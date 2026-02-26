(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Config *)

type config = {
  vocab_size : int;
  n_positions : int;
  n_embd : int;
  n_layer : int;
  n_head : int;
  n_inner : int;
  resid_pdrop : float;
  embd_pdrop : float;
  attn_pdrop : float;
  layer_norm_eps : float;
}

let config ~vocab_size ~n_embd ~n_layer ~n_head ?(n_positions = 1024)
    ?(n_inner = 4 * n_embd) ?(resid_pdrop = 0.1) ?(embd_pdrop = 0.1)
    ?(attn_pdrop = 0.1) ?(layer_norm_eps = 1e-5) () =
  if n_embd mod n_head <> 0 then
    invalid_argf "Gpt2.config: n_embd (%d) not divisible by n_head (%d)" n_embd
      n_head;
  {
    vocab_size;
    n_positions;
    n_embd;
    n_layer;
    n_head;
    n_inner;
    resid_pdrop;
    embd_pdrop;
    attn_pdrop;
    layer_norm_eps;
  }

(* Helpers *)

let fields ~ctx t = Ptree.Dict.fields_exn ~ctx t
let get fs ~name dtype = Ptree.Dict.get_tensor_exn fs ~name dtype
let find ~ctx key fs = Ptree.Dict.find_exn ~ctx key fs

(* Causal self-attention with combined QKV *)

let causal_self_attention (type l) ~(cfg : config)
    ~(dtype : (float, l) Rune.dtype) ~training ?rngs ~params
    (x : (float, l) Rune.t) : (float, l) Rune.t =
  let shape = Rune.shape x in
  let batch = shape.(0) in
  let seq = shape.(1) in
  let h = cfg.n_embd in
  let heads = cfg.n_head in
  let head_dim = h / heads in
  let fs = fields ~ctx:"Gpt2.attention" params in

  (* Combined QKV projection: [batch, seq, 3*h] *)
  let qkv_w = get fs ~name:"qkv_weight" dtype in
  let qkv_b = get fs ~name:"qkv_bias" dtype in
  let qkv = Rune.add (Rune.matmul x qkv_w) qkv_b in

  (* Split into Q, K, V *)
  let qkv_parts = Rune.split ~axis:(-1) 3 qkv in
  let q = List.nth qkv_parts 0 in
  let k = List.nth qkv_parts 1 in
  let v = List.nth qkv_parts 2 in

  let split_heads t =
    Rune.reshape [| batch; seq; heads; head_dim |] t
    |> Rune.transpose ~axes:[ 0; 2; 1; 3 ]
  in
  let q = split_heads q in
  let k = split_heads k in
  let v = split_heads v in

  let dropout_rate =
    if training && cfg.attn_pdrop > 0.0 then Some cfg.attn_pdrop else None
  in
  let dropout_key =
    match dropout_rate with
    | Some _ -> (
        match rngs with
        | Some key -> Some (Rune.Rng.split key).(0)
        | None ->
            invalid_arg
              "Gpt2.attention: requires ~rngs during training with dropout")
    | None -> None
  in

  let attn =
    Rune.dot_product_attention ~is_causal:true ?dropout_rate ?dropout_key q k v
  in

  (* Merge heads *)
  let merged =
    Rune.transpose attn ~axes:[ 0; 2; 1; 3 ]
    |> Rune.contiguous
    |> Rune.reshape [| batch; seq; h |]
  in

  (* Output projection *)
  let o_w = get fs ~name:"o_weight" dtype in
  let o_b = get fs ~name:"o_bias" dtype in
  Rune.add (Rune.matmul merged o_w) o_b

(* Transformer block (pre-norm) *)

let transformer_block (type l) ~(cfg : config) ~(dtype : (float, l) Rune.dtype)
    ~training ?rngs ~params (x : (float, l) Rune.t) : (float, l) Rune.t =
  let fs = fields ~ctx:"Gpt2.block" params in

  (* RNG splitting: [attn_drop, resid_drop1, resid_drop2] *)
  let attn_key, drop1_key, drop2_key =
    match rngs with
    | Some key ->
        let keys = Rune.Rng.split ~n:3 key in
        (Some keys.(0), Some keys.(1), Some keys.(2))
    | None -> (None, None, None)
  in

  (* Pre-norm attention *)
  let ln1_g = get fs ~name:"ln1_gamma" dtype in
  let ln1_b = get fs ~name:"ln1_beta" dtype in
  let x' =
    Rune.layer_norm ~gamma:ln1_g ~beta:ln1_b ~epsilon:cfg.layer_norm_eps x
  in

  let attn_params = find ~ctx:"Gpt2.block" "attention" fs in
  let attn =
    causal_self_attention ~cfg ~dtype ~training ?rngs:attn_key
      ~params:attn_params x'
  in

  (* Residual dropout *)
  let attn =
    if training && cfg.resid_pdrop > 0.0 then
      match drop1_key with
      | Some key -> Rune.dropout ~key ~rate:cfg.resid_pdrop attn
      | None ->
          invalid_arg "Gpt2.block: requires ~rngs during training with dropout"
    else attn
  in
  let x = Rune.add x attn in

  (* Pre-norm FFN *)
  let ln2_g = get fs ~name:"ln2_gamma" dtype in
  let ln2_b = get fs ~name:"ln2_beta" dtype in
  let x' =
    Rune.layer_norm ~gamma:ln2_g ~beta:ln2_b ~epsilon:cfg.layer_norm_eps x
  in

  let ffn_up_w = get fs ~name:"ffn_up_weight" dtype in
  let ffn_up_b = get fs ~name:"ffn_up_bias" dtype in
  let ffn_down_w = get fs ~name:"ffn_down_weight" dtype in
  let ffn_down_b = get fs ~name:"ffn_down_bias" dtype in

  let y = Rune.add (Rune.matmul x' ffn_up_w) ffn_up_b |> Rune.gelu_approx in
  let y = Rune.add (Rune.matmul y ffn_down_w) ffn_down_b in

  (* Residual dropout *)
  let y =
    if training && cfg.resid_pdrop > 0.0 then
      match drop2_key with
      | Some key -> Rune.dropout ~key ~rate:cfg.resid_pdrop y
      | None ->
          invalid_arg "Gpt2.block: requires ~rngs during training with dropout"
    else y
  in
  Rune.add x y

(* Forward: embeddings + transformer stack + final layer norm *)

let decode (type l in_elt) ~(cfg : config) ~params
    ~(dtype : (float, l) Rune.dtype) ~training ?rngs
    (input_ids : (int32, in_elt) Rune.t) : (float, l) Rune.t =
  let input_ids = Rune.cast Rune.int32 input_ids in
  let shape = Rune.shape input_ids in
  let batch = shape.(0) in
  let seq = shape.(1) in

  if seq > cfg.n_positions then
    invalid_argf "Gpt2.decode: seq_len=%d exceeds n_positions=%d" seq
      cfg.n_positions;

  (* Split rngs: [emb_drop, block0, block1, ...] *)
  let keys =
    match rngs with
    | Some key -> Rune.Rng.split ~n:(1 + cfg.n_layer) key
    | None -> [||]
  in
  let key_at i = if Array.length keys > i then Some keys.(i) else None in

  (* Params *)
  let root = fields ~ctx:"Gpt2.decode" params in

  let wte = get root ~name:"wte" dtype in
  let wpe = get root ~name:"wpe" dtype in
  let layers_t = find ~ctx:"Gpt2.decode" "layers" root in

  (* Embedding lookup: token + position *)
  let position_ids =
    Rune.arange_f Rune.float32 0.0 (float_of_int seq) 1.0
    |> Rune.cast Rune.int32
    |> Rune.reshape [| 1; seq |]
    |> Rune.broadcast_to [| batch; seq |]
    |> Rune.contiguous
  in
  let tok = Rune.embedding ~scale:false ~embedding:wte input_ids in
  let pos = Rune.embedding ~scale:false ~embedding:wpe position_ids in
  let x = Rune.add tok pos in

  (* Embedding dropout *)
  let x =
    if training && cfg.embd_pdrop > 0.0 then
      match key_at 0 with
      | Some key -> Rune.dropout ~key ~rate:cfg.embd_pdrop x
      | None ->
          invalid_arg "Gpt2.decode: requires ~rngs during training with dropout"
    else x
  in

  (* Transformer stack *)
  let blocks = Ptree.List.items_exn ~ctx:"Gpt2.decode.layers" layers_t in
  let _, x =
    List.fold_left
      (fun (i, h) block_params ->
        let h =
          transformer_block ~cfg ~dtype ~training
            ?rngs:(key_at (i + 1))
            ~params:block_params h
        in
        (i + 1, h))
      (0, x) blocks
  in

  (* Final layer norm *)
  let ln_f_g = get root ~name:"ln_f_gamma" dtype in
  let ln_f_b = get root ~name:"ln_f_beta" dtype in
  Rune.layer_norm ~gamma:ln_f_g ~beta:ln_f_b ~epsilon:cfg.layer_norm_eps x

(* Parameter initialization *)

let init_block_params ~dtype ~rngs ~n_embd ~n_inner =
  let w = Init.normal ~stddev:0.02 () in
  let keys = Rune.Rng.split ~n:4 rngs in
  let zeros n = Rune.zeros dtype [| n |] in
  let ones n = Rune.ones dtype [| n |] in
  let attn_params =
    Ptree.dict
      [
        ( "qkv_weight",
          Ptree.tensor (w.f keys.(0) [| n_embd; 3 * n_embd |] dtype) );
        ("qkv_bias", Ptree.tensor (zeros (3 * n_embd)));
        ("o_weight", Ptree.tensor (w.f keys.(1) [| n_embd; n_embd |] dtype));
        ("o_bias", Ptree.tensor (zeros n_embd));
      ]
  in
  Ptree.dict
    [
      ("attention", attn_params);
      ("ln1_gamma", Ptree.tensor (ones n_embd));
      ("ln1_beta", Ptree.tensor (zeros n_embd));
      ("ffn_up_weight", Ptree.tensor (w.f keys.(2) [| n_embd; n_inner |] dtype));
      ("ffn_up_bias", Ptree.tensor (zeros n_inner));
      ( "ffn_down_weight",
        Ptree.tensor (w.f keys.(3) [| n_inner; n_embd |] dtype) );
      ("ffn_down_bias", Ptree.tensor (zeros n_embd));
      ("ln2_gamma", Ptree.tensor (ones n_embd));
      ("ln2_beta", Ptree.tensor (zeros n_embd));
    ]

let init_decoder_params ~cfg ~dtype ~rngs =
  let h = cfg.n_embd in
  let w = Init.normal ~stddev:0.02 () in
  let keys = Rune.Rng.split ~n:(2 + cfg.n_layer) rngs in
  let wte = w.f keys.(0) [| cfg.vocab_size; h |] dtype in
  let wpe = w.f keys.(1) [| cfg.n_positions; h |] dtype in
  let blocks =
    List.init cfg.n_layer (fun i ->
        init_block_params ~dtype
          ~rngs:keys.(2 + i)
          ~n_embd:h ~n_inner:cfg.n_inner)
  in
  Ptree.dict
    [
      ("wte", Ptree.tensor wte);
      ("wpe", Ptree.tensor wpe);
      ("layers", Ptree.list blocks);
      ("ln_f_gamma", Ptree.tensor (Rune.ones dtype [| h |]));
      ("ln_f_beta", Ptree.tensor (Rune.zeros dtype [| h |]));
    ]

(* Layers *)

let decoder (cfg : config) () : (int32, float) Layer.t =
  {
    Layer.init =
      (fun ~rngs ~dtype ->
        Layer.make_vars
          ~params:(init_decoder_params ~cfg ~dtype ~rngs)
          ~state:Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?rngs ?ctx x ->
        ignore (state, ctx);
        let y = decode ~cfg ~params ~dtype ~training ?rngs x in
        (y, Ptree.empty));
  }

let for_causal_lm (cfg : config) () : (int32, float) Layer.t =
  {
    Layer.init =
      (fun ~rngs ~dtype ->
        Layer.make_vars
          ~params:(init_decoder_params ~cfg ~dtype ~rngs)
          ~state:Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?rngs ?ctx x ->
        ignore (state, ctx);
        let hidden = decode ~cfg ~params ~dtype ~training ?rngs x in
        (* Tied LM head: logits = hidden @ wte^T *)
        let root = fields ~ctx:"Gpt2.lm_head" params in
        let wte = get root ~name:"wte" dtype in
        let logits = Rune.matmul hidden (Rune.transpose wte ~axes:[ 1; 0 ]) in
        (logits, Ptree.empty));
  }

(* JSON config parsing *)

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let json_to_int = function
  | Jsont.Number (f, _) -> int_of_float f
  | _ -> failwith "expected int"

let json_to_int_option = function
  | Jsont.Number (f, _) -> Some (int_of_float f)
  | _ -> None

let json_to_float_option = function Jsont.Number (f, _) -> Some f | _ -> None

let parse_config json =
  let n_embd = json |> json_mem "n_embd" |> json_to_int in
  config
    ~vocab_size:(json |> json_mem "vocab_size" |> json_to_int)
    ~n_embd
    ~n_layer:(json |> json_mem "n_layer" |> json_to_int)
    ~n_head:(json |> json_mem "n_head" |> json_to_int)
    ?n_positions:(json |> json_mem "n_positions" |> json_to_int_option)
    ?n_inner:(json |> json_mem "n_inner" |> json_to_int_option)
    ?resid_pdrop:(json |> json_mem "resid_pdrop" |> json_to_float_option)
    ?embd_pdrop:(json |> json_mem "embd_pdrop" |> json_to_float_option)
    ?attn_pdrop:(json |> json_mem "attn_pdrop" |> json_to_float_option)
    ?layer_norm_eps:
      (json |> json_mem "layer_norm_epsilon" |> json_to_float_option)
    ()

(* HuggingFace weight mapping *)

let cast_tensor dtype (Ptree.P t) =
  Ptree.P (Rune.cast dtype (Rune.of_nx (Rune.to_nx t)))

let map_hf_weights ~cfg ~dtype hf_weights =
  let tbl = Hashtbl.create (List.length hf_weights) in
  List.iter (fun (name, tensor) -> Hashtbl.add tbl name tensor) hf_weights;
  let hf name =
    match Hashtbl.find_opt tbl name with
    | Some t -> cast_tensor dtype t
    | None -> invalid_argf "from_pretrained: missing HF weight %S" name
  in
  (* GPT-2 stores weights as [in, out] — NO transpose needed *)
  let hf_t name = Ptree.Tensor (hf name) in
  let layer i =
    let p s = Printf.sprintf "h.%d.%s" i s in
    Ptree.dict
      [
        ( "attention",
          Ptree.dict
            [
              ("qkv_weight", hf_t (p "attn.c_attn.weight"));
              ("qkv_bias", hf_t (p "attn.c_attn.bias"));
              ("o_weight", hf_t (p "attn.c_proj.weight"));
              ("o_bias", hf_t (p "attn.c_proj.bias"));
            ] );
        ("ln1_gamma", hf_t (p "ln_1.weight"));
        ("ln1_beta", hf_t (p "ln_1.bias"));
        ("ffn_up_weight", hf_t (p "mlp.c_fc.weight"));
        ("ffn_up_bias", hf_t (p "mlp.c_fc.bias"));
        ("ffn_down_weight", hf_t (p "mlp.c_proj.weight"));
        ("ffn_down_bias", hf_t (p "mlp.c_proj.bias"));
        ("ln2_gamma", hf_t (p "ln_2.weight"));
        ("ln2_beta", hf_t (p "ln_2.bias"));
      ]
  in
  Ptree.dict
    [
      ("wte", hf_t "wte.weight");
      ("wpe", hf_t "wpe.weight");
      ("layers", Ptree.list (List.init cfg.n_layer layer));
      ("ln_f_gamma", hf_t "ln_f.weight");
      ("ln_f_beta", hf_t "ln_f.bias");
    ]

(* Pretrained loading *)

let from_pretrained ?(model_id = "gpt2") () =
  let json = Kaun_hf.load_config ~model_id () in
  let cfg = parse_config json in
  let hf_weights = Kaun_hf.load_weights ~model_id () in
  let params = map_hf_weights ~cfg ~dtype:Rune.float32 hf_weights in
  (cfg, params)
