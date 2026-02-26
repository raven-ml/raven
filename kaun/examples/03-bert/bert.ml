(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let require_float_dtype (type p in_elt) ~ctx (expected : (float, p) Rune.dtype)
    (x : (float, in_elt) Rune.t) : (float, p) Rune.t =
  match Nx_core.Dtype.equal_witness expected (Rune.dtype x) with
  | Some Type.Equal -> x
  | None ->
      invalid_argf "%s: dtype mismatch (expected %s, got %s)" ctx
        (Nx_core.Dtype.to_string expected)
        (Nx_core.Dtype.to_string (Rune.dtype x))

(* Config *)

type config = {
  vocab_size : int;
  max_position_embeddings : int;
  type_vocab_size : int;
  hidden_size : int;
  num_hidden_layers : int;
  num_attention_heads : int;
  intermediate_size : int;
  hidden_dropout_prob : float;
  attention_dropout_prob : float;
  layer_norm_eps : float;
}

let config ~vocab_size ~hidden_size ~num_hidden_layers ~num_attention_heads
    ~intermediate_size ?(max_position_embeddings = 512) ?(type_vocab_size = 2)
    ?(hidden_dropout_prob = 0.1) ?(attention_dropout_prob = 0.1)
    ?(layer_norm_eps = 1e-12) () =
  if hidden_size mod num_attention_heads <> 0 then
    invalid_argf
      "Bert.config: hidden_size (%d) not divisible by num_attention_heads (%d)"
      hidden_size num_attention_heads;
  if hidden_dropout_prob < 0.0 || hidden_dropout_prob >= 1.0 then
    invalid_arg "Bert.config: hidden_dropout_prob must satisfy 0 <= p < 1";
  if attention_dropout_prob < 0.0 || attention_dropout_prob >= 1.0 then
    invalid_arg "Bert.config: attention_dropout_prob must satisfy 0 <= p < 1";
  {
    vocab_size;
    max_position_embeddings;
    type_vocab_size;
    hidden_size;
    num_hidden_layers;
    num_attention_heads;
    intermediate_size;
    hidden_dropout_prob;
    attention_dropout_prob;
    layer_norm_eps;
  }

(* Context keys *)

let token_type_ids_key = "token_type_ids"

(* Helpers *)

let get_from_ctx_int32 ~name ~default ctx =
  match ctx with
  | Some c -> (
      match Context.find c ~name with
      | Some tensor -> Ptree.Tensor.to_typed_exn Rune.int32 tensor
      | None -> default ())
  | None -> default ()

let get_attention_mask_bool ctx ~batch ~seq =
  match ctx with
  | Some c -> (
      match Context.find c ~name:Attention.attention_mask_key with
      | Some tensor -> (
          match Ptree.Tensor.to_typed Rune.bool tensor with
          | Some m -> m
          | None ->
              let int_mask = Ptree.Tensor.to_typed_exn Rune.int32 tensor in
              Rune.not_equal int_mask
                (Rune.zeros Rune.int32 (Rune.shape int_mask)))
      | None -> Rune.broadcast_to [| batch; seq |] (Rune.scalar Rune.bool true))
  | None -> Rune.broadcast_to [| batch; seq |] (Rune.scalar Rune.bool true)

let fields ~ctx t = Ptree.Dict.fields_exn ~ctx t
let get fs ~name dtype = Ptree.Dict.get_tensor_exn fs ~name dtype
let find ~ctx key fs = Ptree.Dict.find_exn ~ctx key fs

(* Self-attention with biased projections *)

let self_attention (type l) ~(cfg : config) ~(dtype : (float, l) Rune.dtype)
    ~training ?rngs ~attention_mask ~params (x : (float, l) Rune.t) :
    (float, l) Rune.t =
  let shape = Rune.shape x in
  let batch = shape.(0) in
  let seq = shape.(1) in
  let h = cfg.hidden_size in
  let heads = cfg.num_attention_heads in
  let head_dim = h / heads in
  let fs = fields ~ctx:"Bert.attention" params in

  let proj name =
    let w = get fs ~name:(name ^ "_weight") dtype in
    let b = get fs ~name:(name ^ "_bias") dtype in
    fun t -> Rune.add (Rune.matmul t w) b
  in
  let q = proj "q" x in
  let k = proj "k" x in
  let v = proj "v" x in

  let split_heads t =
    Rune.reshape [| batch; seq; heads; head_dim |] t
    |> Rune.transpose ~axes:[ 0; 2; 1; 3 ]
  in
  let q = split_heads q in
  let k = split_heads k in
  let v = split_heads v in

  (* Broadcast mask [batch; seq] -> [batch; 1; 1; seq] *)
  let attention_mask = Rune.reshape [| batch; 1; 1; seq |] attention_mask in

  let dropout_rate =
    if training && cfg.attention_dropout_prob > 0.0 then
      Some cfg.attention_dropout_prob
    else None
  in
  let dropout_key =
    match dropout_rate with
    | Some _ -> (
        match rngs with
        | Some key -> Some (Rune.Rng.split key).(0)
        | None ->
            invalid_arg
              "Bert.attention: requires ~rngs during training with dropout")
    | None -> None
  in

  let attn =
    Rune.dot_product_attention ~attention_mask ?dropout_rate ?dropout_key q k v
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

(* Encoder block *)

let encoder_block (type l) ~(cfg : config) ~(dtype : (float, l) Rune.dtype)
    ~training ?rngs ?ctx ~attention_mask ~params (x : (float, l) Rune.t) :
    (float, l) Rune.t =
  ignore ctx;
  let fs = fields ~ctx:"Bert.block" params in

  (* RNG splitting: [attn_drop, hidden_drop1, hidden_drop2] *)
  let attn_key, drop1_key, drop2_key =
    match rngs with
    | Some key ->
        let keys = Rune.Rng.split ~n:3 key in
        (Some keys.(0), Some keys.(1), Some keys.(2))
    | None -> (None, None, None)
  in

  (* Self-attention *)
  let attn_params = find ~ctx:"Bert.block" "attention" fs in
  let attn =
    self_attention ~cfg ~dtype ~training ?rngs:attn_key ~attention_mask
      ~params:attn_params x
  in

  (* Hidden dropout on attention output *)
  let attn =
    if training && cfg.hidden_dropout_prob > 0.0 then
      match drop1_key with
      | Some key -> Rune.dropout ~key ~rate:cfg.hidden_dropout_prob attn
      | None ->
          invalid_arg "Bert.block: requires ~rngs during training with dropout"
    else attn
  in

  (* Residual + LayerNorm (post-norm, original BERT) *)
  let ln1_g = get fs ~name:"attn_ln_gamma" dtype in
  let ln1_b = get fs ~name:"attn_ln_beta" dtype in
  let x =
    Rune.layer_norm ~gamma:ln1_g ~beta:ln1_b ~epsilon:cfg.layer_norm_eps
      (Rune.add x attn)
  in

  (* FFN: up -> GELU -> down *)
  let ffn_up_w = get fs ~name:"ffn_up_weight" dtype in
  let ffn_up_b = get fs ~name:"ffn_up_bias" dtype in
  let ffn_down_w = get fs ~name:"ffn_down_weight" dtype in
  let ffn_down_b = get fs ~name:"ffn_down_bias" dtype in

  let y = Rune.add (Rune.matmul x ffn_up_w) ffn_up_b |> Rune.gelu in
  let y = Rune.add (Rune.matmul y ffn_down_w) ffn_down_b in

  (* Hidden dropout on FFN output *)
  let y =
    if training && cfg.hidden_dropout_prob > 0.0 then
      match drop2_key with
      | Some key -> Rune.dropout ~key ~rate:cfg.hidden_dropout_prob y
      | None ->
          invalid_arg "Bert.block: requires ~rngs during training with dropout"
    else y
  in

  (* Residual + LayerNorm *)
  let ln2_g = get fs ~name:"ffn_ln_gamma" dtype in
  let ln2_b = get fs ~name:"ffn_ln_beta" dtype in
  Rune.layer_norm ~gamma:ln2_g ~beta:ln2_b ~epsilon:cfg.layer_norm_eps
    (Rune.add x y)

(* Forward: embeddings + encoder stack *)

let encode (type l in_elt) ~(cfg : config) ~params
    ~(dtype : (float, l) Rune.dtype) ~training ?rngs ?ctx
    (input_ids : (int32, in_elt) Rune.t) : (float, l) Rune.t =
  let input_ids = Rune.cast Rune.int32 input_ids in
  let shape = Rune.shape input_ids in
  let batch = shape.(0) in
  let seq = shape.(1) in

  if seq > cfg.max_position_embeddings then
    invalid_argf "Bert.encode: seq_len=%d exceeds max_position_embeddings=%d"
      seq cfg.max_position_embeddings;

  (* Read auxiliary inputs from context *)
  let token_type_ids =
    get_from_ctx_int32 ~name:token_type_ids_key ctx ~default:(fun () ->
        Rune.zeros Rune.int32 [| batch; seq |])
  in
  let attention_mask = get_attention_mask_bool ctx ~batch ~seq in

  (* Split rngs: [emb_drop, block0, block1, ...] *)
  let keys =
    match rngs with
    | Some key -> Rune.Rng.split ~n:(1 + cfg.num_hidden_layers) key
    | None -> [||]
  in
  let key_at i = if Array.length keys > i then Some keys.(i) else None in

  (* Params *)
  let root = fields ~ctx:"Bert.encode" params in
  let emb_t = find ~ctx:"Bert.encode" "embeddings" root in
  let layers_t = find ~ctx:"Bert.encode" "layers" root in

  let emb = fields ~ctx:"Bert.embeddings" emb_t in
  let word_emb = get emb ~name:"word" dtype in
  let pos_emb = get emb ~name:"pos" dtype in
  let type_emb = get emb ~name:"type" dtype in
  let ln_g = get emb ~name:"ln_gamma" dtype in
  let ln_b = get emb ~name:"ln_beta" dtype in

  (* Embedding lookup: word + position + token_type *)
  let position_ids =
    Rune.arange_f Rune.float32 0.0 (float_of_int seq) 1.0
    |> Rune.cast Rune.int32
    |> Rune.reshape [| 1; seq |]
    |> Rune.broadcast_to [| batch; seq |]
    |> Rune.contiguous
  in
  let token_type_ids = Rune.contiguous token_type_ids in
  let tok = Rune.embedding ~scale:false ~embedding:word_emb input_ids in
  let pos = Rune.embedding ~scale:false ~embedding:pos_emb position_ids in
  let typ = Rune.embedding ~scale:false ~embedding:type_emb token_type_ids in
  let x = Rune.add tok (Rune.add pos typ) in
  let x =
    Rune.layer_norm ~gamma:ln_g ~beta:ln_b ~epsilon:cfg.layer_norm_eps x
  in

  (* Embedding dropout *)
  let x =
    if training && cfg.hidden_dropout_prob > 0.0 then
      match key_at 0 with
      | Some key -> Rune.dropout ~key ~rate:cfg.hidden_dropout_prob x
      | None ->
          invalid_arg "Bert.encode: requires ~rngs during training with dropout"
    else x
  in

  (* Encoder stack *)
  let blocks = Ptree.List.items_exn ~ctx:"Bert.encode.layers" layers_t in
  let _, x =
    List.fold_left
      (fun (i, h) block_params ->
        let h =
          encoder_block ~cfg ~dtype ~training
            ?rngs:(key_at (i + 1))
            ?ctx ~attention_mask ~params:block_params h
        in
        (i + 1, h))
      (0, x) blocks
  in
  x

(* Parameter initialization *)

let init_block_params ~dtype ~rngs ~hidden ~intermediate =
  let w = Init.normal ~stddev:0.02 () in
  let keys = Rune.Rng.split ~n:6 rngs in
  let zeros n = Rune.zeros dtype [| n |] in
  let ones n = Rune.ones dtype [| n |] in
  let attn_params =
    Ptree.dict
      [
        ("q_weight", Ptree.tensor (w.f keys.(0) [| hidden; hidden |] dtype));
        ("q_bias", Ptree.tensor (zeros hidden));
        ("k_weight", Ptree.tensor (w.f keys.(1) [| hidden; hidden |] dtype));
        ("k_bias", Ptree.tensor (zeros hidden));
        ("v_weight", Ptree.tensor (w.f keys.(2) [| hidden; hidden |] dtype));
        ("v_bias", Ptree.tensor (zeros hidden));
        ("o_weight", Ptree.tensor (w.f keys.(3) [| hidden; hidden |] dtype));
        ("o_bias", Ptree.tensor (zeros hidden));
      ]
  in
  Ptree.dict
    [
      ("attention", attn_params);
      ("attn_ln_gamma", Ptree.tensor (ones hidden));
      ("attn_ln_beta", Ptree.tensor (zeros hidden));
      ( "ffn_up_weight",
        Ptree.tensor (w.f keys.(4) [| hidden; intermediate |] dtype) );
      ("ffn_up_bias", Ptree.tensor (zeros intermediate));
      ( "ffn_down_weight",
        Ptree.tensor (w.f keys.(5) [| intermediate; hidden |] dtype) );
      ("ffn_down_bias", Ptree.tensor (zeros hidden));
      ("ffn_ln_gamma", Ptree.tensor (ones hidden));
      ("ffn_ln_beta", Ptree.tensor (zeros hidden));
    ]

let init_encoder_params ~cfg ~dtype ~rngs =
  let h = cfg.hidden_size in
  let w = Init.normal ~stddev:0.02 () in
  let keys = Rune.Rng.split ~n:(3 + cfg.num_hidden_layers) rngs in
  let word = w.f keys.(0) [| cfg.vocab_size; h |] dtype in
  let pos = w.f keys.(1) [| cfg.max_position_embeddings; h |] dtype in
  let typ = w.f keys.(2) [| cfg.type_vocab_size; h |] dtype in
  let blocks =
    List.init cfg.num_hidden_layers (fun i ->
        init_block_params ~dtype
          ~rngs:keys.(3 + i)
          ~hidden:h ~intermediate:cfg.intermediate_size)
  in
  Ptree.dict
    [
      ( "embeddings",
        Ptree.dict
          [
            ("word", Ptree.tensor word);
            ("pos", Ptree.tensor pos);
            ("type", Ptree.tensor typ);
            ("ln_gamma", Ptree.tensor (Rune.ones dtype [| h |]));
            ("ln_beta", Ptree.tensor (Rune.zeros dtype [| h |]));
          ] );
      ("layers", Ptree.list blocks);
    ]

(* Layers *)

let encoder (cfg : config) () : (int32, float) Layer.t =
  {
    Layer.init =
      (fun ~rngs ~dtype ->
        Layer.make_vars
          ~params:(init_encoder_params ~cfg ~dtype ~rngs)
          ~state:Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?rngs ?ctx x ->
        ignore state;
        let y = encode ~cfg ~params ~dtype ~training ?rngs ?ctx x in
        (y, Ptree.empty));
  }

let pooler (cfg : config) () : (float, float) Layer.t =
  let w_init = Init.normal ~stddev:0.02 () in
  {
    Layer.init =
      (fun ~rngs ~dtype ->
        let key = (Rune.Rng.split rngs).(0) in
        let w = w_init.f key [| cfg.hidden_size; cfg.hidden_size |] dtype in
        let b = Rune.zeros dtype [| cfg.hidden_size |] in
        Layer.make_vars
          ~params:
            (Ptree.dict
               [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ])
          ~state:Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?rngs ?ctx x ->
        ignore (training, rngs, ctx, state);
        let x = require_float_dtype ~ctx:"Bert.pooler" dtype x in
        let fs = fields ~ctx:"Bert.pooler" params in
        let w = get fs ~name:"weight" dtype in
        let b = get fs ~name:"bias" dtype in
        let batch = (Rune.shape x).(0) in
        let cls =
          Rune.slice [ A; R (0, 1) ] x
          |> Rune.reshape [| batch; cfg.hidden_size |]
        in
        (Rune.add (Rune.matmul cls w) b |> Rune.tanh, Ptree.empty));
  }

let for_sequence_classification (cfg : config) ~num_labels () :
    (int32, float) Layer.t =
  let w_init = Init.normal ~stddev:0.02 () in
  {
    Layer.init =
      (fun ~rngs ~dtype ->
        let keys = Rune.Rng.split ~n:3 rngs in
        let enc = init_encoder_params ~cfg ~dtype ~rngs:keys.(0) in
        let pool_w =
          w_init.f keys.(1) [| cfg.hidden_size; cfg.hidden_size |] dtype
        in
        let cls_w = w_init.f keys.(2) [| cfg.hidden_size; num_labels |] dtype in
        Layer.make_vars
          ~params:
            (Ptree.dict
               [
                 ("encoder", enc);
                 ( "pooler",
                   Ptree.dict
                     [
                       ("weight", Ptree.tensor pool_w);
                       ( "bias",
                         Ptree.tensor (Rune.zeros dtype [| cfg.hidden_size |])
                       );
                     ] );
                 ( "classifier",
                   Ptree.dict
                     [
                       ("weight", Ptree.tensor cls_w);
                       ("bias", Ptree.tensor (Rune.zeros dtype [| num_labels |]));
                     ] );
               ])
          ~state:Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?rngs ?ctx x ->
        ignore state;
        let root = fields ~ctx:"Bert.seq_cls" params in
        let enc_params = find ~ctx:"Bert.seq_cls" "encoder" root in
        let pool_params = find ~ctx:"Bert.seq_cls" "pooler" root in
        let cls_params = find ~ctx:"Bert.seq_cls" "classifier" root in

        let hidden =
          encode ~cfg ~params:enc_params ~dtype ~training ?rngs ?ctx x
        in

        (* Pooler: CLS token -> dense -> tanh *)
        let pool_fs = fields ~ctx:"Bert.seq_cls.pooler" pool_params in
        let pool_w = get pool_fs ~name:"weight" dtype in
        let pool_b = get pool_fs ~name:"bias" dtype in
        let batch = (Rune.shape hidden).(0) in
        let cls =
          Rune.slice [ A; R (0, 1) ] hidden
          |> Rune.reshape [| batch; cfg.hidden_size |]
        in
        let pooled = Rune.add (Rune.matmul cls pool_w) pool_b |> Rune.tanh in

        (* Dropout on pooled output during fine-tuning *)
        let pooled =
          if training && cfg.hidden_dropout_prob > 0.0 then
            match rngs with
            | Some key ->
                let k = (Rune.Rng.split key).(0) in
                Rune.dropout ~key:k ~rate:cfg.hidden_dropout_prob pooled
            | None ->
                invalid_arg
                  "Bert.for_sequence_classification: requires ~rngs during \
                   training with dropout"
          else pooled
        in

        (* Classifier *)
        let cls_fs = fields ~ctx:"Bert.seq_cls.classifier" cls_params in
        let cls_w = get cls_fs ~name:"weight" dtype in
        let cls_b = get cls_fs ~name:"bias" dtype in
        (Rune.add (Rune.matmul pooled cls_w) cls_b, Ptree.empty));
  }

let for_masked_lm (cfg : config) () : (int32, float) Layer.t =
  let w_init = Init.normal ~stddev:0.02 () in
  {
    Layer.init =
      (fun ~rngs ~dtype ->
        let keys = Rune.Rng.split ~n:2 rngs in
        let enc = init_encoder_params ~cfg ~dtype ~rngs:keys.(0) in
        let dense_w =
          w_init.f keys.(1) [| cfg.hidden_size; cfg.hidden_size |] dtype
        in
        Layer.make_vars
          ~params:
            (Ptree.dict
               [
                 ("encoder", enc);
                 ( "mlm",
                   Ptree.dict
                     [
                       ("dense_weight", Ptree.tensor dense_w);
                       ( "dense_bias",
                         Ptree.tensor (Rune.zeros dtype [| cfg.hidden_size |])
                       );
                       ( "ln_gamma",
                         Ptree.tensor (Rune.ones dtype [| cfg.hidden_size |]) );
                       ( "ln_beta",
                         Ptree.tensor (Rune.zeros dtype [| cfg.hidden_size |])
                       );
                       ( "decoder_bias",
                         Ptree.tensor (Rune.zeros dtype [| cfg.vocab_size |]) );
                     ] );
               ])
          ~state:Ptree.empty ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?rngs ?ctx x ->
        ignore state;
        let root = fields ~ctx:"Bert.mlm" params in
        let enc_params = find ~ctx:"Bert.mlm" "encoder" root in
        let mlm_params = find ~ctx:"Bert.mlm" "mlm" root in

        let hidden =
          encode ~cfg ~params:enc_params ~dtype ~training ?rngs ?ctx x
        in

        (* MLM transform: dense -> GELU -> LN *)
        let mlm_fs = fields ~ctx:"Bert.mlm.head" mlm_params in
        let dw = get mlm_fs ~name:"dense_weight" dtype in
        let db = get mlm_fs ~name:"dense_bias" dtype in
        let ln_g = get mlm_fs ~name:"ln_gamma" dtype in
        let ln_b = get mlm_fs ~name:"ln_beta" dtype in
        let dec_b = get mlm_fs ~name:"decoder_bias" dtype in

        let h = Rune.add (Rune.matmul hidden dw) db |> Rune.gelu in
        let h =
          Rune.layer_norm ~gamma:ln_g ~beta:ln_b ~epsilon:cfg.layer_norm_eps h
        in

        (* Tied decoder: logits = h @ word_emb^T + bias *)
        let enc_root = fields ~ctx:"Bert.mlm.encoder" enc_params in
        let emb_t = find ~ctx:"Bert.mlm.encoder" "embeddings" enc_root in
        let emb_fs = fields ~ctx:"Bert.mlm.embeddings" emb_t in
        let word_emb = get emb_fs ~name:"word" dtype in
        let logits =
          Rune.add
            (Rune.matmul h (Rune.transpose word_emb ~axes:[ 1; 0 ]))
            dec_b
        in
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
  config
    ~vocab_size:(json |> json_mem "vocab_size" |> json_to_int)
    ~hidden_size:(json |> json_mem "hidden_size" |> json_to_int)
    ~num_hidden_layers:(json |> json_mem "num_hidden_layers" |> json_to_int)
    ~num_attention_heads:(json |> json_mem "num_attention_heads" |> json_to_int)
    ~intermediate_size:(json |> json_mem "intermediate_size" |> json_to_int)
    ?max_position_embeddings:
      (json |> json_mem "max_position_embeddings" |> json_to_int_option)
    ?type_vocab_size:(json |> json_mem "type_vocab_size" |> json_to_int_option)
    ?hidden_dropout_prob:
      (json |> json_mem "hidden_dropout_prob" |> json_to_float_option)
    ?attention_dropout_prob:
      (json |> json_mem "attention_probs_dropout_prob" |> json_to_float_option)
    ?layer_norm_eps:(json |> json_mem "layer_norm_eps" |> json_to_float_option)
    ()

(* HuggingFace weight mapping *)

let transpose_weight (Ptree.P t) = Ptree.P (Rune.transpose t ~axes:[ 1; 0 ])

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
  (* Some checkpoints use LayerNorm.weight/bias, others use
     LayerNorm.gamma/beta. Try both. *)
  let hf_ln_weight prefix =
    let w = prefix ^ ".weight" in
    let g = prefix ^ ".gamma" in
    if Hashtbl.mem tbl w then hf w else hf g
  in
  let hf_ln_bias prefix =
    let b = prefix ^ ".bias" in
    let beta = prefix ^ ".beta" in
    if Hashtbl.mem tbl b then hf b else hf beta
  in
  let hf_t name = Ptree.Tensor (transpose_weight (hf name)) in
  let hf_b name = Ptree.Tensor (hf name) in
  let ln_w prefix = Ptree.Tensor (hf_ln_weight prefix) in
  let ln_b prefix = Ptree.Tensor (hf_ln_bias prefix) in
  let layer i =
    let p s = Printf.sprintf "bert.encoder.layer.%d.%s" i s in
    let attn_ln = p "attention.output.LayerNorm" in
    let ffn_ln = p "output.LayerNorm" in
    Ptree.dict
      [
        ( "attention",
          Ptree.dict
            [
              ("q_weight", hf_t (p "attention.self.query.weight"));
              ("q_bias", hf_b (p "attention.self.query.bias"));
              ("k_weight", hf_t (p "attention.self.key.weight"));
              ("k_bias", hf_b (p "attention.self.key.bias"));
              ("v_weight", hf_t (p "attention.self.value.weight"));
              ("v_bias", hf_b (p "attention.self.value.bias"));
              ("o_weight", hf_t (p "attention.output.dense.weight"));
              ("o_bias", hf_b (p "attention.output.dense.bias"));
            ] );
        ("attn_ln_gamma", ln_w attn_ln);
        ("attn_ln_beta", ln_b attn_ln);
        ("ffn_up_weight", hf_t (p "intermediate.dense.weight"));
        ("ffn_up_bias", hf_b (p "intermediate.dense.bias"));
        ("ffn_down_weight", hf_t (p "output.dense.weight"));
        ("ffn_down_bias", hf_b (p "output.dense.bias"));
        ("ffn_ln_gamma", ln_w ffn_ln);
        ("ffn_ln_beta", ln_b ffn_ln);
      ]
  in
  let emb_ln = "bert.embeddings.LayerNorm" in
  let encoder_params =
    Ptree.dict
      [
        ( "embeddings",
          Ptree.dict
            [
              ("word", hf_b "bert.embeddings.word_embeddings.weight");
              ("pos", hf_b "bert.embeddings.position_embeddings.weight");
              ("type", hf_b "bert.embeddings.token_type_embeddings.weight");
              ("ln_gamma", ln_w emb_ln);
              ("ln_beta", ln_b emb_ln);
            ] );
        ("layers", Ptree.list (List.init cfg.num_hidden_layers layer));
      ]
  in
  let pooler_params =
    let has_pooler = Hashtbl.mem tbl "bert.pooler.dense.weight" in
    if has_pooler then
      Some
        (Ptree.dict
           [
             ("weight", hf_t "bert.pooler.dense.weight");
             ("bias", hf_b "bert.pooler.dense.bias");
           ])
    else None
  in
  let mlm_params =
    let has_mlm = Hashtbl.mem tbl "cls.predictions.transform.dense.weight" in
    if has_mlm then
      let mlm_ln = "cls.predictions.transform.LayerNorm" in
      Some
        (Ptree.dict
           [
             ("dense_weight", hf_t "cls.predictions.transform.dense.weight");
             ("dense_bias", hf_b "cls.predictions.transform.dense.bias");
             ("ln_gamma", ln_w mlm_ln);
             ("ln_beta", ln_b mlm_ln);
             ("decoder_bias", hf_b "cls.predictions.bias");
           ])
    else None
  in
  (encoder_params, pooler_params, mlm_params)

(* Pretrained loading *)

let from_pretrained ?(model_id = "bert-base-uncased") () =
  let json = Kaun_hf.load_config ~model_id () in
  let cfg = parse_config json in
  let hf_weights = Kaun_hf.load_weights ~model_id () in
  let encoder_params, pooler_params, mlm_params =
    map_hf_weights ~cfg ~dtype:Rune.float32 hf_weights
  in
  (cfg, encoder_params, pooler_params, mlm_params)
