(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun
module Hf = Kaun_hf

type layer = Sliding | Full

type config = {
  vocab_size : int;
  dim : int;
  layers : layer list;
  window : int;
  n_heads : int;
  n_kv_heads : int;
  head_dim : int;
  hidden_dim : int;
  experts : int;
  experts_per_token : int;
  swiglu_limit : float;
  norm_eps : float;
  rope : Rope.t;
  attention_scale : float;
  tied : bool;
}

type 'a block = {
  attn_norm : 'a Rms_norm.t;
  attn : 'a Attention.t;
  sinks : 'a;
  ffn_norm : 'a Rms_norm.t;
  router : 'a Linear.t;
  moe : 'a Moe.t;
}

type 'a params = {
  tok : 'a Embedding.t;
  blocks : 'a block list;
  norm : 'a Rms_norm.t;
  head : 'a Linear.t option;
}

type t = Nx.float32_t params
type role = Whole | Column | Row | Experts | Kv_heads

(* Traversals *)

let map_block f b =
  let attn_norm = Rms_norm.map f b.attn_norm in
  let attn = Attention.map f b.attn in
  let sinks = f b.sinks in
  let ffn_norm = Rms_norm.map f b.ffn_norm in
  let router = Linear.map f b.router in
  let moe = Moe.map f b.moe in
  { attn_norm; attn; sinks; ffn_norm; router; moe }

let map f p =
  let tok = Embedding.map f p.tok in
  let blocks = List.map (map_block f) p.blocks in
  let norm = Rms_norm.map f p.norm in
  let head = Option.map (Linear.map f) p.head in
  { tok; blocks; norm; head }

let block_ptree (type b) () :
    (module Nx.Ptree.S with type t = (float, b) Nx.t block) =
  (module struct
    type t = (float, b) Nx.t block

    let map_weight (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) = function
      | Moe.Float w -> Moe.Float (f w)
      | Moe.Quant w -> Moe.Quant (Nx_quant.map f w)

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) b =
      let attn_norm = Rms_norm.map f b.attn_norm in
      let attn = Attention.map f b.attn in
      let sinks = f b.sinks in
      let ffn_norm = Rms_norm.map f b.ffn_norm in
      let router = Linear.map f b.router in
      let gate_up = map_weight f b.moe.Moe.gate_up in
      let gate_up_bias = f b.moe.gate_up_bias in
      let down = map_weight f b.moe.down in
      let down_bias = f b.moe.down_bias in
      let moe = { Moe.gate_up; gate_up_bias; down; down_bias } in
      { attn_norm; attn; sinks; ffn_norm; router; moe }

    let map2_weight (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t)
        w w' =
      match (w, w') with
      | Moe.Float w, Moe.Float w' -> Moe.Float (f w w')
      | Moe.Quant w, Moe.Quant w' -> Moe.Quant (Nx_quant.map2 f w w')
      | _ ->
          invalid_arg
            "Gpt_oss.block_ptree: one block packs its experts, one does not"

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) b b' =
      let attn_norm = Rms_norm.map2 f b.attn_norm b'.attn_norm in
      let attn = Attention.map2 f b.attn b'.attn in
      let sinks = f b.sinks b'.sinks in
      let ffn_norm = Rms_norm.map2 f b.ffn_norm b'.ffn_norm in
      let router = Linear.map2 f b.router b'.router in
      let m = b.moe and m' = b'.moe in
      let gate_up = map2_weight f m.Moe.gate_up m'.Moe.gate_up in
      let gate_up_bias = f m.gate_up_bias m'.gate_up_bias in
      let down = map2_weight f m.down m'.down in
      let down_bias = f m.down_bias m'.down_bias in
      let moe = { Moe.gate_up; gate_up_bias; down; down_bias } in
      { attn_norm; attn; sinks; ffn_norm; router; moe }

    let iter_weight (f : 'a 'c. ('a, 'c) Nx.t -> unit) = function
      | Moe.Float w -> f w
      | Moe.Quant w -> Nx_quant.iter f w

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) b =
      Rms_norm.iter f b.attn_norm;
      Attention.iter f b.attn;
      f b.sinks;
      Rms_norm.iter f b.ffn_norm;
      Linear.iter f b.router;
      iter_weight f b.moe.Moe.gate_up;
      f b.moe.gate_up_bias;
      iter_weight f b.moe.down;
      f b.moe.down_bias
  end)

let ptree (type b) () : (module Nx.Ptree.S with type t = (float, b) Nx.t params)
    =
  let module B =
    (val block_ptree () : Nx.Ptree.S with type t = (float, b) Nx.t block)
  in
  (module struct
    type t = (float, b) Nx.t params

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) p =
      let tok = Embedding.map f p.tok in
      let blocks = List.map (B.map f) p.blocks in
      let norm = Rms_norm.map f p.norm in
      let head = Option.map (Linear.map f) p.head in
      { tok; blocks; norm; head }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) p p' =
      let tok = Embedding.map2 f p.tok p'.tok in
      let blocks = List.map2 (B.map2 f) p.blocks p'.blocks in
      let norm = Rms_norm.map2 f p.norm p'.norm in
      let head =
        match (p.head, p'.head) with
        | None, None -> None
        | Some l, Some l' -> Some (Linear.map2 f l l')
        | _ ->
            invalid_arg "Gpt_oss.ptree: one model ties its head, one does not"
      in
      { tok; blocks; norm; head }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) p =
      Embedding.iter f p.tok;
      List.iter (B.iter f) p.blocks;
      Rms_norm.iter f p.norm;
      Option.iter (Linear.iter f) p.head
  end)

(* Attention: kaun's cached layer with sinks and YaRN's score scale. *)

let attention cfg layer b cache index x =
  let index =
    match layer with
    | Sliding -> Cache_index.window cfg.window index
    | Full -> index
  in
  let head_dim = cfg.head_dim and pos = Cache_index.positions index in
  let q = Rope.apply cfg.rope ~pos (Attention.split ~head_dim b.attn.q x) in
  let k = Rope.apply cfg.rope ~pos (Attention.split ~head_dim b.attn.k x) in
  let v = Attention.split ~head_dim b.attn.v x in
  let k, v, cache = Attention.Cache.extend index cache k v in
  let y =
    Attention.attend ~mask:(Cache_index.mask index) ~scale:cfg.attention_scale
      ~sinks:b.sinks q k v
  in
  (Attention.merge b.attn.out y, cache)

(* Forward passes *)

let block cfg layer b cache index x =
  let eps = cfg.norm_eps in
  let a, cache =
    attention cfg layer b cache index (Rms_norm.apply ~eps b.attn_norm x)
  in
  let x = Nx.add x a in
  let h = Rms_norm.apply ~eps b.ffn_norm x in
  let routing = Moe.route ~k:cfg.experts_per_token (Linear.apply b.router h) in
  let experts = Moe.apply ~limit:cfg.swiglu_limit b.moe routing h in
  (Nx.add x experts, cache)

module Cache = Attention.Cache.List

let cache ?placement cfg ~slots dtype =
  let place x =
    match placement with None -> x | Some p -> Nx.place (p Kv_heads ~axis:1) x
  in
  List.map
    (fun _ ->
      Attention.Cache.map place
        (Attention.Cache.make ~slots ~kv_heads:cfg.n_kv_heads
           ~head_dim:cfg.head_dim dtype))
    cfg.layers

let cached cfg p caches index ids =
  let rec go x rev layers blocks caches =
    match (layers, blocks, caches) with
    | [], [], [] -> (x, List.rev rev)
    | layer :: layers, b :: blocks, c :: caches ->
        let x, c = block cfg layer b c index x in
        go x (c :: rev) layers blocks caches
    | _ ->
        invalid_arg
          "Gpt_oss.cached: the configuration, the parameters and the caches \
           differ in their number of blocks"
  in
  go (Embedding.apply p.tok ids) [] cfg.layers p.blocks caches

let hidden cfg p ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  let nothing = cache cfg ~slots:0 (Nx.dtype p.norm.Rms_norm.gamma) in
  fst (cached cfg p nothing (Cache_index.whole ~batch ~seq ()) ids)

let logits cfg p h =
  let h = Rms_norm.apply ~eps:cfg.norm_eps p.norm h in
  match p.head with
  | Some l -> Linear.apply l h
  | None -> Nx.matmul h (Nx.transpose p.tok.Embedding.table)

(* Configuration from HuggingFace's config.json *)

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let config_of_json json =
  let missing name = failwith ("gpt-oss config.json: missing " ^ name) in
  let number_in json name =
    match json_mem name json with Jsont.Number (f, _) -> f | _ -> missing name
  in
  let number = number_in json in
  let int name = int_of_float (number name) in
  let head_dim = int "head_dim" in
  let scaling = json_mem "rope_scaling" json in
  (match (json_mem "rope_type" scaling, json_mem "truncate" scaling) with
  | Jsont.String ("yarn", _), Jsont.Bool (false, _) -> ()
  | _ ->
      failwith
        "gpt-oss config.json: rope_scaling must be yarn with truncate false");
  let field = number_in scaling in
  let factor = field "factor" in
  let rope =
    Rope.yarn ~theta:(number "rope_theta") ~head_dim ~factor
      ~beta_fast:(field "beta_fast") ~beta_slow:(field "beta_slow")
      ~original_context:
        (int_of_float (field "original_max_position_embeddings"))
  in
  let concentration = (0.1 *. log factor) +. 1.0 in
  let layers =
    match json_mem "layer_types" json with
    | Jsont.Array (kinds, _) ->
        List.map
          (function
            | Jsont.String ("sliding_attention", _) -> Sliding
            | Jsont.String ("full_attention", _) -> Full
            | _ -> failwith "gpt-oss config.json: unknown layer type")
          kinds
    | _ -> missing "layer_types"
  in
  if List.length layers <> int "num_hidden_layers" then
    failwith "gpt-oss config.json: layer_types and num_hidden_layers disagree";
  {
    vocab_size = int "vocab_size";
    dim = int "hidden_size";
    layers;
    window = int "sliding_window";
    n_heads = int "num_attention_heads";
    n_kv_heads = int "num_key_value_heads";
    head_dim;
    hidden_dim = int "intermediate_size";
    experts = int "num_local_experts";
    experts_per_token = int "num_experts_per_tok";
    swiglu_limit = number "swiglu_limit";
    norm_eps = number "rms_norm_eps";
    rope;
    attention_scale = (concentration ** 2.0) /. sqrt (float_of_int head_dim);
    tied =
      (match json_mem "tie_word_embeddings" json with
      | Jsont.Bool (b, _) -> b
      | _ -> false);
  }

(* Importing a HuggingFace checkpoint.

   The file names its tensors model.layers.{i}.self_attn.q_proj.weight, ... and
   stores every projection as [outputs; inputs], the transpose of [Linear]'s
   layout. Its q and k weights are laid out for the rotation that pairs feature
   i with i + head_dim / 2, which is [Rope]'s. Expert weights are either float,
   [experts; inputs; outputs] as [Moe] takes them, or packed as uint8 codes and
   exponents under the [_blocks] and [_scales] suffixes, which stay uint8:
   [Checkpoint.to_float] refuses them. A tied model has no lm_head entry. *)

let of_hf ?placement cfg dt ckpt =
  let place role ~axis x =
    match placement with None -> x | Some p -> Nx.place (p role ~axis) x
  in
  let whole x = place Whole ~axis:0 x in
  let float ~shape name = Checkpoint.to_float ~shape dt name ckpt in
  let norm name =
    { Rms_norm.gamma = whole (float ~shape:[| cfg.dim |] name) }
  in
  (* A column projection is cut along its outputs, bias included; a row
     projection along its inputs, and its bias is added whole. *)
  let linear role ~axis ?(bias = true) ~inputs ~outputs name =
    let b_role = match role with Row -> Whole | role -> role in
    {
      Linear.w =
        place role ~axis
          (Nx.matrix_transpose
             (float ~shape:[| outputs; inputs |] (name ^ ".weight")));
      b =
        (if bias then
           Some
             (place b_role ~axis:0
                (float ~shape:[| outputs |] (name ^ ".bias")))
         else None);
    }
  in
  let column = linear Column ~axis:1 and row = linear Row ~axis:0 in
  let experts ~inputs ~outputs name =
    match Checkpoint.find (name ^ "_blocks") ckpt with
    | None ->
        Moe.Float
          (place Experts ~axis:0
             (float ~shape:[| cfg.experts; inputs; outputs |] name))
    | Some _ ->
        let groups = inputs / 32 in
        let bytes ~shape name = Checkpoint.to_tensor ~shape Nx.uint8 name ckpt in
        let codes =
          bytes ~shape:[| cfg.experts; outputs; groups; 16 |] (name ^ "_blocks")
        in
        Moe.Quant
          (Nx_quant.map (fun x -> place Experts ~axis:0 x)
             (Nx_quant.mxfp4
                ~scales:
                  (bytes
                     ~shape:[| cfg.experts; outputs; groups |]
                     (name ^ "_scales"))
                (Nx.reshape [| cfg.experts; outputs; inputs / 2 |] codes)))
  in
  let q_dim = cfg.n_heads * cfg.head_dim in
  let kv_dim = cfg.n_kv_heads * cfg.head_dim in
  let block i =
    let at leaf = Printf.sprintf "model.layers.%d.%s" i leaf in
    {
      attn_norm = norm (at "input_layernorm.weight");
      attn =
        {
          Attention.q =
            column ~inputs:cfg.dim ~outputs:q_dim (at "self_attn.q_proj");
          k = column ~inputs:cfg.dim ~outputs:kv_dim (at "self_attn.k_proj");
          v = column ~inputs:cfg.dim ~outputs:kv_dim (at "self_attn.v_proj");
          out = row ~inputs:q_dim ~outputs:cfg.dim (at "self_attn.o_proj");
        };
      sinks =
        place Column ~axis:0
          (float ~shape:[| cfg.n_heads |] (at "self_attn.sinks"));
      ffn_norm = norm (at "post_attention_layernorm.weight");
      router =
        linear Whole ~axis:0 ~inputs:cfg.dim ~outputs:cfg.experts
          (at "mlp.router");
      moe =
        {
          Moe.gate_up =
            experts ~inputs:cfg.dim ~outputs:(2 * cfg.hidden_dim)
              (at "mlp.experts.gate_up_proj");
          gate_up_bias =
            place Experts ~axis:0
              (float
                 ~shape:[| cfg.experts; 2 * cfg.hidden_dim |]
                 (at "mlp.experts.gate_up_proj_bias"));
          down =
            experts ~inputs:cfg.hidden_dim ~outputs:cfg.dim
              (at "mlp.experts.down_proj");
          down_bias =
            place Experts ~axis:0
              (float ~shape:[| cfg.experts; cfg.dim |]
                 (at "mlp.experts.down_proj_bias"));
        };
    }
  in
  {
    tok =
      {
        Embedding.table =
          whole
            (float
               ~shape:[| cfg.vocab_size; cfg.dim |]
               "model.embed_tokens.weight");
      };
    blocks = List.mapi (fun i _ -> block i) cfg.layers;
    norm = norm "model.norm.weight";
    head =
      (if cfg.tied then None
       else
         Some
           (column ~bias:false ~inputs:cfg.dim ~outputs:cfg.vocab_size "lm_head"));
  }

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtype_of_string = function
  | "float32" -> Dtype Nx.float32
  | "bfloat16" -> Dtype Nx.bfloat16
  | d -> failwith ("--dtype must be float32 or bfloat16, got " ^ d)

let stored_dtype ckpt =
  let (Rune.Ptree.P table) = Checkpoint.get "model.embed_tokens.weight" ckpt in
  match Nx.dtype table with
  | Nx.BFloat16 -> Dtype Nx.bfloat16
  | Nx.Float32 -> Dtype Nx.float32
  | _ ->
      failwith
        "the checkpoint's embedding table is not a bfloat16 or float32 entry"

let from_file ?placement cfg dt path =
  of_hf ?placement cfg dt (Checkpoint.load path)

let from_pretrained ?placement repo_id dt =
  let cfg = config_of_json (Hf.load_config repo_id) in
  (cfg, of_hf ?placement cfg dt (Hf.load_checkpoint repo_id))
