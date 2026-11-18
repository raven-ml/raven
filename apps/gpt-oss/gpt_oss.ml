(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

type layer = Sliding | Full

type 'd config = {
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
  rope : 'd Rope.t;
  attention_scale : float;
  tied : bool;
}

let place_config at cfg = { cfg with rope = Rope.place at cfg.rope }

type ('a, 'd) projection =
  | Float of ('a, 'd) Linear.t
  | Quant of { w : 'd Nx_quant.t; b : 'a option }

type ('a, 'd) attn = { qkv : ('a, 'd) projection; out : ('a, 'd) projection }

type ('a, 'd) block = {
  attn_norm : ('a, 'd) Rms_norm.t;
  attn : ('a, 'd) attn;
  sinks : 'a;
  ffn_norm : ('a, 'd) Rms_norm.t;
  router : ('a, 'd) Linear.t;
  moe : ('a, 'd) Moe.t;
}

type ('a, 'd) params = {
  tok : ('a, 'd) Embedding.t;
  blocks : ('a, 'd) block list;
  norm : ('a, 'd) Rms_norm.t;
  head : ('a, 'd) projection option;
}

type 'd t = ('d Nx.float32_t, 'd) params
type role = Whole | Column | Row | Experts | Kv_heads

let expert_parallel s role ~axis =
  match role with
  | Experts -> Nx.Placement.split ~axis s
  | Whole | Column | Row | Kv_heads -> Nx.Placement.on s

(* Projections *)

let project p x =
  match p with
  | Float l -> Linear.apply l x
  | Quant { w; b } -> (
      let y = Nx_quant.apply w x in
      match b with None -> y | Some b -> Nx.add y b)

let walk_projection c =
  let open Nx.Ptree.Walk in
  function
  | Float l ->
      case c "float";
      Float (Linear.walk c l)
  | Quant { w; b } ->
      case c "quant";
      let w = field c "w" Nx_quant.walk w in
      let b = field c "b" (option leaf) b in
      Quant { w; b }

let walk_attn c a =
  let open Nx.Ptree.Walk in
  let qkv = field c "qkv" walk_projection a.qkv in
  let out = field c "out" walk_projection a.out in
  { qkv; out }

(* Structures *)

module Block = struct
  type nonrec ('a, 'd) t = ('a, 'd) block

  let walk c b =
    let open Nx.Ptree.Walk in
    let attn_norm = field c "attn_norm" Rms_norm.walk b.attn_norm in
    let attn = field c "attn" walk_attn b.attn in
    let sinks = field c "sinks" leaf b.sinks in
    let ffn_norm = field c "ffn_norm" Rms_norm.walk b.ffn_norm in
    let router = field c "router" Linear.walk b.router in
    let moe = field c "moe" Moe.walk b.moe in
    { attn_norm; attn; sinks; ffn_norm; router; moe }
end

module Params = struct
  type nonrec ('a, 'd) t = ('a, 'd) params

  let walk c p =
    let open Nx.Ptree.Walk in
    let tok = field c "tok" Embedding.walk p.tok in
    let blocks = field c "blocks" (list Block.walk) p.blocks in
    let norm = field c "norm" Rms_norm.walk p.norm in
    let head = field c "head" (option walk_projection) p.head in
    { tok; blocks; norm; head }
end

(* Attention: kaun's cached layer with sinks and YaRN's score scale, over
   projections that may be packed. [split] and [merge] are kaun's
   Attention.split and Attention.merge over {!project}, [split] of the fused
   projection: its outputs are, for each key-value head, the [group] query
   heads that read it, its key head and its value head (of_hf). Query head
   [h] reads key-value head [h / group], so merging the key-value and group
   axes orders the query heads as the checkpoint does. *)

let split cfg p x =
  let y = project p x in
  let batch = Nx.dim 0 y and seq = Nx.dim 1 y in
  let kv = cfg.n_kv_heads and d = cfg.head_dim in
  let group = cfg.n_heads / kv in
  let y = Nx.reshape [| batch; seq; kv; group + 2; d |] y in
  let heads x = Nx.swapaxes 1 2 x in
  let q = Nx.slice [ A; A; A; R (0, group) ] y in
  ( heads (Nx.reshape [| batch; seq; cfg.n_heads; d |] q),
    heads (Nx.slice [ A; A; A; I group ] y),
    heads (Nx.slice [ A; A; A; I (group + 1) ] y) )

let merge p y =
  let batch = Nx.dim 0 y and heads = Nx.dim 1 y in
  let seq = Nx.dim 2 y and d = Nx.dim 3 y in
  project p (Nx.reshape [| batch; seq; heads * d |] (Nx.swapaxes 1 2 y))

let windows cfg index =
  let sliding = Cache_index.window cfg.window index
  and full = Cache_index.window max_int index in
  function Sliding -> sliding | Full -> full

let attention cfg b cache index x =
  let pos = Cache_index.positions index in
  let q, k, v = split cfg b.attn.qkv x in
  let q = Rope.apply cfg.rope ~pos q and k = Rope.apply cfg.rope ~pos k in
  let kv, cache = Attention.Cache.extend index cache k v in
  let y =
    Attention.attend
      ~mask:(Cache_index.visible index)
      ~scale:cfg.attention_scale ~sinks:b.sinks q kv
  in
  (merge b.attn.out y, cache)

(* Forward passes *)

(* [x] closed where it lies: the experts' output, cut along their features,
   is a sum the block asked for (Moe), which a norm reads whole. On one
   device it is [x]. *)
let close x = Nx.place (Nx.Placement.closed (Nx.placement x)) x

let block cfg b cache index x =
  let eps = cfg.norm_eps in
  let a, cache =
    attention cfg b cache index (Rms_norm.apply ~eps b.attn_norm x)
  in
  let x = close (Nx.add x a) in
  let h = Rms_norm.apply ~eps b.ffn_norm x in
  let routing = Moe.route ~k:cfg.experts_per_token (Linear.apply b.router h) in
  let experts = Moe.apply ~limit:cfg.swiglu_limit b.moe routing h in
  (close (Nx.add x experts), cache)

(* The pools, made where the kv-heads role places them. *)
let cache ~placement cfg ~slots dtype =
  let at = placement Kv_heads ~axis:1 in
  List.map
    (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:cfg.n_kv_heads
           ~head_dim:cfg.head_dim dtype at)
    cfg.layers

let cached cfg p caches index ids =
  let index = windows cfg index in
  let rec go x rev layers blocks caches =
    match (layers, blocks, caches) with
    | [], [], [] -> (x, List.rev rev)
    | layer :: layers, b :: blocks, c :: caches ->
        let x, c = block cfg b c (index layer) x in
        go x (c :: rev) layers blocks caches
    | _ ->
        invalid_arg
          "Gpt_oss.cached: the configuration, the parameters and the caches \
           differ in their number of blocks"
  in
  go (Embedding.apply p.tok ids) [] cfg.layers p.blocks caches

(* The whole index reads no cache, so its empty pools lie whole beside the ids,
   and its positions where the ids are. *)
let hidden cfg p ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  let at = Nx.placement ids in
  let whole = Nx.Placement.uncut at in
  let nothing =
    cache
      ~placement:(fun _ ~axis:_ -> whole)
      cfg ~slots:0
      (Nx.dtype p.norm.Rms_norm.gamma)
  in
  let index = Cache_index.place at (Cache_index.whole ~batch ~seq ()) in
  fst (cached cfg p nothing index ids)

let logits cfg p h =
  let h = Rms_norm.apply ~eps:cfg.norm_eps p.norm h in
  match p.head with
  | Some p -> project p h
  | None -> Nx.matmul h (Nx.transpose p.tok.Embedding.table)

(* Configuration from HuggingFace's config.json *)

let config_of_json json =
  let get ?absent name codec =
    match Jsont.Json.decode (Jsont.mem ?absent name codec) json with
    | Ok v -> v
    | Error e -> failwith ("gpt-oss config.json: " ^ e)
  in
  let int name = get name Jsont.int and number name = get name Jsont.number in
  let scaling name codec = get "rope_scaling" (Jsont.mem name codec) in
  if
    scaling "rope_type" Jsont.string <> "yarn"
    || scaling "truncate" Jsont.bool
  then
    failwith
      "gpt-oss config.json: rope_scaling must be yarn with truncate false";
  let head_dim = int "head_dim" and factor = scaling "factor" Jsont.number in
  let rope =
    Rope.yarn ~theta:(number "rope_theta") ~head_dim ~factor
      ~beta_fast:(scaling "beta_fast" Jsont.number)
      ~beta_slow:(scaling "beta_slow" Jsont.number)
      ~original_context:(scaling "original_max_position_embeddings" Jsont.int)
      ~context:(int "max_position_embeddings")
  in
  let concentration = (0.1 *. log factor) +. 1.0 in
  let layer = function
    | "sliding_attention" -> Sliding
    | "full_attention" -> Full
    | _ -> failwith "gpt-oss config.json: unknown layer type"
  in
  let layers =
    get "layer_types" (Jsont.list (Jsont.map ~dec:layer Jsont.string))
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
    tied = get ~absent:false "tie_word_embeddings" Jsont.bool;
  }

(* Importing HuggingFace weights.

   The file names its tensors model.layers.{i}.self_attn.q_proj.weight, ... and
   stores every projection as [outputs; inputs], the transpose of [Linear]'s
   layout. Its q and k weights are laid out for the rotation that pairs feature
   i with i + head_dim / 2, which is [Rope]'s. Expert weights are either float,
   [experts; inputs; outputs] as [Moe] takes them, or packed as uint8 codes and
   exponents under the [_blocks] and [_scales] suffixes, which stay uint8:
   [Nx_io.Archive.float] refuses them. A tied model has no lm_head entry. *)

type packing = Unpacked | Q8_0

let of_hf ?(pack = Unpacked) ~placement cfg dt weights =
  let place role ~axis x = Nx.place (placement role ~axis) x in
  let whole x = place Whole ~axis:0 x in
  let float ~shape name = Nx_io.Archive.float ~shape dt name weights in
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
  (* A packed projection keeps the file's [outputs; inputs], so a cut along
     the outputs is along its axis 0 and one along its inputs along its axis
     1. It is quantised where it is placed, compiled, so a set that computes
     under [Rune.jit] alone quantises too. *)
  let quantize =
    Rune.jit
      Nx.Ptree.(tensor @-> returns (v Nx_quant.walk))
      Nx_quant.quantize_q8_0
  in
  let projection role ~axis ~inputs ~outputs name =
    match pack with
    | Unpacked -> Float (linear role ~axis ~inputs ~outputs name)
    | Q8_0 ->
        let b_role = match role with Row -> Whole | role -> role in
        let w = float ~shape:[| outputs; inputs |] (name ^ ".weight") in
        let b = float ~shape:[| outputs |] (name ^ ".bias") in
        Quant
          {
            w = quantize (place role ~axis:(1 - axis) w);
            b = Some (place b_role ~axis:0 b);
          }
  in
  let row = projection Row ~axis:0 in
  (* The query, key and value projections as one, its outputs grouped by
     key-value head: each one's [group] query heads, then its key head and
     its value head. A cut along the outputs between key-value heads so holds
     on each device the heads that attend together, as cuts of the three
     apart do. The three are placed in the file's [outputs; inputs] and
     joined where they are placed, compiled, so a set that computes under
     [Rune.jit] alone joins them too; the join copies. Q8_0 quantises each
     row of 32-value blocks alone, so the joined weight packs to the rows
     the three would. *)
  let group = cfg.n_heads / cfg.n_kv_heads in
  let join =
    let by_kv heads x =
      let rest = Array.sub (Nx.shape x) 1 (Nx.ndim x - 1) in
      Nx.reshape (Array.append [| cfg.n_kv_heads; heads; cfg.head_dim |] rest) x
    in
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
      (fun q k v ->
        let y =
          Nx.concatenate ~axis:1 [ by_kv group q; by_kv 1 k; by_kv 1 v ]
        in
        let rest = Array.sub (Nx.shape y) 3 (Nx.ndim y - 3) in
        let outputs = cfg.n_kv_heads * (group + 2) * cfg.head_dim in
        Nx.reshape (Array.append [| outputs |] rest) y)
  in
  let qkv at =
    let joined leaf rest =
      let part name heads =
        let shape = Array.append [| heads * cfg.head_dim |] rest in
        place Column ~axis:0 (float ~shape (at ("self_attn." ^ name ^ leaf)))
      in
      join
        (part "q_proj" cfg.n_heads)
        (part "k_proj" cfg.n_kv_heads)
        (part "v_proj" cfg.n_kv_heads)
    in
    let w = joined ".weight" [| cfg.dim |] and b = joined ".bias" [||] in
    match pack with
    | Unpacked -> Float { Linear.w = Nx.matrix_transpose w; b = Some b }
    | Q8_0 -> Quant { w = quantize w; b = Some b }
  in
  let experts ~inputs ~outputs name =
    match Nx_io.Archive.find (name ^ "_blocks") weights with
    | None ->
        Moe.Float
          (place Experts ~axis:0
             (float ~shape:[| cfg.experts; inputs; outputs |] name))
    | Some _ ->
        let groups = inputs / 32 in
        let bytes ~shape name =
          Nx_io.Archive.tensor ~shape Nx.uint8 name weights
        in
        let codes =
          bytes ~shape:[| cfg.experts; outputs; groups; 16 |] (name ^ "_blocks")
        in
        let w =
          Nx_quant.mxfp4
            ~scales:
              (bytes
                 ~shape:[| cfg.experts; outputs; groups |]
                 (name ^ "_scales"))
            (Nx.reshape [| cfg.experts; outputs; inputs / 2 |] codes)
        in
        Moe.Quant (Nx_quant.place (placement Experts ~axis:0) w)
  in
  let q_dim = cfg.n_heads * cfg.head_dim in
  let block i =
    let at leaf = Printf.sprintf "model.layers.%d.%s" i leaf in
    {
      attn_norm = norm (at "input_layernorm.weight");
      attn =
        {
          qkv = qkv at;
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
         let w = float ~shape:[| cfg.vocab_size; cfg.dim |] "lm_head.weight" in
         match pack with
         | Q8_0 ->
             Some (Quant { w = quantize (place Column ~axis:0 w); b = None })
         | Unpacked ->
             (* A copy in [inputs; outputs], since tolk's bfloat16
                matrix-vector kernel reads the transposed view 3 to 4 times
                slower. It is made after the head is placed, as a host copy
                reads the file's pages in transposed order, ten times slower
                cold, and compiled, so a set that computes under [Rune.jit]
                alone makes it too. *)
             let w = place Column ~axis:1 (Nx.matrix_transpose w) in
             Some (Float { Linear.w = Rune.jit' Nx.copy w; b = None }));
  }

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtype_of_string = function
  | "float32" -> Dtype Nx.float32
  | "bfloat16" -> Dtype Nx.bfloat16
  | d -> failwith ("--dtype must be float32 or bfloat16, got " ^ d)

let stored_dtype weights =
  let name = "model.embed_tokens.weight" in
  match Nx_io.Archive.find name weights with
  | None -> failwith (name ^ ": no entry in the archive")
  | Some (Nx.P table) -> (
      match Nx.dtype table with
      | Nx.Dtype.Bfloat16 -> Dtype Nx.bfloat16
      | Nx.Dtype.Float32 -> Dtype Nx.float32
      | _ -> failwith (name ^ " is not a bfloat16 or float32 entry"))
