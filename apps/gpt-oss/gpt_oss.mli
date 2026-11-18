(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** gpt-oss (OpenAI, 2025) from kaun layers.

    A pre-norm decoder whose blocks pair grouped-query attention with a mixture
    of experts ({!Moe}). Attention has biases, one sink logit per query head and
    YaRN rotary positions; layers alternate between a sliding window and full
    attention, the first one sliding, over one {!Kaun.Cache_index}. {!attention}
    composes kaun's attention pieces with the sinks and YaRN's score scale.
    Expert weights stay packed as the checkpoint stores them ({!Nx_quant.t}),
    and {!Nx_quant.apply} multiplies by them. The attention projections and
    an untied head are floats, or packed as Q8_0 where the caller asks
    ({!packing}).

    The model is written on the decode contract: {!hidden}, {!cached} and
    {!logits} are its forward passes. *)

type layer = Sliding | Full  (** The type for attention kinds. *)

type 'd config = {
  vocab_size : int;
  dim : int;
  layers : layer list;  (** One per block. *)
  window : int;  (** Positions a {!Sliding} layer sees, the token included. *)
  n_heads : int;
  n_kv_heads : int;  (** Key-value heads; divides [n_heads]. *)
  head_dim : int;
  hidden_dim : int;  (** The width inside an expert. *)
  experts : int;
  experts_per_token : int;
  swiglu_limit : float;
  norm_eps : float;
  rope : 'd Kaun.Rope.t;  (** {!Kaun.Rope.yarn}, where the model reads it. *)
  attention_scale : float;
      (** YaRN's temperature, [(0.1 * ln factor + 1) ^ 2 / sqrt head_dim], in
          place of [1 / sqrt head_dim]. The reference scales its cosines and
          sines by [0.1 * ln factor + 1] instead, which is the same attention.
      *)
  tied : bool;  (** Whether the head is the token table. *)
}
(** The type for gpt-oss hyperparameters, with the rotary schedule on the set
    ['d]. *)

val place_config : 'e Nx.Placement.t -> 'd config -> 'e config
(** [place_config at cfg] is [cfg] with its rotary schedule at [at]
    ({!Kaun.Rope.place}). {!config_of_json} makes it on the host; a model on
    other devices places it once, where its parameters lie. *)

(** The type for a projection's weights over payload ['a]. *)
type ('a, 'd) projection =
  | Float of ('a, 'd) Kaun.Linear.t
  | Quant of { w : 'd Nx_quant.t; b : 'a option }
      (** A packed weight of shape [[| outputs; inputs |]], as GGUF files
          store it, and its bias. *)

val project :
  ((float, 'b, 'd) Nx.t, 'd) projection ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [project p x] is [p] applied to [x], whose last axis is [p]'s inputs:
    {!Kaun.Linear.apply}, or {!Nx_quant.apply} from the bias, each
    output rounded once. *)

type ('a, 'd) attn = { qkv : ('a, 'd) projection; out : ('a, 'd) projection }
(** The type for the attention's projections over payload ['a]: the query,
    key and value as one product, and the output. Each has a bias. [qkv]'s
    outputs are grouped by key-value head: for each, in order, the
    [n_heads / n_kv_heads] query heads that read it, then its key head and
    its value head, [head_dim] features each. *)

type ('a, 'd) block = {
  attn_norm : ('a, 'd) Kaun.Rms_norm.t;
  attn : ('a, 'd) attn;
  sinks : 'a;  (** [[| n_heads |]]. *)
  ffn_norm : ('a, 'd) Kaun.Rms_norm.t;
  router : ('a, 'd) Kaun.Linear.t;  (** Model width to experts, with a bias. *)
  moe : ('a, 'd) Moe.t;
}
(** The type for one block over payload ['a]. Packed expert weights are uint8
    tensors whatever ['a]. *)

type ('a, 'd) params = {
  tok : ('a, 'd) Kaun.Embedding.t;
  blocks : ('a, 'd) block list;
  norm : ('a, 'd) Kaun.Rms_norm.t;
  head : ('a, 'd) projection option;  (** [None]: tied to [tok]. *)
}
(** The type for gpt-oss parameters over payload ['a]. *)

module Block : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) block
(** A block's structure. It walks [attn_norm], [attn]'s [qkv] and [out],
    [sinks], [ffn_norm], [router] and [moe]. A projection reports its case,
    ["float"], walked as {!Kaun.Linear.walk}, or ["quant"], its [w] by
    {!Nx_quant.walk} and its [b]; the experts' weights report theirs (see
    {!Moe.walk}):
    [Nx.Ptree.instantiate (module Block)] is what a compiled block program takes
    its weights as. *)

module Params : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) params
(** The parameters' structure: [tok], each block at [blocks.i] as {!Block} walks
    it, [norm], then [head] when it is present. The packed expert weights are
    fixed tensors, so [Nx.Ptree.instantiate (module Params)] walks every tensor
    of the model, and [Nx.Ptree.cast (module Params) dt p] converts precision
    and keeps the packed weights. *)

(** {1:placement Placement}

    {!of_hf} and {!cache} make the arrays a model starts from, the file's
    entries and empty pools, and place each leaf they build with
    [placement role ~axis]: [role] is the cut a tensor-parallel or
    expert-parallel placement makes in the leaf and [axis] the axis of the leaf
    that cut runs along, [0] for [Whole]. One device ignores both:
    [fun _ ~axis:_ -> p], and the host is [fun _ ~axis:_ -> Nx.Host.on]. *)

(** The type for the cuts of a tensor-parallel or expert-parallel placement. *)
type role =
  | Whole
      (** Kept whole: the token table, the norms, the router and the attention
          output's bias. *)
  | Column
      (** Cut along its outputs: the query, key and value projection with its
          bias, the sinks, and an untied head. *)
  | Row  (** Cut along its inputs: the attention's output projection. *)
  | Experts
      (** Cut along the expert axis: the expert weights, packed parts included,
          and their biases. *)
  | Kv_heads  (** Cut along its heads: a cache pool. *)

val expert_parallel : 'd Nx.Devices.t -> role -> axis:int -> 'd Nx.Placement.t
(** [expert_parallel s] splits the experts over [s] along their expert axis and
    keeps every other leaf, and the caches, a copy on each device: the
    [placement] of {!of_hf} and {!cache} for expert parallelism. Over one device
    every leaf is whole there. *)

(** {1:attention Attention} *)

val windows :
  'd config -> 'e Kaun.Cache_index.t -> layer -> 'e Kaun.Cache_index.t
(** [windows cfg index layer] is [index] under a layer of kind [layer]'s window
    ({!Kaun.Cache_index.window}): the last [cfg.window] positions for a
    {!Sliding} layer, and for a {!Full} one a window wider than every position,
    which hides nothing. The two differ in their window's size alone, so one
    compiled block serves both. Each is resolved
    ({!Kaun.Cache_index.resolve}), so the blocks of a kind share one
    computation of its addressing. Apply it to [cfg index] once per call: the partial
    application holds both indices. *)

val attention :
  'd config ->
  ((float, 'b, 'd) Nx.t, 'd) block ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t ->
  'd Kaun.Cache_index.t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t * ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t
(** [attention cfg b cache index x] is causal self-attention of the normalised
    tokens [x], of shape [[| batch; seq; dim |]], through [cache], and the
    cache with their keys and values written: {!Kaun.Attention.cached}
    composed from its pieces over {!project}, with [b.sinks] and
    [cfg.attention_scale] given to {!Kaun.Attention.attend}. [index] carries
    the layer's window ({!windows}). *)

(** {1:forward Forward passes}

    One fold over the blocks: [hidden cfg p ids] is
    [fst (cached cfg p caches index ids)] over {!Kaun.Cache_index.whole}, which
    reads and keeps nothing. *)

val block :
  'd config ->
  ((float, 'b, 'd) Nx.t, 'd) block ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t ->
  'd Kaun.Cache_index.t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t * ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t
(** [block cfg b cache index x] is one pre-norm block applied to the residual
    stream [x], of shape [[| batch; seq; dim |]], through [cache], and the
    cache with the tokens' keys and values written. [index] carries the
    block's window ({!windows}). {!cached} folds it over the blocks; a driver
    that runs each block as its own compiled program calls it directly. *)

val hidden :
  'd config ->
  ((float, 'b, 'd) Nx.t, 'd) params ->
  'd Nx.int64_t ->
  (float, 'b, 'd) Nx.t
(** [hidden cfg p ids] is the residual stream after the last block for the
    [[| batch; seq |]] id tensor [ids], of shape [[| batch; seq; dim |]]. Every
    token attends to the tokens before it. *)

val cache :
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  'e config ->
  slots:int ->
  (float, 'b) Nx.dtype ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [cache ~placement cfg ~slots dtype] is an empty decoding state: one
    key-value cache per block, in block order, each of [slots] slots, at the
    parameters' dtype. Its structure is
    [Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))].

    Each pool is made at [placement Kv_heads ~axis:1], where a compiled step
    finds it from its first call. *)

val cached :
  'd config ->
  ((float, 'b, 'd) Nx.t, 'd) params ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list ->
  'd Kaun.Cache_index.t ->
  'd Nx.int64_t ->
  (float, 'b, 'd) Nx.t * ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [cached cfg p caches index ids] is the residual stream of the tokens [ids],
    which sit where [index] says and attend through [caches], each layer under
    its window ({!windows}), and the caches with their keys and values written.
    See {!Kaun.Attention.cached}. *)

val logits :
  'e config ->
  ((float, 'b, 'd) Nx.t, 'd) params ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [logits cfg p h] is the final norm and the language-model head applied to a
    residual stream, per position. Select positions first. *)

(** {1:loading Pretrained weights} *)

val config_of_json : Jsont.json -> Nx.host config
(** [config_of_json json] reads HuggingFace's [config.json].

    Raises [Failure] on a missing field or a rotary scaling other than YaRN
    without truncation. *)

(** The type for how {!of_hf} stores the attention projections and an untied
    head. *)
type packing =
  | Unpacked  (** Floats at the model's dtype: the checkpoint's model. *)
  | Q8_0
      (** {!Nx_quant.quantize_q8_0} of those floats, as GGUF files of gpt-oss
          store them. *)

val of_hf :
  ?pack:packing ->
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  'c config ->
  (float, 'b) Nx.dtype ->
  'e Nx_io.Archive.t ->
  ((float, 'b, 'd) Nx.t, 'd) params
(** [of_hf ~placement cfg dt weights] is the model of the HuggingFace gpt-oss
    [weights], each entry read by its name in the file with the shape [cfg]
    gives it. Float leaves are at [dt], cast where the file stores another
    dtype; experts stored as [_blocks] and [_scales] stay packed as
    {!Nx_quant.mxfp4} weights over the file's bytes. Under [pack], [Unpacked]
    by default, the attention projections and an untied head are stored as
    {!packing} says. Each leaf is placed as it is built, a float leaf at
    [placement role ~axis] (see {!role}) and a packed expert weight at
    [placement Experts ~axis:0].

    Raises [Failure], naming the entry, if one is missing, has another shape
    than [cfg] says, or has a dtype the leaf cannot take. *)

type dtype =
  | Dtype : (float, 'b) Nx.dtype -> dtype
      (** A floating-point dtype chosen at run time. *)

val stored_dtype : 'd Nx_io.Archive.t -> dtype
(** [stored_dtype weights] is the dtype [weights] stores its embedding table at,
    the dtype at which {!of_hf} casts nothing.

    Raises [Failure] if the table is missing or is not a bfloat16 or float32
    entry. *)
