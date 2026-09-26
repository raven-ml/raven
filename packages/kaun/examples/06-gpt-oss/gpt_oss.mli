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
    and {!Nx_quant.apply} multiplies by them.

    The model is written on the decode contract: {!hidden}, {!cached} and
    {!logits} are its forward passes. *)

type layer = Sliding | Full  (** The type for attention kinds. *)

type config = {
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
  rope : Kaun.Rope.t;  (** {!Kaun.Rope.yarn}. *)
  attention_scale : float;
      (** YaRN's temperature, [(0.1 * ln factor + 1) ^ 2 / sqrt head_dim], in
          place of [1 / sqrt head_dim]. The reference scales its cosines and
          sines by [0.1 * ln factor + 1] instead, which is the same attention.
      *)
  tied : bool;  (** Whether the head is the token table. *)
}
(** The type for gpt-oss hyperparameters. The expert count and widths are read
    from the parameters. *)

type 'a block = {
  attn_norm : 'a Kaun.Rms_norm.t;
  attn : 'a Kaun.Attention.t;  (** Every projection has a bias. *)
  sinks : 'a;  (** [[| n_heads |]]. *)
  ffn_norm : 'a Kaun.Rms_norm.t;
  router : 'a Kaun.Linear.t;  (** Model width to experts, with a bias. *)
  moe : 'a Moe.t;
}
(** The type for one block over payload ['a]. Packed expert weights are uint8
    tensors whatever ['a]. *)

type 'a params = {
  tok : 'a Kaun.Embedding.t;
  blocks : 'a block list;
  norm : 'a Kaun.Rms_norm.t;
  head : 'a Kaun.Linear.t option;  (** [None]: tied to [tok]. *)
}
(** The type for gpt-oss parameters over payload ['a]. *)

type t = Nx.float32_t params

module Block : Nx.Ptree.S with type 'a t = 'a block
(** A block's structure. It walks [attn_norm], [attn], [sinks], [ffn_norm],
    [router] and [moe], whose weights report their case (see {!Moe.walk}):
    [Nx.Ptree.instantiate (module Block)] is what a compiled block program takes
    its weights as. *)

module Params : Nx.Ptree.S with type 'a t = 'a params
(** The parameters' structure: [tok], each block at [blocks.i] as {!Block} walks
    it, [norm], then [head] when it is present. The packed expert weights are
    fixed tensors, so [Nx.Ptree.instantiate (module Params)] walks every tensor
    of the model, and [Nx.Ptree.cast (module Params) dt p] converts precision
    and keeps the packed weights. *)

(** {1:placement Placement}

    {!of_hf} and {!cache} place each leaf they build with
    [placement role ~axis]: [role] is the cut a tensor-parallel or
    expert-parallel placement makes in the leaf and [axis] the axis of the leaf
    that cut runs along, [0] for [Whole]. One device ignores both:
    [fun _ ~axis:_ -> p]. *)

(** The type for the cuts of a tensor-parallel or expert-parallel placement. *)
type role =
  | Whole
      (** Kept whole: the token table, the norms, the router and the attention
          output's bias. *)
  | Column
      (** Cut along its outputs: the query, key and value projections with their
          biases, the sinks, and an untied head. *)
  | Row  (** Cut along its inputs: the attention's output projection. *)
  | Experts
      (** Cut along the expert axis: the expert weights, packed parts included,
          and their biases. *)
  | Kv_heads  (** Cut along its heads: a cache pool. *)

val expert_parallel : Nx.Device.t list -> role -> axis:int -> Nx.Placement.t
(** [expert_parallel ds] splits the experts over [ds] along their expert axis
    and keeps every other leaf, and the caches, a copy on each device: the
    [placement] of {!of_hf} and {!cache} for expert parallelism. Over one device
    every leaf is whole there. *)

(** {1:attention Attention} *)

val attention :
  config ->
  layer ->
  (float, 'b) Nx.t block ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t ->
  Kaun.Cache_index.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Kaun.Attention.Cache.t
(** [attention cfg layer b cache index x] is causal self-attention of the
    normalised tokens [x], of shape [[| batch; seq; dim |]], through [cache],
    and the cache with their keys and values written: {!Kaun.Attention.cached}
    composed from its pieces, with [b.sinks] and [cfg.attention_scale] given to
    {!Kaun.Attention.attend}. A {!Sliding} layer sees the last [cfg.window]
    positions, through {!Kaun.Cache_index.window}. *)

(** {1:forward Forward passes}

    One fold over the blocks: [hidden cfg p ids] is
    [fst (cached cfg p caches index ids)] over {!Kaun.Cache_index.whole}, which
    reads and keeps nothing. *)

val block :
  config ->
  layer ->
  (float, 'b) Nx.t block ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t ->
  Kaun.Cache_index.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Kaun.Attention.Cache.t
(** [block cfg layer b cache index x] is one pre-norm block of kind [layer]
    applied to the residual stream [x], of shape [[| batch; seq; dim |]],
    through [cache], and the cache with the tokens' keys and values written.
    {!cached} folds it over the blocks; a driver that runs each block as its own
    compiled program calls it directly. *)

val hidden :
  config ->
  (float, 'b) Nx.t params ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t
(** [hidden cfg p ids] is the residual stream after the last block for the
    [[| batch; seq |]] id tensor [ids], of shape [[| batch; seq; dim |]]. Every
    token attends to the tokens before it. *)

val cache :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  slots:int ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t list
(** [cache cfg ~slots dtype] is an empty decoding state: one key-value cache per
    block, in block order, each of [slots] slots, at the parameters' dtype. Its
    structure is
    [Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))].

    With [placement], each pool is placed with [placement Kv_heads ~axis:1], so
    a compiled step finds the caches where the model is from its first call. *)

val cached :
  config ->
  (float, 'b) Nx.t params ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t list ->
  Kaun.Cache_index.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Kaun.Attention.Cache.t list
(** [cached cfg p caches index ids] is the residual stream of the tokens [ids],
    which sit where [index] says and attend through [caches], and the caches
    with their keys and values written. See {!Kaun.Attention.cached}. *)

val logits :
  config -> (float, 'b) Nx.t params -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [logits cfg p h] is the final norm and the language-model head applied to a
    residual stream, per position. Select positions first. *)

(** {1:loading Pretrained weights} *)

val config_of_json : Jsont.json -> config
(** [config_of_json json] reads HuggingFace's [config.json].

    Raises [Failure] on a missing field or a rotary scaling other than YaRN
    without truncation. *)

val of_hf :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  (float, 'b) Nx.dtype ->
  Kaun.Checkpoint.t ->
  (float, 'b) Nx.t params
(** [of_hf cfg dt ckpt] is the model of the HuggingFace gpt-oss checkpoint
    [ckpt], with its float leaves at [dt]. Each entry is read by its name in the
    file with the shape [cfg] gives it. Projections are transposed to
    [inputs × outputs], a view. Experts stored as [_blocks] and [_scales] are an
    {!Nx_quant.mxfp4} weight over the file's bytes, whatever [dt]. At the file's
    own dtype nothing is copied; at another one each float leaf is cast.

    With [placement], each float leaf is placed with
    [Nx.place (placement role ~axis)] as it is built (see {!role}), and each
    packed weight with [Nx_quant.place (placement Experts ~axis:0)], so a
    function compiled where the model is that captures it uploads nothing and
    the host holds one leaf at a time.

    Raises [Invalid_argument], naming the entry, if one is missing, has another
    shape than [cfg] says, or has a dtype the leaf cannot take. *)

val from_file :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  (float, 'b) Nx.dtype ->
  string ->
  (float, 'b) Nx.t params
(** [from_file cfg dt path] is {!of_hf} on the safetensors file [path]. *)

type dtype =
  | Dtype : (float, 'b) Nx.dtype -> dtype
      (** A floating-point dtype chosen at run time. *)

val dtype_of_string : string -> dtype
(** [dtype_of_string s] is the dtype named ["float32"] or ["bfloat16"]. Raises
    [Failure] on another name. *)

val stored_dtype : Kaun.Checkpoint.t -> dtype
(** [stored_dtype ckpt] is the dtype [ckpt] stores its embedding table at, the
    dtype at which {!of_hf} casts nothing. *)

val from_pretrained :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  string ->
  (float, 'b) Nx.dtype ->
  config * (float, 'b) Nx.t params
(** [from_pretrained repo_id dt] downloads the repository's configuration and
    checkpoint, single-file or sharded (cached afterwards), and is the model at
    [dt]. *)
