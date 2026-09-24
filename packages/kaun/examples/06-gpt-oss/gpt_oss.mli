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
    Expert weights stay packed as the checkpoint stores them ({!Mxfp4}) and are
    dequantised inside the forward pass.

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

val map : ('a -> 'b) -> 'a params -> 'b params
(** [map f p] is [p] with [f] applied to every float leaf; packed expert weights
    are kept. [map (Nx.cast dt) p] converts precision. *)

val ptree : unit -> (module Nx.Ptree.S with type t = (float, 'b) Nx.t params)
(** [ptree ()] is the parameter tree the transformations take: every tensor of
    the model, the packed uint8 ones included, in the order [tok], the blocks
    ([attn_norm], [attn], [sinks], [ffn_norm], then the router, [gate_up], its
    bias, [down], its bias, a packed weight being [blocks] then [scales]),
    [norm], [head]. *)

val block_ptree :
  unit -> (module Nx.Ptree.S with type t = (float, 'b) Nx.t block)
(** [block_ptree ()] is the parameter tree of one block, in {!ptree}'s order:
    the tree a compiled block takes its weights as. *)

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
    reads and keeps nothing. A call of one token per sequence runs the experts
    in their {!Moe.Gather} form and any other call in their {!Moe.Dense} form.
*)

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

module Cache : Nx.Ptree.Uniform with type 'a t = 'a Kaun.Attention.Cache.t list
(** Decoding state: one key-value cache per block, in block order. *)

val cache :
  config -> slots:int -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t Cache.t
(** [cache cfg ~slots dtype] is an empty decoding state of [slots] slots per
    block, at the parameters' dtype. *)

val cached :
  config ->
  (float, 'b) Nx.t params ->
  (float, 'b) Nx.t Cache.t ->
  Kaun.Cache_index.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Cache.t
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
  ?device:string ->
  config ->
  (float, 'b) Nx.dtype ->
  Kaun.Checkpoint.t ->
  (float, 'b) Nx.t params
(** [of_hf cfg dt ckpt] is the model of the HuggingFace gpt-oss checkpoint
    [ckpt], with its float leaves at [dt]. Each entry is read by its name in the
    file with the shape [cfg] gives it. Projections are transposed to
    [inputs × outputs], a view. Experts stored as [_blocks] and [_scales] stay
    packed uint8 tensors, whatever [dt]. At the file's own dtype nothing is
    copied; at another one each float leaf is cast.

    With [device], each leaf, float or uint8, is placed on it with [Nx.place] as
    it is built, so a function compiled for [device] that captures the model
    uploads nothing and the host holds one leaf at a time.

    Raises [Invalid_argument], naming the entry, if one is missing, has another
    shape than [cfg] says, or has a dtype the leaf cannot take. *)

val from_file :
  ?device:string ->
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
  ?device:string ->
  string ->
  (float, 'b) Nx.dtype ->
  config * (float, 'b) Nx.t params
(** [from_pretrained repo_id dt] downloads the repository's configuration and
    checkpoint, single-file or sharded (cached afterwards), and is the model at
    [dt]. *)
