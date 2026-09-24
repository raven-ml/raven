(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Llama (Touvron et al., 2023; Llama 3, 2024) from kaun layers.

    A pre-norm decoder: RMS normalization, grouped-query attention with rotary
    positions, a SwiGLU feed-forward. The model is a plain record of {!Kaun}
    layers written on the decode contract: {!hidden}, {!cached} and {!logits}
    are its forward passes and {!Params} its checkpoint plumbing. {!of_hf}
    adapts the HuggingFace checkpoint — [torch.nn.Linear] orientation and naming
    — onto {!Params}' names, and {!from_pretrained} runs the whole pipeline. *)

type config = {
  vocab_size : int;
  dim : int;
  n_layers : int;
  n_heads : int;
  n_kv_heads : int;  (** Key-value heads; divides [n_heads]. *)
  head_dim : int;
  hidden_dim : int;  (** SwiGLU inner width. *)
  norm_eps : float;
  rope : Kaun.Rope.t;
  tied : bool;  (** Whether the head is the token table (Llama 3.2 1B, 3B). *)
}
(** The type for Llama hyperparameters. *)

type 'a block = {
  attn_norm : 'a Kaun.Rms_norm.t;
  attn : 'a Kaun.Attention.t;
  ffn_norm : 'a Kaun.Rms_norm.t;
  gate : 'a Kaun.Linear.t;
  up : 'a Kaun.Linear.t;
  down : 'a Kaun.Linear.t;
}
(** The type for one block over payload ['a]. No projection has a bias. *)

type 'a params = {
  tok : 'a Kaun.Embedding.t;
  blocks : 'a block list;
  norm : 'a Kaun.Rms_norm.t;
  head : 'a Kaun.Linear.t option;  (** [None]: tied to [tok]. *)
}
(** The type for Llama parameters over payload ['a]. *)

type t = Nx.float32_t params

module Params : Nx.Ptree.Uniform with type 'a t = 'a params
(** The parameter traversals. Leaves are named [tok.table], [blocks.0.attn.q.w],
    [norm.gamma], [head.w], ... [Params.map (Nx.cast dt) p] converts precision;
    the layers keep their float32 islands whatever [dt]. *)

val make : config -> t
(** [make cfg] is a zero-initialized float32 model: the starting point of
    training from scratch, and the [~like] template that
    {!Kaun.Checkpoint.to_params} needs to read back a checkpoint this library
    saved. *)

(** {1:placement Placement}

    {!of_hf} and {!cache} place each leaf they build with
    [placement role ~axis]: [role] is the cut a tensor-parallel placement makes
    in the leaf and [axis] the axis of the leaf that cut runs along, [0] for
    [Whole]. One device ignores both: [fun _ ~axis:_ -> p]. *)

(** The type for the cuts of a tensor-parallel placement. *)
type role =
  | Whole  (** Kept whole: the token table and the norms. *)
  | Column
      (** Cut along its outputs: the query, key and value projections, the gate
          and up projections, and an untied head. *)
  | Row
      (** Cut along its inputs: the attention's output projection and the down
          projection. *)
  | Kv_heads  (** Cut along its heads: a cache pool. *)

(** {1:forward Forward passes}

    One fold over the blocks: [hidden cfg p ids] is
    [fst (cached cfg p caches index ids)] over {!Kaun.Cache_index.whole}, which
    reads and keeps nothing, so training never writes or reads a cache. *)

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
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  slots:int ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t Cache.t
(** [cache cfg ~slots dtype] is an empty decoding state of [slots] slots per
    block, at the parameters' dtype.

    With [placement], each pool is placed with [placement Kv_heads ~axis:1], so
    a compiled step finds the caches where the model is from its first call. *)

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
(** [config_of_json json] reads HuggingFace's [config.json], including the
    ["llama3"] rotary scaling.

    Raises [Failure] on a missing field or another rotary scaling type. *)

val of_hf :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  (float, 'b) Nx.dtype ->
  Kaun.Checkpoint.t ->
  (float, 'b) Nx.t params
(** [of_hf cfg dt ckpt] is the model of the HuggingFace Llama checkpoint [ckpt],
    at [dt]. Each entry is read by its name in the file with the shape [cfg]
    gives it, and every projection is transposed to [inputs × outputs], a view.
    At the file's own dtype nothing is copied; at another one each leaf is cast.

    With [placement], each leaf is placed with [Nx.place (placement role ~axis)]
    as it is built (see {!role}), before the next is read, so at most one leaf's
    cast is alive on the host and a function compiled where the model is that
    captures it uploads nothing.

    Raises [Invalid_argument], naming the entry, if one is missing, has another
    shape than [cfg] says, or is not a floating-point entry. *)

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
(** [dtype_of_string s] is the dtype named ["float32"], ["float16"] or
    ["bfloat16"]. Raises [Failure] on another name. *)

val stored_dtype : Kaun.Checkpoint.t -> dtype
(** [stored_dtype ckpt] is the dtype [ckpt] stores its embedding table at, the
    dtype at which {!of_hf} casts nothing. *)

val default_repo : string
(** An ungated HuggingFace repository of Llama 3.2 1B whose weight file is
    byte-identical to Meta's gated one. *)

val from_pretrained :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  ?repo_id:string ->
  (float, 'b) Nx.dtype ->
  config * (float, 'b) Nx.t params
(** [from_pretrained dt] downloads {!default_repo} (about 2.5 GB, cached
    afterwards), or [repo_id], and is its configuration and its parameters at
    [dt]. *)
