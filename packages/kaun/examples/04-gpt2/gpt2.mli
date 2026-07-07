(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 (Radford et al., 2019) from kaun layers.

    The model is a plain record of {!Kaun} layers; {!logits} is its forward pass
    and {!Params} its checkpoint plumbing. {!of_hf} adapts the HuggingFace
    checkpoint — [h.{i}.attn.c_attn] fused qkv projections, [Conv1D] naming —
    onto {!Params}' names, and {!from_pretrained} runs the whole pipeline:
    download, adapt, extract typed parameters. *)

type config = {
  vocab_size : int;
  n_positions : int;  (** Maximum sequence length. *)
  n_embd : int;
  n_layer : int;
  n_head : int;
  n_inner : int;  (** MLP hidden width, [4 * n_embd] in the released models. *)
  layer_norm_eps : float;
}
(** The type for GPT-2 hyperparameters, as in HuggingFace's [config.json]. *)

type block = {
  ln1 : Kaun.Layer_norm.t;
  attn : Kaun.Attention.t;
  ln2 : Kaun.Layer_norm.t;
  fc : Kaun.Linear.t;  (** MLP up projection, [n_embd → n_inner]. *)
  proj : Kaun.Linear.t;  (** MLP down projection, [n_inner → n_embd]. *)
}
(** The type for one pre-norm transformer block. *)

type t = {
  wte : Kaun.Embedding.t;  (** Token embeddings, also the tied LM head. *)
  wpe : Kaun.Embedding.t;  (** Learned position embeddings. *)
  blocks : block list;
  ln_f : Kaun.Layer_norm.t;
}
(** The type for GPT-2 parameters. *)

module Params : Kaun.Checkpoint.Named with type t = t
(** Checkpoint plumbing for {!type:t}. Leaves are named [wte.table],
    [blocks.0.attn.q.w], [ln_f.gamma], ... *)

val make : config -> t
(** [make cfg] is a zero-initialized model, the [~like] template for
    {!Kaun.Checkpoint.to_params}. *)

val logits : config -> t -> (int32, Nx.int32_elt) Nx.t -> Nx.float32_t
(** [logits cfg p ids] is the next-token logits for the [[| batch; seq |]] id
    tensor [ids], of shape [[| batch; seq; vocab_size |]]. The LM head is tied
    to [p.wte].

    Raises [Invalid_argument] if [ids] has more than [cfg.n_positions]
    positions. *)

type cache = Nx.float32_elt Kaun.Attention.Cache.t list
(** The type for decoding state: one key-value cache per block, in block order.
*)

val cache : config -> len:int -> cache
(** [cache cfg ~len] is an empty decoding state whose caches hold [len]
    positions: the total sequence length (prompt plus generated tokens) must not
    exceed [len]. *)

val logits_cached :
  config ->
  t ->
  pos:(int32, Nx.int32_elt) Nx.t ->
  cache ->
  (int32, Nx.int32_elt) Nx.t ->
  Nx.float32_t * cache
(** [logits_cached cfg p ~pos caches ids] is the next-token logits at the last
    position of [ids] — shape [[| batch; vocab_size |]] — and the updated
    caches, where [ids] holds the positions [pos] to [pos + seq - 1] of the
    sequence and [pos] is a one-element int32 tensor. A whole-prompt call at
    [pos = 0] prefills the caches; a single-token call advances decoding by one
    step. Since [pos] is a tensor, both trace under {!Rune.jit} — one compiled
    single-token step serves the whole decode loop.

    The caller steps [pos] and must keep [pos + seq] at most the cache length
    (see {!Kaun.Attention.apply_cached}) and [cfg.n_positions].

    Raises [Invalid_argument] on the geometry errors of
    {!Kaun.Attention.apply_cached}. *)

val generate : config -> t -> max_tokens:int -> int32 array -> int32 array
(** [generate cfg p ~max_tokens prompt] is [prompt] extended with [max_tokens]
    greedily decoded token ids: each step re-runs {!logits} on the whole
    sequence (the model keeps no key-value cache) and appends the argmax of the
    last position.

    Raises [Invalid_argument] if [prompt] is empty or the sequence outgrows
    [cfg.n_positions]. *)

val of_hf : n_layer:int -> Kaun.Checkpoint.t -> Kaun.Checkpoint.t
(** [of_hf ~n_layer ckpt] adapts the HuggingFace GPT-2 checkpoint [ckpt] to
    {!Params}' names: splits each block's fused [c_attn] weight and bias into
    the [q], [k] and [v] projections and renames everything else. HF's [Conv1D]
    weights are already [inputs × outputs], so no transposes are needed. Entries
    the model does not use (attention mask buffers) are left in place and
    ignored by extraction. *)

val of_checkpoint : config -> Kaun.Checkpoint.t -> t
(** [of_checkpoint cfg ckpt] extracts pretrained parameters from the
    HuggingFace-layout checkpoint [ckpt]: {!of_hf} adaptation followed by typed
    extraction against [make cfg].

    Raises [Invalid_argument] if [ckpt] does not match [cfg] (see
    {!Kaun.Checkpoint.to_params}). *)

val from_file : config -> string -> t
(** [from_file cfg path] is [of_checkpoint cfg (Checkpoint.load path)]: the
    parameters of a local HuggingFace-layout safetensors file, for checkpoints
    already on disk.

    Raises [Failure] on I/O or format errors, [Invalid_argument] as
    {!of_checkpoint}. *)

val from_pretrained : ?repo_id:string -> unit -> config * t
(** [from_pretrained ()] downloads [repo_id] (defaults to ["gpt2"]) from the
    HuggingFace Hub — config and weights — and is the parsed configuration with
    the pretrained parameters.

    Raises [Failure] on download or parse errors. *)
