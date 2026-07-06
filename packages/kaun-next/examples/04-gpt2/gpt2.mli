(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 (Radford et al., 2019) from kaun-next layers.

    The model is a plain record of {!Kaun_next} layers; {!logits} is its forward
    pass and {!Params} its checkpoint plumbing. {!of_hf} adapts the HuggingFace
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
  ln1 : Kaun_next.Layer_norm.t;
  attn : Kaun_next.Attention.t;
  ln2 : Kaun_next.Layer_norm.t;
  fc : Kaun_next.Linear.t;  (** MLP up projection, [n_embd → n_inner]. *)
  proj : Kaun_next.Linear.t;  (** MLP down projection, [n_inner → n_embd]. *)
}
(** The type for one pre-norm transformer block. *)

type t = {
  wte : Kaun_next.Embedding.t;  (** Token embeddings, also the tied LM head. *)
  wpe : Kaun_next.Embedding.t;  (** Learned position embeddings. *)
  blocks : block list;
  ln_f : Kaun_next.Layer_norm.t;
}
(** The type for GPT-2 parameters. *)

module Params : Kaun_next.Checkpoint.Named with type t = t
(** Checkpoint plumbing for {!type:t}. Leaves are named [wte.table],
    [blocks.0.attn.q.w], [ln_f.gamma], ... *)

val make : config -> t
(** [make cfg] is a zero-initialized model, the [~like] template for
    {!Kaun_next.Checkpoint.to_params}. *)

val logits : config -> t -> (int32, Nx.int32_elt) Nx.t -> Nx.float32_t
(** [logits cfg p ids] is the next-token logits for the [[| batch; seq |]] id
    tensor [ids], of shape [[| batch; seq; vocab_size |]]. The LM head is tied
    to [p.wte].

    Raises [Invalid_argument] if [ids] has more than [cfg.n_positions]
    positions. *)

val of_hf : n_layer:int -> Kaun_next.Checkpoint.t -> Kaun_next.Checkpoint.t
(** [of_hf ~n_layer ckpt] adapts the HuggingFace GPT-2 checkpoint [ckpt] to
    {!Params}' names: splits each block's fused [c_attn] weight and bias into
    the [q], [k] and [v] projections and renames everything else. HF's [Conv1D]
    weights are already [inputs × outputs], so no transposes are needed. Entries
    the model does not use (attention mask buffers) are left in place and
    ignored by extraction. *)

val from_pretrained : ?repo_id:string -> unit -> config * t
(** [from_pretrained ()] downloads [repo_id] (defaults to ["gpt2"]) from the
    HuggingFace Hub — config and weights — and is the parsed configuration with
    the pretrained parameters.

    Raises [Failure] on download or parse errors. *)
