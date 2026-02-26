(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 decoder and language model head.

    GPT-2 inputs are passed as int32 [input_ids]:

    {[
      Layer.apply model vars ~training:false input_ids
    ]}

    Position ids are computed automatically from the sequence length. *)

open Kaun

(** {1:config Configuration} *)

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
(** The type for GPT-2 configurations. *)

val config :
  vocab_size:int ->
  n_embd:int ->
  n_layer:int ->
  n_head:int ->
  ?n_positions:int ->
  ?n_inner:int ->
  ?resid_pdrop:float ->
  ?embd_pdrop:float ->
  ?attn_pdrop:float ->
  ?layer_norm_eps:float ->
  unit ->
  config
(** [config ~vocab_size ~n_embd ~n_layer ~n_head ()] is a GPT-2 configuration.

    [n_positions] defaults to [1024]. [n_inner] defaults to [4 * n_embd].
    Dropout rates default to [0.1]. [layer_norm_eps] defaults to [1e-5].

    Raises [Invalid_argument] if [n_embd] is not divisible by [n_head]. *)

(** {1:layers Layers} *)

val decoder : config -> unit -> (int32, float) Layer.t
(** [decoder cfg ()] is the GPT-2 transformer decoder.

    Input: int32 [input_ids] of shape [[batch; seq]]. Output: float hidden
    states of shape [[batch; seq; n_embd]]. *)

val for_causal_lm : config -> unit -> (int32, float) Layer.t
(** [for_causal_lm cfg ()] is decoder + tied LM head.

    Output: logits [[batch; seq; vocab_size]]. Word embeddings are tied with the
    LM head projection. *)

(** {1:pretrained Pretrained loading} *)

val from_pretrained : ?model_id:string -> unit -> config * Ptree.t
(** [from_pretrained ?model_id ()] downloads [model_id] from HuggingFace and
    returns [(cfg, decoder_params)].

    [decoder_params] is ready for {!decoder} or {!for_causal_lm} (the LM head
    reuses the word embedding weights).

    [model_id] defaults to ["gpt2"]. *)
