(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** BERT encoder and task heads.

    BERT inputs are passed as int32 [input_ids] with auxiliary data in
    {!Context}:

    {[
      let ctx =
        Context.empty
        |> Context.set ~name:Bert.token_type_ids_key (Ptree.P token_type_ids)
        |> Context.set ~name:Attention.attention_mask_key
             (Ptree.P attention_mask)
      in
      Layer.apply model vars ~training:false ~ctx input_ids
    ]}

    When absent, [token_type_ids] defaults to zeros and [attention_mask]
    defaults to ones (no padding). *)

open Kaun

(** {1:config Configuration} *)

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
(** The type for BERT configurations. *)

val config :
  vocab_size:int ->
  hidden_size:int ->
  num_hidden_layers:int ->
  num_attention_heads:int ->
  intermediate_size:int ->
  ?max_position_embeddings:int ->
  ?type_vocab_size:int ->
  ?hidden_dropout_prob:float ->
  ?attention_dropout_prob:float ->
  ?layer_norm_eps:float ->
  unit ->
  config
(** [config ~vocab_size ~hidden_size ~num_hidden_layers ~num_attention_heads
     ~intermediate_size ()] is a BERT configuration.

    [max_position_embeddings] defaults to [512]. [type_vocab_size] defaults to
    [2]. [hidden_dropout_prob] and [attention_dropout_prob] default to [0.1].
    [layer_norm_eps] defaults to [1e-12].

    Raises [Invalid_argument] if [hidden_size] is not divisible by
    [num_attention_heads] or if dropout rates are outside [\[0, 1)]. *)

(** {1:context Context keys} *)

val token_type_ids_key : string
(** ["token_type_ids"]. The {!Context} key for segment ids (shape
    [[batch; seq]], int32, values 0 or 1). *)

(** {1:layers Layers} *)

val encoder : config -> unit -> (int32, float) Layer.t
(** [encoder cfg ()] is the base BERT encoder.

    Input: int32 [input_ids] of shape [[batch; seq]]. Output: float hidden
    states of shape [[batch; seq; hidden_size]].

    Reads {!token_type_ids_key} and {!Attention.attention_mask_key} from [ctx].
*)

val pooler : config -> unit -> (float, float) Layer.t
(** [pooler cfg ()] maps [[batch; seq; hidden_size]] to [[batch; hidden_size]]
    by extracting the CLS token (position 0) and applying a dense + tanh. *)

val for_sequence_classification :
  config -> num_labels:int -> unit -> (int32, float) Layer.t
(** [for_sequence_classification cfg ~num_labels ()] is encoder + pooler +
    classifier. Output: logits [[batch; num_labels]]. *)

val for_masked_lm : config -> unit -> (int32, float) Layer.t
(** [for_masked_lm cfg ()] is encoder + MLM head with tied word embeddings.
    Output: logits [[batch; seq; vocab_size]]. *)

(** {1:pretrained Pretrained loading} *)

val from_pretrained :
  ?model_id:string -> unit -> config * Ptree.t * Ptree.t option * Ptree.t option
(** [from_pretrained ?model_id ()] downloads [model_id] from HuggingFace and
    returns [(cfg, encoder_params, pooler_params, mlm_params)].

    [encoder_params] is ready for {!encoder}. [pooler_params] and [mlm_params]
    are [Some _] when the checkpoint contains the corresponding weights.

    [model_id] defaults to ["bert-base-uncased"]. *)
