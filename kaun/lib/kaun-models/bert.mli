(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** BERT: Bidirectional Encoder Representations from Transformers.

    Devlin et al., 2018: "BERT: Pre-training of Deep Bidirectional Transformers
    for Language Understanding"

    A transformer-based model that uses bidirectional self-attention to
    understand context from both directions, revolutionizing NLP tasks. *)

open Rune

(** {1 Configuration} *)

type config = {
  vocab_size : int;  (** Size of vocabulary *)
  hidden_size : int;  (** Hidden dimension (d_model) *)
  num_hidden_layers : int;  (** Number of transformer encoder layers *)
  num_attention_heads : int;  (** Number of attention heads *)
  intermediate_size : int;
      (** FFN intermediate dimension (typically 4 * hidden_size) *)
  hidden_act : [ `gelu | `relu | `swish | `gelu_new ];
      (** Activation function *)
  hidden_dropout_prob : float;  (** Dropout probability for hidden layers *)
  attention_probs_dropout_prob : float;
      (** Dropout for attention probabilities *)
  max_position_embeddings : int;  (** Maximum sequence length *)
  type_vocab_size : int;
      (** Token type vocabulary size (for segment embeddings) *)
  layer_norm_eps : float;  (** Layer normalization epsilon *)
  pad_token_id : int;  (** Padding token ID *)
  position_embedding_type : [ `absolute | `relative ];
      (** Type of position embeddings *)
  use_cache : bool;  (** Whether to cache key/values *)
  classifier_dropout : float option;  (** Dropout for classification head *)
}
(** BERT model configuration *)

val default_config : config
(** Standard BERT configurations *)

val bert_base_uncased : config
(** BERT Base: 12 layers, 768 hidden, 12 heads, 110M parameters *)

val bert_large_uncased : config
(** BERT Large: 24 layers, 1024 hidden, 16 heads, 340M parameters *)

val bert_base_cased : config
(** Same as base_uncased but preserves case information *)

val bert_base_multilingual : config
(** BERT Base for 104 languages *)

(** {1 Model Components} *)

val embeddings : config:config -> unit -> Kaun.module_
(** BERT embeddings combining token, position, and segment embeddings *)

val pooler : hidden_size:int -> unit -> Kaun.module_

type 'a output = {
  last_hidden_state : (float, 'a) Rune.t;
      (** Sequence of hidden states at the last layer
          [batch_size; seq_len; hidden_size] *)
  pooler_output : (float, 'a) Rune.t option;
      (** Pooled [CLS] token representation [batch_size; hidden_size] *)
  hidden_states : (float, 'a) Rune.t list option;
      (** Hidden states from all layers if output_hidden_states=true *)
  attentions : (float, 'a) Rune.t list option;
      (** Attention weights from all layers if output_attentions=true *)
}
(** Model outputs *)

type 'a bert = {
  model : Kaun.module_;
  params : Kaun.params;
  config : config;
  dtype : (float, 'a) dtype;
}
(** Unified BERT model type *)

type inputs = {
  input_ids : (int32, int32_elt) Rune.t;
  attention_mask : (int32, int32_elt) Rune.t;
  token_type_ids : (int32, int32_elt) Rune.t option;
  position_ids : (int32, int32_elt) Rune.t option;
}
(** Input tensors for BERT *)

(** Create a new BERT model *)
val create : ?config:config -> ?add_pooling_layer:bool -> unit -> Kaun.module_
(** [create ?config ?add_pooling_layer ()] creates a new BERT model.

    @param config Model configuration (default: bert_base_uncased)
    @param add_pooling_layer
      Whether to add pooling layer for [CLS] token (default: true) *)

(** Load pretrained BERT from HuggingFace *)
val from_pretrained :
  ?model_id:string ->
  ?revision:Kaun_huggingface.revision ->
  ?cache_config:Kaun_huggingface.Config.t ->
  dtype:(float, 'a) dtype ->
  unit ->
  'a bert
(** [from_pretrained ?model_id ?dtype ()] loads pretrained BERT.

    Default model_id is "bert-base-uncased", device is CPU, dtype is Float32.
    Returns a unified bert record with model, params, and config.

    Example:
    {[
      let bert = BERT.from_pretrained () in
      (* Or with options: *)
      let bert = BERT.from_pretrained ~model_id:"bert-base-multilingual-cased" ()
    ]} *)

(** Forward pass through BERT *)
val forward :
  'a bert ->
  inputs ->
  ?training:bool ->
  ?output_hidden_states:bool ->
  ?output_attentions:bool ->
  ?rngs:Rune.Rng.key ->
  unit ->
  'a output
(** [forward ~model ~params ~input_ids ... ()] performs a forward pass.

    @param input_ids Token IDs [batch_size; seq_len]
    @param attention_mask
      Mask for padding tokens (1 for real tokens, 0 for padding)
    @param token_type_ids Segment IDs for sentence pairs (0 or 1)
    @param position_ids Custom position IDs (default: 0..seq_len-1)
    @param head_mask Mask to nullify specific attention heads
    @param training Whether in training mode (affects dropout)
    @param output_hidden_states Whether to return all hidden states
    @param output_attentions Whether to return attention weights
    @param rngs Source key required when training with dropout *)

(** {1 Task-Specific Heads} *)

(** BERT for masked language modeling *)
module For_masked_lm : sig
  val create : ?config:config -> unit -> Kaun.module_

  val forward :
    model:Kaun.module_ ->
    params:Kaun.params ->
    compute_dtype:(float, 'a) dtype ->
    input_ids:(int32, int32_elt) Rune.t ->
    ?config:config ->
    ?attention_mask:(int32, int32_elt) Rune.t ->
    ?token_type_ids:(int32, int32_elt) Rune.t ->
    ?labels:(int32, int32_elt) Rune.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    unit ->
    (float, 'a) Rune.t * (float, 'a) Rune.t option
  (** Returns (logits, loss) where logits has shape
      [batch_size; seq_len; vocab_size] *)
end

(** BERT for sequence classification *)
module For_sequence_classification : sig
  val create : ?config:config -> num_labels:int -> unit -> Kaun.module_

  val forward :
    model:Kaun.module_ ->
    params:Kaun.params ->
    compute_dtype:(float, 'a) dtype ->
    input_ids:(int32, int32_elt) Rune.t ->
    ?config:config ->
    ?attention_mask:(int32, int32_elt) Rune.t ->
    ?token_type_ids:(int32, int32_elt) Rune.t ->
    ?labels:(int32, int32_elt) Rune.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    unit ->
    (float, 'a) Rune.t * (float, 'a) Rune.t option
  (** Returns (logits, loss) where logits has shape [batch_size; num_labels] *)
end

(** BERT for token classification (NER, POS tagging) *)
module For_token_classification : sig
  val create : ?config:config -> num_labels:int -> unit -> Kaun.module_

  val forward :
    model:Kaun.module_ ->
    params:Kaun.params ->
    compute_dtype:(float, 'a) dtype ->
    input_ids:(int32, int32_elt) Rune.t ->
    ?config:config ->
    ?attention_mask:(int32, int32_elt) Rune.t ->
    ?token_type_ids:(int32, int32_elt) Rune.t ->
    ?labels:(int32, int32_elt) Rune.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    unit ->
    (float, 'a) Rune.t * (float, 'a) Rune.t option
  (** Returns (logits, loss) where logits has shape
      [batch_size; seq_len; num_labels] *)
end

(** {1 Tokenization} *)

module Tokenizer : sig
  type t
  (** BERT tokenizer instance *)

  val create : ?vocab_file:string -> ?model_id:string -> unit -> t
  (** Create a WordPiece tokenizer for BERT. Either provide a vocab_file path or
      a model_id to download from HuggingFace (defaults to bert-base-uncased) *)

  val create_wordpiece : ?vocab_file:string -> ?model_id:string -> unit -> t
  (** Alias for create *)

  val encode_to_array : t -> string -> int array
  (** Encode text to token IDs with [CLS] and [SEP] tokens *)

  val encode : t -> string -> inputs
  (** Encode text directly to input tensors ready for forward pass *)

  val encode_batch :
    t ->
    ?max_length:int ->
    ?padding:bool ->
    string list ->
    (int32, int32_elt) Rune.t
  (** Encode multiple texts with padding and special tokens *)

  val decode : t -> int array -> string
  (** Decode token IDs back to text *)
end

(** {1 Utilities} *)

(** Create attention mask from input IDs *)
val create_attention_mask :
  input_ids:(int32, int32_elt) Rune.t ->
  pad_token_id:int ->
  dtype:(float, 'a) dtype ->
  (float, 'a) Rune.t
(** Creates attention mask where 1.0 for real tokens and 0.0 for padding *)

(** Get BERT embeddings for text analysis *)
val get_embeddings :
  model:Kaun.module_ ->
  params:Kaun.params ->
  input_ids:(int32, int32_elt) Rune.t ->
  ?attention_mask:(int32, int32_elt) Rune.t ->
  layer_index:int ->
  unit ->
  (float, 'a) Rune.t
(** Extract embeddings from a specific layer (0 = embeddings, 1..n = encoder
    layers) *)

val num_parameters : Kaun.params -> int
(** Count total parameters in the model *)

val parameter_stats : Kaun.params -> string
(** Get human-readable parameter statistics *)

(** {1 BERT Configuration Parsing} *)

val parse_bert_config : Yojson.Safe.t -> config
(** Parse BERT configuration from HuggingFace JSON format *)

(** {1 Common Model Configurations} *)

val load_bert_base_uncased : dtype:(float, 'a) dtype -> unit -> 'a bert
(** Load BERT Base Uncased (110M parameters) *)

val load_bert_large_uncased : dtype:(float, 'a) dtype -> unit -> 'a bert
(** Load BERT Large Uncased (340M parameters) *)

val load_bert_base_cased : dtype:(float, 'a) dtype -> unit -> 'a bert
(** Load BERT Base Cased (110M parameters) *)

val load_bert_base_multilingual_cased :
  dtype:(float, 'a) dtype -> unit -> 'a bert
(** Load Multilingual BERT Base Cased (110M parameters, 104 languages) *)
