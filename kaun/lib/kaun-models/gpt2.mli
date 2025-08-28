(** GPT-2: Generative Pre-trained Transformer 2.

    Radford et al., 2019: "Language Models are Unsupervised Multitask Learners"

    A transformer-based autoregressive language model that uses causal
    self-attention for text generation and language understanding tasks. *)

open Rune

(** {1 Configuration} *)

type config = {
  vocab_size : int;  (** Size of vocabulary *)
  n_positions : int;  (** Maximum sequence length *)
  n_embd : int;  (** Hidden dimension (d_model) *)
  n_layer : int;  (** Number of transformer decoder layers *)
  n_head : int;  (** Number of attention heads *)
  n_inner : int option;
      (** FFN intermediate dimension (defaults to 4 * n_embd) *)
  activation_function : [ `gelu | `relu | `swish | `gelu_new ];
      (** Activation function *)
  resid_pdrop : float;  (** Dropout probability for residual connections *)
  embd_pdrop : float;  (** Dropout probability for embeddings *)
  attn_pdrop : float;  (** Dropout for attention probabilities *)
  layer_norm_epsilon : float;  (** Layer normalization epsilon *)
  initializer_range : float;
      (** Standard deviation for weight initialization *)
  scale_attn_weights : bool;  (** Whether to scale attention weights *)
  use_cache : bool;  (** Whether to cache key/values *)
  scale_attn_by_inverse_layer_idx : bool;
      (** Scale attention by 1/sqrt(layer_idx) *)
  reorder_and_upcast_attn : bool;  (** Reorder and upcast attention *)
  bos_token_id : int option;  (** Beginning of sequence token ID *)
  eos_token_id : int option;  (** End of sequence token ID *)
  pad_token_id : int option;  (** Padding token ID *)
}
(** GPT-2 model configuration *)

val default_config : config
(** Standard GPT-2 configurations *)

val gpt2_small : config
(** GPT-2 Small: 12 layers, 768 hidden, 12 heads, 124M parameters *)

val gpt2_medium : config
(** GPT-2 Medium: 24 layers, 1024 hidden, 16 heads, 355M parameters *)

val gpt2_large : config
(** GPT-2 Large: 36 layers, 1280 hidden, 20 heads, 774M parameters *)

val gpt2_xl : config
(** GPT-2 XL: 48 layers, 1600 hidden, 25 heads, 1.5B parameters *)

(** {1 Model Components} *)

val embeddings : config:config -> unit -> Kaun.module_
(** GPT-2 embeddings combining token and position embeddings *)

type ('a, 'dev) output = {
  last_hidden_state : (float, 'a, 'dev) Rune.t;
      (** Sequence of hidden states at the last layer
          [batch_size; seq_len; hidden_size] *)
  hidden_states : (float, 'a, 'dev) Rune.t list option;
      (** Hidden states from all layers if output_hidden_states=true *)
  attentions : (float, 'a, 'dev) Rune.t list option;
      (** Attention weights from all layers if output_attentions=true *)
}
(** Model outputs *)

type ('a, 'dev) gpt2 = {
  model : Kaun.module_;
  params : ('a, 'dev) Kaun.params;
  config : config;
  device : 'dev device;
  dtype : (float, 'a) dtype;
}
(** Unified GPT-2 model type *)

type 'dev inputs = {
  input_ids : (int32, int32_elt, 'dev) Rune.t;
  attention_mask : (int32, int32_elt, 'dev) Rune.t option;
  position_ids : (int32, int32_elt, 'dev) Rune.t option;
}
(** Input tensors for GPT-2 *)

(** Create a new GPT-2 model *)
val create : ?config:config -> unit -> Kaun.module_
(** [create ?config ()] creates a new GPT-2 model.

    @param config Model configuration (default: gpt2_small) *)

(** Load pretrained GPT-2 from HuggingFace *)
val from_pretrained :
  ?model_id:string ->
  ?revision:Kaun_huggingface.revision ->
  ?cache_config:Kaun_huggingface.Config.t ->
  device:'dev device ->
  dtype:(float, 'a) dtype ->
  unit ->
  ('a, 'dev) gpt2
(** [from_pretrained ?model_id ?device ?dtype ()] loads pretrained GPT-2.

    Default model_id is "gpt2", device is CPU, dtype is Float32. Returns a
    unified gpt2 record with model, params, and config.

    Example:
    {[
      let gpt2 = GPT2.from_pretrained () in
      (* Or with options: *)
      let gpt2 = GPT2.from_pretrained ~model_id:"gpt2-medium" ()
    ]} *)

(** Forward pass through GPT-2 *)
val forward :
  ('a, 'dev) gpt2 ->
  'dev inputs ->
  ?training:bool ->
  ?output_hidden_states:bool ->
  ?output_attentions:bool ->
  unit ->
  ('a, 'dev) output
(** [forward ~model ~params ~input_ids ... ()] performs a forward pass.

    @param input_ids Token IDs [batch_size; seq_len]
    @param attention_mask
      Mask for padding tokens (1 for real tokens, 0 for padding)
    @param position_ids Custom position IDs (default: 0..seq_len-1)
    @param training Whether in training mode (affects dropout)
    @param output_hidden_states Whether to return all hidden states
    @param output_attentions Whether to return attention weights *)

(** {1 Task-Specific Heads} *)

(** GPT-2 for causal language modeling *)
module For_causal_lm : sig
  val create : ?config:config -> unit -> Kaun.module_

  val forward :
    model:Kaun.module_ ->
    params:('a, 'dev) Kaun.params ->
    input_ids:(int32, int32_elt, 'dev) Rune.t ->
    ?attention_mask:(int32, int32_elt, 'dev) Rune.t ->
    ?position_ids:(int32, int32_elt, 'dev) Rune.t ->
    ?labels:(int32, int32_elt, 'dev) Rune.t ->
    training:bool ->
    unit ->
    (float, 'a, 'dev) Rune.t * (float, 'a, 'dev) Rune.t option
  (** Returns (logits, loss) where logits has shape
      [batch_size; seq_len; vocab_size] *)
end

(** {1 Tokenization} *)

module Tokenizer : sig
  type t
  (** GPT-2 tokenizer instance with BPE *)

  val create :
    ?vocab_file:string -> ?merges_file:string -> ?model_id:string -> unit -> t
  (** Create a BPE tokenizer for GPT-2. Either provide vocab_file and
      merges_file paths, or a model_id to download from HuggingFace (defaults to
      gpt2) *)

  val encode_to_array : t -> string -> int array
  (** Encode text to token IDs *)

  val encode : t -> string -> device:'dev device -> 'dev inputs
  (** Encode text directly to input tensors ready for forward pass *)

  val encode_batch :
    t ->
    ?max_length:int ->
    ?padding:bool ->
    device:'dev device ->
    string list ->
    (int32, int32_elt, 'dev) Rune.t
  (** Encode multiple texts with optional padding *)

  val decode : t -> int array -> string
  (** Decode token IDs back to text *)

  val get_bos_token_id : t -> int
  (** Get beginning of sequence token ID *)

  val get_eos_token_id : t -> int
  (** Get end of sequence token ID *)

  val get_pad_token_id : t -> int option
  (** Get padding token ID *)

  val get_vocab_size : t -> int
  (** Get vocabulary size *)
end

(** {1 Utilities} *)

val num_parameters : ('a, 'dev) Kaun.params -> int
(** Count total parameters in the model *)

val parameter_stats : ('a, 'dev) Kaun.params -> string
(** Get human-readable parameter statistics *)

(** {1 GPT-2 Configuration Parsing} *)

val parse_gpt2_config : Yojson.Safe.t -> config
(** Parse GPT-2 configuration from HuggingFace JSON format *)

(** {1 Common Model Configurations} *)

val load_gpt2_small :
  device:'dev device -> dtype:(float, 'a) dtype -> unit -> ('a, 'dev) gpt2
(** Load GPT-2 Small (124M parameters) *)

val load_gpt2_medium :
  device:'dev device -> dtype:(float, 'a) dtype -> unit -> ('a, 'dev) gpt2
(** Load GPT-2 Medium (355M parameters) *)

val load_gpt2_large :
  device:'dev device -> dtype:(float, 'a) dtype -> unit -> ('a, 'dev) gpt2
(** Load GPT-2 Large (774M parameters) *)

val load_gpt2_xl :
  device:'dev device -> dtype:(float, 'a) dtype -> unit -> ('a, 'dev) gpt2
(** Load GPT-2 XL (1.5B parameters) *)
