(** Tokenizers library - text tokenization for ML. This module provides fast and
    flexible tokenization for machine learning applications, supporting multiple
    algorithms from simple word splitting to advanced subword tokenization like
    BPE, Unigram, WordLevel, and WordPiece. The API is designed to match Hugging
    Face Tokenizers v0.21 as closely as possible, adapted to idiomatic OCaml
    with functional style, records for configurations, polymorphic variants for
    enums, default values for optionals, and result types for fallible
    operations. The central type is {!Tokenizer.t}, which represents a
    configurable tokenization pipeline.

    {1 Quick Start}
    {[
      open Saga.Tokenizers
      (* Create a character-level tokenizer *)
      let tokenizer = Tokenizer.create ~model:(Model.chars ()) in
      (* Add special tokens *)
      Tokenizer.add_special_tokens tokenizer [Added_token.create ~content:"." ~special:true ()];
      (* Train on data *)
      Tokenizer.train_from_iterator tokenizer (Seq.of_list names) ~trainer:(Trainer.chars ()) ();
      (* Encode with options *)
      let encoding = Tokenizer.encode tokenizer ~sequence:"hello world" ~add_special_tokens:true ();
      (* Get ids and decode *)
      let ids = Encoding.ids encoding in
      let text = Tokenizer.decode tokenizer ids ~skip_special_tokens:true;
    ]}

    {1 Key Concepts}
    - {!Tokenizer.t}: The main tokenizer instance, configurable with model,
      normalizer, etc.
    - {!Models.t}: Core tokenization algorithm (e.g., Chars, BPE).
    - {!Encoding.t}: Result of encoding, with ids, tokens, offsets, masks, etc.,
      exposed as a record.
    - Special tokens: Handled via {!add_special_tokens} and encoding options.
    - All functions handle Unicode correctly via the {!Unicode} module. This API
      aligns with Hugging Face Tokenizers v0.21 (as of 2025), including support
      for fast Rust-backed operations where applicable. *)

(** Either type for API compatibility. *)
module Either : sig
  type ('a, 'b) t = Left of 'a | Right of 'b
end

module Unicode = Unicode
(** Unicode utilities. *)

module Models = Models
module Normalizers = Normalizers
module Pre_tokenizers = Pre_tokenizers
module Processors = Processors
module Decoders = Decoders
module Trainers = Trainers
module Encoding = Encoding
module Bpe = Bpe
module Wordpiece = Wordpiece

(** {1 Enums as Polymorphic Variants} *)

type direction = [ `Left | `Right ]
(** Padding or truncation direction. *)

type split_delimiter_behavior =
  [ `Removed
  | `Isolated
  | `Merged_with_previous
  | `Merged_with_next
  | `Contiguous ]
(** Behavior for splitting delimiters. *)

type strategy = [ `Longest_first | `Only_first | `Only_second ]
(** Truncation strategy. *)

type prepend_scheme = [ `Always | `Never | `First ]
(** Prepend scheme for metaspace. *)

(** {1 Core Types} *)

module Added_token : sig
  type t
  (** Added token with customization, matching HF Added_token. *)

  val create :
    ?content:string ->
    ?single_word:bool ->
    ?lstrip:bool ->
    ?rstrip:bool ->
    ?normalized:bool ->
    ?special:bool ->
    unit ->
    t
  (** Create added token with options and defaults. *)

  val content : t -> string
  val lstrip : t -> bool
  val normalized : t -> bool
  val rstrip : t -> bool
  val single_word : t -> bool
end

module Tokenizer : sig
  type t
  (** Main tokenizer type. *)

  type padding_config = {
    direction : direction;
    pad_id : int;
    pad_type_id : int;
    pad_token : string;
    length : int option;
    pad_to_multiple_of : int option;
  }
  (** Record for padding config. *)

  type truncation_config = {
    max_length : int;
    stride : int;
    strategy : strategy;
    direction : direction;
  }
  (** Record for truncation config. *)

  (** {2 Creation} *)
  val create : model:Models.t -> t
  (** Create with core model. *)

  val from_file : string -> (t, exn) result
  (** From JSON file with result. *)

  val from_str : string -> (t, exn) result
  (** From JSON string with result. *)

  val from_pretrained :
    string -> ?revision:string -> ?token:string -> unit -> (t, exn) result
  (** From pretrained with result and defaults. *)

  val from_buffer : bytes -> (t, exn) result
  (** From buffer with result. *)

  (** {2 Configuration} *)
  val set_normalizer : t -> Normalizers.t option -> unit
  (** Set normalizer. *)

  val get_normalizer : t -> Normalizers.t option
  (** Get normalizer. *)

  val set_pre_tokenizer : t -> Pre_tokenizers.t option -> unit
  (** Set pre-tokenizer. *)

  val get_pre_tokenizer : t -> Pre_tokenizers.t option
  (** Get pre-tokenizer. *)

  val set_post_processor : t -> Processors.t option -> unit
  (** Set post-processor. *)

  val get_post_processor : t -> Processors.t option
  (** Get post-processor. *)

  val set_decoder : t -> Decoders.t option -> unit
  (** Set decoder. *)

  val get_decoder : t -> Decoders.t option
  (** Get decoder. *)

  val set_model : t -> Models.t -> unit
  (** Set model. *)

  val get_model : t -> Models.t
  (** Get model. *)

  (** {2 Padding and Truncation} *)
  val enable_padding : t -> padding_config -> unit
  (** Enable padding with record config. *)

  val no_padding : t -> unit
  (** Disable padding. *)

  val get_padding : t -> padding_config option
  (** Get padding config. *)

  val enable_truncation : t -> truncation_config -> unit
  (** Enable truncation with record config. *)

  val no_truncation : t -> unit
  (** Disable truncation. *)

  val get_truncation : t -> truncation_config option
  (** Get truncation config. *)

  (** {2 Vocabulary Management} *)
  val add_tokens : t -> (string, Added_token.t) Either.t list -> int
  (** Add tokens, return count added. *)

  val add_special_tokens : t -> (string, Added_token.t) Either.t list -> int
  (** Add special tokens. *)

  val get_vocab : t -> ?with_added_tokens:bool -> unit -> (string * int) list
  (** Get vocab list with default. *)

  val get_vocab_size : t -> ?with_added_tokens:bool -> unit -> int
  (** Get size with default. *)

  val get_added_tokens_decoder : t -> (int * Added_token.t) list
  (** Get added tokens. *)

  val token_to_id : t -> string -> int option
  (** Token to id. *)

  val id_to_token : t -> int -> string option
  (** Id to token. *)

  (** {2 Training} *)
  val train : t -> files:string list -> ?trainer:Trainers.t -> unit -> unit
  (** Train from files. *)

  val train_from_iterator :
    t -> string Seq.t -> ?trainer:Trainers.t -> ?length:int -> unit -> unit
  (** Train from text sequence. *)

  (** {2 Encoding and Decoding} *)
  val encode :
    t ->
    sequence:(string, string list) Either.t ->
    ?pair:(string, string list) Either.t ->
    ?is_pretokenized:bool ->
    ?add_special_tokens:bool ->
    unit ->
    Encoding.t
  (** Encode single or pair, allowing pretokenized lists. *)

  val encode_batch :
    t ->
    input:
      ( (string, string list) Either.t,
        (string, string list) Either.t * (string, string list) Either.t )
      Either.t
      list ->
    ?is_pretokenized:bool ->
    ?add_special_tokens:bool ->
    unit ->
    Encoding.t list
  (** Batch encode with flexible inputs. *)

  val decode :
    t ->
    int list ->
    ?skip_special_tokens:bool ->
    ?clean_up_tokenization_spaces:bool ->
    unit ->
    string
  (** Decode with defaults. *)

  val decode_batch :
    t ->
    int list list ->
    ?skip_special_tokens:bool ->
    ?clean_up_tokenization_spaces:bool ->
    unit ->
    string list
  (** Batch decode with defaults. *)

  val post_process :
    t ->
    encoding:Encoding.t ->
    ?pair:Encoding.t ->
    ?add_special_tokens:bool ->
    unit ->
    Encoding.t
  (** Post-process manually. *)

  val num_special_tokens_to_add : t -> is_pair:bool -> int
  (** Number of specials. *)

  (** {2 Serialization} *)
  val save : t -> path:string -> ?pretty:bool -> unit -> unit
  (** Save to file with pretty default. *)

  val save_pretrained : t -> path:string -> unit
  (** Save pretrained format. *)

  val to_str : t -> ?pretty:bool -> unit -> string
  (** To JSON string with default. *)
end
