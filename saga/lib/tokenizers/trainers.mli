(** Training module for tokenization models. *)

type t
(** Main trainer type *)

type training_result = { model : Models.t; special_tokens : string list }
(** Training result *)

(** {1 Training Configurations} *)

val bpe :
  ?vocab_size:int ->
  ?min_frequency:int ->
  ?special_tokens:string list ->
  ?limit_alphabet:int ->
  ?initial_alphabet:string list ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?show_progress:bool ->
  ?max_token_length:int ->
  unit ->
  t
(** Create a BPE trainer.
    @param vocab_size Target vocabulary size (default: 30000)
    @param min_frequency Minimum frequency for tokens (default: 0)
    @param show_progress Show training progress (default: true)
    @param special_tokens List of special tokens (default: [])
    @param limit_alphabet Maximum alphabet size (default: 1000)
    @param initial_alphabet Initial alphabet (default: [])
    @param continuing_subword_prefix
      Prefix for continuing subwords (default: None)
    @param end_of_word_suffix Suffix for end of word (default: None) *)

val wordpiece :
  ?vocab_size:int ->
  ?min_frequency:int ->
  ?special_tokens:string list ->
  ?limit_alphabet:int ->
  ?initial_alphabet:string list ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?unk_token:string ->
  ?show_progress:bool ->
  unit ->
  t
(** Create a WordPiece trainer.
    @param vocab_size Target vocabulary size (default: 30000)
    @param min_frequency Minimum frequency for tokens (default: 0)
    @param show_progress Show training progress (default: true)
    @param special_tokens List of special tokens (default: [])
    @param limit_alphabet Maximum alphabet size (default: 1000)
    @param initial_alphabet Initial alphabet (default: [])
    @param continuing_subword_prefix
      Prefix for continuing subwords (default: "##") *)

val word_level :
  ?vocab_size:int ->
  ?min_frequency:int ->
  ?special_tokens:string list ->
  ?show_progress:bool ->
  unit ->
  t
(** Create a WordLevel trainer.
    @param vocab_size Target vocabulary size (default: 30000)
    @param min_frequency Minimum frequency for tokens (default: 0)
    @param show_progress Show training progress (default: true)
    @param special_tokens List of special tokens (default: []) *)

val unigram :
  ?vocab_size:int ->
  ?n_sub_iterations:int ->
  ?shrinking_factor:float ->
  ?unk_token:string ->
  ?special_tokens:string list ->
  ?show_progress:bool ->
  ?initial_alphabet:string list ->
  ?max_piece_length:int ->
  unit ->
  t
(** Create a Unigram trainer.
    @param vocab_size Target vocabulary size (default: 8000)
    @param show_progress Show training progress (default: true)
    @param special_tokens List of special tokens (default: [])
    @param shrinking_factor Shrinking factor (default: 0.75)
    @param unk_token Unknown token (default: None)
    @param max_piece_length Maximum piece length (default: 16)
    @param n_sub_iterations Number of sub-iterations (default: 2) *)

val chars :
  ?min_frequency:int ->
  ?special_tokens:string list ->
  ?show_progress:bool ->
  unit ->
  t
(** Create a character-level trainer.
    @param min_frequency Minimum frequency for characters (default: 0)
    @param special_tokens List of special tokens (default: [])
    @param show_progress Show training progress (default: true) *)

(** {1 Training Operations} *)

val train : t -> files:string list -> ?model:Models.t -> unit -> training_result
(** Train a model on the given files.
    @param t The trainer configuration
    @param files List of training files
    @param model Optional existing model to continue training from
    @return The trained model and special tokens *)

val train_from_iterator :
  t ->
  iterator:(unit -> string option) ->
  ?model:Models.t ->
  unit ->
  training_result
(** Train a model from an iterator.
    @param t The trainer configuration
    @param iterator Function that returns next line or None
    @param model Optional existing model to continue training from
    @return The trained model and special tokens *)

(** {1 Serialization} *)

val to_json : t -> Yojson.Basic.t
val of_json : Yojson.Basic.t -> t
