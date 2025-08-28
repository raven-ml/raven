(** Byte Pair Encoding (BPE) tokenization module *)

(** {2 Core Types} *)

type t
(** BPE model *)

type vocab = (string, int) Hashtbl.t
(** Vocabulary mapping tokens to indices *)

type merges = (string * string) list
(** List of merge operations *)

type config = {
  vocab : vocab;
  merges : merges;
  cache_capacity : int;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
}
(** BPE configuration *)

(** {2 Model Creation} *)

val create : config -> t
(** [create config] creates a new BPE model with the given configuration *)

val from_files : vocab_file:string -> merges_file:string -> t
(** [from_files ~vocab_file ~merges_file] loads a BPE model from vocab.json and
    merges.txt files *)

val default : unit -> t
(** [default ()] creates a BPE model with default configuration *)

(** {2 Configuration Builder} *)

module Builder : sig
  type builder

  val create : unit -> builder
  (** Create a new builder with default settings *)

  val vocab_and_merges : builder -> vocab -> merges -> builder
  (** Set vocabulary and merges *)

  val cache_capacity : builder -> int -> builder
  (** Set cache capacity (0 to disable) *)

  val dropout : builder -> float -> builder
  (** Set dropout probability (0.0 to 1.0) *)

  val unk_token : builder -> string -> builder
  (** Set unknown token *)

  val continuing_subword_prefix : builder -> string -> builder
  (** Set prefix for continuing subwords *)

  val end_of_word_suffix : builder -> string -> builder
  (** Set suffix for end-of-word tokens *)

  val fuse_unk : builder -> bool -> builder
  (** Set whether to fuse consecutive unknown tokens *)

  val byte_fallback : builder -> bool -> builder
  (** Enable byte-level fallback for unknown characters *)

  val ignore_merges : builder -> bool -> builder
  (** Ignore merges and output words directly if in vocab *)

  val build : builder -> t
  (** Build the BPE model *)
end

(** {2 Tokenization} *)

type token = { id : int; value : string; offsets : int * int }
(** Token with ID, string value, and character offsets *)

val tokenize : t -> string -> token list
(** [tokenize model text] tokenizes text into tokens *)

val token_to_id : t -> string -> int option
(** [token_to_id model token] returns the ID for a token *)

val id_to_token : t -> int -> string option
(** [id_to_token model id] returns the token for an ID *)

(** {2 Vocabulary Management} *)

val get_vocab : t -> (string * int) list
(** [get_vocab model] returns the vocabulary as a list of (token, id) pairs *)

val get_vocab_size : t -> int
(** [get_vocab_size model] returns the size of the vocabulary *)

val get_unk_token : t -> string option
(** [get_unk_token model] returns the unknown token if configured *)

val get_continuing_subword_prefix : t -> string option
(** [get_continuing_subword_prefix model] returns the continuing subword prefix
    if configured *)

val get_end_of_word_suffix : t -> string option
(** [get_end_of_word_suffix model] returns the end-of-word suffix if configured
*)

(** {2 Cache Management} *)

val clear_cache : t -> unit
(** [clear_cache model] clears the tokenization cache *)

val resize_cache : t -> int -> unit
(** [resize_cache model capacity] resizes the cache *)

(** {2 Serialization} *)

val save : t -> path:string -> ?name:string -> unit -> unit
(** [save model ~path ?name ()] saves the model to vocab.json and merges.txt
    files *)

val read_files : vocab_file:string -> merges_file:string -> vocab * merges
(** [read_files ~vocab_file ~merges_file] reads vocabulary and merges from files
*)

(** {2 Training} *)

module Trainer : sig
  type trainer

  type trainer_config = {
    min_frequency : int;
    vocab_size : int;
    show_progress : bool;
    special_tokens : string list;
    limit_alphabet : int option;
    initial_alphabet : char list;
    continuing_subword_prefix : string option;
    end_of_word_suffix : string option;
    max_token_length : int option;
  }

  val default_config : trainer_config
  (** Default trainer configuration *)

  val create : trainer_config -> trainer
  (** Create a new trainer *)

  val feed : trainer -> string list -> unit
  (** Feed training data to the trainer *)

  val train : trainer -> t -> string list
  (** Train the model and return special tokens *)
end
