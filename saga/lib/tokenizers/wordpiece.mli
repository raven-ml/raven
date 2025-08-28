(** WordPiece tokenization module

    WordPiece is the subword tokenization algorithm used by BERT. It uses a
    greedy longest-match-first algorithm to tokenize text. *)

(** {2 Core Types} *)

type t
(** WordPiece model *)

type vocab = (string, int) Hashtbl.t
(** Vocabulary mapping tokens to indices *)

type config = {
  vocab : vocab;
  unk_token : string;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}
(** WordPiece configuration *)

(** {2 Model Creation} *)

val create : config -> t
(** [create config] creates a new WordPiece model with the given configuration
*)

val from_file : vocab_file:string -> t
(** [from_file ~vocab_file] loads a WordPiece model from a vocab.txt file with
    default settings *)

val from_file_with_config :
  vocab_file:string ->
  unk_token:string ->
  continuing_subword_prefix:string ->
  max_input_chars_per_word:int ->
  t
(** [from_file_with_config ~vocab_file ~unk_token ~continuing_subword_prefix
     ~max_input_chars_per_word] loads a WordPiece model from a vocab.txt file
    with custom configuration *)

val default : unit -> t
(** [default ()] creates a WordPiece model with default configuration *)

(** {2 Configuration Builder} *)

module Builder : sig
  type builder

  val create : unit -> builder
  (** Create a new builder with default settings *)

  val files : builder -> string -> builder
  (** Set vocabulary file path *)

  val vocab : builder -> vocab -> builder
  (** Set vocabulary directly *)

  val unk_token : builder -> string -> builder
  (** Set unknown token (default: "[UNK]") *)

  val continuing_subword_prefix : builder -> string -> builder
  (** Set prefix for continuing subwords (default: "##") *)

  val max_input_chars_per_word : builder -> int -> builder
  (** Set maximum input characters per word (default: 100) *)

  val build : builder -> t
  (** Build the WordPiece model *)
end

(** {2 Tokenization} *)

type token = { id : int; value : string; offsets : int * int }
(** Token with ID, string value, and character offsets *)

val tokenize : t -> string -> token list
(** [tokenize model text] tokenizes text into tokens using greedy
    longest-match-first *)

val token_to_id : t -> string -> int option
(** [token_to_id model token] returns the ID for a token *)

val id_to_token : t -> int -> string option
(** [id_to_token model id] returns the token for an ID *)

(** {2 Vocabulary Management} *)

val get_vocab : t -> (string * int) list
(** [get_vocab model] returns the vocabulary as a list of (token, id) pairs *)

val get_vocab_size : t -> int
(** [get_vocab_size model] returns the size of the vocabulary *)

val get_unk_token : t -> string
(** [get_unk_token model] returns the unknown token *)

val get_continuing_subword_prefix : t -> string
(** [get_continuing_subword_prefix model] returns the continuing subword prefix
*)

val get_max_input_chars_per_word : t -> int
(** [get_max_input_chars_per_word model] returns the maximum input characters
    per word *)

(** {2 Serialization} *)

val save : t -> path:string -> ?name:string -> unit -> string
(** [save model ~path ?name ()] saves the model to vocab.txt file and returns
    the filepath *)

val read_file : vocab_file:string -> vocab
(** [read_file ~vocab_file] reads vocabulary from file *)

val read_bytes : bytes -> vocab
(** [read_bytes bytes] reads vocabulary from bytes *)

val to_yojson : t -> Yojson.Basic.t
(** [to_yojson model] converts model to JSON *)

val of_yojson : Yojson.Basic.t -> t
(** [of_yojson json] creates model from JSON *)

val from_bytes : bytes -> t
(** [from_bytes bytes] creates model from serialized bytes *)

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
    continuing_subword_prefix : string;
    end_of_word_suffix : string option;
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

(** {2 Conversion} *)

val from_bpe : Bpe.t -> t
(** [from_bpe bpe] creates a WordPiece model from a BPE model *)
