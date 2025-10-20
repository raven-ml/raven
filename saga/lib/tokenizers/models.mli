(** Tokenization models module. *)

type token = { id : int; value : string; offsets : int * int }
(** Tokenization result *)

type bpe_model = {
  vocab : (string, int) Hashtbl.t;
  merges : (string * string) list;
  cache_capacity : int;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
}
(** Model configurations *)

type wordpiece_model = {
  vocab : (string, int) Hashtbl.t;
  unk_token : string;
  max_input_chars_per_word : int;
}

type wordlevel_model = { vocab : (string, int) Hashtbl.t; unk_token : string }

type unigram_model = {
  vocab : (string * float) list;
  token_map : (string, int) Hashtbl.t;
  tokens : string array;
}

(** Main model type *)
type t =
  | BPE of bpe_model
  | WordPiece of wordpiece_model
  | WordLevel of wordlevel_model
  | Unigram of unigram_model

(** {1 Constructors} *)

val bpe :
  ?vocab:(string * int) list ->
  ?merges:(string * string) list ->
  ?cache_capacity:int ->
  ?dropout:float ->
  ?unk_token:string ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?fuse_unk:bool ->
  ?byte_fallback:bool ->
  ?ignore_merges:bool ->
  unit ->
  t
(** Create a BPE model *)

val wordpiece :
  ?vocab:(string * int) list ->
  ?unk_token:string ->
  ?continuing_subword_prefix:string ->
  ?max_input_chars_per_word:int ->
  unit ->
  t
(** Create a WordPiece model *)

val word_level : ?vocab:(string * int) list -> ?unk_token:string -> unit -> t
(** Create a WordLevel model *)

val unigram :
  ?vocab:(string * float) list ->
  ?unk_token:string ->
  ?byte_fallback:bool ->
  ?max_piece_length:int ->
  ?n_sub_iterations:int ->
  ?shrinking_factor:float ->
  unit ->
  t
(** Create a Unigram model *)

val chars : unit -> t
(** Create a character-level model *)

val regex : string -> t
(** Create a regex-based model *)

val from_file : vocab:string -> ?merges:string -> unit -> t
(** Load model from files *)

(** {1 Operations} *)

val tokenize : t -> string -> token list
(** Tokenize a string into tokens *)

val token_to_id : t -> string -> int option
(** Get the ID for a token *)

val id_to_token : t -> int -> string option
(** Get the token for an ID *)

val get_vocab : t -> (string * int) list
(** Get the vocabulary *)

val get_vocab_size : t -> int
(** Get the vocabulary size *)

val add_tokens : t -> string list -> int
(** Add tokens to the model's vocabulary. Returns number of tokens added. *)

val save : t -> folder:string -> ?prefix:string -> unit -> string list
(** Save the model to files *)

(** {1 Serialization} *)

val to_json : t -> Yojson.Basic.t
val of_json : Yojson.Basic.t -> t
