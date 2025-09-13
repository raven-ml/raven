(** Encoding module - represents the output of a tokenizer *)

type t
(** The main encoding type - abstract to users *)

val create :
  ids:int array ->
  type_ids:int array ->
  tokens:string array ->
  words:int option array ->
  offsets:(int * int) array ->
  special_tokens_mask:int array ->
  attention_mask:int array ->
  overflowing:t list ->
  sequence_ranges:(int, int * int) Hashtbl.t ->
  t
(** Create a new encoding - for internal use *)

val with_capacity : int -> t
(** Create an empty encoding with given capacity *)

val from_tokens : (int * string * (int * int)) list -> type_id:int -> t
(** Create encoding from tokens *)

val is_empty : t -> bool
(** Check if encoding is empty *)

val length : t -> int
(** Get the length of the encoding *)

val n_sequences : t -> int
(** Get the number of sequences in the encoding *)

val set_sequence_id : t -> int -> t
(** Set sequence id for the whole encoding *)

(** {2 Accessors} *)

val get_ids : t -> int array
(** Get IDs *)

val get_type_ids : t -> int array
(** Get type IDs *)

val set_type_ids : t -> int array -> t
(** Set type IDs *)

val get_tokens : t -> string array
(** Get tokens *)

val get_word_ids : t -> int option array
(** Get word IDs *)

val get_sequence_ids : t -> int option array
(** Get sequence IDs for each token *)

val get_offsets : t -> (int * int) array
(** Get offsets *)

val get_special_tokens_mask : t -> int array
(** Get special tokens mask *)

val get_attention_mask : t -> int array
(** Get attention mask *)

val get_overflowing : t -> t list
(** Get overflowing encodings *)

val set_overflowing : t -> t list -> t
(** Set overflowing encodings *)

val take_overflowing : t -> t * t list
(** Take overflowing encodings (removes them from encoding) *)

(** {2 Token/Word/Char mappings} *)

val token_to_sequence : t -> int -> int option
(** Get the sequence index containing the given token *)

val token_to_word : t -> int -> (int * int) option
(** Get the word containing the given token *)

val token_to_chars : t -> int -> (int * (int * int)) option
(** Get the character offsets of the given token *)

val word_to_tokens : t -> word:int -> sequence_id:int -> (int * int) option
(** Get the tokens corresponding to the given word *)

val word_to_chars : t -> word:int -> sequence_id:int -> (int * int) option
(** Get the character offsets of the given word *)

val char_to_token : t -> pos:int -> sequence_id:int -> int option
(** Get the token containing the given character position *)

val char_to_word : t -> pos:int -> sequence_id:int -> int option
(** Get the word containing the given character position *)

(** {2 Operations} *)

(** Truncation direction *)
type truncation_direction = Left | Right

val truncate :
  t -> max_length:int -> stride:int -> direction:truncation_direction -> t
(** Truncate the encoding *)

val merge : t list -> growing_offsets:bool -> t
(** Merge multiple encodings *)

val merge_with : t -> t -> growing_offsets:bool -> t
(** Merge with another encoding in place *)

(** Padding direction *)
type padding_direction = Left | Right

val pad :
  t ->
  target_length:int ->
  pad_id:int ->
  pad_type_id:int ->
  pad_token:string ->
  direction:padding_direction ->
  t
(** Pad the encoding *)
