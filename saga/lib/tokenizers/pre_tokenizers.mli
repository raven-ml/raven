(** Pre-tokenizers implementation for text processing *)

type t_with_offsets = string -> (string * (int * int)) list
(** Type for pre-tokenizer functions that return pieces with offsets *)

val bert : t_with_offsets
(** BERT pre-tokenizer: splits on whitespace and punctuation *)

val byte_level :
  ?add_prefix_space:bool -> ?use_regex:bool -> unit -> t_with_offsets
(** ByteLevel pre-tokenizer for GPT-2 style tokenization
    @param add_prefix_space
      whether to add a space prefix if text doesn't start with space
    @param use_regex whether to use GPT-2 regex pattern for splitting *)

val whitespace : t_with_offsets
(** Whitespace tokenizer with pattern \w+|[^\w\s]+ *)

val whitespace_split : t_with_offsets
(** Simple whitespace split (removes whitespace) *)

type behavior =
  [ `Isolated  (** Keep delimiter as separate token *)
  | `Removed  (** Remove delimiter *)
  | `Merged_with_previous  (** Merge delimiter with previous token *)
  | `Merged_with_next  (** Merge delimiter with next token *)
  | `Contiguous  (** Group consecutive delimiters *) ]
(** Split delimiter behavior *)

val punctuation : ?behavior:behavior -> unit -> t_with_offsets
(** Punctuation splitter with configurable behavior *)

val split :
  pattern:string -> behavior:behavior -> ?invert:bool -> unit -> t_with_offsets
(** Split on pattern with configurable behavior
    @param pattern the string pattern to split on
    @param behavior how to handle the delimiter
    @param invert if true, split on everything except the pattern *)

val char_delimiter_split : char -> t_with_offsets
(** Character delimiter split (removes delimiter) *)

val digits : ?individual_digits:bool -> unit -> t_with_offsets
(** Digits splitter
    @param individual_digits if true, each digit is a separate token *)

type prepend_scheme =
  [ `First  (** Only prepend to first piece *)
  | `Never  (** Never prepend *)
  | `Always  (** Always prepend if not starting with space *) ]
(** Prepend scheme for metaspace *)

val metaspace :
  ?replacement:string ->
  ?prepend_scheme:prepend_scheme ->
  ?split:bool ->
  ?is_first:bool ->
  unit ->
  t_with_offsets
(** Metaspace pre-tokenizer (replaces spaces with replacement char)
    @param replacement the character to replace spaces with (default "▁")
    @param prepend_scheme when to prepend the replacement
    @param split whether to split on replacement character
    @param is_first
      whether this is the first piece in a sequence (for `First scheme) *)

val sequence : t_with_offsets list -> t_with_offsets
(** Apply a sequence of pre-tokenizers *)

val fixed_length : length:int -> t_with_offsets
(** FixedLength pre-tokenizer: split into fixed-length chunks
    @param length number of characters per chunk *)

val unicode_scripts : t_with_offsets
(** UnicodeScripts pre-tokenizer: split on script boundaries *)

(** {1 Internal Helpers - exposed for testing} *)

val split_gpt2_pattern : string -> string list
(** Split text using GPT-2 regex pattern *)

val byte_level_decode : string -> string
(** Decode ByteLevel encoded text back to original bytes *)
