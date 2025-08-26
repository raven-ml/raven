(** Pre-tokenizers module - splits text before tokenization

    Following HuggingFace tokenizers API design *)

type t = string -> string list
(** Pre-tokenizer type - a function that splits text into pieces *)

type t_with_offsets = string -> (string * (int * int)) list
(** Pre-tokenizer with offset tracking *)

(** {2 Pre-tokenizer Implementations} *)

val whitespace_split : t
(** Whitespace pre-tokenizer - splits on whitespace *)

val whitespace : t
(** Whitespace pre-tokenizer with regex pattern \w+|[^\w\s]+ *)

(** ByteLevel pre-tokenizer for GPT-2 style tokenization *)
val byte_level : ?add_prefix_space:bool -> ?use_regex:bool -> unit -> t
(** [byte_level ~add_prefix_space ~use_regex ()] creates a ByteLevel
    pre-tokenizer.
    - [add_prefix_space]: Whether to add a space to the first word (default:
      true)
    - [use_regex]: Use GPT-2 specific regex for splitting (default: true) *)

val bert : t
(** BertPreTokenizer - splits on spaces and punctuation *)

val punctuation :
  ?behavior:
    [ `Isolated
    | `Removed
    | `MergedWithPrevious
    | `MergedWithNext
    | `Contiguous ] ->
  unit ->
  t
(** Split on punctuation *)

val split :
  pattern:string ->
  behavior:
    [ `Isolated
    | `Removed
    | `MergedWithPrevious
    | `MergedWithNext
    | `Contiguous ] ->
  ?invert:bool ->
  unit ->
  t
(** Split using a custom pattern *)

val char_delimiter_split : char -> t
(** Character delimiter split *)

val digits : ?individual_digits:bool -> unit -> t
(** Digits pre-tokenizer *)

val metaspace :
  ?replacement:string ->
  ?prepend_scheme:[ `Always | `Never | `First ] ->
  ?split:bool ->
  unit ->
  t
(** Metaspace pre-tokenizer (like SentencePiece) *)

val sequence : t list -> t
(** Sequence of pre-tokenizers applied in order *)

(** {2 Utility Functions} *)

val byte_level_alphabet : unit -> string list
(** Get the ByteLevel alphabet (256 chars mapping bytes to visible chars) *)
