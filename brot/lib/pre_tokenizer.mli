(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pre-tokenization.

    Pre-tokenizers split raw text into pieces before vocabulary-based
    tokenization (BPE, WordPiece, etc.) is applied. Each piece carries byte
    offsets into the original text.

    See {!Brot} for the full tokenization pipeline. *)

type t
(** The type for pre-tokenizers. *)

(** {1:constructors Constructors} *)

val whitespace : unit -> t
(** [whitespace ()] splits on whitespace using pattern [\w+|[^\w\s]+].

    Groups word characters (letters, digits, underscore) together and groups
    non-word, non-space characters together. Whitespace is used as delimiter but
    not included in output. *)

val whitespace_split : unit -> t
(** [whitespace_split ()] splits on any whitespace characters.

    Removes whitespace from output. Simplest and fastest pre-tokenizer. *)

val bert : unit -> t
(** [bert ()] applies BERT-style pre-tokenization.

    Splits on whitespace, isolates punctuation, and separates CJK characters
    individually. *)

val byte_level :
  ?add_prefix_space:bool -> ?use_regex:bool -> ?trim_offsets:bool -> unit -> t
(** [byte_level ()] is a byte-level pre-tokenizer. Used by GPT-2, GPT-3,
    RoBERTa.

    Converts text to byte representation and applies GPT-2's regex pattern for
    splitting.

    - [add_prefix_space]: add space at beginning if text does not start with
      whitespace. Default: [true].
    - [use_regex]: use GPT-2's regex pattern for splitting. Default: [true].
    - [trim_offsets]: adjust offsets for byte-level encoding. Default: [true].
*)

type behavior =
  [ `Isolated  (** Keep delimiter as separate piece *)
  | `Removed  (** Remove delimiter *)
  | `Merged_with_previous  (** Merge delimiter with previous piece *)
  | `Merged_with_next  (** Merge delimiter with next piece *)
  | `Contiguous  (** Group consecutive delimiters together *) ]
(** Delimiter handling behavior for splitting operations. *)

val punctuation : ?behavior:behavior -> unit -> t
(** [punctuation ()] separates punctuation from alphanumeric content.

    [behavior] defaults to [`Isolated]. *)

val split : pattern:string -> ?behavior:behavior -> ?invert:bool -> unit -> t
(** [split ~pattern ()] splits on a literal string [pattern].

    [behavior] defaults to [`Removed]. When [invert] is [true], splits on
    everything {e except} the pattern; defaults to [false]. *)

val char_delimiter : char -> t
(** [char_delimiter c] splits on character [c], removing it from output.

    Equivalent to [split ~pattern:(String.make 1 c) ~behavior:`Removed ()]. *)

val digits : ?individual_digits:bool -> unit -> t
(** [digits ()] splits on digit boundaries.

    When [individual_digits] is [true], each digit is a separate piece; when
    [false] (default), consecutive digits are grouped. *)

type prepend_scheme =
  [ `First  (** Only prepend to first piece *)
  | `Never  (** Never prepend *)
  | `Always  (** Always prepend if not starting with space *) ]
(** Controls when metaspace prepends the replacement character. *)

val metaspace :
  ?replacement:char ->
  ?prepend_scheme:prepend_scheme ->
  ?split:bool ->
  unit ->
  t
(** [metaspace ()] replaces whitespace with a visible marker. Used by
    SentencePiece models.

    - [replacement]: character to replace spaces with. Default: ['_'].
    - [prepend_scheme]: when to prepend the replacement character. Default:
      [`Always].
    - [split]: whether to split on the replacement character. Default: [true].
*)

val unicode_scripts : unit -> t
(** [unicode_scripts ()] splits on Unicode script boundaries.

    Separates text when the writing system changes (e.g., Latin to Cyrillic,
    Latin to Han). *)

val fixed_length : int -> t
(** [fixed_length n] splits into fixed-length character chunks.

    The last chunk may be shorter than [n]. *)

val sequence : t list -> t
(** [sequence ts] chains multiple pre-tokenizers left-to-right.

    Each pre-tokenizer processes the pieces from the previous one. Offsets are
    composed correctly through the chain. *)

(** {1 Operations} *)

val pre_tokenize : t -> string -> (string * (int * int)) list
(** [pre_tokenize t text] splits [text] into pieces with character offsets.

    Returns a list of [(piece, (start, end_))] where [start] and [end_] are byte
    positions in the original [text]. Offsets are non-overlapping and in
    ascending order. *)

(** {1 Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] formats [t] for inspection. *)

(** {1:byte_level_decode Byte-level decoding} *)

val byte_level_decode : string -> string
(** [byte_level_decode s] reverses byte-level encoding by converting the special
    Unicode codepoints back to original byte values. *)

(** {1 Serialization} *)

val to_json : t -> Jsont.json
(** [to_json t] serializes [t] to HuggingFace JSON format. *)

val of_json : Jsont.json -> (t, string) result
(** [of_json json] is a pre-tokenizer from HuggingFace JSON format. Errors if
    [json] is not an object, has a missing or unknown ["type"] field, or has
    invalid parameters. *)
