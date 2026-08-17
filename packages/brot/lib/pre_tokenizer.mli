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
(** [split ~pattern ()] splits on a literal string [pattern]. HuggingFace's
    regular expression patterns have no equivalent here.

    [behavior] defaults to [`Removed]. When [invert] is [true], splits on
    everything {e except} the pattern; defaults to [false]. *)

val char_delimiter : string -> t
(** [char_delimiter c] splits on the character [c], removing it from the output.
    Equivalent to [split ~pattern:c ~behavior:`Removed ()].

    Raises [Invalid_argument] if [c] is not exactly one character. *)

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
  ?replacement:string ->
  ?prepend_scheme:prepend_scheme ->
  ?split:bool ->
  unit ->
  t
(** [metaspace ()] replaces spaces with a visible marker. Used by SentencePiece
    models.

    - [replacement]: the marker spaces are replaced with. Default: ["▁"]
      (U+2581). It must be exactly one character.
    - [prepend_scheme]: when to prepend the marker, which happens only if the
      marked text does not start with one already. Default: [`Always].
    - [split]: whether to split before each marker. Default: [true].

    Pieces are those of the marked text, and so are offsets unless [split] is
    [false].

    Raises [Invalid_argument] if [replacement] is not exactly one character. *)

val unicode_scripts : unit -> t
(** [unicode_scripts ()] splits on Unicode script boundaries.

    A piece opens where the writing system changes (e.g. Latin to Cyrillic,
    Latin to Han) and runs to the next change. Hiragana and Katakana count as
    Han, as does the prolonged sound mark ["ー"] (U+30FC).

    Spaces and characters of no known script join the piece they follow, so a
    leading run of them belongs to no piece: unlike the other pre-tokenizers,
    the pieces need not cover the input. *)

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
    [json] is not an object, has a missing or unknown ["type"] field, has
    invalid parameters, or is a ["Split"] whose pattern is a regular expression
    ([{"Regex": ...}]) rather than a literal ([{"String": ...}]). *)

(** {1:internals Internals}

    Pre-tokenization as the encode path sees it: byte ranges written into a
    buffer, rather than pieces. {!pre_tokenize} is these functions plus the
    strings.

    These are for {!Brot}'s own use and are not part of the stable interface.
    [Spans.t] belongs to a module the library does not export, so {!fill} has no
    caller outside it. *)

(** The type for text rewrites that precede a walk. *)
type rewrite =
  | Verbatim  (** Spans index the text as it was given. *)
  | Prefix_space
      (** A [' '] is prepended unless the text starts with one; offsets shift
          back by one. *)
  | Space_marker of { marker : string; prepend : bool }
      (** Spaces become [marker], and one is prepended when [prepend] holds and
          the text does not already start with a space. Offsets are those of the
          marked text. *)

(** The type for how a pre-tokenizer takes part in the walking path.
    [splittable] is [true] when cutting the input at a space that separates two
    non-whitespace characters and walking the halves gives the spans of the
    whole. *)
type plan =
  | Walk of { rewrite : rewrite; splittable : bool }
  | Pieces  (** Not walkable: {!pre_tokenize} is the only implementation. *)

val plan : t -> plan
(** [plan t] is how [t] takes part in the walking path. *)

val fill : t -> string -> pos:int -> stop:int -> Spans.t -> int
(** [fill t text ~pos ~stop spans] appends to [spans] the pretoken spans of the
    bytes of [text] between [pos] and [stop], treating that range as the whole
    text, and is the position to resume from: [stop] once the range is
    exhausted, and otherwise the start of the first span that did not fit. Spans
    are never empty and are never cut short. Positions must fit in 32 bits.

    [text] needs not be valid UTF-8: no byte outside the range is read.

    A sequence can also return [pos] having appended nothing, when one of its
    members needs more room than [spans] holds for a single span of the member
    before it. A caller that reaches that state makes no progress until it calls
    again with a larger buffer.

    Raises [Invalid_argument] if [plan t] is [Pieces], or if [pos] and [stop]
    are not within [0] and [String.length text]. *)
