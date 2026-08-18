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
  [ `Isolated  (** Keep each delimiter as a piece of its own *)
  | `Removed  (** Drop the delimiters *)
  | `Merged_with_previous  (** Keep a delimiter with what precedes it *)
  | `Merged_with_next  (** Keep a delimiter with what follows it *)
  | `Contiguous
    (** Keep neighbours that are both delimiters, or both not, as one piece *)
  ]
(** Delimiter handling behavior for splitting operations.

    [`Merged_with_previous] and [`Merged_with_next] merge a delimiter into the
    text beside it only: a delimiter that follows another one is a piece of its
    own. *)

val punctuation : ?behavior:behavior -> unit -> t
(** [punctuation ()] separates punctuation characters from the text around them,
    whitespace included. Every punctuation character is a delimiter.

    [behavior] defaults to [`Isolated]. *)

val split : pattern:string -> ?behavior:behavior -> ?invert:bool -> unit -> t
(** [split ~pattern ()] splits on a literal string [pattern]. HuggingFace's
    regular expression patterns have no equivalent here.

    [behavior] defaults to [`Removed]. When [invert] is [true] the delimiters
    are the runs of text between the occurrences of [pattern], and those
    occurrences are what they separate; defaults to [false].

    An empty [pattern] matches at every position, so the pieces are the
    characters — and none of them when [invert] is [true] and [behavior] is
    [`Removed]. *)

val char_delimiter : string -> t
(** [char_delimiter c] splits on the character [c], removing it from the output.
    Equivalent to [split ~pattern:c ~behavior:`Removed ()].

    Raises [Invalid_argument] if [c] is not exactly one character. *)

val digits : ?individual_digits:bool -> unit -> t
(** [digits ()] splits on digit boundaries.

    When [individual_digits] is [true], each digit is a separate piece; when
    [false] (default), consecutive digits are grouped. *)

type prepend_scheme =
  [ `First  (** Prepend to the piece that opens the document only *)
  | `Never  (** Never prepend *)
  | `Always  (** Prepend to every piece not starting with a space *) ]
(** Controls when metaspace prepends the replacement character. In a {!sequence}
    the pieces are those of the member before it, so [`First] marks the first of
    them and [`Always] each — and the {!Brot} pipeline counts a piece as opening
    the document only when nothing, an added token included, comes before it. *)

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

    Pieces are those of the marked text; their offsets are the bytes of the text
    they were made from. A marker that replaced a space stands at that space,
    and a prepended one at the character it opens.

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
(** [pre_tokenize t text] splits [text] into pieces with byte offsets.

    Returns a list of [(piece, (start, end_))] where [start] and [end_] are byte
    positions in [text], ascending and within it. A piece is the bytes of its
    span unless [t] rewrote or encoded them, in which case the span is where the
    bytes it was made from lie.

    Two spans can cover the same bytes. A span is widened to whole characters,
    so pieces of a [text] that is not valid UTF-8 can share one; and the pieces
    a member of a {!sequence} cuts from one that was rewritten more than once,
    cut by a fixed-length member, or encoded all report that piece's span,
    nothing in it being placeable more finely than the whole of it. *)

(** {1 Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] formats [t] for inspection. *)

(** {1:byte_level_decode Byte-level decoding} *)

val byte_level_decode : string -> string
(** [byte_level_decode s] is the bytes the byte-level alphabet spells [s] from:
    one byte per character. A single character outside the alphabet, an
    invalidly encoded one included, leaves [s] as it is — the fallback is over
    the whole string, not the character.

    The result needs not be valid UTF-8: it is bytes, and turning them into text
    is the caller's step. *)

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
  | Space_marker of { marker : string; prepend : prepend_scheme }
      (** Spaces become [marker], and one is prepended when [prepend] asks for
          it — on [`First] only to text that opens the document — and the text
          is neither empty nor already starting with a space or with [marker].
          Offsets are those of the marked text the caller filled, which
          {!pre_tokenize} maps back to the text it was given. *)

(** The type for how a pre-tokenizer takes part in the walking path.
    [splittable] is [true] when cutting the input at a space that separates two
    non-whitespace characters and walking the halves gives the spans of the
    whole. *)
type plan =
  | Walk of { rewrite : rewrite; splittable : bool }
      (** One walk over the text once [rewrite] has been applied to it. *)
  | Segmented of { outer : t; rewrite : rewrite; inner : t; splittable : bool }
      (** [outer] walks the text as it is into segments, and [inner] walks each
          segment once [rewrite] has been applied to it, which is how a
          {!sequence} rewrites: member by member over the pieces of the one
          before. Both parts are walkable, and neither is a rewrite of the
          other's text: [inner] is handed the rewritten segment. *)
  | Pieces  (** Not walkable: {!pre_tokenize} is the only implementation. *)

val plan : t -> plan
(** [plan t] is how [t] takes part in the walking path. *)

val rewriter : rewrite -> first:bool Lazy.t -> string -> Normalizer.t option
(** [rewriter rewrite ~first text] is the normalizer that performs [rewrite] on
    [text], or [None] when it would leave [text] as it is. [first] says whether
    [text] opens the document, and is only forced by a [Space_marker] that
    prepends on [`First]. The normalizers are built when [rewriter rewrite] is
    applied, so a caller with many texts applies it once. *)

val pieces : t -> first:bool Lazy.t -> string -> (string * (int * int)) list
(** [pieces t ~first text] is {!pre_tokenize} for a [text] that opens the
    document iff [first] does: the marker a metaspace prepends on [`First] goes
    to the piece at [0] only then. [first] is forced only by such a metaspace.
    {!pre_tokenize} is [pieces] with [first] holding. *)

val encodes_bytes : t -> bool
(** [encodes_bytes t] is [true] iff the pieces of [t] are byte-level encoded,
    that is iff [t] is a byte-level pre-tokenizer or a sequence ending in one.
    {!fill} does no such rewriting, so a model behind one is handed the raw
    bytes of a span and has to match its vocabulary against those. *)

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

    Raises [Invalid_argument] unless [plan t] is [Walk], or if [pos] and [stop]
    are not within [0] and [String.length text]. The parts of a [Segmented] plan
    are filled on their own. *)
