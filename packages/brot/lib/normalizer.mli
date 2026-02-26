(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Text normalization.

    Normalizers transform text before tokenization: lowercasing, accent removal,
    Unicode normalization, whitespace cleanup, and model-specific preprocessing.
    They are the first stage in the tokenization pipeline, applied before
    {!Pre_tokenizer} and vocabulary-based encoding.

    Compose normalizers with {!val-sequence}:
    {[
      let n =
        Normalizer.sequence
          [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]
      in
      Normalizer.apply n "Caf\u{00E9}"
      (* "cafe" *)
    ]}

    See {!Brot} for the full tokenization pipeline. *)

type t
(** The type for normalizers. *)

(** {1:normalizers Normalizers} *)

(** {2:unicode Unicode normalization} *)

val nfc : t
(** [nfc] is Unicode NFC normalization (canonical composition). *)

val nfd : t
(** [nfd] is Unicode NFD normalization (canonical decomposition). *)

val nfkc : t
(** [nfkc] is Unicode NFKC normalization (compatibility composition). *)

val nfkd : t
(** [nfkd] is Unicode NFKD normalization (compatibility decomposition). *)

(** {2:text Text transforms} *)

val lowercase : t
(** [lowercase] is Unicode case folding to lowercase. *)

val strip_accents : t
(** [strip_accents] removes combining marks after NFD decomposition. Applies
    {!val-nfd} before stripping. *)

val strip : ?left:bool -> ?right:bool -> unit -> t
(** [strip ?left ?right ()] is a normalizer that strips Unicode whitespace from
    text boundaries. [left] and [right] default to [true]. *)

val replace : pattern:string -> replacement:string -> t
(** [replace ~pattern ~replacement] is a normalizer that replaces all [pattern]
    matches with [replacement]. [pattern] is a PCRE regular expression, compiled
    once at construction time.

    Raises [Re.Pcre.Parse_error] if [pattern] is not valid PCRE. *)

val prepend : string -> t
(** [prepend s] is a normalizer that prepends [s] to non-empty text. Empty text
    is returned unchanged. *)

(** {2:byte_level Byte-level encoding} *)

val byte_level : ?add_prefix_space:bool -> unit -> t
(** [byte_level ?add_prefix_space ()] is GPT-2 style byte-level encoding. Each
    byte is mapped to a printable Unicode codepoint using the GPT-2
    byte-to-unicode table.
    - [add_prefix_space] adds a space prefix when the text does not start with
      whitespace. Defaults to [false]. *)

(** {2:model Model-specific} *)

val bert :
  ?clean_text:bool ->
  ?handle_chinese_chars:bool ->
  ?strip_accents:bool option ->
  ?lowercase:bool ->
  unit ->
  t
(** [bert ()] is a BERT normalizer.

    - [clean_text]: remove control characters and normalize whitespace. Default:
      [true].
    - [handle_chinese_chars]: pad CJK ideographs with spaces. Default: [true].
    - [strip_accents]: strip accents after NFD decomposition. When [None],
      accents are stripped iff [lowercase] is [true]. Default: [None].
    - [lowercase]: lowercase text via Unicode case folding. Default: [true]. *)

(** {2:composition Composition} *)

val sequence : t list -> t
(** [sequence ns] is the composition of normalizers [ns], applied left to right.
*)

(** {1:applying Applying} *)

val apply : t -> string -> string
(** [apply n s] is [s] normalized by [n]. *)

(** {1:formatting Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf n] formats [n] for inspection. *)

(** {1:serialization Serialization} *)

val to_json : t -> Jsont.json
(** [to_json n] is [n] serialized to HuggingFace-compatible JSON. *)

val of_json : Jsont.json -> (t, string) result
(** [of_json json] is a normalizer deserialized from HuggingFace JSON. Errors if
    [json] is not an object, has a missing or unknown ["type"] field, or has
    invalid parameters. *)
