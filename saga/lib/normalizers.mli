(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Text normalization module matching HuggingFace tokenizers.

    Normalizers are responsible for cleaning and transforming text before
    tokenization. This includes operations like lowercasing, accent removal,
    Unicode normalization, and handling special characters.

    {1 Overview}

    Normalization is the first stage in tokenization pipelines, applied before
    pre-tokenization and vocabulary-based encoding. It ensures consistent text
    representation, handles Unicode quirks, and removes irrelevant variations.

    Common normalization operations:
    - Lowercasing for case-insensitive models
    - Accent removal (é → e) for language-agnostic matching
    - Unicode normalization (NFC/NFD/NFKC/NFKD) for canonical representation
    - Whitespace cleanup and control character removal
    - BERT-specific preprocessing (CJK handling, accent stripping)

    {1 When to Use Normalization}

    Apply normalization when you want to:
    - Reduce vocabulary size by merging case variants (Hello/hello)
    - Handle accented characters uniformly (café/cafe)
    - Clean noisy text (control characters, extra whitespace)
    - Match model-specific preprocessing (BERT, GPT-2)

    Skip normalization when you need to:
    - Preserve case distinctions (proper nouns, acronyms)
    - Keep accent information (é vs e are different)
    - Process code or structured data with meaningful formatting

    {1 Usage Examples}

    Simple lowercasing:
    {[
      let normalizer = Normalizers.lowercase () in
      let result = Normalizers.normalize_str normalizer "Hello World!" in
      (* result = "hello world!" *)
    ]}

    BERT-style normalization:
    {[
      let normalizer = Normalizers.bert ~lowercase:true () in
      let result = Normalizers.normalize_str normalizer "  Héllo\tWorld!  " in
      (* Cleans whitespace, removes accents, lowercases *)
    ]}

    Combining multiple normalizers:
    {[
      let normalizer = Normalizers.sequence [
        Normalizers.nfd ();  (* Decompose accented chars *)
        Normalizers.strip_accents ();  (* Remove accent marks *)
        Normalizers.lowercase ();  (* Convert to lowercase *)
        Normalizers.strip ~left:true ~right:true ();  (* Trim whitespace *)
      ] in
      let result = Normalizers.normalize_str normalizer "  Café  " in
      (* result = "cafe" *)
    ]}

    {1 Unicode Normalization Forms}

    Unicode provides four normalization forms for canonical representation:

    - {b NFC (Canonical Composition)}: Decomposes then recomposes characters.
      Preferred for most text processing. é stored as single character U+00E9.

    - {b NFD (Canonical Decomposition)}: Decomposes characters into base +
      combining marks. é stored as e (U+0065) + ́ (U+0301). Useful before accent
      removal.

    - {b NFKC (Compatibility Composition)}: Replaces compatibility characters
      with canonical equivalents, then composes. Converts ﬁ (ligature) → fi.
      Lossy but reduces variation.

    - {b NFKD (Compatibility Decomposition)}: Compatibility decomposition
      without recomposition. Most aggressive normalization, useful for search.

    Typical usage:
    - Use NFC for storage and display (most compact)
    - Use NFD before accent stripping
    - Use NFKC/NFKD for fuzzy matching and search *)

type normalized_string = {
  normalized : string;  (** The normalized text *)
  original : string;  (** The original text *)
  alignments : (int * int) array;
      (** Alignment mappings from normalized to original positions *)
}
(** Type representing a normalized string with alignment information *)

type t
(** Main normalizer type *)

(** {1 Constructors} *)

val bert :
  ?clean_text:bool ->
  ?handle_chinese_chars:bool ->
  ?strip_accents:bool option ->
  ?lowercase:bool ->
  unit ->
  t
(** [bert ~clean_text ~handle_chinese_chars ~strip_accents ~lowercase ()]
    creates a BERT normalizer.

    @param clean_text
      Remove control characters and normalize whitespace (default: true)
    @param handle_chinese_chars Add spaces around CJK characters (default: true)
    @param strip_accents
      Strip accents (None means auto-detect based on lowercase) (default: None)
    @param lowercase Convert to lowercase (default: true) *)

val strip : ?left:bool -> ?right:bool -> unit -> t
(** [strip ~left ~right ()] removes whitespace from text boundaries.

    Trims whitespace characters from the beginning and/or end of text. Does not
    affect internal whitespace.

    @param left Strip whitespace from left (beginning). Default: true.
    @param right Strip whitespace from right (end). Default: true.

    {[
      let normalizer = Normalizers.strip ~left:true ~right:true () in
      let result = Normalizers.normalize_str normalizer "  Hello  " in
      (* result = "Hello" *)
    ]} *)

val strip_accents : unit -> t
(** [strip_accents ()] removes accent marks from characters.

    Converts accented characters to their base forms (é → e, ñ → n). Uses
    Unicode NFD decomposition followed by removal of combining marks. Typically
    applied after NFD normalization.

    {[
      let normalizer = Normalizers.strip_accents () in
      let result = Normalizers.normalize_str normalizer "Café résumé" in
      (* result = "Cafe resume" *)
    ]} *)

val nfc : unit -> t
(** [nfc ()] applies Unicode NFC normalization.

    Canonical Decomposition followed by Canonical Composition. Decomposes
    characters (é → e + ́), then recomposes them into precomposed forms (e + ́ →
    é). Produces canonical composed representation.

    Use for: Standard text storage, ensuring consistent representation.

    {[
      let normalizer = Normalizers.nfc () in
      let result = Normalizers.normalize_str normalizer "e\u{0301}" in
      (* Combining e + accent → composed é *)
    ]} *)

val nfd : unit -> t
(** [nfd ()] applies Unicode NFD normalization.

    Canonical Decomposition. Splits precomposed characters into base character
    + combining marks (é → e + ́). Essential before accent stripping.

    Use for: Accent removal pipelines, character-level analysis.

    {[
      let normalizer = Normalizers.nfd () in
      let result = Normalizers.normalize_str normalizer "é" in
      (* Composed é → e + combining accent *)
    ]} *)

val nfkc : unit -> t
(** [nfkc ()] applies Unicode NFKC normalization.

    Compatibility Decomposition followed by Canonical Composition. Replaces
    compatibility characters with canonical equivalents, then composes. Converts
    ligatures (ﬁ → fi), full-width characters (Ａ → A), subscripts. Lossy
    transformation.

    Use for: Fuzzy search, aggressive text normalization.

    {[
      let normalizer = Normalizers.nfkc () in
      let result = Normalizers.normalize_str normalizer "ﬁle" in
      (* Ligature ﬁ → fi, result = "file" *)
    ]} *)

val nfkd : unit -> t
(** [nfkd ()] applies Unicode NFKD normalization.

    Compatibility Decomposition. Most aggressive Unicode normalization.
    Decomposes compatibility characters and canonical characters. Useful for
    maximum normalization and search applications.

    Use for: Aggressive fuzzy matching, search indexing.

    {[
      let normalizer = Normalizers.nfkd () in
      let result = Normalizers.normalize_str normalizer "ﬁ" in
      (* Decomposes ligatures, compatibility forms *)
    ]} *)

val lowercase : unit -> t
(** [lowercase ()] converts text to lowercase.

    Applies Unicode lowercase transformation. Language-agnostic but may not
    handle all language-specific casing rules correctly (e.g., Turkish i).

    {[
      let normalizer = Normalizers.lowercase () in
      let result = Normalizers.normalize_str normalizer "Hello World!" in
      (* result = "hello world!" *)
    ]} *)

val replace : pattern:string -> replacement:string -> unit -> t
(** [replace ~pattern ~replacement ()] replaces text matching regex pattern.

    Finds all matches of pattern and replaces them with replacement string.
    Useful for custom text transformations.

    @param pattern Regular expression pattern (Str module syntax)
    @param replacement Replacement string (supports backreferences)

    {[
      let normalizer = Normalizers.replace ~pattern:"[0-9]+" ~replacement:"<NUM>" () in
      let result = Normalizers.normalize_str normalizer "I have 123 apples" in
      (* result = "I have <NUM> apples" *)
    ]} *)

val prepend : prepend:string -> t
(** [prepend ~prepend] prepends string to text.

    Adds fixed string to the beginning of text. Useful for adding prefixes or
    special markers.

    @param prepend String to add at beginning

    {[
      let normalizer = Normalizers.prepend ~prepend:">> " in
      let result = Normalizers.normalize_str normalizer "Hello" in
      (* result = ">> Hello" *)
    ]} *)

val byte_level : ?add_prefix_space:bool -> ?use_regex:bool -> unit -> t
(** [byte_level ~add_prefix_space ~use_regex ()] applies byte-level
    normalization.

    Converts text to byte representation using special Unicode characters. Used
    by GPT-2 style models for robust handling of any byte sequence.

    @param add_prefix_space
      Add space prefix to text if it doesn't start with whitespace. Default:
      false.
    @param use_regex Use regex-based processing. Default: false.

    {[
      let normalizer = Normalizers.byte_level ~add_prefix_space:true () in
      let result = Normalizers.normalize_str normalizer "Hello" in
      (* Converts to byte representation, adds prefix space *)
    ]} *)

val sequence : t list -> t
(** [sequence normalizers] combines multiple normalizers into a sequence.

    Applies normalizers left-to-right. Each normalizer processes the output of
    the previous one. Useful for building complex normalization pipelines.

    @param normalizers List of normalizers to apply in order

    {[
      let normalizer = Normalizers.sequence [
        Normalizers.nfd ();
        Normalizers.strip_accents ();
        Normalizers.lowercase ();
      ] in
      let result = Normalizers.normalize_str normalizer "Café" in
      (* Applies: NFD decomposition → accent removal → lowercase *)
      (* result = "cafe" *)
    ]} *)

(** {1 Operations} *)

val normalize : t -> string -> normalized_string
(** [normalize t text] applies normalization to a string, preserving alignment
    information. *)

val normalize_str : t -> string -> string
(** [normalize_str t text] applies normalization to a string, returning only the
    normalized text. *)

(** {1 Serialization} *)

val to_json : t -> Jsont.json
(** [to_json t] converts normalizer to JSON representation. *)

val of_json : Jsont.json -> t
(** [of_json json] creates normalizer from JSON representation. *)
