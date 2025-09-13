(** Text normalization module matching HuggingFace tokenizers.

    Normalizers are responsible for cleaning and transforming text before
    tokenization. This includes operations like lowercasing, accent removal,
    Unicode normalization, and handling special characters. *)

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
(** Create a BERT normalizer.
    @param clean_text
      Remove control characters and normalize whitespace (default: true)
    @param handle_chinese_chars Add spaces around CJK characters (default: true)
    @param strip_accents
      Strip accents (None means auto-detect based on lowercase) (default: None)
    @param lowercase Convert to lowercase (default: true) *)

val strip : ?left:bool -> ?right:bool -> unit -> t
(** Create a strip normalizer.
    @param left Strip whitespace from left (default: false)
    @param right Strip whitespace from right (default: true) *)

val strip_accents : unit -> t
(** Create an accent stripping normalizer *)

val nfc : unit -> t
(** Unicode NFC (Canonical Decomposition, followed by Canonical Composition)
    normalizer *)

val nfd : unit -> t
(** Unicode NFD (Canonical Decomposition) normalizer *)

val nfkc : unit -> t
(** Unicode NFKC (Compatibility Decomposition, followed by Canonical
    Composition) normalizer *)

val nfkd : unit -> t
(** Unicode NFKD (Compatibility Decomposition) normalizer *)

val lowercase : unit -> t
(** Simple lowercase normalizer *)

val nmt : unit -> t
(** NMT normalizer - handles special spacing around punctuation *)

val precompiled : bytes -> t
(** Create a normalizer from precompiled data *)

val replace : pattern:string -> replacement:string -> unit -> t
(** Create a replace normalizer.
    @param pattern Regex pattern to match
    @param replacement Replacement string *)

val prepend : prepend:string -> t
(** Create a prepend normalizer.
    @param prepend String to prepend *)

val byte_level : ?add_prefix_space:bool -> ?use_regex:bool -> unit -> t
(** Create a byte-level normalizer.
    @param add_prefix_space Add space prefix to first word (default: false)
    @param use_regex Use regex for splitting (default: false) *)

val sequence : t list -> t
(** Combine multiple normalizers into a sequence *)

(** {1 Operations} *)

val normalize : t -> string -> normalized_string
(** Apply normalization to a string, preserving alignment information *)

val normalize_str : t -> string -> string
(** Apply normalization to a string, returning only the normalized text *)

(** {1 Serialization} *)

val to_json : t -> Yojson.Basic.t
(** Convert normalizer to JSON representation *)

val of_json : Yojson.Basic.t -> t
(** Create normalizer from JSON representation *)
