(** Decoding tokens back to text.

    Decoders reverse the tokenization process, converting token strings back
    into natural text. They handle removing special markers (prefixes,
    suffixes), reversing byte-level encoding, normalizing whitespace, and other
    post-processing needed to reconstruct readable text.

    Decoders operate on token strings (not IDs). The typical flow is: 1. Convert
    IDs to token strings via vocabulary 2. Apply decoder to token string list 3.
    Result is final decoded text

    Multiple decoders can be chained with {!sequence} to compose
    transformations. *)

type t
(** Decoder that transforms token strings into natural text.

    Decoders are composable and can be chained. *)

(** {1 Decoder Types} *)

val bpe : ?suffix:string -> unit -> t
(** [bpe ?suffix ()] creates BPE decoder.

    Removes end-of-word suffixes added during tokenization.

    @param suffix Suffix to strip from tokens (default: empty string). *)

val byte_level : unit -> t
(** [byte_level ()] creates decoder for byte-level tokenization.

    Reverses byte-to-Unicode encoding used by GPT-2 style tokenizers. Converts
    special byte representations back to original characters. *)

val byte_fallback : unit -> t
(** [byte_fallback ()] creates decoder for byte fallback encoding.

    Converts byte tokens (e.g., "<0x41>") back to characters. *)

val wordpiece : ?prefix:string -> ?cleanup:bool -> unit -> t
(** [wordpiece ?prefix ?cleanup ()] creates WordPiece decoder.

    Removes continuing subword prefixes and merges tokens into words.

    @param prefix Prefix to remove from non-initial subwords (default: "##").
    @param cleanup Normalize whitespace and remove artifacts (default: true). *)

val metaspace : ?replacement:char -> ?add_prefix_space:bool -> unit -> t
(** [metaspace ?replacement ?add_prefix_space ()] creates metaspace decoder.

    Converts metaspace markers back to regular spaces.

    @param replacement
      Metaspace character used during tokenization (default: '▁').
    @param add_prefix_space
      Whether tokenizer added prefix space (affects decoding) (default: true).
*)

val ctc :
  ?pad_token:string ->
  ?word_delimiter_token:string ->
  ?cleanup:bool ->
  unit ->
  t
(** [ctc ?pad_token ?word_delimiter_token ?cleanup ()] creates CTC decoder for
    speech recognition models.

    Removes CTC blank tokens and formats word boundaries.

    @param pad_token Padding token to remove (default: "<pad>").
    @param word_delimiter_token Word boundary marker (default: "|").
    @param cleanup Remove extra whitespace and artifacts (default: true). *)

val sequence : t list -> t
(** [sequence decoders] chains multiple decoders.

    Applies decoders left-to-right. Output of each decoder feeds into next.
    Useful for combining transformations (e.g., byte-level + wordpiece +
    whitespace cleanup). *)

val replace : pattern:string -> content:string -> unit -> t
(** [replace ~pattern ~content ()] creates pattern replacement decoder.

    Replaces all occurrences of [pattern] with [content] in decoded text. Uses
    literal string matching (not regex).

    @param pattern String to find.
    @param content Replacement string. *)

val strip : ?left:bool -> ?right:bool -> ?content:char -> unit -> t
(** [strip ?left ?right ?content ()] creates whitespace stripping decoder.

    Removes specified characters from text edges.

    @param left Strip from start of text (default: false).
    @param right Strip from end of text (default: false).
    @param content Character to strip (default: space ' '). *)

val fuse : unit -> t
(** [fuse ()] creates decoder that merges all tokens without delimiters.

    Concatenates token strings with no spaces. Useful when tokens already
    contain appropriate spacing. *)

(** {1 Operations} *)

val decode : t -> string list -> string
(** [decode decoder tokens] converts token strings to text.

    Applies decoder transformations to reconstruct natural text from token list.

    @param decoder Decoder to apply.
    @param tokens List of token strings (not IDs).
    @return Decoded text. *)

(** {1 Serialization} *)

val to_json : t -> Yojson.Basic.t
(** [to_json decoder] serializes decoder to HuggingFace JSON format. *)

val of_json : Yojson.Basic.t -> t
(** [of_json json] deserializes decoder from HuggingFace JSON format.

    @raise Yojson.Json_error if JSON is malformed. *)
