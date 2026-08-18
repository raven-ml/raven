(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Decoding tokens back to text.

    Decoders convert token strings back into natural text by reversing
    encoding-specific transformations (prefix/suffix removal, byte-level
    decoding, whitespace normalization, etc.).

    Decoders operate on token {e strings}, not IDs. Convert IDs to strings via
    vocabulary first, then apply {!decode}.

    A decoder rewrites a token list into another token list, and {!decode} is
    the concatenation of the result. Most decoders rewrite each token on its own
    ({!bpe}, {!wordpiece}, {!metaspace}, {!replace}, {!strip}); {!byte_fallback}
    and {!ctc} also join or drop tokens; {!byte_level} and {!fuse} collapse the
    whole list into one token. The distinction matters when composing decoders
    with {!sequence}: a collapsing decoder hides token boundaries from the
    decoders that follow it. *)

type t
(** The type for decoders. *)

(** {1:constructors Constructors} *)

val bpe : ?suffix:string -> unit -> t
(** [bpe ~suffix ()] is a decoder for BPE-encoded tokens. Every occurrence of
    [suffix], which marks the end of a word, becomes the space that follows it,
    except in the last token where it is dropped. [suffix] defaults to ["</w>"].
*)

val byte_level : unit -> t
(** [byte_level ()] is a collapsing decoder that reads GPT-2 style
    byte-to-Unicode tokens back as the text their bytes spell.

    A token is mapped character by character, and one character outside the
    byte-level alphabet leaves the whole token to stand for its own bytes. The
    bytes of every token are then read as one text, so a character spelled
    across two tokens comes back whole, and every maximal ill-formed byte
    sequence becomes one U+FFFD — four stray bytes cost four, while a four-byte
    character cut short costs one. *)

val byte_fallback : unit -> t
(** [byte_fallback ()] is a decoder for byte fallback tokens. Each run of hex
    byte tokens (e.g. ["<0x41>"]) becomes the text those bytes spell; a run that
    is not valid UTF-8 becomes one U+FFFD per byte. Other tokens pass through
    unchanged. *)

val wordpiece : ?prefix:string -> ?cleanup:bool -> unit -> t
(** [wordpiece ~prefix ~cleanup ()] is a decoder for WordPiece tokens. Strips
    continuation [prefix] (default ["##"]) from non-initial subwords and gives
    the others a leading space. When [cleanup] is [true] (default), applies the
    detokenization cleanup to every token, once its joining space is prepended:
    the space before [.], [?], [!], [,] and the English contractions is taken
    back, and [" do not"] is rewritten to [" don't"]. So ["hello"; ","; "world"]
    decodes to ["hello, world"], while ["3"; "."; "14"] decodes to ["3. 14"]
    because the space after the full stop was never the decoder's to remove. *)

val metaspace :
  ?replacement:string ->
  ?prepend_scheme:Pre_tokenizer.prepend_scheme ->
  unit ->
  t
(** [metaspace ~replacement ~prepend_scheme ()] converts metaspace markers back
    to spaces. [replacement] defaults to ["\u{2581}"]. Unless [prepend_scheme]
    is [`Never], the marker was prepended to the text rather than standing for a
    space, so every occurrence of it in the {e first} token is dropped instead
    of becoming a space — on [`First] as on [`Always], the two prepending to the
    same first token. [prepend_scheme] defaults to [`Always]. *)

val ctc :
  ?pad_token:string ->
  ?word_delimiter_token:string ->
  ?cleanup:bool ->
  unit ->
  t
(** [ctc ~pad_token ~word_delimiter_token ~cleanup ()] is a decoder for
    {{:https://distill.pub/2017/ctc/}CTC (Connectionist Temporal
     Classification)} output. Deduplicates consecutive tokens, then cuts every
    occurrence of [pad_token] (default ["<pad>"]) out of each token, wherever in
    it they fall, and drops the tokens left empty. When [cleanup] is [true]
    (default), applies the same detokenization cleanup as {!wordpiece} to every
    token and then replaces [word_delimiter_token] (default ["|"]) with spaces.
*)

val sequence : t list -> t
(** [sequence decoders] chains [decoders] left-to-right. Each decoder's output
    token list feeds into the next. *)

val replace : pattern:string -> by:string -> unit -> t
(** [replace ~pattern ~by ()] replaces every literal occurrence of [pattern]
    with [by] in each token. *)

val strip : ?content:string -> ?start:int -> ?stop:int -> unit -> t
(** [strip ~content ~start ~stop ()] removes up to [start] leading and [stop]
    trailing occurrences of [content] from each token. [content] defaults to
    [" "], [start] and [stop] to [0]. A token left with nothing between the two
    cuts becomes empty.

    HuggingFace reads [content] as a single character, so {!to_json} on a
    decoder whose [content] is longer, or empty, writes a decoder it rejects. *)

val fuse : unit -> t
(** [fuse ()] is a collapsing decoder that concatenates all tokens into a single
    string with no delimiter. *)

(** {1:ops Operations} *)

val decode : t -> string list -> string
(** [decode decoder tokens] applies [decoder] to [tokens] and returns the
    decoded text. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf decoder] formats [decoder] for debugging. *)

(** {1:serialization Serialization} *)

val to_json : t -> Jsont.json
(** [to_json decoder] serializes [decoder] to HuggingFace JSON format. *)

val of_json : Jsont.json -> (t, string) result
(** [of_json json] is a decoder from HuggingFace JSON format. Errors if [json]
    is not an object, has a missing or unknown ["type"] field, or has invalid
    parameters. *)
