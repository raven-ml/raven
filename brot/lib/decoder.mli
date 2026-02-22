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

    Some decoders transform each token independently ({e per-token}: {!bpe},
    {!metaspace}, {!replace}, {!strip}, {!byte_fallback}), while others collapse
    the entire token list into a single result ({e collapsing}: {!byte_level},
    {!wordpiece}, {!fuse}). This distinction matters when composing decoders
    with {!sequence}. *)

type t
(** The type for decoders. *)

(** {1:constructors Constructors} *)

val bpe : ?suffix:string -> unit -> t
(** [bpe ~suffix ()] is a per-token decoder for BPE-encoded tokens. Strips
    [suffix] from end-of-word tokens and inserts spaces between words. [suffix]
    defaults to [""]. *)

val byte_level : unit -> t
(** [byte_level ()] is a collapsing decoder that reverses GPT-2 style
    byte-to-Unicode encoding back to original bytes. *)

val byte_fallback : unit -> t
(** [byte_fallback ()] is a per-token decoder for byte fallback tokens. Converts
    hex byte tokens (e.g. ["<0x41>"]) back to their byte values, accumulating
    consecutive byte tokens into strings. Non-byte tokens pass through
    unchanged. *)

val wordpiece : ?prefix:string -> ?cleanup:bool -> unit -> t
(** [wordpiece ~prefix ~cleanup ()] is a collapsing decoder for WordPiece
    tokens. Strips continuation [prefix] (default ["##"]) from non-initial
    subwords and joins tokens into words. When [cleanup] is [true] (default),
    normalizes whitespace in the result. *)

val metaspace : ?replacement:char -> ?add_prefix_space:bool -> unit -> t
(** [metaspace ~replacement ~add_prefix_space ()] is a per-token decoder that
    converts metaspace markers back to regular spaces. [replacement] defaults to
    ['_']. When [add_prefix_space] is [true] (default), the leading replacement
    character on the first token is stripped. *)

val ctc :
  ?pad_token:string ->
  ?word_delimiter_token:string ->
  ?cleanup:bool ->
  unit ->
  t
(** [ctc ~pad_token ~word_delimiter_token ~cleanup ()] is a per-token decoder
    for
    {{:https://distill.pub/2017/ctc/}CTC (Connectionist Temporal
     Classification)} output. Deduplicates consecutive tokens, removes
    [pad_token] (default ["<pad>"]), and when [cleanup] is [true] (default),
    replaces [word_delimiter_token] (default ["|"]) with spaces. *)

val sequence : t list -> t
(** [sequence decoders] chains [decoders] left-to-right. Each decoder's output
    token list feeds into the next. *)

val replace : pattern:string -> by:string -> unit -> t
(** [replace ~pattern ~by ()] is a collapsing decoder that joins the token list,
    replaces all literal occurrences of [pattern] with [by] in the result, and
    returns a single-element list. *)

val strip : ?left:bool -> ?right:bool -> ?content:char -> unit -> t
(** [strip ~left ~right ~content ()] is a collapsing decoder that joins the
    token list and removes leading (when [left] is [true]) and/or trailing (when
    [right] is [true]) occurrences of [content] from the result. [left] and
    [right] default to [false]; [content] defaults to [' ']. *)

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
