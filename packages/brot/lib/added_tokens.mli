(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Added tokens.

    Added tokens are vocabulary entries matched atomically in the input, ahead
    of normalization and pre-tokenization. They carry their own identifier,
    which may lie outside the model vocabulary. *)

type token = {
  content : string;  (** The token text. *)
  id : int;  (** The identifier emitted for [content]. *)
  special : bool;  (** Whether decoding may skip the token. *)
  single_word : bool;  (** Whether the token only matches whole words. *)
  lstrip : bool;  (** Whether the match extends over preceding white space. *)
  rstrip : bool;  (** Whether the match extends over following white space. *)
  normalized : bool;
      (** Whether the token is matched against normalized text rather than
          against the raw input. *)
}
(** The type for added tokens. *)

type t
(** The type for added token tables. *)

val make : normalize:(string -> string) -> token list -> t
(** [make ~normalize tokens] is the table of [tokens]. [normalize] is the
    tokenizer's normalizer; it is applied to the content of the tokens whose
    [normalized] is [true], since those are matched against normalized text.
    Tokens whose content is empty, or normalizes to empty, are dropped. Two
    tokens with the same content are one: it keeps the identifier of the first
    and the flags of the last. *)

val is_empty : t -> bool
(** [is_empty t] is [true] iff [t] has no added token. *)

val tokens : t -> token list
(** [tokens t] is the added tokens of [t], in the order given to {!make}. *)

val token_to_id : t -> string -> int option
(** [token_to_id t content] is the identifier of the added token [content]. The
    content of a token matched against normalized text is its unnormalized form.
*)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the content of the added token with identifier [id].
*)

val is_special : t -> int -> bool
(** [is_special t id] is [true] iff [id] is that of an added token whose
    [special] is [true]. *)

val find_raw : t -> string -> pos:int -> (int * int * int) option
(** [find_raw t text ~pos] is [Some (start, stop, id)] for the first added token
    of [t] matched against raw text at or after [pos] in [text], and [None] if
    there is none.

    At a given position the longest content wins, and earlier positions win over
    later ones. [start] and [stop] delimit the match in [text], extended over
    the neighbouring white space of a token with [lstrip] or [rstrip], but never
    before [pos]. A [single_word] token whose neighbouring character is a word
    character is discarded and the search resumes past it. [text] is a whole
    sentence: [pos] bounds the search, not the neighbourhood a [single_word]
    token is tested against. *)

val find_normalized : t -> string -> pos:int -> (int * int * int) option
(** [find_normalized t text ~pos] is {!find_raw} for the added tokens of [t]
    matched against normalized text, [text] being the normalized form of a
    sentence. *)
