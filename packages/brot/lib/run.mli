(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pretoken runs.

    {b Internal module.} What a document's token ids were built from: the spans
    the model encoded, and how a byte range of the text they index maps back to
    the text the caller passed in. {!Encoding} derives its tokens, offsets and
    word ids from a run the first time they are read.

    Four coordinates meet here. A span indexes a frame's [text], which a
    pre-tokenizer may have rewritten from the normalized text; that in turn
    comes from a stretch of the input. {!place}, {!frame.rewrite},
    {!frame.align} and {!frame.base} carry a span across the three steps. *)

type deferred = {
  normalizer : Normalizer.t;
  source : string;
  mutable alignment : Normalizer.alignment option;
      (** The alignment of [Normalizer.apply_aligned normalizer source] on
          [source], once something has asked for it. *)
}
(** The type for a normalization whose alignment has not been worked out yet.
    Working out where every byte came from costs about as much again as
    normalizing, and only {!offsets} needs it. The frames of one stretch share a
    value, so the stretch is normalized once however many frames it is cut into.
*)

(** The type for a map back to the text a normalization was run on. *)
type alignment = Known of Normalizer.alignment | Deferred of deferred

(** The type for where a frame's spans fall in the normalized text. *)
type place =
  | Shifted of int
      (** Spans are byte ranges of the frame's text, which starts at that offset
          in the normalized text. *)
  | Fixed of { start : int; stop : int }
      (** Every span stands for one and the same range of the normalized text: a
          pre-tokenizer that hands back text of its own rather than ranges can
          place its pieces, but not the tokens inside them. *)

type frame = {
  text : string;  (** The text the spans index. *)
  literal : bool;
      (** Whether each span of the frame is a single token whose string is the
          text of the span. *)
  place : place;
  rewrite : alignment option;
      (** How the frame's text maps back to the normalized text, when a
          pre-tokenizer rewrote it rather than cutting it. Only consulted for
          {!constructor-Shifted}. *)
  align : alignment;
      (** How the normalized text maps back to the input from [base]. *)
  base : int;  (** Where the normalized text starts in the input. *)
}
(** The type for the stretches of a document a run of spans was walked over. A
    document is one frame, plus one for each added token found in it. *)

type t = {
  frames : frame array;
  frame_stop : int array;
      (** Per frame, one past the index of its last span. *)
  span_start : int array;  (** Per span, its first byte in its frame's text. *)
  span_stop : int array;  (** Per span, one past its last byte. *)
  marks : int array;  (** Per span, one past the index of its last id. *)
  token_table : string array;  (** An id to its token string. *)
  len_table : int array;
      (** An id to the number of source bytes an occurrence of it accounts for,
          or [0] when that is a property of the text rather than of the id. *)
}
(** The type for pretoken runs. *)

val tokens : t -> ids:int array -> string array
(** [tokens t ~ids] is the token string of each id. A span of a literal frame
    reads its own text instead: an added token matched with [lstrip] or [rstrip]
    stands for the white space it took in too, which its identifier alone does
    not describe. *)

val offsets : t -> ids:int array -> (int * int) array
(** [offsets t ~ids] is the byte span of each id in the input.

    The tokens of a span tile it in the order they were emitted, each covering
    the bytes its identifier accounts for. An identifier whose {!len_table}
    entry is [0] stands for whatever the others do not describe, so the last
    such token of a span takes the bytes up to where the ones after it begin;
    several share what is left, a character each and the rest to the last. *)

val words : t -> ids:int array -> int option array
(** [words t ~ids] is the index of the span each id came from, counting from
    [0]. *)
