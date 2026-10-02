(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Typora-style cursor-aware markdown block segmentation.

    Parses a markdown source string into top-level blocks with byte ranges.
    Consumers use this to render inactive blocks formatted and the active block
    (containing the cursor) as raw text for editing. *)

(** {1:types Types} *)

type span = {
  first : int;  (** Inclusive start byte offset in source (zero-based). *)
  last : int;  (** Inclusive end byte offset in source (zero-based). *)
}
(** A byte range within the source string. *)

type block_kind =
  | Paragraph
  | Heading of int
  | Block_quote
  | List
  | Thematic_break
  | Table
  | Blank  (** The kind of a top-level markdown block. *)

type block = { span : span; kind : block_kind }
(** A top-level block extracted from a markdown source string. *)

type t
(** A parsed markdown source split into blocks with byte ranges. *)

(** {1:parse Parsing} *)

val parse : string -> t
(** [parse source] parses [source] into blocks with byte ranges. *)

val source : t -> string
(** [source t] is the original source string. *)

val blocks : t -> block list
(** [blocks t] is the list of top-level blocks in document order. *)

(** {1:query Queries} *)

val active_block : t -> cursor:int -> block option
(** [active_block t ~cursor] is the block containing byte offset [cursor], or
    [None] if [cursor] is outside all blocks. *)

val block_source : t -> block -> string
(** [block_source t block] extracts the raw source substring for [block]. *)

(** {1:render Rendering} *)

val block_to_html : t -> block -> string
(** [block_to_html t block] renders [block] to an HTML fragment. *)

val to_html : string -> string
(** [to_html source] renders CommonMark [source] to an HTML fragment. *)
