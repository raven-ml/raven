(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Packed bitmaps.

    A bitmap is a sequence of bits packed eight to a byte in a [uint8] tensor,
    from a bit offset: bit [i] is bit [(offset + i) mod 8] of byte
    [(offset + i) / 8], the least significant bit first. This is Arrow's
    validity layout, so a bitmap reads and writes Arrow's validity buffers
    without a copy. The bits of the bytes outside the range are unspecified:
    operations may set them, and {!count} and {!to_bool} ignore them.

    The library [nx.bits] is built from nx's operations over the bytes, so a
    bitmap is placed, traced and compiled as its bytes are.

    Conditions take booleans: [Nx.where (Nx_bits.to_bool b) x y]. *)

type t
(** The type for bitmaps. *)

(** {1:make Bitmaps} *)

val v : ?offset:int -> length:int -> Nx.uint8_t -> t
(** [v ~offset ~length bytes] is the [length] bits of [bytes] from bit [offset]
    on. [offset] defaults to [0].

    Raises [Invalid_argument] if [bytes] is not 1-D, if [offset] or [length] is
    negative, or if the bits reach past the bytes. *)

val of_bool : Nx.bool_t -> t
(** [of_bool m] is the bitmap of the 1-D mask [m]: bit [i] is set iff [m.{i}].
    Its offset is [0].

    Raises [Invalid_argument] if [m] is not 1-D. *)

(** {1:observe Observing} *)

val length : t -> int
(** [length b] is the number of bits of [b]. *)

val bytes : t -> Nx.uint8_t * int
(** [bytes b] is [(bytes, offset)]: the bytes that hold [b]'s bits and the
    position of its first bit, in \[[0], [7]\]. [bytes] has exactly the
    [(offset + length b + 7) / 8] bytes the bits reach, and
    [v ~offset ~length:(length b) bytes] is [b]. *)

val to_bool : t -> Nx.bool_t
(** [to_bool b] is the mask of [b]: [(to_bool b).{i}] is [true] iff bit [i] is
    set. Its shape is [[|length b|]]. *)

val count : t -> Nx.int64_t
(** [count b] is the number of bits set in [b], a scalar. *)

(** {1:logic Logic}

    Operands at one offset combine byte by byte; operands at different offsets
    combine through {!to_bool}, and the result has offset [0]. *)

val logand : t -> t -> t
(** [logand a b] has bit [i] set iff it is set in [a] and in [b].

    Raises [Invalid_argument] if [a] and [b] have different lengths. *)

val logor : t -> t -> t
(** [logor a b] has bit [i] set iff it is set in [a] or in [b].

    Raises [Invalid_argument] if [a] and [b] have different lengths. *)

val lognot : t -> t
(** [lognot b] has bit [i] set iff it is not set in [b]. *)

(** {1:select Selecting} *)

val sub : t -> offset:int -> length:int -> t
(** [sub b ~offset ~length] is bits [offset] to [offset + length - 1] of [b],
    over [b]'s bytes, in O(1).

    Raises [Invalid_argument] if [offset] or [length] is negative, or if
    [offset + length > length b]. *)

val take : indices:Nx.int64_t -> t -> t
(** [take ~indices b] is the bitmap whose bit [j] is bit [indices.{j}] of [b].
    An index outside \[[0], [length b]) reads an unset bit. Its offset is [0].

    Raises [Invalid_argument] if [indices] is not 1-D. *)

val concat : t list -> t
(** [concat bs] is the bits of [bs] one after the other. A single bitmap is
    returned as it is; otherwise the result has offset [0].

    Raises [Invalid_argument] if [bs] is empty. *)
