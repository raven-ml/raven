(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reading the scalar values of UTF-8 bytes, and looking a scalar up in a table
    of ranges.

    Every function reads a [buf], the bytes of a column, and trusts them to be
    valid UTF-8, as every text column is. *)

type buf = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** Bytes of text. *)

val starts_scalar : buf -> int -> bool
(** [starts_scalar v j] is [true] iff byte [j] of [v] starts a scalar value. *)

val width : buf -> int -> int -> int
(** [width v j stop] is the length of the UTF-8 sequence at byte [j] of [v] that
    ends by [stop], or 1 where none does. *)

val scalar : buf -> int -> int -> int
(** [scalar v j w] is the scalar value of the [w] bytes at [j], [w] > 1. *)

val decode : buf -> int -> int -> int
(** [decode v j stop] is the scalar value of the sequence at byte [j] of [v],
    which ends by [stop]. *)

val decode_before : buf -> int -> int -> int
(** [decode_before v first j] is the scalar value that ends just before byte
    [j] of [v], [j] past [first]. *)

val blit : buf -> int -> buf -> int -> int -> unit
(** [blit src lo dst at len] copies the bytes [lo, lo + len) of [src] to [dst]
    from [at]. *)

val within : int array -> int -> int -> int -> bool
(** [within bounds u lo hi] is [true] iff [u] lies in the ranges that the
    ascending [bounds] start and end by turns, searched in [\[lo, hi)]. *)
