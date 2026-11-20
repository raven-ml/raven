(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Codecs' decisions as {!Spread} checks read them. *)

val digit_edges : float array
(** [digit_edges] is the edges an image token's numbers are rounded between:
    [-2.5], [-1.5], ..., [2.5]. *)

val digits : int array -> int array
(** [digits k] is each number of each image token of [k], in order, rounded, as
    [q + 3] for the integer [q] it is rounded to: the count of {!digit_edges}
    below it, of the widest of AION-1's levels, seven. *)
