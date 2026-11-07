(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Exact reading of decimals into binary formats, for the suites that check
    printed digits. It computes on naturals in integer limbs, so it answers the
    same on every system, whose C [strtod] need not round correctly. A decimal
    is a pair [(d, e)] of digits [d] and the exponent [e] of the last one: it
    is [d] × 10{^ [e]}. *)

type format = { precision : int; emin : int }
(** The type for binary formats: [precision] significand bits, the leading one
    included, from 1 to 53, and [emin] the exponent of the least normal value.
    The exponent is unbounded above. *)

val compare : string * int -> float -> int
(** [compare (d, e) x] compares [d] × 10{^ [e]} with [|x|], for a finite
    [x]. *)

val reads : format -> float -> string * int -> bool
(** [reads f x de] is [true] iff the decimal [de] reads as [|x|] in [f], to
    nearest, ties to even: it lies strictly between the midpoints to [|x|]'s
    neighbours, or on one when [|x|]'s significand is even. Past the largest
    finite value the neighbour above is the next value [f] would have. [x] is
    a finite value of [f]. *)
