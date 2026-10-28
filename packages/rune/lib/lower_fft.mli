(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fourier transforms as tolk graphs.

    Each function takes and gives nodes as {!Lower_arith} holds elements, of
    [float32] or [float64] parts, and computes what {!Nx_kernel.S.c2c},
    {!Nx_kernel.S.r2c} and {!Nx_kernel.S.c2r} state along each of [axes], which
    must be strictly increasing and not empty.

    {b Cost.} A transform of a line of [n] points is [O(n log n)] work. It
    splits the line by [n]'s prime factors. A prime factor [p] up to [250] costs
    one kernel of [O(p)] work per point. A prime factor above [250] costs
    [O(log p)] work per point in all, in [O(log m)] kernels, where [m] is the
    least power of two at least [2p - 1]. A real transform computes the complex
    transform of its points, twice a real one's work.

    {b Error bound.} This bound implements the transform rows of
    {!Rune.section-numerics}. Barring overflow and underflow, and to first
    order in the unit roundoff [u] of the parts, a transform [ŷ] of [N] points,
    [N] the product of the extents along [axes], has [‖ŷ - y‖₂ <= ε u √N ‖x‖₂],
    where [y] is the exact transform. [‖x‖₂] is its operand's norm, or [√2]
    times it for {!c2r}, whose operand's unread imaginary parts count as [0]
    along one axis. [ε] is the sum over [axes] of [e(n)], where [n] is the
    number of points along the axis and [e(n) = Σ c(p) + 15 (r - 1)] over [n]'s
    [r] prime factors [p], with multiplicity. [c(2) = 1], [c(p) = (2p + 15) √p]
    for [2 < p <= 250] and, above [250],

    {[
      c(p) = (2 e(m) + 18) m / √p + (e(m) + 12) √m + 15
    ]}

    for [m] as above. As [m] is a power of two, [e(m) = 16 log₂ m - 15].

    A NaN or infinite point makes NaN results where it enters, with an
    unspecified sign and payload. *)

open Tolk

val c2c : Nx_kernel.direction -> axes:int list -> Ops.t -> Ops.t
(** [c2c d ~axes x] is the transform [d] of the complex [x] along each of
    [axes].

    Raises [Invalid_argument] for parts other than [float32] and [float64]. *)

val r2c : axes:int list -> Ops.t -> Ops.t
(** [r2c ~axes x] is the [Forward] transform of the real [x], keeping bins [0]
    to [n/2] along the last of [axes], where [n] is [x]'s extent there. For
    [n = 0] it is one bin of [+0].

    Raises [Invalid_argument] for parts other than [float32] and [float64]. *)

val c2r : axes:int list -> n:int -> Ops.t -> Ops.t
(** [c2r ~axes ~n x] is the [Inverse] transform of the complex [x] along each of
    [axes] but the last, then along the last the real parts of the [Inverse]
    transform of the [n] bins [X'] completed from those [n/2 + 1] bins [X].
    [X'[k]] is [X[k]] for [k <= n/2] and [conj X[n - k]] above, with the
    imaginary parts of bin [0], and of bin [n/2] for an even [n], taken as [0].

    Raises [Invalid_argument] for parts other than [float32] and [float64]. *)
