(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fourier transforms as tolk graphs.

    Each function takes and gives nodes as {!Lower_arith} holds elements, of
    [float32] or [float64] parts, and computes what {!Nx_kernel.S.c2c},
    {!Nx_kernel.S.r2c} and {!Nx_kernel.S.c2r} state, along each of [axes],
    strictly increasing and not empty.

    {1:algorithm Algorithm}

    A line of [n] points splits by its least prime factor [p] into [p]-point
    transforms, a twiddle and transforms of [n / p] points, recursively
    (Cooley and Tukey, 1965). A [2]-point transform is a sum and a
    difference. Another prime's up to [250] is a sum of [p] products with a
    [p × p] table, one kernel of [O(p)] work per point. A prime's above [250]
    is Bluestein's convolution (1970): the points times a chirp, a cyclic
    convolution of [m] points, [m] the least power of two at least [2p - 1],
    by three transforms of [m] points, then the chirp again: [O(log p)] work
    per point, in [O(log m)] kernels. A transform of [n] points is
    thus [O(n log n)] work. A real transform computes the complex transform
    of its points, twice a real one's work.

    {1:bounds Error bound}

    Barring overflow and underflow, and to first order in the unit roundoff
    [u] of the parts, a transform [ŷ] of [N] points, the product of the
    points along each of [axes], is within [‖ŷ - y‖₂ <= ε u √N ‖x‖₂] of the
    exact [y], [‖x‖₂] its operand's norm, [√2] times it for {!c2r}, whose
    operand's unread imaginary parts count as [0] along one axis. [ε] is the
    sum over [axes]
    of [e(n)], [n] the points along the axis: [e(n) = Σ c(p) + 15 (r - 1)]
    over [n]'s [r] prime factors [p], with multiplicity, [c(2) = 1],
    [c(p) = (2p + 15) √p] for [2 < p <= 250] and, above,

    {[
      c(p) = (2 e(m) + 18) m / √p + (e(m) + 12) √m + 15
    ]}

    [m] Bluestein's points, so that [e(m) = 16 log₂ m - 15]:

    - A twiddle [w] is within [12 u] of [e^(±2πi k/n)], exact at whole
      quarter turns: its parts are the sines {!Lower_real} states within 4
      ulps of arguments of at most [π/2], which carry at most [4 u] of
      relative error.
    - A twiddle's product errs by at most [√2 γ(2) + 12 u <= 15 u] relative.
    - A [p]-point transform of the points [z] scales their norm by [√p]
      exactly, and is within [c(p) u] of its result's norm. A [2]-point
      transform rounds each part once, [u] relative to its result. One by
      the table is within [(√2 (p + 2) u + 12 u) Σ|z|] of each result.
    - Bluestein's: the chirp's product, a twiddle's, errs by [15 u ‖z‖₂] in
      norm, and the forward transform, of norm [√m ‖z‖₂], adds [e(m) u] of
      it. The kernel [K], the transform of the [2p - 1] chirp values over
      [m], is at most [(2p - 1) / m <= 1] in magnitude, and within
      [(e(m) + 12) u √(2p - 1) / √m] of it in norm, which moves the product
      by [(e(m) + 12) u √p ‖z‖₂], a transform's entries being at most
      [√p ‖z‖₂]; the product rounds within [3 u]. The inverse transform
      multiplies these errors by [√m] and adds [e(m) u m ‖z‖₂], and the
      last chirp [15 u √p ‖z‖₂], against a result of norm [√p ‖z‖₂].

    A NaN or infinite point makes NaNs where it enters, of no promised sign
    or payload. *)

open Tolk

val c2c : Nx_kernel.direction -> axes:int list -> Ops.t -> Ops.t
(** [c2c d ~axes x] is the transform [d] of the complex [x] along each of
    [axes].

    Raises [Invalid_argument] for parts other than [float32] and [float64]. *)

val r2c : axes:int list -> Ops.t -> Ops.t
(** [r2c ~axes x] is the [Forward] transform of the real [x], its bins [0] to
    [n/2] along the last of [axes], [n] [x]'s extent there; one bin of [+0]
    for [n = 0].

    Raises [Invalid_argument] for parts other than [float32] and [float64]. *)

val c2r : axes:int list -> n:int -> Ops.t -> Ops.t
(** [c2r ~axes ~n x] is the [Inverse] transform of the complex [x] along each of
    [axes] but the last, then along the last the real parts of the [Inverse]
    transform of the [n] bins [X'] completed from those [n/2 + 1] bins [X]:
    [X'[k]] is [X[k]] for [k <= n/2], the imaginary parts of bin [0] and,
    for an even [n], of bin [n/2] taken as [0], and [conj X[n - k]] above.

    Raises [Invalid_argument] for parts other than [float32] and [float64]. *)
