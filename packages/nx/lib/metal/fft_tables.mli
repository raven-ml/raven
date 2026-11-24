(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The tables of the Fourier transforms: roots of unity, Bluestein's chirp and
    kernel, and Rader's kernel and logarithms.

    A table is a run of complex numbers, each two [float32], real part first,
    little-endian, 8 bytes a number. [w_l = e{^-2πi/l}] is the forward
    transform's root of order [l]. Every table is the forward transform's: an
    inverse transform conjugates its points around the forward one's arithmetic.

    Each number is computed in double-double arithmetic, a value [hi + lo] of
    two [float]s, from IEEE additions, fused multiply-adds and divisions alone,
    so its bits are the same on every host: no libm function enters a table. A
    root is the sine and cosine of its angle by their series, the angle taken
    within an eighth of a turn by exact integer steps; the chirp's numbers are
    products of two roots, and a kernel is an FFT of such numbers. A root is
    within [2{^-98}] of its exact value, a chirp's number within [2{^-96}], and
    a kernel's [m] numbers within [(1 + log₂ m) 2{^-95} ‖B‖₂] of theirs in norm,
    [‖B‖₂] the norm of the exact ones, at most [2{^-89} ‖B‖₂] for every [m]
    admitted. Each part is then rounded once to [float32]: the nearest [float32]
    to [hi], or where [hi] lies halfway between two, the one on [lo]'s side, the
    even one where [lo] is zero. A part that rounds below [2{^-126}], the least
    normal [float32], rounds to zero, as the GPU reads it, and every zero is
    stored as [+0]. So a root's or a chirp's stored part is the nearest
    [float32] to its exact value, unless that value lies within its bound of a
    halfway point. A kernel's bound is on its error's norm, so each entry errs
    by at most it: a part [x] of an entry, its real or its imaginary one, with
    [|x| >= 2{^-64} ‖B‖₂] is the nearest [float32] to its exact value unless
    that value lies within [2{^-89} ‖B‖₂] of a halfway point; a smaller part
    lies within [2{^-89} ‖B‖₂ + 2{^-24} |x|] of its exact value, and may be ulps
    from its nearest.

    A table's bytes lie outside the OCaml heap. The chirp takes [32 n] bytes of
    the heap while it is made, and a kernel [32 m] and a bit a number for the
    FFT of [m] points that makes it, in place, in time that grows as [m log m].
*)

type bigbytes = Plan.bigbytes
(** The type for a table's bytes. *)

(** {1:roots Roots of unity} *)

val roots : int -> start:int -> step:int -> count:int -> bigbytes
(** [roots l ~start ~step ~count] is the [count] roots [w_l{^start + i step}],
    [i < count].

    Raises [Invalid_argument] if [l] is not positive or [4 l] is past [2{^53}],
    or if [start], [step] or [count] is negative. *)

val low_bits : int
(** [low_bits] is 10: a long line's root [w_l{^k}] is the product
    [lo.(k mod 2{^10}) hi.(k / 2{^10})] of two tables ({!global}). *)

val global : int -> bigbytes * bigbytes
(** [global l] is [(lo, hi)], the [2{^low_bits}] roots [w_l{^k}] and the
    [⌈l / 2{^low_bits}⌉] roots [w_l{^2{^low_bits} k}].

    Raises [Invalid_argument] as {!roots} does. *)

(** {1:convolutions Convolutions} *)

val chirp : int -> bigbytes
(** [chirp n] is Bluestein's chirp of [n] points,
    [c_j = e{^-πi j² / n} = w_2n{^j² mod 2n}], [j < n].

    Raises [Invalid_argument] if [n] is not positive or [8 n] is past [2{^53}].
*)

val bluestein : n:int -> m:int -> bigbytes
(** [bluestein ~n ~m] is Bluestein's kernel for [n] points convolved at [m]: the
    [m] numbers [B_k = (1 / m) Σ_j b_j w_m{^jk}], [b_j = b_(m-j) = conj c_j] for
    [j < n], {!chirp}'s, and [b_j = 0] between.

    Raises [Invalid_argument] if [n] is not positive, if [m] is below [2 n - 1]
    or has a prime factor past 7, or if [4 m] is past [2{^53}]. *)

val rader : int -> bigbytes * bigbytes
(** [rader n] is [(kernel, logs)], Rader's tables for a prime [n] whose
    [m = n - 1] has no prime factor past 7, [g] the least primitive root of [n]:
    the [m] numbers [B_k = (1 / m) Σ_s w_n{^g{^-s} mod n} w_m{^sk}], and the [n]
    [int32] logarithms [logs.(g{^s} mod n) = s], [s < m], little-endian, entry 0
    zero.

    Raises [Invalid_argument] if [n] is not such a prime or is past [2{^31}]. *)

(**/**)

type dd = { hi : float; lo : float }
(** The type for double-doubles, the value [hi + lo]. *)

type cdd = { re : dd; im : dd }
(** The type for complex double-doubles. *)

val root : int -> int -> cdd
(** [root l k] is [w_l{^k}] before rounding. *)

val bluestein_dd : n:int -> m:int -> cdd array
(** [bluestein_dd ~n ~m] is {!bluestein}'s forward kernel before rounding. *)

val rader_dd : int -> cdd array
(** [rader_dd n] is {!rader}'s forward kernel before rounding. *)

val round : dd -> float
(** [round x] is [x.hi +. x.lo] rounded once to [float32], as the tables round
    each part, a zero of its sign below [2{^-126}]; a table then stores each
    zero as [+0]. *)

(**/**)
