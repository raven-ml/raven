(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The float kinds on [float32] and [float64] nodes, as nx's kernels compute
    them.

    Each function computes its kind by the algorithm nx's kinds write for its
    compute type ([nx_kinds.h] and [nx_kinds_real.h] of {!Nx_kernel}): the same
    operations in the same order, with a fused multiply-add where they write one
    and nowhere else, so that a node gives the bits nx's kernels give on a
    target that rounds as those headers assume, NaN operands included. *)

open Tolk

val unary : Nx_kernel.Prog.unary -> Ops.t -> Ops.t
(** [unary k x] is [k] of each element of the [float32] or [float64] node [x].
    Raises [Invalid_argument] for a kind floats do not take. *)

val binary : Nx_kernel.Prog.binary -> Ops.t -> Ops.t -> Ops.t
(** [binary k x y] is [k] of the elements of [x] and [y], two nodes of one float
    type. Raises [Invalid_argument] for a kind floats do not take. *)

val compare : Nx_kernel.Prog.compare -> Ops.t -> Ops.t -> Ops.t
(** [compare k x y] is the boolean comparison [k] of the elements of [x] and
    [y]. *)

val fma : Ops.t -> Ops.t -> Ops.t -> Ops.t
(** [fma a b c] is [a * b + c] rounded once. *)

(** {1:parts Parts of the complex kinds}

    The pieces of the float kinds that the complex kinds ([nx_kinds_complex.h])
    compute with. A constant is a pair: its value at [float32], then at
    [float64]. *)

val pick : Ops.t -> 'a * 'a -> 'a
(** [pick x p] is the member of [p] for [x]'s type. *)

val con : Ops.t -> float * float -> Ops.t
(** [con x p] is the constant {!pick}[ x p] of [x]'s type and shape. *)

val ln2_hi : float * float
(** [ln2_hi] is ln 2 rounded short, so that its products with an exponent are
    exact. *)

val ln2_lo : float * float
(** [ln2_lo] is ln 2 less {!ln2_hi}. *)

val log2e : float * float
(** [log2e] is 1 / ln 2 rounded. *)

val log2e_lo : float * float
(** [log2e_lo] is 1 / ln 2 less {!log2e}. *)

val pio2_hi : float * float
(** [pio2_hi] is π/2 rounded. *)

val pio2_lo : float * float
(** [pio2_lo] is π/2 less {!pio2_hi}. *)

val exp2_lo : float * float
(** [exp2_lo] is where [exp2] clamps its operand from below. *)

val shifter : float * float
(** [shifter] is the float whose addition rounds a float below half of it to the
    nearest integer. *)

val sub_scale : float * float
(** [sub_scale] is the power of two that scales a subnormal to a normal. *)

val mbits : int * int
(** [mbits] is the number of fraction bits. *)

val bias : int * int
(** [bias] is the exponent's bias. *)

val min_normal_bits : int64 * int64
(** [min_normal_bits] is the bits of the least normal. *)

val sqrt_half_bits : int64 * int64
(** [sqrt_half_bits] is the bits of √½ rounded. *)

val copysign : Ops.t -> Ops.t -> Ops.t
(** [copysign m s] is [m] with [s]'s sign bit. *)

val scale : Ops.t -> Ops.t -> Ops.t
(** [scale y k] is [y 2{^k}] for the [int32] node [k], rounded once where
    [y 2{^k/2}] is normal. *)

val two_sum : Ops.t -> Ops.t -> Ops.t * Ops.t
(** [two_sum a b] is [(hi, lo)], [hi] the rounded sum and [a + b = hi + lo]
    exactly. *)

val nearest : Ops.t -> Ops.t * Ops.t
(** [nearest t] is, from [t = v + ]{!shifter}, the integer nearest [v] as a
    float and as an [int32] node. *)

val exp_reduce : Ops.t -> Ops.t * Ops.t * Ops.t
(** [exp_reduce x] is [(r, rc, k)] with [x = k ln2 + r], [|r| <= ln2/2], [rc]
    [r]'s rounding error and [k] an [int32] node. *)

val exp_q : Ops.t -> Ops.t
(** [exp_q r] is [Q(r)] of [expm1 r = r + r{^2} Q(r)]. *)

val exp2_poly : Ops.t -> Ops.t
(** [exp2_poly r] is [2{^r}] for [|r| <= 1/2]. *)

(** The table a reduction by π/2 reads. *)
type table =
  | Two_over_pi  (** 2/π, for [x] itself. *)
  | Two_ln2_over_pi  (** 2 ln 2 / π, for [x ln 2]. *)

val trig_big : Ops.t -> Ops.t
(** [trig_big x] is whether [|x|] is finite and at or past the switch from Cody
    and Waite's reduction to Payne and Hanek's. *)

val rem_big : table -> Ops.t -> Ops.t * Ops.t * Ops.t
(** [rem_big t x] is [(hi, lo, q)] with [x k = q π/2 + hi + lo], [k] the number
    [t] is of [x], [q] an [int32] node: Payne and Hanek's reduction, for [x]
    where {!trig_big} holds of [x k]. *)

val sin_at : Ops.t -> Ops.t * Ops.t * Ops.t -> Ops.t
(** [sin_at x r] is [sin x] from [x]'s reduction [r]. *)

val cos_at : Ops.t -> Ops.t * Ops.t * Ops.t -> Ops.t
(** [cos_at x r] is [cos x] from [x]'s reduction [r]. *)
