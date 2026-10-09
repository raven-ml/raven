(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar kinds.

    A kind is what one step of a scalar program computes from one element of
    each operand. Each computes, per compute type, as the function of its name
    in [nx_kinds.h] ([Exp] as [nx_exp_f32] on float32), or as code that gives
    the same bits. That header states the compute types, the transcendental
    kinds' bounds in ulps, their special values and the NaNs each kind gives. *)

(** {1:kinds Kinds} *)

(** The type for kinds of one operand, of its dtype. Trigonometric kinds take
    radians. *)
type unary =
  | Neg
  | Recip  (** On integers, [x] for [1] and [-1], [0] otherwise. *)
  | Abs
  | Sign  (** [-1], [0] or [1]; NaN for a NaN. *)
  | Sqrt
  | Exp
  | Exp2  (** [2{^x}], exact at integers whose power is in the dtype. *)
  | Log
  | Log2  (** Exact at powers of two. *)
  | Log1p  (** [log (1 + x)], accurate near [0]. *)
  | Expm1  (** [exp x - 1], accurate near [0]. *)
  | Sin
  | Cos
  | Tan
  | Asin  (** In \[[-π/2], [π/2]\]. *)
  | Acos  (** In \[[0], [π]\]. *)
  | Atan  (** In \[[-π/2], [π/2]\]. *)
  | Sinh
  | Cosh
  | Tanh
  | Erf  (** [2/√π ∫₀ˣ e{^-t²} dt]. *)
  | Floor  (** Toward negative infinity; the identity on integers. *)
  | Ceil  (** Toward positive infinity; the identity on integers. *)
  | Round
      (** To the nearest integer, half away from zero; the identity on
          integers. *)
  | Trunc  (** Toward zero; the identity on integers. *)

(** The type for kinds of two operands of one dtype, of that dtype. Integers
    wrap. *)
type binary =
  | Add
  | Sub
  | Mul
  | Fdiv  (** The IEEE 754 quotient of floats. *)
  | Idiv
      (** The integer quotient truncated toward zero: [0] by zero, and a signed
          dtype's least value by [-1] is that value. *)
  | Mod
      (** The remainder, of the dividend's sign: the dividend by zero on
          integers, [fmod] on floats. *)
  | Pow  (** The first operand to the power of the second. *)
  | Atan2  (** The angle of [(y, x)], [y] the first operand, in \]-π, π\]. *)
  | Maximum
      (** The IEEE 754 maximum: NaN propagates and [-0] orders below [+0]. *)
  | Minimum  (** As [Maximum]. *)
  | And  (** Bitwise on integers, logical on booleans. *)
  | Or  (** As [And]. *)
  | Xor  (** As [And]. *)
  | Threefry
      (** Threefry-2x32 with 20 rounds of the counter, the first operand, under
          the key, the second: [uint64] words, the low half first. *)

(** The type for comparisons of two operands of one dtype, to booleans.
    Unsigned dtypes order unsigned. *)
type compare = Equal | Not_equal | Less | Less_equal

(** {1:arities Kinds by arity} *)

(** The type for kinds of no operand. *)
type op0 =
  | Fill of string
      (** [Fill b] is the element whose bits are [b], in the result's dtype and
          the host's byte order. *)
  | Iota of int
      (** [Iota i] is each element's index along the result's axis [i], in the
          result's dtype. *)

(** The type for kinds of one operand. *)
type op1 =
  | Copy  (** The operand's bits, NaN payloads included, into its own dtype. *)
  | Unary of unary
  | Cast
      (** The operand's element stored into the result's dtype. A float stores
          as {!Nx_array.Dtype.of_float} says. An integer stores as its exact
          value rounded once to a float format, modulo the width to an integer
          dtype, and as [x <> 0] to a boolean. A boolean stores as [0] or [1].
          Into a complex dtype, these rules give the real part and the imaginary
          part is [0.]. A complex number stores part by part into a complex
          dtype, as [true] into a boolean if either part is non-zero, and by its
          real part into any other dtype. A float that keeps its format, as a
          float32 into a complex64's real part, keeps its bits. Into the
          operand's own dtype it is [Copy]. *)
  | Bitcast  (** The operand's bits read in the result's dtype, of one width. *)

(** The type for kinds of two operands. *)
type op2 = Binary of binary | Compare of compare

(** The type for kinds of three operands. *)
type op3 =
  | Where
      (** The second operand's element where the first's, a boolean, is
          [true], and the third's elsewhere. *)
  | Fma
      (** The product of the first two plus the third, rounded once; wrapping
          on integers. *)
