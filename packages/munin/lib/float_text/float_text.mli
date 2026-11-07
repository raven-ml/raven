(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Decimal digits of floats, the same on every system.

    {b Internal module.} A finite float's value is an exact decimal. These
    functions compute with that decimal in integers. Each float operation they
    perform is exact: [frexp], [ldexp], [fmod], and a subtraction and a
    division whose results are integers below 2{^ 53}. So the digits are the
    same on every system and under js_of_ocaml. Each returns a pair
    [(d, e)] of the digits [d] of a decimal, without leading or trailing zeros,
    and the exponent [e] of its last digit, so that the decimal is
    [d] × 10{^ [e]}. [("15", -1)] is 1.5 and [("12", 3)] is 12000. Zero is
    [("0", 0)].

    They read [|x|]. The caller writes the sign, NaN and the infinities as its
    format wants, and takes the sign from [Float.sign_bit x] rather than from
    [x < 0.], so that [-0.] keeps its sign. [decimals 2 (-0.001)] is
    [("0", 0)], which C's [%.2f] writes [-0.00].

    Every function raises [Invalid_argument] if [x] is NaN or infinite. *)

type format = { precision : int; emin : int }
(** The type for binary floating-point formats. [precision] is the number of
    significand bits, the leading one included, from 1 to 53, and [emin] the
    exponent of the least normal value, 2{^ [emin]}. The exponent is unbounded
    above: the reals that read as the largest finite value are those nearer to
    it than to the next value the format would have. Where that value's
    significand is odd, as in every IEEE 754 format, this is IEEE's overflow
    threshold.

    Binary16 is [{ precision = 11; emin = -14 }], bfloat16 [{ 8; -126 }],
    float8 e5m2 [{ 3; -14 }], e4m3fn [{ 4; -6 }] and float4 e2m1fn
    [{ 2; 0 }]. *)

val binary64 : format
(** [binary64] is [{ precision = 53; emin = -1022 }]. *)

val binary32 : format
(** [binary32] is [{ precision = 24; emin = -126 }]. *)

val shortest : format -> float -> string * int
(** [shortest f x] is the decimal with the fewest digits that reads as [x] in
    [f], reading to nearest, ties to even; of those, the nearest to [x]; and at
    a tie between two, the one whose last digit is even.
    [shortest binary64 0.1] is [("1", -1)], and so is [shortest binary32 x] for
    the float32 nearest 0.1, held in a double.

    Raises [Invalid_argument] if [f]'s precision is outside 1 to 53, or if [x]
    is not a value of [f]. *)

val significant : int -> float -> string * int
(** [significant n x] is [|x|] rounded to [n] significant digits, to nearest,
    ties to even. [significant 3 135.5625] is [("136", 0)] and
    [significant 2 0.125] is [("12", -2)].

    Raises [Invalid_argument] if [n < 1]. *)

val decimals : int -> float -> string * int
(** [decimals n x] is [|x|] rounded to a multiple of 10{^ -[n]}, to nearest,
    ties to even. [decimals 2 0.125] is [("12", -2)] and [decimals 0 0.5] is
    [("0", 0)]. [n] may be negative: [decimals (-2) 1250.] is [("12", 2)]. *)
