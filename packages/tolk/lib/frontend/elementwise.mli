(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Element-wise operations.

    Binary operations broadcast their operands to a common shape and promote
    them to a common dtype before applying. Broadcasting follows the standard
    rule: shapes are aligned from the last axis, and an axis of size [1] is
    stretched to match. Promotion picks the least dtype that both operands cast
    into without loss where possible. Use {!Tensor.i}, {!Tensor.f}, and
    {!Tensor.b} to lift scalar operands.

    Broadcasting and promotion themselves are {!Tensor.broadcasted}, whose
    implementation lives in {!Op}. *)

val contiguous : Tensor.t -> Tensor.t
(** [contiguous t] forces [t] to be materialised into a contiguous buffer,
    breaking a chain of lazy views. It is a value-preserving layout hint. *)

(** {1 Arithmetic} *)

val add : Tensor.t -> Tensor.t -> Tensor.t
(** [add a b] is the element-wise sum. *)

val sub : Tensor.t -> Tensor.t -> Tensor.t
(** [sub a b] is the element-wise difference [a - b]. *)

val mul : Tensor.t -> Tensor.t -> Tensor.t
(** [mul a b] is the element-wise product. *)

val div : Tensor.t -> Tensor.t -> Tensor.t
(** [div a b] is element-wise true division. Integer operands are promoted to
    float, so the result is always floating point. *)

val floordiv : Tensor.t -> Tensor.t -> Tensor.t
(** [floordiv a b] rounds the quotient toward negative infinity. *)

val mod_ : Tensor.t -> Tensor.t -> Tensor.t
(** [mod_ a b] is the remainder with the sign of the divisor (floor
    modulo), matching {!floordiv}. *)

val neg : Tensor.t -> Tensor.t
(** [neg t] negates [t]; on a boolean tensor it is logical negation. *)

(** {1 Comparisons}

    Each returns a boolean tensor. *)

val lt : Tensor.t -> Tensor.t -> Tensor.t
(** [lt a b] is [a < b]. *)

val gt : Tensor.t -> Tensor.t -> Tensor.t
(** [gt a b] is [a > b]. *)

val le : Tensor.t -> Tensor.t -> Tensor.t
(** [le a b] is [a <= b]. *)

val ge : Tensor.t -> Tensor.t -> Tensor.t
(** [ge a b] is [a >= b]. *)

val ne : Tensor.t -> Tensor.t -> Tensor.t
(** [ne a b] is [a <> b]. *)

val eq : Tensor.t -> Tensor.t -> Tensor.t
(** [eq a b] is [a = b]. *)

val logical_not : Tensor.t -> Tensor.t
(** [logical_not t] casts [t] to boolean and negates it. *)

(** {1 Bitwise}

    Operands must be integer or boolean. *)

val bitwise_and : Tensor.t -> Tensor.t -> Tensor.t
val bitwise_or : Tensor.t -> Tensor.t -> Tensor.t
val bitwise_xor : Tensor.t -> Tensor.t -> Tensor.t
val bitwise_not : Tensor.t -> Tensor.t

val threefry : Tensor.t -> Tensor.t -> Tensor.t
(** [threefry x key] mixes the 64-bit counters [x] with the 64-bit [key]
    using the Threefry-2x32 block cipher, producing uniformly distributed
    bits. Both operands must be [uint64] tensors of the same shape. This is
    the primitive underlying the {!Rand} generators. *)

(** {1 Selection} *)

val where : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
(** [where cond x y] selects from [x] where [cond] is true and from [y]
    elsewhere. [cond] is cast to boolean and all three are broadcast together. *)

val maximum : Tensor.t -> Tensor.t -> Tensor.t
(** [maximum a b] is the element-wise larger value. *)

val masked_fill : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
(** [masked_fill t mask value] replaces the elements of [t] where [mask] is
    true with [value], leaving the rest unchanged. *)

val minimum : Tensor.t -> Tensor.t -> Tensor.t
(** [minimum a b] is the element-wise smaller value. *)

val inverse : Tensor.t -> Tensor.t
(** [inverse t] is the additive inverse [neg t] for floating-point [t] and the
    bitwise complement for integer [t]. It is the reflection used to turn a
    maximum into a minimum. *)

val usum : Tensor.t -> Tensor.t list -> Tensor.t
(** [usum t ts] folds [t] and [ts] together, adding them when [t] is numeric
    and taking their logical or when [t] is boolean. *)

val uprod : Tensor.t -> Tensor.t list -> Tensor.t
(** [uprod t ts] folds [t] and [ts] together, multiplying them when [t] is
    numeric and taking their logical and when [t] is boolean. *)

(** {1 Unary math}

    Transcendental functions promote integer inputs to float. *)

val reciprocal : Tensor.t -> Tensor.t
val sqrt : Tensor.t -> Tensor.t
val rsqrt : Tensor.t -> Tensor.t
val sin : Tensor.t -> Tensor.t
val cos : Tensor.t -> Tensor.t
val exp : Tensor.t -> Tensor.t
val exp2 : Tensor.t -> Tensor.t
val log : Tensor.t -> Tensor.t
val log2 : Tensor.t -> Tensor.t
val trunc : Tensor.t -> Tensor.t
val floor : Tensor.t -> Tensor.t
val ceil : Tensor.t -> Tensor.t
val square : Tensor.t -> Tensor.t
val abs : Tensor.t -> Tensor.t
val sign : Tensor.t -> Tensor.t

(** {1 Activations} *)

val relu : Tensor.t -> Tensor.t
val sigmoid : Tensor.t -> Tensor.t
val tanh : Tensor.t -> Tensor.t

(** {1 Powers, shifts, and truncating division} *)

val pow : Tensor.t -> Tensor.t -> Tensor.t
(** [pow a b] is [a] raised to the power [b], element-wise. When [a] is integer
    and [b] floating point the result is rounded and cast back to [a]'s dtype. *)

val cdiv : Tensor.t -> Tensor.t -> Tensor.t
(** [cdiv a b] divides [a] by [b] rounding the quotient toward zero (C-style),
    as opposed to {!floordiv} which rounds toward negative infinity. *)

val fmod : Tensor.t -> Tensor.t -> Tensor.t
(** [fmod a b] is the remainder of truncating division, taking the sign of the
    dividend [a], as opposed to {!mod_} which takes the sign of the divisor. *)

val lshift : Tensor.t -> Tensor.t -> Tensor.t
(** [lshift a b] shifts [a] left by [b] bits. Operands must be integer. *)

val rshift : Tensor.t -> Tensor.t -> Tensor.t
(** [rshift a b] shifts [a] right by [b] bits, arithmetically. Operands must be
    integer. *)

(** {1 Sign and interpolation} *)

val copysign : Tensor.t -> Tensor.t -> Tensor.t
(** [copysign a b] has the magnitude of [a] and the sign of [b], element-wise. *)

val logaddexp : Tensor.t -> Tensor.t -> Tensor.t
(** [logaddexp a b] is [log (exp a + exp b)], computed stably by factoring out
    the element-wise maximum. *)

val lerp : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
(** [lerp t end_ weight] linearly interpolates from [t] to [end_] by [weight],
    i.e. [t + (end_ - t) * weight]. *)

(** {1 Rounding and clamping} *)

val round : Tensor.t -> Tensor.t
(** [round t] rounds to the nearest integer, ties to even. *)

val clamp : ?min:Tensor.t -> ?max:Tensor.t -> Tensor.t -> Tensor.t
(** [clamp ~min ~max t] bounds [t] below by [min] and above by [max]. Either
    bound may be omitted.

    @raise Invalid_argument if neither bound is given. *)

val clip : ?min:Tensor.t -> ?max:Tensor.t -> Tensor.t -> Tensor.t
(** [clip] is an alias for {!clamp}. *)

(** {1 More unary math} *)

val log10 : Tensor.t -> Tensor.t
val tan : Tensor.t -> Tensor.t
val asin : Tensor.t -> Tensor.t
val acos : Tensor.t -> Tensor.t
val atan : Tensor.t -> Tensor.t
val sinh : Tensor.t -> Tensor.t
val cosh : Tensor.t -> Tensor.t
val atanh : Tensor.t -> Tensor.t
val asinh : Tensor.t -> Tensor.t
val acosh : Tensor.t -> Tensor.t

val erf : Tensor.t -> Tensor.t
(** [erf t] is the Gauss error function, approximated by a rational polynomial. *)

(** {1 Floating-point classification}

    Each returns a boolean tensor. *)

val isnan : Tensor.t -> Tensor.t
(** [isnan t] is true where [t] is NaN. *)

val isinf : ?detect_positive:bool -> ?detect_negative:bool -> Tensor.t -> Tensor.t
(** [isinf t] is true where [t] is an infinity. [detect_positive] and
    [detect_negative] (both default [true]) restrict which sign counts. *)

val isfinite : Tensor.t -> Tensor.t
(** [isfinite t] is true where [t] is neither infinite nor NaN. *)

val isclose : ?rtol:float -> ?atol:float -> ?equal_nan:bool -> Tensor.t -> Tensor.t -> Tensor.t
(** [isclose t other] is true where [t] and [other] agree within [atol] plus
    [rtol] times [|other|]. Matching infinities compare equal; NaNs compare
    equal only when [equal_nan] is set. *)

(** {1 More activations} *)

val relu6 : Tensor.t -> Tensor.t
val hardswish : Tensor.t -> Tensor.t
val hardsigmoid : ?alpha:float -> ?beta:float -> Tensor.t -> Tensor.t
val hardtanh : ?min_val:float -> ?max_val:float -> Tensor.t -> Tensor.t
val leaky_relu : ?neg_slope:float -> Tensor.t -> Tensor.t
val quick_gelu : Tensor.t -> Tensor.t

val gelu : Tensor.t -> Tensor.t
(** [gelu t] is the Gaussian error linear unit, tanh approximation. *)

val swish : Tensor.t -> Tensor.t
val silu : Tensor.t -> Tensor.t
(** [silu] is an alias for {!swish}. *)

val elu : ?alpha:float -> Tensor.t -> Tensor.t
val celu : ?alpha:float -> Tensor.t -> Tensor.t
val selu : ?alpha:float -> ?gamma:float -> Tensor.t -> Tensor.t
val softplus : ?beta:float -> Tensor.t -> Tensor.t
val mish : Tensor.t -> Tensor.t
val logsigmoid : Tensor.t -> Tensor.t
val softsign : Tensor.t -> Tensor.t
