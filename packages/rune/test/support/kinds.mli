(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Float kinds compiled on a set's first device, against nx's eager kernels on
    the host, bit for bit.

    rune.mli's Numerics section states these kinds exact: a compiled result has
    the bits of eager's, a NaN equal to any NaN. Operands are drawn to break
    that: both sides of each switch nx's kinds take, the greatest finite floats,
    subnormals, [-0.], infinities, NaNs with payloads and uniform bit patterns.
*)

type kind
(** The type for a float kind of one or two operands. *)

val name : kind -> string
(** [name k] is [k]'s name, as nx's function. *)

val sin : kind
val cos : kind
val tan : kind
val exp : kind
val exp2 : kind
val expm1 : kind
val sinh : kind
val cosh : kind
val tanh : kind
val asin : kind
val acos : kind
val atan : kind
val atan2 : kind
val erf : kind
val pow : kind
val mod_ : kind
val log : kind
val log2 : kind
val log1p : kind

val clamped_exp : kind
(** [clamped_exp] is {!exp} of its operand clamped to [[-80, 80]] by two
    selections on comparisons inside the compiled program, a NaN taking [80].
    Its law covers the clamped operand. *)

val clamped_log : kind
(** [clamped_log] is {!log} of its operand clamped to [[1e-30, 1e4]] as
    {!clamped_exp} clamps, a NaN taking [1e4]. *)

(** {1:composed Composed functions}

    Functions nx composes from kinds and exact operations, at [float32] and
    [float64]: their operands are drawn as a kind's are, from the switches
    their formulas take and their domains' edges, and a function of several
    operands also draws tuples from its domain. *)

val asinh : kind
val sinc : kind
val acosh : kind
val atanh : kind
val rsqrt : kind
val hypot : kind
val erfc : kind
val erfinv : kind
val ndtr : kind
val log_ndtr : kind
val ndtri : kind
val lgamma : kind
val digamma : kind
val lbeta : kind
val gammainc : kind
val gammaincc : kind
val log_gammainc : kind
val log_gammaincc : kind
val gammaincinv : kind
val gammainccinv : kind
val betainc : kind
val betaincc : kind
val log_betainc : kind
val log_betaincc : kind
val betaincinv : kind
val betainccinv : kind
val i0e : kind
val i1e : kind

(** {1:complex Complex kinds}

    Kinds of [complex64] and [complex128] operands, named as C99's functions.
    Their parts are drawn as a float kind's operands are, and also as Annex G's
    special values (ISO/IEC 9899, G.6) and as the kind's own: both sides of each
    switch [nx_kinds_complex.h] takes, the branch cuts, and parts that overflow
    or underflow the kind's naive formula. *)

val cmul : kind
val cdiv : kind
val crecip : kind
val csqrt : kind
val csign : kind
val clog : kind
val clog2 : kind
val cexp : kind
val cexp2 : kind
val cpow : kind
val csinh : kind
val ccosh : kind
val ctanh : kind
val csin : kind
val ccos : kind
val ctan : kind
val casin : kind
val cacos : kind
val catan : kind
val casinh : kind
val cacosh : kind
val catanh : kind

(** {1:cases Cases} *)

type case
(** The type for a case: elements of [float32] and of [float64], each one value
    per operand, or for a complex kind one per part of each operand, the real
    part first: [complex64] and [complex128] elements. *)

val gen : kind -> case Windtrap.Gen.t
(** [gen k] generates cases of [k]'s operands. *)

val example : kind -> case
(** [example k] is a case of [k]'s fixed elements (every window of 2/π that a
    trigonometric kind's large reduction reads, Annex F's pairs for a kind of
    two operands, every number of Annex G's parts for a complex kind), every
    switch point of [k] and its neighbours, both signs, every special value and
    a Weyl sequence over the codes. *)

(** {1:laws Laws} *)

val law :
  ?f32:bool -> (module Nx.Devices) -> f64:bool -> kind -> case -> unit
(** [law s ~f64 k c] checks that [k] of [c]'s elements compiled on [s]'s first
    device, with no call evaluated, gives the bits of [k] computed eagerly on
    the host, a NaN equal to any NaN, part by part for a complex kind. The
    [float32] elements are checked only with [f32] (default [true]), the
    [float64] ones only with [f64]. It covers each class of operand [k]'s
    switches tell apart and each special value. *)

val narrow : (module Nx.Devices) -> kind -> unit
(** [narrow s k] checks the law of {!law} for [k] at [float16], [bfloat16],
    [float8_e4m3fn], [float8_e5m2] and [float4_e2m1fn]: at every code of each,
    or for two operands at every pair of codes of the 8- and 4-bit floats and at
    a grid of pairs of the 16-bit ones.

    Raises [Invalid_argument] for a complex kind or one of three operands. *)

val folded : (module Nx.Devices) -> f64:bool -> kind -> unit
(** [folded s ~f64 k] checks that [k] of a constant inside a compiled function,
    which the compiler may compute while it compiles, gives eager's bits at each
    switch point of [k], its neighbours and the special values, as {!law}
    states, at [float32] and, with [f64], [float64]; for a complex kind, at
    tuples of its parts' constants and Annex G's numbers (for two operands, each
    number of zeros, ones, infinities and a NaN beside two others), at
    [complex64] and, with [f64], [complex128]. *)
