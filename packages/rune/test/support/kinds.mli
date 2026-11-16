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
val log : kind
val log2 : kind
val log1p : kind

(** {1:cases Cases} *)

type case
(** The type for a case: elements of [float32] and of [float64], each one value
    per operand. *)

val gen : kind -> case Windtrap.Gen.t
(** [gen k] generates cases of [k]'s operands. *)

val example : kind -> case
(** [example k] is a case of every switch point of [k] and its neighbours, both
    signs, and every special value. *)

(** {1:laws Laws} *)

val law : (module Nx.Devices) -> f64:bool -> kind -> case -> unit
(** [law s ~f64 k c] checks that [k] of [c]'s elements compiled on [s]'s first
    device, with no call evaluated, gives the bits of [k] computed eagerly on
    the host, a NaN equal to any NaN. The [float64] elements are checked only
    with [f64]. It covers each class of operand [k]'s switches tell apart and
    each special value. *)

val narrow : (module Nx.Devices) -> kind -> unit
(** [narrow s k] checks the law of {!law} for [k] at [float16], [bfloat16],
    [float8_e4m3fn], [float8_e5m2] and [float4_e2m1fn]: at every code of each,
    or for two operands at every pair of codes of the 8- and 4-bit floats and at
    a grid of pairs of the 16-bit ones. *)

val folded : (module Nx.Devices) -> f64:bool -> kind -> unit
(** [folded s ~f64 k] checks that [k] of a constant inside a compiled function,
    which the compiler may compute while it compiles, gives eager's bits at each
    switch point of [k], its neighbours and the special values, as {!law}
    states, at [float32] and, with [f64], [float64]. *)
