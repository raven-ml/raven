(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {!Nx.exp} compiled on a set's first device, in ulps of the correctly
    rounded result.

    The reference is libm's [exp] at [float64] rounded once to the operand's
    dtype. It is the correctly rounded result wherever the [float64] value lies
    further than [2{^-16}] of an ulp of the dtype from a midpoint between two of
    its codes, which holds for any libm within [2{^13}] ulps at [float64].
    Nearer a midpoint, both of its codes count as the reference. Distances are
    in ranks of the dtype's codes, as rune.mli's Numerics count them, [-0.] and
    [+0.] one rank apart. *)

val gen : int32 array Windtrap.Gen.t
(** [gen] generates [float32] codes: the zeros, infinities, NaNs, least
    subnormals and greatest finite floats of both signs, uniform bit patterns,
    values spread over the range where the result is finite and non-zero, and
    values near the arguments where it overflows, becomes subnormal and
    underflows to zero. *)

val law :
  (module Nx.Devices) -> ulps:int -> flushes:bool -> int32 array -> unit
(** [law s ~ulps ~flushes xs] checks that [Nx.exp] of the [float32] values of
    the codes [xs], compiled on [s]'s first device with no call evaluated, is
    within [ulps] of the reference. Where the operand is a zero, an infinity or
    a NaN, the result is Annex F's [1.], [+inf], [+0.] or a NaN. With
    [flushes], a result below the least normal [float32] is not checked, as
    rune.mli exempts a device that flushes subnormals. It covers each zero and
    infinity as an operand, a result that rounds to [+inf], one of [1.] from a
    non-zero operand and, without [flushes], a subnormal result and one that
    rounds to [+0.]. *)

val law16 :
  (module Nx.Devices) -> flushes:bool -> (float, 's) Nx.dtype -> unit
(** [law16 s ~flushes dt] checks that [Nx.exp] of every code of [dt],
    [float16] or [bfloat16], compiled on [s]'s first device with no call
    evaluated, is within one ulp of the reference, as rune.mli states for a
    float narrower than [float32]; Annex F's values and NaNs are as in {!law}.
    With [flushes], a result below the least normal [float32], as [bfloat16]'s
    subnormals are, is not checked.

    Raises [Invalid_argument] unless [dt] is [float16] or [bfloat16]. *)
