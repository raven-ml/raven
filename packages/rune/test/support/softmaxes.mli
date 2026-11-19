(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {!Nx.softmax} compiled on a set's first device, against its definition
    as {!Nx_softmax_support} computes it. *)

type case
(** The type for a case: a dtype, lanes along the last axis and a scale. *)

val gen : case Windtrap.Gen.t
(** [gen] generates cases at [float64], [float32], [float16] and [bfloat16]:
    lanes of one element to a few thousand, of values within a few units to
    past [exp]'s range, about [0.] or far from it; [-inf] elements, or a NaN
    or a [+inf] in every other lane; scales of [1.] and others. *)

val law :
  (module Nx.Devices) -> f64:bool -> chain:bool -> exp_ulps:int -> case -> unit
(** [law s ~f64 ~chain ~exp_ulps c] checks that [Nx.softmax] of [c],
    compiled on [s]'s first device, is within rune.mli's row for it of the
    reference, as {!Nx_softmax_support.check} checks it, [E] being
    [exp_ulps], [β] being [β(n)], or the larger of [β(n)] and [n - 1] with
    [chain], for a device where a sum may keep one running sum, and a
    narrow dtype's rounding adding its unit roundoff. An element whose
    [scale x] is [-inf] in a lane of finite greatest is [+0.], and a lane
    holding a NaN or of an infinite greatest is NaN. Without [f64] it
    discards a [float64] case, for a device whose kernels lack [float64].
    It covers lanes of one element, lanes far from [0.], [-inf] elements,
    NaN lanes and {!Nx_softmax_support.covers}' cases. *)
