(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Casts between [float32] and the 16-bit floats, compiled on a set's first
    device, against eager's on the host. *)

val law : (module Nx.Devices) -> (float, 's) Nx.dtype -> unit
(** [law s dt] checks, for [dt] [float16] or [bfloat16], that a call compiled
    on [s]'s first device casts as eager does on the host, with no call
    evaluated:
    - every code of [dt] cast to [float32];
    - every finite code's [float32] value, the [float32] halfway to the next
      code, as if the exponent were unbounded, and that midpoint's two
      [float32] neighbours, of both signs, cast to [dt];
    - every code cast to [float32] and back, and to the other 16-bit format.

    A result is eager's bits where eager's is not a NaN. Where it is, the
    result is a NaN too, quiet where its operand is quiet, as nx's NaN rule
    states ({!Nx.Dtype.section-stores}); quiet and signalling [float32] NaNs of
    both signs are cast as well.

    Raises [Invalid_argument] unless [dt] is [float16] or [bfloat16]. *)
