(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values placed in the buffers of runtime devices. *)

val device : Nx_device.t -> Nx_effect.device
(** [device d] is the device that holds placed values in [d]'s buffers, as
    {!Nx.Device.of_runtime} specifies. *)
