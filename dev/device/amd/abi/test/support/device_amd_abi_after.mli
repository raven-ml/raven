(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The words allocated after device_amd_abi initialises. *)

val after : float
(** [after] is {!Device_amd_abi_before.allocated} when this module initialises,
    after device_amd_abi. *)
