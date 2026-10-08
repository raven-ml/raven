(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The words allocated after rig_nv_abi initialises. *)

val after : float
(** [after] is {!Rig_nv_abi_before.allocated} when this module initialises,
    after rig_nv_abi. *)
