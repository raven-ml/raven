(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The firmware images each driver-less driver boots with: its
    [gen/headers/firmware.tsv], from which [gen/gen.py] also makes the driver's
    pins. *)

val amd : string
(** [amd] is rig.amd.pci's list. *)

val nv : string
(** [nv] is rig.nv.pci's list. *)
