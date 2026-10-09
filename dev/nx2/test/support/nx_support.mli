(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices for nx's suites: memory devices, opened once per process. *)

val memory : int -> Rig.t
(** [memory k] is the memory device [k], [0 <= k < 4], named ["m<k>"]: its
    memory is the host's and its work runs in this process
    ({!Rig.memory_device}). *)
