(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Host memory as windows, for the suites. *)

val window : int -> Device_pci.Window.t
(** [window n] is a window on [n] new zeroed bytes of host memory, kept alive
    for the process. *)
