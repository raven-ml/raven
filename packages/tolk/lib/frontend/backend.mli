(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Selection of frontend execution devices. *)

val device : unit -> Tolk.Device.t
(** [device ()] is the execution device selected by the current [DEV] context,
    or the first usable backend when [DEV] is unset. Devices are opened once
    through the shared runtime registry. *)

val device_name : unit -> string
(** [device_name ()] is the name of {!device} [()]. *)
