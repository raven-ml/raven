(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rig's public API against a reference written from [rig.mli], over every kind
    of device the machine has. *)

val commands : two:bool -> fork:bool -> Windtrap.command list
(** [commands ~two ~fork] is the API's calls and the reference's, for programs
    whose calls run on one domain, or with [two], on two at once. With [fork] a
    call forks a child that reads every buffer: a process that ran a domain
    cannot fork. The GPUs present are probed at the first draw of a device. *)

val leaves_nothing : fork:bool -> unit
(** [leaves_nothing ~fork] checks that programs making every kind of call on
    every kind of device, and with [fork] forking a child that reads their
    buffers, leave the process's C heap and open descriptors as they found them
    once warm: after eight programs uncounted, two counts eight programs apart.
    Each count follows four rounds of a full collection, a wait for every
    device's work and its cache given back. Skips where the system counts
    neither. *)
