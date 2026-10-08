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
    cannot fork. *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once the process holds the machine's GPU lock, where
    the machine has a GPU. A suite calls it before [Windtrap.run]. *)
