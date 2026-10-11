(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Holds a {!Rig_remote_proxy.Link}'s transfer or abort at a point of its
    protocol, or a sender whose send failed once it released the link
    ([Cut]). Each observation waits at most five seconds, and so does the
    hold. [resume] releases it, including after a failed assertion. Only one
    scenario may run at once. *)

type point = Sending | Sent | Arrived | Aborting | Cut

val arm : point -> unit

val held : unit -> bool
(** [held ()] observes the held transfer, or [false] on timeout. *)

val releases : unit -> int * int
(** [releases ()] observes two release entrants, or times out. Its result is
    their number and the number waiting for an active transfer. *)

val resume : unit -> unit
