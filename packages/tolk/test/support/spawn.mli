(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A process that cannot start another. *)

val forbid : unit -> unit
(** [forbid ()] forbids this process to start any other, for the rest of its
    life: on Windows by a job that holds one active process, this one;
    elsewhere by a limit of one process for its user, which already runs more.
    A suite calls it in a process of its own.

    Raises [Failure] if the system refuses the limit. *)
