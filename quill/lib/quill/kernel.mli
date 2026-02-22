(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Code execution kernels.

    A kernel executes code and produces outputs. The interface is abstract to
    support different backends: in-process toplevels, subprocess-based kernels,
    remote kernels, etc. *)

(** {1:status Kernel status} *)

type status =
  | Starting
  | Idle
  | Busy
  | Shutting_down  (** The type for kernel lifecycle status. *)

(** {1:events Kernel events} *)

type event =
  | Output of { cell_id : Cell.id; output : Cell.output }
  | Finished of { cell_id : Cell.id; success : bool }
  | Status_changed of status
      (** The type for kernel events.

          - [Output] is emitted for each piece of output during execution.
          - [Finished] signals that execution of a cell has completed.
          - [Status_changed] signals a kernel lifecycle change. *)

(** {1:kernel Kernel interface} *)

type t = {
  execute : cell_id:Cell.id -> code:string -> unit;
  interrupt : unit -> unit;
  complete : code:string -> pos:int -> string list;
  status : unit -> status;
  shutdown : unit -> unit;
}
(** The type for kernel handles.

    - [execute ~cell_id ~code] submits code for execution. Results are delivered
      as {!event} values through the callback registered at kernel creation
      time.
    - [interrupt ()] requests interruption of the current execution.
    - [complete ~code ~pos] returns completion candidates at the given cursor
      position in [code].
    - [status ()] returns the current kernel status.
    - [shutdown ()] initiates graceful shutdown. *)
