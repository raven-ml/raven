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

(** {1:completions Completions} *)

type completion_kind =
  | Value
  | Type
  | Module
  | Module_type
  | Constructor
  | Label  (** The type for completion item kinds. *)

type completion_item = {
  label : string;
  kind : completion_kind;
  detail : string;
}
(** The type for completion items. [label] is the identifier name, [kind]
    classifies it, and [detail] is a formatted type signature. *)

(** {1:intellisense Intellisense} *)

type severity =
  | Error
  | Warning  (** The type for diagnostic severity levels. *)

type diagnostic = {
  from_pos : int;
  to_pos : int;
  severity : severity;
  message : string;
}
(** The type for diagnostics. Positions are byte offsets within the cell. *)

type type_info = {
  typ : string;
  doc : string option;
  from_pos : int;
  to_pos : int;
}
(** The type for type-at-position results. [typ] is the formatted type, [doc] is
    the optional documentation string, and positions delimit the expression
    span. *)

(** {1:kernel Kernel interface} *)

type t = {
  execute : cell_id:Cell.id -> code:string -> unit;
  interrupt : unit -> unit;
  complete : code:string -> pos:int -> completion_item list;
  type_at : (code:string -> pos:int -> type_info option) option;
  diagnostics : (code:string -> diagnostic list) option;
  is_complete : (string -> bool) option;
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
    - [type_at] when [Some f], [f ~code ~pos] returns type information at the
      given cursor position.
    - [diagnostics] when [Some f], [f ~code] returns parse and type errors.
    - [is_complete] when [Some f], [f code] returns [true] if [code] contains a
      complete toplevel phrase ready for execution.
    - [status ()] returns the current kernel status.
    - [shutdown ()] initiates graceful shutdown. *)
