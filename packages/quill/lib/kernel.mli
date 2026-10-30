(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Code execution kernels.

    A kernel runs the code cells of one {e language}, the first word of the info
    string of the cells it runs, such as ["ocaml"]. It reports what a run
    produces as {!event} values, through the callback given to {!make}. Any
    backend that can run code (an in-process toplevel, a subprocess, a remote
    machine) becomes a kernel through {!make}.

    A frontend holds one kernel and gives it cells, never code alone, so the
    kernel checks each cell's language. A cell in another language is
    {e refused}. The backend never sees it, and the kernel reports an [Error]
    output and a failed [Finished] for it, as for a cell that raised. *)

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

(** {1:kernel Kernels} *)

type t
(** The type for kernels. *)

val make :
  language:string ->
  on_event:(event -> unit) ->
  execute:(cell_id:Cell.id -> code:string -> unit) ->
  interrupt:(unit -> unit) ->
  complete:(code:string -> pos:int -> completion_item list) ->
  ?type_at:(code:string -> pos:int -> type_info option) ->
  ?diagnostics:(code:string -> diagnostic list) ->
  ?is_complete:(string -> bool) ->
  status:(unit -> status) ->
  shutdown:(unit -> unit) ->
  unit ->
  t
(** [make ~language ~on_event ~execute ~interrupt ~complete ~status ~shutdown
     ()] is the kernel that runs the cells of [language] with these functions of
    its backend:
    - [execute ~cell_id ~code] runs [code] for the cell [cell_id] and reports
      its outputs, then its [Finished], through [on_event].
    - [interrupt ()] requests that the running code stop.
    - [complete ~code ~pos] is the completions at byte [pos] of [code].
    - [type_at ~code ~pos] is the type at byte [pos] of [code]. It defaults to
      [None].
    - [diagnostics ~code] is the parse and type errors of [code]. It defaults to
      [[]].
    - [is_complete code] is [true] iff [code] holds a complete phrase, ready to
      run. It defaults to [true].
    - [status ()] is the backend's status.
    - [shutdown ()] starts a graceful shutdown.

    [on_event] must be the callback through which [execute] reports, so that a
    refusal ({!val-execute}) reaches the frontend the same way as a run.

    Raises [Invalid_argument] if [language] is not a language
    ({!Cell.is_language}). *)

val language : t -> string
(** [language k] is the language [k] runs, such as ["ocaml"]. *)

val execute : t -> Cell.t -> unit
(** [execute k c] runs the code cell [c] on [k]'s backend if [c]'s language is
    [language k]. Otherwise [k] refuses [c] and reports
    [Output {cell_id; output = Error msg}], where [msg] names both languages,
    then [Finished {cell_id; success = false}].

    Raises [Invalid_argument] if [c] is a text cell. *)

(** {1:intellisense_queries Intellisense}

    A text cell, or a code cell in another language than [language k], has no
    completions, type or diagnostics, and is complete. *)

val complete : t -> Cell.t -> pos:int -> completion_item list
(** [complete k c ~pos] is the completions at byte [pos] of [c]'s source. *)

val type_at : t -> Cell.t -> pos:int -> type_info option
(** [type_at k c ~pos] is the type at byte [pos] of [c]'s source, if any. *)

val diagnostics : t -> Cell.t -> diagnostic list
(** [diagnostics k c] is the parse and type errors of [c]'s source. *)

val is_complete : t -> Cell.t -> bool
(** [is_complete k c] is [true] iff [c]'s source holds a complete phrase, ready
    to run. *)

(** {1:control Control} *)

val interrupt : t -> unit
(** [interrupt k] requests that the code [k] runs stop. *)

val status : t -> status
(** [status k] is [k]'s status. *)

val shutdown : t -> unit
(** [shutdown k] starts [k]'s graceful shutdown. *)
