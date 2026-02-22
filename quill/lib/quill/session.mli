(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Interactive notebook sessions.

    A session manages document state with undo/redo history and transient cell
    execution statuses. It is purely functional: all operations return a new
    session value.

    Sessions do not own a kernel. The caller is responsible for driving kernel
    execution and feeding results back via {!apply_output} and
    {!finish_execution}. *)

(** {1:status Cell status} *)

type cell_status =
  | Idle
  | Queued
  | Running  (** The type for transient cell execution status. *)

(** {1:sessions Sessions} *)

type t
(** The type for notebook sessions. *)

val create : ?history_capacity:int -> Doc.t -> t
(** [create ?history_capacity doc] creates a session from [doc].
    [history_capacity] defaults to [100]. *)

(** {1:accessors Accessors} *)

val doc : t -> Doc.t
(** [doc s] is the current document of session [s]. *)

val cell_status : Cell.id -> t -> cell_status
(** [cell_status id s] is the execution status of cell [id] in [s]. *)

val can_undo : t -> bool
(** [can_undo s] is [true] if an undo operation is available. *)

val can_redo : t -> bool
(** [can_redo s] is [true] if a redo operation is available. *)

(** {1:document Document operations}

    Structural operations ({!insert_cell}, {!remove_cell}, {!move_cell},
    {!set_cell_kind}) record undo history automatically. Source edits via
    {!update_source} do not -- call {!checkpoint} when the edit sequence is
    complete. *)

val update_source : Cell.id -> string -> t -> t
(** [update_source id source s] updates the source of cell [id]. Does not record
    undo history. Call {!checkpoint} when the edit sequence ends. *)

val checkpoint : t -> t
(** [checkpoint s] saves the current document to the undo history. Call this at
    natural boundaries: before execution, before save, on cell focus change.
    No-op if the document hasn't changed since the last checkpoint. *)

val insert_cell : pos:int -> Cell.t -> t -> t
(** [insert_cell ~pos cell s] inserts [cell] at position [pos]. *)

val remove_cell : Cell.id -> t -> t
(** [remove_cell id s] removes the cell with identifier [id]. *)

val move_cell : Cell.id -> pos:int -> t -> t
(** [move_cell id ~pos s] moves cell [id] to position [pos]. *)

val clear_outputs : Cell.id -> t -> t
(** [clear_outputs id s] clears the outputs of cell [id]. *)

val clear_all_outputs : t -> t
(** [clear_all_outputs s] clears outputs from all code cells. *)

val set_cell_kind : Cell.id -> [ `Code | `Text ] -> t -> t
(** [set_cell_kind id kind s] changes cell [id] to the given [kind]. *)

(** {1:execution Execution state}

    Update transient cell status. These do not touch the kernel -- the caller is
    responsible for driving kernel execution. *)

val mark_running : Cell.id -> t -> t
(** [mark_running id s] marks cell [id] as running. *)

val mark_queued : Cell.id -> t -> t
(** [mark_queued id s] marks cell [id] as queued. *)

val mark_idle : Cell.id -> t -> t
(** [mark_idle id s] marks cell [id] as idle. *)

val apply_output : Cell.id -> Cell.output -> t -> t
(** [apply_output id output s] appends [output] to cell [id] in the document.
    The output is visible immediately via {!doc}. *)

val finish_execution : Cell.id -> success:bool -> t -> t
(** [finish_execution id ~success s] marks cell [id] as idle and increments its
    execution count. *)

(** {1:history History} *)

val undo : t -> t
(** [undo s] restores the previous document state. Returns [s] unchanged if no
    undo is available. *)

val redo : t -> t
(** [redo s] restores the next document state. Returns [s] unchanged if no redo
    is available. *)

(** {1:reload Reload} *)

val reload : Doc.t -> t -> t
(** [reload doc s] replaces the document, clearing history and statuses. *)
