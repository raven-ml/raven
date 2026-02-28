(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** WebSocket protocol for notebook communication.

    Defines the message types exchanged between the web frontend and the
    notebook server, with JSON serialization.

    - {{!client}Client messages}
    - {{!server}Server messages} *)

(** {1:client Client messages} *)

type client_msg =
  | Update_source of { cell_id : string; source : string }
      (** Update the source text of cell [cell_id]. *)
  | Checkpoint  (** Create an undo checkpoint of the current notebook state. *)
  | Execute_cell of { cell_id : string }  (** Execute cell [cell_id]. *)
  | Execute_cells of { cell_ids : string list }
      (** Execute the cells [cell_ids] in order. *)
  | Execute_all  (** Execute all code cells in document order. *)
  | Interrupt  (** Interrupt the currently running execution. *)
  | Insert_cell of { pos : int; kind : [ `Code | `Text ] }
      (** Insert a new cell of the given [kind] at position [pos]. *)
  | Delete_cell of { cell_id : string }  (** Delete cell [cell_id]. *)
  | Move_cell of { cell_id : string; pos : int }
      (** Move cell [cell_id] to position [pos]. *)
  | Set_cell_kind of { cell_id : string; kind : [ `Code | `Text ] }
      (** Change the kind of cell [cell_id] to [kind]. *)
  | Clear_outputs of { cell_id : string }
      (** Clear outputs of cell [cell_id]. *)
  | Clear_all_outputs  (** Clear outputs of all cells. *)
  | Save  (** Save the notebook to disk. *)
  | Undo  (** Undo the last checkpoint. *)
  | Redo  (** Redo the last undone checkpoint. *)
  | Complete of { request_id : string; code : string; pos : int }
      (** Request completions for [code] at cursor position [pos]. [request_id]
          correlates the response. *)

val client_msg_of_json : string -> (client_msg, string) result
(** [client_msg_of_json s] parses a JSON string into a client message. Returns
    [Error msg] if [s] is not valid JSON, if the ["type"] field is missing or
    unknown, or if required fields are absent. *)

(** {1:server Server messages} *)

val notebook_to_json :
  cells:(Quill.Cell.t * Quill.Session.cell_status) list ->
  can_undo:bool ->
  can_redo:bool ->
  string
(** [notebook_to_json ~cells ~can_undo ~can_redo] is a ["notebook"] JSON message
    with the full notebook state. Each cell is paired with its execution status.
*)

val cell_output_to_json : cell_id:string -> Quill.Cell.output -> string
(** [cell_output_to_json ~cell_id output] is a ["cell_output"] JSON message for
    [output] of cell [cell_id]. *)

val cell_status_to_json : cell_id:string -> Quill.Session.cell_status -> string
(** [cell_status_to_json ~cell_id status] is a ["cell_status"] JSON message for
    cell [cell_id]. *)

val cell_inserted_to_json :
  pos:int -> Quill.Cell.t -> Quill.Session.cell_status -> string
(** [cell_inserted_to_json ~pos cell status] is a ["cell_inserted"] JSON message
    for [cell] at position [pos]. *)

val cell_deleted_to_json : cell_id:string -> string
(** [cell_deleted_to_json ~cell_id] is a ["cell_deleted"] JSON message for cell
    [cell_id]. *)

val cell_moved_to_json : cell_id:string -> pos:int -> string
(** [cell_moved_to_json ~cell_id ~pos] is a ["cell_moved"] JSON message for cell
    [cell_id] moved to position [pos]. *)

val cell_updated_to_json : Quill.Cell.t -> Quill.Session.cell_status -> string
(** [cell_updated_to_json cell status] is a ["cell_updated"] JSON message for
    [cell] with [status]. *)

val completions_to_json :
  request_id:string -> Quill.Kernel.completion_item list -> string
(** [completions_to_json ~request_id items] is a ["completions"] JSON message
    with completion [items] for the given [request_id]. *)

val saved_to_json : unit -> string
(** [saved_to_json ()] is a ["saved"] JSON message. *)

val undo_redo_to_json : can_undo:bool -> can_redo:bool -> string
(** [undo_redo_to_json ~can_undo ~can_redo] is an ["undo_redo"] JSON message
    with the current undo/redo availability. *)

val error_to_json : string -> string
(** [error_to_json msg] is an ["error"] JSON message with [msg]. *)
