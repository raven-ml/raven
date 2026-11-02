(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Runs as a listing reads them: the representation of {!Run.t}, and the status
    rule ({!Store_format.section-status}). *)

type status =
  [ `Running | `Silent | `Crashed | `Finished | `Failed of string | `Killed ]

type stamp = (float * int * int) option

type t = {
  id : string;
  dir : string;
  start : Record.start;
  started_at : float;
  tags : string list;
  notes : string option;
  diff : string option;
  status : status;
  ended_at : float option;
  updated_at : float;
  read_at : float;  (** The wall clock when this was read. *)
  tags_stamp : stamp;
  notes_stamp : stamp;
}

val log : string -> string
(** [log dir] is the path of the log of the run in [dir]. *)

val status :
  held:bool option ->
  writer:string option * string option ->
  log_mtime:float ->
  (float * Record.status) option ->
  status * float option
(** [status ~held ~writer ~log_mtime last] is the run's status and end time,
    where [held] is the probe's result taken before [last] was read, if the log
    could be opened, and [writer] the machine and boot of the run's latest
    writer. *)

val tags : string -> string list
(** [tags dir] is the tags of the run in [dir], sorted; damaged tag files are
    left out. *)

val read_tag : string -> string -> string option
(** [read_tag dir name] is the tag of the tag file [name] of the run in [dir],
    or [None] if it is damaged. *)

val tag_file : string -> string -> string
(** [tag_file dir tag] is the path of [tag]'s file in the run [dir]. *)

val notes : string -> string option

val writer : string -> string option * string option
(** [writer dir] is the machine and boot of the run's latest writer, from its
    [writer] file. *)

type error = [ `Not_a_run | `Unreadable of string ]

val read : ?prev:t -> id:string -> string -> (t, error) result
(** [read ?prev ~id dir] reads the run in [dir], reusing what [prev], an earlier
    read of it, says that cannot have changed. *)

val first_line : Unix.file_descr -> string option
(** [first_line fd] is the log's first whole line, without its line feed. *)

val last_line : Unix.file_descr -> size:int -> string option
(** [last_line fd ~size] is the last line of the first [size] bytes that ends
    with a line feed, without it. *)

val decode_start : string -> (Record.start * float, error) result
