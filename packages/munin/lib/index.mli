(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The run index: the header of each run in a store, by run id.

    The runs of a store are the directories under [experiments/*/runs] that
    hold a manifest. The index in [index.json] at the store root caches their
    headers, so a store lists and finds runs without reading each manifest.
    Whatever the index holds, a run whose directory holds a manifest is
    found, from its manifest if the index lacks it, and an entry whose
    directory holds none is ignored. Each change writes back the headers of
    the runs on disk.

    A session writes a run's status to its event log first and to the index
    after, so a process killed between the two leaves the index behind. The
    log wins. Each entry records the log's length when its header was taken,
    and a run whose log now has another length takes its status from the
    log's last event ({!Event_log.last}): the status of a [Finished] event,
    and [`Running] for any other. Those runs are the ones whose log has
    grown since the index recorded it, which are the runs logging events,
    and every run of an index written without lengths.

    {!read}, {!find} and every change stat each run's manifest and event
    log, and read the last 4 KiB of the log of each run above. A store whose
    [N] running runs log events pays up to [N] such reads per listing, and
    per change under the exclusive lock.

    A change holds an exclusive [Unix.lockf] lock on [index.lock] beside the
    index while it reads the index, changes it and writes it back, and
    {!read} holds a shared one. A write goes to [index.json.tmp], which then
    replaces the index. So where the file system honours [lockf] across its
    clients, the changes of the processes, threads and domains that change
    one store apply one after another and none is lost, and a reader sees
    the index before a change or after it, whole. Where the file system
    refuses the lock, a change removes [index.json] instead, and reads take
    every header from the manifests.

    Raises [Unix.Unix_error] or [Sys_error] if a change cannot open the lock
    file or write the index. *)

type status = [ `Running | `Finished | `Failed | `Killed ]
(** The type for run statuses, those of {!Run.status}. *)

type entry = {
  experiment : string;
  name : string option;
  group : string option;
  parent_id : string option;
  status : status;
  tags : string list;
  started_at : float;  (** Seconds since the Unix epoch. *)
}
(** The type for the header of a run. *)

val read : string -> (string, entry) Hashtbl.t
(** [read root] is the header of each run of the store at [root], by run id.
    An index entry with no experiment is ignored, and one with an unknown
    status reads as [`Running]. *)

val find : string -> id:string -> entry option
(** [find root ~id] is the header of the run [id] of the store at [root], if
    it has one. *)

val add : string -> id:string -> entry -> unit
(** [add root ~id entry] binds [id] to [entry] in the index of [root]. *)

val update_status : string -> id:string -> status -> unit
(** [update_status root ~id status] sets the status of [id]'s entry, if
    any. *)

val remove : string -> id:string -> unit
(** [remove root ~id] removes [id]'s entry, if any. *)
