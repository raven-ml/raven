(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stores: a root directory and what it holds.

    A {e store} is a directory that holds runs, artifacts and blobs in the
    files {!Store_format} specifies. Opening one reads and creates nothing. A
    missing root is an empty store, and the first write creates it. This
    module lists and finds what a store holds, changes what may change
    after it was written (a run's tags and notes, an artifact's aliases),
    and deletes and collects. A {!Session} writes runs and
    publishes versions.

    Unless a value's doc says otherwise, a value that reads or writes the
    store raises [Sys_error] when a file cannot be read or written, and a
    value that takes a run's id raises [Failure] if the store has no run
    of that id. *)

type t
(** The type for stores. *)

val open_ : ?root:string -> unit -> t
(** [open_ ()] is the store at [root], made absolute against the current
    directory. [root] defaults to the first of these that applies:
    - [$RAVEN_TRACKING_DIR], when set and not empty;
    - [$XDG_DATA_HOME/raven/munin], when [$XDG_DATA_HOME] is set and an
      absolute path;
    - [$HOME/.local/share/raven/munin], when [$HOME] is set and not empty;
    - on Windows, [%LOCALAPPDATA%\raven\munin].

    Raises [Failure] if [root] is not given and none applies. *)

val root : t -> string
(** [root s] is the absolute path of [s]'s directory. *)

(** {1:runs Runs} *)

val runs : ?prev:Run.t list -> t -> Run.t list
(** [runs s] is the runs of [s], newest first by start time, then by id.
    For each run it reads the first and last records of its log, its tags
    and its notes, and probes the lock of a run without an end
    ({!Store_format.section-status}). It reads no other part of a log. A
    directory that is not a readable run of this format is left out, and
    {!check} names it. So is a run whose log cannot be opened, such as a
    live run on an SMB mount ({!Store_format.section-network}).

    [prev] is a listing of [s] made earlier. With it, [runs] reads the
    first record only of the runs [prev] lacks, and the last record and
    the lock only of those [prev] saw without an end. For every run [prev]
    saw, it stats the run's [tags/] directory and [notes] file and reads
    them again only when their modification time, size or inode differs
    from what [prev] read, or when the modification time is within a
    second of [prev]'s read, since a file system that keeps whole seconds
    gives a change within that second the time it had. A file system whose
    times are coarser than a second, such as FAT's two seconds, can still
    hide a change, until the next one. A poll of a store that has not changed then
    costs a read of [runs/], two [stat]s per run, and for each run without
    an end an open, a probe and a read of its last record. *)

val run : t -> string -> Run.t option
(** [run s id] is the run [id] of [s], read as {!runs} reads it, or [None]
    if [s] has no run [id].

    Raises [Failure] if [id]'s log does not start as a run of this
    format. *)

val add_tags : t -> string -> string list -> unit
(** [add_tags s id tags] adds [tags] to the run [id]. Its other tags stay.
    Any process may call it, while the run is written and after it ended
    ({!Store_format.section-annotations}).

    Raises [Invalid_argument] if one of [tags] is not a tag
    ({!Store_format.section-names}). *)

val remove_tags : t -> string -> string list -> unit
(** [remove_tags s id tags] removes [tags] from the run [id], ignoring those
    it lacks. Any process may call it, as {!add_tags}. *)

val set_notes : t -> string -> string option -> unit
(** [set_notes s id (Some text)] makes [text] the notes of the run [id],
    replacing them whole, and [set_notes s id None] removes them. When two
    processes set one run's notes at once, the last to finish wins.

    Raises [Invalid_argument] if [text] is not UTF-8. *)

val delete_run : t -> string -> unit
(** [delete_run s id] removes the run [id] and every file it holds. It
    takes the run's lock as a resume does, and removes the run while it
    holds it ({!Store_format.section-resume}). The versions it produced
    stay, and still name it as their producer.

    A [`Silent] run, whose writer cannot be probed, is refused. When its
    writer is known to be gone, {!Session.resume} it, {!Session.finish} it
    as [`Killed], then delete it.

    A user who may change the store's [runs/] and not write the run's log
    deletes it through a read-only descriptor, which a local file system
    locks as any other; on NFS the lock then needs write access, and the
    call raises [Sys_error].

    Once the run is deleted, a failure to remove its files, as when
    another program holds one open on Windows, does not raise: {!gc}
    removes what is left.

    Raises [Failure] if another writer holds the run's lock, or the run is
    [`Silent] ({!Run.status}). *)

(** {1:artifacts Artifacts} *)

val artifacts : ?name:string -> t -> Artifact.t list
(** [artifacts s] is the versions of [s] not deleted, by name, then by
    number. With [name], it is the versions of [name] alone, and reads no
    other name's files. *)

val artifact :
  t -> string -> [ `Latest | `Version of int | `Alias of string ] ->
  Artifact.t option
(** [artifact s name sel] is the version of [name] that [sel] selects: its
    latest ({!Store_format.section-artifacts}), the version numbered [n],
    or the version the alias names. It is [None] if there is no such version or
    it was deleted. *)

val aliases : t -> string -> (string * int) list
(** [aliases s name] is [name]'s aliases and the numbers of the versions
    they name, sorted by alias. *)

val set_alias : t -> Artifact.t -> string -> unit
(** [set_alias s a alias] makes [alias] name the version [a] of its
    artifact, replacing what it named. When two processes set one alias
    at once, the last to finish wins.

    Raises [Invalid_argument] if [alias] is not a name
    ({!Store_format.section-names}). *)

val remove_alias : t -> name:string -> string -> unit
(** [remove_alias s ~name alias] removes [name]'s alias [alias], if it
    has one. *)

val delete_artifact : t -> Artifact.t -> unit
(** [delete_artifact s a] deletes the version [a] and the aliases that name
    it. Its number never names another version. Its blob stays until a
    {!gc} finds that no version names it. *)

(** {1:maintenance Maintenance} *)

val gc : t -> int
(** [gc s] removes the blobs of [s] that no version names, the [.]-named
    work that crashed writers left under [blobs/] and [artifacts/], and
    what a {!delete_run} cut short left under [runs/], and is the number
    of entries it removed. It waits for the
    publications in progress, and those that start wait for it
    ({!Store_format.section-blobs}). *)

val check : t -> (string * string) list
(** [check s] is each path of [s] that breaks the format, with what is
    wrong, sorted by path:
    - a run directory that is not a readable run, or whose log cannot be
      opened;
    - a damaged line in a log, and a line that is not whole in the log of
      a run that has crashed or ended ({!Store_format.section-reading});
    - a damaged tag file;
    - a statistics hint that does not parse or is of another format, and
      one whose [rules] and [length] match its log but whose statistics
      differ from the log's ({!Store_format.section-stats});
    - a file under a run's [files/] that no record names;
    - an alias that names no version;
    - a version whose blob is missing, or whose blob's content does not
      match its digest;
    - a [.]-named entry a writer left;
    - a run directory that a {!delete_run} cut short left, until {!gc}
      removes it.

    It reads every log, every tag file and every blob. *)

(**/**)

(* Tests hold the store's shared lock, as a publication in progress does,
   to see what gc does meanwhile. *)

val with_publication_lock : t -> (unit -> 'a) -> 'a

(**/**)
