(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Writing a run.

    A {e session} is the one writer of one run
    ({!Store_format.section-status}). While it is open its process holds
    the run's lock, so no other session can write the run, and readers
    under its boot see it running. A process that exits without finishing
    its session releases the lock too, and its run then reads as crashed
    to those readers.

    The life cycle of a session:
    + {!start} creates a run and is its session, or {!resume} is a new
      session of a run whose writer is gone. {!within} brackets either
      with {!finish}.
    + {!val-metric} declares the metrics the run logs and {!log} and
      {!log_metrics} log their samples, and {!sample} samples readings of
      the machine and its devices on a schedule.
    + {!log_file}, {!set_summary}, {!log_artifact} and {!use_artifact}
      record the rest.
    + {!finish} stops the samplers, writes the run's end and releases the
      lock.

    Each write appends one record to the run's log. A reader in another
    process sees the record once the call returns, and the record survives
    the program's crash. It survives a power loss once synced. {!start},
    {!resume}, {!log_artifact}, {!sync} and {!finish} sync the log before
    they return, and between them the system writes it out on its own
    schedule ({!Store_format.section-durability}).

    The session's {e latest step} is the step of the last [metrics] or
    [resume] record it wrote, or [0] before any
    ({!Store_format.section-log}).

    Unless a value's doc says otherwise:
    - a session's values may be called from any domain or thread, and they
      take effect one at a time;
    - a value that writes raises [Invalid_argument] on a finished session;
    - a write the system refuses, as on a full disk, raises [Sys_error] and
      leaves the log as it was before the call;
    - once a sync has failed, every write raises [Failure], since the
      system may have dropped records written before it.

    A training loop that records its loss:
    {[
      Munin.Session.within
        (Munin.Session.start ~experiment:"mnist"
           ~params:[ ("lr", `Float lr) ] ~steps:999 ())
      @@ fun s ->
      let loss = Munin.Session.metric s ~goal:`Minimize "train/loss" in
      for step = 0 to 999 do
        Munin.Session.log loss ~step (train_step ())
      done
    ]} *)

type t
(** The type for sessions. *)

(** {1:lifecycle Starting and ending} *)

val start :
  ?store:Store.t ->
  ?name:string ->
  ?parent:string ->
  ?tags:string list ->
  ?params:(string * Value.t) list ->
  ?steps:int ->
  ?provenance:Provenance.t ->
  experiment:string ->
  unit ->
  t
(** [start ~experiment ()] creates a run of [experiment] in [store] and is
    its session. The run's [start] record holds [experiment], [name],
    [parent], [params], [steps] and [provenance], which never change; its
    tags and notes may ({!Store.add_tags}, {!Store.set_notes}). It syncs
    the log.

    - [store] defaults to {!Store.open_}[ ()].
    - [parent] is the id of the run of [store] that this run belongs to,
      such as the run that drives a sweep.
    - [steps] is the largest step the run will log, so that readers can
      show progress.
    - [provenance] defaults to {!Provenance.detect}[ ()]. When it says the
      work tree is dirty, [start] keeps the output of
      [git diff --binary HEAD], run in its [cwd], beside the log.

    Raises [Invalid_argument] if one of [tags] is not a tag, [experiment],
    [name] or a param's key is not a text ({!Store_format.section-names}),
    a param's key repeats, a string value is not UTF-8, or [steps] is not a step. Raises
    [Failure] if [parent] names no run of [store] or [git diff] fails, and
    [Sys_error] if the store cannot be written or its file system refuses
    locks. *)

val resume : ?provenance:Provenance.t -> Run.t -> step:int -> t
(** [resume run ~step] is a new session of [run] that continues from
    [step]. The program has restored a state that includes every sample
    logged at [step] and before, and logs later steps. Samples and files
    [run] logged with a step above [step] become orphans
    ({!Store_format.section-reading}).

    It takes the run's lock, waiting out readers' probes, and reads the
    log once. The read gives whether the run ended, the kind of each key,
    so that the session refuses a key of the other kind as {!start}'s
    does ({!Store_format.section-log}), and each key's statistics, which
    the session keeps for the run's statistics hint. It costs one pass
    over the log. The session then cuts what the last writer left that is
    not whole lines, replaces the run's [writer] with this machine's
    identities, appends a [resume] record holding [step] and
    [provenance], and syncs the log ({!Store_format.section-resume}).
    [provenance] defaults to {!Provenance.detect}[ ()].

    On Windows, it blocks briefly while another handle holds the log, and
    raises [Sys_error] if the handle outlasts the wait.

    Raises [Invalid_argument] if [step] is not a step, and [Failure] if the
    run has ended or another writer holds its lock. *)

val within : t -> (t -> 'a) -> 'a
(** [within s f] is [f s], after which it finishes [s] as [`Finished]. When
    [f] raises [Sys.Break] it finishes [s] as [`Killed], and when [f]
    raises another exception [e], as [`Failed (Printexc.to_string e)]; it
    then raises [f]'s exception again with its backtrace. If finishing [s]
    raises after [f] raised, [f]'s exception is raised, and the run, left
    without an end, reads as crashed. *)

val finish : ?status:[ `Finished | `Failed of string | `Killed ] -> t -> unit
(** [finish s] stops [s]'s samplers, waiting for a reading in progress,
    writes the run's end with [status], syncs the log, writes the run's
    statistics hint ({!Store_format.section-stats}) and releases the
    lock. [status] defaults to [`Finished]. [s] is finished even if this
    raises, and finishing a finished session does nothing.

    Raises what stopped a sampler ({!sample}), after it has written the
    end with [status]. Raises [Sys_error] if the hint cannot be written
    after the end is synced; the run has ended all the same, and readers
    fold its log. *)

val sync : t -> unit
(** [sync s] makes every record [s] wrote survive a power loss. It blocks
    until the storage reports the write done. *)

val id : t -> string
(** [id s] is the id of [s]'s run. *)

val store : t -> Store.t
(** [store s] is the store that holds [s]'s run. *)

(** {1:metrics Metrics and summary}

    A {e handle} logs the samples of one metric of its session's run.
    {!val-metric} declares the metric and is its handle. *)

type metric
(** The type for handles. *)

val metric :
  t -> ?goal:Metric.goal -> ?step_metric:metric -> string -> metric
(** [metric s key] declares the metric [key] in [s]'s run and is the
    handle that logs its samples. [goal] says whether lower or higher
    values are better, which {!Metric.best} needs. [step_metric] is
    another metric of [s] that readers use as [key]'s x axis, paired by
    step ({!Store_format.section-reading}). The first declaration of [key]
    in the run appends a [metric] record; an equal one later, as after a
    resume, writes nothing and is another handle of [key].

    Raises [Invalid_argument] if [key] is not a key
    ({!Store_format.section-names}), the run declared [key] with another
    goal or step metric, a reading of [s]'s run gave [key], or
    [step_metric] is another session's. *)

val key : metric -> string
(** [key m] is [m]'s key. *)

val log : metric -> step:int -> ?time:float -> float -> unit
(** [log m ~step v] is {!log_metrics}[ s ~step ?time [ (m, v) ]] for [m]'s
    session [s]. Steps need not increase from one sample to the next. *)

val log_metrics :
  t -> step:int -> ?time:float -> (metric * float) list -> unit
(** [log_metrics s ~step pairs] appends one [metrics] record that holds a
    sample of each metric of [pairs], all at [step] and [time]. [time]
    defaults to the wall clock now. An empty [pairs] writes nothing.

    Raises [Invalid_argument] if [step] is not a step, [time] is not
    finite, or a metric of [pairs] is another session's or appears
    twice. *)

val set_summary : t -> (string * Value.t) list -> unit
(** [set_summary s values] appends a [summary] record, in which each value
    replaces the run's earlier value of its key. A summary holds what the
    program states about the run, such as a test score. Readers compute
    the statistics of metrics ({!History.val-stats}).

    Raises [Invalid_argument] if a key is not a text or repeats, or a
    string value is not UTF-8. *)

(** {1:sampling Sampling}

    A {e reading} reads some facts now and gives them as pairs of a key
    and a number, such as [("sys/cpu_busy_pct", 37.5)]. It owns what it
    reads. A reading of a rate keeps the counters of its previous call and
    gives the mean since then, with the window it covers under a key of
    its own. Readings of the machine and of this process are in
    [munin.sys], and a program builds readings of its devices from what
    their library reports. A {e sampler} calls readings on a schedule and
    logs what they give, and knows nothing of what they read. *)

type reading = unit -> (string * float) list
(** The type for readings. *)

val sample : t -> every:float -> reading list -> unit
(** [sample s ~every readings] calls each reading now, in the caller's
    thread, then every [every] seconds on a thread of its own until [s]
    finishes. It logs each call's pairs as one [reading] record, at the
    time of the call and the session's latest step, and a call that gives
    no pair logs nothing. A reading of rates gives nothing at its first
    call, so its first samples come [every] seconds after [sample]. Samples of readings are never orphans
    ({!Store_format.section-reading}). Each reading must be safe to call
    from another thread, and must be given to one sampler only, since each
    call moves its window.

    A call that raises [Sys_error] gives nothing that time. A call that
    raises another exception, or gives a key that is not a key, a key
    twice or the key of a metric of [s]'s run, stops the sampler, and so does a
    write that fails. {!finish} raises the first exception that stopped a
    sampler of [s].

    Raises [Invalid_argument] if [every] is not positive and finite, and
    what a reading's first call raises, other than [Sys_error]. *)

(** {1:files Files} *)

val log_file : t -> step:int -> key:string -> string -> unit
(** [log_file s ~step ~key path] copies the file [path] into [s]'s run and
    appends a [file] record that names the copy at [step]
    ({!Store_format.section-files}). It returns once the copy and its
    record are written; other threads' writes to [s] proceed meanwhile.
    Readers find the file with {!History.files}.

    Raises [Invalid_argument] if [key] is not a key or [step] is not a
    step, and [Sys_error] if [path] cannot be read. *)

(** {1:artifacts Artifacts} *)

val log_artifact :
  t -> ?step:int -> ?metadata:(string * Value.t) list -> name:string ->
  string -> Artifact.t
(** [log_artifact s ~name path] publishes the file or directory [path] as
    a new version of the artifact [name], produced by [s]'s run at [step],
    and is that version. It copies [path] once, computing the digest of the
    bytes it copies, so the version's digest names what was stored even
    if [path] changes meanwhile. Content already stored is not stored
    again, and still makes a new version. It syncs the blob, the version
    file and the run's log ({!Store_format.section-durability}). It
    returns once the version is published; other threads' writes to [s]
    proceed meanwhile, and a {!Store.gc} waits for the publication
    ({!Store_format.section-blobs}).

    Raises [Invalid_argument] if [name] is not a name
    ({!Store_format.section-names}), [step] is not a step, a metadata key
    is not a text or repeats, a string value is not UTF-8, or [path] holds
    a symbolic link to a directory; and [Sys_error] if [path] cannot be
    read. *)

val use_artifact : t -> Artifact.t -> unit
(** [use_artifact s a] appends a [use] record: [s]'s run used the version
    [a]. *)
