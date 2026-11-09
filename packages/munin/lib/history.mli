(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What a run's log says, folded.

    A {e history} is what a run's log says up to a point: its metrics' and
    readings' series and statistics, its summary, the files it logged, the
    artifact versions it used, its resumes, and the run with its status
    ({!Store_format.section-reading}). {!read} folds a log from its start,
    and {!follow} folds what the log gained since. Both read each byte
    once, so following a live run costs its new records, whatever its
    length. A history keeps every key's statistics, and the samples of the
    keys it was asked for, in columns of 24 bytes a sample.

    A history is a value. Following one gives a new history and leaves the
    old one valid and unchanged, and its accessors do no I/O. A follow
    appends the new samples into columns it shares with the history it
    follows. The first follow of a history claims that history's columns;
    a later follow of the same history, on any domain, finds them claimed
    and copies the columns of the keys it extends, at the cost of their
    samples.

    A history may be read from any domain. Following a history on one
    domain leaves every other history valid, including those another
    domain is reading. A history passed to another domain goes through
    something that synchronises the two, as any value does.

    A reader that wants a run's latest state follows its history at
    intervals:
    {[
      let rec watch h =
        let h = Munin.History.follow h in
        draw (Munin.History.run h) (Munin.History.series h "train/loss");
        Unix.sleepf 1.;
        watch h
      in
      watch (Munin.History.read ~series:[ "train/loss" ] run)
    ]} *)

type t
(** The type for histories. *)

(** {1:reading Reading} *)

val read : ?limit:int -> ?series:string list -> Run.t -> t
(** [read run] is [run]'s history up to the log's last whole line, or, with
    [limit], up to the first whole line that ends past [limit] bytes from
    the log's start. It probes [run]'s lock before it reads
    ({!Store_format.section-status}), and reads the run's tags and notes
    again. [series] names the keys whose samples the history keeps, and
    defaults to every key; the statistics of every key are kept.

    Raises [Sys_error] if the log cannot be read. *)

val follow : ?limit:int -> t -> t
(** [follow h] is [h] with the whole lines its log gained since folded in,
    or, with [limit], up to the first whole line that ends past [limit]
    bytes after what [h] read. It keeps the keys [h] keeps. It opens the
    log again, probes the lock, reads only the new bytes, and reads the
    run's tags, each digested to check it, and notes again. A log shorter
    than [h] read, or another file than the one [h] read, such as one a
    copy replaced, is read again from its start, and the result is
    {!restarted}. A resume that orphans samples of a key whose samples
    [h] does not keep also makes [follow] read the log again from its
    start, since that key's statistics cannot be recomputed otherwise, and
    that result is {!restarted} too.

    [series (follow h) k] extends [series h k], unless the result is
    {!restarted} or a resume read by the follow orphaned samples of [k].

    It costs the new bytes and a lookup per sample among the run's keys,
    and a [resume] record costs the samples of the keys it orphans. When
    the log has not grown it costs an open, a probe, the read of the last
    record, and the reads of the tags and notes.

    Raises [Sys_error] if the log cannot be read, and [Failure] if a log
    read again from its start no longer starts as a run of this format. *)

val restarted : t -> bool
(** [restarted h] is [true] iff the {!follow} that made [h] read its log
    again from its start. *)

(** {1:samples Metrics and readings} *)

val metrics : t -> string list
(** [metrics h] is the keys of the run's metrics that have a sample that is
    not an orphan, sorted: the metrics whose {!stats} is [Some]. A key whose
    every sample is an orphan is left out; {!orphans} still gives them. *)

val readings : t -> string list
(** [readings h] is the keys of the run's readings, sorted. *)

val goal : t -> string -> Metric.goal option
(** [goal h k] is the goal of [k]'s latest [metric] record, if it gave
    one. *)

val step_metric : t -> string -> string option
(** [step_metric h k] is the key that [k]'s latest [metric] record names as
    its x axis, if it names one. A sample of [k] pairs with that key's
    latest sample at the same step ({!Store_format.section-reading}). *)

val series : t -> string -> Metric.series
(** [series h k] is [k]'s samples that are not orphans, in log order. It
    is empty for a key with none.

    Raises [Invalid_argument] if [h] does not keep [k]'s samples
    ({!read}). *)

val orphans : t -> string -> Metric.series
(** [orphans h k] is [k]'s orphans, in log order.

    Raises [Invalid_argument] if [h] does not keep [k]'s samples. *)

val stats : t -> string -> Metric.stats option
(** [stats h k] is the statistics of [k]'s samples that are not orphans,
    or [None] if there is none. *)

(** {1:other Summary, files, uses and resumes} *)

val summary : t -> (string * Value.t) list
(** [summary h] is the run's summary: each key's latest value, sorted by
    key. *)

type file = {
  key : string;  (** The key it was logged under. *)
  step : int;  (** The step it was logged at. *)
  time : float;  (** When it was logged. *)
  path : string;  (** Its absolute path. *)
}
(** The type for files a run logged. *)

val files : t -> file list
(** [files h] is the files the run logged that are not orphans, in log
    order. *)

val uses : t -> (string * int) list
(** [uses h] is the artifact versions the run used, as names and numbers,
    in log order. *)

type resume = {
  step : int;  (** The step the run continued from. *)
  time : float;  (** When it resumed. *)
  provenance : Provenance.t;  (** The resuming process. *)
}
(** The type for resumes. *)

val resumes : t -> resume list
(** [resumes h] is the run's resumes, in log order. *)

val damaged : t -> int
(** [damaged h] is the number of damaged lines read
    ({!Store_format.section-reading}). *)

(** {1:ended Statistics of a run without its history}

    An ended run keeps its keys' statistics in a hint beside its log
    ({!Store_format.section-stats}), so that a listing can show each
    run's best values without folding long logs. *)

type key_stats = {
  kind : [ `Metric | `Reading ];  (** Whether the key is a metric's. *)
  goal : Metric.goal option;  (** The goal of a metric's key, if given. *)
  stats : Metric.stats;  (** The key's statistics. *)
}
(** The type for a key's statistics. *)

val stats_of : Run.t -> (string * key_stats) list
(** [stats_of run] is the statistics of each key of [run] that has a
    sample, sorted by key, equal to what {!stats} gives for the run's
    history. When [run] has a hint whose rules and length match
    ({!Store_format.section-stats}), it reads the hint alone; otherwise
    it folds the log as {!read}[ ~series:[]] does, at the cost of the
    whole log. A hint that does not parse or is of another format is
    ignored.

    Raises [Sys_error] if the log cannot be read. *)

(** {1:run The run} *)

val run : t -> Run.t
(** [run h] is the run as [h] last read it: its tags, notes and status,
    and its end if [h] read one. *)
