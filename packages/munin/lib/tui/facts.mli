(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What a poll knows of a run and of an experiment, each fact computed once
    per poll from what it read. *)

type run = {
  run : Munin.Run.t;  (** The run as last read. *)
  state : Watch.state;
  stats : (string * Munin.History.key_stats) list;
      (** Each key's statistics, sorted by key; [[]] until read. *)
  metrics : string list;
      (** Its metrics: losses with a goal, then the other metrics with a
          goal, then the rest, each by key. *)
  latest : int option;  (** The greatest step of its metrics' samples. *)
  non_finite : (string * Munin.Metric.sample * int) option;
      (** The metric of its earliest sample that is not finite, that sample,
          and how many metrics have one. *)
  last_arrival : float option;  (** When a sample last arrived. *)
  active : float;
      (** Its active time: from its start, less its pauses, to its end, to
          the poll while a writer may hold it, or else to its last write. *)
  rate : float option;
      (** Its steps per second over the last 10 minutes, or the last tenth of
          its active time, once that window holds 60 s; known only for a run
          whose every key was read. *)
  x_end : float;
      (** The end of its x domain: its [steps] when no step passes them,
          else {!bound} of its greatest step, and [1.] without a step. *)
}
(** The type for what a poll knows of a run. *)

val of_run :
  now:float ->
  history:Munin.History.t option ->
  hint:(string * Munin.History.key_stats) list option ->
  watch:Watch.t option ->
  pauses:(float * float) list ->
  Munin.Run.t ->
  run
(** [of_run ~now ~history ~hint ~watch ~pauses r] is what is known of [r] at
    [now] from its history, if held, else its hint; [pauses] as {!pauses}
    gives them. *)

val pauses : Munin.History.t -> (float * float) list
(** [pauses h] is each resume's time with the pauses up to it, the latest
    first. A pause runs from the last metric sample before a resume to the
    resume. It reads every kept series once. *)

val active_at : (float * float) list -> started:float -> float -> float
(** [active_at pauses ~started t] is the active time at [t] of a run that
    started at [started]. *)

val name : Munin.Run.t -> string
(** [name r] is [r]'s name, or its id. *)

val stats : run -> string -> Munin.History.key_stats option
(** [stats f key] is [key]'s statistics. *)

val goal : run -> string -> Munin.Metric.goal option
(** [goal f key] is [key]'s goal. *)

val best : run -> string -> Munin.Metric.sample option
(** [best f key] is [key]'s best sample by its goal. *)

val glyph : run -> string
(** [glyph f] is the run's state glyph: [●] live or writing, [○] running or
    starting, [◌] stalled or silent, [✓] finished, [■] killed, [✗] failed or
    crashed, and [×] for a run, not failed, with a metric sample that is not
    finite. *)

val commit : Munin.Provenance.t -> string option
(** [commit p] is the first seven characters of [p]'s commit. *)

val value : Munin.Value.t -> string
(** [value v] is a param's value as the program gave it. *)

val bound : float -> float
(** [bound x] is the least of 1, 2 or 5 times a power of ten at or above
    [x], the largest float past them, and [1.] for [x <= 0.]. *)

(** {1:experiments Experiments} *)

type experiment = {
  name : string;
  runs : string list;  (** Its runs' ids, in the listing's order. *)
  key : string option;
      (** Its key metric: among its runs' metrics in the order of their
          keys, since a history keeps no order of declaration, the first with
          a goal outside [train/] whose key ends in [loss], else the first
          with a goal outside [train/], else the first with a goal, else the
          first. *)
  varying : string list;
      (** The params whose value differs between its runs, a run without
          one counting as another value. *)
}
(** The type for what a poll knows of an experiment. *)

val experiments : run list -> experiment list
(** [experiments runs] is the experiments of [runs], in the order of their
    first run. *)
