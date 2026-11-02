(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Facts about a run that more than one screen prints, each defined once. *)

val commit : ?dirty:string -> Munin.Provenance.t -> string option
(** [commit p] is the first seven characters of [p]'s commit, or fewer when it
    has fewer, followed by [dirty] (["+"] by default) when the work tree was
    dirty. *)

val arrow : Munin.Metric.goal option -> string
(** [arrow goal] is [↓], [↑] or [""]. *)

val latest_step : (string * Munin.History.key_stats) list -> int option
(** [latest_step stats] is the largest step of the latest samples of the metrics
    of [stats]. *)

val wall : Munin.Run.t -> float
(** [wall r] is the time from [r]'s start to its end, or to its last write. *)

val pauses : Munin.History.t -> (float * float) list
(** [pauses h] is each resume's time with the pauses up to it, the latest first.
    A pause runs from the last metric sample before a resume to the resume. It
    reads every kept series once. *)

val active_at : (float * float) list -> float -> float -> float
(** [active_at pauses started t] is the active time at [t] of a run that started
    at [started]. *)
