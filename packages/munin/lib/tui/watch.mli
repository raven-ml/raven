(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the dashboard has seen of a run since it began watching it: the
    arrivals of its metric samples, timed on the dashboard's clock, and the
    state they give with the run's status (the States section of
    {!Munin_tui}). *)

type t
(** The type for what a dashboard saw of a run. *)

val start : samples:int -> t
(** [start ~samples] is a watch that began on a run whose log held [samples]
    metric samples. Those are no arrival. *)

val observe : t -> now:float -> samples:int -> t
(** [observe w ~now ~samples] is [w] after a poll at [now] read [samples]
    metric samples: an arrival when they are more than before. *)

val last_arrival : t -> float option
(** [last_arrival w] is the time of [w]'s latest arrival. *)

(** The type for a run's states. *)
type state =
  | Starting  (** A writer holds the run and it has no metric sample. *)
  | Running  (** A writer holds it and no sample arrived since watching. *)
  | Live  (** A writer holds it and a sample arrived recently. *)
  | Stalled  (** A writer holds it and no sample arrived for long. *)
  | Writing  (** Its lock is not probed and a sample arrived recently. *)
  | Silent  (** Its lock is not probed and no sample arrived recently. *)
  | Crashed
  | Finished
  | Failed
  | Killed

val state : Munin.Run.status -> t option -> now:float -> state
(** [state status w ~now] is the state of a run of [status] watched by [w]. A
    sample arrived recently when it arrived within three usual gaps, the
    median gap between the last 32 arrivals, and at least 10 s ago at most;
    with fewer than two arrivals timed, any arrival is recent. *)

val word : state -> string
(** [word s] is [s]'s word: [starting], [running], [live], ... *)

val live : state -> bool
(** [live s] is [true] iff a writer may still log: [Starting], [Running],
    [Live] or [Writing]. *)

val broken : state -> bool
(** [broken s] is [true] iff [s] is [Failed] or [Crashed]. *)
