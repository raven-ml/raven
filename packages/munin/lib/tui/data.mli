(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the dashboard read of a store, and the poll that reads it again.

    A value of [t] is what one poll left: the listing, the histories it holds,
    what it knows of each run and experiment ({!Facts}), and the reductions of
    the series the screen draws ({!Reduce}). A poll runs off the loop and
    makes the next value from the last one and from what the screen {e wants};
    only a poll makes a value, and one runs at a time, so the value a poll
    starts from is the one held when it returns. Values are immutable: the
    loop draws from one while a poll makes the next.

    A history is held while its run has no end, or while a want needs what
    only its samples give: a reduction the wants name that no poll made
    whole, from a history that read its run's end, a readout's value or a
    comparison's difference; its keys are those the wants ask of it. A poll
    that lists follows every history it holds until the history reads its
    run's end, the listing that first shows the end included; one that does
    not list reads only what the wants lack, from the histories held and the
    last listing. A whole reduction stays after no want names it, at the dot
    counts the screen draws, while its run is listed: coming back to an
    ended run reads nothing of it. A resize names other dot counts, so an
    ended run's history that no other want holds is read again from its
    log, as its first draw read it. The statistics of an ended run that no
    want names come from its hint, read once. *)

type axis = Step | Time
(** The type for a plot's x: the step, or the run's active time. *)

type plot = {
  run : string;
  key : string;
  axis : axis;
  x : float * float;  (** The x domain. *)
  dots : int;  (** The plot's dot columns. *)
  smooth : bool;  (** Whether the reduction has a smoothed form. *)
}
(** The type for a series reduced for a plot. *)

type at =
  | Latest of string  (** The latest step of a run, as the poll finds it. *)
  | At of int
(** The type for the step of a readout. *)

type readout = {
  key : string;
  runs : string list;
  at : at;
  tau : float;  (** The time constant of the smoothed values. *)
}
(** The type for the values of runs at a step. *)

type compare = { focus : string; other : string; keys : string list }
(** The type for the differences of two runs at their latest common step. *)

type wants = {
  focus : string option;  (** The cursor's run. *)
  every : bool;  (** Hold the focus's history even when it has ended. *)
  plots : (plot * (float -> int)) list;
      (** Each plot drawn, with the dot column of an x of its domain. *)
  readouts : readout list;
  compare : compare option;
}
(** The type for what the screen draws. *)

val nothing : wants
(** [nothing] wants nothing. *)

val same : wants -> wants -> bool
(** [same a b] is [true] iff [a] and [b] want the same, their plots
    compared by their records. *)

type t
(** The type for what was read. *)

val empty : t
(** [empty] has read nothing. *)

val poll : Munin.Store.t -> now:float -> list:bool -> wants -> t -> t
(** [poll store ~now ~list wants d] is [d] after a poll at [now] that lists
    [store] when [list] and reads what [wants] needs. A run that cannot be
    read keeps what was held of it and is named in {!errors}. *)

val time : t -> float
(** [time d] is the time of the poll that made [d]. *)

val listed : t -> bool
(** [listed d] is [true] once a poll listed the store. *)

val runs : t -> Facts.run list
(** [runs d] is what is known of each listed run, newest first. *)

val run : t -> string -> Facts.run option
(** [run d id] is what is known of the run [id]. *)

val experiment : t -> string -> Facts.experiment option
(** [experiment d name] is what is known of the experiment [name]. *)

val history : t -> string -> Munin.History.t option
(** [history d id] is the history held of the run [id]. *)

val reductions :
  t -> run:string -> key:string -> axis:axis -> x:float * float -> Reduce.t list
(** [reductions d ~run ~key ~axis ~x] is the reductions a poll made of the
    series [key] of [run] over the x domain [x], at each dot count it made
    one. *)

val values : t -> readout -> (string * float option * float option) list
(** [values d r] is, for each run of [r] whose history the poll held, its raw
    and smoothed values at [r]'s step: those of its latest sample at or
    before the step. A step [Latest id] is that of the poll's facts of
    [id]. *)

val difference : t -> compare -> string -> (int * float * float) option
(** [difference d c key] is the latest step at which both runs of [c] logged
    a finite [key], with the focus's value and the other's there. *)

val errors : t -> (string * string) list
(** [errors d] is each run the last poll could not read, by id, and why; the
    id [""] for the listing. *)
