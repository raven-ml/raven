(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Samples, series and their statistics.

    A {e sample} is a step, a time and a float that a run logged under a
    key, as a metric's sample or a reading's ({!Store_format.section-log}).
    A reader gets a key's samples that are not orphans as a {!type-series}, and
    their statistics as {!type-stats} ({!History.series},
    {!History.val-stats}). A program declares a metric and logs its samples
    through a session ({!Session.val-metric}).

    A sample is {e finite} when its value is neither NaN nor infinite.
    Statistics that compare or add values range over finite samples alone,
    so a NaN or an infinity never becomes a minimum, a maximum, a mean or a
    best value; the samples that are not finite are counted and dated
    beside them. *)

type goal = [ `Minimize | `Maximize ]
(** The type for goals: whether lower or higher values are better. *)

type sample = { step : int; time : float; value : float }
(** The type for samples: the step the program gave, the time the sample
    was taken, and its value, read back exactly
    ({!Store_format.section-values}). *)

(** {1:series Series}

    A {e series} is a key's samples that are not orphans, in log order. It
    is a snapshot, in the sense that reading more of the log gives a new
    series and leaves this one as it was. Its samples are indexed from [0]
    to [length s - 1], and the accessors raise [Invalid_argument] for any
    other index. *)

type series = Series.t
(** The type for series. *)

val length : series -> int
(** [length s] is the number of samples of [s]. *)

val step : series -> int -> int
(** [step s i] is the step of [s]'s [i]th sample. *)

val time : series -> int -> float
(** [time s i] is the time of [s]'s [i]th sample. *)

val value : series -> int -> float
(** [value s i] is the value of [s]'s [i]th sample. *)

val sample : series -> int -> sample
(** [sample s i] is [s]'s [i]th sample. *)

(** {1:stats Statistics}

    Among samples of equal value the earliest in log order wins, and
    [-0.] equals [0.]. *)

type finite = {
  least : sample;  (** The finite sample of least value. *)
  greatest : sample;  (** The finite sample of greatest value. *)
  mean : float;
      (** The mean of the finite values, summed with compensation so that
          its rounding error does not grow with the number of samples. It
          lies between [least]'s and [greatest]'s values, so it is finite
          even when the values' sum is past the largest float. *)
}
(** The type for the statistics of a series' finite samples. *)

type stats = {
  count : int;  (** The number of samples. *)
  last : sample;
      (** The latest sample ({!Store_format.section-reading}), finite or
          not. *)
  finite : finite option;
      (** The statistics of the finite samples, [None] iff none is
          finite. *)
  non_finite : (int * sample) option;
      (** The number of samples that are not finite and the earliest of
          them, [None] iff every sample is finite. *)
}
(** The type for the statistics of a series of at least one sample. *)

val best : goal -> stats -> sample option
(** [best goal st] is the finite sample that best meets [goal]: the
    [least] of [st]'s finite samples under [`Minimize] and the [greatest]
    under [`Maximize], the earliest among equals. It is [None] if no
    sample is finite. *)

