(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A metric's series reduced to the dot columns of a plot.

    A chart draws one value per braille dot column, so a series of any length
    costs a chart what the plot shows. A {e reduction} holds, for each dot
    column, what every way of drawing the series needs: the mean of the
    column's finite samples, arithmetic and geometric, of the raw samples and
    of the smoothed ones; the raw samples' least and greatest values; and
    whether a sample that is not finite fell in it. A chart then chooses
    smoothing and the y scale as it draws, and waits for no new reduction.

    {b Smoothing} is a debiased exponential mean over the steps, of time
    constant [tau] in x: the samples of one step are pooled into their mean
    [y{_i}], and the smoothed value at the step [x] is the mean of the steps
    [x{_i}] up to it, each weighted by
    [(1 - exp (-(x{_i} - x{_i-1}) / tau)) exp (-(x - x{_i}) / tau)], where
    the step before the first is [x{_1} - (x{_2} - x{_1})]. The weights
    depend on the steps alone, so runs that log at different cadences are
    smoothed alike. A sample that is not finite is left out, and a step
    lower than the one before starts the mean again. The latest step's
    value is known once a later step arrives; until then a reduction counts
    it as it stands.

    A reduction is extended by the samples a series gained, at the cost of
    those samples, and is then the reduction of the whole series. It holds
    its columns and nothing a sample: making or extending one allocates a
    few words a sample, which the minor heap reclaims. *)

type grid = {
  lo : float;
  hi : float;  (** The plot's x domain, [lo < hi]. *)
  dots : int;  (** Its dot columns, at least 1. *)
  dot : float -> int;
      (** The dot column, in \[[0];[dots - 1]\], in which the plot draws an x
          of \[[lo];[hi]\]. *)
}
(** The type for the dot columns of a plot. *)

type t
(** The type for reductions. *)

val make : ?tau:float -> grid -> x:(int -> float) -> Munin.Metric.series -> t
(** [make ?tau grid ~x s] is [s] reduced to [grid]'s columns, sample [i] at
    x [x i], with a smoothed form of time constant [tau] when given,
    [tau > 0]. A sample whose x is NaN is
    left out; one whose x is outside the domain is in no column, and the
    finite one nearest each side of it is kept ({!edges}). *)

val extend : t -> x:(int -> float) -> Munin.Metric.series -> t
(** [extend r ~x s] is [make] of [s] with [r]'s grid and smoothing, at the
    cost of the samples of [s] past those [r] read. [s] extends the series
    [r] reduced, and [x] gives those samples the x [r] gave them. *)

val grid : t -> grid
(** [grid r] is the grid [r] was made for. *)

val length : t -> int
(** [length r] is the number of samples [r] read. *)

val smoothed : t -> bool
(** [smoothed r] is [true] iff [r] has a smoothed form. *)

val mean : t -> smooth:bool -> log:bool -> int -> float
(** [mean r ~smooth ~log k] is the mean of column [k]'s finite samples,
    smoothed when [smooth] and [r] has a smoothed form, geometric when [log].
    It is NaN for a column without one, and, when [log], for a column with
    one at or below 0. Within the finite floats, a mean of values whose sum
    passes the largest float included. *)

val extent : t -> int -> (float * float) option
(** [extent r k] is the least and greatest of column [k]'s finite raw
    samples. *)

type event = [ `Nan | `Pos_inf | `Neg_inf ]
(** The type for samples that are not finite. *)

val event : t -> int -> event option
(** [event r k] is the non-finite sample of column [k], NaN over an
    infinity. *)

val edges :
  t -> smooth:bool -> (float * float) option * (float * float) option
(** [edges r ~smooth] is the finite sample, as [(x, y)], nearest the domain
    on its left and on its right, so that a line can run to the plot's edges;
    of two at one x, the later. *)

val common :
  Munin.Metric.series -> Munin.Metric.series -> (int * float * float) option
(** [common a b] is the largest step at which [a] and [b] both have a finite
    sample, with the value of each one's last finite sample there in log
    order, or [None] if they have no such step. Two ordered series
    ({!Munin.Metric.ordered}) cost the samples past that step; others cost
    their length. *)

val at :
  ?tau:float -> x:(int -> float) -> Munin.Metric.series -> float -> float option
(** [at ?tau ~x s v] is [s]'s value at the x [v], from its latest sample
    in log order whose x is at most [v]: that sample's value, or, with
    [tau], the smoothed value of its step as {!make} smooths the samples up
    to it. A sample that is not finite is its own value, smoothed or not. It
    allocates a few words a sample. *)
