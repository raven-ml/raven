(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What a chart draws and over which steps.

    {b The x domain} of a chart is the step, from [0] to the greatest end
    of the runs drawn ({!Facts.run.x_end}): a run's declared steps, or, when
    none are declared or a step passes them, its largest step rounded up to
    1, 2 or 5 times a power of ten. Every chart and strip on screen shares
    it, so a reduction made for it stays valid while the runs log within
    it. At the zoom depth the chart shows a window of it: the domain halved
    [n] times, the part of that size that holds the zoom's cursor. *)

type role =
  | Context  (** A run of the cursor's experiment, in grey. *)
  | Overlay of char  (** An overlay, with its letter. *)
  | Cursor  (** The cursor's run, drawn last. *)
(** The type for what a run is to a chart. *)

val smoothable : string -> bool
(** [smoothable key] is [true] iff [key] is smoothed by default: it starts
    with [train/]. *)

val domain : Facts.run list -> float * float
(** [domain runs] is [(0., e)], [e] the greatest {!Facts.run.x_end} of
    [runs], and [1.] for no run. *)

val tau : float * float -> float
(** [tau (lo, hi)] is smoothing's time constant over the x domain
    [(lo, hi)]: a thirtieth of its width. Every run a chart draws shares
    it, whatever its cadence or length. *)

val window : float * float -> at:float -> narrow:int -> float * float
(** [window (lo, hi) ~at ~narrow] is the part of [(lo, hi)] that holds
    [at], of a [2{^narrow}]th of its width, aligned to that width from
    [lo]: moving [at] inside it keeps it. [at] outside [(lo, hi)] is taken
    at the nearer end. *)

val log : Facts.run list -> string -> bool
(** [log runs key] is [true] iff every finite sample of [key] in [runs] is
    above [0.] and their greatest is at least a hundred times their least. *)

val y_ticks : log:bool -> most:int -> float * float -> float list
(** [y_ticks ~log ~most (lo, hi)] is the y ticks of the domain [(lo, hi)],
    each the float nearest its decimal value: on a log scale its powers of
    ten, every [k]th when they are more than [most]; else the multiples in
    it of the finest step of 1, 2 or 5 times a power of ten that gives at
    most [most]. Only ticks {!Numbers.tick} labels are taken, so a step too
    fine for them gives way to a coarser one, and a domain no labelled
    value falls in has none. *)

(** {1:charts Charts} *)

type guides = [ `Both | `Y | `None ]
(** The type for a chart's axes: both, the y axis alone, or none. *)

type line = {
  role : role;
  colour : int;  (** The run's colour. *)
  reductions : Reduce.t list;
      (** The run's series reduced over the chart's x domain, at any dot
          counts; the one made on the plot's dot columns is drawn. *)
}
(** The type for a run a chart draws. *)

type t = {
  x : float * float;  (** The x domain, [lo < hi]. *)
  log : bool;  (** A log y scale, where every value drawn is above [0.]. *)
  smooth : bool;  (** Draw the smoothed means. *)
  guides : guides;
  now : float option;  (** A live run's latest step. *)
  cursor : float option;  (** The zoom's step. *)
  extents : bool;  (** Draw each column's raw extent behind the line. *)
  fill : bool;  (** Fill each line down to the plot's floor. *)
  lines : line list;  (** In the order they are drawn, later over earlier. *)
}
(** The type for charts. *)

val dots :
  Theme.t -> t -> width:int -> height:int -> (int * (float -> int)) option
(** [dots th c ~width ~height] is the number of dot columns of [c]'s plot
    drawn in a region of [width] by [height] cells, and the dot column in
    which it draws an x of its domain; [None] if the region leaves the plot
    no column. They depend on [c]'s domain, scale and guides, never on its
    lines: a reduction made on them before the lines are known is drawn
    where it was binned. *)

val draw : Theme.t -> t -> Matrix_grid.t -> unit
(** [draw th c g] draws [c] over the whole of [g]:

    - The line of each run whose reduction was made on [c]'s dot columns
      ({!dots} at [g]'s size), through each column's mean, raw or smoothed
      as [c.smooth] says, geometric on a log scale, to the nearest samples
      on each side of the plot. A run without such a reduction draws no
      line. [Context] runs are faint, the others in their colour. With
      [c.fill], the region under each line to the plot's floor.
    - The y domain covers every raw sample of those lines in the plot, a
      twentieth wider each way; on a log scale from the power of ten below
      when it is near. A log scale with a value at or below [0.] is
      linear.
    - Each extreme raw sample of the cursor's run further than three dot
      rows from its line, smoothed or not, as a faint dot; with [c.extents],
      each column's raw extent of the cursor's run, faint, in place of
      them.
    - On a chart with axes, from the first column of the cursor's run with
      a sample that is not finite to the plot's right edge, a faint hatch,
      labelled.
    - A dotted rule at [c.now] labelled [now], and a rule at [c.cursor]
      whose step the x axis labels.
    - Without colour, each overlay's letter one cell right of its last
      point in the plot, or on it at the plot's edge.
    - A label starts one cell right of its point, moved inside the plot
      where the plot's edge would cut it.
    - y ticks about three rows apart or more, at the multiples of
      the finest step of 1, 2 or 5 times a power of ten whose labels give
      their values exactly in six columns ({!Numbers.tick}), or at powers
      of ten on a log scale; a domain where no such value falls has no y
      label. x labels compact, [step] at the end. *)
