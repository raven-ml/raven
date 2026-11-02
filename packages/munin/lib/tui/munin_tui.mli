(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A terminal dashboard of a munin store.

    The dashboard shows a {!Munin.Store.t} in a terminal, reads it again once a
    second, and never writes it. It has two {e screens}. {e Runs} lists the
    store's runs, to find a run and compare runs by their params and metrics.
    {e Run} monitors one run, the {e focus}: its state, its metrics drawn over
    its steps, and its machine. A {e marked} run is drawn in every chart of Run
    beside the focus, so the focus reads against earlier runs of a sweep. Run's
    charts follow a {e metric cursor}, and Runs shows the cursor's metric, the
    {e focus metric}, in a column. Three views open over the screens: {e Zoom},
    one metric of Run at full size with a step cursor; {e Details}, every fact
    about one run; and {e Help}.

    Each screen's last line names its keys ({!section-keys}). The screens form a
    stack whose bottom is Runs. Enter opens the run under the cursor in Run, and
    Esc goes back one screen. A dashboard that starts on Run has Runs below it,
    so Esc from either start ends on Runs, and only [q] and Ctrl+C quit.

    The screens state each fact in words or glyphs, and colour repeats what they
    say, so a terminal without colour, or under [NO_COLOR], shows every fact.
    Text takes the terminal's own foreground and background, and other colours
    are the 16 colours of the terminal's palette.

    The dashboard reads a run as its library does ({!Munin.Store_format}).
    Unless a section says otherwise, a value shown is the run's as last read,
    and a time is the dashboard's clock against a time the store recorded. *)

(** {1:runs Runs}

    Runs has a row per run of the store, the newest started first, and a cursor
    on one row. Its first line counts the runs, the [live] ones, and those that
    need attention: the [failed] and [crashed] ones, and those with a metric
    sample that is not finite. A value that every row shows equal, such as the
    experiment of a filtered sweep or the host, is stated once on that line.

    The columns, left to right, are the run's state word ({!section-states}),
    its name or else its id, its experiment and its parent, its start time, the
    time from its start to its end or its last write, the age of its last write,
    its latest step with its progress while short of its [steps]
    ({!section-numbers}), the focus metric's value and the step of that value,
    the first sample of the focus metric that is not finite, as [NaN 4.5k], its
    host, its params and its commit. The value is the run's best sample by the
    metric's goal ({!Munin.History.best}), or its latest without a goal, and a
    [*] follows the best value of each experiment among the rows shown. The
    experiment, the parent, the host, each param and the commit are columns only
    when their values differ among the rows shown. Columns that do not fit are
    left out, in this order: the commit, the step of the value, the time from
    its start, the start time, the third and later params, the age, the second
    param, the first, the host, the parent, the experiment.

    The focus metric is chosen by the first listing whose rows have a metric:
    the metric with a goal that the most shown rows have, the first by name
    among equals. After that only [left] and [right] move it, through the
    metrics that any shown row has, by name, and a row without the focus metric
    shows [—] in its column.

    Below the rows, a terminal of 120 columns or more whose rows leave 14 lines
    shows a pane with the cursor run's details and its focus metric drawn over
    its steps; otherwise the line above the keys states the cursor run.

    A {e filter}, typed after [/], keeps the rows of which each word of the
    filter is part of the name, the id, the experiment, the parent, a tag, the
    state word or a param written [key=value], ignoring case. The rows are
    filtered as each key is typed. Enter keeps the filter and Esc clears it.

    The values come from each run's statistics ({!Munin.History.stats_of}),
    which are an ended run's hint, and for a run without an end a history the
    dashboard follows while Runs is shown. *)

(** {1:run Run}

    Run's first line states the focus in words ({!section-states}): its state
    word, its name, its progress, its rate, the time it has left, its active
    time, and the age of its last arrival or its last write ({!section-states}).
    The second line, shown only when something needs attention, starts with [!]
    and names a sample that is not finite in the focus or a marked run, a
    stalled or crashed run, a run whose lock is held and from which no metric
    sample has arrived in the 30 s since the dashboard began watching it, or
    damaged lines in a log. Then come the charts ({!section-charts}), and the
    readings of the run's machine and devices last.

    When the terminal has room for a chart per metric, three to a line at most,
    each at least 40 columns by 8 lines, with the system rows below them, every
    metric has a chart, the charts share one x domain, and the system rows add
    each reading drawn over the run's time. Otherwise Run draws one chart, the
    cursor's metric, above one row per metric with its latest value, a trend and
    its best value. A metric is a key of the run's metrics with a sample
    ({!Munin.History.metrics}).

    The system rows follow the USE method, with a row per resource that gives
    its utilisation, its saturation and its errors. The keys are those of
    [Munin_sys]. munin logs no errors, which the rows state once. A device's row
    prints its [used] of its [memory], the device's whole memory, apart from the
    [allocated] bytes of the run against its [budget]. *)

(** {1:views Zoom, Details and Help}

    Zoom draws the cursor's metric at the terminal's size with a step cursor on
    the focus's samples, which [left] and [right] move one sample, and a table
    of each run's latest value, best value and its step, value at the cursor,
    and counts of samples, non-finite samples and orphans. Below 120 columns the
    table keeps each run's value at the cursor and its best value. Details lists
    the run's start and end, its source, its params, marking those that differ
    from a marked run's with the marked run's value, its metrics' latest and
    best values, its notes and tags, the artifact versions it used, the files it
    logged, its resumes and its damaged lines. It scrolls. Help lists the keys
    and the words and glyphs of the screens. Below 40 columns or 10 lines every
    screen is one line, which holds the focus's state word, name and progress,
    or the counts of Runs. *)

(** {1:charts Charts}

    A chart draws the focus's samples of a metric as a line in braille dots,
    over a domain of x and of y, and each marked run's as a level line.

    {b x.} A chart's x is the step, or with [x] the metric's step metric, or the
    time. A sample on the step metric's axis takes the value of the step
    metric's sample at the same step ({!Munin.Store_format.section-reading}),
    and a sample without one is not drawn; the title counts them. The time is
    the run's active time, the time from its start less the time between a
    writer's last record and the next writer's resume. The x domain runs from
    [0] to the focus's [steps] on the step axis of a run that declared them and
    has no sample past them, and otherwise to the least of [1], [2] or [5] times
    a power of ten at or above 1.1 times the largest x, so that it changes a few
    times over a run and holds every sample.

    {b Reduction.} The samples of a {e dot column}, the samples whose x falls in
    that column of the plot, are reduced to the first, the least, the greatest
    and the last of its finite samples, joined in order. The line through them
    inks the dots that the line through every sample inks.

    {b Non-finite samples.} A sample that is NaN or infinite takes no part in a
    domain or in smoothing, and the line breaks across it. The row above the
    plot marks each dot column that holds one of the focus's with [×] for NaN,
    [↑] for an infinity and [↓] for a negative one, [×] winning. A marked run's
    are named in the chart's title and on the attention line.

    {b Orphans and resumes.} A sample that a resume orphans is never drawn
    ({!Munin.History.orphans}). Each resume is a dotted vertical line at its
    step. Where a run's steps decrease without a resume, its line breaks, and
    smoothing starts again.

    {b y.} The y domain is the extent of the finite values drawn in the x
    window, the focus's and the marked runs'. Nothing is trimmed, so a spike
    sets the domain; a narrower x window ([+]) refits y to the samples in it.
    With [L] a chart whose finite values in the window are all above [0] draws y
    on a log scale, and another says [linear: values <= 0]. The ticks are at
    [1], [2], [2.5] or [5] times a power of ten, and a chart too short for two
    ticks prints its range in its title.

    {b Smoothing}, off by default, draws a smoothed line over the raw one, which
    stays drawn, dimmed. The smoothed value at a step [x_i] is the debiased mean
    of the finite samples at and before it,
    [s_i = y_1 + Σ_j w_j (y_j - y_1) / Σ_j w_j], with
    [w_j = (1 - exp (-(x_j - x_(j-1)) / τ)) * exp (-(x_i - x_j) / τ)], where
    [x_0 = x_1 - (x_2 - x_1)], or [x_1 - 1] for a lone sample. The samples that
    share a step count as one sample, their mean, at that step. [s] cycles [τ]
    through a hundredth, a thirtieth and a tenth of the shown x span, rounded to
    [1], [2] or [5] times a power of ten, and the legend prints it. A latest and
    a best value are always the raw samples'.

    {b Marked runs.} A marked run's {e level line} joins, per dot column, the
    mean of its finite values in that column, smoothed when smoothing is on. It
    is dotted, drawn before the focus so that the focus loses no dot, and
    carries its letter, [a] or [b], at its end and in cells along it that the
    focus does not ink. Where the two lines share a cell the focus's dots
    replace the marked run's, so a level line shows the level of a marked run
    and promises none of its extremes. Its points are means over a dot column,
    so a narrower chart averages more samples per point; a marked run's values
    are read from Zoom's table. At most two runs are marked. A third level line
    shares the cells of the other two at braille resolution, and without colour
    no reader can tell it from them. *)

(** {1:states States}

    A run's {e state word} comes from its status as read ({!Munin.Run.status})
    and from when its samples arrived:
    - [starting]: a process holds the run's lock and the run has no metric
      sample;
    - [running]: a process holds the lock and no metric sample has arrived since
      the dashboard began watching the run;
    - [live]: a process holds the lock and a metric sample has arrived, within
      three usual gaps once they are known;
    - [stalled]: a process holds the lock and no metric sample has arrived for
      longer than three usual gaps;
    - [writing]: the lock is not probed and a metric sample has arrived, within
      three usual gaps once they are known;
    - [silent]: the lock is not probed, and no metric sample has arrived since
      watching began, or for longer than three usual gaps;
    - [crashed], [finished], [failed] and [killed]: the run's status.

    The dashboard times each arrival on its own clock. A poll that finds new
    metric samples of a run is an arrival at that poll's time, and a run's
    samples that the first read finds are none. The {e usual gap} is the median
    of the gaps between the run's last 32 arrivals, known once two arrivals have
    been timed, and three of them are never less than 10 s. No time the run
    recorded judges its state. Run's first line gives the age of the last
    arrival, or before one, the age of the last write ({!Munin.Run.updated_at})
    labelled as such and how long the dashboard has watched. A run that is
    [writing] or [silent] may have exited, which its line says by
    [lock not probed]. *)

(** {1:numbers Numbers}

    - A value prints with the fewest significant digits from 4 to 17 that print
      differently every two values shown together that differ, as a column or a
      chart's latest and best. NaN prints [NaN], and the infinities [inf] and
      [-inf].
    - A count prints with separators, as [12,400], or in a narrow cell with
      three significant digits, as [12.4k].
    - A duration prints in at most two units, as [45s], [1h37m] or [3d4h], and
      an age in a narrow cell in one, as [2h].
    - Memory prints in GiB and a rate of bytes in MB/s.
    - The {e latest step} is the largest step of the latest samples of the run's
      metrics. Progress is the latest step over the run's [steps]
      ({!Munin.Run.steps}), and absent without them.
    - The {e rate} is the steps per second over the last 10 minutes, or over the
      last tenth of the active time when that is longer, once that window holds
      60 s of samples. The {e time left} is the steps left over the rate, shown
      as [~59m left] while the run is [live] or [writing].
    - Times of day are local, as [10:41] today and [Oct 31 16:20] before. *)

(** {1:keys Keys}

    A key means one thing on every screen, and a key whose object the screen
    lacks does nothing. Keys are matched with their case.

    {v
    up down, k j     move the cursor: run, metric, line
    left right, h l  move sideways: focus metric, metric, step cursor
    g G              first, last
    Enter            open: the run in Run, the metric in Zoom
    Esc              back one screen; in the filter, clear it; on Runs,
                     clear a kept filter; close Help
    q, Ctrl+C        quit, except in the filter, where q is typed
    ?                Help
    /                filter the rows of Runs, the metrics of Run
    Space            mark or unmark the run: the cursor's on Runs, the
                     focus elsewhere
    [ ]              focus the previous, next run of Runs's order
    o O              order Runs by the next column, reverse the order
    s                smoothing: off, then three time constants
    x                x axis: step, step metric, time
    L                log y
    + - 0            narrow, widen, reset the x window, at the latest step
    < >              pan the x window
    i                Details of the run
    y                copy the run's id to the clipboard
    v}

    [y] sends the terminal its sequence to set the clipboard (OSC 52). A
    terminal may ignore it, which the dashboard cannot see.

    The dashboard leaves the mouse to the terminal, so the terminal selects text
    as it does for any program. *)

(** {1:running Running} *)

(** The type for the screen a dashboard starts on. *)
type screen =
  | Runs  (** The store's runs. *)
  | Run of string option
      (** The run of this id, or with [None] the newest run without an end that
          has not crashed, else the newest run, else the first run to start. *)

val run : Munin.Store.t -> screen -> (unit, string) result
(** [run store screen] shows the dashboard of [store], starting on [screen], on
    the process's terminal until the user quits. On Runs a poll reads the
    store's listing as {!Munin.Store.runs} reads it with the previous listing,
    and follows the history of each run without an end, with statistics alone,
    and the cursor's run's focus metric while its pane shows. On Run it follows
    the focus's history with every key, and the marked runs' with the keys its
    charts show. A key kept costs 24 bytes a sample. When the cursor leaves an
    ended run whose pane's chart was drawn from the run's end, that chart is
    kept and its history is not: the run's log no longer changes, so coming back
    reads nothing, and the kept charts cost a reduction of the pane's width per
    row. A change of the focus metric reads them again. A history is held while
    a screen of the stack needs its run, and charts are reduced as polls read,
    so a key and a frame cost what the screen shows.

    [Error why] if standard input or output is not a terminal, or no run of
    [store] has the id given. *)

(**/**)

(* What the tests drive. [app ~clock store screen] is the program [run] runs,
   which reads the wall clock from [clock], for Mosaic_test's emulated terminal.
   [Curve] reduces a metric's series to a plot's dot columns, as the charts draw
   them: the laws of the Charts section, for windtrap. [Chart.draw] draws a
   chart as Run draws it, with any number of marked runs, for the test that
   shows how many the medium carries, and [Chart.paired] and [Chart.domain] give
   the x of the charts' laws. [Watch] states the state words without a store.
   [key], [resize] and [frame] let the bench drive the program by its
   messages. *)

type model
type msg

val app :
  clock:(unit -> float) -> Munin.Store.t -> screen -> (model, msg) Mosaic.app

module Curve : sig
  type event = [ `Nan | `Pos_inf | `Neg_inf ]

  type plot = { lo : float; hi : float; dots : int }
  (** The x domain shown and the dot columns of the plot. A sample at [x] in
      [[lo;hi]] is in the dot column in which {!Matrix_charts} draws [x] on a
      plot of [dots] dot columns over that domain. *)

  type t

  val reduce :
    ?tau:float -> plot -> x:(int -> float) -> Munin.Metric.series -> t
  (** [reduce plot ~x s] is [s]'s samples at x [x i], reduced to [plot]'s dot
      columns, smoothed with [tau] when given. A sample whose [x] is NaN has no
      x and is left out. *)

  val extend : t -> x:(int -> float) -> Munin.Metric.series -> t
  (** [extend c ~x s] is [reduce] of [s] with [c]'s plot and [tau], at the cost
      of [s]'s samples past those [c] reduced. [s] extends the series [c]
      reduced, and [x] gives the samples [c] reduced their x. *)

  val points : t -> (float * float) array
  (** [points c] is the line's points in order, a NaN [y] where it breaks. *)

  val level : t -> (float * float) array
  (** [level c] is the level line's points, one per dot column that holds a
      finite sample. *)

  val events : t -> (int * event) array
  (** [events c] is each dot column that holds a sample that is not finite, with
      its mark, by column. *)
end

module Chart : sig
  val draw :
    width:int ->
    height:int ->
    lo:float ->
    hi:float ->
    key:string ->
    Munin.History.t ->
    (char * Munin.History.t) list ->
    Matrix_grid.t
  (** [draw ~width ~height ~lo ~hi ~key focus marked] is the chart of [key] that
      Run draws for the focus [focus] and the marked runs [marked] with their
      letters, over the steps [lo] to [hi], in a grid of [width] by [height]
      cells. *)

  val paired : Munin.History.t -> string -> int -> float
  (** [paired h key i] is the x of the sample [i] of [key]'s series on its step
      metric's axis, NaN when it has none. *)

  val domain : Munin.History.t -> string -> float * float
  (** [domain h key] is the x domain of [key] on the step axis. *)
end

(** The state words from the arrivals a dashboard timed. *)
module Watch : sig
  type t

  val start : now:float -> samples:int -> t
  (** [start ~now ~samples] is a watch that began at [now] on a run whose log
      held [samples] metric samples, which are no arrival. *)

  val observe : t -> now:float -> samples:int -> t
  (** [observe w ~now ~samples] is [w] after a poll at [now] read [samples]: an
      arrival when they are more than before. *)

  val word : Munin.Run.status -> t option -> now:float -> string
  (** [word status w ~now] is the state word of a run of [status]. *)
end

val key : Matrix.Input.Keystroke.t -> msg
val resize : width:int -> height:int -> msg

val frame : model -> Matrix_grid.t
(** [frame m] is the screen [m] draws, at its size: what the bench times after a
    key. *)

(**/**)
