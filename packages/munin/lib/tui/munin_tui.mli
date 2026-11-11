(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A terminal dashboard of a munin store.

    The dashboard shows a {!Munin.Store.t} in a terminal, reads it again once a
    second, and never writes it. It is one screen: a bar at the top, three
    panes, and a status bar at the bottom. The {e runs pane} at the left lists
    the store's runs under their experiments, and its cursor picks the run the
    rest shows. The {e centre} shows that run's metrics at one of three
    {e depths}: charts, strips and zoom. The {e run pane} at the right states
    the run: its state, its progress, its params, its machine and its source.
    Up to four {e overlays}, runs the user picked, are drawn beside the
    cursor's run in every chart, each in a colour it keeps for the session.

    The status bar names the depth and the keys that act at it
    ({!section-keys}); where they do not fit, they leave from the last, and
    [? keys], which lists the rest, stays. Colour repeats what a glyph or a word says, so a terminal
    without colour, or [NO_COLOR], shows every fact: bold and faint remain, the
    cursor's row is reverse video, and each overlay's line ends in its letter.
    The colours follow the terminal's background, dark or light.

    A text cut to its column ends in […] and keeps a blank before the next
    column, and a number is never cut: a run's name, a param's key or a
    command gives way first. The run pane prints a param's number as values
    print, and [i] gives it as logged. Param keys take the longest key's
    width, up to half the run pane, and a strip's key up to a quarter of the
    centre. The line of values keeps the values that fit whole. The banner
    gives an error's first line, with […] when it has more, and [i] the
    rest; both wrap at their words, and [i] keeps a command's arguments
    whole.

    The dashboard reads a run as its library does ({!Munin.Store_format}).
    Unless a section says otherwise, a value shown is the run's as last read,
    and a time is the dashboard's clock against a time the store recorded. *)

(** {1:screen The screen}

    The top bar counts the runs and experiments, the runs [live], the runs
    [failed] or [crashed], and those with a metric sample that is not finite,
    and names the host when every run shares one.

    The panes take their widths from the terminal's: at 150 columns or more
    the runs pane has 42 and the run pane 34; at 100 or more, 34 and 28, the
    run pane hidden at the strips and zoom depths; below 100, neither, and a
    line under the top bar names the cursor's run. [1] and [3] show or hide
    either pane at any width. Below 60 columns or 16 lines the screen is one
    line per live run and asks for a larger terminal. A store without runs
    shows how a program starts one.

    {b The runs pane} has a header per experiment, open or folded to one line
    that counts its runs and its failed or live ones, and a line per run of an
    open experiment: its name, its state glyph ({!section-states}), its
    experiment's key metric drawn over its steps, and its best value of it,
    with [★] on the best of its experiment among the runs shown. Runs that
    share a name among those listed or drawn each follow it with the end of
    their id, the fewest of its last characters, four at the least, that
    tell them apart, as [cnn-adam 130c], wherever a name is shown. Beyond 30
    runs, the experiments start folded but the cursor's. The cursor stops at
    each run of an open experiment and at the header of a folded one. Runs
    are newest first, or by their best key metric, or by name ([o]).

    A {e filter}, typed after [/], keeps the runs of which each word matches
    the name, the id, the experiment, a tag, the state word or a param written
    [key=value], ignoring case, or, for a word [key<v] or [key>v] with [v] a
    decimal such as [0.002] or [2e-3], whose param [key] is a number below or
    above [v].

    The {e key metric} of an experiment is, among its runs' metrics by key, the
    first with a goal outside [train/] whose key ends in [loss], else the first
    with a goal outside [train/], else the first with a goal, else the first.

    {b The centre} starts with a {e banner} when the cursor's run failed,
    crashed or logged a sample that is not finite, then the {e hero}: the key
    metric's latest value in large digits, its best, and the latest values of
    up to three other metrics with a goal, four from 150 columns, smoothed
    where their charts smooth them. Below 100 columns or 30 lines one line of
    those values takes the hero's place.

    At the {e charts} depth the centre has the metric tabs, which [left] and
    [right] move, a legend that gives each drawn run's value at the cursor's
    run's latest step, one large chart and, from 100 columns and 30 lines, two
    small charts of the other metrics with a goal, three from 150 columns and
    44 lines. A run without a metric sample shows, in their place, [no
    metric sample yet], or, once it ended, [the run logged no metric]. At
    the {e strips} depth each metric of the run is a strip two
    lines high on the same steps, with its latest value and, with an overlay,
    the difference to the first overlay at the latest step at which both
    logged a finite value; while the run is live, strips of its machine end
    the list, over its active time. At the {e zoom} depth one metric fills the
    centre with each column's raw extent faint behind the smoothed line, a
    step cursor and a readout of each drawn run at the cursor. Enter goes one
    depth deeper and Esc one back.

    [i] shows everything about the run over the centre: its params, notes,
    tags, source, every metric's latest and best values, the artifact versions
    it used, its files, its resumes and its damaged lines. [?] shows the
    keys. *)

(** {1:charts Charts}

    A chart draws a metric of the cursor's run over the step, and each overlay
    beside it.

    + The x axis is the step, from [0] to the greatest end of the runs drawn.
      A run that has ended, crashed or gone silent ends at its largest step,
      so that one that stopped at step 2,812 of 3,000 is drawn on [0] to
      2,812. A run that is starting, running, live, stalled or writing ends
      at the least of 1, 2 or 5 times a power of ten at or above its latest
      step, at most its [steps] while it is within them, so that at step 80
      of 3,000 it is drawn on [0] to 100 and the axis changes at most three
      times a decade as it grows, a pause included. The run pane says how
      many steps are left. A machine's strip is over the run's active time, to
      the same rounding of it. Zoom shows a window of the domain: halved by
      each [+], the part that holds its cursor.
    + A line is one value per braille dot column, the mean of the column's
      samples, joined to the next column's.
    + A column's least and greatest raw samples of the cursor's run, where
      further than three dot rows from its line, smoothed or not, are faint
      dots, and the y range covers the raw samples, so that no spike hides.
    + A metric under [train/] is smoothed by default, with a debiased
      exponential mean over the steps whose time constant is a thirtieth of
      the steps shown, so that runs logging at different cadences are
      smoothed alike; [s] turns smoothing on and off for every chart.
    + y is on a log scale when every value drawn is above 0 and they span two
      decades or more, with ticks at the powers of ten; else linear with round
      ticks. [y] asks for the other scale, per metric.
    + On a chart with axes, from the cursor's run's first sample that is not
      finite, the plot is hatched and labelled [× NaN], [× inf] or [× -inf].
    + The other runs of the experiment ([r]) are drawn first, in grey, then the
      overlays, then the cursor's run, its spikes and raw extent with it. Runs
      that share a cell all show there, in the colour of the run drawn later.
      A rule shows where no dots are; at zoom, the chip on the x axis marks
      the cursor's step.
    + Without colour each overlay's line ends in its letter, [a] to [d].
    + While the run is live a faint rule marks its latest step, [now], on a
      chart with axes.
    + A machine's reading is filled down to the plot's floor.
    + The large chart and zoom have both axes, a small chart the y axis
      alone, a strip and a run's trend none. A y label is a value that its
      digits give exactly in at most six columns, at a multiple of 1, 2 or 5
      times a power of ten, the labels three rows apart or more; a y range
      where no such value falls has no label. An axis's labels share one
      notation: plain, as [0.25] or [12500], if each fits; else with the
      suffix [k], [M] or [G], as [1.5M]; else with an exponent, as [2e-05].
      x labels are compact steps, and [step] ends the axis, except where
      zoom's step cursor meets it.

    A chart is drawn from a reduction of each series to its dot columns,
    made off the loop by the poll for the chart's x domain and size, so that
    drawing costs what the chart shows, whatever the series' length. A chart
    whose reduction is not made yet, as after a resize, draws its axes
    alone. *)

(** {1:states States}

    A run's {e state word} comes from its status as read
    ({!Munin.Run.val-status}) and from when its samples arrived:
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

    Its glyph is [●] for [live] and [writing], [○] for [running] and
    [starting], [◌] for [stalled] and [silent], [✓] [finished], [■] [killed],
    [✗] [failed] and [crashed], and [×] for a run, not failed, with a metric
    sample that is not finite.

    The dashboard times each arrival on its own clock. A poll that finds new
    metric samples of a run is an arrival at that poll's time, and a run's
    samples that the first read finds are none. The {e usual gap} is the median
    of the gaps between the run's last 32 arrivals, known once two arrivals have
    been timed, and three of them are never less than 10 s. No time the run
    recorded judges its state. *)

(** {1:numbers Numbers}

    - A whole value of six digits or fewer prints in full, as [348513]. Any
      other value prints with four significant digits, its trailing zeros
      kept, as [0.4077] or [6.200], with the suffix [k] from a thousand up, as
      [2.196k], and in scientific notation below [1e-4] and from [1e6] in
      magnitude: a value that does not read back as logged shows a decimal
      point, a [k] or an exponent. An accuracy, a metric with [acc] in its
      key and a value in [[0;1]], prints as a percentage with two decimals.
      NaN prints [NaN], and the infinities [inf] and [-inf].
    - A step or a count prints with separators, as [12,400], and compactly on
      an axis, as [12k].
    - A difference prints relative, as [-6.0%], or in points for an accuracy,
      as [+0.15 pt], and [same] within the digits shown.
    - A duration prints in two units, as [4m 10s]; a time of day as [14:22].
    - Progress is the latest step over the run's [steps], rounded down, so it
      reads [100%] only once the steps are reached. The rate is the steps per
      second over the last 10 minutes, and the time left the steps left over
      the rate, while the run is live. *)

(** {1:keys Keys}

    A key means one thing at every depth, and a key whose object is missing
    does nothing. Arrows and [h j k l] alike.

    {v
    up down, k j     the run; the strip at strips
    left right, h l  the metric; the step at zoom, shift for ten
    [ ]              the previous, next run, at any depth
    g G              the first, last run
    Enter            one depth deeper: charts, strips, zoom; open a folded
                     experiment
    Esc              one depth back; at charts, clear the overlays, then the
                     filter, then the grey runs; close the run's facts
    Space            overlay the cursor's run, or take it off (four at most)
    r                the experiment's other runs, in grey
    c                clear the overlays
    s                smoothing on, off
    y                the other y scale for this metric
    + -              narrow, widen the steps shown, about zoom's cursor
    /                filter the runs
    o                order the runs: started, best, name
    e                fold or open the cursor's experiment
    i                everything about the run
    1 3              the runs pane, the run pane
    ?                these keys
    q, Ctrl+C        quit, except in the filter, where q is typed
    v}

    The dashboard leaves the mouse to the terminal, so the terminal selects
    text as it does for any program. *)

(** {1:running Running} *)

(** The type for where a dashboard starts. *)
type screen =
  | Runs  (** The cursor on the newest run. *)
  | Run of string option
      (** The cursor on the run of this id, or with [None] on the newest run
          without an end that has not crashed, waiting for one to start. *)

val run : Munin.Store.t -> screen -> (unit, string) result
(** [run store screen] shows the dashboard of [store], starting on [screen], on
    the process's terminal until the user quits. Once a second a poll lists
    the store as {!Munin.Store.val-runs} lists it with the previous listing,
    follows the history of the cursor's run while it has no end, or while
    [i] shows it, with every key, each other run drawn with the keys drawn,
    and each run without an end with statistics alone. A key or a resize
    that changes what the screen draws starts a poll that reads only what
    it lacks. Only one poll runs at a time. An ended run is never followed
    and its charts outlive the poll, so that coming back to it reads nothing
    of it. A key kept costs 24 bytes a sample.

    [Error why] if standard input or output is not a terminal, or no run of
    [store] has the id given. *)

(**/**)

(* What the tests drive. [app ~clock store screen] is the program [run]
   runs, which reads the wall clock from [clock], for Mosaic_test's emulated
   terminal. *)

type model
type msg

val app :
  clock:(unit -> float) -> Munin.Store.t -> screen -> (model, msg) Mosaic.app

(* The reduction the charts draw, the drawing of a chart, the numbers it
   prints, the facts its domain comes from, the screen's tokens and the watch
   the state words come from, whose laws the tests state over inputs no
   screen reaches. A reduction's arithmetic is stated on [Reduce], since a
   drawing shows it only to a dot. *)

module Reduce = Reduce
module Plot = Plot
module Numbers = Numbers
module Facts = Facts
module Theme = Theme
module Watch = Watch

(**/**)
