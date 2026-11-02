# The dashboard

`munin` without a command, on a terminal, shows the store's runs.
`munin watch` shows one run as it trains. The two are one program, the
dashboard of [`munin_tui.mli`](../lib/tui/munin_tui.mli), which reads the
store again once a second and never writes it. This page walks through a
session over a sweep. Every rule it relies on is stated in that `.mli`.

<!-- $MDX skip -->
```sh
munin                 # the store's runs
munin watch           # the newest run that is still going
munin watch RUN       # one run
munin --root DIR      # another store
```

Without a terminal, `munin` prints its help and `munin watch` fails, so
a script that calls them by mistake stops.

To try it, run the example that writes a sweep into a temporary store,
starts a run that trains while you watch, and opens the dashboard on
that store; `q` quits, ends the run and removes the store:

<!-- $MDX skip -->
```sh
dune exec packages/munin/examples/08-dashboard/main.exe
```

## Runs

A sweep of six learning rates over a ResNet, one still training, and two
older runs of an MLP:

<!-- $MDX skip -->
```text
8 runs · ● 1 live · ✗ 1 failed · ✗ 1 crashed · × 1 not finite · on gpu-01

   state       run                step  val/loss ↓  not finite          lr
 cifar-resnet  6 runs · ● 1 live · ✗ 1 failed or crashed
   ● live      tidy-otter  12,400  62%     0.4077                   0.0003
   ✓ finished  amber-fox        20,000     0.3841*                   0.001
   ✓ finished  brisk-wren       20,000     0.4071                   0.0001
   ✓ finished  calm-lynx        20,000     0.3942                    0.003
   ✓ finished  dusty-mole       20,000     0.7665   × NaN at 4,250    0.01
   ✗ failed    eager-hawk        1,100      1.670                     0.03
 mnist-mlp  2 runs · ✗ 1 failed or crashed
   ✓ finished  mlp-wide          4,690    0.05831*                     0.1
   ✗ crashed   mlp-base     2,900  62%    0.06576                      0.1
```

The first line answers the morning's question: how many runs, which
are live, and which need a look, each count matching the rows that show
its glyph. Each experiment's runs sit under a header row that counts
them. The state word comes first on each row, after a glyph, since it is
what a glance looks for, and it is a word, so the screen reads the same
without colour; finished runs are dim, so the eye skips them. `crashed`
means the run's writer exited without an end; it is never a guess (the
States section of the `.mli`).

The metric column shows each run's best value by the metric's goal, here
the least validation loss, or its latest value for a metric without a
goal. A star marks the best of each experiment, under its header: a loss
on CIFAR-10 is not compared with one on MNIST. `not finite` gives the
step where a metric of the run first went NaN or infinite, so the run
that diverged is named with the step to look at. `left` and `right` move
the column through the metrics any shown row has. Every value prints with
four significant digits, and every step with separators.

The columns after the metric are the params whose values differ among
the rows shown, the param a sweep varies first. In the lines the table
leaves, the cursor's run is drawn: its params, its metric's latest and
best values, and the metric over its steps. Typing `/cifar` keeps the
sweep:

<!-- $MDX skip -->
```text
6 of 8 runs · ● 1 live · ✗ 1 failed · × 1 not finite · cifar-resnet
```

The experiment, being the same in every row, moves to the first line.
A word of the filter matches a name, an id, a tag, a state word or a
param written `key=value`, so `/warmup=2000 finished` keeps the finished
runs of that warmup. Enter keeps the filter, and Esc clears it.

## Watching a run

Enter opens the run under the cursor. `space` marks runs to compare
against first; here `amber-fox` is marked `a` and `dusty-mole` `b`:

<!-- $MDX skip -->
```text
● live  tidy-otter   12,400/20,000 62% · ~59m left · last sample 13s ago
! b dusty-mole: train/acc NaN from step 4,250 (and 7 other metrics)
val/loss ↓   0.4077   best 0.4077 at 12,000
━ tidy-otter   a amber-fox  lr 0.001   b dusty-mole  lr 0.01

            2.002 ┤⢣⡀
                  │ ⢣⡀
                  │  ⢣
                  │   ⠱⡀
                1 ┤    ⠈⠑⢄  b
                  │       ⠉⠢⣀⡀
           0.3841 ┤          ⠈⠉⠑⠒⠢⠤⠤⠤⠤⣀⣀⣀⣀⣀⣀⣀⡀⢀⢀a⢀⢀ ⡀⡀⡀⡀⡀⡀⡀⡀⡀⡀⡀a
                  └┬─────────────────────┬────────────────────┬
                   0                   10,000               20,000
  a amber-fox      ⡀⢀⠤⠔⠢⢄⠔⠤⠒⠒⠒⠊⠑⠒⠒⠒⠒⠉⠉⠉⠒⠊⠉⠉⠉⠑⠂┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄       +0.01379
    focus − a      ⠈⠁                                                    worse
  b dusty-mole     ⡀┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄       -0.04625
    focus − b      ⠈⠒⠢⠤⠤⣀⣀⣀⣀ b NaN from step 4,500                      better
```

The first line states the run in words. `live` means a process holds the
run's lock and the dashboard has seen a sample arrive. When the
dashboard has just opened, it has seen none, and the run reads `running`
with how long it has watched; a run that logs nothing for three of its
usual gaps reads `stalled`. These words come from what the dashboard
observed on its own clock, so a program that logs samples with times of
its own, or a writer on another machine with another clock, cannot make
a hung run look alive.

The second line, drawn only when something needs attention, names the
run, the metric and the step, a marked run's included. One chart takes
most of the screen: the focus as a solid line, each marked run as a
level line with its letter at its end and along it, a dot after the
latest sample. The y axis is labelled at the least and the greatest
value drawn, so the region the run is in always has a number. The x
axis runs to the run's declared `steps`, so the blank right part of the
chart is the work left. The line is drawn from every sample: each column
of braille dots shows the least and the greatest value it covers, so a
spike between two columns of a long run stays on screen. A NaN or an
infinity is a mark in the row above the plot, and a metric that stays
NaN reads as one span.

Where the focus runs over a marked run, the overlay cannot show both, so
under the chart each marked run has two rows: the focus minus that run
at the same step, around a zero rule, with the latest difference and
whether the focus is `worse` or `better` by the metric's goal. A marked
run that stops early, as `dusty-mole` does at its first NaN, says where.

Under them every metric is a strip, drawn over the same x as the chart
and in its columns, so reading down a column reads one step across every
metric. Strips are grouped by the prefix of their keys, `train/` and
`val/`, and `up` and `down` move the chart to the next metric.

Last come the machine's readings, one row per resource, and on wide
terminals each reading drawn over the run's time:

<!-- $MDX skip -->
```text
machine gpu-01 · read every 15s · errors not logged
CPU       71% busy · load 1.3 per core · process 9.6 cores
memory    33.0 of 62.7 GiB · swap 0.0 GiB · process 18.2 GiB
CUDA:0    97% busy · 71 °C · 274 W
          used 21.3 of 24.0 GiB · allocated 20.9 of 22.0 GiB budget
disk      read 14 MB/s · write 2.8 MB/s · 5.5% busy
```

A device's `used` of its `memory` is every process's, and the run's
`allocated` against its runtime's `budget` is the run's alone, so a full
device with a small allocation points at another process. For a run that
is no longer live, the first row says when the readings were last taken.

Enter zooms the chart, with a step cursor whose values the table below
the chart prints for each run. `s` smooths the line with a debiased mean
whose time constant the title prints, `x` changes the x axis to the
metric's step metric or to time, `L` draws y on a log scale where every
value is above 0, and `+` and `-` narrow and widen the x window at the
latest step; a setting away from its default is written at the right of
the title. Esc goes back, and `q` quits from anywhere.

## Keys

`?` lists the keys of every screen and of the screen it opens over, and
the words and glyphs the screens print. One key means one thing
on every screen; the table is the Keys section of
[`munin_tui.mli`](../lib/tui/munin_tui.mli). The dashboard leaves the
mouse to the terminal, so text on screen can be selected and copied as
in any program, and `y` copies the run's id.
