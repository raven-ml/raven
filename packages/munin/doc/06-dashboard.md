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

## Runs

A sweep of six learning rates over a ResNet, one still training, and two
older runs of an MLP:

<!-- $MDX skip -->
```text
munin  8 runs: 1 live, 1 failed, 1 crashed, 1 not finite · on gpu-01

  STATE     RUN         EXPERIMENT               STEP  val/loss↓  NONFINITE
▸ live      tidy-otter  cifar-resnet   12.4k/20k  62%    0.4077
  finished  amber-fox   cifar-resnet              20k    0.3841*
  finished  brisk-wren  cifar-resnet              20k    0.4071
  finished  calm-lynx   cifar-resnet              20k    0.3942
  finished  dusty-mole  cifar-resnet              20k    0.7665   NaN 4.5k
  failed    eager-hawk  cifar-resnet    1.1k/20k   6%      1.67
  finished  mlp-wide    mnist-mlp               4.69k   0.05831*
  crashed   mlp-base    mnist-mlp     2.9k/4.69k  62%   0.06576
```

The first line answers the morning's question: how many runs, which
are live, and which need a look. The state word comes first on each row,
since it is what a glance looks for, and it is a word, so the screen
reads the same without colour. `crashed` means the run's writer exited
without an end; it is never a guess (the States section of the `.mli`).

The metric column shows each run's best value by the metric's goal, here
the least validation loss, or its latest value for a metric without a
goal. A star marks the best of each experiment among the rows shown: a
loss on CIFAR-10 is not compared with one on MNIST. `NONFINITE` gives the
step where the metric first went NaN or infinite, so the run that
diverged is named with the step to look at. `left` and `right` move the
column through the metrics any shown row has.

The columns after the metric are the ones whose values differ among the
rows shown. Typing `/cifar` keeps the sweep, and its knobs appear:

<!-- $MDX skip -->
```text
munin  6 of 8 runs: 1 live, 1 failed, 1 not finite · cifar-resnet · on gpu-01

  STATE     RUN                   STEP  val/loss↓  NONFINITE      lr  warmup
▸ live      tidy-otter  12.4k/20k  62%    0.4077              0.0003     500
  finished  amber-fox              20k    0.3841*              0.001     500
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
live  tidy-otter  step 12.4k/20k 62%  2.14/s  ~59m left  last sample 13s ago
! b dusty-mole: val/loss NaN from step 4,500, 7 other metrics from 4,250
val/loss ↓ 0.4077  best 0.4077 @12k  b NaN 4.5k        a amber-fox  b dusty-mole

   2.0 ┤⠱⡀
       │ ⠑⡄
   1.5 ┤  ⠘⢄
       │    ⠑⢄⡀
   1.0 ┤      ⠑⢄⡀
       │        ⠈⠒⠤⣀⡀
       │           ⠈⠈⠑⠢⠤⢄⣀
   0.5 ┤                  ⠉⠉⠉⠑⠒⠒⠒⠤⠤⠤⠤⠤⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⡀ ⡀⡀⡀⡀⢀⢀⢀ a⡀⢀⢀⢀ ⡀⡀⡀⡀⢀⢀⢀ ⡀⡀⡀⢀a
       └┬────────────────┬────────────────┬────────────────┬────────────────┬
        0              5,000            10,000           15,000           20,000
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
run, the metric and the step. The chart draws the focus as a solid line
and each marked run as a dotted level line with its letter at its end
and along it. The x axis runs to the run's declared `steps`, so the
blank right part of the chart is the work left. The line is drawn from
every sample: each column of braille dots shows the least and the
greatest value it covers, so a spike between two columns of a long run
stays on screen. A NaN or an infinity is a mark in the row above the
plot.

Wider terminals give every metric a chart, aligned on one x axis, and
draw the machine's readings over the run's time:

<!-- $MDX skip -->
```text
SYSTEM  gpu-01 · readings every 15s · errors not logged
CPU     busy ▕███████████▍    ▏  71%      load 1.3 per core   process 9.6 cores
MEM     used ▕████████▍       ▏  53%  33.0 of 62.7 GiB   swap 0.0 GiB
CUDA:0  busy ▕███████████████▌▏  97%      used 21.3 of 24.0 GiB 89%   71 °C   274 W
             alloc 20.9 of 22.0 GiB budget
```

A device's `used` of its `memory` is every process's, and the run's
`alloc` against its runtime's `budget` is the run's alone, so a full
device with a small `alloc` points at another process.

Enter on a metric zooms it, with a step cursor whose values the table
below the chart prints for each run. `s` smooths the line with a
debiased mean whose time constant the legend prints, `x` changes the x
axis to the metric's step metric or to time, `L` draws y on a log scale
where every value is above 0, and `+` and `-` narrow and widen the x
window at the latest step. Esc goes back, and `q` quits from anywhere.

## Keys

`?` lists the keys of the screen it opens over. One key means one thing
on every screen; the table is the Keys section of
[`munin_tui.mli`](../lib/tui/munin_tui.mli). The dashboard leaves the
mouse to the terminal, so text on screen can be selected and copied as
in any program, and `y` copies the run's id.
