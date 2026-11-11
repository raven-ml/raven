# The dashboard

`munin` without a command, on a terminal, shows the store's runs.
`munin watch` shows one run as it trains. The two are one program, the
dashboard of [`munin_tui.mli`](../lib/tui/munin_tui.mli), which reads the
store again once a second and never writes it. This page walks through a
session over a sweep. Every rule it relies on is stated in that `.mli`.

<!-- $MDX skip -->
```sh
munin                 # the store's runs
munin watch           # waits for a run that is still going
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

## One screen

A sweep of six learning rates over a ResNet, one still training, and two
older runs of an MLP, on a terminal of 120 columns:

<!-- $MDX skip -->
```text
 ᛗ munin                                              8 runs · 2 experiments   ● 1 live   ✗ 2 failed   × 1 NaN   gpu-01
                                 │                                                          │
 ¹ runs                          │ val/loss ↓                                               │ ³ run
                                 │ █▀█   █ █ █▀█ ▀▀█ ▀▀█      train/loss  0.2762  smoothed  │
 ▾ cifar-resnet  6      val/loss │ █ █   ▀▀█ █ █   █   █      train/acc   95.40%  smoothed  │ ● tidy-otter  live
▌ ■ tidy-otter  ● ⠢⣀⣀⡀   0.4077  │ ▀▀▀ ▀   ▀ ▀▀▀   ▀   ▀      val/acc     94.18%            │ last sample 0s ago
  · amber-fox   ✓ ⠢⣀⣀⣀⣀  0.3841 ★│ best 0.4077 at 12,000                                    │
  · brisk-wren  ✓ ⠢⣀⣀⣀⣀  0.4071  │                                                          │ ━━━━━━━━━━━━━──────── 62%
  · calm-lynx   ✓ ⠢⣀⣀⣀⣀  0.3942  │ ‹   val/loss   train/acc   val/acc   val/ece   ›         │ 12,400 of 20,000 steps
  · dusty-mole  × ⢢⡀     0.7665  │     ▔▔▔▔▔▔▔▔                                             │ 59m 11s left · ends 13:17
  · eager-hawk  ✗ ⠄       1.670  │                                                          │ 2.1 steps/s · 1h 37m in
 ▾ mnist-mlp  2         val/loss │ ━ tidy-otter 0.4077   r others                           │
  · mlp-wide    ✓ ⠈⢆⣀⣀⣀ 0.05831 ★│                                                          │ params ──────────────────
  · mlp-base    ✗ ⠈⢆⣀⡀  0.06576  │      2┤⡄                            ┊now                 │ lr     0.0003000   varies
                                 │       │⢱                            ┊                    │ warmup 500         varies
                                 │       │⠘⡄                           ┊                    │ batch  256
                                 │       │ ⢣                           ┊                    │ epochs 10
                                 │    1.5┤ ⠘⡄                          ┊                    │ wd     0.05000
                                 │       │  ⢣                          ┊                    │ +1 more   i: all
                                 │       │   ⢣                         ┊                    │
                                 │       │   ⠈⢆                        ┊                    │ machine ─────────────────
                                 │       │    ⠈⢆                       ┊                    │ GPU  ━━━━━━━━━━━━━━━╸ 97%
                                 │      1┤     ⠈⢢                      ┊                    │ VRAM ━━━━━━━━━━─ 21.3/24G
                                 │       │       ⢣                     ┊                    │ CPU  ━━━━━━━━━━━╸──── 71%
                                 │       │        ⠑⠢⡀                  ┊                    │ RAM  ━━━━╸──── 33.0/62.7G
                                 │       │          ⠈⠒⢄⡀               ┊                    │ 71°C  274 W
                                 │    0.5┤             ⠈⠉⠒⠤⠤⢄⣀⣀        ┊                    │
                                 │       │                     ⠉⠉⠉⠉⠉⠉⠉⠒┊                    │ source ──────────────────
                                 │       └┬───────────┬───────────┬───────────┬──────────┬  │ 5eaf82b dirty
                                 │        0           5k         10k         15k      step  │ train.exe --lr 0.0003 --…
                                 │                                                          │
                                 │ train/loss                   train/acc                   │
                                 │       │⡄           ┊now            │      ⢀⡠⠤⠤⠒⠒⠂now     │
                                 │      2┤⠘⡄          ┊               │    ⡠⠊⠁     ┊        │
                                 │       │⠐⢱          ┊            0.8┤  ⠄⡔⠁       ┊        │
                                 │       │ ⢁⢣         ┊               │ ⡈⡜         ┊        │
                                 │      1┤  ⢂⢣        ┊            0.6┤⠠⡸          ┊        │
                                 │       │    ⠑⢄⡀     ┊               │⢠⠃          ┊        │
                                 │       │      ⠈⠑⠒⠢⠤⠤⠄               │⠃           ┊        │
 charts › strips › zoom   ↑↓ run   ←→ metric   enter strips   space overlay   / filter   ? keys
```

The top bar answers the morning's question: how many runs, which are
live, and which need a look, each count matching the rows that show its
glyph. Below it the screen has three panes. The runs pane at the left
lists the runs under their experiments; its cursor, the row starting
with `▌`, picks the run the rest of the screen shows. The centre draws
that run's metrics. The run pane at the right states the run: its state,
its progress, its params, its machine and its source. The status bar
names the depth the centre is at and the keys that act there; on a narrow
terminal they leave from the last, and `? keys`, which lists them all,
stays.

The panes take their widths from the terminal's. On a narrower terminal
the side panes give way to the centre, and `1` and `3` bring either back
(The screen, in the `.mli`).

## The runs

Each row is a run: a swatch of its colour when a chart draws it, its
name, its state glyph, its
experiment's key metric drawn over its steps, and its best value of it,
here the least validation loss. A star marks the best of each
experiment: a loss on CIFAR-10 is not compared with one on MNIST. The
key metric is chosen by a rule, a goal metric outside `train/` whose key
ends in `loss` first, so that every experiment of a store reads the same
way without being told.

A glyph says the run's state, and the run pane says it in a word:
`live` means a process holds the run's lock and the dashboard has seen
a sample arrive. A run the dashboard has just opened reads `running`
until one arrives, and a run that logs nothing for three of its usual
gaps reads `stalled`. These words come from what the dashboard observed
on its own clock, so a program that logs times of its own, or a writer
on another machine, cannot make a hung run look alive (States, in the
`.mli`). `×` marks a run with a sample that is not finite, as
`dusty-mole`, whose losses turned NaN.

`/` filters the rows as you type. A word matches a name, an id, an
experiment, a tag, a state word or a param written `key=value`, and
`lr<0.002` keeps the runs whose `lr` is below `0.002`. Enter keeps the
filter, and Esc clears it. `o` orders the runs by start, by best value
or by name, and `e` folds an experiment to one line; a store of more
than 30 runs opens with every experiment folded but the cursor's.

## Charts

The centre starts with the hero: the key metric's latest value in large
digits, its best, and the latest values of other goal metrics. Under it
the tabs name the run's metrics, `left` and `right` move between them,
and the large chart draws the one chosen; small charts of the other goal
metrics follow when the terminal has the room.

A chart's x axis runs to the largest step of the runs it draws that
logged its metric. While a run may still log (it is starting, running,
live, stalled or writing), its end is rounded up to 1, 2 or 5 times a
power of ten, and to no more
than its declared `steps`, so the axis changes at most three times a
decade as the run grows, and not at a pause; the dotted rule `now` marks
the latest step of a live run, and the run pane says how many steps are
left. A line is one value per
braille dot column, the mean of the samples in it, and a raw sample far from the line stays
on screen as a faint dot, so a spike is never averaged away. A metric
under `train/` is smoothed by default; `s` turns smoothing off and on. `y` switches the metric between a linear and a log scale. Each
y label is a round value that its digits give exactly, so a label never
reads as a number the chart does not hold. From the run's first sample
that is not finite, the plot is hatched and labelled `× NaN`.

Each chart is drawn from a reduction of the series to its dot columns,
made off the loop, so a run of millions of samples draws as fast as one
of thousands. Right after a resize a chart can show its axes alone until
the next reduction arrives (Charts, in the `.mli`).

## Comparing runs

`space` overlays the run under the cursor; up to four overlays are drawn
in every chart beside the cursor's run, each in a colour it keeps. `r`
draws the experiment's other runs in grey behind them. The legend gives
each drawn run's value at the cursor run's latest step. Here
`amber-fox` and `dusty-mole` are overlaid, on a terminal without colour,
where each overlay's line ends in its letter:

<!-- $MDX skip -->
```text
│ ━ tidy-otter 0.4077   b dusty-mole NaN                   │
│ a amber-fox 0.3939   r others                            │
│                                                          │
│      2┤⡄                            ┊now                 │
│       │⢱                            ┊                    │
│       │⠈⡆                           ┊                    │
│       │ ⢱                           ┊                    │
│    1.5┤ ⠈⣆                          ┊                    │
│       │  ⠘⣆                         ┊                    │
│       │   ⠸⡄                        ┊                    │
│       │    ⠹⣄                       ┊                    │
│      1┤     ⠘⢦                      ┊                    │
│       │      ⠈⢷⡀                    ┊                    │
│       │        ⠙⢷b                  ┊                    │
│       │          ⠙⠲⢄⡀               ┊                    │
│    0.5┤             ⠈⠉⠒⠶⠦⢤⣄⣀⣀⣀⡀     ┊                    │
│       │                     ⠉⠉⠉⠉⠉⠉⠛⠛⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒a  │
│       └┬───────────┬───────────┬───────────┬──────────┬  │
│        0           5k         10k         15k      step  │
│                                                          │
```

`dusty-mole`'s line stops at its first NaN. Where two lines cross a
cell, both show, in the colour of the one drawn later; the cursor's run
is drawn last. `c` clears the overlays, and so does Esc at the charts.

## Strips and zoom

Enter goes one depth deeper. At the strips depth every metric of the run
is a strip two lines high on one step axis, the largest end of the runs
the strips draw, so reading down a column reads one step across every
metric. Right of each is its latest value
and, with an overlay, the difference to the first overlay at the latest
step both logged:

<!-- $MDX skip -->
```text
│ ━ tidy-otter   ━ amber-fox                                     latest   vs amber-fox
│
│  train/loss      ⠉⠒⠤⢄⣀⡀                                           0.2642        +11%
│  ↓ min                ⠈⠉⠓⠒⠒⠦⠤⠤⠤⠤⢤⣤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀
│
│ ▌val/loss        ⠑⠢⢄⣀                                             0.4077       +3.5%
│  ↓ min               ⠉⠉⠒⠒⠢⠤⠤⠤⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀
```

While the run is live, its machine's readings end the list, each filled
down to its floor:

<!-- $MDX skip -->
```text
│ machine ────────────────────────────────────────────────────────────────────────────
│  GPU busy        ⣼⣾⣾⣾⣿⣿⣿⣿⣷⣷⣷⣷⣿⣿⣾⣿⣿⣿⣾⣾⣷⣷⣿⣿⣿⣷⣿⣿⣾⣾⣾⣿⣿⣿⣿⣷⣷⣷⣷⣿⣿           97%
│                  ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
```

`up` and `down` move between the strips, and Enter on one zooms it: the
metric fills the centre with each column's raw extent faint behind the
smoothed line, and a step cursor that `left` and `right` move, ten
samples at a time with shift. A readout beside the chart gives each
drawn run's value at that step, smoothed and raw. `+` and `-` narrow and
widen the steps shown about the cursor. Esc goes back a depth.

## The run's facts and the keys

`i` shows everything about the cursor's run over the centre: its params,
notes, tags, source, every metric's latest and best values, the artifact
versions it used, its files, its resumes and any damaged lines of its
log. `?` lists the keys; the same table is the Keys section of
[`munin_tui.mli`](../lib/tui/munin_tui.mli). A key means one thing at
every depth, and `q` quits from anywhere but the filter. The dashboard
leaves the mouse to the terminal, so text on screen can be selected and
copied as in any program.

## What it reads

Once a second the dashboard lists the store and follows the runs that
are still going; a key or a resize reads only what the new screen
lacks. A run that has ended is read once: coming back to it reads
nothing of it. A metric's samples cost 24 bytes each while their run is
held (Running, in the `.mli`).
