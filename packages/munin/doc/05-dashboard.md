# Terminal Dashboard

Munin includes a terminal-based dashboard for monitoring runs in real time.
It renders braille-resolution charts, status indicators, and system resource
bars directly in the terminal.

## Launching

The dashboard is started with `munin watch`. With no arguments it auto-detects
the most recently started run:

<!-- $MDX skip -->
```
munin watch
```

To open a specific run, pass its ID:

<!-- $MDX skip -->
```
munin watch 20260317T143201_abc
```

To pick the latest run in a given experiment:

<!-- $MDX skip -->
```
munin watch --experiment mnist-sweep
```

## Layout

The dashboard has three sections stacked vertically:

### Header

A single-line bar showing:

- **Experiment and run name** (with run ID in parentheses)
- **Tags** as inline badges
- **Epoch** counter (e.g. `Epoch 3/10`) when an `epoch` metric is logged and
  `epochs` is set in the params
- **Step** counter (the highest step across all metrics)
- **Elapsed time** in `HH:MM:SS`
- **Status badge** on the right: a colored dot and label

### Metrics panel

A grid of braille-resolution line charts, one per user metric (system metrics
prefixed with `sys/` are excluded). Each chart shows the metric name, latest
value, and best value when a goal is defined.

Charts are arranged in a responsive grid: 2 columns when the terminal is wide
enough (at least 50 characters), 1 column otherwise. Rows are sized at 14
characters tall. When there are more metrics than fit on screen, they are split
into batches and navigated with `<` / `>`.

The currently selected chart has a white border; unselected charts have a dim
border. Pressing Enter on the selected chart opens the detail view.

### System panel

A side panel (right 34% of the screen) showing four resource bars when system
metrics are available:

- **CPU** -- combined user (green) + system (cyan) percentage with sparkline
- **Mem** -- system memory percentage and absolute GB with sparkline
- **Proc** -- process CPU percentage with sparkline
- **RSS** -- process resident set size in MB with sparkline

The bars change color based on utilization: green below 50%, yellow 50-80%,
red above 80%.

Toggle the system panel on/off with `[` or `]`.

### Footer

A hint bar showing available keyboard shortcuts for the current mode.

## Keyboard shortcuts

### Dashboard mode

| Key          | Action                                     |
|--------------|--------------------------------------------|
| Arrow keys   | Navigate the metric chart grid              |
| Enter/Space  | Open the selected metric in detail view     |
| `<` / `>`    | Previous / next batch of metrics            |
| `[` / `]`    | Toggle the system panel                     |
| `q` / Escape | Quit the dashboard                          |

### Detail view

| Key          | Action                                     |
|--------------|--------------------------------------------|
| `S`          | Cycle EMA smoothing: Off, Light (1), Medium (2), Heavy (3) |
| `q` / Escape | Return to dashboard                        |

## Status detection

The dashboard determines run status from the event log:

- **Live** (green) -- new events are arriving.
- **Stopped** (gray) -- no events received for 5 seconds. The run process may
  have crashed or been suspended.
- **Done** (blue) -- a `Finished` event with status `finished` was received.
- **Failed** (red) -- a `Finished` event with status `failed` was received.
- **Killed** (yellow) -- a `Finished` event with status `killed` was received.

The dashboard polls the event log on every tick and transitions between states
automatically.

## Detail view

Pressing Enter on a metric chart opens a full-screen detail view. The chart
fills 80% of the screen with full axis labels and gridlines.

**EMA smoothing** can be toggled by pressing `S`, cycling through four levels:

| Level  | Alpha | Effect                        |
|--------|-------|-------------------------------|
| Off    | --    | Raw values                    |
| Light  | 0.5   | Mild smoothing                |
| Medium | 0.3   | Moderate smoothing            |
| Heavy  | 0.15  | Aggressive smoothing          |

When smoothing is active, the chart title shows "(EMA)" and the footer displays
the current level number.

The best value (determined by the metric's declared goal, or heuristically for
keys containing "loss" or "error") is displayed below the chart.
