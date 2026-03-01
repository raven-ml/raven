# Hugin Examples

Learn Hugin through progressively complex examples. Start with `01-line-plot`
and work through the numbered examples in order.

## Examples

| Example | Concept | Key Functions |
|---------|---------|---------------|
| [`01-line-plot`](./01-line-plot/) | Your first plot | `line`, `render_png` |
| [`02-styling`](./02-styling/) | Colors, line styles, markers | `~color`, `~line_style`, `~marker`, `~alpha` |
| [`03-scatter`](./03-scatter/) | Scatter plots and color mapping | `point`, `~color_by` |
| [`04-bar-chart`](./04-bar-chart/) | Bar charts with categorical axes | `bar`, `xlabel`, `ylabel`, `xticks` |
| [`05-histogram`](./05-histogram/) | Histograms and density | `hist`, `~bins`, `~density` |
| [`06-layers`](./06-layers/) | Overlaying marks and legends | `layers`, `fill_between`, `hline`, `legend` |
| [`07-decorations`](./07-decorations/) | Axis control and grid lines | `xscale`, `xlim`, `ylim`, `xtick_format`, `grid_lines` |
| [`08-grid-layout`](./08-grid-layout/) | Multi-panel layouts | `Layout.grid` |
| [`09-themes`](./09-themes/) | Themes and context scaling | `Theme.default`, `Theme.dark`, `Theme.talk` |
| [`10-showcase`](./10-showcase/) | Full showcase with multiple outputs | All mark types, `heatmap`, `render_svg` |
| [`11-errorbar`](./11-errorbar/) | Measurement uncertainty | `errorbar`, `~yerr`, `~cap_size` |

## Running Examples

All examples can be run with:

```bash
dune exec dev/hugin/examples/<name>/main.exe
```

For example:

```bash
dune exec dev/hugin/examples/01-line-plot/main.exe
```

## Quick Reference

### Single Plot

```ocaml
open Hugin

let x = Nx.linspace Nx.float32 0. 6.28 100 in
let y = Nx.sin x in
line ~x ~y () |> title "Sine" |> render_png "plot.png"
```

### Multiple Marks on Shared Axes

```ocaml
layers
  [
    line ~x ~y:(Nx.sin x) ~label:"sin" ();
    line ~x ~y:(Nx.cos x) ~label:"cos" ~line_style:`Dashed ();
  ]
|> legend |> render_png "plot.png"
```

### Grid Layout

```ocaml
let p1 = line ~x ~y:(Nx.sin x) () |> title "sin" in
let p2 = line ~x ~y:(Nx.cos x) () |> title "cos" in
Layout.grid [ [ p1; p2 ] ] |> render_png "grid.png"
```
