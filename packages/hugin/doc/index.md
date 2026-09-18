# Hugin

Hugin creates publication-quality plots from Nx arrays using a declarative, pipeline-oriented API.

## What Hugin Does

Hugin turns immutable plot specifications into rendered output. You build a specification from mark constructors (`line`, `point`, `bar`, `hist`), decorate it with `title`, `xlabel`, and axis controls via the `|>` pipeline, and render with `render_png`, `render_svg`, or `show`.

Internally, rendering proceeds in three stages: the user-facing spec is compiled to a prepared tree (histograms binned, data bounds computed, marks auto-colored), then resolved to device-pixel coordinates, then drawn by a backend. Data compilation happens once; layout resolution is cheap and repeatable at different sizes.

## Quick Start

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 100 in
  let y = Nx.sin x in
  line ~x ~y () |> title "Sine wave" |> render_png "sine.png"
```

Two marks on shared axes:

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 100 in
  layers [
    line ~x ~y:(Nx.sin x) ~label:"sin" ();
    line ~x ~y:(Nx.cos x) ~label:"cos" ~line_style:`Dashed ();
  ]
  |> legend |> render_png "trig.png"
```

## Next Steps

- [Getting Started](01-getting-started.md) — installation, first plot, key concepts
- [Marks and Styling](02-marks-and-styling.md) — mark catalog, visual properties
- [Layout and Decorations](03-layout-and-decorations.md) — axes, scales, themes, multi-panel
- [Colors and Colormaps](04-colors-and-colormaps.md) — OKLCH colors, palettes, colormaps
- [Matplotlib Comparison](05-matplotlib-comparison.md) — side-by-side with Python
