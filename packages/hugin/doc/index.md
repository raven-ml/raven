# Hugin

Hugin creates publication-quality plots from Nx arrays using a declarative, pipeline-oriented API.

## What Hugin Does

Hugin turns immutable plot specifications into rendered output. You build a specification from mark constructors (`line`, `point`, `bar`, `hist`), decorate it with `title`, `xlabel`, and axis controls via the `|>` pipeline, and render with `render_png`, `render_svg`, or `show`.

Internally, rendering proceeds in three stages: the user-facing spec is compiled to a prepared tree (histograms binned, data bounds computed, marks auto-colored), then resolved to device-pixel coordinates, then drawn by a backend. Data compilation happens once; layout resolution is cheap and repeatable at different sizes.

## System Requirements

Hugin needs Cairo and SDL2 for rendering:

<!-- $MDX skip -->
```bash
# macOS
brew install cairo sdl2

# Ubuntu/Debian
apt install libcairo2-dev libsdl2-dev
```

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

- [Getting Started](/docs/hugin/getting-started/) — installation, first plot, key concepts
- [Marks and Styling](/docs/hugin/marks-and-styling/) — mark catalog, visual properties
- [Layout and Decorations](/docs/hugin/layout-and-decorations/) — axes, scales, themes, multi-panel
- [Colors and Colormaps](/docs/hugin/colors-and-colormaps/) — OKLCH colors, palettes, colormaps
- [Matplotlib Comparison](/docs/hugin/matplotlib-comparison/) — side-by-side with Python
