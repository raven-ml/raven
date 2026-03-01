# Getting Started

This guide covers installation, your first plot, and the key concepts behind Hugin.

## Installation

Install system dependencies:

<!-- $MDX skip -->
```bash
# macOS
brew install cairo sdl2

# Ubuntu/Debian
apt install libcairo2-dev libsdl2-dev
```

Then install hugin:

<!-- $MDX skip -->
```bash
opam install hugin
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build dev/hugin
```

Add to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries hugin))
```

## Your First Plot

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 100 in
  let y = Nx.sin x in
  line ~x ~y () |> title "Sine wave" |> render_png "sine.png"
```

This creates a 1-D array of 100 points, computes the sine, builds a line specification, adds a title, and writes a PNG file.

## Key Concepts

### Marks

A mark constructor (`line`, `point`, `bar`, `hist`, `heatmap`, etc.) takes data arrays and optional visual properties and returns an immutable plot specification of type `t`. A mark is already a complete spec — you can render it directly:

<!-- $MDX skip -->
```ocaml
line ~x ~y () |> render_png "plot.png"
```

### Decorations

Decoration functions add metadata to a spec. They are designed for the `|>` pipeline:

<!-- $MDX skip -->
```ocaml
line ~x ~y ()
|> title "My Plot"
|> xlabel "Time (s)"
|> ylabel "Amplitude"
|> xlim 0. 10.
|> grid_lines true
```

Decorations include `title`, `xlabel`, `ylabel`, `xlim`, `ylim`, `xscale`, `yscale`, `grid_lines`, `legend`, `xticks`, `yticks`, `xinvert`, `yinvert`, `with_theme`, and tick formatting.

### Composition

`layers` overlays multiple marks on shared axes:

<!-- $MDX skip -->
```ocaml
layers [
  line ~x ~y:(Nx.sin x) ~label:"sin" ();
  line ~x ~y:(Nx.cos x) ~label:"cos" ~line_style:`Dashed ();
]
|> legend |> render_png "overlay.png"
```

You can mix mark types freely. A `line` with `point` markers, a `bar` chart with `hline` reference lines — anything goes.

### Layout

`Layout.grid` arranges specs in rows and columns:

<!-- $MDX skip -->
```ocaml
let p1 = line ~x ~y:(Nx.sin x) () |> title "sin" in
let p2 = line ~x ~y:(Nx.cos x) () |> title "cos" in
Layout.grid [ [ p1; p2 ] ] |> render_png "grid.png"
```

`Layout.hstack` and `Layout.vstack` are shorthands for single-row and single-column grids.

### Rendering

Four output modes:

| Function | Output |
|----------|--------|
| `render_png "file.png" t` | PNG image file |
| `render_svg "file.svg" t` | SVG document file |
| `render_pdf "file.pdf" t` | PDF document file |
| `show t` | Interactive SDL window (resize, Esc to close) |

All renderers accept optional `~width` and `~height` (default 1600×1200) and `~theme`.

`render_svg_to_string` and `render_to_buffer` return the output as a string instead of writing a file.

## Common Marks

### Line

<!-- $MDX skip -->
```ocaml
line ~x ~y ()
line ~x ~y ~color:Color.blue ~line_style:`Dashed ~line_width:2.0 ()
line ~x ~y ~step:`Post ()  (* staircase plot *)
```

### Scatter

<!-- $MDX skip -->
```ocaml
point ~x ~y ()
point ~x ~y ~color_by:values ~size:8. ~marker:Star ()
point ~x ~y ~size_by:weights ()  (* variable marker size *)
```

### Bar Chart

<!-- $MDX skip -->
```ocaml
bar ~x:categories ~height:values ()
bar ~x:categories ~height:values ~width:0.5 ~color:Color.orange ()
```

### Histogram

<!-- $MDX skip -->
```ocaml
hist ~x:data ()
hist ~x:data ~bins:(`Num 30) ~density:true ~color:Color.green ()
```

### Heatmap

<!-- $MDX skip -->
```ocaml
(* data has shape [|rows; cols|] *)
heatmap ~data ()
heatmap ~data ~annotate:true ~cmap:Cmap.viridis ()
```

### Fill Between

<!-- $MDX skip -->
```ocaml
fill_between ~x ~y1:(Nx.sub y err) ~y2:(Nx.add y err) ~alpha:0.3 ()
```

### Error Bars

<!-- $MDX skip -->
```ocaml
errorbar ~x ~y ~yerr:(`Symmetric err) ()
errorbar ~x ~y ~yerr:(`Asymmetric (lo, hi)) ~xerr:(`Symmetric xerr) ()
```

## Next Steps

- [Marks and Styling](/docs/hugin/marks-and-styling/) — full mark catalog and visual properties
- [Layout and Decorations](/docs/hugin/layout-and-decorations/) — axes, scales, themes, multi-panel
- [Colors and Colormaps](/docs/hugin/colors-and-colormaps/) — OKLCH colors and colormap reference
