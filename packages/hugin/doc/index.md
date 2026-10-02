# Hugin

Hugin draws figures from nx tensors. A figure is an immutable value: marks
read tensors through channels, scales place the values, and axes and legends
follow from the scales. Figures render to PNG, SVG and PDF with a bundled
font, and a figure renders to the same bytes every time.

## Quick Start

Five noisy training curves, one tensor of shape `[5; 200]`, drawn as five
lines coloured by run:

```ocaml
open Hugin

let steps = Nx.linspace Nx.float64 0. 1. 200

let losses =
  let rates = Nx.create Nx.float64 [| 5; 1 |] [| 2.; 2.5; 3.; 3.5; 4. |] in
  let noise = Nx.Rng.normal (Nx.Rng.key 0) Nx.float64 [| 5; 200 |] in
  Nx.add (Nx.exp (Nx.neg (Nx.mul rates steps))) (Nx.mul_s noise 0.01)

let () =
  line
    ~x:(num ~title:(Text.v "progress") steps)
    ~y:(num ~title:(Text.v "loss") losses)
    ~stroke:(dim ~title:(Text.v "run") 0)
    ()
  |> title (Text.v "Five runs")
  |> save "runs.png"
```

`num` reads a tensor as quantities and `dim 0` reads the index along axis 0
as a category, so each run gets its own colour and the legend lists the runs.

## How Hugin Works

- **A figure is a value.** `Hugin.t` describes what to draw. Building one
  reads no data, so a figure can be stored, compared with `equal` and
  rendered again.
- **Marks bind channels to roles.** `dot`, `line`, `area`, `rect`, `rule`,
  `text`, `image` and `contour` take channels as labelled arguments named
  after their roles: `~x`, `~y`, `~fill`, `~stroke`, `~size` and others.
- **Channels broadcast.** The channels of a mark broadcast together like nx
  operands, and each element of the result is one row. A `[5; 200]` tensor
  is 1,000 rows of one mark.
- **Scales are found by name.** Each channel reads a scale, by default the
  one named after its role. Marks that read one scale share its domain, and
  each scale yields one axis or legend.
- **Composition decides sharing.** `layer` draws figures over one another
  on shared scales, `grid` arranges them in cells with scales of their own,
  and the facet channels `~fx` and `~fy` split one mark into panels.
- **Rendering is three pure stages.** `resolve` fits the scales, `layout`
  measures text and places panels, and `draw` paints a picture. `save` runs
  the three and writes a file.

## Libraries

| Library | Module | Contents |
|---|---|---|
| `hugin` | `Hugin` | Figures, marks, channels, composition, rendering |
| `hugin.kit` | `Hugin_kit` | Scales, ticks, number and time formats, colour schemes, symbols, curves, statistics |
| `hugin.gg` | `Hugin_gg` | Points, boxes, affine maps, paths, strokes, colours |
| `hugin.gg.kit` | `Hugin_gg_kit` | Polygons, and the isolines and isobands of sampled fields |
| `hugin.font` | `Hugin_font` | OpenType fonts, the bundled Inter faces, glyph runs |
| `hugin.text` | `Hugin_text` | Rich text and its layout |
| `hugin.vg` | `Hugin_vg` | Pictures and renderables |
| `hugin.vg.raster` | `Hugin_vg_raster` | Rendering to pixels and PNG |
| `hugin.vg.svg` | `Hugin_vg_svg` | Rendering to SVG |
| `hugin.vg.pdf` | `Hugin_vg_pdf` | Rendering to PDF |

`Hugin` re-exports the modules of the lower libraries that figures use, so
most programs link `hugin` alone.

## Next Steps

- [Getting Started](01-getting-started.md): a first figure, channels and
  data, saving, notebooks, errors and warnings
- [Marks](02-marks.md): each built-in mark and the figures it makes
- [Scales, Colour and Guides](03-scales-colour-and-guides.md): naming and
  configuring scales, colour schemes, axes and legends
- [Composition and Layout](04-composition-and-layout.md): layers, grids,
  facets, titles, sizes and themes
- [Rendering and Extending](05-rendering-and-extending.md): the three
  stages, views, and writing your own marks
- [Porting from the First Hugin](06-porting.md): each function of the
  first hugin and its counterpart
- [Examples](../examples/01-line/README.md): seventeen runnable figures,
  from a first line to a paper figure
