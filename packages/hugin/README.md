# Hugin

Figures from nx tensors, for OCaml.

Hugin is part of the Raven ecosystem. A figure is an immutable value: marks
read tensors through channels, scales place their values, and axes and
legends follow from the scales. Figures compose into layers, grids and
facets, and render to PNG, SVG and PDF with a bundled font.

## Features

- Marks: dots, lines, areas, rectangles (bars, heatmaps, histograms, spans),
  rules, text, images and filled contours, and `Mark.v` to write your own
- Channels that broadcast like nx operands: one tensor of shape `[5; 200]`
  is five curves
- Linear, logarithmic, symmetric-log, power and band scales, shared by name
  across a figure
- Facets, grids with aligned data areas, spans, titles and fixed aspect ratios
- Sequential, diverging and qualitative colour schemes, and colours mixed in
  Oklab
- Large data reduced where it lives: dense lines, scatters and images draw
  only what the output's pixels can show
- PNG, SVG and PDF written in pure OCaml, the same bytes on every run
- Display in Quill notebooks through `Hugin.pp`

## Quick Start

```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float64 0. 6.28 100 in
  line ~x:(num x) ~y:(num (Nx.sin x)) ()
  |> title "Sine wave"
  |> save "sine.png"
```

See the [documentation](doc/index.md) and the [examples](examples/README.md).

## Contributing

See the [Raven monorepo README](../../README.md) for guidelines.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
