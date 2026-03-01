# Hugin

Declarative plotting and visualization library for OCaml.

Hugin is part of the Raven ecosystem, providing a functional API to create
publication-quality charts and figures from Nx arrays. You build immutable
plot specifications with mark constructors, compose them with `|>` pipelines,
and render to PNG, SVG, PDF, or an interactive SDL window.

## Features

- Line, scatter, bar, histogram, error bar, fill-between, hline/vline, hspan/vspan
- Heatmap, colormapped image display (`imshow`), contour plots
- Multi-panel layouts with `Layout.grid`, `Layout.hstack`, `Layout.vstack`
- Perceptually uniform OKLCH colors with colorblind-friendly Okabe-Ito palette
- Predefined colormaps: viridis, plasma, inferno, magma, cividis, coolwarm
- Themes with context scaling (paper, notebook, talk, poster)
- Axis scales: linear, log, sqrt, asinh, symlog
- Cairo rendering (PNG, PDF), pure-OCaml SVG backend, interactive SDL display
- Format printer for Quill notebooks (`#install_printer`)

## Quick Start

<!-- $MDX skip -->
```ocaml
open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. 6.28 100 in
  let y = Nx.sin x in
  line ~x ~y () |> title "Sine wave" |> render_png "sine.png"
```

## Contributing

See the [Raven monorepo README](../../README.md) for guidelines.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
