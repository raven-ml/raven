# Hugin

Publication-quality plotting and visualization library for OCaml

Hugin is part of the Raven ecosystem, providing a flexible and powerful API to create
publication-quality charts, plots, and figures from Ndarray data. It supports 2D and 3D
visualizations with multiple backends (Cairo+SDL) and seamless integration in notebooks.

## Features

- 2D plots: line, scatter, bar, histogram, step, error bars, fill-between
- 3D plots: line3d, scatter3d
- Image display: imshow for array visualization
- Matrix heatmap: matshow for 2D data as images
- Text annotation: titles, labels, and arbitrary text
- Figure and axes abstraction with customizable ticks, labels, and layouts
- Data from Ndarray: seamless plotting of numerical arrays
- Multiple rendering backends: Cairo+SDL on native platforms

## Quick Start

```ocaml
open Ndarray
open Hugin

(* Create a simple line plot *)
let x = create float32 [|0; 1; 2; 3|] [|0.; 1.; 2.; 3.|] in
let y = create float32 [|0; 1; 2; 3|] [|0.; 1.; 4.; 9.|] in

let fig = Figure.create () in
let ax = Figure.subplot fig 1 1 0 in
plot ~x ~y ax;
Figure.save fig "plot.png"
```

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
