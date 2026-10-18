# hugin

Figures from nx tensors. A figure is an immutable value: marks read tensors
through channels, scales place values, axes and legends follow from scales.
It renders to PNG, SVG and PDF with a bundled font.

- hugin depends on nx and its own sub-libraries only; never on talon.
- A figure renders to the same bytes on every system. Floats are written
  rounded once by hugin's own code.
- Example images are refreshed with `dune promote` after reading the diff.
