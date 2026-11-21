# hugin

Figures from nx tensors. A figure is an immutable value: marks read tensors
through channels, scales place values, axes and legends follow from scales.
It renders to PNG, SVG and PDF with a bundled font.

- hugin depends on nx and its own sub-libraries only; never on talon.
- A figure's SVG and PDF are the same bytes on every system. Floats are
  written rounded once by hugin's own code.
- Its PNG is the same image within a few levels a channel, not the same
  bytes: OCaml fuses multiply-adds on arm64 and not on x86-64, and libm's
  last bits differ. Tests compare rasters within a stated tolerance
  (test_draw.ml's goldens, examples/support/agree.ml).
- Example images are refreshed with `dune promote` after reading the diff.
