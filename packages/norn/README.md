# Norn

Probabilistic inference for OCaml.

Norn turns a log density over a structure of your own type into draws and
diagnostics. A position is your structure with a leading chain axis, and a
density maps it to one log density per chain.

- `Norn.Bij`: bijectors from unconstrained coordinates onto a value's
  support.
