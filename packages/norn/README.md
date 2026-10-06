# Norn

Probabilistic inference for OCaml.

Norn turns a log density over a structure of your own type into draws and
diagnostics. A position is your structure with a leading chain axis, and a
density maps it to one log density per chain.

- `Norn.with_gradient`: a density with a gradient of its own, such as an
  adjoint solver's.
- `Norn.Bij`: bijectors from unconstrained coordinates onto a value's
  support.
- `Norn.Draws` and `Norn.Stats`: draws of your structure with `[chain; draw]`
  axes, and the statistics of each transition.
