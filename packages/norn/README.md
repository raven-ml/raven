# Norn

Probabilistic inference for OCaml.

Norn turns a log density over a structure of your own type into draws and
diagnostics. A position is your structure with a leading chain axis, and a
density maps it to one log density per chain.

- `Norn.with_gradient`: a density with a gradient of its own, such as an
  adjoint solver's.
- `Norn.Dist`: distributions over whole tensors, for priors and
  likelihoods.
- `Norn.Support` and `Norn.Bij`: supports, and the bijectors that map
  unconstrained coordinates onto them.
- `Norn.Gaussian`: diagonal-plus-low-rank Gaussians over your structure,
  the geometry samplers adapt.
- `Norn.Nuts`: the No-U-Turn sampler over many chains at once, with warmup.
- `Norn.Draws` and `Norn.Stats`: draws of your structure with `[chain; draw]`
  axes, and the statistics of each transition.
- `Norn.Diag`: R-hat, effective sample sizes and calibration, each a value
  of your structure: `(Norn.Diag.rhat schools post).tau` is the R-hat of
  `tau`.
- `Norn.Summary`: a table of draws by element, with its findings.

The model language, `norn.model`, turns one generative function into these
densities, and draws back into values of your types.
