# norn

Probabilistic inference over structures: a log density over a value of the
user's type becomes draws and diagnostics returned as values of that type.
`norn.model` gives the density from one generative function.

- norn depends on nx, rune and jera. It uses rune only for what it states
  about its own computations; it never calls `jit`.
- Goldens come from `test/gen/` through `uv run`: `diag.py` (ArviZ),
  `dist.py`, `posteriordb.py`.
- A sampler test is statistical. Its threshold states the false-failure
  rate.
