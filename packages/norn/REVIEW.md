# Reviewing norn

## Design

- A sampler special-cased for one model or one distribution: what would the
  general sampler need instead?
- A diagnostic computed differently for one structure.

## Semantics

References: the papers each `.mli` cites (NUTS, its adaptation, SMC, nested
sampling, the diagnostics). `test/gen/diag.py`'s ArviZ goldens are values
to compare with.

Failure here is draws from the wrong distribution, which no single run
shows. Then diagnostics that disagree with their published definition.

- A sampler change: a test whose statistic would detect the bias, with its
  false-failure rate stated.
- Densities and bijectors: the log-Jacobian's sign, support edges,
  infinities in the tails.
- Chains independent: one chain's divergence leaves the others alone.

## Skip

`test/golden/`, `bench/norn.thumper`.
