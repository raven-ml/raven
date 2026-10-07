# `08-laplace`

A Laplace approximation is the Gaussian whose precision is the curvature at the
mode. This example finds a mode with jera, builds the approximation, samples it,
and uses it as NUTS's geometry.

```bash
dune exec packages/norn/examples/08-laplace/main.exe
```

## What You'll Learn

- `Gaussian.of_precision` at a mode found by `Jera.Minimize`
- `Gaussian.sample`, `variance` and `log_density`
- A density over chains from a one-position density with `Rune.vmap'`
- A Gaussian as `Nuts.init ~geometry`

## Key Functions

| Function                                | Purpose                     |
| --------------------------------------- | --------------------------- |
| `Gaussian.of_precision u dtype ~mean p` | A Gaussian from a precision |
| `Gaussian.sample u k ~n g`              | Draws                       |
| `Nuts.init ~geometry`                   | Start preconditioned        |

## Next Steps

Continue to [09-calibration](../09-calibration/).
