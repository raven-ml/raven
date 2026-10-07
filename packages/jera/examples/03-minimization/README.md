# `03-minimization`

Gradient methods take rune's gradient of a plain objective. This example
minimises Rosenbrock's function with each method, inside a box, fits a curve by
least squares, and minimises in one variable.

```bash
dune exec packages/jera/examples/03-minimization/main.exe
```

## What You'll Learn

- BFGS, L-BFGS, Newton and Nelder–Mead through `Minimize.solve`
- Box constraints with `~within`
- Least squares with `Minimize.levenberg_marquardt`
- Elementwise minima in brackets with `Minimize.bracket`

## Key Functions

| Function                                       | Purpose                      |
| ---------------------------------------------- | ---------------------------- |
| `Minimize.solve x m ?within ~tol ~budget f x0` | A local minimum from `x0`    |
| `Minimize.bfgs`, `lbfgs`, `newton`             | Gradient methods             |
| `Minimize.levenberg_marquardt r`               | Least squares from residuals |
| `Minimize.nelder_mead`                         | No gradient                  |
| `Minimize.bracket ~tol f ~lo ~hi`              | A minimum in each bracket    |

## Next Steps

Continue to [04-linear-systems](../04-linear-systems/).
