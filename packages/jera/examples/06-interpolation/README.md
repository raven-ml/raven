# `06-interpolation`

Splines, fits and their derivatives and integrals are one value, a piecewise
Chebyshev series. This example interpolates samples, keeps a step monotone, fits
a function to a tolerance, and interpolates over two axes.

```bash
dune exec packages/jera/examples/06-interpolation/main.exe
```

## What You'll Learn

- Cubic splines and broken lines through samples
- `Piecewise.derivative` and `Piecewise.integral`
- Steffen's monotone interpolant
- Fitting a function to a tolerance with `Piecewise.adapt`
- Extending past the domain with `Piecewise.extend`
- Tensor-product splines with `Grid.cubic`

## Key Functions

| Function                                       | Purpose                    |
| ---------------------------------------------- | -------------------------- |
| `Piecewise.cubic ends x y`                     | A cubic spline             |
| `Piecewise.steffen x y`                        | A monotone interpolant     |
| `Piecewise.adapt s ~degree ~tol ~budget f a b` | A fit to a tolerance       |
| `Piecewise.eval p x`                           | Evaluate                   |
| `Grid.cubic ends ~axes values`                 | A spline over several axes |

## Next Steps

Continue to [07-odes](../07-odes/).
