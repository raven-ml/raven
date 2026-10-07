# `05-quadrature`

Integrals over ranges are elementwise. This example uses a fixed rule, an
adaptive solve, double-exponential rules for a singularity and an infinite
range, cumulative integrals, and two methods over a box.

```bash
dune exec packages/jera/examples/05-quadrature/main.exe
```

## What You'll Learn

- Fixed Gauss rules with `Quad.fixed`, many integrals at once
- Adaptive Gauss–Kronrod with `Quad.adaptive`
- `Quad.tanh_sinh` for endpoint singularities and infinite ranges
- `Quad.cumulative` between knots
- `Quad.cubature` and `Quad.qmc` over boxes

## Key Functions

| Function                               | Purpose                    |
| -------------------------------------- | -------------------------- |
| `Quad.fixed r f range`                 | A rule's sum               |
| `Quad.adaptive r ~tol ~budget f range` | An integral to a tolerance |
| `Quad.tanh_sinh ~tol f range`          | Double-exponential rules   |
| `Quad.cubature`, `Quad.qmc`            | Integrals over boxes       |
| `Quad.Range.v`, `from`, `line`         | Ranges                     |

## Next Steps

Continue to [06-interpolation](../06-interpolation/).
