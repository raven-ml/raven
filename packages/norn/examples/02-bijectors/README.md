# `02-bijectors`

A bijector maps unconstrained coordinates onto a support and reports its
log-determinant. This example maps coordinates onto intervals, simplices and
ordered vectors, and pulls a density back to coordinates.

```bash
dune exec packages/norn/examples/02-bijectors/main.exe
```

## What You'll Learn

- `Bij.forward` and `Bij.inverse`
- Elementwise and vector bijectors
- Pulling a density back: density plus log-determinant
- `Dist.transform`, the distribution of a bijector's image

## Key Functions

| Function                                    | Purpose                   |
| ------------------------------------------- | ------------------------- |
| `Bij.forward b u`                           | Value and log-determinant |
| `Bij.inverse b x`                           | Coordinates of a value    |
| `Bij.exp`, `interval`, `simplex`, `ordered` | Bijectors                 |
| `Dist.transform b d`                        | An image distribution     |

## Next Steps

Continue to [03-nuts](../03-nuts/).
