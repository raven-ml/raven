# `04-linear-systems`

A linear system is given by its operator's product, never its matrix. This
example solves one system with each solver and differentiates its solution.

```bash
dune exec packages/jera/examples/04-linear-systems/main.exe
```

## What You'll Learn

- `Linear.dense` and `Linear.banded`, which probe the operator
- `Linear.cg` and `Linear.gmres`, which iterate on it
- The residual check every answer gets, read with `Solution.error`
- The derivative of a solution in its right-hand side

## Key Functions

| Function               | Purpose                |
| ---------------------- | ---------------------- |
| `Linear.solve x s a r` | The `u` with `a u = r` |
| `Linear.dense`         | Materialise and factor |
| `Linear.banded ~width` | A banded operator      |
| `Linear.cg`            | Conjugate gradients    |
| `Linear.gmres`         | Restarted GMRES        |

## Next Steps

Continue to [05-quadrature](../05-quadrature/).
