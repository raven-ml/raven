# `01-distributions`

A distribution describes one whole tensor value. This example evaluates
densities, broadcasts parameters, draws samples, and reads each family's support
and coordinates.

```bash
dune exec packages/norn/examples/01-distributions/main.exe
```

## What You'll Learn

- Constructing distributions and reading `log_density`, `factors` and `quantile`
- Broadcasting parameters and `iid` copies
- Sampling from a key
- Discrete and vector families
- Values outside the support, and parameters outside their domain
- `support` and `coords`

## Key Functions

| Function                  | Purpose                              |
| ------------------------- | ------------------------------------ |
| `Dist.normal ~loc ~scale` | A family                             |
| `Dist.log_density d x`    | The log density, a scalar            |
| `Dist.factors d x`        | One log density per independent unit |
| `Dist.sample k d`         | A draw                               |
| `Dist.iid s d`            | Independent copies                   |
| `Dist.coords d`           | The bijector onto `d`'s support      |

## Next Steps

Continue to [02-bijectors](../02-bijectors/).
