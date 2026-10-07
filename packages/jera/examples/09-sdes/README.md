# `09-sdes`

A problem is a drift, a diffusion and a Brownian path. This example checks an
Ornstein–Uhlenbeck process against its law and compares the strong error of four
methods on one path.

```bash
dune exec packages/jera/examples/09-sdes/main.exe
```

## What You'll Learn

- Brownian paths with `Sde.Brownian.v`
- `Sde.march` with a drift and a diffusion
- Strong order: Euler–Maruyama, Milstein, SRA1 and reversible Heun on one path

## Key Functions

| Function                                          | Purpose                      |
| ------------------------------------------------- | ---------------------------- |
| `Sde.Brownian.v key dtype ~shape ~t0 ~t1 ~depth`  | A Brownian path              |
| `Sde.march y m ~steps ~drift ~diffusion w ~at y0` | States along the path        |
| `Sde.sra1`                                        | Order 3/2 for additive noise |

## Next Steps

Continue to [10-splitting](../10-splitting/).
