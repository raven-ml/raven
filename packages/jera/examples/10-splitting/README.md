# `10-splitting`

For a separable Hamiltonian, a splitting composes the exact flows of its parts.
This example compares the energy error of four splittings and of rk4 over a long
march.

```bash
dune exec packages/jera/examples/10-splitting/main.exe
```

## What You'll Learn

- Kicks and drifts as flows
- `Split.march` with leapfrog, McLachlan and Yoshida schemes
- Bounded energy error against rk4's drift
- Time reversibility of `Split.step`

## Key Functions

| Function                                     | Purpose             |
| -------------------------------------------- | ------------------- |
| `Split.march s m ~steps ~kick ~drift ~at s0` | States at each time |
| `Split.step m ~kick ~drift h s`              | One step            |
| `Split.leapfrog`, `yoshida4`, ...            | Schemes             |

## Next Steps

Continue to [11-compile-and-batch](../11-compile-and-batch/).
