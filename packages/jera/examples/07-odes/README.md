# `07-odes`

A problem is a field and an initial state of any structure. This example marches
a pendulum with fixed steps, solves it to a tolerance, samples it, builds its
path as a function of time, and stops at an event.

```bash
dune exec packages/jera/examples/07-odes/main.exe
```

## What You'll Learn

- `Ode.march` with a fixed-step method
- `Ode.sample` and `Ode.solve` with adaptive steps
- `Ode.path`: the solution as a piecewise series
- `Ode.event`: the first zero of an event function

## Key Functions

| Function                                  | Purpose                            |
| ----------------------------------------- | ---------------------------------- |
| `Ode.march y m ~steps f ~at y0`           | Fixed steps                        |
| `Ode.solve y m ~tol ~budget f ~t0 ~t1 y0` | The state at `t1`                  |
| `Ode.sample`                              | States at given times              |
| `Ode.path`                                | The solution as a function of time |
| `Ode.event`                               | Stop at a crossing                 |

## Next Steps

Continue to [08-stiff-and-dae](../08-stiff-and-dae/).
