# `01-roots`

Root finders are elementwise: each element of the input is its own problem with
its own status. This example brackets, runs Newton's method, reads a partly
failed answer, and differentiates a zero in its function's parameter.

```bash
dune exec packages/jera/examples/01-roots/main.exe
```

## What You'll Learn

- Bracketing solves with `Root.bracket` and a tolerance in ulps
- Newton's method with `Root.newton`, steered by a slope
- Reading answers: `Solution.get`, `best`, `ok`, `is`, `evaluations`, `pp`
- The implicit derivative of a zero through `Rune.grad'`

## Key Functions

| Function                               | Purpose                                    |
| -------------------------------------- | ------------------------------------------ |
| `Root.bracket ~tol f ~lo ~hi`          | A zero of `f` in each bracket              |
| `Root.newton ~tol ~budget ~slope f x0` | A zero near `x0`                           |
| `Solution.get s`                       | The answer, if every lane converged        |
| `Solution.is st s`                     | Where the status is `st`                   |
| `Tol.ulps k`                           | A tolerance of `k` units in the last place |

## Next Steps

Continue to [02-systems](../02-systems/).
