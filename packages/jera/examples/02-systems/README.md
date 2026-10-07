# `02-systems`

A system maps a value to a value of the same structure. This example solves one
system with Newton's, Broyden's and Anderson's methods, and many small systems
at once with `System.lanes`.

```bash
dune exec packages/jera/examples/02-systems/main.exe
```

## What You'll Learn

- `System.newton` with the Jacobian's product from `Rune.jvp'`
- `System.broyden`, with no derivative
- `System.anderson` for fixed points
- `System.lanes`: one small system per row

## Key Functions

| Function                                        | Purpose                    |
| ----------------------------------------------- | -------------------------- |
| `System.solve x m ~linear ~tol ~budget f guess` | A zero of `f` near `guess` |
| `System.newton ~derivative`                     | Newton's method            |
| `System.broyden`                                | Broyden's method           |
| `System.anderson ~memory`                       | Anderson mixing            |
| `System.lanes ~tol ~budget ~jacobian f guess`   | A zero per lane            |

## Next Steps

Continue to [03-minimization](../03-minimization/).
