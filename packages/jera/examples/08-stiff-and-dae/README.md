# `08-stiff-and-dae`

On a stiff field an explicit method spends its budget; `Ode.kvaerno5` steps at
the tolerance's pace. A singular mass matrix turns equations into constraints.

```bash
dune exec packages/jera/examples/08-stiff-and-dae/main.exe
```

## What You'll Learn

- Recognising stiffness: an explicit solve ends `Budget_spent`
- `Ode.kvaerno5` with the field's Jacobian product
- A differential-algebraic system through `~mass`

## Key Functions

| Function                                | Purpose                                      |
| --------------------------------------- | -------------------------------------------- |
| `Ode.kvaerno5 ?mass ~linear derivative` | An implicit method for stiff fields and DAEs |
| `Solution.is Budget_spent s`            | Where the budget ran out                     |

## Next Steps

Continue to [09-sdes](../09-sdes/).
