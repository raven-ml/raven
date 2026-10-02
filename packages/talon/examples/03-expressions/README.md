# `03-expressions`

Expressions over a table of training losses: values per row, reductions per
group, and reductions put back on the rows with `over`.

```bash
dune exec ./main.exe
```

- `over ~by:[ "run" ] ~order:[ Order.asc "step" ]` evaluates an expression in
  each run's rows, ordered by step. `shift 1 loss` is the previous evaluation's
  loss and `rank loss` its rank in its run.
- In `aggregate`, each output reduces one group. `over (min loss)` inside it
  is the group's minimum on each of its rows, so
  `first (if_ (loss = over (min loss)) step null)` is the step of the best loss.
- Nulls propagate through arithmetic, reductions skip them, and `is_null` tests
  for them.

```
table 2 rows × 5 columns
 run     evals  best     best_step  improved
 string  int64  float64  int32      bool
 a           4  1.55000        400  true
 b           3  1.52000        400  true
```
