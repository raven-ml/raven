# `07-features`

Turn flights into a feature matrix and labels for a model.

```bash
dune exec ./main.exe
```

- `over (mean x)` and `over (std x)` put whole-table statistics on each row,
  so `standardize` is one expression.
- `over ~by:[ "carrier" ] (mean delay)` is each carrier's mean delay, on each
  of its rows.
- `Talon.take` with `Nx.Rng.permutation` shuffles every column with one
  permutation.
- `Talon.to_tensor Nx.float32` copies the feature columns into one
  `(rows, 3)` matrix, converting each value. `Column.to_tensor` returns the
  label column's own buffer without a copy.

`flights.csv` is the file of `02-flights`.
