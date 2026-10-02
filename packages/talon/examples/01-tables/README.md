# `01-tables`

Build a table from OCaml arrays, read a column back, and run a first query.

```bash
dune exec ./main.exe
```

- `Column.v` and `Column.of_options` make columns of a type from OCaml values,
  `None` being null. `Column.of_tensor` takes a 1-D nx tensor without a copy.
- `Talon.v` puts named columns of equal length into a table.
- `Column.options Kind.float` reads a column back, nulls as `None`.
- `Query.(of_table t |> filter … |> derive …)` describes a table, and
  `Query.run` computes it.
- `Query.values` decodes an expression row by row to OCaml values, and
  `Talon.to_tensor` copies numeric columns into a matrix.

The literals in `temp *. 1.8 +. 32.` take the type of the column they meet,
so `temp_f` is `float32`, like `temp`:

```
table 2 rows × 5 columns
 station  day         temp     rain   temp_f
 string   date        float32  bool   float32
 lima     2024-03-01  22.0000  false  71.6000
 pune     2024-03-01  31.5000  true   88.7000
```
