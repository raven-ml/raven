# Talon Examples

Each example is a program that builds and runs with the test suite, and
`main.expected` holds what it prints. Run one from its directory, since the
examples read their data files by relative path:

```bash
cd packages/talon/examples/02-flights
dune exec ./main.exe
```

| Example | Shows | Key functions |
|---|---|---|
| [`01-tables`](./01-tables/) | Tables from OCaml values, a first query | `Talon.v`, `Column.v`, `Query.run`, `Query.values` |
| [`02-flights`](./02-flights/) | A pipeline over a CSV and a Parquet file | `Talon_csv.file`, `Query.aggregate`, `Query.join` |
| [`03-expressions`](./03-expressions/) | Row expressions, reductions, frames | `Expr.over`, `Expr.shift`, `Expr.rank` |
| [`04-joins`](./04-joins/) | Join kinds and match assertions | `Join.eq`, `Join.kind`, `~each_left` |
| [`05-formats`](./05-formats/) | Reading and writing CSV, reading Parquet | `Talon_csv.sniff`, `Talon_csv.encode`, `Talon_parquet.sniff` |
| [`06-mistakes`](./06-mistakes/) | Plan problems and failures in data | `Invalid_argument`, `Error.pp` |
| [`07-features`](./07-features/) | From a file to a feature matrix | `Talon.take`, `Talon.to_tensor`, `Column.to_tensor` |
