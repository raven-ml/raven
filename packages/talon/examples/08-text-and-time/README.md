# `08-text-and-time`

Text and temporal expressions over a table of recording sessions.

```bash
dune exec ./main.exe
```

- `Expr.Str` counts and slices Unicode scalar values, matches literal
  patterns, and splits and replaces on literals. `Str.split` gives a
  `list[string]` column.
- `Expr.Temporal.field` reads calendar fields, `floor` truncates to a period,
  `add` moves an instant by a span, and `diff` is the span between two
  instants.
- `Temporal.parse` and `Temporal.format` read and write text in a
  `%Y-%m-%d`-style format.

```
table 3 rows × 4 columns
 weekday  day                   end                   since
 int64    datetime[ms, UTC]     datetime[ms, UTC]     duration[ms]
       5  2024-03-15T00:00:00Z  2024-03-15T11:00:00Z  0s
       6  2024-03-16T00:00:00Z  2024-03-16T11:02:03Z  24h2m3s
       4  2024-03-21T00:00:00Z  2024-03-21T10:03:20Z  143h3m20s
```
