# `10-custom-source`

Write a source of your own and fold a query over its batches.

```bash
dune exec ./main.exe
```

- `Source.v` takes a name, a schema and a function from a request to parts,
  each of which opens a reader of batches. A batch holds the columns the
  request asks for.
- `~pushdown` answers, for each conjunct of a filter, whether the source
  applies it. This source applies comparisons of `i` with an integer exactly,
  so the optimized plan hands `i >= 2500` to the source and keeps the modulo:

```
query → i int64, square int64
derive ["square" := i * i]
└ filter (i mod 7 = 0)
  └ counter (1 column, 10000 rows) ~filters:[i >= 2500]
```

- `~sorted` claims the order of the rows, which talon checks as they arrive.
- `Query.fold` runs the query batch by batch, so the result is never held
  whole.
