# `11-nested-columns`

Columns of lists, tensors, records and an extension type.

```bash
dune exec ./main.exe
```

- A `list[int32]` column is offsets into one flat buffer.
  `Column.ragged Nx.int32` reads both as an `Nx_ragged.t` without a copy.
- `Column.of_tensor` on a `(rows, 2, 2)` tensor makes a column holding one
  2×2 tensor per row.
- `Record.add` builds record cells field by field, each a value or null.
- `Ext.v` declares an extension type, here lengths in metres stored as
  `float64`. No handle reads an extension column: `Ext.col` does, through the
  declaration, and `Query.values` decodes with it.

```
id int8, tokens list[int32], image tensor[float32, 2×2], point record[x float64, y float64], length ext[example.metres, float64]
```
