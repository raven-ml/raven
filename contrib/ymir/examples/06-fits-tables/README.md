# `06-fits-tables`

A binary table holds rows of typed columns. This example writes a table with
text, vector and scalar columns, a unit per column and an undefined cell, then
reads each column back.

```bash
dune exec contrib/ymir/examples/06-fits-tables/main.exe
```

## What You'll Learn

- Describing columns as `Fits.Table.data`: `Array`, `Text`
- Column keywords such as `TUNIT`, and validity masks for undefined cells
- Reading a table's description with `Fits.Table.of_hdu`
- Reading columns as tensors with `values`, row ranges with `~rows`, text with
  `ragged`

## Key Functions

| Function                                 | Purpose                          |
| ---------------------------------------- | -------------------------------- |
| `Fits.Table.hdu h columns`               | A binary table HDU               |
| `Fits.Table.of_hdu`                      | The table's rows and columns     |
| `Fits.Table.values ?rows dtype name hdu` | A numeric column's values        |
| `Fits.Table.validity`                    | Which cells are defined          |
| `Fits.Table.unit`                        | A column's `TUNIT` as a unit     |
| `Fits.Table.ragged`                      | Text and variable-length columns |

## Next Steps

Continue to [07-fits-wcs](../07-fits-wcs/).
