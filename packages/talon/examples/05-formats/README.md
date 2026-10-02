# `05-formats`

Read CSV text with a sniffed format, change a column's type, write the table
back, and read a Parquet file whose row groups the filter can skip.

```bash
dune exec ./main.exe
```

- `Talon_csv.sniff` infers the separator, the header and each column's type.
  `Talon_csv.with_type` declares another type for a column.
- `Talon_csv.encode` writes what `Talon_csv.decode` reads back: a quoted empty
  field is the empty string, an unquoted one is null.
- A field that does not read as its type fails the read with its line, column
  and text:

```
line 3, column 6: "warm": column "temp": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
```

- `Talon_parquet.sniff` reads a file's footer. In the optimized plan, the
  filter is handed to the source, which uses row group statistics to skip
  groups, and stays in the plan for the rows the statistics cannot rule out.

`carriers.parquet` is the file of `02-flights`.
