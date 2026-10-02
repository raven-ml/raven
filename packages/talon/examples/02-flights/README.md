# `02-flights`

The pipeline of the guide: read flights from a CSV file and carriers from a
Parquet file, keep the late departures, average them per carrier, join each
carrier's name and sort.

```bash
dune exec ./main.exe
```

- `Talon_csv.file ~nulls:[ "NA" ]` sniffs the file's columns and types, `NA`
  reading as null. `Talon_parquet.file` reads the types from the footer.
- `~each_left:One` asserts that every carrier finds exactly one name.
- `Query.pp` prints the plan, and `Query.optimize` the plan a run runs, which
  reads two of the CSV file's eleven columns.

```
table 7 rows × 4 columns
 carrier  mean_delay  flights  name
 string   float64     int64    string
 DL          38.0000        5  Delta Air Lines Inc.
 HA          36.6667        3  Hawaiian Airlines Inc.
 B6          31.0000        1  JetBlue Airways
 OO          29.0000        3  ∅
 F9          27.0000        2  Frontier Airlines Inc.
 AA          27.0000        1  American Airlines Inc.
 WN          26.0000        1  Southwest Airlines Co.
```

The flights are made up. `uv run data.py` writes both files again.
