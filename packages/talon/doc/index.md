# Talon

Talon is a library of tables: named, typed columns of equal length. A
missing value is null, a state of its own that no number stands for, and rows
have no labels. Columns are laid out over nx buffers, so a numeric column
becomes an nx tensor without a copy.

## Quick Start

Read flights from a CSV file and carriers from a Parquet file, average the
late departures per carrier, and name each carrier:

```ocaml
open Talon

let ( let* ) = Result.bind
let delay = Col.int "dep_delay"

let late_by_carrier () =
  let* flights = Talon_csv.file ~nulls:[ "NA" ] "flights.csv" in
  let* carriers = Talon_parquet.file "carriers.parquet" in
  Query.(
    of_source flights
    |> filter Expr.(delay > int 15)
    |> aggregate ~by:[ "carrier" ]
         Expr.[ "mean_delay" := mean delay; "flights" := rows ]
    |> join ~on:(Join.keys [ "carrier" ]) ~each_left:One (of_source carriers)
    |> sort [ Order.desc "mean_delay" ]
    |> run)
```

```text
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

`∅` is null: the carriers file has no name for `OO`.

## How Talon Works

- **A table is data.** `Talon.t` holds columns, each a typed array with a
  bitmap of the rows that are null.
- **A query is a description.** `Query.t` is a table or a file transformed by
  verbs: `select`, `derive`, `filter`, `sort`, `slice`, `aggregate`, `join`
  and `append`. Building one reads nothing, and `Query.run` computes it.
- **Expressions are typed.** `Col.int "dep_delay"` names a column and reads it
  as `int`. An expression's type also says whether it has one value per row
  or one per group, so a column where a reduction belongs does not compile.
- **Plans fail before data.** Each verb checks its expressions against its
  input's columns when it is applied and reports every problem at once.
- **Failures in data are values.** A malformed file or a value that breaks a
  query's contract is an `Error`, which names the file, line, plan step or
  row.

## Libraries

| Library | Module | Contents |
|---|---|---|
| `talon` | `Talon` | Tables, columns, types, expressions and queries |
| `talon.csv` | `Talon_csv` | CSV files: sniffing, reading and writing |
| `talon.parquet` | `Talon_parquet` | Parquet files, read mapped in memory |

## Next Steps

- [Getting Started](01-getting-started.md): tables, a first query, reading
  files and handling mistakes
- [Expressions and Frames](02-expressions-and-frames.md): types, nulls, verbs,
  reductions and windows over groups
- [Joins](03-joins.md): conditions, kinds and assertions on matches
- [Formats](04-formats.md): CSV, Parquet and your own sources
- [pandas Comparison](05-pandas-comparison.md): side-by-side reference
- [Examples](../examples/01-tables/README.md): runnable programs, from a
  first table to a feature matrix
