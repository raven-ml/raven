# Talon

Tables for OCaml: named, typed columns of equal length, queried with typed
expressions, read from CSV and Parquet files. Talon is part of the
[Raven](https://github.com/raven-ml/raven) ecosystem and is built on
[Nx](../nx/).

## Features

- **Typed columns**: integers, floats, booleans, text, categoricals, bytes,
  dates, datetimes, durations, lists, records and tensors, laid out over nx
  buffers
- **Nulls**: one null state for every type, distinct from NaN
- **Queries as plans**: `select`, `derive`, `filter`, `sort`, `slice`,
  `aggregate`, `join` and `append`, optimized before they run
- **Typed expressions**: column handles, arithmetic, comparisons, reductions,
  windows over groups with `over`, text and time
- **Checked plans**: every problem of a verb reported at once, before any data
  is read; failures in data returned as `Error` values with their location
- **Formats**: `talon.csv` reads and writes CSV, `talon.parquet` reads Parquet
  files mapped in memory, skipping row groups by their statistics
- **Nx interop**: a numeric column becomes a tensor without a copy, and
  `to_tensor` builds a feature matrix

## Quick Start

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

let () =
  match late_by_carrier () with
  | Ok t -> Format.printf "%a@." Talon.pp t
  | Error e -> Format.eprintf "%a@." Error.pp e
```

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

## Installation

```bash
opam install talon
```

## Documentation

- [Guide](doc/index.md): getting started, expressions and frames, joins,
  formats, and a comparison with pandas
- [Examples](examples/): runnable programs

## License

ISC
