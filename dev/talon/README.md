# Talon

A dataframe library for OCaml with heterogeneous column types, inspired by pandas and polars.

## Features

- **Heterogeneous columns**: Mix numeric tensors, strings, and booleans in a single dataframe
- **Null handling**: Built-in support for missing values across all column types
- **Rich API**: Filtering, grouping, sorting, aggregations, and joins
- **I/O support**: CSV and JSON serialization through sublibraries
- **Nx integration**: Seamless interop with the Nx tensor library

## Installation

```bash
opam install talon
```

## Quick Example

```ocaml
open Talon

(* Create a dataframe *)
let df = create [
  ("name", Col.string_list ["Alice"; "Bob"; "Charlie"]);
  ("age", Col.int32_list [25l; 30l; 35l]);
  ("score", Col.float64_list [85.5; 92.0; 78.5])
]

(* Filter rows *)
let adults = filter_by df Row.(map (int32 "age") ~f:(fun age -> age > 25l))

(* Aggregations *)
let avg_score = Agg.Float.mean df "score"
let total = Agg.Int.sum df "age"

(* Group by computed key *)
let by_grade = group_by df Row.(
  map (float64 "score") ~f:(fun s ->
    if s >= 90.0 then "A" 
    else if s >= 80.0 then "B" 
    else "C"))
```

## CSV and JSON Support

```ocaml
(* CSV I/O *)
let df = Talon_csv.read "data.csv"
Talon_csv.write df "output.csv"

(* JSON I/O *)
let json = Talon_json.to_string ~orient:`Records df
let df2 = Talon_json.from_string ~orient:`Columns json_str
```

## License

ISC