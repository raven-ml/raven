# talon

Fast and elegant dataframes for OCaml with type-safe row operations.

## overview

Talon brings the power of pandas and Polars to OCaml with compile-time type safety. Built on top of Nx arrays, it provides efficient operations for wide dataframes while maintaining OCaml's strong typing guarantees.

## key features

- **Type-safe heterogeneous columns** - Mix strings, floats, integers, and booleans safely
- **Optimized for wide dataframes** - Efficient handling of 100+ columns
- **Functional row operations** - Use applicative functors for elegant transformations
- **pandas-compatible API** - All the operations you know: join, merge, pivot, melt, groupby
- **Built on Nx** - Leverages fast array operations with multiple backend support

## quick example

```ocaml
open Talon

(* Create a dataframe *)
let df = create [
  ("name", Col.string_list ["Alice"; "Bob"; "Charlie"]);
  ("age", Col.int32_list [25l; 30l; 35l]);
  ("score", Col.float64_list [92.5; 87.3; 95.1])
]

(* Add computed columns *)
let pass_values = 
  match to_float64_array df "score" with
  | Some scores -> Array.map (fun s -> s >= 90.0) scores
  | None -> [||]
let df = add_column df "pass" (Col.bool_list (Array.to_list pass_values))

(* Row-wise aggregations *)
let df_scores = create [
  ("score1", Col.float64_list [85.0; 90.0; 88.0]);
  ("score2", Col.float64_list [88.0; 92.0; 85.0]);
  ("score3", Col.float64_list [91.0; 87.0; 92.0])
]
let total = Row.Agg.sum df_scores ~names:["score1"; "score2"; "score3"]

(* Filter and sort *)
let df = filter_by df Row.(bool "pass")
let df = sort_values ~ascending:false df "score"
```

## learn more

- [Getting Started](/docs/talon/getting-started/) - Installation and first steps
- [Comparison with Pandas](/docs/talon/pandas-comparison/) - Coming from Python
