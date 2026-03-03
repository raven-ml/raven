# talon

Talon provides type-safe DataFrames for OCaml, built on Nx arrays. It is the Raven ecosystem's equivalent of pandas and Polars.

## Features

- **Heterogeneous columns** — mix strings, floats, integers, and booleans
- **Applicative Row operations** — type-safe, composable row-wise computations
- **First-class null handling** — explicit null masks for numeric columns, Option types for strings and bools
- **Vectorized aggregations** — column-wise and row-wise reductions backed by Nx
- **CSV I/O** — read and write CSV files with auto-detection
- **Built on Nx** — columns are 1-D Nx tensors

## Quick Start

```ocaml
open Talon

let () =
  let df = create [
    ("name", Col.string [|"Alice"; "Bob"; "Charlie"|]);
    ("age", Col.int32 [|25l; 30l; 35l|]);
    ("score", Col.float64 [|92.5; 87.3; 95.1|]);
  ] in
  print df
```
Shape: (3, 3)
name	age	score
Alice	25	92.5
Bob	30	87.3
Charlie	35	95.1

## Next Steps

- [Getting Started](/docs/talon/getting-started/) — installation, creating and inspecting DataFrames
- [Row Operations](/docs/talon/row-operations/) — the applicative Row system, computed columns, filtering
- [pandas Comparison](/docs/talon/pandas-comparison/) — side-by-side reference
