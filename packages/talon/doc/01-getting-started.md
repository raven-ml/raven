# Getting Started with Talon

## installation

Talon is part of the Raven ecosystem and will be available through OPAM:

<!-- $MDX skip -->
```bash
opam install talon
```

For now, you can build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven.git
cd raven
opam install . --deps-only
dune build
```

## your first dataframe

Let's create a simple dataframe and explore its features:

```ocaml
open Talon

(* Create a dataframe from arrays *)
let df = create [
  ("name", Col.string [|"Alice"; "Bob"; "Charlie"; "Dana"|]);
  ("age", Col.int32 [|25l; 30l; 35l; 28l|]);
  ("height", Col.float64 [|1.70; 1.80; 1.75; 1.65|]);
  ("weight", Col.float64 [|65.0; 82.0; 90.0; 70.0|]);
  ("active", Col.bool [|true; false; true; true|])
]

(* Check the shape *)
let () = Printf.printf "Rows: %d, Columns: %d\n"
  (num_rows df) (num_columns df)

(* Print the dataframe *)
let () = print df
```

## adding computed columns

One of Talon's strengths is type-safe row operations:

```ocaml
(* Calculate BMI: weight / (height^2) *)
let df = with_column df "bmi" Nx.float64
  Row.(map2 (number "weight") (number "height")
    ~f:(fun w h -> w /. (h ** 2.)))

(* Add a categorical column based on BMI *)
let categories =
  match to_array Nx.float64 df "bmi" with
  | Some arr ->
    Array.map (fun bmi ->
      if bmi < 18.5 then "underweight"
      else if bmi < 25.0 then "normal"
      else if bmi < 30.0 then "overweight"
      else "obese") arr
  | None -> [||]

let df = add_column df "category" (Col.string categories)
```

## row-wise operations

Talon excels at operations across many columns:

```ocaml
(* Sum across multiple columns *)
let df_scores = create [
  ("student", Col.string [|"Alice"; "Bob"|]);
  ("math", Col.float64 [|92.0; 85.0|]);
  ("science", Col.float64 [|88.0; 92.0|]);
  ("history", Col.float64 [|95.0; 78.0|]);
  ("english", Col.float64 [|90.0; 88.0|])
]

(* Calculate total score *)
let scores = Agg.row_sum df_scores
  ~names:["math"; "science"; "history"; "english"]
let df_scores = add_column df_scores "total" scores

(* Calculate average score *)
let avg = Agg.row_mean df_scores
  ~names:["math"; "science"; "history"; "english"]
let df_scores = add_column df_scores "average" avg
```

## filtering and sorting

```ocaml
(* Filter students with average >= 90 *)
let top_students = filter_by df_scores
  Row.(map (number "average") ~f:(fun avg -> avg >= 90.0))

(* Sort by total score descending *)
let sorted = sort_values ~ascending:false df_scores "total"
```

## working with column selectors

Talon provides composable column selection via `select_columns`:

```ocaml
(* Select all numeric columns *)
let numeric_cols = select_columns df `Numeric

(* Select columns by prefix using standard list operations *)
let price_cols =
  List.filter (fun n -> String.starts_with ~prefix:"price_" n) (column_names df)

(* Select all except specific columns *)
let feature_cols =
  List.filter (fun n -> not (List.mem n ["id"; "name"; "timestamp"]))
    (column_names df)

(* Use selectors in operations *)
let row_totals = Agg.row_sum df ~names:numeric_cols
```

## functional transformations

Use applicative functors for elegant row transformations:

```ocaml
let df = create [
  ("a", Col.float64 [|1.0; 2.0; 3.0|]);
  ("b", Col.float64 [|4.0; 5.0; 6.0|]);
  ("c", Col.float64 [|7.0; 8.0; 9.0|]);
  ("x", Col.float64 [|10.0; 20.0; 30.0|]);
  ("y", Col.float64 [|0.5; 0.5; 0.5|]);
  ("z", Col.float64 [|1.0; 2.0; 3.0|])
]

(* Add multiple computed columns *)
let df = with_column df "sum" Nx.float64
  Row.(map3 (number "a") (number "b") (number "c")
    ~f:(fun a b c -> a +. b +. c))

let df = with_column df "product" Nx.float64
  Row.(map3 (number "a") (number "b") (number "c")
    ~f:(fun a b c -> a *. b *. c))

(* Use applicative operations *)
let df = with_column df "result" Nx.float64
  Row.(map3 (number "x") (number "y") (number "z")
    ~f:(fun a b c -> a *. b +. c))
```

## data manipulation

### joins

```ocaml
let df1 = create [
  ("id", Col.int32 [|1l; 2l; 3l|]);
  ("name", Col.string [|"Alice"; "Bob"; "Charlie"|])
]

let df2 = create [
  ("id", Col.int32 [|2l; 3l; 4l|]);
  ("score", Col.float64 [|85.0; 92.0; 88.0|])
]

(* Inner join *)
let joined = join df1 df2 ~on:"id" ~how:`Inner ()

(* Left join *)
let left_joined = join df1 df2 ~on:"id" ~how:`Left ()
```

### pivot tables

```ocaml
let sales = create [
  ("date", Col.string [|"2024-01"; "2024-01"; "2024-02"; "2024-02"|]);
  ("product", Col.string [|"A"; "B"; "A"; "B"|]);
  ("amount", Col.float64 [|100.0; 150.0; 120.0; 180.0|])
]

let pivoted = pivot sales ~index:"date" ~columns:"product" ~values:"amount" ()
```

## loading and saving data

Use `Talon_csv` to control types and null handling on I/O:

<!-- $MDX skip -->
```ocaml
let df = Talon_csv.read ~dtype_spec:["id", `Int64; "score", `Float64] "data.csv"
let () = Talon_csv.write "clean.csv" df
```

## Next Steps
Check out the [Comparison with Pandas](/docs/talon/pandas-comparison/) to see how Talon's functional approach differs from Pandas.
