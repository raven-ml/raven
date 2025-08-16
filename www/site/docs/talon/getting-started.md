# Getting Started with Talon

## installation

Talon is part of the Raven ecosystem and will be available through OPAM:

```bash
opam install talon
```

For now, you can build from source:

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

(* Create a dataframe from lists *)
let df = create [
  ("name", Col.string_list ["Alice"; "Bob"; "Charlie"; "Dana"]);
  ("age", Col.int32_list [25l; 30l; 35l; 28l]);
  ("height", Col.float64_list [1.70; 1.80; 1.75; 1.65]);
  ("active", Col.bool_list [true; false; true; true])
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
  match get_column_exn df "bmi" with
  | Col.F64 arr ->
    let arr = Nx.to_array arr in
    Array.map (fun bmi ->
      if bmi < 18.5 then "underweight"
      else if bmi < 25.0 then "normal"
      else if bmi < 30.0 then "overweight"
      else "obese") arr
  | _ -> [||]
in
let df = add_column df "category" 
  (Col.string_list (Array.to_list categories))
```

## row-wise operations

Talon excels at operations across many columns:

```ocaml
(* Sum across multiple columns *)
let df_scores = create [
  ("student", Col.string_list ["Alice"; "Bob"]);
  ("math", Col.float64_list [92.0; 85.0]);
  ("science", Col.float64_list [88.0; 92.0]);
  ("history", Col.float64_list [95.0; 78.0]);
  ("english", Col.float64_list [90.0; 88.0])
]

(* Calculate total score *)
let scores = Row.Agg.sum df_scores 
  ~names:["math"; "science"; "history"; "english"]
let df_scores = add_column df_scores "total" scores

(* Calculate average score *)
let avg = Row.Agg.mean df_scores 
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

Talon provides powerful column selection utilities:

```ocaml
(* Select all numeric columns *)
let numeric_cols = Cols.numeric df

(* Select columns by prefix *)
let price_cols = Cols.with_prefix df "price_"

(* Select columns by suffix *)
let total_cols = Cols.with_suffix df "_total"

(* Select all except specific columns *)
let feature_cols = Cols.except df ["id"; "name"; "timestamp"]

(* Use selectors in operations *)
let row_totals = Row.Agg.sum df ~names:numeric_cols
```

## functional transformations

Use applicative functors for elegant row transformations:

```ocaml
(* Map over multiple columns at once *)
let df = with_columns_map df
  Row.([
    ("sum", Nx.float64, 
      map3 (number "a") (number "b") (number "c") ~f:(fun a b c -> a +. b +. c));
    ("product", Nx.float64, 
      map3 (number "a") (number "b") (number "c") ~f:(fun a b c -> a *. b *. c));
    ("mean", Nx.float64, 
      map3 (number "a") (number "b") (number "c") ~f:(fun a b c -> (a +. b +. c) /. 3.0))
  ])

(* Use applicative operations *)
let df = with_column df "result" Nx.float64
  Row.(map3 (number "x") (number "y") (number "z") 
    ~f:(fun a b c -> a *. b +. c))
```

## data manipulation

### joins

```ocaml
let df1 = create [
  ("id", Col.int32_list [1l; 2l; 3l]);
  ("name", Col.string_list ["Alice"; "Bob"; "Charlie"])
]

let df2 = create [
  ("id", Col.int32_list [2l; 3l; 4l]);
  ("score", Col.float64_list [85.0; 92.0; 88.0])
]

(* Inner join *)
let joined = join df1 df2 ~on:"id" ~how:`Inner ()

(* Left join *)
let left_joined = join df1 df2 ~on:"id" ~how:`Left ()
```

### pivot tables

```ocaml
let sales = create [
  ("date", Col.string_list ["2024-01"; "2024-01"; "2024-02"; "2024-02"]);
  ("product", Col.string_list ["A"; "B"; "A"; "B"]);
  ("amount", Col.float64_list [100.0; 150.0; 120.0; 180.0])
]

let pivoted = pivot sales ~index:"date" ~columns:"product" ~values:"amount" ()
```
