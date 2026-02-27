# Row Operations

Talon's `Row` module is an applicative functor for type-safe, row-wise computations. This is Talon's most distinctive feature compared to pandas, where column operations are typically stringly-typed.

## The Row Applicative

A `'a row` is a computation that, when executed against a dataframe, produces a value of type `'a` for each row. You build row computations declaratively, then apply them with `with_column`, `filter_by`, or `map`.

### Column accessors

Extract typed values from named columns:

<!-- $MDX skip -->
```ocaml
open Talon

Row.float64 "height"         (* float row — from a float64 column *)
Row.int32 "age"              (* int32 row — from an int32 column *)
Row.string "name"            (* string row — from a string column *)
Row.bool "active"            (* bool row — from a bool column *)
Row.number "score"           (* float row — coerces any numeric type to float *)
```

`number` is the most flexible accessor — it works with any numeric column type (int32, int64, float32, float64) and coerces to `float`.

### Transforming with map

Apply a function to each row's value:

<!-- $MDX skip -->
```ocaml
(* Double every value *)
let doubled = Row.map (Row.float64 "price") ~f:(fun x -> x *. 2.)

(* Combine two columns *)
let full_name =
  Row.map2 (Row.string "first") (Row.string "last")
    ~f:(fun f l -> f ^ " " ^ l)

(* Three columns at once *)
let weighted_score =
  Row.map3
    (Row.number "math") (Row.number "science") (Row.number "english")
    ~f:(fun m s e -> m *. 0.4 +. s *. 0.35 +. e *. 0.25)
```

## Adding Computed Columns

`with_column` applies a row computation to create a new column:

<!-- $MDX skip -->
```ocaml
(* BMI = weight / height² *)
let df = with_column df "bmi" Nx.Float64
  Row.(map2 (number "weight") (number "height")
    ~f:(fun w h -> w /. (h *. h)))

(* Category from numeric value *)
let df = with_column df "grade" Nx.Int32
  Row.(map (number "score") ~f:(fun s ->
    if s >= 90. then 4l
    else if s >= 80. then 3l
    else if s >= 70. then 2l
    else 1l))
```

For multiple computed columns in one pass, use `with_columns_map`:

<!-- $MDX skip -->
```ocaml
let df = with_columns_map df [
  ("bmi", Nx.Float64,
    Row.(map2 (number "weight") (number "height")
      ~f:(fun w h -> w /. (h *. h))));
  ("is_tall", Nx.Int32,
    Row.(map (number "height") ~f:(fun h ->
      if h > 1.80 then 1l else 0l)));
]
```

## Filtering

`filter_by` keeps rows where a row computation returns `true`:

<!-- $MDX skip -->
```ocaml
(* Simple filter *)
let adults = filter_by df
  Row.(map (int32 "age") ~f:(fun a -> a >= 18l))

(* Compound filter *)
let qualified = filter_by df
  Row.(map2 (number "score") (bool "active")
    ~f:(fun s a -> s >= 80. && a))
```

## Working with Multiple Columns

### numbers and map_list

For operations across a dynamic list of columns:

<!-- $MDX skip -->
```ocaml
(* Average across all score columns *)
let score_cols = ["math"; "science"; "english"; "history"] in
let avg = Row.map_list (Row.numbers score_cols) ~f:(fun scores ->
  List.fold_left (+.) 0. scores /. float (List.length scores))

let df = with_column df "average" Nx.Float64 avg
```

### fold_list

More memory-efficient than `map_list` for reductions:

<!-- $MDX skip -->
```ocaml
(* Total across quarterly columns *)
let total = Row.fold_list
  (Row.numbers ["q1"; "q2"; "q3"; "q4"])
  ~init:0. ~f:(+.)
```

### sequence

Collect values from multiple computations into a list:

<!-- $MDX skip -->
```ocaml
let all_scores = Row.sequence [
  Row.number "math";
  Row.number "science";
  Row.number "english";
]
(* all_scores : float list row *)
```

## Row-wise Aggregations (Row.Agg)

`Row.Agg` provides efficient horizontal aggregations — computing statistics across columns within each row:

<!-- $MDX skip -->
```ocaml
(* Sum across score columns *)
let total = Row.Agg.sum (Row.numbers ["math"; "science"; "english"])

(* Mean, ignoring nulls *)
let avg = Row.Agg.mean ~skipna:true (Row.numbers ["q1"; "q2"; "q3"; "q4"])

(* Min and max across columns *)
let best = Row.Agg.max (Row.numbers ["test1"; "test2"; "test3"])
let worst = Row.Agg.min (Row.numbers ["test1"; "test2"; "test3"])

(* Weighted sum *)
let weights = Nx.create Nx.Float64 [|3|] [|0.4; 0.35; 0.25|]
let weighted = Row.Agg.dot weights (Row.numbers ["math"; "science"; "english"])

(* Boolean reductions *)
let all_pass = Row.Agg.all [
  Row.(map (number "math") ~f:(fun x -> x >= 50.));
  Row.(map (number "science") ~f:(fun x -> x >= 50.));
]
```

These are more efficient than `map_list` followed by manual reduction because they use vectorized Nx operations internally.

## Row Metadata

### index

Access the current row index:

<!-- $MDX skip -->
```ocaml
let df = with_column df "row_num" Nx.Int32
  Row.(map index ~f:Int32.of_int)
```

## Nullable Columns

For columns that may contain null values, use the `_opt` accessors:

<!-- $MDX skip -->
```ocaml
(* Returns float option row instead of float row *)
let maybe_score = Row.float64_opt "score"

(* Handle nulls explicitly *)
let filled = Row.map maybe_score ~f:(function
  | Some v -> v
  | None -> 0.)
```

## Column-wise Aggregations (Agg)

For completeness, Talon also provides column-wise aggregations via the top-level `Agg` module. These produce scalar results from entire columns:

<!-- $MDX skip -->
```ocaml
(* Column-wise: single value from entire column *)
let avg_score = Agg.Float.mean df "score"
let total = Agg.Float.sum df "revenue"
let maximum = Agg.Float.max df "temperature"
let row_count = Agg.count df "name"
```

## Next Steps

- [Getting Started](/docs/talon/getting-started/) — basic DataFrame creation and manipulation
- [pandas Comparison](/docs/talon/pandas-comparison/) — side-by-side reference
