# Talon vs. pandas – A Practical Comparison

This guide explains how Talon's dataframe model relates to Python's [pandas](https://pandas.pydata.org/), focusing on:

* How core concepts map (DataFrame, Series, dtypes, nulls)
* Where the APIs feel similar vs. deliberately different
* How to translate common pandas patterns into Talon

If you already use pandas, this should be enough to become productive in Talon quickly.

---

## 1. Big-Picture Differences

| Aspect          | pandas (Python)                                           | Talon (OCaml)                                                       |
| --------------- | --------------------------------------------------------- | ------------------------------------------------------------------- |
| Language        | Dynamic, interpreted                                      | Statically typed, compiled                                          |
| Core table type | `pd.DataFrame`                                            | `Talon.t`                                                           |
| Column type     | `pd.Series`                                               | `Talon.Col.t` (abstract)                                            |
| Numeric backend | NumPy                                                     | Nx                                                                  |
| Typing model    | Dtypes checked at runtime                                 | Dtypes tracked at type & value-level via GADTs                      |
| Null semantics  | NaN, `pd.NA`, object `None`, nullable dtypes              | Explicit null masks for numerics, `option` values for strings/bools |
| Row-wise logic  | `DataFrame.apply`, vectorized ops                         | `Row` applicative combinators, compiled to loops                    |
| Groupby / joins | `groupby`, `merge`, `join`                                | `group_by`, `join`                                                  |
| Reshaping       | `pivot`, `melt`, `stack`, `unstack`                       | `pivot`, `melt`                                                     |
| I/O             | `read_csv`, `to_csv`, `read_json`, `to_json`, etc.        | `Talon_csv.read/write`                                              |
| Mutability      | DataFrames mutably changed by convention (but not always) | Immutable `Talon.t`; operations return new dataframes               |

**Talon semantics to know (read once):**
- Null keys never match in joins; inner joins drop null-key rows.
- Row-wise reducers default to `skipna=true`; set `~skipna:false` to propagate nulls.
- `Row.number` coerces numerics to float64; use `Row.float64`/`int32`/`int64` to avoid upcasting.
- Dataframes are immutable; every operation returns a new `Talon.t`.

---

## 2. Core Data Model: DataFrame & Column

### 2.1 DataFrame

**pandas**

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age":  [25, 30],
})
```

**Talon**

```ocaml
open Talon

let df =
  create
    [
      ("name", Col.string [| "Alice"; "Bob" |]);
      ("age",  Col.int32   [| 25l; 30l |]);
    ]
```

Key parallels:

* Both are *row-oriented* logical tables with named, homogeneous columns.
* `Talon.t` is immutable; every transformation returns a new dataframe.

### 2.2 Column Representation

**pandas** columns are `Series`, dynamically typed: dtype is metadata, but Python won't stop you from doing `df["name"] + 1` until runtime.

**Talon** columns are opaque (`Col.t`), internally storing:

* Numeric data backed by 1D Nx tensors with an optional null mask
* String data as `string option array`
* Boolean data as `bool option array`

This gives Talon:

* Runtime knowledge of *exact* numeric dtype.
* Explicit representation of nulls instead of overloading special values.

---

## 3. Dtypes & Type Safety

### 3.1 Creating Columns

**pandas**

```python
pd.Series([1.0, 2.0, 3.0], dtype="float64")
pd.Series([1, 2, 3], dtype="int32")
pd.Series(["a", "b"], dtype="string[python]")
```

**Talon**

```ocaml
let _ = Col.float64 [| 1.0; 2.0; 3.0 |]
let _ = Col.int32   [| 1l; 2l; 3l |]
let _ = Col.string  [| "a"; "b" |]
```

Nullable equivalents:

```ocaml
let _ = Col.float64_opt [| Some 1.0; None; Some 3.0 |]
let _ = Col.int32_opt   [| Some 42l; None; Some 100l |]
let _ = Col.string_opt  [| Some "x"; None; Some "y" |]
let _ = Col.bool_opt    [| Some true; None; Some false |]
```

### 3.2 Consequences of Strong Typing

* Talon will fail fast if you try to use a column with the wrong type accessor (e.g. `Row.int32 "name"`).
* Numeric aggregations (`Agg.sum`, `Agg.mean`, etc.) coerce any numeric column to float, so you don't need separate modules for float vs int aggregations.
* String and boolean operations live in dedicated sub-modules (`Agg.String`, `Agg.Bool`).

Where pandas often says "this is probably fine, let's try", Talon tends to say "pick the right function for this dtype".

---

## 4. Null / Missing Data Semantics

This is one of the biggest conceptual differences.

### 4.1 Representation

**pandas**

* Historically: NaN for numeric, `None`/`np.nan` for object; increasingly `pd.NA` and nullable dtypes.
* Null semantics vary slightly by dtype (especially between legacy and new nullable dtypes).

**Talon**

* **Numeric columns**: explicit optional boolean mask; payload values (including `nan`, `Int32.min_int`, etc.) are treated as *regular data* unless masked.
* **String / Bool columns**: `None` = null, `Some v` = non-null.

So the "source of truth" for missingness is:

* Numeric: the mask attached to the column.
* String/Bool: `option` constructors.

### 4.2 Column-level utilities

**pandas**

```python
df["score"].isna()
df["score"].notna()
df["score"].dropna()
df["score"].fillna(0.0)
```

**Talon**

<!-- $MDX skip -->
```ocaml
(* Column-level *)
let has_nulls   = Col.has_nulls col
let null_count  = Col.null_count col
let no_nulls    = Col.drop_nulls col

(* Fill nulls with a single-element column of the same type *)
let filled = Col.fill_nulls col ~value:(Col.float64 [| 0.0 |])
```

### 4.3 DataFrame-level null handling

**pandas**

```python
df.dropna()                 # drop rows with any null
df.dropna(subset=["col1"])  # only check some columns
df["col"].isna()            # mask
df["col"].fillna(0)
```

**Talon**

<!-- $MDX skip -->
```ocaml
let cleaned   = drop_nulls df              (* all columns *)
let cleaned_x = drop_nulls ~subset:["x"] df

let col = get_column_exn df "x"
let has = Col.has_nulls col
let n   = Col.null_count col

let df' = fill_null df "score" ~with_value:(`Float 0.0)
```

`drop_nulls` and `fill_null` are the closest conceptual equivalents to `dropna` and `fillna`.

---

## 5. Constructing DataFrames (and I/O)

### 5.1 From in-memory data

**pandas**

```python
df = pd.DataFrame(
    {"name": ["Alice", "Bob"], "age": [25, 30], "score": [85.5, 92.0]},
)
```

**Talon**

```ocaml
let df =
  create
    [
      ("name",  Col.string  [| "Alice"; "Bob" |]);
      ("age",   Col.int32   [| 25l; 30l |]);
      ("score", Col.float64 [| 85.5; 92.0 |]);
    ]
```

From Nx tensors:

<!-- $MDX skip -->
```ocaml
let t1 = Nx.create Nx.float64 [| 3 |] [| 1.0; 2.0; 3.0 |]
let t2 = Nx.create Nx.float64 [| 3 |] [| 4.0; 5.0; 6.0 |]
let df = of_tensors ~names:[ "x"; "y" ] [ t1; t2 ]
```

From a 2D tensor:

<!-- $MDX skip -->
```ocaml
let t = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
let df = of_nx ~names:[ "x"; "y"; "z" ] t
```

### 5.2 CSV I/O

**pandas**

```python
df = pd.read_csv("data.csv")
df.to_csv("out.csv", index=False)
```

**Talon**

<!-- $MDX skip -->
```ocaml
let df =
  Talon_csv.read
    ~sep:','
    ~na_values:[""; "NA"; "N/A"; "null"; "NULL"]
    "data.csv"

let () = Talon_csv.write ~sep:',' "out.csv" df
```

From/to string:

<!-- $MDX skip -->
```ocaml
let csv = Talon_csv.to_string df
let df' = Talon_csv.of_string csv
```

---

## 6. Selecting & Inspecting Columns

### 6.1 Column discovery

**pandas**

```python
df.columns
df.dtypes
len(df)
df.empty
```

**Talon**

```ocaml
let (rows, cols) = shape df
let n_rows       = num_rows df
let n_cols       = num_columns df
let names        = column_names df

let types : (string * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String | `Other ]) list =
  column_types df

let is_empty = is_empty df
```

Type-based selection (roughly `df.select_dtypes`):

```ocaml
let numeric_cols = select_columns df `Numeric   (* floats + ints *)
let float_cols   = select_columns df `Float     (* float32/64 *)
let int_cols     = select_columns df `Int
let bool_cols    = select_columns df `Bool
let string_cols  = select_columns df `String
```

Name-based selection uses standard list operations:

<!-- $MDX skip -->
```ocaml
let prefixed = List.filter (fun n -> String.starts_with ~prefix:"temp_" n)
  (column_names df)
let suffixed = List.filter (fun n -> String.ends_with ~suffix:"_score" n)
  (column_names df)
let others = List.filter (fun n -> not (List.mem n ["id"; "target"]))
  (column_names df)
```

### 6.2 Getting and manipulating single columns

**pandas**

```python
age_series = df["age"]
df["ratio"] = df["a"] / df["b"]
```

**Talon**

```ocaml
let age_col   = get_column_exn df "age"

let df' =
  add_column df "ratio"
    (Col.float64 [| 1.0; 2.0 |])  (* or use with_column, see below *)
```

Drop / rename:

```ocaml
let df_no_age   = drop_column df "age"
let df_relabel  = rename_column df ~old_name:"age" ~new_name:"age_years"
let df_pruned   = drop_columns df [ "name"; "score" ]
```

Select subsets:

```ocaml
let df_small  = select df [ "name"; "age" ]        (* error if missing *)
let df_loose  = select ~strict:false df [ "name"; "maybe" ] (* ignores missing *)
let df_reordered = reorder_columns df [ "age"; "name" ]
```

Extract as arrays, like `df["x"].to_numpy()`:

<!-- $MDX skip -->
```ocaml
let xs : float array option    = to_array Nx.float64 df "x"
let ys : int32 array option    = to_array Nx.int32   df "y"
let zs : string option array option = to_string_array df "z"
```

---

## 7. Row-wise Computations

pandas often uses:

* vectorized operations (`df["a"] + df["b"]`)
* `DataFrame.apply` / `Series.apply`.

Talon uses the `Row` applicative to define per-row computations.

### 7.1 Basic accessors

**pandas**

```python
# per-row access
df.apply(lambda row: row["a"] + row["b"], axis=1)
```

**Talon**

<!-- $MDX skip -->
```ocaml
open Row

let sum_ab : float row =
  map2 (float64 "a") (float64 "b") ~f:( +. )
```

Use this with `map` / `with_column`:

<!-- $MDX skip -->
```ocaml
let df' =
  with_column df "sum_ab" Nx.float64 sum_ab
```

Available accessors:

* `float32`, `float64`, `int32`, `int64`
* `string`, `bool`
* `number` – numeric column coerced to float
* Option-aware variants: `float64_opt`, `int32_opt`, `string_opt`, `bool_opt`
* `index` – row index
* Helpers: `map`, `map2`, `map3`, `both`, `sequence`, `fold_list`

### 7.2 Filtering rows

**pandas**

```python
adults = df[df["age"] > 25]
```

**Talon**

```ocaml
let adults =
  filter_by df
    Row.(
      map (int32 "age") ~f:(fun age -> age > 25l)
    )
```

Or with boolean mask like `df[df["mask"]]`:

<!-- $MDX skip -->
```ocaml
let mask : bool array = [|true; false; true|]
let filtered = filter df mask
```

### 7.3 Adding multiple derived columns

**pandas**

```python
df["sum"]   = df["a"] + df["b"]
df["ratio"] = df["a"] / df["b"]
```

**Talon**

<!-- $MDX skip -->
```ocaml
let df' = df
  |> fun df -> with_column df "sum" Nx.float64
    Row.(map2 (float64 "a") (float64 "b") ~f:( +. ))
  |> fun df -> with_column df "ratio" Nx.float64
    Row.(map2 (float64 "a") (float64 "b") ~f:( /. ))
```

Or add multiple pre-computed columns at once with `with_columns`:

<!-- $MDX skip -->
```ocaml
let df' = with_columns df [
  ("col1", Col.float64 [| 1.0; 2.0 |]);
  ("col2", Col.float64 [| 3.0; 4.0 |]);
]
```

---

## 8. Column-wise Aggregations & Descriptives

### 8.1 Simple aggregations

**pandas**

```python
df["score"].sum()
df["score"].mean()
df["score"].std()
df["score"].min()
df["score"].max()
df["score"].median()
df["score"].quantile(0.25)
```

**Talon**

All numeric aggregations coerce to float, so a single set of functions works for any numeric column:

```ocaml
let sum_score   = Agg.sum  df "score"
let mean_score  = Agg.mean df "score"
let std_score   = Agg.std  df "score"
let min_score   = Agg.min  df "score"
let max_score   = Agg.max  df "score"
let median      = Agg.median   df "score"
let q25         = Agg.quantile df "score" ~q:0.25
```

Integer columns work with the same functions:

```ocaml
let total  = Agg.sum  df "age"   (* returns float *)
let min_a  = Agg.min  df "age"   (* returns float option *)
let mean_a = Agg.mean df "age"   (* returns float *)
```

### 8.2 Strings and booleans

**pandas**

```python
df["name"].min()
df["name"].max()
df["name"].mode()
df["name"].nunique()
(df["flag"]).all()
(df["flag"]).any()
(df["flag"]).mean()  # proportion true
```

**Talon**

<!-- $MDX skip -->
```ocaml
let s_min    = Agg.String.min     df "name"
let s_max    = Agg.String.max     df "name"
let s_mode   = Agg.String.mode    df "name"
let s_unique = Agg.String.unique  df "name"  (* string array *)
let s_nuniq  = Agg.String.nunique df "name"

let b_all    = Agg.Bool.all  df "flag"
let b_any    = Agg.Bool.any  df "flag"
let b_sum    = Agg.Bool.sum  df "flag"
let b_mean   = Agg.Bool.mean df "flag" (* proportion true *)
```

### 8.3 Generic quantities

**pandas**

```python
df["x"].count()
df["x"].nunique()
df["x"].value_counts()
df["x"].isna()
```

**Talon**

<!-- $MDX skip -->
```ocaml
let count     = Agg.count df "x"
let nunique   = Agg.nunique df "x"

let vc = value_counts df "x"
(* vc is a dataframe with "value" and "count" columns *)

let null_col : Col.t = is_null df "x"
(* null_col is a boolean column where true indicates null *)
```

### 8.4 `describe`

**pandas**

```python
df.describe()
```

**Talon**

<!-- $MDX skip -->
```ocaml
let stats_df = describe df
```

* `describe` in Talon returns a `Talon.t` whose rows are `"count"`, `"mean"`, `"std"`, `"min"`, `"25%"`, `"50%"`, `"75%"`, `"max"` and columns are numeric column names.

---

## 9. Row-wise Aggregations (`axis=1` in pandas)

**pandas**

```python
df["row_sum"]   = df[["a", "b", "c"]].sum(axis=1)
df["row_mean"]  = df[["a", "b", "c"]].mean(axis=1)
df["row_max"]   = df[["a", "b", "c"]].max(axis=1)
df["dot"]       = df[["x", "y"]] @ np.array([0.2, 0.8])
df["any_flag"]  = df[["f1", "f2", "f3"]].any(axis=1)
df["all_flag"]  = df[["f1", "f2", "f3"]].all(axis=1)
```

**Talon**

Use `Agg.row_*` (vectorized across columns):

```ocaml
let df_row =
  create
    [
      ("a", Col.int32 [| 1l; 2l; 3l |]);
      ("b", Col.int32 [| 4l; 5l; 6l |]);
      ("c", Col.int32 [| 7l; 8l; 9l |]);
      ("x", Col.float32 [| 1.0; 2.0; 3.0 |]);
      ("y", Col.float32 [| 0.2; 0.8; 1.0 |]);
      ("f1", Col.bool [| true; false; true |]);
      ("f2", Col.bool [| true; true; false |]);
      ("f3", Col.bool [| false; true; true |]);
    ]

let row_sum_col   = Agg.row_sum  df_row ~names:[ "a"; "b"; "c" ]
let row_mean_col  = Agg.row_mean df_row ~names:[ "a"; "b"; "c" ]
let row_max_col   = Agg.row_max  df_row ~names:[ "a"; "b"; "c" ]

let dot_col =
  Agg.dot df_row ~names:[ "x"; "y" ] ~weights:[| 0.2; 0.8 |]

let any_flag_col  = Agg.row_any df_row ~names:[ "f1"; "f2"; "f3" ]
let all_flag_col  = Agg.row_all df_row ~names:[ "f1"; "f2"; "f3" ]

let df' =
  with_columns df_row
    [
      ("row_sum",  row_sum_col);
      ("dot",      dot_col);
      ("any_flag", any_flag_col);
    ]
```

These are direct analogues of `axis=1` aggregations in pandas, implemented with Nx for performance.

---

## 10. Sorting, Sampling, and Slicing

### 10.1 Sorting

**pandas**

```python
df.sort_values("age")
df.sort_values("age", ascending=False)
```

**Talon**

```ocaml
let df_sorted     = sort_values df "age"
let df_descending = sort_values ~ascending:false df "age"
```

Custom key sort (like `df.sort_values(key=...)`):

```ocaml
let people =
  create
    [
      ("first", Col.string [| "Ada"; "Bob"; "Cara"; "Dan" |]);
      ("last", Col.string [| "Zane"; "Young"; "Zane"; "Xue" |]);
    ]

let df_sorted_by_composite =
  sort people
    Row.(
      map2 (string "last") (string "first")
        ~f:(fun l f -> l ^ ", " ^ f)
    )
    ~compare:String.compare
```

### 10.2 Sampling

**pandas**

```python
df.sample(n=10, replace=True, random_state=42)
df.sample(frac=0.1)
```

**Talon**

<!-- $MDX skip -->
```ocaml
let s1 = sample ~n:10   ~replace:true  ~seed:42 df
let s2 = sample ~frac:0.1              df
```

Exactly one of `n` or `frac` must be provided.

### 10.3 Head / tail / slice

**pandas**

```python
df.head(5)
df.tail(5)
df.iloc[10:20]
```

**Talon**

```ocaml
let df_slice =
  create
    [
      ( "age",
        Col.int32 [| 18l; 22l; 25l; 27l; 30l; 31l; 35l; 40l; 42l; 44l; 48l; 50l |]
      );
    ]

let first5  = head df_slice          (* default n=5 *)
let last5   = tail df_slice
let mid     = slice df_slice ~start:2 ~stop:8
```

---

## 11. Grouping

### 11.1 Group by existing column

**pandas**

```python
for key, group in df.groupby("category"):
    ...
```

**Talon**

```ocaml
let grouped =
  create
    [
      ("category", Col.string [| "A"; "A"; "B"; "B"; "C" |]);
      ("score", Col.float64 [| 85.; 92.; 78.; 88.; 95. |]);
    ]

let groups : (string * t) list = group_by grouped (Row.string "category")

let () =
  List.iter
    (fun (key, group_df) ->
      Printf.printf "Group %s: rows=%d\n" key (num_rows group_df)
    )
    groups
```

### 11.2 Group by computed key

**pandas**

```python
df.groupby(df["score"].apply(lambda s: "A" if s >= 90 else "B"))
```

**Talon**

```ocaml
let scored =
  create
    [
      ("score", Col.float64 [| 85.; 92.; 78.; 88.; 95. |]);
    ]

let groups =
  group_by scored
    Row.(
      map (float64 "score") ~f:(fun s ->
        if s >= 90.0 then "A"
        else if s >= 80.0 then "B"
        else "C")
    )
(* groups : (string * t) list *)
```

`group_by` takes a `Row` computation as the key, which covers both column-based and computed grouping.

---

## 12. Joins and Merges

### 12.1 API shape

**pandas**

```python
df1.merge(df2, on="id", how="inner")
df1.merge(df2, left_on="a", right_on="b", how="left")
df1.join(df2.set_index("id"), on="id", how="outer")
```

**Talon**

```ocaml
let df1 =
  create
    [
      ("id", Col.int32 [| 1l; 2l |]);
      ("value", Col.float64 [| 10.0; 20.0 |]);
    ]

let df2 =
  create
    [
      ("id", Col.int32 [| 1l; 2l |]);
      ("value", Col.float64 [| 100.0; 200.0 |]);
    ]

(* Same key name on both sides *)
let joined =
  join df1 df2 ~on:"id" ~how:`Inner ()

(* Different key names *)
let df_left =
  create
    [
      ("a", Col.int32 [| 1l; 2l |]);
      ("val1", Col.float64 [| 10.0; 20.0 |]);
    ]

let df_right =
  create
    [
      ("b", Col.int32 [| 1l; 2l |]);
      ("val2", Col.float64 [| 100.0; 200.0 |]);
    ]

let merged =
  join df_left df_right
    ~on:"a" ~right_on:"b"
    ~how:`Left
    ()
```

Join types: `` `Inner | `Left | `Right | `Outer ``

Column name collisions:

* The join key appears once (for `join` on same name).
* Other duplicate names get suffixes `("_x", "_y")` by default.
* Customize via `~suffixes:("_left", "_right")`.

Null semantics for join keys:

* Null keys never match each other (similar to SQL semantics; different from some pandas corner cases).
* Inner joins drop null-keyed rows entirely.
* Outer joins keep null-keyed rows, but they don't match across sides.

---

## 13. Reshaping: Pivot & Melt

### 13.1 Pivot

**pandas**

```python
pd.pivot_table(
    df,
    index="date",
    columns="product",
    values="amount",
    aggfunc="sum"
)
```

**Talon**

```ocaml
let df_pivot =
  create
    [
      ("date", Col.string [| "2024-01"; "2024-01"; "2024-02"; "2024-02" |]);
      ("product", Col.string [| "A"; "B"; "A"; "B" |]);
      ("amount", Col.float64 [| 100.0; 150.0; 120.0; 180.0 |]);
    ]

let pivoted =
  pivot df_pivot
    ~index:"date"
    ~columns:"product"
    ~values:"amount"
    ~agg_func:`Sum
    ()
```

Supported `agg_func`: `` `Sum | `Mean | `Count | `Min | `Max ``.

### 13.2 Melt

**pandas**

```python
pd.melt(
    df,
    id_vars=["id"],
    value_vars=["A", "B"],
    var_name="variable",
    value_name="value",
)
```

**Talon**

```ocaml
let df_melt =
  create
    [
      ("id", Col.int32 [| 1l; 2l |]);
      ("A", Col.float64 [| 10.0; 20.0 |]);
      ("B", Col.float64 [| 30.0; 40.0 |]);
    ]

let melted =
  melt df_melt
    ~id_vars:["id"]
    ~value_vars:["A"; "B"]
    ~var_name:"variable"
    ~value_name:"value"
    ()
```

If `value_vars` is omitted, Talon uses all columns not in `id_vars`, just like pandas.

---

## 14. Converting to Nx (vs NumPy)

**pandas**

```python
arr = df[["x", "y"]].to_numpy(dtype="float32")
```

**Talon**

<!-- $MDX skip -->
```ocaml
let tensor : (float, Bigarray.float32_elt) Nx.t =
  to_nx df
```

* `to_nx` stacks **numeric** columns only (floats and ints).
* All numeric columns are cast to `float32`.
* Nulls become `NaN`.

For more control, extract specific columns and use `Nx.stack` manually.

---

## 15. When to Reach for Talon vs pandas

**Use Talon when:**

* You're writing OCaml (obviously) and want a dataframe story compatible with Nx and type-safe numeric code.
* You want null semantics that are explicit and consistent across operations.
* You care about compile-time guidance: you'd rather have `Agg.String.min` only accept strings than debug runtime dtype errors.
* You like functional, immutable pipelines and row computations expressed as pure combinators.

**Use pandas when:**

* You're in Python, especially in a notebook-heavy, exploratory environment.
* You need the huge ecosystem around pandas (plotting, scikit-learn, statsmodels, etc.).
* You rely on advanced pandas features Talon doesn't yet model (MultiIndex, time-series index semantics, categorical dtypes, etc.).

---

## 16. Quick Cheat Sheet

| Task                      | pandas                                | Talon                                                                          |
| ------------------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
| Create DF from columns    | `pd.DataFrame({...})`                 | `create [ ("col", Col.float64 [| ... |]); ... ]`                              |
| Read CSV                  | `pd.read_csv("file.csv")`             | `Talon_csv.read "file.csv"`                                                    |
| Filter rows               | `df[df["age"] > 25]`                  | `filter_by df Row.(map (int32 "age") ~f:(fun a -> a > 25l))`                   |
| Select columns            | `df[["a", "b"]]`                      | `select df ["a"; "b"]`                                                         |
| Drop null rows            | `df.dropna()`                         | `drop_nulls df`                                                                |
| Fill nulls                | `df["x"].fillna(0)`                   | `fill_null df "x" ~with_value:(\`Float 0.0)`                                   |
| Column sum                | `df["x"].sum()`                       | `Agg.sum df "x"`                                                               |
| Value counts              | `df["x"].value_counts()`              | `value_counts df "x"`                                                           |
| Group by column           | `df.groupby("key")`                   | `group_by df (Row.string "key")`                                               |
| Join on column            | `df1.merge(df2, on="id", how="left")` | `join df1 df2 ~on:"id" ~how:\`Left ()`                                         |
| Pivot                     | `pd.pivot_table(df, index=..., ...)`  | `pivot df ~index ~columns ~values ~agg_func ()`                                |
| Melt                      | `pd.melt(df, ...)`                    | `melt df ~id_vars ~value_vars ()`                                              |
| Describe numeric columns  | `df.describe()`                       | `describe df`                                                                  |
| Head / tail               | `df.head(5)`, `df.tail(5)`            | `head ~n:5 df`, `tail ~n:5 df`                                                 |
| Row sum (axis=1)          | `df[cols].sum(axis=1)`                | `let s = Agg.row_sum df ~names:cols in add_column df "row_sum" s`              |
| Convert to numeric matrix | `df[cols].to_numpy(dtype="float32")`  | `to_nx df`                                                                     |
