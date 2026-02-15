# Talon vs. pandas – A Practical Comparison

This guide explains how Talon’s dataframe model relates to Python’s [pandas](https://pandas.pydata.org/), focusing on:

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
| Column type     | `pd.Series`                                               | `Talon.Col.t` (GADT)                                                |
| Numeric backend | NumPy                                                     | Nx                                                                  |
| Typing model    | Dtypes checked at runtime                                 | Dtypes tracked at type & value-level via GADTs                      |
| Null semantics  | NaN, `pd.NA`, object `None`, nullable dtypes              | Explicit null masks for numerics, `option` values for strings/bools |
| Row-wise logic  | `DataFrame.apply`, vectorized ops                         | `Row` applicative combinators, compiled to loops                    |
| Groupby / joins | `groupby`, `merge`, `join`                                | `group_by`, `group_by_column`, `join`, `merge`                      |
| Reshaping       | `pivot`, `melt`, `stack`, `unstack`                       | `pivot`, `melt`                                                     |
| I/O             | `read_csv`, `to_csv`, `read_json`, `to_json`, etc.        | `Talon_csv.read/write`, `Talon_json.from/to_*`                      |
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
      ("name", Col.string_list [ "Alice"; "Bob" ]);
      ("age",  Col.int32_list   [ 25l; 30l ]);
    ]
```

Key parallels:

* Both are *row-oriented* logical tables with named, homogeneous columns.
* `Talon.t` is immutable; every transformation returns a new dataframe.

### 2.2 Column Representation

**pandas** columns are `Series`, dynamically typed: dtype is metadata, but Python won’t stop you from doing `df["name"] + 1` until runtime.

**Talon** columns are:

<!-- $MDX skip -->
```ocaml
module Col : sig
  type t =
    | P : ('a, 'b) Nx.dtype * ('a, 'b) Nx.t * bool array option -> t
    | S : string option array
    | B : bool option array
end
```

* `P` (“packed”) = any Nx numeric dtype + optional null mask.
* `S` = string column (`string option array`).
* `B` = bool column (`bool option array`).

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
* Many operations are organized into *type-specific* modules (e.g. `Agg.Float`, `Agg.Int`, `Agg.String`), which prevents applying inappropriate aggregations at runtime.

Where pandas often says “this is probably fine, let’s try”, Talon tends to say “pick the right function for this dtype”.

---

## 4. Null / Missing Data Semantics

This is one of the biggest conceptual differences.

### 4.1 Representation

**pandas**

* Historically: NaN for numeric, `None`/`np.nan` for object; increasingly `pd.NA` and nullable dtypes.
* Null semantics vary slightly by dtype (especially between legacy and new nullable dtypes).

**Talon**

* **Numeric** (`Col.P`): explicit optional boolean mask; payload values (including `nan`, `Int32.min_int`, etc.) are treated as *regular data* unless masked.
* **String / Bool** (`Col.S` / `Col.B`): `None` = null, `Some v` = non-null.

So the “source of truth” for missingness is:

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

(* Type-specific filling *)
let filled_f32  = Col.fill_nulls_float32 col ~value:0.0
let filled_i32  = Col.fill_nulls_int32  col ~value:0l
let filled_str  = Col.fill_nulls_string col ~value:"(missing)"
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

let has_nulls = has_nulls df "x"
let n_nulls   = null_count df "x"

let df' = fill_missing df "score" ~with_value:(`Float 0.0)
```

`drop_nulls` and `fill_missing` are the closest conceptual equivalents to `dropna` and `fillna`.

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
      ("name",  Col.string_list [ "Alice"; "Bob" ]);
      ("age",   Col.int32_list   [ 25l; 30l ]);
      ("score", Col.float64_list [ 85.5; 92.0 ]);
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
let df = from_nx ~names:[ "x"; "y"; "z" ] t
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

### 5.3 JSON I/O

**pandas**

```python
df.to_json(orient="records")
pd.read_json(..., orient="records")
```

**Talon**

<!-- $MDX skip -->
```ocaml
let json = Talon_json.to_string ~orient:`Records df
let df'  = Talon_json.from_string ~orient:`Records json
```

File-based equivalents exist (`Talon_json.to_file`, `Talon_json.from_file`).

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
module Cols = Talon.Cols

let numeric_cols = Cols.numeric df      (* floats + ints *)
let float_cols   = Cols.float df        (* float32/64 *)
let int_cols     = Cols.int df
let bool_cols    = Cols.bool df
let string_cols  = Cols.string df

let only_numeric_and_bool =
  Cols.select_dtypes df [ `Numeric; `Bool ]
```

Name-based selection:

<!-- $MDX skip -->
```ocaml
let re = Re.(compile (re "score_.*"))
let score_cols = Cols.matching df re
let prefixed   = Cols.with_prefix df "temp_"
let suffixed   = Cols.with_suffix df "_score"
let others     = Cols.except df [ "id"; "target" ]
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
let df_pruned   = drop_columns df [ "col1"; "col2" ]
```

Select subsets:

```ocaml
let df_small  = select df [ "name"; "age" ]        (* error if missing *)
let df_loose  = select_loose df [ "name"; "maybe" ](* ignores missing *)
let df_reordered = reorder_columns df [ "id"; "target" ]
```

Extract as arrays, like `df["x"].to_numpy()`:

```ocaml
let xs : float array option    = to_float64_array df "x"
let ys : int32 array option    = to_int32_array  df "y"
let zs : string array option   = to_string_array df "z"

let xs_opt : float option array option = to_float64_options df "x"
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
* Helpers: `map`, `map2`, `map3`, `both`, `sequence`, `map_list`, `fold_list`

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

### 7.3 Adding multiple derived columns in one pass

**pandas**

```python
df["sum"]   = df["a"] + df["b"]
df["ratio"] = df["a"] / df["b"]
```

**Talon**

<!-- $MDX skip -->
```ocaml
let df' =
  with_columns_map df
    [
      ( "sum",
        Nx.float64,
        Row.map2 (Row.float64 "a") (Row.float64 "b") ~f:( +. ) );
      ( "ratio",
        Nx.float64,
        Row.map2 (Row.float64 "a") (Row.float64 "b") ~f:( /. ) );
    ]
```

`with_columns_map` is the Talon equivalent of “compute several derived columns at once”, minimizing passes over the data.

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

Use type-specific modules:

```ocaml
let sum_score   = Agg.Float.sum  df "score"
let mean_score  = Agg.Float.mean df "score"
let std_score   = Agg.Float.std  df "score"
let min_score   = Agg.Float.min  df "score"
let max_score   = Agg.Float.max  df "score"
let median      = Agg.Float.median   df "score"
let q25         = Agg.Float.quantile df "score" ~q:0.25
```

For integer semantics (returning `int64`):

```ocaml
(* Integer aggregations require integer dtypes. Create a dedicated numeric sample df. *)
let int_df = create [ ("count", Col.int32_list [ 1l; 2l; 3l; 4l ]) ]

let total = Agg.Int.sum  int_df "count"
let min_c = Agg.Int.min  int_df "count"
let max_c = Agg.Int.max  int_df "count"
let mean_c = Agg.Int.mean int_df "count"
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

```ocaml
let s_min    = Agg.String.min     df "name"
let s_max    = Agg.String.max     df "name"
let s_mode   = Agg.String.mode    df "name"
let s_unique = Agg.String.unique  df "name"  (* string array *)
let s_nuniq  = Agg.String.nunique df "name"

(* Boolean aggregations also require boolean input. Create one for this section. *)
let bool_df = create [ ("flag", Col.bool_list [ true; false; true; true ]) ]

let b_all    = Agg.Bool.all  bool_df "flag"
let b_any    = Agg.Bool.any  bool_df "flag"
let b_sum    = Agg.Bool.sum  bool_df "flag"
let b_mean   = Agg.Bool.mean bool_df "flag" (* proportion true *)
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

```ocaml
let count     = Agg.count df "x"
let nunique   = Agg.nunique df "x"

let (values_col, counts) = Agg.value_counts df "x"
(* values_col is a Col.t, counts : int array *)

let null_mask : bool array = Agg.is_null df "x"
```

### 8.4 `describe`

**pandas**

```python
df.describe()
```

**Talon**

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

Use `Row.Agg` (vectorized across columns):

```ocaml
module RA = Row.Agg

let df_row =
  create
    [
      ("a", Col.int32_list [ 1l; 2l; 3l ]);
      ("b", Col.int32_list [ 4l; 5l; 6l ]);
      ("c", Col.int32_list [ 7l; 8l; 9l ]);
      ("x", Col.float32_list [ 1.0; 2.0; 3.0 ]);
      ("y", Col.float32_list [ 0.2; 0.8; 1.0 ]);
      ("f1", Col.bool_list [ true; false; true ]);
      ("f2", Col.bool_list [ true; true; false ]);
      ("f3", Col.bool_list [ false; true; true ]);
    ]

let row_sum_col   = RA.sum  df_row ~names:[ "a"; "b"; "c" ]
let row_mean_col  = RA.mean df_row ~names:[ "a"; "b"; "c" ]
let row_max_col   = RA.max  df_row ~names:[ "a"; "b"; "c" ]

let dot_col =
  RA.dot df_row ~names:[ "x"; "y" ] ~weights:[| 0.2; 0.8 |]

let any_flag_col  = RA.any df_row ~names:[ "f1"; "f2"; "f3" ]
let all_flag_col  = RA.all df_row ~names:[ "f1"; "f2"; "f3" ]

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
      ("first", Col.string_list [ "Ada"; "Bob"; "Cara"; "Dan" ]);
      ("last", Col.string_list [ "Zane"; "Young"; "Zane"; "Xue" ]);
    ]

let df_sorted_by_composite =
  let df =
    people
  in
  sort df
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
        Col.int32_list [ 18l; 22l; 25l; 27l; 30l; 31l; 35l; 40l; 42l; 44l; 48l; 50l ]
      );
    ]

let first5  = head df_slice          (* default n=5 *)
let last5   = tail df_slice
let mid     = slice df_slice ~start:10 ~stop:20
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
      ("category", Col.string_list [ "A"; "A"; "B"; "B"; "C" ]);
      ("score", Col.float64_list [ 85.; 92.; 78.; 88.; 95. ]);
    ]

let groups : (Col.t * t) list = group_by_column grouped "category"

(* key column (with single value) + group df *)
let () =
  List.iter
    (fun (key_col, group_df) ->
      Printf.printf "Group: null_count=%d, rows=%d\n"
        (Col.null_count key_col) (num_rows group_df)
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
      ("score", Col.float64_list [ 85.; 92.; 78.; 88.; 95. ]);
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

`group_by` takes a `Row` computation as the key; `group_by_column` is the shortcut when you already have a column.

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
      ("id", Col.int32_list [ 1l; 2l ]);
      ("value", Col.float64_list [ 10.0; 20.0 ]);
    ]

let df2 =
  create
    [
      ("id", Col.int32_list [ 1l; 2l ]);
      ("value", Col.float64_list [ 100.0; 200.0 ]);
    ]

(* Same key name on both sides *)
let joined =
  join df1 df2 ~on:"id" ~how:`Inner ()

(* Different key names *)
let df_left =
  create
    [
      ("a", Col.int32_list [ 1l; 2l ]);
      ("val1", Col.float64_list [ 10.0; 20.0 ]);
    ]

let df_right =
  create
    [
      ("b", Col.int32_list [ 1l; 2l ]);
      ("val2", Col.float64_list [ 100.0; 200.0 ]);
    ]

let merged =
  merge df_left df_right
    ~left_on:"a" ~right_on:"b"
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
* Outer joins keep null-keyed rows, but they don’t match across sides.

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
      ("date", Col.string_list [ "2024-01"; "2024-01"; "2024-02"; "2024-02" ]);
      ("product", Col.string_list [ "A"; "B"; "A"; "B" ]);
      ("amount", Col.float64_list [ 100.0; 150.0; 120.0; 180.0 ]);
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
      ("id", Col.int32_list [ 1l; 2l ]);
      ("A", Col.float64_list [ 10.0; 20.0 ]);
      ("B", Col.float64_list [ 30.0; 40.0 ]);
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

* You’re writing OCaml (obviously) and want a dataframe story compatible with Nx and type-safe numeric code.
* You want null semantics that are explicit and consistent across operations.
* You care about compile-time guidance: you’d rather have `Agg.String.min` only accept strings than debug runtime dtype errors.
* You like functional, immutable pipelines and row computations expressed as pure combinators.

**Use pandas when:**

* You’re in Python, especially in a notebook-heavy, exploratory environment.
* You need the huge ecosystem around pandas (plotting, scikit-learn, statsmodels, etc.).
* You rely on advanced pandas features Talon doesn’t yet model (MultiIndex, time-series index semantics, categorical dtypes, etc.).

---

## 16. Quick Cheat Sheet

| Task                      | pandas                                | Talon                                                                         |
| ------------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| Create DF from columns    | `pd.DataFrame({...})`                 | `create [ ("col", Col.float64_list [...]); ... ]`                             |
| Read CSV                  | `pd.read_csv("file.csv")`             | `Talon_csv.read "file.csv"`                                                   |
| Filter rows               | `df[df["age"] > 25]`                  | `filter_by df Row.(map (int32 "age") ~f:(fun a -> a > 25l))`                  |
| Select columns            | `df[["a", "b"]]`                      | `select df ["a"; "b"]`                                                        |
| Drop null rows            | `df.dropna()`                         | `drop_nulls df`                                                               |
| Fill nulls                | `df["x"].fillna(0)`                   | `fill_missing df "x" ~with_value:(\`Float 0.0)`                               |
| Column sum                | `df["x"].sum()`                       | `Agg.Float.sum df "x"`                                                        |
| Value counts              | `df["x"].value_counts()`              | `Agg.value_counts df "x"`                                                     |
| Group by column           | `df.groupby("key")`                   | `group_by_column df "key"`                                                    |
| Join on column            | `df1.merge(df2, on="id", how="left")` | `join df1 df2 ~on:"id" ~how:\`Left ()`                                        |
| Pivot                     | `pd.pivot_table(df, index=..., ...)`  | `pivot df ~index ~columns ~values ~agg_func ()`                               |
| Melt                      | `pd.melt(df, ...)`                    | `melt df ~id_vars ~value_vars ()`                                             |
| Describe numeric columns  | `df.describe()`                       | `describe df`                                                                 |
| Head / tail               | `df.head(5)`, `df.tail(5)`            | `head ~n:5 df`, `tail ~n:5 df`                                                |
| Row sum (axis=1)          | `df[cols].sum(axis=1)`                | `let row_sum = Row.Agg.sum df ~names:cols in add_column df "row_sum" row_sum` |
| Convert to numeric matrix | `df[cols].to_numpy(dtype="float32")`  | `to_nx df`                                                                    |
