# Talon vs Pandas Comparison

This document compares the Talon dataframe library (OCaml) with Pandas (Python), highlighting similarities, differences, and providing equivalent code examples.

- [Talon vs Pandas Comparison](#talon-vs-pandas-comparison)
  1. [Overview](#overview)
  2. [Creating DataFrames](#creating-dataframes)
  3. [Basic Operations](#basic-operations)
  4. [Filtering and Sorting](#filtering-and-sorting)
  5. [Column Operations](#column-operations)
  6. [Row-Wise Operations](#row-wise-operations)
  7. [Data Manipulation](#data-manipulation)
  8. [Feature Comparison Matrix](#feature-comparison-matrix)

## 1. Overview

Talon is a dataframe library for OCaml that brings the power of Pandas and Polars to OCaml with compile-time type safety. Built on top of Nx arrays, it provides efficient operations for wide dataframes while maintaining OCaml's strong typing guarantees.

Key characteristics of Talon vs Pandas:

- **Type-Safe Heterogeneous Columns**: Talon enforces type safety at compile time, preventing runtime type errors. Pandas operates dynamically with type checking at runtime.
- **Ecosystem Integration**: Talon is designed to work seamlessly with Nx (arrays) and Kaun (deep learning), just as Pandas integrates with NumPy and scikit-learn.
- **Wide DataFrame Optimization**: Talon is optimized for dataframes with 100+ columns, while Pandas works well across all sizes.
- **Functional Row Operations**: Talon uses applicative functors for elegant row transformations, while Pandas uses imperative NumPy-style operations.
- **API Similarity**: Both provide pandas-compatible operations like join, merge, pivot, melt, and groupby.

## 2. Creating DataFrames

### Creating a DataFrame from Lists

**Talon (OCaml)**:
```ocaml
open Talon

let df = create [
  ("name", Col.string_list ["Alice"; "Bob"; "Charlie"; "Dana"]);
  ("age", Col.int32_list [25l; 30l; 35l; 28l]);
  ("height", Col.float64_list [1.70; 1.80; 1.75; 1.65]);
  ("active", Col.bool_list [true; false; true; true])
]
```

**Pandas (Python)**:
```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Dana"],
    "age": [25, 30, 35, 28],
    "height": [1.70, 1.80, 1.75, 1.65],
    "active": [True, False, True, True]
})
```

### Checking DataFrame Shape

**Talon (OCaml)**:
```ocaml
let rows = num_rows df  (* 4 *)
let cols = num_columns df  (* 4 *)
let () = Printf.printf "Rows: %d, Columns: %d\n" rows cols
```

**Pandas (Python)**:
```python
rows, cols = df.shape
print(f"Rows: {rows}, Columns: {cols}")
# (4, 4)
```

### Printing a DataFrame

**Talon (OCaml)**:
```ocaml
let () = print df
```

**Pandas (Python)**:
```python
print(df)
```

## 3. Basic Operations

### Accessing Columns

**Talon (OCaml)**:
```ocaml
(* Get column as array *)
match to_float64_array df "height" with
| Some arr -> Printf.printf "Height array retrieved\n"
| None -> Printf.printf "Column not found or wrong type\n"
```

**Pandas (Python)**:
```python
height = df["height"].values
# array([1.7 , 1.8 , 1.75, 1.65])
```

### Adding a Simple Column

**Talon (OCaml)**:
```ocaml
let df = add_column df "pass" (Col.bool_list [true; false; true; true])
```

**Pandas (Python)**:
```python
df["pass"] = [True, False, True, True]
```

### Adding Computed Columns

**Talon (OCaml)**:
```ocaml
open Talon

(* Calculate BMI from weight and height *)
let df = with_column df "bmi" Nx.float64
  Row.(map2 (number "weight") (number "height")
    ~f:(fun w h -> w /. (h ** 2.)))
```

**Pandas (Python)**:
```python
df["bmi"] = df["weight"] / (df["height"] ** 2)
```

### Renaming Columns

**Talon (OCaml)**:
```ocaml
(* Talon uses `add_column` with computed values to effectively rename *)
let df_renamed = add_column df "new_name" 
  (get_column_exn df "old_name")
```

**Pandas (Python)**:
```python
df = df.rename(columns={"old_name": "new_name"})
```

## 4. Filtering and Sorting

### Filtering Rows

**Talon (OCaml)**:
```ocaml
(* Filter students with height > 1.70 *)
let tall_students = filter_by df
  Row.(map (number "height") ~f:(fun h -> h > 1.70))
```

**Pandas (Python)**:
```python
tall_students = df[df["height"] > 1.70]
```

### Filtering with Multiple Conditions

**Talon (OCaml)**:
```ocaml
(* Filter active students with age > 28 *)
let filtered = filter_by df
  Row.(map2 (number "age") (bool "active")
    ~f:(fun age active -> age > 28 && active))
```

**Pandas (Python)**:
```python
filtered = df[(df["age"] > 28) & (df["active"] == True)]
```

### Sorting Values

**Talon (OCaml)**:
```ocaml
(* Sort by age in ascending order *)
let sorted = sort_values ~ascending:true df "age"

(* Sort by age in descending order *)
let sorted = sort_values ~ascending:false df "age"
```

**Pandas (Python)**:
```python
sorted = df.sort_values(by="age", ascending=True)
sorted = df.sort_values(by="age", ascending=False)
```

## 5. Column Operations

### Selecting Column Types

**Talon (OCaml)**:
```ocaml
(* Select all numeric columns *)
let numeric_cols = Cols.numeric df

(* Select columns by prefix *)
let price_cols = Cols.with_prefix df "price_"

(* Select columns by suffix *)
let total_cols = Cols.with_suffix df "_total"

(* Select all except specific columns *)
let feature_cols = Cols.except df ["id"; "name"; "timestamp"]
```

**Pandas (Python)**:
```python
# Select numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns

# Select columns by prefix
price_cols = [col for col in df.columns if col.startswith("price_")]

# Select columns by suffix
total_cols = [col for col in df.columns if col.endswith("_total")]

# Select all except specific columns
feature_cols = df.drop(columns=["id", "name", "timestamp"]).columns
```

### Dropping Columns

**Talon (OCaml)**:
```ocaml
let feature_cols = Cols.except df ["id"; "name"; "timestamp"]
let df = select_columns df feature_cols
```

**Pandas (Python)**:
```python
df = df.drop(columns=["id", "name", "timestamp"])
```

## 6. Row-Wise Operations

### Summing Across Multiple Columns

**Talon (OCaml)**:
```ocaml
let df_scores = create [
  ("student", Col.string_list ["Alice"; "Bob"]);
  ("math", Col.float64_list [92.0; 85.0]);
  ("science", Col.float64_list [88.0; 92.0]);
  ("history", Col.float64_list [95.0; 78.0]);
  ("english", Col.float64_list [90.0; 88.0])
]

(* Calculate total score *)
let total_scores = Row.Agg.sum df_scores
  ~names:["math"; "science"; "history"; "english"]

let df_scores = add_column df_scores "total" total_scores
```

**Pandas (Python)**:
```python
df_scores = pd.DataFrame({
    "student": ["Alice", "Bob"],
    "math": [92.0, 85.0],
    "science": [88.0, 92.0],
    "history": [95.0, 78.0],
    "english": [90.0, 88.0]
})

df_scores["total"] = df_scores[["math", "science", "history", "english"]].sum(axis=1)
```

### Computing Mean Across Columns

**Talon (OCaml)**:
```ocaml
(* Calculate average score *)
let avg_scores = Row.Agg.mean df_scores
  ~names:["math"; "science"; "history"; "english"]

let df_scores = add_column df_scores "average" avg_scores
```

**Pandas (Python)**:
```python
df_scores["average"] = df_scores[["math", "science", "history", "english"]].mean(axis=1)
```

### Multiple Transformations

**Talon (OCaml)**:
```ocaml
(* Map over multiple columns at once *)
let df = with_columns_map df
  Row.([
    ("sum", Nx.float64,
      map3 (number "a") (number "b") (number "c") 
        ~f:(fun a b c -> a +. b +. c));
    ("product", Nx.float64,
      map3 (number "a") (number "b") (number "c") 
        ~f:(fun a b c -> a *. b *. c));
    ("mean", Nx.float64,
      map3 (number "a") (number "b") (number "c") 
        ~f:(fun a b c -> (a +. b +. c) /. 3.0))
  ])
```

**Pandas (Python)**:
```python
df["sum"] = df["a"] + df["b"] + df["c"]
df["product"] = df["a"] * df["b"] * df["c"]
df["mean"] = (df["a"] + df["b"] + df["c"]) / 3.0
```

## 7. Data Manipulation

### Inner Join

**Talon (OCaml)**:
```ocaml
let df1 = create [
  ("id", Col.int32_list [1l; 2l; 3l]);
  ("name", Col.string_list ["Alice"; "Bob"; "Charlie"])
]

let df2 = create [
  ("id", Col.int32_list [2l; 3l; 4l]);
  ("score", Col.float64_list [85.0; 92.0; 88.0])
]

let joined = join df1 df2 ~on:"id" ~how:`Inner ()
```

**Pandas (Python)**:
```python
df1 = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

df2 = pd.DataFrame({
    "id": [2, 3, 4],
    "score": [85.0, 92.0, 88.0]
})

joined = df1.merge(df2, on="id", how="inner")
```

### Left Join

**Talon (OCaml)**:
```ocaml
let left_joined = join df1 df2 ~on:"id" ~how:`Left ()
```

**Pandas (Python)**:
```python
left_joined = df1.merge(df2, on="id", how="left")
```

### Pivot Table

**Talon (OCaml)**:
```ocaml
let sales = create [
  ("date", Col.string_list ["2024-01"; "2024-01"; "2024-02"; "2024-02"]);
  ("product", Col.string_list ["A"; "B"; "A"; "B"]);
  ("amount", Col.float64_list [100.0; 150.0; 120.0; 180.0])
]

let pivoted = pivot sales ~index:"date" ~columns:"product" ~values:"amount" ()
```

**Pandas (Python)**:
```python
sales = pd.DataFrame({
    "date": ["2024-01", "2024-01", "2024-02", "2024-02"],
    "product": ["A", "B", "A", "B"],
    "amount": [100.0, 150.0, 120.0, 180.0]
})

pivoted = sales.pivot(index="date", columns="product", values="amount")
```
