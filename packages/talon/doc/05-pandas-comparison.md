# Talon vs. pandas

This page translates common pandas code into talon. The examples share one
small table of flights:

```ocaml
open Talon

let flights =
  Query.of_table
    (Talon.v
       [
         ("carrier", Column.v Type.string [| "AA"; "AA"; "DL"; "UA"; "DL" |]);
         ("origin", Column.v Type.string [| "JFK"; "LGA"; "JFK"; "EWR"; "LGA" |]);
         ("dep_delay", Column.of_options Type.int64 [| Some 4; Some 130; None; Some (-2); Some 17 |]);
         ("distance", Column.v Type.int64 [| 1089; 733; 2475; 2565; 762 |]);
       ])

let carriers =
  Query.of_table
    (Talon.v
       [
         ("carrier", Column.v Type.string [| "AA"; "DL"; "UA" |]);
         ("name", Column.v Type.string [| "American"; "Delta"; "United" |]);
       ])

let delay = Col.int "dep_delay"
let distance = Col.int "distance"
```

## Key Differences

| | pandas | Talon |
|---|---|---|
| Table | `pd.DataFrame`, mutable | `Talon.t`, immutable |
| Computation | eager, one call at a time | a `Query.t` plan, computed by `Query.run` |
| Column access | `df["dep_delay"]`, checked when it runs | `Col.int "dep_delay"`, a typed handle checked when the verb is applied |
| Missing values | `NaN`, `None`, `pd.NA` or `NaT`, by dtype | null, one state for every type, distinct from NaN |
| Row labels | an index on every frame | none: rows are ordered, and keys are columns |
| Wrong column name | `KeyError` at the line that uses it | every problem of a verb in one report, before any data is read |
| Bad data in a file | often a silent `NaN` or an `object` column | an `Error` with the file, line and column |

## Creating and Reading

```python
df = pd.DataFrame({"carrier": ["AA", "DL"], "dep_delay": [4, None]})
df = pd.read_csv("flights.csv", na_values=["NA"])
df = pd.read_parquet("carriers.parquet")
```

```ocaml
let df =
  Talon.v
    [
      ("carrier", Column.v Type.string [| "AA"; "DL" |]);
      ("dep_delay", Column.of_options Type.int64 [| Some 4; None |]);
    ]

let files () =
  let ( let* ) = Result.bind in
  let* flights = Talon_csv.file ~nulls:[ "NA" ] "flights.csv" in
  let* carriers = Talon_parquet.file "carriers.parquet" in
  Ok (Query.of_source flights, Query.of_source carriers)
```

A file is a source: nothing is read until a query that uses it runs, and
then only the columns the query needs.

## Selecting and Filtering

| pandas | Talon |
|---|---|
| `df[["carrier", "dep_delay"]]` | `Query.select Expr.[ keep Sel.(names [ "carrier"; "dep_delay" ]) ]` |
| `df.drop(columns=["origin"])` | `Kit.drop Sel.(names [ "origin" ])` |
| `df.rename(columns={"origin": "from"})` | `Kit.rename [ ("origin", "from") ]` |
| `df[df.dep_delay > 15]` | `Query.filter Expr.(delay > int 15)` |
| `df[df.origin.isin(["JFK", "LGA"])]` | `Query.filter Expr.(is_in [ "JFK"; "LGA" ] (Col.string "origin"))` |
| `df.dropna(subset=["dep_delay"])` | `Query.filter Expr.(not (is_null delay))` |
| `df.head(10)`, `df.tail(10)` | `Kit.head 10`, `Kit.tail 10` |
| `df.iloc[20:30]` | `Query.slice ~offset:20 ~length:10` |

```ocaml
let late_from_jfk =
  Query.(
    flights
    |> filter Expr.(delay > int 15 && Col.string "origin" = string "JFK")
    |> select Expr.[ keep Sel.(names [ "carrier"; "dep_delay" ]) ])
```

A comparison with a null is null, and `filter` keeps only the rows where its
predicate is `true`, so a missing delay is dropped by `delay > int 15` and by
`not (delay > int 15)` alike.

## Derived Columns

```python
df["hours"] = df.dep_delay / 60
df["long"] = df.distance > 1500
df["dep_delay"] = df.dep_delay.fillna(0)
```

```ocaml
let derived =
  Query.(
    flights
    |> derive
         Expr.[
           "hours" := cast Type.float64 delay /. float 60.;
           "long" := distance > int 1500;
           "dep_delay" := coalesce [ delay; int 0 ];
         ])
```

`derive` adds columns, and an output named like an input column replaces it
in place. Outputs read the input's columns, never each other, so `hours` is
null where the input's delay is, although `dep_delay` is filled in the
result. Integer division stays integer, so the delay is cast to a float
first.

## Group By

```python
df.groupby("carrier").agg(mean_delay=("dep_delay", "mean"), flights=("dep_delay", "size"))
df.groupby("carrier", sort=False).dep_delay.transform("mean")
df.dep_delay.value_counts()
```

```ocaml
let by_carrier =
  Query.(
    flights
    |> aggregate ~by:[ "carrier" ] Expr.[ "mean_delay" := mean delay; "flights" := rows ])

let carrier_mean =
  Query.(flights |> derive Expr.[ "carrier_mean" := over ~by:[ "carrier" ] (mean delay) ])

let counts = Kit.value_counts "dep_delay" flights
```

- Groups come in order of first appearance. A null key forms a group of its
  own, as does NaN.
- `over ~by` is `transform`: it evaluates a reduction per group and puts the
  result back on each of the group's rows.
- `mean` returns `float64`; `sum` of integers returns `int64`. Reductions skip
  nulls, as pandas skips NaN.
- `Kit.value_counts` counts null as a value of its own.

## Sorting and Ranking

| pandas | Talon |
|---|---|
| `df.sort_values("dep_delay", ascending=False)` | `Query.sort [ Order.desc "dep_delay" ]` |
| `df.sort_values(["carrier", "dep_delay"])` | `Query.sort [ Order.asc "carrier"; Order.asc "dep_delay" ]` |
| `df.nlargest(3, "dep_delay")` | `Kit.top_k 3 [ Order.desc "dep_delay" ]` |
| `df.dep_delay.rank(method="min")` | `Expr.(over (rank delay))` |
| `df.groupby("carrier").dep_delay.shift(1)` | `Expr.(over ~by:[ "carrier" ] (shift 1 delay))` |

Sorting is stable and puts nulls last in both directions, unless
`Order.nulls_first`.

## Joins and Concatenation

| pandas | Talon |
|---|---|
| `df.merge(carriers, on="carrier")` | `Query.join ~on:(Join.keys [ "carrier" ]) carriers` |
| `df.merge(c, left_on="carrier", right_on="code", how="left")` | `Query.join ~kind:Left ~on:(Join.eq "carrier" "code") c` |
| `df.merge(carriers, on="carrier", validate="many_to_one")` | `Query.join ~each_right:At_most_one ~on:(Join.keys [ "carrier" ]) carriers` |
| `df[df.carrier.isin(carriers.carrier)]` | `Query.join ~kind:Semi ~on:(Join.keys [ "carrier" ]) carriers` |
| `pd.concat([df1, df2])` | `Query.append df2 df1` |
| `pd.concat([df1, df2])` with different columns | `Kit.union df2 df1` |

```ocaml
let named =
  Query.(flights |> join ~on:(Join.keys [ "carrier" ]) ~each_left:One carriers)
```

`~each_left:One` checks that every flight finds exactly one carrier, and the
run fails naming the key if one does not. A column name on both sides is a
problem: talon adds no suffixes, so rename one side first.

## Summaries

| pandas | Talon |
|---|---|
| `df.describe()` | `Kit.describe` |
| `df.isna().sum()` | `Kit.null_count` |
| `df.drop_duplicates()` | `Kit.distinct` |
| `df.dep_delay.nunique()` | `Query.aggregate ~by:[] Expr.[ "n" := n_unique delay ]` |
| `pd.get_dummies(df.carrier)` | `Kit.one_hot "carrier"`, after `Kit.categorize [ "carrier" ]` |

## Text and Dates

| pandas | Talon |
|---|---|
| `s.str.len()` | `Expr.Str.length s` |
| `s.str.startswith("sub")` | `Expr.Str.(matches (prefix "sub") s)` |
| `s.str.split("_")` | `Expr.Str.split "_" s` |
| `s.str.replace("a", "b", regex=False)` | `Expr.Str.replace "a" ~by:"b" s` |
| `pd.to_datetime(s, format="%d/%m/%Y")` | `Expr.Temporal.parse "%d/%m/%Y" Type.date s` |
| `t.dt.year` | ``Expr.Temporal.field `Year t`` |
| `t.dt.floor("D")` | `Expr.Temporal.floor (Time.Days 1) t` |
| `t + pd.Timedelta(minutes=90)` | `Expr.(Temporal.add t (span (Time.Span.minutes 90)))` |

## To Arrays

| pandas | Talon |
|---|---|
| `df[["a", "b"]].to_numpy(dtype="float32")` | `Talon.to_tensor Nx.float32 [ "a"; "b" ] t` |
| `df.a.to_numpy()` | `Column.to_tensor Nx.int64 (Talon.column t "a")` |
| `df.sample(frac=1, random_state=0)` | `Talon.take (Nx.Rng.permutation (Nx.Rng.key 0) (Talon.rows t)) t` |
| `df.a.tolist()` | `Column.options Kind.int (Talon.column t "a")` |

`Column.to_tensor` shares the column's buffer without a copy, and refuses a
column with nulls. `Talon.to_tensor` makes the one copy that turning columns
into a matrix requires.

## Not in Talon

Talon has no row index and no in-place mutation. It has no `pivot` or
`melt`, no rolling or expanding windows, no as-of joins, and reads no JSON or
Arrow files.
