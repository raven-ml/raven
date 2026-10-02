# Expressions and Frames

An expression computes values from the columns of a **frame**: the ordered
rows that a verb, or an enclosing expression, gives it. This page covers how
expressions are typed, how they treat nulls, the verbs that take them, and
the frames they run in.

The examples use a table of training losses, with one evaluation missing:

```ocaml
open Talon

let losses =
  Talon.v
    [
      ("run", Column.v Type.string [| "a"; "a"; "a"; "a"; "b"; "b"; "b"; "b" |]);
      ("step", Column.v Type.int32 [| 100; 200; 300; 400; 100; 200; 300; 400 |]);
      ( "loss",
        Column.of_options Type.float64
          [| Some 2.31; Some 1.87; Some 1.98; Some 1.55;
             Some 2.40; None; Some 1.71; Some 1.52 |] );
    ]

let step = Col.int "step"
let loss = Col.float "loss"
```

## Handles

A handle names a column and the kind its values read as: `Col.bool`,
`Col.int`, `Col.float`, `Col.string`, `Col.binary`, `Col.date`,
`Col.instant` and `Col.span`. `Col.v` takes any kind, as in
`Col.v (Kind.list Kind.int) "tokens"`.

A handle of a kind binds a column whose type that kind reads: `Col.int` binds
`int8` to `uint64`, and `Col.string` binds text and categorical columns. A
verb binds its handles when it is applied, and a missing column, or one of
another kind, is a problem that the verb reports.

## Types

Every expression has a column type once its verb binds it, and the verb
infers its output schema from them:

```ocaml
# #install_printer Schema.pp;;
# Query.of_table losses
  |> Query.derive
       Expr.[
         "next" := step + int 100;
         "half" := loss /. float 2.;
         "one" := int 1;
       ]
  |> Query.schema;;
- : Schema.t =
run string, step int32, loss float64, next int32, half float64, one int64
```

- **Operands meet at a common type**, the one of their types that holds every
  value of the others: `int8` and `int16` meet at `int16`, and a categorical
  and a string at `string`. Types that do not meet, such as `int8` and
  `uint8`, are a problem: `cast` one first.
- **A literal takes the type of the operand it meets**, so `step + int 100`
  stays `int32`. The literal must fit that type. Where a literal meets no
  column, it takes its kind's default: `int64`, `float64`, `bool`, `string`,
  `date`, `datetime[ns, UTC]` or `duration[ns]`.
- **Results follow from operand types.** Integer arithmetic stays at the
  common type and wraps on overflow, `count` and `rows` are `int64`, and
  `mean`, `std` and `quantile` are `float64`.

`cast` converts between types. It is exact: a float with a fraction, or a
value outside the target's range, fails the run, and the error names the
value and the row.

## Nulls and Comparisons

Elementwise operations are null where an operand is null. Integer division
and `mod` are also null where the divisor is zero.

Comparisons follow one total order, the order that sorting and grouping use.
Floats order from `neg_infinity` to `infinity`, then NaN; `-0.` equals `0.`,
and NaN equals NaN. A comparison with a null is null, and `&&`, `||` and `not`
treat null as unknown: `false && null` is `false`, and `true && null` is null.
`filter` keeps only the rows where its predicate is `true`.

These functions handle nulls explicitly:

| Expression | Value |
|---|---|
| `is_null x` | `true` where `x` is null, never null |
| `coalesce [ x; y ]` | the first of `x` and `y` that is not null |
| `if_ c a b` | `a` where `c` is `true`, `b` where it is `false` or null |
| `is_in [ v0; v1 ] x` | `true` where `x` is one of the values, `false` elsewhere, null `x` included |

```ocaml
let flags =
  Query.(
    of_table losses
    |> select
         Expr.[
           "loss" := loss;
           "high" := loss > float 2.;
           "level" := if_ (loss > float 2.) (string "high") (string "low");
           "filled" := coalesce [ loss; float 0. ];
         ])
```

```text
table 8 rows × 4 columns
 loss     high   level   filled
 float64  bool   string  float64
 2.31000  true   high    2.31000
 1.87000  false  low     1.87000
 1.98000  false  low     1.98000
 1.55000  false  low     1.55000
 2.40000  true   high    2.40000
       ∅  ∅      low     0.00000
 1.71000  false  low     1.71000
 1.52000  false  low     1.52000
```

## Verbs

| Verb | Rows | Columns |
|---|---|---|
| `select os` | the input's | exactly the outputs `os` |
| `derive os` | the input's | the input's, then `os`; an output replaces the column of its name |
| `filter p` | those where `p` is `true` | the input's |
| `sort keys` | reordered, stably | the input's |
| `slice ~offset ~length` | a range; a negative `offset` counts from the end | the input's |
| `aggregate ~by os` | one per group | the keys, then `os` |
| `join ~on right` | pairs of matching rows | see [Joins](03-joins.md) |
| `append rest` | the input's, then `rest`'s | the input's, each at the common type |

Each verb states the order of its rows, and outputs of one verb read the
input's columns, never each other. Sort keys are `Order.asc` and
`Order.desc` of a column name. Nulls go last in both directions unless
`Order.nulls_first`.

`aggregate ~by` groups rows whose keys are the same, null being one key and
NaN another, and lists the groups in order of first appearance. With
`~by:[]` it makes one group of every row, so one row of output even from no
rows.

`Kit` holds compositions of the verbs, each documented by its definition:
`head`, `tail`, `top_k`, `distinct`, `count_by`, `value_counts`, `describe`,
`null_count`, `drop`, `rename`, `complete`, `one_hot`, `union` and
`categorize`:

```ocaml
let summary = Kit.describe (Query.of_table losses)
```

```text
table 2 rows × 10 columns
 column  count  nulls  mean       std         min        q25        median     q75        max
 string  int64  int64  float64    float64     float64    float64    float64    float64    float64
 step        8      0  250.00000  119.522861  100.00000  175.00000  250.00000  325.00000  400.00000
 loss        7      1    1.90571    0.348370    1.52000    1.63000    1.87000    2.14500    2.40000
```

## Reductions

An expression's type carries its **shape**: `row`, one value per row of the
frame, or `agg`, one value per frame. Handles are `row`, a reduction takes a
`row` expression to `agg`, and elementwise operations keep their operands'
shape, so `sum (w *. x) /. sum w` is `agg`. `select`, `derive` and `filter`
take `row` expressions, and `aggregate` takes `agg` ones. Passing the wrong
shape is a type error.

| Reduction | Value |
|---|---|
| `rows` | the frame's number of rows |
| `count x` | the number of non-null values |
| `sum x`, `mean x`, `std x`, `var x` | arithmetic over the non-null values |
| `min x`, `max x`, `median x`, `quantile p x` | order statistics |
| `first x`, `last x` | the first and last non-null values in frame order |
| `only x` | the one distinct non-null value; several fail the run |
| `n_unique x` | the number of distinct values, null counting as one |
| `arg_min x`, `arg_max x` | the position in the frame of the first extreme |

Reductions skip nulls, except `rows`, `count` and `n_unique`. Over no values,
`sum`, `count` and `n_unique` are `0` and the others are null.

```ocaml
let per_run =
  Query.(
    of_table losses
    |> aggregate ~by:[ "run" ]
         Expr.[
           "evals" := count loss;
           "best" := min loss;
           "improved" := last loss < first loss;
         ])
```

```text
table 2 rows × 4 columns
 run     evals  best     improved
 string  int64  float64  bool
 a           4  1.55000  true
 b           3  1.52000  true
```

## Frames

A verb gives its expressions a frame: `select`, `derive` and `filter` their
input's rows, and `aggregate` each group's rows, in input order. `over`
evaluates an expression in a frame of its own and returns the results to the
rows:

- `over e` evaluates `e` over the whole enclosing frame. A reduction is
  broadcast to every row, so `loss -. over (mean loss)` centres the losses.
- `over ~by:[ "run" ] e` partitions the frame by the columns `by` and
  evaluates `e` in each partition.
- `over ~order:[ Order.asc "step" ] e` orders each partition first.

Two row expressions use the frame's order: `shift n x` is `x` from `n` rows
earlier, null past the edge, and `rank x` is `x`'s 1-based rank among the
frame's non-null values, ties taking the lowest rank.

```ocaml
let in_run e = Expr.over ~by:[ "run" ] ~order:[ Order.asc "step" ] e

let progress =
  Query.(
    of_table losses
    |> derive
         Expr.[
           "change" := loss -. in_run (shift 1 loss);
           "rank" := in_run (rank loss);
           "z" := (loss -. over (mean loss)) /. over (std loss);
         ])
```

```text
table 8 rows × 6 columns
 run     step   loss     change     rank   z
 string  int32  float64  float64    int64  float64
 a         100  2.31000          ∅      4   1.160506
 a         200  1.87000  -0.440000      2  -0.102518
 a         300  1.98000   0.110000      3   0.213238
 a         400  1.55000  -0.430000      1  -1.021081
 b         100  2.40000          ∅      3   1.418851
 b         200        ∅          ∅      ∅          ∅
 b         300  1.71000          ∅      2  -0.561799
 b         400  1.52000  -0.190000      1  -1.107196
```

The change at `b`, step 300 is null because the loss before it is null.

Frames nest. Inside `aggregate`, `over (min loss)` is the group's minimum on
each of the group's rows, so this is the step of each run's best loss:

```ocaml
let best_step =
  Query.(
    of_table losses
    |> aggregate ~by:[ "run" ]
         Expr.[ "best_step" := first (if_ (loss = over (min loss)) step null) ])
```

A whole-input `over` at the top of `derive` or `filter` needs every row
before it can return one, so that step holds its input in memory.

## Many Columns at Once

A selector chooses columns by name, type or kind, and a verb resolves it
against its input: `Sel.all`, `Sel.names`, `Sel.prefix`, `Sel.suffix`,
`Sel.of_kind` and `Sel.where`, combined with `+`, `-` and `Sel.inter`.

- `keep sel` outputs the selected columns unchanged.
- `across kind sel f` applies `f` to each selected column, read as `kind`.
- `each sel { column }` applies a function that works on a column of any
  type.

```ocaml
let standardize x = Expr.((x -. over (mean x)) /. over (std x))

let standardized =
  Query.(
    of_table losses
    |> derive
         Expr.[ across Kind.float Sel.(of_kind Kind.float) (fun n x -> n := standardize x) ])

let nulls =
  Query.(
    of_table losses
    |> aggregate ~by:[]
         Expr.[ each Sel.all { column = (fun n x -> n := rows - count x) } ])
```

## OCaml Functions

`const f $ x $ y` applies an OCaml function to each row's values, and is null
where an argument is. `option x` passes nulls as `None`, and `of_option`
turns `None` back into null. The result's type comes from what it meets, or
from `store`:

```ocaml
let labels =
  Query.(
    of_table losses
    |> select
         Expr.[
           "label" := store Type.string (const (Printf.sprintf "%s@%d") $ Col.string "run" $ step);
         ])
```

`nx { f }` applies an elementwise nx function to numeric values, as in
`nx { f = Nx.exp } x`. It runs inside talon's computation, at nx's speed.

## Text

`Expr.Str` works on text by Unicode scalar values: `length`, `slice`,
`matches` with a `literal`, `prefix`, `suffix` or `pieces` pattern, `split`
and `replace` on literals, and `parse`, which reads text as a typed value.

## Time

Dates read as `Time.date`, datetimes as `Time.instant` and durations and
times of day as `Time.span`. `Expr.Temporal` computes with them:

```ocaml
let at s = Option.get (Time.of_s (Int64.of_int s))

let sessions =
  Talon.v
    [
      ("file", Column.v Type.string
         [| "sub-01_ses-01.nwb"; "sub-01_ses-02.nwb"; "sub-02_ses-01.nwb" |]);
      ("start", Column.v (Type.datetime ~zone:"UTC" Type.Ms)
         [| at 1710495000; at 1710581523; at 1711010000 |]);
    ]

let file = Col.string "file"
let start = Col.instant "start"

let described =
  Query.(
    of_table sessions
    |> select
         Expr.[
           "parts" := Str.split "_" (Str.replace ".nwb" ~by:"" file);
           "weekday" := Temporal.field `Weekday start;
           "end" := Temporal.add start (span (Time.Span.minutes 90));
           "since" := Temporal.diff start (over (min start));
         ])
```

```text
table 3 rows × 4 columns
 parts                 weekday  end                   since
 list[string]          int64    datetime[ms, UTC]     duration[ms]
 ["sub-01"; "ses-01"]        5  2024-03-15T11:00:00Z  0s
 ["sub-01"; "ses-02"]        6  2024-03-16T11:02:03Z  24h2m3s
 ["sub-02"; "ses-01"]        4  2024-03-21T10:03:20Z  143h3m20s
```

`Temporal.floor` and `Temporal.offset` move dates and datetimes by
`Time.step`s: calendar months, weeks and days, or exact spans. `parse` and
`format` read and write text in a `%Y-%m-%d`-style format. The calendar of a
datetime with a zone is read in UTC; asking for it in another zone is a
problem.

## How a Query Runs

`Query.run` optimizes the plan first, and `Query.optimize` shows the result:

- filters move toward the files, past the steps that keep the rows they see,
  and a file applies those it can, as Parquet does with row group statistics;
- each file reads only the columns that some step reads;
- `sort` followed by `slice` from the start sorts only the rows it keeps;
- literal arithmetic is computed once, and equal subplans run once.

The rewrites never change a result. A run fails at the first row, in plan
order, where a step fails, whatever the batches or the number of cores, so
the same query over the same data gives the same rows and the same error.

`Query.fold` runs a query batch by batch, for results that need not fit in
memory at once, and `Query.run` folds the batches into one table.

The examples [03-expressions](../examples/03-expressions/README.md) and
[07-features](../examples/07-features/README.md) run expressions like these.
