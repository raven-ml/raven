# Getting Started

This page builds a table, queries it, reads files, and shows what happens
when a query is wrong.

## Installation

<!-- $MDX skip -->
```bash
opam install talon
```

Add the libraries you use to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries talon talon.csv talon.parquet))
```

## A Table

A table is a list of named columns of equal length. `Column.v` makes a column
of a type from an OCaml array, and `Column.of_options` does the same with a
null for each `None`:

```ocaml
open Talon

let day d = Option.get (Time.Date.of_civil (2024, 3, d))

let readings =
  Talon.v
    [
      ("station", Column.v Type.string [| "oslo"; "oslo"; "lima"; "lima"; "pune" |]);
      ("day", Column.v Type.date (Array.map day [| 1; 2; 1; 2; 1 |]));
      ( "temp",
        Column.of_options Type.float32
          [| Some (-3.5); Some (-1.25); Some 22.; None; Some 31.5 |] );
      ("rain", Column.of_tensor (Nx.create Nx.bool [| 5 |] [| false; true; false; false; true |]));
    ]

let () = Format.printf "%a@." Talon.pp readings
```

```text
table 5 rows × 4 columns
 station  day         temp      rain
 string   date        float32   bool
 oslo     2024-03-01  -3.50000  false
 oslo     2024-03-02  -1.25000  true
 lima     2024-03-01  22.00000  false
 lima     2024-03-02         ∅  false
 pune     2024-03-01  31.50000  true
```

`Column.of_tensor` makes a column of a 1-D nx tensor without copying it. In
quill, tables print the same way.

## Types and Kinds

A column's **type** is what it stores: `float32`, `int16`, `date`,
`datetime[ms, UTC]`, `list[string]`. Its **kind** is the OCaml type its values
read as. Several types share a kind: `int8` to `uint64` all read as `int`, and
`float16` to `float64` as `float`.

Code names kinds, never storage widths. `Column.options Kind.float` reads any
float column:

```ocaml
let temp = Talon.column readings "temp"
let temps : float option array = Column.options Kind.float temp
```

`Column.values` is the same for a column without nulls, and raises on a null.

## A Query

A query describes a table to compute. It starts from a table or a file and
goes through verbs, written inside `Query.( … )`, with expressions written
inside `Expr.( … )`:

```ocaml
let temp = Col.float "temp"

let warm =
  Query.(
    of_table readings
    |> filter Expr.(temp > float 0.)
    |> derive Expr.[ "temp_f" := (temp *. float 1.8) +. float 32. ])
```

`Col.float "temp"` is a **handle**: it names a column and reads it as
`float`. It is bound to no table, so a module of handles serves as a schema.
`:=` names an output column.

Building `warm` read nothing. Printing it shows the plan and its schema,
which the verbs inferred:

```ocaml
let () = Format.printf "%a@." Query.pp warm
```

```text
query → station string, day date, temp float32, rain bool, temp_f float32
derive ["temp_f" := temp *. 1.8 +. 32.]
└ filter (temp > 0.)
  └ table (4 columns, 5 rows)
```

`Query.run` computes the rows:

```ocaml
let warm_table = Error.get_ok (Query.run warm)
```

```text
table 2 rows × 5 columns
 station  day         temp     rain   temp_f
 string   date        float32  bool   float32
 lima     2024-03-01  22.0000  false  71.6000
 pune     2024-03-01  31.5000  true   88.7000
```

The filter dropped `lima`'s second day: its temperature is null, so
`temp > 0.` is null there, and a filter keeps only the rows where its
predicate is `true`. The literals in `temp *. 1.8 +. 32.` took the type of the
column they meet, so `temp_f` is `float32`.

## Getting Values Out

`Query.values` evaluates an expression on each row and decodes it to OCaml.
`const f $ x $ y` applies an OCaml function to the values of a row:

```ocaml
# Query.values
    Expr.(const (Printf.sprintf "%s %.1f") $ Col.string "station" $ temp)
    warm;;
- : (string array, Error.t) result = Ok [|"lima 22.0"; "pune 31.5"|]
```

For numerical work, `Talon.to_tensor` copies numeric columns into an nx
matrix, one column per name, and `Column.to_tensor` returns a column's own
buffer without a copy:

```ocaml
let m = Talon.to_tensor Nx.float32 [ "temp"; "temp_f" ] warm_table
let rain = Column.to_tensor Nx.bool (Talon.column warm_table "rain")
```

## Reading Files

`Talon_csv.file` and `Talon_parquet.file` return a **source**: a file whose
columns are known and whose rows a query reads when it runs. A CSV file's
types are sniffed from its first rows, and a Parquet file's come from its
footer:

```ocaml
let ( let* ) = Result.bind
let delay = Col.int "dep_delay"

let late_by_carrier () =
  let* flights = Talon_csv.file ~nulls:[ "NA" ] "flights.csv" in
  let* carriers = Talon_parquet.file "carriers.parquet" in
  Ok
    Query.(
      of_source flights
      |> filter Expr.(delay > int 15)
      |> aggregate ~by:[ "carrier" ]
           Expr.[ "mean_delay" := mean delay; "flights" := rows ]
      |> join ~on:(Join.keys [ "carrier" ]) ~each_left:One (of_source carriers)
      |> sort [ Order.desc "mean_delay" ])
```

- `~nulls:[ "NA" ]` reads the text `NA` as null.
- `aggregate ~by:[ "carrier" ]` makes one row per carrier, with the carrier
  and the outputs, each reduced over the carrier's rows.
- `~each_left:One` asserts that every left row finds exactly one carrier. A
  missing or duplicated carrier fails the run and names the key.

`Query.optimize` gives the plan a run runs. It reads two of the CSV file's
eleven columns:

```text
query → carrier string, mean_delay float64, flights int64, name string
sort [desc "mean_delay"]
└ join ~on:(keys ["carrier"]) ~each_left:One
  ├ aggregate ~by:["carrier"] ["mean_delay" := mean dep_delay;
  │                            "flights" := rows]
  │ └ filter (dep_delay > 15)
  │   └ csv "flights.csv" (11 columns) ~columns:["carrier"; "dep_delay"]
  └ parquet "carriers.parquet" (2 columns, 8 rows)
```

## When Something Is Wrong

A query fails in one of two ways.

**Problems in the plan** are programming errors: a missing column, a handle
of the wrong kind, two outputs of one name. A verb finds them when it is
applied, before any data is read, and raises one `Invalid_argument` that lists
them all:

```text
aggregate: 3 problems
  ~by: no column "carier". Did you mean "carrier"?
  "mean_delay" := mean dep_dly
    no column "dep_dly". Did you mean "dep_delay"?
  "late" := mean carrier
    Col.float reads float16, float32 or float64, but "carrier" is string.
  input (3 columns): carrier string, origin string, dep_delay int64
```

Some mistakes do not compile. A row expression where a reduction belongs is a
type error:

```text
Query.aggregate ~by:[ "carrier" ] Expr.[ "d" := delay ]
                                         ^^^^^^^^^^^^
Error: This expression has type row out
       but an expression was expected of type agg out
```

**Failures in the data** are values. A field that does not parse, a cast that
loses a value, or a broken join assertion makes `Query.run` return `Error e`,
which `Error.pp` prints with the plan step and the row of its input:

```text
derive ["delay" := cast int8 dep_delay]: row 1: cannot cast 130 to int8.
```

`Error.get_ok` raises `Failure` with that text, for scripts that stop at the
first failure.

The examples [01-tables](../examples/01-tables/README.md),
[02-flights](../examples/02-flights/README.md) and
[06-mistakes](../examples/06-mistakes/README.md) run this page's code.
