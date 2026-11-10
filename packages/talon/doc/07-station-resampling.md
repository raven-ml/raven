# Station resampling

A weather station reports once a day, and an analysis often wants months. A
daily archive holds sentinels for missing readings, quality flags and one row
per element per day. A month counts only when enough of its days were
measured. This page uses the terms of [Expressions and
Frames](02-expressions-and-frames.md).

The fixture [`data/ghcn_daily.csv`](data/ghcn_daily.csv) follows the daily
files of the Global Historical Climatology Network. It holds two stations and
two months of daily maximum and minimum temperature, in tenths of a degree
Celsius. `-9999` stands for a value the station did not report, and a quality
flag sits beside each reading:

```text
station,date,element,value,qflag
LIMA,2024-01-01,TMAX,268,
LIMA,2024-01-01,TMIN,201,
```

```ocaml
open Talon

let show q = Format.printf "%a@." Talon.pp (Error.get_ok (Query.run q))
let path = "data/ghcn_daily.csv"
```

## Reading sentinels as null

`-9999` is a number the file writes in place of a reading it lacks. Naming it
in `~nulls` makes it null as the file is read, so no arithmetic can use it. A
blank field is null too, and `qflag` is blank where a reading passed every
check. `Talon_csv.file` sniffs the format here, and each query that reads the
source opens the file again:

```ocaml
let daily = Error.get_ok (Talon_csv.file ~nulls:[ "-9999" ] path)
```

```ocaml
# #install_printer Schema.pp;;
# Query.(schema (of_source daily));;
- : Schema.t =
station string, date date, element string, value int64, qflag string
```

## Quality and units

A row with a flag failed a check, and the filter drops it. `value` counts
tenths of a degree, so a cast and a division turn it into degrees:

```ocaml
let clean =
  Query.(
    of_source daily
    |> filter Expr.(is_null (Col.string "qflag"))
    |> select
         Expr.[
           keep Sel.(names [ "station"; "date"; "element" ]);
           "value" := cast Type.float64 (Col.int "value") /. float 10.;
         ])
```

## One column per element

`Kit.wider` gives each element its own column, one row per station and date.
Its columns are the categories of the key's categorical type, so the plan's
schema is known before a row is read. This archive has two elements, and the
cast below states them. `Kit.categorize` reads the dictionary from the data
instead, running a query to do so:

```ocaml
let elements = [| "TMAX"; "TMIN" |]

let wide =
  clean
  |> Query.derive
       Expr.[
         "element" :=
           cast (Type.categorical elements) (Col.string "element");
       ]
  |> Kit.wider ~by:[ "station"; "date" ] ~key:"element" ~value:"value"
```

An element missing from `elements` fails the run, and the error names the step
and the row. A cell of `wide` is null when its reading was flagged or was the
sentinel, and a date the file lacks has no row.

## Months with enough days

`Expr.Temporal.floor` moves a date to the first day of its calendar month, and
the month is a group key. `count tmax` is the number of non-null values, the
days with a usable maximum, and `mean` skips nulls. Here a month needs 25 such
days, and its mean is null otherwise:

```ocaml
let month = Expr.Temporal.floor (Time.Months 1) (Col.date "date")
let tmax = Col.float "TMAX"
let tmin = Col.float "TMIN"
let enough = 25

let monthly_mean x = Expr.(if_ (count x >= int enough) (mean x) null)

let monthly =
  Query.(
    wide
    |> derive Expr.[ "month" := month ]
    |> aggregate ~by:[ "station"; "month" ]
         Expr.[
           "tmax" := monthly_mean tmax;
           "tmin" := monthly_mean tmin;
           "tmax_days" := count tmax;
           "tmin_days" := count tmin;
         ]
    |> sort [ Order.asc "station"; Order.asc "month" ])
```

```ocaml
# let () = show monthly;;
table 4 rows × 6 columns
 station  month       tmax      tmin      tmax_days  tmin_days
 string   date        float64   float64   int64      int64
 LIMA     2024-01-01  27.09032  20.30645         31         31
 LIMA     2024-02-01  27.27778  20.37241         27         29
 OSLO     2024-01-01   1.49032  -4.99355         31         31
 OSLO     2024-02-01         ∅  -4.98889         21         27
```

Oslo's February has 21 days with a usable maximum, after two absent rows, five
sentinels and one flagged reading, so its `tmax` is null and the row stays in
the table. `tmin_days` differs from `tmax_days` because `TMIN` has neither
flags nor sentinels in this file.

## Rolling over gaps

`over ~by` makes one frame per station, and `window ~on` keys it by date, so
it reduces the rows of the previous seven calendar days: dates after the row's
date minus seven days, up to its own. A window without `on` keys rows by their
place and would reach back past a gap for seven rows. The filter comes after
the `derive`, so each window sees every row of its station:

```ocaml
let week = Expr.span (Time.Span.days 7)
let feb d = Expr.date (Option.get (Time.Date.of_civil (2024, 2, d)))

let last_week r =
  Expr.(over ~by:[ "station" ] (window ~on:(Col.date "date") ~before:week r))

let rolling =
  Query.(
    wide
    |> derive
         Expr.[
           "tmax_7d" := last_week (mean tmax);
           "days_7d" := last_week (count tmax);
         ]
    |> filter
         Expr.(
           Col.string "station" = string "OSLO"
           && Col.date "date" >= feb 3
           && Col.date "date" <= feb 9)
    |> select Expr.[ keep Sel.(names [ "date"; "TMAX"; "tmax_7d"; "days_7d" ]) ])
```

```ocaml
# let () = show rolling;;
table 5 rows × 4 columns
 date        TMAX     tmax_7d  days_7d
 date        float64  float64  int64
 2024-02-03  4.00000  3.70000        7
 2024-02-04  3.60000  3.84286        7
 2024-02-07  1.70000  3.54000        5
 2024-02-08        ∅  3.37500        4
 2024-02-09        ∅  3.10000        3
```

The file has no rows for 5 and 6 February, so the window ending on 7 February
holds five rows and `days_7d` is 5. The sentinels from 8 February on leave
their rows in place and lower the count of usable days further.

## Contracts

In [`talon_csv.mli`](../lib/csv/talon_csv.mli), the Values section states
blank fields and null tokens, and the Reading section `Talon_csv.file`.

In [`talon.mli`](../lib/talon.mli):

- nulls in elementwise operations and reductions, the preamble of `Expr`;
  `Expr.count`, its Reductions section
- `Expr.cast` and `Expr.if_`, the Elementwise operations section of `Expr`
- `Kit.wider`, the Queries section of `Kit`; `Kit.categorize`, its Runs section
- what a failed run reports, the Running section of `Query`
- `Expr.Temporal.floor`, the Time section of `Expr`
- `over` and `window`, the Frames and Windows sections of `Expr`
