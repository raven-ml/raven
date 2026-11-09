# Peri-Event Analysis

An electrophysiology session records when each neuron, or unit, fired and when
each trial's events happened. A peri-event analysis asks how firing changes
around an event. This page counts each unit's spikes in 100 ms bins around the
cue that starts each trial, over the periods where the recording is good, and
ends with a tensor of trials by units by bins. It uses the terms of
[Expressions and Frames](02-expressions-and-frames.md) and
[Joins](03-joins.md).

The unit table holds each unit's spike times as a list. Trial times share the
spike clock, and `side` is where the stimulus appeared. Unit 2 did not fire,
and its row stays so that the tensor has a place for it:

```ocaml
open Talon

let ( let* ) = Result.bind
let show q = Format.printf "%a@." Talon.pp (Error.get_ok (Query.run q))
let ints a = Column.v Type.int64 a
let floats a = Column.v Type.float64 a

let units =
  Talon.v
    [
      ("unit", ints [| 0; 1; 2 |]);
      ( "spike_times",
        Column.v
          (Type.list Type.float64)
          [|
            [| 0.92; 1.01; 1.03; 1.21; 1.50; 1.95;
               2.12; 2.95; 3.02; 3.92; 4.05; 4.15 |];
            [| 0.98; 1.05; 1.11; 2.02; 2.24; 4.22 |];
            [||];
          |] );
    ]

let trials =
  Talon.v
    [
      ("trial", ints [| 0; 1; 2; 3 |]);
      ("cue", floats [| 1.; 2.; 3.; 4. |]);
      ("side", Column.v Type.string [| "left"; "right"; "left"; "right" |]);
    ]
```

## Spikes as Rows

`Query.unnest` makes one row per element of a list and repeats the row's other
columns, so a join can pair each spike with a trial. An empty list gives no
row, so unit 2 has none here. The `select` drops the list, which the rows no
longer need:

```ocaml
let spikes =
  Query.(
    of_table units
    |> unnest Expr.[ "time" := Col.v Kind.(list float) "spike_times" ]
    |> select Expr.[ keep Sel.(names [ "unit"; "time" ]) ])
```

## Good Periods

The animal was running in two periods, and an artifact spoilt part of one. A
table of `s` and `e` columns is a set of periods, each holding its start and
excluding its stop. `Kit.subtract_intervals` removes `artifacts` from
`running` and returns the result in normal form, sorted by start, each period
stopping before the next one starts:

```ocaml
let running =
  Query.of_table
    (Talon.v [ ("s", floats [| 0.5; 3.8 |]); ("e", floats [| 3.5; 5.0 |]) ])

let artifacts =
  Query.of_table (Talon.v [ ("s", floats [| 2.85 |]); ("e", floats [| 3.05 |]) ])

let good = running |> Kit.subtract_intervals ("s", "e") artifacts
```

```ocaml
# let () = show good;;
table 3 rows × 2 columns
 s         e
 float64   float64
 0.500000  2.85000
 3.050000  3.50000
 3.800000  5.00000
```

## Trials Inside the Good Periods

A trial counts when its whole window lies inside one good period. Two
comparisons state that, and a `Semi` join keeps each trial that has a match,
once:

```ocaml
let pre = 0.1
let post = 0.3
let cue = Col.float "cue"

let windows =
  Query.(
    of_table trials
    |> derive Expr.[ "start" := cue -. float pre; "stop" := cue +. float post ]
    |> join ~kind:Semi ~on:Join.(ge "start" "s" && le "stop" "e") good
    |> derive Expr.[ "trial_i" := rank (Col.int "trial") - int 1 ])
```

Trial 2's window, from `2.9` to `3.3`, crosses the artifact and has no match.
`trial_i` numbers the survivors from zero for the tensor. `rank` is 1-based
and each trial has one row, so subtracting one suffices.

## Binning

`Join.inside` pairs each spike with the window that holds it, which includes
its start and excludes its stop. An inner join drops a spike outside every
window, and keeps the right side's columns, so `cue` is there to subtract.
`Expr.cut` then assigns each spike's time relative to the cue to a bin. Bin
`i` runs from `edges.(i)`, included, to `edges.(i + 1)`, excluded, so bin 0 is
the 100 ms before the cue:

```ocaml
let edges = [| -0.1; 0.; 0.1; 0.2; 0.3 |]
let width = 0.1
let n_bins = Array.length edges - 1

let binned =
  Query.(
    spikes
    |> join ~on:(Join.inside "time" ("start", "stop")) windows
    |> derive Expr.[ "bin" := cut edges (Col.float "time" -. cue) ]
    |> aggregate ~by:[ "trial"; "unit"; "bin" ] Expr.[ "spikes" := rows ])
```

## Bins With No Spikes

A combination of trial, unit and bin with no spike has no row in `binned`, and
a mean over `binned` would average only the bins that fired. The grid holds
one row per trial, unit and bin. A left join from it keeps every row, its
count null where `binned` has none, and `coalesce` turns those nulls into
zeros:

```ocaml
let bins = Query.of_table (Talon.v [ ("bin", ints (Array.init n_bins Fun.id)) ])

let unit_ids =
  Query.(of_table units |> select Expr.[ keep Sel.(names [ "unit" ]) ])

let cells =
  Query.(
    windows
    |> select Expr.[ keep Sel.(names [ "trial"; "trial_i"; "side" ]) ]
    |> join ~on:Join.all unit_ids
    |> join ~on:Join.all bins
    |> join ~kind:Left ~each_left:At_most_one
         ~on:(Join.keys [ "trial"; "unit"; "bin" ])
         binned
    |> derive Expr.[ "spikes" := coalesce [ Col.int "spikes"; int 0 ] ])
```

The grid comes from `units`, `windows` and `bins`. `Kit.complete` builds the
combinations of the values a query already holds, so over `binned` it would
miss a unit or bin with no spike.

## From Counts to Rates

With zeros in place, a unit's peri-stimulus time histogram is the mean over
trials of each bin's count, divided by the width of a bin:

```ocaml
let psth =
  Query.(
    cells
    |> aggregate ~by:[ "side"; "unit"; "bin" ]
         Expr.[ "rate_hz" := mean (Col.int "spikes") /. float width ]
    |> filter Expr.(Col.int "unit" = int 0)
    |> sort [ Order.asc "side"; Order.asc "bin" ])
```

```ocaml
# let () = show psth;;
table 8 rows × 4 columns
 side    unit   bin    rate_hz
 string  int64  int64  float64
 left        0      0  10.00000
 left        0      1  20.00000
 left        0      2   0.00000
 left        0      3  10.00000
 right       0      0  10.00000
 right       0      1   5.00000
 right       0      2  10.00000
 right       0      3   0.00000
```

Only trial 0 remains on the left, and trials 1 and 3 are on the right. In bin
1, unit 0 fired once in trial 3 and not in trial 1, a mean count of `0.5` and
a rate of `0.5 / 0.1 = 5` spikes per second.

## The Tensor

`Talon.to_dense` scatters a column into a tensor, one axis per key column. Its
axes take integers from zero. `trial_i` and `bin` hold those, and so does
`unit` in this fixture; other unit ids would need the numbering `trial_i` got:

```ocaml
let tensor =
  let* t = Query.run cells in
  let* w = Query.run windows in
  let x, validity =
    Talon.to_dense Nx.int64
      ~axes:[ ("trial_i", Talon.rows w); ("unit", 3); ("bin", n_bins) ]
      "spikes" t
  in
  Ok (x, Option.is_none validity)
```

```ocaml
# let () =
    let x, complete = Error.get_ok tensor in
    Format.printf "shape %a, every cell observed: %b@.%a@." Nx.pp_shape
      (Nx.shape x) complete Nx.pp x;;
shape [3; 3; 4], every cell observed: true
int64 [3; 3; 4]
[[[1, 2, 0, 1],
  [1, 1, 1, 0],
  [0, 0, 0, 0]],
 [[1, 0, 1, 0],
  [0, 1, 0, 1],
  [0, 0, 0, 0]],
 [[1, 1, 1, 0],
  [0, 0, 0, 1],
  [0, 0, 0, 0]]]
```

`cells` names every combination, so `to_dense`'s validity is `None`, and every
cell holds a count.

## Contracts

In [`talon.mli`](../lib/talon.mli):

- `Query.unnest`, the Verbs section of `Query`, and the Lists section of `Expr`
- `Kit.subtract_intervals` and its normal form, the Interval sets section of
  `Kit`
- `Semi`, `Left` and the inner join, the Kinds and counts section of `Join`;
  `Join.inside`, its Conditions section
- `Expr.rank`, the Frames section of `Expr`; `Expr.cut` and `Expr.coalesce`, its
  Elementwise operations section
- `Kit.complete`, the Queries section of `Kit`
- `Talon.to_dense`, the Tables section
