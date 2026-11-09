# Point-in-Time Features

A model that predicts from an event's features may use only what was known
when the event happened. A training table therefore asks, for each labelled
event, what the features said just before it. An answer that used a later row
would leak the future into training. This page uses the terms of [Expressions
and Frames](02-expressions-and-frames.md) and [Joins](03-joins.md).

Each label records whether a customer left, stamped when the model would have
been asked. The features are computed from the transactions. Day `d` below is
`d` days after 1 January 2024:

```ocaml
open Talon

let ( let* ) = Result.bind
let show q = Format.printf "%a@." Talon.pp (Error.get_ok (Query.run q))

let jan_1 = 1_704_067_200 (* 2024-01-01T00:00:00Z, in seconds *)
let day d = Option.get (Time.of_s (Int64.of_int (jan_1 + (d * 86_400))))
let stamps a = Column.v (Type.instant Type.S) (Array.map day a)

let transactions =
  Talon.v
    [
      ("user", Column.v Type.string
         [| "a"; "a"; "a"; "a"; "b"; "b"; "c" |]);
      ("ts", stamps [| 1; 3; 9; 20; 2; 15; 5 |]);
      ("amount", Column.v Type.float64
         [| 10.; 20.; 5.; 30.; 7.; 1.; 100. |]);
    ]

let labels =
  Talon.v
    [
      ("user", Column.v Type.string [| "a"; "a"; "b"; "c" |]);
      ("at", stamps [| 10; 20; 30; 4 |]);
      ("churned", Column.v Type.bool [| false; true; false; true |]);
    ]

let ts = Col.instant "ts"
let at = Col.instant "at"
let amount = Col.float "amount"
```

## Features at Each Transaction

A feature here is each customer's rolling total over the previous seven days,
one value per transaction. `over ~by` makes one frame per customer. A window
keyed by the timestamp reduces the rows of its frame whose keys lie after the
row's minus `~before` and up to its own, so it holds the seven days ending at
the row:

```ocaml
let week = Expr.span (Time.Span.days 7)
let last_week r = Expr.(over ~by:[ "user" ] (window ~on:ts ~before:week r))

let features =
  Query.(
    of_table transactions
    |> derive
         Expr.[
           "spend_7d" := last_week (sum amount);
           "txs_7d" := last_week rows;
         ]
    |> Kit.drop Sel.(names [ "amount" ]))
```

```ocaml
# let () = show features;;
table 7 rows × 4 columns
 user    ts                    spend_7d   txs_7d
 string  datetime[s, UTC]      float64    int64
 a       2024-01-02T00:00:00Z   10.00000       1
 a       2024-01-04T00:00:00Z   30.00000       2
 a       2024-01-10T00:00:00Z   25.00000       2
 a       2024-01-21T00:00:00Z   30.00000       1
 b       2024-01-03T00:00:00Z    7.00000       1
 b       2024-01-16T00:00:00Z    1.00000       1
 c       2024-01-06T00:00:00Z  100.00000       1
```

On day 9, customer `a` has spent 25 over the last week, on days 3 and 9. The
transaction on day 1 is eight days earlier and outside the window.

## Joining as of the Event

`Kit.join_asof` gives each left row its pick among the right rows of the same
customer, the feature row with the latest timestamp not after the label's.
`~strict:true` leaves out a feature row stamped at exactly the label's time,
so a transaction on day 20 cannot be the feature for a prediction on day 20. A
left row with no pick stays, its right columns null:

```ocaml
let joined =
  Query.of_table labels
  |> Kit.join_asof ~by:[ "user" ] ~strict:true ~on:("at", "ts") features
```

```ocaml
# let () = show joined;;
table 4 rows × 6 columns
 user    at                    churned  ts                    spend_7d  txs_7d
 string  datetime[s, UTC]      bool     datetime[s, UTC]      float64   int64
 a       2024-01-11T00:00:00Z  false    2024-01-10T00:00:00Z  25.00000       2
 a       2024-01-21T00:00:00Z  true     2024-01-10T00:00:00Z  25.00000       2
 b       2024-01-31T00:00:00Z  false    2024-01-16T00:00:00Z   1.00000       1
 c       2024-01-05T00:00:00Z  true     ∅                            ∅       ∅
```

`c`'s label comes before any transaction. `a`'s label on day 20 picks day 9:
its `spend_7d` is the total of the week ending on day 9, as it stood then.

## Features Too Old to Count

A feature computed weeks before a label may be stale. A tolerance is a
comparison on the pick's timestamp, here its age at the label. `each` applies
a function to every selected column whatever its type: `n` is the column's
name and `x` its expression. `if_` nulls the pick's columns where the
comparison is false, and the row stays. Outputs of one verb read the input's
columns, so `fresh` reads each pick's `ts` before it is replaced:

```ocaml
let ttl = Expr.span (Time.Span.days 14)
let fresh = Expr.(Temporal.diff at ts <= ttl)

let bounded =
  joined
  |> Query.derive
       Expr.[
         each
           Sel.(names [ "ts"; "spend_7d"; "txs_7d" ])
           { column = (fun n x -> n := if_ fresh x null) };
       ]
```

```ocaml
# let () = show bounded;;
table 4 rows × 6 columns
 user    at                    churned  ts                    spend_7d  txs_7d
 string  datetime[s, UTC]      bool     datetime[s, UTC]      float64   int64
 a       2024-01-11T00:00:00Z  false    2024-01-10T00:00:00Z   25.0000       2
 a       2024-01-21T00:00:00Z  true     2024-01-10T00:00:00Z   25.0000       2
 b       2024-01-31T00:00:00Z  false    ∅                            ∅       ∅
 c       2024-01-05T00:00:00Z  true     ∅                            ∅       ∅
```

Customer `b`'s transaction on day 15 is 15 days old on day 30, a day past the
limit, so the pick's columns are null. Where a row has no pick, `fresh` is
null, and `if_` reads null as false, so that row stays null too.

## A Matrix and a Split

An absent feature is information, so a flag records it and zero stands in for
the value. A customer's rows must stay in one fold, or a model would be tested
on a customer it trained on. `Kit.folds ~by` adds a column `fold` that gives
every row of a customer one of its `2` folds, and `~seed` repeats the
assignment:

```ocaml
let spend = Col.float "spend_7d"
let txs = Col.int "txs_7d"

let training =
  bounded
  |> Query.select
       Expr.[
         "known" := not (is_null spend);
         "spend_7d" := coalesce [ spend; float 0. ];
         "txs_7d" := coalesce [ txs; int 0 ];
         keep Sel.(names [ "user"; "churned" ]);
       ]
  |> Kit.folds ~seed:7 ~by:[ "user" ] 2
```

```ocaml
# let () = show training;;
table 4 rows × 6 columns
 known  spend_7d  txs_7d  user    churned  fold
 bool   float64   int64   string  bool     int64
 true    25.0000       2  a       false        0
 true    25.0000       2  a       true         0
 false    0.0000       0  b       false        1
 false    0.0000       0  c       true         0
```

`Talon.to_tensor` copies numeric and boolean columns into a matrix, and
`Column.to_tensor` reads the label column without a copy, booleans as
`Nx.bit`. Both raise on a null, which the `coalesce` above rules out for the
features. Here fold 0 becomes the matrix, and the other fold is the same
filter with `int 1`:

```ocaml
let tensors =
  let* t = Query.run (Query.filter Expr.(Col.int "fold" = int 0) training) in
  let x = Talon.to_tensor Nx.float32 [ "known"; "spend_7d"; "txs_7d" ] t in
  let y = Column.to_tensor Nx.bit (Talon.column t "churned") in
  Ok (Nx.shape x, Nx.shape y)
```

```ocaml
# let () =
    let x, y = Error.get_ok tensors in
    Format.printf "x %a, y %a@." Nx.pp_shape x Nx.pp_shape y;;
x [3; 3], y [3]
```

## Contracts

In [`talon.mli`](../lib/talon.mli):

- `over` and `window`, the Frames and Windows sections of `Expr`
- `Kit.join_asof`, the As-of joins section of `Kit`, and tolerances, its
  Tolerances subsection
- `Expr.each`, the Outputs section of `Expr`; `Query.derive`, the Verbs section
  of `Query`
- nulls in comparisons, the preamble of `Expr`; `Expr.if_` and `Expr.coalesce`,
  its Elementwise operations section
- `Kit.folds`, the Randomness section of `Kit`
- `Talon.to_tensor`, the Tables section; `Column.to_tensor`, the Tensors and
  ragged arrays section of `Column`
