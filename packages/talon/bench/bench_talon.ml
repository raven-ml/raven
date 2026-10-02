(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Filter, group, join and sort over the 40k-row fixture, the same work as
   bench_talon.py does in pandas. *)

open Talon

(* Resolve fixtures next to the executable, not the working directory: the bench
   rule runs with the bench dir as cwd, dune exec with the project root — the
   exe path is the one stable anchor in both. *)
let data_dir = Filename.concat (Filename.dirname Sys.executable_name) "data"

(* [load name columns categories] reads the fixture [name] as [columns], its
   text columns [categories] as categoricals, as pandas reads them. *)
let load name columns categories =
  let format = Talon_csv.format columns in
  let path = Filename.concat data_dir name in
  let source = Error.get_ok (Talon_csv.file ~format path) in
  let q = Error.get_ok (Kit.categorize categories (Query.of_source source)) in
  Error.get_ok (Query.run q)

let string = Type.Any Type.string
let int32 = Type.Any Type.int32
let float64 = Type.Any Type.float64

let transactions =
  load "transactions.csv"
    [
      ("transaction_id", int32);
      ("customer_id", int32);
      ("region", string);
      ("category", string);
      ("channel", string);
      ("amount", float64);
      ("quantity", int32);
      ("discount", float64);
      ("promo", string);
      ("event_date", string);
    ]
    [ "region"; "category"; "channel"; "promo" ]

let customers =
  load "customers.csv"
    [
      ("customer_id", int32);
      ("segment", string);
      ("region", string);
      ("status", string);
      ("loyalty_score", float64);
      ("tenure_years", int32);
    ]
    [ "segment"; "region"; "status" ]

let amount = Col.float "amount"

(* [total q] is the sum of [q]'s amounts. *)
let total q =
  let sums = Query.aggregate ~by:[] Expr.[ "amount" := sum amount ] q in
  (Column.values Kind.float (column (Error.get_ok (Query.run sums)) "amount")).(0)

let filtered () =
  Query.(
    of_table transactions
    |> filter
         Expr.(
           amount > float 120.
           && Col.int "quantity" >= int 3
           && Col.string "region" = string "EMEA"))

let grouped () =
  Query.(
    of_table transactions
    |> aggregate ~by:[ "category"; "region" ] Expr.[ "amount" := sum amount ])

(* Both tables have a [region]; talon adds no suffixes, so the customers' is
   renamed. *)
let joined () =
  Query.(
    of_table transactions
    |> join ~kind:Left
         ~on:(Join.keys [ "customer_id" ])
         (of_table customers |> Kit.rename [ ("region", "customer_region") ]))

let sorted () = Query.(of_table transactions |> sort [ Order.desc "amount" ])

let first_amount q =
  (Column.values Kind.float (column (Error.get_ok (Query.run q)) "amount")).(0)

(* Checks *)

let check name ~rows:expected_rows ~amount:expected q value =
  let got_rows = rows (Error.get_ok (Query.run q)) in
  let close = Float.abs (value -. expected) <= 1e-9 *. Float.abs expected in
  if got_rows <> expected_rows || not close then
    failwith
      (Printf.sprintf "%s: %d rows and %.17g, expected %d and %.17g" name
         got_rows value expected_rows expected)

(* The rows and amounts that bench_talon.py's pandas computes. *)
let () =
  check "filter" ~rows:511 ~amount:80012.63 (filtered ()) (total (filtered ()));
  check "group" ~rows:24 ~amount:2587546.84 (grouped ()) (total (grouped ()));
  check "join" ~rows:40000 ~amount:2587546.84 (joined ()) (total (joined ()));
  check "sort" ~rows:40000 ~amount:450.51 (sorted ()) (first_amount (sorted ()))

let () =
  Thumper.run "talon"
    [
      Thumper.group "Talon"
        [
          Thumper.bench "Filter/high_value" (fun () -> total (filtered ()));
          Thumper.bench "Group/category_region" (fun () -> total (grouped ()));
          Thumper.bench "Join/customer_lookup" (fun () -> total (joined ()));
          Thumper.bench "Sort/amount_desc" (fun () -> first_amount (sorted ()));
        ];
    ]
