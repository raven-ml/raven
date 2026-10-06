(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Filter, group, join and sort over the 40k-row fixture, the same work as
   bench_talon.py does in pandas; then the operations that read validities, over
   a generated table with nulls, and a read of a Parquet file with nulls. *)

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

(* Nulls *)

(* A table of 2^20 rows drawn from a seeded key: an int64 [key] in [0, 1000)
   with 1% nulls, a float64 [f] and a string [s] with 10% nulls each, two bool
   columns [a] and [b] with 10% nulls each, and the columns [i] (int64) and [g]
   (float64) without nulls. *)
let n = 1 lsl 20
let keys = Nx.Rng.split ~n:13 (Nx.Rng.key 22)

let valid k p =
  Nx.greater_equal_s (Nx.Rng.uniform keys.(k) Nx.float64 [| n |]) p

let ints k high = Nx.cast Nx.int64 (Nx.Rng.randint keys.(k) ~high [| n |])
let bools k = Nx.Rng.bernoulli keys.(k) (Nx.full Nx.float64 [| n |] 0.5)

let nullable ?valid x =
  Column.of_tensor ?validity:(Option.map (Nx.cast Nx.bit) valid) x

let key_values = ints 0 1000
let key_valid = valid 1 0.01

let strings =
  let values = Nx.to_array (ints 2 100_000) in
  let valid = Nx.to_array (valid 3 0.1) in
  Column.of_options Type.string
    (Array.init n (fun r ->
         if valid.(r) then Some ("s" ^ Int64.to_string values.(r)) else None))

let nulls =
  Talon.v
    [
      ("key", nullable ~valid:key_valid key_values);
      ( "f",
        nullable ~valid:(valid 4 0.1)
          (Nx.Rng.uniform keys.(5) Nx.float64 [| n |]) );
      ("s", strings);
      ("a", nullable ~valid:(valid 6 0.1) (bools 7));
      ("b", nullable ~valid:(valid 8 0.1) (bools 9));
      ("i", nullable (ints 10 1_000_000));
      ("g", nullable (Nx.Rng.uniform keys.(11) Nx.float64 [| n |]));
    ]

let run q = Error.get_ok (Query.run q)

(* [filter th] keeps the rows whose [key] is below [th]: [th / 10] percent of
   them. *)
let filter th =
  run Query.(of_table nulls |> filter Expr.(Col.int "key" < int th))

let kleene e = run Query.(of_table nulls |> select Expr.[ "r" := e ])

let take_random =
  let indices = ints 12 n in
  let t = Talon.v [ ("key", nullable ~valid:key_valid key_values) ] in
  fun () -> Talon.take indices t

(* Three parts of odd lengths, the second and third starting inside a byte of
   the validity. *)
let concat_3 =
  let bits = Nx.cast Nx.bit key_valid in
  let part offset length =
    let values = Nx.slice [ Nx.R (offset, offset + length) ] key_values in
    let validity = Nx.slice [ Nx.R (offset, offset + length) ] bits in
    Talon.v [ ("key", Column.of_tensor ~validity values) ]
  in
  let third = n / 3 in
  let t =
    Talon.of_batches
      [
        part 0 (third + 5);
        part (third + 5) (third - 3);
        part ((2 * third) + 2) (n - (2 * third) - 2);
      ]
  in
  fun () -> Talon.column t "key"

let joined_nulls =
  let right =
    Talon.v
      [
        ("key", Column.v Type.int64 (Array.init 500 (fun k -> 2 * k)));
        ("w", Column.v Type.float64 (Array.init 500 float_of_int));
      ]
  in
  Query.(
    of_table nulls |> join ~kind:Left ~on:(Join.keys [ "key" ]) (of_table right))

let parquet = Filename.concat data_dir "nulls.parquet"

let read_parquet () =
  run (Query.of_source (Error.get_ok (Talon_parquet.file parquet)))

let () =
  Thumper.run "talon"
    [
      Thumper.group "Talon"
        [
          Thumper.bench "Filter/high_value" (fun () -> total (filtered ()));
          Thumper.bench "Group/category_region" (fun () -> total (grouped ()));
          Thumper.bench "Join/customer_lookup" (fun () -> total (joined ()));
          Thumper.bench "Sort/amount_desc" (fun () -> first_amount (sorted ()));
          Thumper.bench "Nulls/filter-1pct" (fun () -> filter 10);
          Thumper.bench "Nulls/filter-50pct" (fun () -> filter 500);
          Thumper.bench "Nulls/filter-90pct" (fun () -> filter 900);
          Thumper.bench "Nulls/kleene-and" (fun () ->
              kleene Expr.(Col.bool "a" && Col.bool "b"));
          Thumper.bench "Nulls/kleene-or" (fun () ->
              kleene Expr.(Col.bool "a" || Col.bool "b"));
          Thumper.bench "Nulls/take-random" take_random;
          Thumper.bench "Nulls/concat-3" concat_3;
          Thumper.bench "Nulls/sort" (fun () ->
              run Query.(of_table nulls |> sort [ Order.asc "key" ]));
          Thumper.bench "Nulls/group-sum" (fun () ->
              run
                Query.(
                  of_table nulls
                  |> aggregate ~by:[ "key" ] Expr.[ "f" := sum (Col.float "f") ]));
          Thumper.bench "Nulls/join-left" (fun () -> run joined_nulls);
        ];
      Thumper.group "Talon_parquet"
        [ Thumper.bench "read-nullable" read_parquet ];
    ]
  |> exit
