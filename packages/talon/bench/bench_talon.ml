(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Talon dataframe benchmarks using realistic CSV fixtures. *)

module Row = Talon.Row

module Fixtures = struct
  let data_dir = Filename.concat (Sys.getcwd ()) "talon/bench/data"

  let load_csv name dtype_spec =
    Talon_csv.read ~dtype_spec (Filename.concat data_dir name)

  let transactions =
    lazy
      (load_csv "transactions.csv"
         [
           ("transaction_id", `Int32);
           ("customer_id", `Int32);
           ("region", `String);
           ("category", `String);
           ("channel", `String);
           ("amount", `Float64);
           ("quantity", `Int32);
           ("discount", `Float64);
           ("promo", `String);
           ("event_date", `String);
         ])

  let customers =
    lazy
      (load_csv "customers.csv"
         [
           ("customer_id", `Int32);
           ("segment", `String);
           ("region", `String);
           ("status", `String);
           ("loyalty_score", `Float64);
           ("tenure_years", `Int32);
         ])

  let transactions () = Lazy.force transactions
  let customers () = Lazy.force customers
end

let force_float_sum df column =
  let total = Talon.Agg.Float.sum df column in
  ignore total

let bench_filter df =
  let filtered =
    Talon.filter_by df
      Row.(
        map3 (float64 "amount") (int32 "quantity") (string "region")
          ~f:(fun amount quantity region ->
            amount > 120.
            && Int32.compare quantity 3l >= 0
            && String.equal region "EMEA"))
  in
  force_float_sum filtered "amount"

let bench_group df =
  let groups =
    Talon.group_by df
      Row.(
        map2 (string "category") (string "region") ~f:(fun category region ->
            category ^ "|" ^ region))
  in
  let total =
    List.fold_left
      (fun acc (_key, group_df) -> acc +. Talon.Agg.Float.sum group_df "amount")
      0. groups
  in
  ignore total

let bench_join df customers =
  let joined = Talon.join df customers ~on:"customer_id" ~how:`Left () in
  force_float_sum joined "amount"

let bench_sort df =
  let sorted = Talon.sort_values ~ascending:false df "amount" in
  force_float_sum sorted "amount"

let all_benchmarks =
  let transactions = Fixtures.transactions () in
  let customers = Fixtures.customers () in
  [
    Ubench.bench "Filter/high_value" (fun () -> bench_filter transactions);
    Ubench.bench "Group/category_region" (fun () -> bench_group transactions);
    Ubench.bench "Join/customer_lookup" (fun () ->
        bench_join transactions customers);
    Ubench.bench "Sort/amount_desc" (fun () -> bench_sort transactions);
  ]
  |> fun benches -> [ Ubench.group "Talon" benches ]

let () = Ubench.run_cli all_benchmarks
