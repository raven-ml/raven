(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The texts of customers.csv and transactions.csv, the filter, group, join
   and sort rows' fixture, drawn from Nx's generator with one key, so they
   are the same on every machine. bench_talon.exe reads them from memory,
   and bench_talon.py from the files write_fixtures.exe writes. *)

let customers = 1500
let transactions = 40000
let keys = Nx.Rng.split ~n:15 (Nx.Rng.key 20240127)

(* [uniform k n] is [n] draws from [[0, 1)] of [keys.(k)]. *)
let uniform k n : float array =
  Nx.to_array (Nx.Rng.uniform ~key:keys.(k) Nx.float64 [| n |])

let normal k n : float array =
  Nx.to_array (Nx.Rng.normal ~key:keys.(k) Nx.float64 [| n |])

(* [ints k lo hi n] is [n] draws from [[lo, hi]] of [keys.(k)]. *)
let ints k lo hi n : int array =
  Array.map Int32.to_int
    (Nx.to_array (Nx.Rng.randint ~key:keys.(k) ~low:lo ~high:(hi + 1) [| n |]))

(* [pick names weights u] is the name of [names] whose share of [weights],
   which sum to 1, holds [u]. *)
let pick names weights u =
  let rec go i acc =
    let acc = acc +. weights.(i) in
    if u < acc || i = Array.length names - 1 then names.(i) else go (i + 1) acc
  in
  go 0 0.

let regions = [| "EMEA"; "AMER"; "APAC"; "LATAM" |]
let segments = [| "Enterprise"; "Growth"; "SMB"; "Consumer" |]
let statuses = [| "active"; "at_risk"; "inactive" |]

let categories =
  [| "electronics"; "grocery"; "fashion"; "home"; "sports"; "toys" |]

let channels = [| "online"; "retail"; "mobile"; "partner" |]
let promos = [| "none"; "coupon"; "bundle"; "loyalty" |]

let line b cells =
  Buffer.add_string b (String.concat "," cells);
  Buffer.add_char b '\n'

(* [generate ()] is [(customers, transactions)]: 1500 customers with a
   region, segment, status, loyalty score in [[20, 98)] and tenure of 1 to 12
   years; then 40000 transactions, each of a customer, with a category,
   channel and promotion drawn by weight, an amount of lognormal(4, 0.5)
   times a seasonal factor in [[0.85, 1.25)], a quantity of 1 to 8, a
   discount in [[0, 0.25)] and a date in 2024. *)
let generate () =
  let n = customers in
  let region = ints 0 0 3 n and segment = uniform 1 n in
  let status = uniform 2 n and loyalty = uniform 3 n in
  let tenure = ints 4 1 12 n in
  let c = Buffer.create 65536 in
  line c
    [
      "customer_id"; "segment"; "region"; "status"; "loyalty_score";
      "tenure_years";
    ];
  for i = 0 to n - 1 do
    line c
      [
        string_of_int (1000 + i);
        pick segments [| 0.25; 0.2; 0.3; 0.25 |] segment.(i);
        regions.(region.(i));
        pick statuses [| 0.65; 0.2; 0.15 |] status.(i);
        Printf.sprintf "%.2f" (20. +. (78. *. loyalty.(i)));
        string_of_int tenure.(i);
      ]
  done;
  let n = transactions in
  let customer = ints 5 0 (customers - 1) n in
  let category = uniform 6 n and channel = uniform 7 n in
  let quantity = ints 8 1 8 n and z = normal 9 n and season = uniform 10 n in
  let discount = uniform 11 n and promo = uniform 12 n in
  let month = ints 13 1 12 n and day = ints 14 1 28 n in
  let t = Buffer.create (1 lsl 22) in
  line t
    [
      "transaction_id"; "customer_id"; "region"; "category"; "channel";
      "amount"; "quantity"; "discount"; "promo"; "event_date";
    ];
  for i = 0 to n - 1 do
    let amount = exp (4. +. (0.5 *. z.(i))) *. (0.85 +. (0.4 *. season.(i))) in
    line t
      [
        string_of_int (i + 1);
        string_of_int (1000 + customer.(i));
        regions.(region.(customer.(i)));
        pick categories [| 0.2; 0.25; 0.15; 0.18; 0.12; 0.1 |] category.(i);
        pick channels [| 0.45; 0.35; 0.15; 0.05 |] channel.(i);
        Printf.sprintf "%.2f" amount;
        string_of_int quantity.(i);
        Printf.sprintf "%.3f" (0.25 *. discount.(i));
        pick promos [| 0.55; 0.2; 0.15; 0.1 |] promo.(i);
        Printf.sprintf "2024-%02d-%02d" month.(i) day.(i);
      ]
  done;
  (Buffer.contents c, Buffer.contents t)

let write dir =
  let customers, transactions = generate () in
  let put name text =
    Out_channel.with_open_bin (Filename.concat dir name) (fun oc ->
        output_string oc text)
  in
  put "customers.csv" customers;
  put "transactions.csv" transactions
