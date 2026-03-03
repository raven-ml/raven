(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let () =
  let df =
    create
      [
        ("city", Col.string [| "Paris"; "Paris"; "Lyon"; "Lyon"; "Nice" |]);
        ("sales", Col.float64 [| 1200.; 800.; 450.; 900.; 500. |]);
        ("units", Col.int32 [| 10l; 8l; 5l; 9l; 6l |]);
      ]
  in

  Printf.printf "== original ==\n";
  print df;

  Printf.printf "\n== sorted by sales desc ==\n";
  let df_sorted = sort_values ~ascending:false df "sales" in
  print df_sorted;

  Printf.printf "\n== group by city, show sums ==\n";
  let groups = group_by df (Row.string "city") in
  List.iter
    (fun (city_name, sub) ->
      let total_sales = Agg.sum sub "sales" in
      let total_units = Agg.sum sub "units" in
      Printf.printf "- %s: sales=%.0f units=%.0f\n" city_name total_sales
        total_units)
    groups;
  ()
