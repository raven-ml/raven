(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let () =
  let df =
    create
      [
        ("city", Col.string_list [ "Paris"; "Paris"; "Lyon"; "Lyon"; "Nice" ]);
        ("sales", Col.float64_list [ 1200.; 800.; 450.; 900.; 500. ]);
        ("units", Col.int32_list [ 10l; 8l; 5l; 9l; 6l ]);
      ]
  in

  Printf.printf "== original ==\n";
  print df;

  Printf.printf "\n== sorted by sales desc ==\n";
  let df_sorted = sort_values ~ascending:false df "sales" in
  print df_sorted;

  Printf.printf "\n== group by city, show sums ==\n";
  let groups = group_by_column df "city" in
  List.iter
    (fun (key_col, sub) ->
      (* key_col is the "city" value as a Col.t with one element *)
      let city_name =
        match key_col with Col.S [| Some s |] -> s | _ -> "<unknown>"
      in
      let total_sales = Agg.Float.sum sub "sales" in
      let total_units = Agg.Int.sum sub "units" in
      Printf.printf "- %s: sales=%.0f units=%Ld\n" city_name total_sales
        total_units)
    groups;
  ()
