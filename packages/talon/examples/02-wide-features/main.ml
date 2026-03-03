(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let () =
  (* A "wide" frame: 5 rows x 8 numeric features *)
  let df =
    create
      [
        ("id", Col.string [| "u1"; "u2"; "u3"; "u4"; "u5" |]);
        ("feat_1", Col.float64 [| 1.; 4.; 2.; 3.; 1. |]);
        ("feat_2", Col.float64 [| 0.; 1.; 1.; 1.; 2. |]);
        ("feat_3", Col.float64 [| 3.; 0.; 1.; 2.; 0. |]);
        ("feat_4", Col.float64 [| 5.; 2.; 0.; 1.; 3. |]);
        ("feat_5", Col.float64 [| 2.; 2.; 2.; 2.; 2. |]);
        ("feat_6", Col.float64 [| 1.; 0.; 1.; 0.; 1. |]);
        ("feat_7", Col.float64 [| 0.5; 0.2; 0.1; 0.3; 0.4 |]);
        ("feat_8", Col.float64 [| 10.; 9.; 7.; 13.; 8. |]);
      ]
  in

  (* Select all feature columns by prefix *)
  let feats =
    List.filter
      (fun n -> String.starts_with ~prefix:"feat_" n)
      (column_names df)
  in

  (* Row-wise sum across many columns (vectorized) *)
  let df = add_column df "row_sum" (Agg.row_sum ~skipna:true df ~names:feats) in

  (* Weighted score (dot product) *)
  let weights = [| 0.1; 0.1; 0.1; 0.1; 0.1; 0.05; 0.05; 0.4 |] in
  let df = add_column df "score" (Agg.dot df ~names:feats ~weights) in

  (* Sort by score descending *)
  let df = sort_values ~ascending:false df "score" in

  print ~max_rows:10 df
