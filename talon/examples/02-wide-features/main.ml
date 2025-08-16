open Talon

let () =
  (* A “wide” frame: 5 rows x 8 numeric features *)
  let df =
    create
      [
        ("id", Col.string_list [ "u1"; "u2"; "u3"; "u4"; "u5" ]);
        ("feat_1", Col.float64_list [ 1.; 4.; 2.; 3.; 1. ]);
        ("feat_2", Col.float64_list [ 0.; 1.; 1.; 1.; 2. ]);
        ("feat_3", Col.float64_list [ 3.; 0.; 1.; 2.; 0. ]);
        ("feat_4", Col.float64_list [ 5.; 2.; 0.; 1.; 3. ]);
        ("feat_5", Col.float64_list [ 2.; 2.; 2.; 2.; 2. ]);
        ("feat_6", Col.float64_list [ 1.; 0.; 1.; 0.; 1. ]);
        ("feat_7", Col.float64_list [ 0.5; 0.2; 0.1; 0.3; 0.4 ]);
        ("feat_8", Col.float64_list [ 10.; 9.; 7.; 13.; 8. ]);
      ]
  in

  (* Select all feature columns by prefix *)
  let feats = Cols.with_prefix df "feat_" in

  (* Row-wise sum across many columns (vectorized) *)
  let df = add_column df "row_sum" (Row.Agg.sum ~skipna:true df ~names:feats) in

  (* Weighted score (dot product) *)
  let weights = Array.of_list [ 0.1; 0.1; 0.1; 0.1; 0.1; 0.05; 0.05; 0.4 ] in
  let df = add_column df "score" (Row.Agg.dot df ~names:feats ~weights) in

  (* Sort by score descending *)
  let df = sort_values ~ascending:false df "score" in

  print ~max_rows:10 df
