open Talon

let () =
  (* Demonstrate Row.Agg reductions and skipna semantics *)
  let nan = Stdlib.nan in
  let df =
    create
      [
        ("a", Col.float64_list [ 1.; nan; 3.; 4. ]);
        ("b", Col.float64_list [ 0.; 2.; nan; 1. ]);
        (* ints can use sentinels to represent nulls in Talon (see docs). Here
           we keep them valid for simplicity. *)
        ("c", Col.int32_list [ 10l; 20l; 30l; 40l ]);
      ]
  in

  let nums = Cols.numeric df in

  (* Row-wise sum/mean across all numeric columns *)
  let df =
    add_column df "sum_skipna" (Row.Agg.sum ~skipna:true df ~names:nums)
  in
  let df =
    add_column df "mean_skipna" (Row.Agg.mean ~skipna:true df ~names:nums)
  in

  (* Strict variant (NaN participates) *)
  let df =
    add_column df "sum_strict" (Row.Agg.sum ~skipna:false df ~names:nums)
  in
  let df =
    add_column df "mean_strict" (Row.Agg.mean ~skipna:false df ~names:nums)
  in

  print ~max_rows:10 df
