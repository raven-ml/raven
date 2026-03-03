(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let () =
  (* Demonstrate row-wise reductions and skipna semantics *)
  let nan = Stdlib.nan in
  let df =
    create
      [
        ("a", Col.float64 [| 1.; nan; 3.; 4. |]);
        ("b", Col.float64 [| 0.; 2.; nan; 1. |]);
        (* ints can use sentinels to represent nulls in Talon (see docs). Here
           we keep them valid for simplicity. *)
        ("c", Col.int32 [| 10l; 20l; 30l; 40l |]);
      ]
  in

  let nums = select_columns df `Numeric in

  (* Row-wise sum/mean across all numeric columns *)
  let df =
    add_column df "sum_skipna" (Agg.row_sum ~skipna:true df ~names:nums)
  in
  let df =
    add_column df "mean_skipna" (Agg.row_mean ~skipna:true df ~names:nums)
  in

  (* Strict variant (NaN participates) *)
  let df =
    add_column df "sum_strict" (Agg.row_sum ~skipna:false df ~names:nums)
  in
  let df =
    add_column df "mean_strict" (Agg.row_mean ~skipna:false df ~names:nums)
  in

  print ~max_rows:10 df
