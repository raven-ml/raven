open Nx

(* Statistical Functions and Reductions *)

let print_separator () = Printf.printf "\n%s\n\n" (String.make 50 '-')

let () =
  (* Create a sample 2D array for demonstrations *)
  let data =
    init float64 [| 4; 5 |] (fun idx ->
        float_of_int ((idx.(0) * 5) + idx.(1) + 1))
  in

  Printf.printf "Data array (4x5):\n%s\n" (to_string data);
  print_separator ();

  (* 1. Basic statistics on the entire array *)
  Printf.printf "1. Basic Statistics (Entire Array)\n";

  let sum_all = sum data in
  Printf.printf "Sum of all elements: %s\n" (to_string sum_all);

  let mean_all = mean data in
  Printf.printf "Mean of all elements: %s\n" (to_string mean_all);

  let min_all = min data in
  Printf.printf "Minimum value: %s\n" (to_string min_all);

  let max_all = max data in
  Printf.printf "Maximum value: %s\n" (to_string max_all);

  let prod_all = prod data in
  Printf.printf "Product of all elements: %s\n" (to_string prod_all);

  print_separator ();

  (* 2. Statistics along specific axes *)
  Printf.printf "2. Statistics Along Axes\n";

  (* Along axis 0 (column-wise) *)
  let sum_cols = sum ~axes:[| 0 |] data in
  Printf.printf "Sum along columns (axis 0):\n%s\n" (to_string sum_cols);

  let mean_cols = mean ~axes:[| 0 |] data in
  Printf.printf "Mean along columns (axis 0):\n%s\n" (to_string mean_cols);

  let min_cols = min ~axes:[| 0 |] data in
  Printf.printf "Min along columns (axis 0):\n%s\n" (to_string min_cols);

  let max_cols = max ~axes:[| 0 |] data in
  Printf.printf "Max along columns (axis 0):\n%s\n" (to_string max_cols);

  (* Along axis 1 (row-wise) *)
  let sum_rows = sum ~axes:[| 1 |] data in
  Printf.printf "Sum along rows (axis 1):\n%s\n" (to_string sum_rows);

  let mean_rows = mean ~axes:[| 1 |] data in
  Printf.printf "Mean along rows (axis 1):\n%s\n" (to_string mean_rows);

  print_separator ();

  (* 3. Variance and standard deviation *)
  Printf.printf "3. Variance and Standard Deviation\n";

  let var_all = var data in
  Printf.printf "Variance of all elements: %s\n" (to_string var_all);

  let std_all = std data in
  Printf.printf "Standard deviation of all elements: %s\n" (to_string std_all);

  let var_cols = var ~axes:[| 0 |] data in
  Printf.printf "Variance along columns (axis 0):\n%s\n" (to_string var_cols);

  let std_cols = std ~axes:[| 0 |] data in
  Printf.printf "Standard deviation along columns (axis 0):\n%s\n"
    (to_string std_cols);

  print_separator ();

  (* 4. Finding indices of min/max values *)
  Printf.printf "4. Finding Indices of Min/Max Values\n";

  let argmax_all = argmax data in
  Printf.printf "Index of maximum value: %s\n" (to_string argmax_all);

  let argmin_all = argmin data in
  Printf.printf "Index of minimum value: %s\n" (to_string argmin_all);

  let argmax_cols = argmax ~axis:0 data in
  Printf.printf "Indices of maximum values along columns (axis 0):\n%s\n"
    (to_string argmax_cols);

  let argmin_rows = argmin ~axis:1 data in
  Printf.printf "Indices of minimum values along rows (axis 1):\n%s\n"
    (to_string argmin_rows);

  print_separator ();

  (* 5. Sorting and unique values *)
  Printf.printf "5. Sorting and Unique Values\n";

  (* Create a smaller array with some repeated values *)
  let unsorted =
    create float64 [| 2; 4 |] [| 5.0; 2.0; 8.0; 2.0; 1.0; 5.0; 8.0; 3.0 |]
  in
  Printf.printf "Unsorted array with duplicates:\n%s\n" (to_string unsorted);

  let sorted = sort unsorted in
  Printf.printf "Sorted array (entire array):\n%s\n" (to_string sorted);

  let sorted_axis0 = sort ~axis:0 unsorted in
  Printf.printf "Sorted along axis 0 (columns):\n%s\n" (to_string sorted_axis0);

  let sorted_axis1 = sort ~axis:1 unsorted in
  Printf.printf "Sorted along axis 1 (rows):\n%s\n" (to_string sorted_axis1);

  let unique_values = unique (flatten unsorted) in
  Printf.printf "Unique values:\n%s\n" (to_string unique_values);

  print_separator ();

  (* 6. Working with NaN and Inf values (if supported) *)
  Printf.printf "6. Working with Non-finite Values\n";

  (* Create array with some non-finite values *)
  let special_values =
    create float64 [| 2; 3 |]
      [| 1.0; Float.nan; 3.0; Float.infinity; 5.0; Float.neg_infinity |]
  in
  Printf.printf "Array with NaN and Inf values:\n%s\n"
    (to_string special_values);

  let is_nan = isnan special_values in
  Printf.printf "isnan mask:\n%s\n" (to_string is_nan);

  let is_inf = isinf special_values in
  Printf.printf "isinf mask:\n%s\n" (to_string is_inf);

  let is_finite = isfinite special_values in
  Printf.printf "isfinite mask:\n%s\n" (to_string is_finite);

  print_separator ();

  (* 7. Custom reduction using fold *)
  Printf.printf "7. Custom Reductions with fold\n";

  let small_array = create float64 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Printf.printf "Small array:\n%s\n" (to_string small_array);

  (* Compute geometric mean using fold *)
  let n = size small_array in
  let log_sum = fold (fun acc x -> acc +. Stdlib.log x) 0.0 small_array in
  let geom_mean = Stdlib.exp (log_sum /. float_of_int n) in

  Printf.printf "Geometric mean (using fold): %.4f\n" geom_mean
