(** Summarize data with reductions — means, variances, and aggregations along
    any axis.

    Analyze daily temperature readings across four cities. Compute averages,
    find extremes, track running totals, and flag outliers. *)

open Nx
open Nx.Infix

let () =
  (* Daily temperatures (°C) for 4 cities over 7 days. Rows = cities, columns =
     days. *)
  let temps =
    create float64 [| 4; 7 |]
      [|
        22.0;
        24.0;
        19.0;
        25.0;
        23.0;
        21.0;
        26.0;
        (* Paris *)
        30.0;
        32.0;
        35.0;
        31.0;
        29.0;
        33.0;
        34.0;
        (* Cairo *)
        -5.0;
        -8.0;
        -3.0;
        -10.0;
        -2.0;
        -7.0;
        -4.0;
        (* Helsinki *)
        15.0;
        14.0;
        16.0;
        13.0;
        17.0;
        15.0;
        14.0;
        (* London *)
      |]
  in
  let cities = [| "Paris"; "Cairo"; "Helsinki"; "London" |] in
  Printf.printf "Daily temperatures (4 cities × 7 days):\n%s\n\n"
    (data_to_string temps);

  (* --- Per-city statistics (reduce along axis 1 = across days) --- *)
  let city_means = mean ~axes:[ 1 ] temps in
  let city_stds = std ~axes:[ 1 ] temps in
  Printf.printf "City averages:\n";
  for i = 0 to 3 do
    Printf.printf "  %-10s  mean=%.1f  std=%.1f\n" cities.(i)
      (item [ i ] city_means) (item [ i ] city_stds)
  done;
  print_newline ();

  (* --- Hottest day per city (argmax along axis 1) --- *)
  let hottest_day = argmax ~axis:1 temps in
  Printf.printf "Hottest day per city:\n";
  for i = 0 to 3 do
    Printf.printf "  %-10s  day %ld\n" cities.(i) (item [ i ] hottest_day)
  done;
  print_newline ();

  (* --- Global extremes --- *)
  Printf.printf "Warmest reading: %.1f°C\n" (item [] (max temps));
  Printf.printf "Coldest reading: %.1f°C\n\n" (item [] (min temps));

  (* --- Cumulative sum: running total of Cairo's temperatures --- *)
  let cairo = temps.${[ I 1; A ]} in
  let cumulative = cumsum ~axis:0 cairo in
  Printf.printf "Cairo daily:      %s\n" (data_to_string cairo);
  Printf.printf "Cairo cumulative: %s\n\n" (data_to_string cumulative);

  (* --- Outlier detection with z-scores --- *)
  let mu = mean ~axes:[ 1 ] ~keepdims:true temps in
  let sigma = std ~axes:[ 1 ] ~keepdims:true temps in
  let z_scores = (temps - mu) / sigma in
  let outlier_mask = greater_s (abs z_scores) 1.5 in
  Printf.printf "Z-scores:\n%s\n" (data_to_string z_scores);
  Printf.printf "Outliers (|z| > 1.5): %s\n\n" (data_to_string outlier_mask);

  (* --- Check if all/any values meet a condition --- *)
  let all_above_zero = all (greater_s temps 0.0) in
  let any_below_neg5 = any (less_s temps (-5.0)) in
  Printf.printf "All temps > 0?   %b\n" (item [] all_above_zero);
  Printf.printf "Any temp < -5?   %b\n" (item [] any_below_neg5)
