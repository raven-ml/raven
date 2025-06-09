(* Test numerical stability of log_sigmoid *)
open Nx

let test_log_sigmoid_stability () =
  (* Test values that can cause numerical instability *)
  let test_values = [| -100.; -50.; -20.; -10.; 0.; 10.; 20.; 50.; 100.; 1000.; -1000. |] in
  
  Printf.printf "Testing log_sigmoid numerical stability:\n";
  
  let has_nan_or_inf = ref false in
  
  Array.iter (fun x ->
    let input = scalar float32 x in
    let result = log_sigmoid input in
    let y = get_item [] result in
    
    Printf.printf "  log_sigmoid(%.1f) = %.6f" x y;
    
    if Float.is_nan y then (
      Printf.printf " (NaN!)\n";
      has_nan_or_inf := true
    ) else if Float.is_infinite y then (
      Printf.printf " (Inf!)\n"; 
      has_nan_or_inf := true
    ) else
      Printf.printf "\n"
  ) test_values;
  
  if !has_nan_or_inf then (
    Printf.printf "\nFAIL: log_sigmoid produces NaN or Inf values!\n";
    exit 1
  ) else
    Printf.printf "\nPASS: log_sigmoid is numerically stable\n"

let () = test_log_sigmoid_stability ()