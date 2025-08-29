(* Test if gather respects view offsets *)
open Nx

let () =
  Printf.printf "Testing gather on views with offsets\n";

  (* Create a 3x4x5 tensor with values 0-59 *)
  let t = create float32 [| 3; 4; 5 |] (Array.init 60 float_of_int) in

  (* Method 1: Direct indexing - should get row 1, columns 0 and 2 *)
  let direct = slice [ Nx.I 1; Nx.L [ 0; 2 ]; Nx.A ] t in
  Printf.printf "Direct indexing shape: [%s]\n"
    (String.concat "; "
       (Array.to_list (Array.map string_of_int (shape direct))));

  (* Method 2: Step-by-step with views *)
  (* First select row 1 *)
  let row1 = slice [ Nx.I 1; Nx.A; Nx.A ] t in
  (* Shape [4,5] *)
  Printf.printf "Row 1 shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (shape row1))));

  (* First, let's verify row1 has the correct data *)
  let row1_first_5 = Array.init 5 (fun i -> item [ 0; i ] row1) in
  Printf.printf "Row 1, first column values: [%s] (should be 20-24)\n"
    (String.concat "; "
       (Array.to_list (Array.map string_of_float row1_first_5)));

  (* Then select columns 0 and 2 from row 1 *)
  let step_by_step = slice [ Nx.L [ 0; 2 ]; Nx.A ] row1 in
  Printf.printf "Step-by-step shape: [%s]\n"
    (String.concat "; "
       (Array.to_list (Array.map string_of_int (shape step_by_step))));

  (* Compare values *)
  let get_values tensor =
    Array.init (numel tensor) (fun i -> item [ i ] (flatten tensor))
  in

  let direct_vals = get_values direct in
  let step_vals = get_values step_by_step in

  Printf.printf "Direct values: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_float direct_vals)));
  Printf.printf "Step-by-step values: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_float step_vals)));

  (* They should be equal *)
  let all_equal = Array.for_all2 ( = ) direct_vals step_vals in

  if all_equal then
    Printf.printf "SUCCESS: Both methods produce the same result\n"
  else (
    Printf.printf "FAILURE: Methods produce different results\n";
    Printf.printf "This indicates gather doesn't properly handle view offsets\n")
