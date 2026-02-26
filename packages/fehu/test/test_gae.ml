open Fehu
open Windtrap

let f = float 1e-6

(* Compute *)

let test_compute_simple () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let values = [| 0.5; 0.5; 0.5 |] in
  let terminated = [| false; false; false |] in
  let truncated = [| false; false; false |] in
  let next_values = [| 0.5; 0.5; 0.5 |] in
  let advantages, returns =
    Gae.compute ~rewards ~values ~terminated ~truncated ~next_values ~gamma:0.99
      ~lambda:0.95
  in
  equal ~msg:"lengths match" int 3 (Array.length advantages);
  for i = 0 to 2 do
    equal ~msg:"returns = advantages + values" f returns.(i)
      (advantages.(i) +. values.(i))
  done

let test_compute_termination () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let values = [| 0.5; 0.5; 0.5 |] in
  let terminated = [| false; true; false |] in
  let truncated = [| false; false; false |] in
  let next_values = [| 0.5; 0.5; 0.5 |] in
  let advantages, _returns =
    Gae.compute ~rewards ~values ~terminated ~truncated ~next_values ~gamma:0.99
      ~lambda:0.95
  in
  (* At step 1 (terminated), bootstrap is 0: delta = 1.0 + 0.99*0 - 0.5 = 0.5 *)
  equal ~msg:"terminal advantage" f 0.5 advantages.(1)

let test_compute_truncation () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let values = [| 0.5; 0.5; 0.5 |] in
  let terminated = [| false; false; false |] in
  let truncated = [| false; true; false |] in
  let next_values = [| 0.5; 2.0; 0.5 |] in
  let advantages_trunc, _returns =
    Gae.compute ~rewards ~values ~terminated ~truncated ~next_values ~gamma:0.99
      ~lambda:0.95
  in
  (* At step 1 (truncated), bootstrap uses next_values.(1) = 2.0 delta = 1.0 +
     0.99*2.0 - 0.5 = 2.48 *)
  let terminated_fake = [| false; true; false |] in
  let advantages_term, _returns_term =
    Gae.compute ~rewards ~values ~terminated:terminated_fake
      ~truncated:[| false; false; false |] ~next_values ~gamma:0.99 ~lambda:0.95
  in
  (* With termination instead, bootstrap would be 0: delta = 1.0 + 0.99*0 - 0.5
     = 0.5 These must differ because truncation uses next_values. *)
  not_equal ~msg:"truncation differs from termination" f advantages_trunc.(1)
    advantages_term.(1)

let test_compute_length_mismatch () =
  raises_invalid_arg "Gae: all arrays must have the same length" (fun () ->
      Gae.compute ~rewards:[| 1.0; 1.0 |] ~values:[| 0.5 |]
        ~terminated:[| false; false |] ~truncated:[| false; false |]
        ~next_values:[| 0.5; 0.5 |] ~gamma:0.99 ~lambda:0.95)

let test_compute_empty () =
  let advantages, returns =
    Gae.compute ~rewards:[||] ~values:[||] ~terminated:[||] ~truncated:[||]
      ~next_values:[||] ~gamma:0.99 ~lambda:0.95
  in
  equal ~msg:"empty advantages" int 0 (Array.length advantages);
  equal ~msg:"empty returns" int 0 (Array.length returns)

(* Returns *)

let test_returns_simple () =
  let ret =
    Gae.returns ~rewards:[| 1.0; 1.0; 1.0 |]
      ~terminated:[| false; false; false |] ~truncated:[| false; false; false |]
      ~gamma:1.0
  in
  equal ~msg:"ret[0]" f 3.0 ret.(0);
  equal ~msg:"ret[1]" f 2.0 ret.(1);
  equal ~msg:"ret[2]" f 1.0 ret.(2)

let test_returns_gamma_zero () =
  let ret =
    Gae.returns ~rewards:[| 1.0; 2.0; 3.0 |]
      ~terminated:[| false; false; false |] ~truncated:[| false; false; false |]
      ~gamma:0.0
  in
  equal ~msg:"ret[0]" f 1.0 ret.(0);
  equal ~msg:"ret[1]" f 2.0 ret.(1);
  equal ~msg:"ret[2]" f 3.0 ret.(2)

let test_returns_terminated () =
  let ret =
    Gae.returns ~rewards:[| 1.0; 1.0; 1.0 |]
      ~terminated:[| false; true; false |] ~truncated:[| false; false; false |]
      ~gamma:1.0
  in
  (* Step 2: acc = 1.0 Step 1: terminated, so acc = 1.0 + 1.0*0.0*1.0 = 1.0 Step
     0: acc = 1.0 + 1.0*1.0*1.0 = 2.0 *)
  equal ~msg:"ret[0]" f 2.0 ret.(0);
  equal ~msg:"ret[1]" f 1.0 ret.(1);
  equal ~msg:"ret[2]" f 1.0 ret.(2)

let test_returns_truncated () =
  let ret =
    Gae.returns ~rewards:[| 1.0; 1.0; 1.0 |]
      ~terminated:[| false; false; false |] ~truncated:[| false; true; false |]
      ~gamma:1.0
  in
  (* Truncation at step 1 resets accumulation, same as terminated *)
  equal ~msg:"ret[0]" f 2.0 ret.(0);
  equal ~msg:"ret[1]" f 1.0 ret.(1);
  equal ~msg:"ret[2]" f 1.0 ret.(2)

let test_returns_length_mismatch () =
  raises_invalid_arg
    "Gae.returns: rewards, terminated, and truncated must have the same length"
    (fun () ->
      Gae.returns ~rewards:[| 1.0; 1.0 |] ~terminated:[| false |]
        ~truncated:[| false; false |] ~gamma:0.99)

(* Compute from values *)

let test_compute_from_values_simple () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let values = [| 0.5; 0.5; 0.5 |] in
  let terminated = [| false; false; false |] in
  let truncated = [| false; false; false |] in
  let last_value = 0.5 in
  let advantages, returns =
    Gae.compute_from_values ~rewards ~values ~terminated ~truncated ~last_value
      ~gamma:0.99 ~lambda:0.95
  in
  (* next_values should be [| 0.5; 0.5; 0.5 |] (values shifted + last_value) *)
  let advantages2, returns2 =
    Gae.compute ~rewards ~values ~terminated ~truncated
      ~next_values:[| 0.5; 0.5; 0.5 |] ~gamma:0.99 ~lambda:0.95
  in
  for i = 0 to 2 do
    equal ~msg:"advantages match" f advantages2.(i) advantages.(i);
    equal ~msg:"returns match" f returns2.(i) returns.(i)
  done

let test_compute_from_values_shifted () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let values = [| 1.0; 2.0; 3.0 |] in
  let terminated = [| false; false; false |] in
  let truncated = [| false; false; false |] in
  let last_value = 4.0 in
  let advantages, _returns =
    Gae.compute_from_values ~rewards ~values ~terminated ~truncated ~last_value
      ~gamma:0.99 ~lambda:0.95
  in
  (* next_values = [| 2.0; 3.0; 4.0 |] *)
  let advantages2, _returns2 =
    Gae.compute ~rewards ~values ~terminated ~truncated
      ~next_values:[| 2.0; 3.0; 4.0 |] ~gamma:0.99 ~lambda:0.95
  in
  for i = 0 to 2 do
    equal ~msg:"advantages match" f advantages2.(i) advantages.(i)
  done

(* Normalize *)

let test_normalize_mean_zero () =
  let arr = [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let normed = Gae.normalize arr in
  let mean = ref 0.0 in
  Array.iter (fun x -> mean := !mean +. x) normed;
  mean := !mean /. Float.of_int (Array.length normed);
  equal ~msg:"mean near 0" f 0.0 !mean

let test_normalize_std_one () =
  let arr = [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let normed = Gae.normalize arr in
  let n = Array.length normed in
  let mean = ref 0.0 in
  Array.iter (fun x -> mean := !mean +. x) normed;
  mean := !mean /. Float.of_int n;
  let var = ref 0.0 in
  Array.iter
    (fun x ->
      let d = x -. !mean in
      var := !var +. (d *. d))
    normed;
  var := !var /. Float.of_int n;
  let std = sqrt !var in
  is_true ~msg:"std near 1" (Float.abs (std -. 1.0) < 0.01)

let test_normalize_empty () =
  let normed = Gae.normalize [||] in
  equal ~msg:"empty" int 0 (Array.length normed)

let test_normalize_single () =
  let normed = Gae.normalize [| 42.0 |] in
  equal ~msg:"single normalizes to 0" f 0.0 normed.(0)

let () =
  run "Fehu.Gae"
    [
      group "compute"
        [
          test "simple" test_compute_simple;
          test "termination" test_compute_termination;
          test "truncation" test_compute_truncation;
          test "length mismatch" test_compute_length_mismatch;
          test "empty" test_compute_empty;
        ];
      group "returns"
        [
          test "simple" test_returns_simple;
          test "gamma zero" test_returns_gamma_zero;
          test "terminated resets" test_returns_terminated;
          test "truncated resets" test_returns_truncated;
          test "length mismatch" test_returns_length_mismatch;
        ];
      group "compute_from_values"
        [
          test "matches compute" test_compute_from_values_simple;
          test "shifted values" test_compute_from_values_shifted;
        ];
      group "normalize"
        [
          test "mean near zero" test_normalize_mean_zero;
          test "std near one" test_normalize_std_one;
          test "empty" test_normalize_empty;
          test "single element" test_normalize_single;
        ];
    ]
