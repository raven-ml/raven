(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Edge case and special-value tests for Nx operations.

   Algebraic properties (commutativity, identity, inverse, etc.) are covered by
   property-based tests in nx/test/props/test_nx_props.ml. This file retains
   tests for NaN/Inf behavior, error conditions, and specific numerical accuracy
   checks. *)

open Windtrap
open Test_nx_support

(* ───── Generic Test Helpers ───── *)

let test_broadcast_error ~op ~op_name ~dtype ~a_shape ~b_shape () =
  let a = Nx.zeros dtype a_shape in
  let b = Nx.zeros dtype b_shape in
  check_invalid_arg
    (Printf.sprintf "%s incompatible broadcast" op_name)
    (Printf.sprintf
       "broadcast: cannot broadcast %s with %s (dim 0: 3\226\137\1604)"
       (Nx_core.Shape.to_string a_shape)
       (Nx_core.Shape.to_string b_shape))
    (fun () -> ignore (op a b))

let test_nan_propagation ~op ~op_name () =
  let a = Nx.create Nx.float32 [| 3 |] [| Float.nan; 1.0; 2.0 |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 5.0; Float.nan; 3.0 |] in
  let result = op a b in
  equal
    ~msg:(Printf.sprintf "%s nan[0]" op_name)
    bool true
    (Float.is_nan (Nx.item [ 0 ] result));
  equal
    ~msg:(Printf.sprintf "%s nan[1]" op_name)
    bool true
    (Float.is_nan (Nx.item [ 1 ] result))

let test_unary_op ~op ~op_name ~dtype ~shape ~input ~expected () =
  let t = Nx.create dtype shape input in
  let result = op t in
  check_t op_name shape expected result

(* ───── Add Edge Cases ───── *)

(* ───── Sub Edge Cases ───── *)

(* ───── Div Edge Cases ───── *)

(* ───── Pow Edge Cases ───── *)

(* ───── Math Function Edge Cases ───── *)

(* ───── Comparison Edge Cases ───── *)

(* ───── Reduction Edge Cases ───── *)

let reduction_edge_cases =
  [
    test "sum axis=1 keepdims" (fun () ->
        let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let result = Nx.sum ~axes:[ 1 ] ~keepdims:true t in
        check_t "sum axis=1 keepdims" [| 2; 1 |] [| 6.; 15. |] result);
    test "empty array mean" (fun () ->
        let _t = Nx.create Nx.float32 [| 0 |] [||] in
        (* Skip for now - mean of empty array behavior needs investigation *)
        ());
    test "min/max with nan" (fun () ->
        let t = Nx.create Nx.float32 [| 3 |] [| 1.; Float.nan; 3. |] in
        let min_result = Nx.min t in
        let max_result = Nx.max t in
        equal ~msg:"min with nan" bool true
          (Float.is_nan (Nx.item [] min_result));
        equal ~msg:"max with nan" bool true
          (Float.is_nan (Nx.item [] max_result)));
    (* Element p of a float sum's run goes to lane p mod 16 strided or not, so a
       column and its contiguous copy sum to the same bits. A strided run summed
       into one lane: -146.946198 against -146.949036. *)
    test "a float sum's bits do not depend on the stride" (fun () ->
        let wide =
          Nx.create Nx.float32 [| 1003; 2 |]
            (Array.init 2006 (fun i -> sin (float_of_int i) *. 1e3))
        in
        let strided = Nx.slice [ Nx.A; Nx.I 0 ] wide in
        let bits t = Int32.bits_of_float (Nx.item [] (Nx.sum t)) in
        equal ~msg:"strided and contiguous" int32
          (bits (Nx.copy strided))
          (bits strided));
  ]

(* ───── Rounding Edge Cases ───── *)

(* ───── Cumulative Tests ───── *)

let cumulative_tests =
  [
    test "cumsum default axis" (fun () ->
        let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let result = Nx.cumsum t in
        check_t ~eps:1e-6 "cumsum flatten" [| 2; 2 |] [| 1.; 3.; 6.; 10. |]
          result);
    test "cumsum axis=1" (fun () ->
        let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let result = Nx.cumsum ~axis:1 t in
        check_t ~eps:1e-6 "cumsum axis=1" [| 2; 3 |]
          [| 1.; 3.; 6.; 4.; 9.; 15. |]
          result);
    test "cumprod axis=-1" (fun () ->
        let t = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l; 2l; 2l; 2l |] in
        let result = Nx.cumprod ~axis:(-1) t in
        check_t "cumprod axis=-1" [| 2; 3 |] [| 1l; 2l; 6l; 2l; 4l; 8l |] result);
    test "cummax nan propagation" (fun () ->
        let t = Nx.create Nx.float32 [| 4 |] [| 1.; Float.nan; 2.; 3. |] in
        let result = Nx.cummax t in
        equal ~msg:"cummax nan[1]" bool true
          (Float.is_nan (Nx.item [ 1 ] result));
        equal ~msg:"cummax nan[2]" bool true
          (Float.is_nan (Nx.item [ 2 ] result)));
    test "cummin axis option" (fun () ->
        let t =
          Nx.create Nx.int32 [| 2; 4 |] [| 4l; 2l; 3l; 1l; 5l; 6l; 2l; 7l |]
        in
        let result = Nx.cummin ~axis:0 t in
        check_t "cummin axis=0" [| 2; 4 |]
          [| 4l; 2l; 3l; 1l; 4l; 2l; 2l; 1l |]
          result);
  ]

(* ───── Bitwise Edge Cases ───── *)

(* ───── Log/Standardize Tests ───── *)

let test_log_softmax_basic () =
  let input = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let result = Nx.log_softmax input in
  let data = [| 1.; 2.; 3. |] in
  let max_x = Array.fold_left Float.max data.(0) data in
  let denom =
    Array.fold_left (fun acc v -> acc +. Float.exp (v -. max_x)) 0. data
  in
  let log_den = Float.log denom in
  let expected = Array.map (fun v -> v -. max_x -. log_den) data in
  check_t ~eps:1e-6 "log_softmax basic" [| 3 |] expected result

let test_log_softmax_with_scale () =
  let input = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 2. |] in
  let scale = 0.5 in
  let result = Nx.log_softmax ~scale input in
  let data = [| 0.; 1.; 2. |] in
  let max_x = Array.fold_left Float.max data.(0) data in
  let denom =
    Array.fold_left
      (fun acc v -> acc +. Float.exp (scale *. (v -. max_x)))
      0. data
  in
  let log_den = Float.log denom in
  let expected = Array.map (fun v -> (scale *. (v -. max_x)) -. log_den) data in
  check_t ~eps:1e-6 "log_softmax scale" [| 3 |] expected result

let test_logsumexp_basic () =
  let input = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let result = Nx.logsumexp input in
  let data = [| 1.; 2.; 3. |] in
  let max_x = Array.fold_left Float.max data.(0) data in
  let denom =
    Array.fold_left (fun acc v -> acc +. Float.exp (v -. max_x)) 0. data
  in
  let expected = Float.log denom +. max_x in
  check_t ~eps:1e-6 "logsumexp basic" [||] [| expected |] result

let test_logsumexp_axis () =
  let input = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let result = Nx.logsumexp ~axes:[ 1 ] input in
  let rows = [| [| 1.; 2. |]; [| 3.; 4. |] |] in
  let expected =
    Array.map
      (fun row ->
        let max_x = Array.fold_left Float.max row.(0) row in
        let denom =
          Array.fold_left (fun acc v -> acc +. Float.exp (v -. max_x)) 0. row
        in
        Float.log denom +. max_x)
      rows
  in
  check_t ~eps:1e-6 "logsumexp axis" [| 2 |] expected result

let test_logmeanexp_basic () =
  let input = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let result = Nx.logmeanexp input in
  let data = [| 1.; 2.; 3. |] in
  let max_x = Array.fold_left Float.max data.(0) data in
  let denom =
    Array.fold_left (fun acc v -> acc +. Float.exp (v -. max_x)) 0. data
  in
  let log_sum = Float.log denom +. max_x in
  let expected = log_sum -. Float.log (float_of_int (Array.length data)) in
  check_t ~eps:1e-6 "logmeanexp basic" [||] [| expected |] result

let test_logmeanexp_axis () =
  let input = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let result = Nx.logmeanexp ~axes:[ 1 ] input in
  let rows = [| [| 1.; 2. |]; [| 3.; 4. |] |] in
  let expected =
    Array.map
      (fun row ->
        let max_x = Array.fold_left Float.max row.(0) row in
        let denom =
          Array.fold_left (fun acc v -> acc +. Float.exp (v -. max_x)) 0. row
        in
        let log_sum = Float.log denom +. max_x in
        log_sum -. Float.log (float_of_int (Array.length row)))
      rows
  in
  check_t ~eps:1e-6 "logmeanexp axis" [| 2 |] expected result

let test_standardize_global () =
  let input = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let standardized = Nx.standardize input in
  let mean = Nx.item [] (Nx.mean standardized) in
  let variance = Nx.item [] (Nx.var standardized) in
  equal ~msg:"standardize mean ~ 0" (float 1e-5) 0. mean;
  equal ~msg:"standardize var ~ 1" (float 1e-4) 1. variance

let test_standardize_axes_with_params () =
  let input = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let axes = [ 0 ] in
  let mean = Nx.mean ~axes ~keepdims:true input in
  let variance = Nx.var ~axes ~keepdims:true input in
  let expected =
    let eps = 1e-5 in
    let denom = Nx.sqrt (Nx.add variance (Nx.scalar Nx.float32 eps)) in
    Nx.div (Nx.sub input mean) denom
  in
  let result = Nx.standardize ~axes ~mean ~variance input in
  check_nx ~epsilon:1e-6 "standardize with params" expected result;
  let auto = Nx.standardize ~axes input in
  check_nx ~epsilon:1e-6 "standardize axes" expected auto

let log_tests =
  [
    test "log_softmax basic" test_log_softmax_basic;
    test "log_softmax scale" test_log_softmax_with_scale;
    test "logsumexp basic" test_logsumexp_basic;
    test "logsumexp axis" test_logsumexp_axis;
    test "logmeanexp basic" test_logmeanexp_basic;
    test "logmeanexp axis" test_logmeanexp_axis;
  ]

let standardize_tests =
  [
    test "standardize global" test_standardize_global;
    test "standardize axes with params" test_standardize_axes_with_params;
  ]

(* [erfinv] at float64 is a seven-digit guess refined by Newton steps, so it is
   held to reference values at double precision, and to [erf] round trip over
   the interval; float32 keeps the guess's seven digits. *)
(* Test Suite Organization *)

let suite =
  [
    group "Reduction Edge Cases" reduction_edge_cases;
    group "Cumulative" cumulative_tests;
    group "Log" log_tests;
    group "Standardize" standardize_tests;
  ]

let () = exit (run "Nx Ops" suite)
