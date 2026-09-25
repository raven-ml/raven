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

let add_edge_cases =
  [
    test "broadcast error"
      (test_broadcast_error ~op:Nx.add ~op_name:"add" ~dtype:Nx.float32
         ~a_shape:[| 3 |] ~b_shape:[| 4 |]);
    test "nan propagation" (test_nan_propagation ~op:Nx.add ~op_name:"add");
    test "inf arithmetic" (fun () ->
        let a =
          Nx.create Nx.float32 [| 2 |] [| Float.infinity; Float.neg_infinity |]
        in
        let b = Nx.create Nx.float32 [| 2 |] [| 5.0; 10.0 |] in
        let result = Nx.add a b in
        equal ~msg:"inf + 5" (float 1e-6) Float.infinity (Nx.item [ 0 ] result);
        equal ~msg:"-inf + 10" (float 1e-6) Float.neg_infinity
          (Nx.item [ 1 ] result));
    test "inf + inf" (fun () ->
        let a =
          Nx.create Nx.float32 [| 2 |] [| Float.infinity; Float.infinity |]
        in
        let b =
          Nx.create Nx.float32 [| 2 |] [| Float.infinity; Float.neg_infinity |]
        in
        let result = Nx.add a b in
        equal ~msg:"inf + inf" (float 1e-6) Float.infinity
          (Nx.item [ 0 ] result);
        equal ~msg:"inf + -inf" bool true (Float.is_nan (Nx.item [ 1 ] result)));
  ]

(* ───── Sub Edge Cases ───── *)

let sub_edge_cases =
  [
    test "inf - inf" (fun () ->
        let a =
          Nx.create Nx.float32 [| 2 |] [| Float.infinity; Float.infinity |]
        in
        let b =
          Nx.create Nx.float32 [| 2 |] [| Float.infinity; Float.neg_infinity |]
        in
        let result = Nx.sub a b in
        equal ~msg:"inf - inf" bool true (Float.is_nan (Nx.item [ 0 ] result));
        equal ~msg:"inf - -inf" (float 1e-6) Float.infinity
          (Nx.item [ 1 ] result));
  ]

(* ───── Div Edge Cases ───── *)

let div_edge_cases =
  [
    test "div by zero float" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.0; -1.0; 0.0 |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 0.0; 0.0; 0.0 |] in
        let result = Nx.div a b in
        equal ~msg:"1/0" (float 1e-6) Float.infinity (Nx.item [ 0 ] result);
        equal ~msg:"-1/0" (float 1e-6) Float.neg_infinity (Nx.item [ 1 ] result);
        equal ~msg:"0/0" bool true (Float.is_nan (Nx.item [ 2 ] result)));
  ]

(* ───── Pow Edge Cases ───── *)

let pow_edge_cases =
  [
    test "zero^zero" (fun () ->
        let a = Nx.create Nx.float32 [| 1 |] [| 0.0 |] in
        let b = Nx.create Nx.float32 [| 1 |] [| 0.0 |] in
        let result = Nx.pow a b in
        equal ~msg:"0^0" (float 1e-6) 1.0 (Nx.item [ 0 ] result));
    test "negative base fractional exp" (fun () ->
        let a = Nx.create Nx.float32 [| 1 |] [| -2.0 |] in
        let b = Nx.create Nx.float32 [| 1 |] [| 0.5 |] in
        let result = Nx.pow a b in
        equal ~msg:"(-2)^0.5" bool true (Float.is_nan (Nx.item [ 0 ] result)));
    test "pow overflow" (fun () ->
        let a = Nx.create Nx.float32 [| 1 |] [| 10.0 |] in
        let b = Nx.create Nx.float32 [| 1 |] [| 100.0 |] in
        let result = Nx.pow a b in
        equal ~msg:"10^100" (float 1e-6) Float.infinity (Nx.item [ 0 ] result));
  ]

(* ───── Math Function Edge Cases ───── *)

let math_edge_cases =
  [
    test "exp overflow" (fun () ->
        let t = Nx.create Nx.float32 [| 1 |] [| 1000.0 |] in
        let result = Nx.exp t in
        equal ~msg:"exp(1000)" (float 1e-6) Float.infinity
          (Nx.item [ 0 ] result));
    test "exp underflow" (fun () ->
        let t = Nx.create Nx.float32 [| 1 |] [| -1000.0 |] in
        let result = Nx.exp t in
        equal ~msg:"exp(-1000)" (float 1e-6) 0.0 (Nx.item [ 0 ] result));
    test "log negative" (fun () ->
        let t = Nx.create Nx.float32 [| 1 |] [| -1.0 |] in
        let result = Nx.log t in
        equal ~msg:"log(-1)" bool true (Float.is_nan (Nx.item [ 0 ] result)));
    test "log zero" (fun () ->
        let t = Nx.create Nx.float32 [| 1 |] [| 0.0 |] in
        let result = Nx.log t in
        equal ~msg:"log(0)" (float 1e-6) Float.neg_infinity
          (Nx.item [ 0 ] result));
    test "sqrt negative" (fun () ->
        let t = Nx.create Nx.float32 [| 1 |] [| -1.0 |] in
        let result = Nx.sqrt t in
        equal ~msg:"sqrt(-1)" bool true (Float.is_nan (Nx.item [ 0 ] result)));
    test "asin out of domain" (fun () ->
        let t = Nx.create Nx.float32 [| 1 |] [| 2.0 |] in
        let result = Nx.asin t in
        equal ~msg:"asin(2)" bool true (Float.is_nan (Nx.item [ 0 ] result)));
  ]

(* ───── Comparison Edge Cases ───── *)

let comparison_edge_cases =
  [
    test "nan comparisons" (fun () ->
        let t1 = Nx.create Nx.float32 [| 3 |] [| Float.nan; 1.; Float.nan |] in
        let t2 = Nx.create Nx.float32 [| 3 |] [| Float.nan; Float.nan; 1. |] in
        let eq_result = Nx.equal t1 t2 in
        let ne_result = Nx.not_equal t1 t2 in
        check_t "nan equal" [| 3 |] [| false; false; false |] eq_result;
        check_t "nan not_equal" [| 3 |] [| true; true; true |] ne_result);
  ]

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

let rounding_edge_cases =
  [
    test "clip" (fun () ->
        let t = Nx.create Nx.float32 [| 5 |] [| -1.; 2.; 5.; 8.; 10. |] in
        let result = Nx.clamp ~min:0. ~max:7. t in
        check_t "clip" [| 5 |] [| 0.; 2.; 5.; 7.; 7. |] result);
  ]

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

let bitwise_edge_cases =
  [
    test "invert"
      (test_unary_op ~op:Nx.bitwise_not ~op_name:"invert" ~dtype:Nx.int32
         ~shape:[| 3 |] ~input:[| 5l; 0l; 7l |] ~expected:[| -6l; -1l; -8l |]);
  ]

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
let erfinv_tests =
  [
    test "reference values at float64" (fun () ->
        List.iter
          (fun (x, expected) ->
            equal
              ~msg:(Printf.sprintf "erfinv %g" x)
              (float (1e-15 *. Float.abs expected))
              expected
              (Nx.item [] (Nx.erfinv (Nx.scalar Nx.float64 x))))
          [
            (0.5, 0.47693627620446987);
            (-0.3, -0.27246271472675435);
            (0.9, 1.1630871536766742);
            (0.999999, 3.4589107372754988);
            (0.99999999, 4.0522372432687634);
            (1.0 -. 1e-13, 5.2614833313726763);
            (-.(1.0 -. 1e-16), -5.8635847487551679);
          ]);
    test "erf round trip at float64" (fun () ->
        let x = Nx.linspace Nx.float64 (-0.999) 0.999 1999 in
        let back = Nx.erf (Nx.erfinv x) in
        let worst = Nx.item [] (Nx.max (Nx.abs (Nx.sub back x))) in
        equal ~msg:"max |erf (erfinv x) - x|" (float 1e-14) 0.0 worst);
    test "float32 keeps seven digits" (fun () ->
        let x = Nx.linspace Nx.float32 (-0.999) 0.999 999 in
        let back = Nx.erf (Nx.erfinv x) in
        let worst = Nx.item [] (Nx.max (Nx.abs (Nx.sub back x))) in
        equal ~msg:"max |erf (erfinv x) - x|" (float 2e-6) 0.0 worst);
    test "endpoints and outside" (fun () ->
        let v =
          Nx.to_array
            (Nx.erfinv (Nx.create Nx.float64 [| 4 |] [| 1.0; -1.0; 1.5; 0.0 |]))
        in
        equal ~msg:"erfinv 1" float_exact Float.infinity v.(0);
        equal ~msg:"erfinv -1" float_exact Float.neg_infinity v.(1);
        equal ~msg:"erfinv 1.5" bool true (Float.is_nan v.(2));
        equal ~msg:"erfinv 0" float_exact 0.0 v.(3));
  ]

(* Test Suite Organization *)

let suite =
  [
    group "Add Edge Cases" add_edge_cases;
    group "Sub Edge Cases" sub_edge_cases;
    group "Div Edge Cases" div_edge_cases;
    group "Pow Edge Cases" pow_edge_cases;
    group "Math Edge Cases" math_edge_cases;
    group "Comparison Edge Cases" comparison_edge_cases;
    group "Reduction Edge Cases" reduction_edge_cases;
    group "Rounding Edge Cases" rounding_edge_cases;
    group "Cumulative" cumulative_tests;
    group "Bitwise Edge Cases" bitwise_edge_cases;
    group "Log" log_tests;
    group "Standardize" standardize_tests;
    group "Erfinv" erfinv_tests;
  ]

let () = run "Nx Ops" suite
