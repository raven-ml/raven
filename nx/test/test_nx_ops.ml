(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Comprehensive operation tests for Nx following the test plan *)

open Windtrap
open Test_nx_support

(* ───── Generic Test Helpers ───── *)

let test_binary_op ~op ~op_name ~dtype ~shape ~a_data ~b_data ~expected () =
  let a = Nx.create dtype shape a_data in
  let b = Nx.create dtype shape b_data in
  let result = op a b in
  check_t
    (Printf.sprintf "%s %s" op_name (Nx.shape_to_string shape))
    shape expected result

let test_binary_op_float ~eps ~op ~op_name ~dtype ~shape ~a_data ~b_data
    ~expected () =
  let a = Nx.create dtype shape a_data in
  let b = Nx.create dtype shape b_data in
  let result = op a b in
  check_t ~eps
    (Printf.sprintf "%s %s" op_name (Nx.shape_to_string shape))
    shape expected result

let test_binary_op_0d ~op ~op_name ~dtype ~a_val ~b_val ~expected () =
  let a = Nx.scalar dtype a_val in
  let b = Nx.scalar dtype b_val in
  let result = op a b in
  check_t (Printf.sprintf "%s 0d" op_name) [||] [| expected |] result

let test_broadcast ~op ~op_name ~dtype ~a_shape ~a_data ~b_shape ~b_data
    ~result_shape () =
  let a = Nx.create dtype a_shape a_data in
  let b = Nx.create dtype b_shape b_data in
  let result = op a b in
  check_shape
    (Printf.sprintf "%s broadcast %s + %s" op_name
       (Nx.shape_to_string a_shape)
       (Nx.shape_to_string b_shape))
    result_shape result

let test_broadcast_error ~op ~op_name ~dtype ~a_shape ~b_shape () =
  let a = Nx.zeros dtype a_shape in
  let b = Nx.zeros dtype b_shape in
  check_invalid_arg
    (Printf.sprintf "%s incompatible broadcast" op_name)
    (Printf.sprintf
       "broadcast: cannot broadcast %s to %s (dim 0: 3\226\137\1604)\n\
        hint: broadcasting requires dimensions to be either equal or 1"
       (Nx.shape_to_string a_shape)
       (Nx.shape_to_string b_shape))
    (fun () -> ignore (op a b))

let test_nan_propagation ~op ~op_name () =
  let a = Nx.create Nx.float32 [| 3 |] [| Float.nan; 1.0; 2.0 |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 5.0; Float.nan; 3.0 |] in
  let result = op a b in
  equal ~msg:(Printf.sprintf "%s nan[0]" op_name) bool true
    (Float.is_nan (Nx.item [ 0 ] result));
  equal ~msg:(Printf.sprintf "%s nan[1]" op_name) bool true
    (Float.is_nan (Nx.item [ 1 ] result))

let test_inplace ~iop ~op_name ~dtype ~shape ~a_data ~b_data ~expected () =
  let a = Nx.create dtype shape a_data in
  let b = Nx.create dtype shape b_data in
  let result = iop a b in
  check_t (Printf.sprintf "i%s result" op_name) shape expected a;
  equal ~msg:(Printf.sprintf "i%s returns a" op_name) bool true (result == a)

let test_scalar_op ~op_s ~op_name ~dtype ~shape ~data ~scalar ~expected () =
  let a = Nx.create dtype shape data in
  let result = op_s a scalar in
  check_t (Printf.sprintf "%s_s" op_name) shape expected result

let test_unary_op ~op ~op_name ~dtype ~shape ~input ~expected () =
  let t = Nx.create dtype shape input in
  let result = op t in
  check_t op_name shape expected result

let test_unary_op_float ~eps ~op ~op_name ~dtype ~shape ~input ~expected () =
  let t = Nx.create dtype shape input in
  let result = op t in
  check_t ~eps op_name shape expected result

(* ───── Arithmetic Operations Tests ───── *)

module Add_tests = struct
  let op = Nx.add
  let op_name = "add"

  let tests =
    [
      test "0d + 0d"
        (test_binary_op_0d ~op ~op_name ~dtype:Nx.float32 ~a_val:5.0 ~b_val:3.0
           ~expected:8.0);
      test "1d + 1d"
        (test_binary_op ~op ~op_name ~dtype:Nx.float32 ~shape:[| 5 |]
           ~a_data:[| 1.; 2.; 3.; 4.; 5. |] ~b_data:[| 5.; 4.; 3.; 2.; 1. |]
           ~expected:[| 6.; 6.; 6.; 6.; 6. |]);
      test "2d + 2d"
        (test_binary_op ~op ~op_name ~dtype:Nx.float32 ~shape:[| 2; 2 |]
           ~a_data:[| 1.; 2.; 3.; 4. |] ~b_data:[| 5.; 6.; 7.; 8. |]
           ~expected:[| 6.; 8.; 10.; 12. |]);
      test "5d + 5d"
        (fun () ->
          let shape = [| 2; 3; 4; 5; 6 |] in
          let size = Array.fold_left ( * ) 1 shape in
          test_binary_op ~op ~op_name ~dtype:Nx.float32 ~shape
            ~a_data:(Array.make size 1.0) ~b_data:(Array.make size 2.0)
            ~expected:(Array.make size 3.0) ());
      test "scalar broadcast"
        (fun () ->
          let a = Nx.scalar Nx.float32 5.0 in
          let b = Nx.create Nx.float32 [| 3; 4 |] (Array.make 12 1.0) in
          let result = Nx.add a b in
          check_t "add scalar broadcast" [| 3; 4 |] (Array.make 12 6.0) result);
      test "1d broadcast to 2d"
        (test_broadcast ~op ~op_name ~dtype:Nx.float32 ~a_shape:[| 4 |]
           ~a_data:[| 1.; 2.; 3.; 4. |] ~b_shape:[| 3; 4 |]
           ~b_data:(Array.make 12 10.0) ~result_shape:[| 3; 4 |]);
      test "broadcast error"
        (test_broadcast_error ~op ~op_name ~dtype:Nx.float32 ~a_shape:[| 3 |]
           ~b_shape:[| 4 |]);
      test "nan propagation" (test_nan_propagation ~op ~op_name);
      test "inf arithmetic"
        (fun () ->
          let a =
            Nx.create Nx.float32 [| 2 |]
              [| Float.infinity; Float.neg_infinity |]
          in
          let b = Nx.create Nx.float32 [| 2 |] [| 5.0; 10.0 |] in
          let result = Nx.add a b in
          equal ~msg:"inf + 5" (float 1e-6) Float.infinity (Nx.item [ 0 ] result);
          equal ~msg:"-inf + 10" (float 1e-6) Float.neg_infinity
            (Nx.item [ 1 ] result));
      test "inf + inf"
        (fun () ->
          let a =
            Nx.create Nx.float32 [| 2 |] [| Float.infinity; Float.infinity |]
          in
          let b =
            Nx.create Nx.float32 [| 2 |]
              [| Float.infinity; Float.neg_infinity |]
          in
          let result = Nx.add a b in
          equal ~msg:"inf + inf" (float 1e-6) Float.infinity (Nx.item [ 0 ] result);
          equal ~msg:"inf + -inf" bool true (Float.is_nan (Nx.item [ 1 ] result)));
      test "iadd"
        (test_inplace ~iop:Nx.iadd ~op_name ~dtype:Nx.float32 ~shape:[| 2; 2 |]
           ~a_data:[| 1.; 2.; 3.; 4. |] ~b_data:[| 5.; 6.; 7.; 8. |]
           ~expected:[| 6.; 8.; 10.; 12. |]);
      test "add_s"
        (test_scalar_op ~op_s:Nx.add_s ~op_name ~dtype:Nx.float32 ~shape:[| 3 |]
           ~data:[| 1.; 2.; 3. |] ~scalar:5.0 ~expected:[| 6.; 7.; 8. |]);
    ]
end

module Sub_tests = struct
  let op = Nx.sub
  let op_name = "sub"

  let tests =
    [
      test "2d - 2d"
        (test_binary_op ~op ~op_name ~dtype:Nx.float32 ~shape:[| 2; 2 |]
           ~a_data:[| 5.; 6.; 7.; 8. |] ~b_data:[| 1.; 2.; 3.; 4. |]
           ~expected:[| 4.; 4.; 4.; 4. |]);
      test "inf - inf"
        (fun () ->
          let a =
            Nx.create Nx.float32 [| 2 |] [| Float.infinity; Float.infinity |]
          in
          let b =
            Nx.create Nx.float32 [| 2 |]
              [| Float.infinity; Float.neg_infinity |]
          in
          let result = Nx.sub a b in
          equal ~msg:"inf - inf" bool true (Float.is_nan (Nx.item [ 0 ] result));
          equal ~msg:"inf - -inf" (float 1e-6) Float.infinity (Nx.item [ 1 ] result));
      test "isub"
        (test_inplace ~iop:Nx.isub ~op_name ~dtype:Nx.float32 ~shape:[| 2; 2 |]
           ~a_data:[| 10.; 11.; 12.; 13. |] ~b_data:[| 1.; 2.; 3.; 4. |]
           ~expected:[| 9.; 9.; 9.; 9. |]);
      test "sub_s"
        (test_scalar_op ~op_s:Nx.sub_s ~op_name ~dtype:Nx.float32 ~shape:[| 3 |]
           ~data:[| 5.; 6.; 7. |] ~scalar:2.0 ~expected:[| 3.; 4.; 5. |]);
    ]
end

module Mul_tests = struct
  let op = Nx.mul
  let op_name = "mul"

  let tests =
    [
      test "2d * 2d"
        (test_binary_op ~op ~op_name ~dtype:Nx.float32 ~shape:[| 2; 2 |]
           ~a_data:[| 1.; 2.; 3.; 4. |] ~b_data:[| 5.; 6.; 7.; 8. |]
           ~expected:[| 5.; 12.; 21.; 32. |]);
      test "imul"
        (test_inplace ~iop:Nx.imul ~op_name ~dtype:Nx.int32 ~shape:[| 3 |]
           ~a_data:[| 1l; 2l; 3l |] ~b_data:[| 5l; 6l; 7l |]
           ~expected:[| 5l; 12l; 21l |]);
      test "mul_s"
        (test_scalar_op ~op_s:Nx.mul_s ~op_name ~dtype:Nx.float32 ~shape:[| 3 |]
           ~data:[| 1.; 2.; 3. |] ~scalar:3.0 ~expected:[| 3.; 6.; 9. |]);
    ]
end

module Div_tests = struct
  let op = Nx.div
  let op_name = "div"

  let tests =
    [
      test "2d / 2d float32"
        (test_binary_op_float ~eps:1e-6 ~op ~op_name ~dtype:Nx.float32
           ~shape:[| 2; 2 |] ~a_data:[| 5.; 6.; 7.; 8. |]
           ~b_data:[| 1.; 2.; 4.; 5. |] ~expected:[| 5.; 3.; 1.75; 1.6 |]);
      test "2d / 2d int32"
        (test_binary_op ~op ~op_name ~dtype:Nx.int32 ~shape:[| 2; 2 |]
           ~a_data:[| 10l; 21l; 30l; 40l |] ~b_data:[| 3l; 5l; 4l; 4l |]
           ~expected:[| 3l; 4l; 7l; 10l |]);
      test "div by zero float"
        (fun () ->
          let a = Nx.create Nx.float32 [| 3 |] [| 1.0; -1.0; 0.0 |] in
          let b = Nx.create Nx.float32 [| 3 |] [| 0.0; 0.0; 0.0 |] in
          let result = Nx.div a b in
          equal ~msg:"1/0" (float 1e-6) Float.infinity (Nx.item [ 0 ] result);
          equal ~msg:"-1/0" (float 1e-6) Float.neg_infinity (Nx.item [ 1 ] result);
          equal ~msg:"0/0" bool true (Float.is_nan (Nx.item [ 2 ] result)));
      test "idiv"
        (test_inplace ~iop:Nx.idiv ~op_name ~dtype:Nx.float32 ~shape:[| 3 |]
           ~a_data:[| 10.; 20.; 30. |] ~b_data:[| 2.; 2.; 2. |]
           ~expected:[| 5.; 10.; 15. |]);
      test "div_s"
        (test_scalar_op ~op_s:Nx.div_s ~op_name ~dtype:Nx.float32 ~shape:[| 3 |]
           ~data:[| 6.; 9.; 12. |] ~scalar:3.0 ~expected:[| 2.; 3.; 4. |]);
    ]
end

module Pow_tests = struct
  let op = Nx.pow
  let op_name = "pow"

  let tests =
    [
      test "basic pow"
        (test_binary_op_float ~eps:1e-5 ~op ~op_name ~dtype:Nx.float32
           ~shape:[| 3 |] ~a_data:[| 2.; 3.; 4. |] ~b_data:[| 3.; 2.; 0.5 |]
           ~expected:[| 8.; 9.; 2. |]);
      test "zero^zero"
        (fun () ->
          let a = Nx.create Nx.float32 [| 1 |] [| 0.0 |] in
          let b = Nx.create Nx.float32 [| 1 |] [| 0.0 |] in
          let result = Nx.pow a b in
          equal ~msg:"0^0" (float 1e-6) 1.0 (Nx.item [ 0 ] result));
      test "negative base fractional exp"
        (fun () ->
          let a = Nx.create Nx.float32 [| 1 |] [| -2.0 |] in
          let b = Nx.create Nx.float32 [| 1 |] [| 0.5 |] in
          let result = Nx.pow a b in
          equal ~msg:"(-2)^0.5" bool true (Float.is_nan (Nx.item [ 0 ] result)));
      test "pow overflow"
        (fun () ->
          let a = Nx.create Nx.float32 [| 1 |] [| 10.0 |] in
          let b = Nx.create Nx.float32 [| 1 |] [| 100.0 |] in
          let result = Nx.pow a b in
          equal ~msg:"10^100" (float 1e-6) Float.infinity (Nx.item [ 0 ] result));
      test "ipow"
        (fun () ->
          let a = Nx.create Nx.float32 [| 3 |] [| 2.; 3.; 4. |] in
          let b = Nx.create Nx.float32 [| 3 |] [| 2.; 1.; 0.5 |] in
          let result = Nx.ipow a b in
          check_t ~eps:1e-5 "ipow result" [| 3 |] [| 4.; 3.; 2. |] a;
          equal ~msg:"ipow returns a" bool true (result == a));
      test "pow_s"
        (fun () ->
          let a = Nx.create Nx.float32 [| 3 |] [| 2.; 3.; 4. |] in
          let result = Nx.pow_s a 2.0 in
          check_t ~eps:1e-5 "pow_s" [| 3 |] [| 4.; 9.; 16. |] result);
    ]
end

module Mod_tests = struct
  let op = Nx.mod_
  let op_name = "mod"

  let tests =
    [
      test "basic mod"
        (test_binary_op ~op ~op_name ~dtype:Nx.int32 ~shape:[| 3 |]
           ~a_data:[| 10l; 11l; 12l |] ~b_data:[| 3l; 5l; 4l |]
           ~expected:[| 1l; 1l; 0l |]);
      test "imod"
        (test_inplace ~iop:Nx.imod ~op_name ~dtype:Nx.int32 ~shape:[| 3 |]
           ~a_data:[| 10l; 11l; 12l |] ~b_data:[| 3l; 5l; 4l |]
           ~expected:[| 1l; 1l; 0l |]);
      test "mod_s"
        (test_scalar_op ~op_s:Nx.mod_s ~op_name ~dtype:Nx.int32 ~shape:[| 3 |]
           ~data:[| 10l; 11l; 12l |] ~scalar:3l ~expected:[| 1l; 2l; 0l |]);
    ]
end

(* ───── Math Function Tests ───── *)

module Math_tests = struct
  let test_exp =
    [
      test "exp basic"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.exp ~op_name:"exp"
           ~dtype:Nx.float32 ~shape:[| 3 |] ~input:[| 0.; 1.; 2. |]
           ~expected:[| 1.; 2.71828; 7.38906 |]);
      test "exp overflow"
        (fun () ->
          let t = Nx.create Nx.float32 [| 1 |] [| 1000.0 |] in
          let result = Nx.exp t in
          equal ~msg:"exp(1000)" (float 1e-6) Float.infinity (Nx.item [ 0 ] result));
      test "exp underflow"
        (fun () ->
          let t = Nx.create Nx.float32 [| 1 |] [| -1000.0 |] in
          let result = Nx.exp t in
          equal ~msg:"exp(-1000)" (float 1e-6) 0.0 (Nx.item [ 0 ] result));
    ]

  let test_log =
    [
      test "log basic"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.log ~op_name:"log"
           ~dtype:Nx.float32 ~shape:[| 3 |] ~input:[| 1.; 2.71828; 7.38905 |]
           ~expected:[| 0.; 1.; 2. |]);
      test "log negative"
        (fun () ->
          let t = Nx.create Nx.float32 [| 1 |] [| -1.0 |] in
          let result = Nx.log t in
          equal ~msg:"log(-1)" bool true (Float.is_nan (Nx.item [ 0 ] result)));
      test "log zero"
        (fun () ->
          let t = Nx.create Nx.float32 [| 1 |] [| 0.0 |] in
          let result = Nx.log t in
          equal ~msg:"log(0)" (float 1e-6) Float.neg_infinity (Nx.item [ 0 ] result));
    ]

  let test_sqrt =
    [
      test "sqrt basic"
        (test_unary_op ~op:Nx.sqrt ~op_name:"sqrt" ~dtype:Nx.float32
           ~shape:[| 3 |] ~input:[| 4.; 9.; 16. |] ~expected:[| 2.; 3.; 4. |]);
      test "sqrt negative"
        (fun () ->
          let t = Nx.create Nx.float32 [| 1 |] [| -1.0 |] in
          let result = Nx.sqrt t in
          equal ~msg:"sqrt(-1)" bool true (Float.is_nan (Nx.item [ 0 ] result)));
    ]

  let test_abs =
    [
      test "abs int32"
        (test_unary_op ~op:Nx.abs ~op_name:"abs" ~dtype:Nx.int32 ~shape:[| 4 |]
           ~input:[| -1l; 0l; 5l; -10l |] ~expected:[| 1l; 0l; 5l; 10l |]);
      test "abs float32"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.abs ~op_name:"abs"
           ~dtype:Nx.float32 ~shape:[| 3 |] ~input:[| -1.5; 0.0; 5.2 |]
           ~expected:[| 1.5; 0.0; 5.2 |]);
    ]

  let test_sign =
    [
      test "sign float32"
        (test_unary_op ~op:Nx.sign ~op_name:"sign" ~dtype:Nx.float32
           ~shape:[| 4 |] ~input:[| -5.0; 0.0; 3.2; -0.0 |]
           ~expected:[| -1.0; 0.0; 1.0; 0.0 |]);
      test "sign int32"
        (test_unary_op ~op:Nx.sign ~op_name:"sign" ~dtype:Nx.int32 ~shape:[| 3 |]
           ~input:[| -5l; 0l; 3l |] ~expected:[| -1l; 0l; 1l |]);
    ]

  let tests =
    test_exp @ test_log @ test_sqrt @ test_abs @ test_sign
    @ [
        test "neg"
          (test_unary_op ~op:Nx.neg ~op_name:"neg" ~dtype:Nx.float32
             ~shape:[| 3 |] ~input:[| 1.0; -2.0; 0.0 |]
             ~expected:[| -1.0; 2.0; 0.0 |]);
        test "square"
          (test_unary_op ~op:Nx.square ~op_name:"square" ~dtype:Nx.float32
             ~shape:[| 4 |] ~input:[| 1.; 2.; -3.; 0. |]
             ~expected:[| 1.; 4.; 9.; 0. |]);
      ]
end

(* ───── Trigonometric Function Tests ───── *)

module Trig_tests = struct
  let pi = Float.pi

  let tests =
    [
      test "sin"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.sin ~op_name:"sin"
           ~dtype:Nx.float32 ~shape:[| 4 |]
           ~input:[| 0.; pi /. 6.; pi /. 4.; pi /. 2. |]
           ~expected:[| 0.; 0.5; 0.707107; 1. |]);
      test "cos"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.cos ~op_name:"cos"
           ~dtype:Nx.float32 ~shape:[| 4 |]
           ~input:[| 0.; pi /. 6.; pi /. 4.; pi /. 2. |]
           ~expected:[| 1.; 0.866025; 0.707107; 0. |]);
      test "tan"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.tan ~op_name:"tan"
           ~dtype:Nx.float32 ~shape:[| 3 |]
           ~input:[| 0.; pi /. 6.; pi /. 4. |]
           ~expected:[| 0.; 0.577350; 1. |]);
      test "asin"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.asin ~op_name:"asin"
           ~dtype:Nx.float32 ~shape:[| 3 |] ~input:[| 0.; 0.5; 1. |]
           ~expected:[| 0.; pi /. 6.; pi /. 2. |]);
      test "asin out of domain"
        (fun () ->
          let t = Nx.create Nx.float32 [| 1 |] [| 2.0 |] in
          let result = Nx.asin t in
          equal ~msg:"asin(2)" bool true (Float.is_nan (Nx.item [ 0 ] result)));
      test "sinh"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.sinh ~op_name:"sinh"
           ~dtype:Nx.float32 ~shape:[| 3 |] ~input:[| -1.; 0.; 1. |]
           ~expected:[| -1.17520; 0.; 1.17520 |]);
      test "cosh"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.cosh ~op_name:"cosh"
           ~dtype:Nx.float32 ~shape:[| 3 |] ~input:[| -1.; 0.; 1. |]
           ~expected:[| 1.54308; 1.; 1.54308 |]);
      test "tanh"
        (test_unary_op_float ~eps:1e-5 ~op:Nx.tanh ~op_name:"tanh"
           ~dtype:Nx.float32 ~shape:[| 3 |] ~input:[| -1.; 0.; 1. |]
           ~expected:[| -0.761594; 0.; 0.761594 |]);
    ]
end

(* ───── Comparison Operation Tests ───── *)

module Comparison_tests = struct
  let test_comparison ~op ~op_name ~input1 ~input2 ~expected () =
    let t1 = Nx.create Nx.float32 [| 3 |] input1 in
    let t2 = Nx.create Nx.float32 [| 3 |] input2 in
    let result = op t1 t2 in
    check_t op_name [| 3 |] expected result

  let tests =
    [
      test "equal"
        (test_comparison ~op:Nx.equal ~op_name:"equal" ~input1:[| 1.; 2.; 3. |]
           ~input2:[| 1.; 5.; 3. |] ~expected:[| true; false; true |]);
      test "not_equal"
        (test_comparison ~op:Nx.not_equal ~op_name:"not_equal"
           ~input1:[| 1.; 2.; 3. |] ~input2:[| 1.; 5.; 3. |]
           ~expected:[| false; true; false |]);
      test "greater"
        (test_comparison ~op:Nx.greater ~op_name:"greater"
           ~input1:[| 1.; 3.; 2. |] ~input2:[| 2.; 1.; 4. |]
           ~expected:[| false; true; false |]);
      test "greater_equal"
        (test_comparison ~op:Nx.greater_equal ~op_name:"greater_equal"
           ~input1:[| 1.; 3.; 4. |] ~input2:[| 2.; 3.; 3. |]
           ~expected:[| false; true; true |]);
      test "less"
        (test_comparison ~op:Nx.less ~op_name:"less" ~input1:[| 1.; 3.; 2. |]
           ~input2:[| 2.; 1.; 4. |] ~expected:[| true; false; true |]);
      test "less_equal"
        (test_comparison ~op:Nx.less_equal ~op_name:"less_equal"
           ~input1:[| 1.; 3.; 4. |] ~input2:[| 2.; 3.; 3. |]
           ~expected:[| true; true; false |]);
      test "nan comparisons"
        (fun () ->
          let t1 =
            Nx.create Nx.float32 [| 3 |] [| Float.nan; 1.; Float.nan |]
          in
          let t2 =
            Nx.create Nx.float32 [| 3 |] [| Float.nan; Float.nan; 1. |]
          in
          let eq_result = Nx.equal t1 t2 in
          let ne_result = Nx.not_equal t1 t2 in
          check_t "nan equal" [| 3 |] [| false; false; false |] eq_result;
          check_t "nan not_equal" [| 3 |] [| true; true; true |] ne_result);
    ]
end

(* ───── Reduction Operation Tests ───── *)

module Reduction_tests = struct
  let test_reduction_all ~op ~op_name ~dtype ~shape ~input ~expected () =
    let t = Nx.create dtype shape input in
    let result = op t in
    check_t (Printf.sprintf "%s all" op_name) [||] [| expected |] result

  let tests =
    [
      test "sum all"
        (test_reduction_all ~op:Nx.sum ~op_name:"sum" ~dtype:Nx.float32
           ~shape:[| 2; 2 |] ~input:[| 1.; 2.; 3.; 4. |] ~expected:10.);
      test "sum axis=0"
        (fun () ->
          let t =
            Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let result = Nx.sum ~axes:[ 0 ] t in
          check_t "sum axis=0" [| 3 |] [| 5.; 7.; 9. |] result);
      test "sum axis=1 keepdims"
        (fun () ->
          let t =
            Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let result = Nx.sum ~axes:[ 1 ] ~keepdims:true t in
          check_t "sum axis=1 keepdims" [| 2; 1 |] [| 6.; 15. |] result);
      test "prod all"
        (fun () ->
          let t = Nx.create Nx.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
          let result = Nx.prod t in
          check_t "prod all" [||] [| 24l |] result);
      test "mean all"
        (fun () ->
          let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
          let result = Nx.mean t in
          check_t "mean all" [||] [| 2.5 |] result);
      test "max all"
        (fun () ->
          let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
          let result = Nx.max t in
          check_t "max all" [||] [| 4. |] result);
      test "min all"
        (fun () ->
          let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
          let result = Nx.min t in
          check_t "min all" [||] [| 1. |] result);
      test "var 1d"
        (fun () ->
          let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
          let result = Nx.var t in
          check_t "var 1d" [||] [| 2. |] result);
      test "std 1d"
        (fun () ->
          let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
          let result = Nx.std t in
          check_t ~eps:1e-5 "std 1d" [||] [| 1.41421 |] result);
      test "empty array mean"
        (fun () ->
          let _t = Nx.create Nx.float32 [| 0 |] [||] in
          (* Skip for now - mean of empty array behavior needs investigation *)
          ());
      test "min/max with nan"
        (fun () ->
          let t = Nx.create Nx.float32 [| 3 |] [| 1.; Float.nan; 3. |] in
          let min_result = Nx.min t in
          let max_result = Nx.max t in
          equal ~msg:"min with nan" bool true (Float.is_nan (Nx.item [] min_result));
          equal ~msg:"max with nan" bool true (Float.is_nan (Nx.item [] max_result)));
    ]
end

(* ───── Min/Max Operation Tests ───── *)

module MinMax_tests = struct
  let tests =
    [
      test "maximum"
        (test_binary_op ~op:Nx.maximum ~op_name:"maximum" ~dtype:Nx.float32
           ~shape:[| 3 |] ~a_data:[| 1.; 3.; 2. |] ~b_data:[| 2.; 1.; 4. |]
           ~expected:[| 2.; 3.; 4. |]);
      test "maximum_s"
        (test_scalar_op ~op_s:Nx.maximum_s ~op_name:"maximum" ~dtype:Nx.float32
           ~shape:[| 3 |] ~data:[| 1.; 5.; 3. |] ~scalar:2.0
           ~expected:[| 2.; 5.; 3. |]);
      test "imaximum"
        (test_inplace ~iop:Nx.imaximum ~op_name:"maximum" ~dtype:Nx.float32
           ~shape:[| 3 |] ~a_data:[| 1.; 3.; 2. |] ~b_data:[| 2.; 1.; 4. |]
           ~expected:[| 2.; 3.; 4. |]);
      test "minimum"
        (test_binary_op ~op:Nx.minimum ~op_name:"minimum" ~dtype:Nx.int32
           ~shape:[| 3 |] ~a_data:[| 1l; 3l; 4l |] ~b_data:[| 2l; 1l; 4l |]
           ~expected:[| 1l; 1l; 4l |]);
      test "minimum_s"
        (test_scalar_op ~op_s:Nx.minimum_s ~op_name:"minimum" ~dtype:Nx.float32
           ~shape:[| 3 |] ~data:[| 1.; 5.; 3. |] ~scalar:2.0
           ~expected:[| 1.; 2.; 2. |]);
      test "iminimum"
        (test_inplace ~iop:Nx.iminimum ~op_name:"minimum" ~dtype:Nx.int32
           ~shape:[| 3 |] ~a_data:[| 1l; 3l; 4l |] ~b_data:[| 2l; 1l; 4l |]
           ~expected:[| 1l; 1l; 4l |]);
    ]
end

(* ───── Rounding Operation Tests ───── *)

module Rounding_tests = struct
  let tests =
    [
      test "round"
        (test_unary_op ~op:Nx.round ~op_name:"round" ~dtype:Nx.float32
           ~shape:[| 3 |] ~input:[| 1.6; 2.4; -1.7 |] ~expected:[| 2.; 2.; -2. |]);
      test "floor"
        (test_unary_op ~op:Nx.floor ~op_name:"floor" ~dtype:Nx.float32
           ~shape:[| 3 |] ~input:[| 1.6; 2.4; -1.7 |] ~expected:[| 1.; 2.; -2. |]);
      test "ceil"
        (test_unary_op ~op:Nx.ceil ~op_name:"ceil" ~dtype:Nx.float32
           ~shape:[| 3 |] ~input:[| 1.6; 2.4; -1.7 |] ~expected:[| 2.; 3.; -1. |]);
      test "clip"
        (fun () ->
          let t = Nx.create Nx.float32 [| 5 |] [| -1.; 2.; 5.; 8.; 10. |] in
          let result = Nx.clip ~min:0. ~max:7. t in
          check_t "clip" [| 5 |] [| 0.; 2.; 5.; 7.; 7. |] result);
    ]
end

(* ───── Bitwise Operation Tests ───── *)

module Cumulative_tests = struct
  let tests =
    [
      test "cumsum default axis"
        (fun () ->
          let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
          let result = Nx.cumsum t in
          check_t ~eps:1e-6 "cumsum flatten" [| 2; 2 |] [| 1.; 3.; 6.; 10. |]
            result);
      test "cumsum axis=1"
        (fun () ->
          let t =
            Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let result = Nx.cumsum ~axis:1 t in
          check_t ~eps:1e-6 "cumsum axis=1" [| 2; 3 |]
            [| 1.; 3.; 6.; 4.; 9.; 15. |]
            result);
      test "cumprod axis=-1"
        (fun () ->
          let t = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l; 2l; 2l; 2l |] in
          let result = Nx.cumprod ~axis:(-1) t in
          check_t "cumprod axis=-1" [| 2; 3 |]
            [| 1l; 2l; 6l; 2l; 4l; 8l |]
            result);
      test "cummax nan propagation"
        (fun () ->
          let t = Nx.create Nx.float32 [| 4 |] [| 1.; Float.nan; 2.; 3. |] in
          let result = Nx.cummax t in
          equal ~msg:"cummax nan[1]" bool true (Float.is_nan (Nx.item [ 1 ] result));
          equal ~msg:"cummax nan[2]" bool true (Float.is_nan (Nx.item [ 2 ] result)));
      test "cummin axis option"
        (fun () ->
          let t =
            Nx.create Nx.int32 [| 2; 4 |] [| 4l; 2l; 3l; 1l; 5l; 6l; 2l; 7l |]
          in
          let result = Nx.cummin ~axis:0 t in
          check_t "cummin axis=0" [| 2; 4 |]
            [| 4l; 2l; 3l; 1l; 4l; 2l; 2l; 1l |]
            result);
    ]
end

(* ───── Bitwise Operation Tests ───── *)

module Bitwise_tests = struct
  let tests =
    [
      test "bitwise_and"
        (test_binary_op ~op:Nx.bitwise_and ~op_name:"bitwise_and" ~dtype:Nx.int32
           ~shape:[| 3 |] ~a_data:[| 5l; 3l; 7l |] ~b_data:[| 3l; 5l; 6l |]
           ~expected:[| 1l; 1l; 6l |]);
      test "bitwise_or"
        (test_binary_op ~op:Nx.bitwise_or ~op_name:"bitwise_or" ~dtype:Nx.int32
           ~shape:[| 3 |] ~a_data:[| 5l; 3l; 7l |] ~b_data:[| 3l; 5l; 6l |]
           ~expected:[| 7l; 7l; 7l |]);
      test "bitwise_xor"
        (test_binary_op ~op:Nx.bitwise_xor ~op_name:"bitwise_xor" ~dtype:Nx.int32
           ~shape:[| 3 |] ~a_data:[| 5l; 3l; 7l |] ~b_data:[| 3l; 5l; 6l |]
           ~expected:[| 6l; 6l; 1l |]);
      test "invert"
        (test_unary_op ~op:Nx.invert ~op_name:"invert" ~dtype:Nx.int32
           ~shape:[| 3 |] ~input:[| 5l; 0l; 7l |] ~expected:[| -6l; -1l; -8l |]);
    ]
end

(* Log/standardize related tests *)

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

module Log_tests = struct
  let tests =
    [
      test "log_softmax basic" test_log_softmax_basic;
      test "log_softmax scale" test_log_softmax_with_scale;
      test "logsumexp basic" test_logsumexp_basic;
      test "logsumexp axis" test_logsumexp_axis;
      test "logmeanexp basic" test_logmeanexp_basic;
      test "logmeanexp axis" test_logmeanexp_axis;
    ]
end

module Standardize_tests = struct
  let tests =
    [
      test "standardize global" test_standardize_global;
      test "standardize axes with params" test_standardize_axes_with_params;
    ]
end

(* ───── Broadcasting Tests ───── *)

module Broadcasting_tests = struct
  let tests =
    [
      test "broadcast 1d to 2d"
        (fun () ->
          let t2d =
            Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let t1d = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
          let result = Nx.add t2d t1d in
          check_t "broadcast 1d to 2d" [| 2; 3 |]
            [| 11.; 22.; 33.; 14.; 25.; 36. |]
            result);
      test "broadcast 2x1 to 2x3"
        (fun () ->
          let t21 = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 2. |] in
          let t23 =
            Nx.create Nx.float32 [| 2; 3 |] [| 10.; 11.; 12.; 20.; 21.; 22. |]
          in
          let result = Nx.add t21 t23 in
          check_t "broadcast 2x1 to 2x3" [| 2; 3 |]
            [| 11.; 12.; 13.; 22.; 23.; 24. |]
            result);
      test "broadcast (4,1) * (3,)"
        (fun () ->
          let t41 =
            Nx.reshape [| 4; 1 |]
              (Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
          in
          let t3 = Nx.create Nx.float32 [| 3 |] [| 10.; 100.; 1000. |] in
          let result = Nx.mul t41 t3 in
          check_t "broadcast (4,1) * (3,)" [| 4; 3 |]
            [|
              10.;
              100.;
              1000.;
              20.;
              200.;
              2000.;
              30.;
              300.;
              3000.;
              40.;
              400.;
              4000.;
            |]
            result);
    ]
end

(* Test Suite Organization *)

let suite =
  [
    group "Add" Add_tests.tests;
    group "Sub" Sub_tests.tests;
    group "Mul" Mul_tests.tests;
    group "Div" Div_tests.tests;
    group "Pow" Pow_tests.tests;
    group "Mod" Mod_tests.tests;
    group "Math Functions" Math_tests.tests;
    group "Trigonometric" Trig_tests.tests;
    group "Comparison" Comparison_tests.tests;
    group "Reduction" Reduction_tests.tests;
    group "Min/Max" MinMax_tests.tests;
    group "Rounding" Rounding_tests.tests;
    group "Cumulative" Cumulative_tests.tests;
    group "Bitwise" Bitwise_tests.tests;
    group "Broadcasting" Broadcasting_tests.tests;
    group "Log" Log_tests.tests;
    group "Standardize" Standardize_tests.tests;
  ]

let () = run "Nx Ops" suite
