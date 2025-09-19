(* Comprehensive operation tests for Nx following the test plan *)

open Alcotest
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
  let a = Nx.create Nx_core.Dtype.float32 [| 3 |] [| Float.nan; 1.0; 2.0 |] in
  let b = Nx.create Nx_core.Dtype.float32 [| 3 |] [| 5.0; Float.nan; 3.0 |] in
  let result = op a b in
  check bool
    (Printf.sprintf "%s nan[0]" op_name)
    true
    (Float.is_nan (Nx.item [ 0 ] result));
  check bool
    (Printf.sprintf "%s nan[1]" op_name)
    true
    (Float.is_nan (Nx.item [ 1 ] result))

let test_inplace ~iop ~op_name ~dtype ~shape ~a_data ~b_data ~expected () =
  let a = Nx.create dtype shape a_data in
  let b = Nx.create dtype shape b_data in
  let result = iop a b in
  check_t (Printf.sprintf "i%s result" op_name) shape expected a;
  check bool (Printf.sprintf "i%s returns a" op_name) true (result == a)

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
      ( "0d + 0d",
        `Quick,
        test_binary_op_0d ~op ~op_name ~dtype:Nx_core.Dtype.float32 ~a_val:5.0
          ~b_val:3.0 ~expected:8.0 );
      ( "1d + 1d",
        `Quick,
        test_binary_op ~op ~op_name ~dtype:Nx_core.Dtype.float32 ~shape:[| 5 |]
          ~a_data:[| 1.; 2.; 3.; 4.; 5. |] ~b_data:[| 5.; 4.; 3.; 2.; 1. |]
          ~expected:[| 6.; 6.; 6.; 6.; 6. |] );
      ( "2d + 2d",
        `Quick,
        test_binary_op ~op ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 2; 2 |] ~a_data:[| 1.; 2.; 3.; 4. |]
          ~b_data:[| 5.; 6.; 7.; 8. |] ~expected:[| 6.; 8.; 10.; 12. |] );
      ( "5d + 5d",
        `Quick,
        fun () ->
          let shape = [| 2; 3; 4; 5; 6 |] in
          let size = Array.fold_left ( * ) 1 shape in
          test_binary_op ~op ~op_name ~dtype:Nx_core.Dtype.float32 ~shape
            ~a_data:(Array.make size 1.0) ~b_data:(Array.make size 2.0)
            ~expected:(Array.make size 3.0) () );
      ( "scalar broadcast",
        `Quick,
        fun () ->
          let a = Nx.scalar Nx_core.Dtype.float32 5.0 in
          let b =
            Nx.create Nx_core.Dtype.float32 [| 3; 4 |] (Array.make 12 1.0)
          in
          let result = Nx.add a b in
          check_t "add scalar broadcast" [| 3; 4 |] (Array.make 12 6.0) result
      );
      ( "1d broadcast to 2d",
        `Quick,
        test_broadcast ~op ~op_name ~dtype:Nx_core.Dtype.float32
          ~a_shape:[| 4 |] ~a_data:[| 1.; 2.; 3.; 4. |] ~b_shape:[| 3; 4 |]
          ~b_data:(Array.make 12 10.0) ~result_shape:[| 3; 4 |] );
      ( "broadcast error",
        `Quick,
        test_broadcast_error ~op ~op_name ~dtype:Nx_core.Dtype.float32
          ~a_shape:[| 3 |] ~b_shape:[| 4 |] );
      ("nan propagation", `Quick, test_nan_propagation ~op ~op_name);
      ( "inf arithmetic",
        `Quick,
        fun () ->
          let a =
            Nx.create Nx_core.Dtype.float32 [| 2 |]
              [| Float.infinity; Float.neg_infinity |]
          in
          let b = Nx.create Nx_core.Dtype.float32 [| 2 |] [| 5.0; 10.0 |] in
          let result = Nx.add a b in
          check (float 1e-6) "inf + 5" Float.infinity (Nx.item [ 0 ] result);
          check (float 1e-6) "-inf + 10" Float.neg_infinity
            (Nx.item [ 1 ] result) );
      ( "inf + inf",
        `Quick,
        fun () ->
          let a =
            Nx.create Nx_core.Dtype.float32 [| 2 |]
              [| Float.infinity; Float.infinity |]
          in
          let b =
            Nx.create Nx_core.Dtype.float32 [| 2 |]
              [| Float.infinity; Float.neg_infinity |]
          in
          let result = Nx.add a b in
          check (float 1e-6) "inf + inf" Float.infinity (Nx.item [ 0 ] result);
          check bool "inf + -inf" true (Float.is_nan (Nx.item [ 1 ] result)) );
      ( "iadd",
        `Quick,
        test_inplace ~iop:Nx.iadd ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 2; 2 |] ~a_data:[| 1.; 2.; 3.; 4. |]
          ~b_data:[| 5.; 6.; 7.; 8. |] ~expected:[| 6.; 8.; 10.; 12. |] );
      ( "add_s",
        `Quick,
        test_scalar_op ~op_s:Nx.add_s ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~data:[| 1.; 2.; 3. |] ~scalar:5.0
          ~expected:[| 6.; 7.; 8. |] );
    ]
end

module Sub_tests = struct
  let op = Nx.sub
  let op_name = "sub"

  let tests =
    [
      ( "2d - 2d",
        `Quick,
        test_binary_op ~op ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 2; 2 |] ~a_data:[| 5.; 6.; 7.; 8. |]
          ~b_data:[| 1.; 2.; 3.; 4. |] ~expected:[| 4.; 4.; 4.; 4. |] );
      ( "inf - inf",
        `Quick,
        fun () ->
          let a =
            Nx.create Nx_core.Dtype.float32 [| 2 |]
              [| Float.infinity; Float.infinity |]
          in
          let b =
            Nx.create Nx_core.Dtype.float32 [| 2 |]
              [| Float.infinity; Float.neg_infinity |]
          in
          let result = Nx.sub a b in
          check bool "inf - inf" true (Float.is_nan (Nx.item [ 0 ] result));
          check (float 1e-6) "inf - -inf" Float.infinity (Nx.item [ 1 ] result)
      );
      ( "isub",
        `Quick,
        test_inplace ~iop:Nx.isub ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 2; 2 |] ~a_data:[| 10.; 11.; 12.; 13. |]
          ~b_data:[| 1.; 2.; 3.; 4. |] ~expected:[| 9.; 9.; 9.; 9. |] );
      ( "sub_s",
        `Quick,
        test_scalar_op ~op_s:Nx.sub_s ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~data:[| 5.; 6.; 7. |] ~scalar:2.0
          ~expected:[| 3.; 4.; 5. |] );
    ]
end

module Mul_tests = struct
  let op = Nx.mul
  let op_name = "mul"

  let tests =
    [
      ( "2d * 2d",
        `Quick,
        test_binary_op ~op ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 2; 2 |] ~a_data:[| 1.; 2.; 3.; 4. |]
          ~b_data:[| 5.; 6.; 7.; 8. |] ~expected:[| 5.; 12.; 21.; 32. |] );
      ( "imul",
        `Quick,
        test_inplace ~iop:Nx.imul ~op_name ~dtype:Nx_core.Dtype.int32
          ~shape:[| 3 |] ~a_data:[| 1l; 2l; 3l |] ~b_data:[| 5l; 6l; 7l |]
          ~expected:[| 5l; 12l; 21l |] );
      ( "mul_s",
        `Quick,
        test_scalar_op ~op_s:Nx.mul_s ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~data:[| 1.; 2.; 3. |] ~scalar:3.0
          ~expected:[| 3.; 6.; 9. |] );
    ]
end

module Div_tests = struct
  let op = Nx.div
  let op_name = "div"

  let tests =
    [
      ( "2d / 2d float32",
        `Quick,
        test_binary_op_float ~eps:1e-6 ~op ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 2; 2 |] ~a_data:[| 5.; 6.; 7.; 8. |]
          ~b_data:[| 1.; 2.; 4.; 5. |] ~expected:[| 5.; 3.; 1.75; 1.6 |] );
      ( "2d / 2d int32",
        `Quick,
        test_binary_op ~op ~op_name ~dtype:Nx_core.Dtype.int32 ~shape:[| 2; 2 |]
          ~a_data:[| 10l; 21l; 30l; 40l |] ~b_data:[| 3l; 5l; 4l; 4l |]
          ~expected:[| 3l; 4l; 7l; 10l |] );
      ( "div by zero float",
        `Quick,
        fun () ->
          let a =
            Nx.create Nx_core.Dtype.float32 [| 3 |] [| 1.0; -1.0; 0.0 |]
          in
          let b = Nx.create Nx_core.Dtype.float32 [| 3 |] [| 0.0; 0.0; 0.0 |] in
          let result = Nx.div a b in
          check (float 1e-6) "1/0" Float.infinity (Nx.item [ 0 ] result);
          check (float 1e-6) "-1/0" Float.neg_infinity (Nx.item [ 1 ] result);
          check bool "0/0" true (Float.is_nan (Nx.item [ 2 ] result)) );
      ( "idiv",
        `Quick,
        test_inplace ~iop:Nx.idiv ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~a_data:[| 10.; 20.; 30. |] ~b_data:[| 2.; 2.; 2. |]
          ~expected:[| 5.; 10.; 15. |] );
      ( "div_s",
        `Quick,
        test_scalar_op ~op_s:Nx.div_s ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~data:[| 6.; 9.; 12. |] ~scalar:3.0
          ~expected:[| 2.; 3.; 4. |] );
    ]
end

module Pow_tests = struct
  let op = Nx.pow
  let op_name = "pow"

  let tests =
    [
      ( "basic pow",
        `Quick,
        test_binary_op_float ~eps:1e-5 ~op ~op_name ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~a_data:[| 2.; 3.; 4. |] ~b_data:[| 3.; 2.; 0.5 |]
          ~expected:[| 8.; 9.; 2. |] );
      ( "zero^zero",
        `Quick,
        fun () ->
          let a = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 0.0 |] in
          let b = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 0.0 |] in
          let result = Nx.pow a b in
          check (float 1e-6) "0^0" 1.0 (Nx.item [ 0 ] result) );
      ( "negative base fractional exp",
        `Quick,
        fun () ->
          let a = Nx.create Nx_core.Dtype.float32 [| 1 |] [| -2.0 |] in
          let b = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 0.5 |] in
          let result = Nx.pow a b in
          check bool "(-2)^0.5" true (Float.is_nan (Nx.item [ 0 ] result)) );
      ( "pow overflow",
        `Quick,
        fun () ->
          let a = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 10.0 |] in
          let b = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 100.0 |] in
          let result = Nx.pow a b in
          check (float 1e-6) "10^100" Float.infinity (Nx.item [ 0 ] result) );
      ( "ipow",
        `Quick,
        fun () ->
          let a = Nx.create Nx_core.Dtype.float32 [| 3 |] [| 2.; 3.; 4. |] in
          let b = Nx.create Nx_core.Dtype.float32 [| 3 |] [| 2.; 1.; 0.5 |] in
          let result = Nx.ipow a b in
          check_t ~eps:1e-5 "ipow result" [| 3 |] [| 4.; 3.; 2. |] a;
          check bool "ipow returns a" true (result == a) );
      ( "pow_s",
        `Quick,
        fun () ->
          let a = Nx.create Nx_core.Dtype.float32 [| 3 |] [| 2.; 3.; 4. |] in
          let result = Nx.pow_s a 2.0 in
          check_t ~eps:1e-5 "pow_s" [| 3 |] [| 4.; 9.; 16. |] result );
    ]
end

module Mod_tests = struct
  let op = Nx.mod_
  let op_name = "mod"

  let tests =
    [
      ( "basic mod",
        `Quick,
        test_binary_op ~op ~op_name ~dtype:Nx_core.Dtype.int32 ~shape:[| 3 |]
          ~a_data:[| 10l; 11l; 12l |] ~b_data:[| 3l; 5l; 4l |]
          ~expected:[| 1l; 1l; 0l |] );
      ( "imod",
        `Quick,
        test_inplace ~iop:Nx.imod ~op_name ~dtype:Nx_core.Dtype.int32
          ~shape:[| 3 |] ~a_data:[| 10l; 11l; 12l |] ~b_data:[| 3l; 5l; 4l |]
          ~expected:[| 1l; 1l; 0l |] );
      ( "mod_s",
        `Quick,
        test_scalar_op ~op_s:Nx.mod_s ~op_name ~dtype:Nx_core.Dtype.int32
          ~shape:[| 3 |] ~data:[| 10l; 11l; 12l |] ~scalar:3l
          ~expected:[| 1l; 2l; 0l |] );
    ]
end

(* ───── Math Function Tests ───── *)

module Math_tests = struct
  let test_exp =
    [
      ( "exp basic",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.exp ~op_name:"exp"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~input:[| 0.; 1.; 2. |]
          ~expected:[| 1.; 2.71828; 7.38906 |] );
      ( "exp overflow",
        `Quick,
        fun () ->
          let t = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 1000.0 |] in
          let result = Nx.exp t in
          check (float 1e-6) "exp(1000)" Float.infinity (Nx.item [ 0 ] result)
      );
      ( "exp underflow",
        `Quick,
        fun () ->
          let t = Nx.create Nx_core.Dtype.float32 [| 1 |] [| -1000.0 |] in
          let result = Nx.exp t in
          check (float 1e-6) "exp(-1000)" 0.0 (Nx.item [ 0 ] result) );
    ]

  let test_log =
    [
      ( "log basic",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.log ~op_name:"log"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |]
          ~input:[| 1.; 2.71828; 7.38905 |] ~expected:[| 0.; 1.; 2. |] );
      ( "log negative",
        `Quick,
        fun () ->
          let t = Nx.create Nx_core.Dtype.float32 [| 1 |] [| -1.0 |] in
          let result = Nx.log t in
          check bool "log(-1)" true (Float.is_nan (Nx.item [ 0 ] result)) );
      ( "log zero",
        `Quick,
        fun () ->
          let t = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 0.0 |] in
          let result = Nx.log t in
          check (float 1e-6) "log(0)" Float.neg_infinity (Nx.item [ 0 ] result)
      );
    ]

  let test_sqrt =
    [
      ( "sqrt basic",
        `Quick,
        test_unary_op ~op:Nx.sqrt ~op_name:"sqrt" ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~input:[| 4.; 9.; 16. |] ~expected:[| 2.; 3.; 4. |] );
      ( "sqrt negative",
        `Quick,
        fun () ->
          let t = Nx.create Nx_core.Dtype.float32 [| 1 |] [| -1.0 |] in
          let result = Nx.sqrt t in
          check bool "sqrt(-1)" true (Float.is_nan (Nx.item [ 0 ] result)) );
    ]

  let test_abs =
    [
      ( "abs int32",
        `Quick,
        test_unary_op ~op:Nx.abs ~op_name:"abs" ~dtype:Nx_core.Dtype.int32
          ~shape:[| 4 |] ~input:[| -1l; 0l; 5l; -10l |]
          ~expected:[| 1l; 0l; 5l; 10l |] );
      ( "abs float32",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.abs ~op_name:"abs"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |]
          ~input:[| -1.5; 0.0; 5.2 |] ~expected:[| 1.5; 0.0; 5.2 |] );
    ]

  let test_sign =
    [
      ( "sign float32",
        `Quick,
        test_unary_op ~op:Nx.sign ~op_name:"sign" ~dtype:Nx_core.Dtype.float32
          ~shape:[| 4 |] ~input:[| -5.0; 0.0; 3.2; -0.0 |]
          ~expected:[| -1.0; 0.0; 1.0; 0.0 |] );
      ( "sign int32",
        `Quick,
        test_unary_op ~op:Nx.sign ~op_name:"sign" ~dtype:Nx_core.Dtype.int32
          ~shape:[| 3 |] ~input:[| -5l; 0l; 3l |] ~expected:[| -1l; 0l; 1l |] );
    ]

  let tests =
    test_exp @ test_log @ test_sqrt @ test_abs @ test_sign
    @ [
        ( "neg",
          `Quick,
          test_unary_op ~op:Nx.neg ~op_name:"neg" ~dtype:Nx_core.Dtype.float32
            ~shape:[| 3 |] ~input:[| 1.0; -2.0; 0.0 |]
            ~expected:[| -1.0; 2.0; 0.0 |] );
        ( "square",
          `Quick,
          test_unary_op ~op:Nx.square ~op_name:"square"
            ~dtype:Nx_core.Dtype.float32 ~shape:[| 4 |]
            ~input:[| 1.; 2.; -3.; 0. |] ~expected:[| 1.; 4.; 9.; 0. |] );
      ]
end

(* ───── Trigonometric Function Tests ───── *)

module Trig_tests = struct
  let pi = Float.pi

  let tests =
    [
      ( "sin",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.sin ~op_name:"sin"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 4 |]
          ~input:[| 0.; pi /. 6.; pi /. 4.; pi /. 2. |]
          ~expected:[| 0.; 0.5; 0.707107; 1. |] );
      ( "cos",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.cos ~op_name:"cos"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 4 |]
          ~input:[| 0.; pi /. 6.; pi /. 4.; pi /. 2. |]
          ~expected:[| 1.; 0.866025; 0.707107; 0. |] );
      ( "tan",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.tan ~op_name:"tan"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |]
          ~input:[| 0.; pi /. 6.; pi /. 4. |]
          ~expected:[| 0.; 0.577350; 1. |] );
      ( "asin",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.asin ~op_name:"asin"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~input:[| 0.; 0.5; 1. |]
          ~expected:[| 0.; pi /. 6.; pi /. 2. |] );
      ( "asin out of domain",
        `Quick,
        fun () ->
          let t = Nx.create Nx_core.Dtype.float32 [| 1 |] [| 2.0 |] in
          let result = Nx.asin t in
          check bool "asin(2)" true (Float.is_nan (Nx.item [ 0 ] result)) );
      ( "sinh",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.sinh ~op_name:"sinh"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~input:[| -1.; 0.; 1. |]
          ~expected:[| -1.17520; 0.; 1.17520 |] );
      ( "cosh",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.cosh ~op_name:"cosh"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~input:[| -1.; 0.; 1. |]
          ~expected:[| 1.54308; 1.; 1.54308 |] );
      ( "tanh",
        `Quick,
        test_unary_op_float ~eps:1e-5 ~op:Nx.tanh ~op_name:"tanh"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~input:[| -1.; 0.; 1. |]
          ~expected:[| -0.761594; 0.; 0.761594 |] );
    ]
end

(* ───── Comparison Operation Tests ───── *)

module Comparison_tests = struct
  let test_comparison ~op ~op_name ~input1 ~input2 ~expected () =
    let t1 = Nx.create Nx_core.Dtype.float32 [| 3 |] input1 in
    let t2 = Nx.create Nx_core.Dtype.float32 [| 3 |] input2 in
    let result = op t1 t2 in
    check_t op_name [| 3 |] expected result

  let tests =
    [
      ( "equal",
        `Quick,
        test_comparison ~op:Nx.equal ~op_name:"equal" ~input1:[| 1.; 2.; 3. |]
          ~input2:[| 1.; 5.; 3. |] ~expected:[| 1; 0; 1 |] );
      ( "not_equal",
        `Quick,
        test_comparison ~op:Nx.not_equal ~op_name:"not_equal"
          ~input1:[| 1.; 2.; 3. |] ~input2:[| 1.; 5.; 3. |]
          ~expected:[| 0; 1; 0 |] );
      ( "greater",
        `Quick,
        test_comparison ~op:Nx.greater ~op_name:"greater"
          ~input1:[| 1.; 3.; 2. |] ~input2:[| 2.; 1.; 4. |]
          ~expected:[| 0; 1; 0 |] );
      ( "greater_equal",
        `Quick,
        test_comparison ~op:Nx.greater_equal ~op_name:"greater_equal"
          ~input1:[| 1.; 3.; 4. |] ~input2:[| 2.; 3.; 3. |]
          ~expected:[| 0; 1; 1 |] );
      ( "less",
        `Quick,
        test_comparison ~op:Nx.less ~op_name:"less" ~input1:[| 1.; 3.; 2. |]
          ~input2:[| 2.; 1.; 4. |] ~expected:[| 1; 0; 1 |] );
      ( "less_equal",
        `Quick,
        test_comparison ~op:Nx.less_equal ~op_name:"less_equal"
          ~input1:[| 1.; 3.; 4. |] ~input2:[| 2.; 3.; 3. |]
          ~expected:[| 1; 1; 0 |] );
      ( "nan comparisons",
        `Quick,
        fun () ->
          let t1 =
            Nx.create Nx_core.Dtype.float32 [| 3 |]
              [| Float.nan; 1.; Float.nan |]
          in
          let t2 =
            Nx.create Nx_core.Dtype.float32 [| 3 |]
              [| Float.nan; Float.nan; 1. |]
          in
          let eq_result = Nx.equal t1 t2 in
          let ne_result = Nx.not_equal t1 t2 in
          check_t "nan equal" [| 3 |] [| 0; 0; 0 |] eq_result;
          check_t "nan not_equal" [| 3 |] [| 1; 1; 1 |] ne_result );
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
      ( "sum all",
        `Quick,
        test_reduction_all ~op:Nx.sum ~op_name:"sum"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 2; 2 |]
          ~input:[| 1.; 2.; 3.; 4. |] ~expected:10. );
      ( "sum axis=0",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 2; 3 |]
              [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let result = Nx.sum ~axes:[| 0 |] t in
          check_t "sum axis=0" [| 3 |] [| 5.; 7.; 9. |] result );
      ( "sum axis=1 keepdims",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 2; 3 |]
              [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let result = Nx.sum ~axes:[| 1 |] ~keepdims:true t in
          check_t "sum axis=1 keepdims" [| 2; 1 |] [| 6.; 15. |] result );
      ( "prod all",
        `Quick,
        fun () ->
          let t = Nx.create Nx_core.Dtype.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
          let result = Nx.prod t in
          check_t "prod all" [||] [| 24l |] result );
      ( "mean all",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
          in
          let result = Nx.mean t in
          check_t "mean all" [||] [| 2.5 |] result );
      ( "max all",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
          in
          let result = Nx.max t in
          check_t "max all" [||] [| 4. |] result );
      ( "min all",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
          in
          let result = Nx.min t in
          check_t "min all" [||] [| 1. |] result );
      ( "var 1d",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
          in
          let result = Nx.var t in
          check_t "var 1d" [||] [| 2. |] result );
      ( "std 1d",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
          in
          let result = Nx.std t in
          check_t ~eps:1e-5 "std 1d" [||] [| 1.41421 |] result );
      ( "empty array mean",
        `Quick,
        fun () ->
          let _t = Nx.create Nx_core.Dtype.float32 [| 0 |] [||] in
          (* Skip for now - mean of empty array behavior needs investigation *)
          () );
      ( "min/max with nan",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 3 |] [| 1.; Float.nan; 3. |]
          in
          let min_result = Nx.min t in
          let max_result = Nx.max t in
          check bool "min with nan" true (Float.is_nan (Nx.item [] min_result));
          check bool "max with nan" true (Float.is_nan (Nx.item [] max_result))
      );
    ]
end

(* ───── Min/Max Operation Tests ───── *)

module MinMax_tests = struct
  let tests =
    [
      ( "maximum",
        `Quick,
        test_binary_op ~op:Nx.maximum ~op_name:"maximum"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~a_data:[| 1.; 3.; 2. |]
          ~b_data:[| 2.; 1.; 4. |] ~expected:[| 2.; 3.; 4. |] );
      ( "maximum_s",
        `Quick,
        test_scalar_op ~op_s:Nx.maximum_s ~op_name:"maximum"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~data:[| 1.; 5.; 3. |]
          ~scalar:2.0 ~expected:[| 2.; 5.; 3. |] );
      ( "imaximum",
        `Quick,
        test_inplace ~iop:Nx.imaximum ~op_name:"maximum"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~a_data:[| 1.; 3.; 2. |]
          ~b_data:[| 2.; 1.; 4. |] ~expected:[| 2.; 3.; 4. |] );
      ( "minimum",
        `Quick,
        test_binary_op ~op:Nx.minimum ~op_name:"minimum"
          ~dtype:Nx_core.Dtype.int32 ~shape:[| 3 |] ~a_data:[| 1l; 3l; 4l |]
          ~b_data:[| 2l; 1l; 4l |] ~expected:[| 1l; 1l; 4l |] );
      ( "minimum_s",
        `Quick,
        test_scalar_op ~op_s:Nx.minimum_s ~op_name:"minimum"
          ~dtype:Nx_core.Dtype.float32 ~shape:[| 3 |] ~data:[| 1.; 5.; 3. |]
          ~scalar:2.0 ~expected:[| 1.; 2.; 2. |] );
      ( "iminimum",
        `Quick,
        test_inplace ~iop:Nx.iminimum ~op_name:"minimum"
          ~dtype:Nx_core.Dtype.int32 ~shape:[| 3 |] ~a_data:[| 1l; 3l; 4l |]
          ~b_data:[| 2l; 1l; 4l |] ~expected:[| 1l; 1l; 4l |] );
    ]
end

(* ───── Rounding Operation Tests ───── *)

module Rounding_tests = struct
  let tests =
    [
      ( "round",
        `Quick,
        test_unary_op ~op:Nx.round ~op_name:"round" ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~input:[| 1.6; 2.4; -1.7 |] ~expected:[| 2.; 2.; -2. |]
      );
      ( "floor",
        `Quick,
        test_unary_op ~op:Nx.floor ~op_name:"floor" ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~input:[| 1.6; 2.4; -1.7 |] ~expected:[| 1.; 2.; -2. |]
      );
      ( "ceil",
        `Quick,
        test_unary_op ~op:Nx.ceil ~op_name:"ceil" ~dtype:Nx_core.Dtype.float32
          ~shape:[| 3 |] ~input:[| 1.6; 2.4; -1.7 |] ~expected:[| 2.; 3.; -1. |]
      );
      ( "clip",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 5 |] [| -1.; 2.; 5.; 8.; 10. |]
          in
          let result = Nx.clip ~min:0. ~max:7. t in
          check_t "clip" [| 5 |] [| 0.; 2.; 5.; 7.; 7. |] result );
    ]
end

(* ───── Bitwise Operation Tests ───── *)

module Cumulative_tests = struct
  let tests =
    [
      ( "cumsum default axis",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
          in
          let result = Nx.cumsum t in
          check_t ~eps:1e-6 "cumsum flatten" [| 2; 2 |]
            [| 1.; 3.; 6.; 10. |]
            result );
      ( "cumsum axis=1",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 2; 3 |]
              [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let result = Nx.cumsum ~axis:1 t in
          check_t ~eps:1e-6 "cumsum axis=1" [| 2; 3 |]
            [| 1.; 3.; 6.; 4.; 9.; 15. |]
            result );
      ( "cumprod axis=-1",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.int32 [| 2; 3 |]
              [| 1l; 2l; 3l; 2l; 2l; 2l |]
          in
          let result = Nx.cumprod ~axis:(-1) t in
          check_t "cumprod axis=-1" [| 2; 3 |]
            [| 1l; 2l; 6l; 2l; 4l; 8l |]
            result );
      ( "cummax nan propagation",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.float32 [| 4 |]
              [| 1.; Float.nan; 2.; 3. |]
          in
          let result = Nx.cummax t in
          check bool "cummax nan[1]" true
            (Float.is_nan (Nx.item [ 1 ] result));
          check bool "cummax nan[2]" true
            (Float.is_nan (Nx.item [ 2 ] result)) );
      ( "cummin axis option",
        `Quick,
        fun () ->
          let t =
            Nx.create Nx_core.Dtype.int32 [| 2; 4 |]
              [| 4l; 2l; 3l; 1l; 5l; 6l; 2l; 7l |]
          in
          let result = Nx.cummin ~axis:0 t in
          check_t "cummin axis=0" [| 2; 4 |]
            [| 4l; 2l; 3l; 1l; 4l; 2l; 2l; 1l |]
            result );
    ]
end

(* ───── Bitwise Operation Tests ───── *)

module Bitwise_tests = struct
  let tests =
    [
      ( "bitwise_and",
        `Quick,
        test_binary_op ~op:Nx.bitwise_and ~op_name:"bitwise_and" ~dtype:Nx.int32
          ~shape:[| 3 |] ~a_data:[| 5l; 3l; 7l |] ~b_data:[| 3l; 5l; 6l |]
          ~expected:[| 1l; 1l; 6l |] );
      ( "bitwise_or",
        `Quick,
        test_binary_op ~op:Nx.bitwise_or ~op_name:"bitwise_or" ~dtype:Nx.int32
          ~shape:[| 3 |] ~a_data:[| 5l; 3l; 7l |] ~b_data:[| 3l; 5l; 6l |]
          ~expected:[| 7l; 7l; 7l |] );
      ( "bitwise_xor",
        `Quick,
        test_binary_op ~op:Nx.bitwise_xor ~op_name:"bitwise_xor" ~dtype:Nx.int32
          ~shape:[| 3 |] ~a_data:[| 5l; 3l; 7l |] ~b_data:[| 3l; 5l; 6l |]
          ~expected:[| 6l; 6l; 1l |] );
      ( "invert",
        `Quick,
        test_unary_op ~op:Nx.invert ~op_name:"invert" ~dtype:Nx.int32
          ~shape:[| 3 |] ~input:[| 5l; 0l; 7l |] ~expected:[| -6l; -1l; -8l |]
      );
    ]
end

(* ───── Broadcasting Tests ───── *)

module Broadcasting_tests = struct
  let tests =
    [
      ( "broadcast 1d to 2d",
        `Quick,
        fun () ->
          let t2d =
            Nx.create Nx_core.Dtype.float32 [| 2; 3 |]
              [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let t1d =
            Nx.create Nx_core.Dtype.float32 [| 3 |] [| 10.; 20.; 30. |]
          in
          let result = Nx.add t2d t1d in
          check_t "broadcast 1d to 2d" [| 2; 3 |]
            [| 11.; 22.; 33.; 14.; 25.; 36. |]
            result );
      ( "broadcast 2x1 to 2x3",
        `Quick,
        fun () ->
          let t21 = Nx.create Nx_core.Dtype.float32 [| 2; 1 |] [| 1.; 2. |] in
          let t23 =
            Nx.create Nx_core.Dtype.float32 [| 2; 3 |]
              [| 10.; 11.; 12.; 20.; 21.; 22. |]
          in
          let result = Nx.add t21 t23 in
          check_t "broadcast 2x1 to 2x3" [| 2; 3 |]
            [| 11.; 12.; 13.; 22.; 23.; 24. |]
            result );
      ( "broadcast (4,1) * (3,)",
        `Quick,
        fun () ->
          let t41 =
            Nx.reshape [| 4; 1 |]
              (Nx.create Nx_core.Dtype.float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
          in
          let t3 =
            Nx.create Nx_core.Dtype.float32 [| 3 |] [| 10.; 100.; 1000. |]
          in
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
            result );
    ]
end

(* Test Suite Organization *)

let suite =
  [
    ("Ops :: Add", Add_tests.tests);
    ("Ops :: Sub", Sub_tests.tests);
    ("Ops :: Mul", Mul_tests.tests);
    ("Ops :: Div", Div_tests.tests);
    ("Ops :: Pow", Pow_tests.tests);
    ("Ops :: Mod", Mod_tests.tests);
    ("Ops :: Math Functions", Math_tests.tests);
    ("Ops :: Trigonometric", Trig_tests.tests);
    ("Ops :: Comparison", Comparison_tests.tests);
    ("Ops :: Reduction", Reduction_tests.tests);
    ("Ops :: Min/Max", MinMax_tests.tests);
    ("Ops :: Rounding", Rounding_tests.tests);
    ("Ops :: Cumulative", Cumulative_tests.tests);
    ("Ops :: Bitwise", Bitwise_tests.tests);
    ("Ops :: Broadcasting", Broadcasting_tests.tests);
  ]

let () = Alcotest.run "Nx Ops" suite
