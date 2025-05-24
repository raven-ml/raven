open Alcotest
module Nd = Nx

(* Approximate equality for floating-point nxs *)
let approx_equal epsilon a b =
  if Nd.shape a <> Nd.shape b then false
  else
    let diff = Nd.sub a b in
    let abs_diff = Nd.abs diff in
    let max_diff = Nd.max abs_diff in
    Nd.get_item [||] max_diff < epsilon

(* Testable types *)
let nx_float32 : (float, Nd.float32_elt) Nd.t testable =
  Alcotest.testable Nd.pp (approx_equal 1e-5)

let nx_float64 : (float, Nd.float64_elt) Nd.t testable =
  Alcotest.testable Nd.pp (approx_equal 1e-10)

let nx_int16 : (int, Nd.int16_elt) Nd.t testable =
  Alcotest.testable Nd.pp Nd.array_equal

let nx_int32 : (int32, Nd.int32_elt) Nd.t testable =
  Alcotest.testable Nd.pp Nd.array_equal

let nx_uint8 : (int, Nd.uint8_elt) Nd.t testable =
  Alcotest.testable Nd.pp Nd.array_equal

let test_add_two_2x2_float32 () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let result = Nd.add t1 t2 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 6.0; 8.0; 10.0; 12.0 |] in
  Alcotest.(check nx_float32) "Add two 2x2 float32" expected result

let test_add_inplace_2x2_float32 () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let t_res = Nd.add_inplace t1 t2 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 6.0; 8.0; 10.0; 12.0 |] in
  Alcotest.(check nx_float32) "Add inplace 2x2 float32" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_add_two_2x2_int32 () =
  let t1 = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let t2 = Nd.create Nd.int32 [| 2; 2 |] [| 5l; 6l; 7l; 8l |] in
  let result = Nd.add t1 t2 in
  let expected = Nd.create Nd.int32 [| 2; 2 |] [| 6l; 8l; 10l; 12l |] in
  Alcotest.(check nx_int32) "Add two 2x2 int32" expected result

let test_add_scalar () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let result = Nd.add_scalar t 3.0 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 4.0; 5.0; 6.0 |] in
  Alcotest.(check nx_float32) "Add scalar to 1D float32" expected result

let test_mul_two_2x2_float32 () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let result = Nd.mul t1 t2 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 12.0; 21.0; 32.0 |] in
  Alcotest.(check nx_float32) "Multiply two 2x2 float32" expected result

let test_mul_inplace_1d_int32 () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 5l; 6l; 7l |] in
  let t_res = Nd.mul_inplace t1 t2 in
  let expected = Nd.create Nd.int32 [| 3 |] [| 5l; 12l; 21l |] in
  Alcotest.(check nx_int32) "Multiply inplace 1D int32" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_mul_scalar () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let result = Nd.mul_scalar t 3.0 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 3.0; 6.0; 9.0 |] in
  Alcotest.(check nx_float32) "Multiply 1D float32 by scalar" expected result

let test_sub_two_2x2_float32 () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let result = Nd.sub t1 t2 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 4.0; 4.0; 4.0; 4.0 |] in
  Alcotest.(check nx_float32) "Subtract two 2x2 float32" expected result

let test_sub_inplace_1d_to_2d_broadcast () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 10.0; 11.0; 12.0; 13.0 |] in
  let t2 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t_res = Nd.sub_inplace t1 t2 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 9.0; 9.0; 11.0; 11.0 |] in
  Alcotest.(check nx_float32) "Subtract inplace 1D->2D broadcast" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_sub_scalar () =
  let t = Nd.create Nd.float32 [| 3 |] [| 5.0; 6.0; 7.0 |] in
  let result = Nd.sub_scalar t 2.0 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 3.0; 4.0; 5.0 |] in
  Alcotest.(check nx_float32) "Subtract scalar from 1D float32" expected result

let test_div_two_2x2_float32 () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 4.0; 5.0 |] in
  let result = Nd.div t1 t2 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 3.0; 1.75; 1.6 |] in
  Alcotest.(check nx_float32) "Divide two 2x2 float32" expected result

let test_div_inplace_scalar_broadcast () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let t2 = Nd.scalar Nd.float32 2.0 in
  let t_res = Nd.div_inplace t1 t2 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 5.0; 10.0; 15.0 |] in
  Alcotest.(check nx_float32) "Divide inplace scalar broadcast" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_div_two_2x2_int32 () =
  let t1 = Nd.create Nd.int32 [| 2; 2 |] [| 10l; 21l; 30l; 40l |] in
  let t2 = Nd.create Nd.int32 [| 2; 2 |] [| 3l; 5l; 4l; 4l |] in
  let result = Nd.div t1 t2 in
  let expected = Nd.create Nd.int32 [| 2; 2 |] [| 3l; 4l; 7l; 10l |] in
  Alcotest.(check nx_int32) "Divide two 2x2 int32" expected result

let test_div_scalar () =
  let t = Nd.create Nd.float32 [| 3 |] [| 6.0; 9.0; 12.0 |] in
  let result = Nd.div_scalar t 3.0 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  Alcotest.(check nx_float32) "Divide 1D float32 by scalar" expected result

let test_rem_int32 () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 10l; 11l; 12l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 3l; 5l; 4l |] in
  let result = Nd.rem t1 t2 in
  let expected = Nd.create Nd.int32 [| 3 |] [| 1l; 1l; 0l |] in
  Alcotest.(check nx_int32) "Remainder int32" expected result

let test_rem_scalar () =
  let t = Nd.create Nd.int32 [| 3 |] [| 10l; 11l; 12l |] in
  let result = Nd.rem_scalar t 3l in
  let expected = Nd.create Nd.int32 [| 3 |] [| 1l; 2l; 0l |] in
  Alcotest.(check nx_int32) "Remainder 1D int32 by scalar" expected result

let test_rem_inplace () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 10l; 11l; 12l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 3l; 5l; 4l |] in
  let t_res = Nd.rem_inplace t1 t2 in
  let expected = Nd.create Nd.int32 [| 3 |] [| 1l; 1l; 0l |] in
  Alcotest.(check nx_int32) "Remainder inplace int32" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_pow_float32 () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 3.0; 2.0; 0.5 |] in
  let result = Nd.pow t1 t2 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 8.0; 9.0; 2.0 |] in
  Alcotest.(check nx_float32) "Power float32" expected result

let test_pow_scalar () =
  let t = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  let result = Nd.pow_scalar t 2.0 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 4.0; 9.0; 16.0 |] in
  Alcotest.(check nx_float32) "Power 1D float32 by scalar" expected result

let test_pow_inplace () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 2.0; 1.0; 0.5 |] in
  let t_res = Nd.pow_inplace t1 t2 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 4.0; 3.0; 2.0 |] in
  Alcotest.(check nx_float32) "Power inplace float32" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_exp_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let result = Nd.exp t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.71828; 7.38906 |] in
  Alcotest.(check nx_float32) "Exponential float32" expected result

let test_log_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.71828; 7.38905 |] in
  let result = Nd.log t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  Alcotest.(check nx_float32) "Log float32" expected result

let test_log_non_positive () =
  let t = Nd.create Nd.float64 [| 3 |] [| 1.0; 0.0; -1.0 |] in
  let result = Nd.log t in
  check bool "NaN check" true (Float.is_nan (Nd.get_item [| 2 |] result))

let test_abs_int32 () =
  let t = Nd.create Nd.int32 [| 4 |] [| -1l; 0l; 5l; -10l |] in
  let result = Nd.abs t in
  let expected = Nd.create Nd.int32 [| 4 |] [| 1l; 0l; 5l; 10l |] in
  Alcotest.(check nx_int32) "Absolute value int32" expected result

let test_abs_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| -1.5; 0.0; 5.2 |] in
  let result = Nd.abs t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.5; 0.0; 5.2 |] in
  Alcotest.(check nx_float32) "Absolute value float32" expected result

let test_neg_float64 () =
  let t = Nd.create Nd.float64 [| 3 |] [| 1.0; -2.0; 0.0 |] in
  let result = Nd.neg t in
  let expected = Nd.create Nd.float64 [| 3 |] [| -1.0; 2.0; 0.0 |] in
  Alcotest.(check nx_float64) "Negation float64" expected result

let test_sign_float32 () =
  let t = Nd.create Nd.float32 [| 4 |] [| -5.0; 0.0; 3.2; -0.0 |] in
  let result = Nd.sign t in
  let expected = Nd.create Nd.float32 [| 4 |] [| -1.0; 0.0; 1.0; 0.0 |] in
  Alcotest.(check nx_float32) "Sign float32" expected result

let test_sign_int32 () =
  let t = Nd.create Nd.int32 [| 3 |] [| -5l; 0l; 3l |] in
  let result = Nd.sign t in
  let expected = Nd.create Nd.int32 [| 3 |] [| -1l; 0l; 1l |] in
  Alcotest.(check nx_int32) "Sign int32" expected result

let test_sqrt_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 4.0; 9.0; 16.0 |] in
  let result = Nd.sqrt t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  Alcotest.(check nx_float32) "Square root float32" expected result

let test_sqrt_negative () =
  let t = Nd.create Nd.float32 [| 3 |] [| 4.0; -9.0; 16.0 |] in
  let result = Nd.sqrt t in
  check bool "NaN check" true (Float.is_nan (Nd.get_item [| 1 |] result))

let test_square_float32 () =
  let t = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; -3.0; 0.0 |] in
  let result = Nd.square t in
  let expected = Nd.create Nd.float32 [| 4 |] [| 1.0; 4.0; 9.0; 0.0 |] in
  Alcotest.(check nx_float32) "Square float32" expected result

let test_maximum_float32 () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 1.0; 3.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
  let result = Nd.maximum t1 t2 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  Alcotest.(check nx_float32) "Maximum float32" expected result

let test_maximum_scalar () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 5.0; 3.0 |] in
  let result = Nd.maximum_scalar t 2.0 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 5.0; 3.0 |] in
  Alcotest.(check nx_float32) "Maximum 1D float32 with scalar" expected result

let test_maximum_inplace () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 1.0; 3.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
  let t_res = Nd.maximum_inplace t1 t2 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  Alcotest.(check nx_float32) "Maximum inplace float32" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_minimum_int32 () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 1l; 3l; 4l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 2l; 1l; 4l |] in
  let result = Nd.minimum t1 t2 in
  let expected = Nd.create Nd.int32 [| 3 |] [| 1l; 1l; 4l |] in
  Alcotest.(check nx_int32) "Minimum int32" expected result

let test_minimum_scalar () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 5.0; 3.0 |] in
  let result = Nd.minimum_scalar t 2.0 in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 2.0 |] in
  Alcotest.(check nx_float32) "Minimum 1D float32 with scalar" expected result

let test_minimum_inplace () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 1l; 3l; 4l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 2l; 1l; 4l |] in
  let t_res = Nd.minimum_inplace t1 t2 in
  let expected = Nd.create Nd.int32 [| 3 |] [| 1l; 1l; 4l |] in
  Alcotest.(check nx_int32) "Minimum inplace int32" expected t1;
  Alcotest.(check bool) "Returned tensor is t1" true (t_res == t1)

let test_equal_float32 () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 1.0; 5.0; 3.0 |] in
  let result = Nd.equal t1 t2 in
  let expected = Nd.create Nd.uint8 [| 3 |] [| 1; 0; 1 |] in
  Alcotest.(check nx_uint8) "Equal float32" expected result

let test_greater_float32 () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 1.0; 3.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
  let result = Nd.greater t1 t2 in
  let expected = Nd.create Nd.uint8 [| 3 |] [| 0; 1; 0 |] in
  Alcotest.(check nx_uint8) "Greater float32" expected result

let test_greater_equal_int32 () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 1l; 3l; 4l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 2l; 3l; 3l |] in
  let result = Nd.greater_equal t1 t2 in
  let expected = Nd.create Nd.uint8 [| 3 |] [| 0; 1; 1 |] in
  Alcotest.(check nx_uint8) "Greater equal int32" expected result

let test_less_float32 () =
  let t1 = Nd.create Nd.float32 [| 3 |] [| 1.0; 3.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
  let result = Nd.less t1 t2 in
  let expected = Nd.create Nd.uint8 [| 3 |] [| 1; 0; 1 |] in
  Alcotest.(check nx_uint8) "Less float32" expected result

let test_less_equal_int32 () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 1l; 3l; 4l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 2l; 3l; 3l |] in
  let result = Nd.less_equal t1 t2 in
  let expected = Nd.create Nd.uint8 [| 3 |] [| 1; 1; 0 |] in
  Alcotest.(check nx_uint8) "Less equal int32" expected result

let test_sin_float32 () =
  let t =
    Nd.create Nd.float32 [| 4 |]
      [| 0.0; Float.pi /. 6.0; Float.pi /. 4.0; Float.pi /. 2.0 |]
  in
  let result = Nd.sin t in
  let expected = Nd.create Nd.float32 [| 4 |] [| 0.0; 0.5; 0.707107; 1.0 |] in
  Alcotest.(check nx_float32) "Sine float32" expected result

let test_cos_float32 () =
  let t =
    Nd.create Nd.float32 [| 4 |]
      [| 0.0; Float.pi /. 6.0; Float.pi /. 4.0; Float.pi /. 2.0 |]
  in
  let result = Nd.cos t in
  let expected =
    Nd.create Nd.float32 [| 4 |] [| 1.0; 0.866025; 0.707107; 0.0 |]
  in
  Alcotest.(check nx_float32) "Cosine float32" expected result

let test_tan_float32 () =
  let t =
    Nd.create Nd.float32 [| 3 |] [| 0.0; Float.pi /. 6.0; Float.pi /. 4.0 |]
  in
  let result = Nd.tan t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 0.0; 0.577350; 1.0 |] in
  Alcotest.(check nx_float32) "Tangent float32" expected result

let test_asin_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 0.0; 0.5; 1.0 |] in
  let result = Nd.asin t in
  let expected =
    Nd.create Nd.float32 [| 3 |] [| 0.0; Float.pi /. 6.0; Float.pi /. 2.0 |]
  in
  Alcotest.(check nx_float32) "Arcsine float32" expected result

let test_acos_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 0.5; 0.0 |] in
  let result = Nd.acos t in
  let expected =
    Nd.create Nd.float32 [| 3 |] [| 0.0; Float.pi /. 3.0; Float.pi /. 2.0 |]
  in
  Alcotest.(check nx_float32) "Arccosine float32" expected result

let test_atan_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 0.0; 1.0; -1.0 |] in
  let result = Nd.atan t in
  let expected =
    Nd.create Nd.float32 [| 3 |] [| 0.0; Float.pi /. 4.0; -.Float.pi /. 4.0 |]
  in
  Alcotest.(check nx_float32) "Arctangent float32" expected result

let test_sinh_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let result = Nd.sinh t in
  let expected = Nd.create Nd.float32 [| 3 |] [| -1.17520; 0.0; 1.17520 |] in
  Alcotest.(check nx_float32) "Hyperbolic Sine float32" expected result

let test_cosh_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let result = Nd.cosh t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.54308; 1.0; 1.54308 |] in
  Alcotest.(check nx_float32) "Hyperbolic Cosine float32" expected result

let test_tanh_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let result = Nd.tanh t in
  let expected = Nd.create Nd.float32 [| 3 |] [| -0.761594; 0.0; 0.761594 |] in
  Alcotest.(check nx_float32) "Hyperbolic Tangent float32" expected result

let test_asinh_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| -1.0; 0.0; 2.0 |] in
  let result = Nd.asinh t in
  let expected = Nd.create Nd.float32 [| 3 |] [| -0.881374; 0.0; 1.44364 |] in
  Alcotest.(check nx_float32) "Inverse Hyperbolic Sine float32" expected result

let test_acosh_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let result = Nd.acosh t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 0.0; 1.31696; 1.76275 |] in
  Alcotest.(check nx_float32)
    "Inverse Hyperbolic Cosine float32" expected result

let test_atanh_float32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| -0.5; 0.0; 0.5 |] in
  let result = Nd.atanh t in
  let expected = Nd.create Nd.float32 [| 3 |] [| -0.549306; 0.0; 0.549306 |] in
  Alcotest.(check nx_float32)
    "Inverse Hyperbolic Tangent float32" expected result

let test_broadcast_add_1d_to_2d () =
  let t2d =
    Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let t1d = Nd.create Nd.float32 [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let result = Nd.add t2d t1d in
  let expected =
    Nd.create Nd.float32 [| 2; 3 |] [| 11.0; 22.0; 33.0; 14.0; 25.0; 36.0 |]
  in
  Alcotest.(check nx_float32) "Broadcasting: Add 1D to 2D" expected result

let test_broadcast_add_2x1_to_2x3 () =
  let t21 = Nd.create Nd.float32 [| 2; 1 |] [| 1.0; 2.0 |] in
  let t23 =
    Nd.create Nd.float32 [| 2; 3 |] [| 10.0; 11.0; 12.0; 20.0; 21.0; 22.0 |]
  in
  let result = Nd.add t21 t23 in
  let expected =
    Nd.create Nd.float32 [| 2; 3 |] [| 11.0; 12.0; 13.0; 22.0; 23.0; 24.0 |]
  in
  Alcotest.(check nx_float32) "Broadcasting: Add 2x1 to 2x3" expected result

let test_broadcast_mul_4x1_3 () =
  let t41 =
    Nd.reshape [| 4; 1 |]
      (Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |])
  in
  let t3 = Nd.create Nd.float32 [| 3 |] [| 10.0; 100.0; 1000.0 |] in
  let result = Nd.mul t41 t3 in
  let expected =
    Nd.create Nd.float32 [| 4; 3 |]
      [|
        10.0;
        100.0;
        1000.0;
        20.0;
        200.0;
        2000.0;
        30.0;
        300.0;
        3000.0;
        40.0;
        400.0;
        4000.0;
      |]
  in
  Alcotest.(check nx_float32)
    "Broadcasting: Multiply (4,1) * (3,)" expected result

let test_add_scalar_to_1d () =
  let s = Nd.scalar Nd.float32 2.0 in
  let t1d = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let result = Nd.add s t1d in
  let expected = Nd.create Nd.float32 [| 3 |] [| 3.0; 4.0; 5.0 |] in
  Alcotest.(check nx_float32) "Add scalar to 1D array" expected result

let test_sum_2x2_all () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let result = Nd.sum t in
  let expected = Nd.scalar Nd.float32 10.0 in
  Alcotest.(check nx_float32) "Sum of 2x2 array (all)" expected result

let test_sum_2x3_axis0 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let result = Nd.sum ~axes:[| 0 |] t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 5.0; 7.0; 9.0 |] in
  Alcotest.(check nx_float32) "Sum of 2x3 array axis=0" expected result

let test_sum_2x3_axis1_keepdims () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let result = Nd.sum ~axes:[| 1 |] ~keepdims:true t in
  let expected = Nd.create Nd.float32 [| 2; 1 |] [| 6.0; 15.0 |] in
  Alcotest.(check nx_float32) "Sum of 2x3 array axis=1 keepdims" expected result

let test_prod_1d_int32 () =
  let t = Nd.create Nd.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
  let result = Nd.prod t in
  let expected = Nd.scalar Nd.int32 24l in
  Alcotest.(check nx_int32) "Product of 1D array int32" expected result

let test_prod_2x3_axis1 () =
  let t = Nd.create Nd.int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let result = Nd.prod ~axes:[| 1 |] t in
  let expected = Nd.create Nd.int32 [| 2 |] [| 6l; 120l |] in
  Alcotest.(check nx_int32) "Product of 2x3 axis=1" expected result

let test_mean_2x2_all () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let result = Nd.mean t in
  let expected = Nd.scalar Nd.float32 2.5 in
  Alcotest.(check nx_float32) "Mean of 2x2 array (all)" expected result

let test_mean_2x3_axis0_keepdims () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let result = Nd.mean ~axes:[| 0 |] ~keepdims:true t in
  let expected = Nd.create Nd.float32 [| 1; 3 |] [| 2.5; 3.5; 4.5 |] in
  Alcotest.(check nx_float32)
    "Mean of 2x3 array axis=0 keepdims" expected result

let test_max_2x2_all () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let result = Nd.max t in
  let expected = Nd.scalar Nd.float32 4.0 in
  Alcotest.(check nx_float32) "Max of 2x2 array (all)" expected result

let test_max_2x3_axis1 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 5.0; 3.0; 4.0; 2.0; 6.0 |] in
  let result = Nd.max ~axes:[| 1 |] t in
  let expected = Nd.create Nd.float32 [| 2 |] [| 5.0; 6.0 |] in
  Alcotest.(check nx_float32) "Max of 2x3 array axis=1" expected result

let test_min_2x2_all () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let result = Nd.min t in
  let expected = Nd.scalar Nd.float32 1.0 in
  Alcotest.(check nx_float32) "Min of 2x2 array (all)" expected result

let test_min_2x3_axis0_keepdims () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 5.0; 3.0; 4.0; 2.0; 6.0 |] in
  let result = Nd.min ~axes:[| 0 |] ~keepdims:true t in
  let expected = Nd.create Nd.float32 [| 1; 3 |] [| 1.0; 2.0; 3.0 |] in
  Alcotest.(check nx_float32) "Min of 2x3 array axis=0 keepdims" expected result

let test_var_1d () =
  let t = Nd.create Nd.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let result = Nd.var t in
  let expected = Nd.scalar Nd.float32 2.0 in
  Alcotest.(check nx_float32) "Variance of 1D array" expected result

let test_var_2x3_axis1 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let result = Nd.var ~axes:[| 1 |] t in
  let expected = Nd.create Nd.float32 [| 2 |] [| 0.666667; 0.666667 |] in
  Alcotest.(check nx_float32) "Variance of 2x3 axis=1" expected result

let test_std_1d () =
  let t = Nd.create Nd.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let result = Nd.std t in
  let expected = Nd.scalar Nd.float32 1.41421 in
  Alcotest.(check nx_float32) "Standard Deviation of 1D array" expected result

let test_std_2x3_axis0_keepdims () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let result = Nd.std ~axes:[| 0 |] ~keepdims:true t in
  let expected = Nd.create Nd.float32 [| 1; 3 |] [| 1.5; 1.5; 1.5 |] in
  Alcotest.(check nx_float32)
    "Standard Deviation of 2x3 axis=0 keepdims" expected result

let test_fma () =
  let a = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  let b = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let c = Nd.create Nd.float32 [| 3 |] [| 5.0; 6.0; 7.0 |] in
  let result = Nd.fma a b c in
  let expected = Nd.create Nd.float32 [| 3 |] [| 7.0; 12.0; 19.0 |] in
  Alcotest.(check nx_float32) "FMA float32" expected result

let test_fma_inplace () =
  let a = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  let b = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let c = Nd.create Nd.float32 [| 3 |] [| 5.0; 6.0; 7.0 |] in
  let t_res = Nd.fma_inplace a b c in
  let expected = Nd.create Nd.float32 [| 3 |] [| 7.0; 12.0; 19.0 |] in
  Alcotest.(check nx_float32) "FMA inplace float32" expected a;
  Alcotest.(check bool) "Returned tensor is a" true (t_res == a)

let test_round () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.6; 2.4; -1.7 |] in
  let result = Nd.round t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 2.0; -2.0 |] in
  Alcotest.(check nx_float32) "Round float32" expected result

let test_floor () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.6; 2.4; -1.7 |] in
  let result = Nd.floor t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; -2.0 |] in
  Alcotest.(check nx_float32) "Floor float32" expected result

let test_ceil () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.6; 2.4; -1.7 |] in
  let result = Nd.ceil t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; -1.0 |] in
  Alcotest.(check nx_float32) "Ceil float32" expected result

let test_clip () =
  let t = Nd.create Nd.float32 [| 5 |] [| -1.0; 2.0; 5.0; 8.0; 10.0 |] in
  let result = Nd.clip ~min:0.0 ~max:7.0 t in
  let expected = Nd.create Nd.float32 [| 5 |] [| 0.0; 2.0; 5.0; 7.0; 7.0 |] in
  Alcotest.(check nx_float32) "Clip float32" expected result

let test_bitwise_and () =
  let t1 = Nd.create Nd.int16 [| 3 |] [| 5; 3; 7 |] in
  (* 101, 011, 111 *)
  let t2 = Nd.create Nd.int16 [| 3 |] [| 3; 5; 6 |] in
  (* 011, 101, 110 *)
  let result = Nd.bitwise_and t1 t2 in
  let expected = Nd.create Nd.int16 [| 3 |] [| 1; 1; 6 |] in
  (* 001, 001, 110 *)
  Alcotest.(check nx_int16) "Bitwise AND int16" expected result

let test_bitwise_or () =
  let t1 = Nd.create Nd.int16 [| 3 |] [| 5; 3; 7 |] in
  (* 101, 011, 111 *)
  let t2 = Nd.create Nd.int16 [| 3 |] [| 3; 5; 6 |] in
  (* 011, 101, 110 *)
  let result = Nd.bitwise_or t1 t2 in
  let expected = Nd.create Nd.int16 [| 3 |] [| 7; 7; 7 |] in
  (* 111, 111, 111 *)
  Alcotest.(check nx_int16) "Bitwise OR int16" expected result

let test_bitwise_xor () =
  let t1 = Nd.create Nd.int16 [| 3 |] [| 5; 3; 7 |] in
  (* 101, 011, 111 *)
  let t2 = Nd.create Nd.int16 [| 3 |] [| 3; 5; 6 |] in
  (* 011, 101, 110 *)
  let result = Nd.bitwise_xor t1 t2 in
  let expected = Nd.create Nd.int16 [| 3 |] [| 6; 6; 1 |] in
  (* 110, 110, 001 *)
  Alcotest.(check nx_int16) "Bitwise XOR int16" expected result

let test_invert () =
  let t = Nd.create Nd.int16 [| 3 |] [| 5; 0; 7 |] in
  (* 101, 000, 111 *)
  let result = Nd.invert t in
  let expected = Nd.create Nd.int16 [| 3 |] [| -6; -1; -8 |] in
  (* ~101, ~000, ~111 *)
  Alcotest.(check nx_int16) "Invert int16" expected result

let element_wise_tests =
  [
    ("Add two 2x2 float32", `Quick, test_add_two_2x2_float32);
    ("Add inplace 2x2 float32", `Quick, test_add_inplace_2x2_float32);
    ("Add two 2x2 int32", `Quick, test_add_two_2x2_int32);
    ("Add 1D float32 by scalar", `Quick, test_add_scalar);
    ("Multiply two 2x2 float32", `Quick, test_mul_two_2x2_float32);
    ("Multiply inplace 1D int32", `Quick, test_mul_inplace_1d_int32);
    ("Multiply 1D float32 by scalar", `Quick, test_mul_scalar);
    ("Subtract two 2x2 float32", `Quick, test_sub_two_2x2_float32);
    ( "Subtract inplace 1D->2D broadcast",
      `Quick,
      test_sub_inplace_1d_to_2d_broadcast );
    ("Subtract scalar from 1D float32", `Quick, test_sub_scalar);
    ("Divide two 2x2 float32", `Quick, test_div_two_2x2_float32);
    ( "Divide inplace scalar broadcast",
      `Quick,
      test_div_inplace_scalar_broadcast );
    ("Divide two 2x2 int32", `Quick, test_div_two_2x2_int32);
    ("Divide 1D float32 by scalar", `Quick, test_div_scalar);
    ("Remainder int32", `Quick, test_rem_int32);
    ("Remainder 1D int32 by scalar", `Quick, test_rem_scalar);
    ("Remainder inplace int32", `Quick, test_rem_inplace);
    ("Power float32", `Quick, test_pow_float32);
    ("Power 1D float32 by scalar", `Quick, test_pow_scalar);
    ("Power inplace float32", `Quick, test_pow_inplace);
    ("Exponential float32", `Quick, test_exp_float32);
    ("Log float32", `Quick, test_log_float32);
    ("Log of non-positive", `Quick, test_log_non_positive);
    ("Absolute value int32", `Quick, test_abs_int32);
    ("Absolute value float32", `Quick, test_abs_float32);
    ("Negation float64", `Quick, test_neg_float64);
    ("Sign float32", `Quick, test_sign_float32);
    ("Sign int32", `Quick, test_sign_int32);
    ("Square root float32", `Quick, test_sqrt_float32);
    ("Square root of negative", `Quick, test_sqrt_negative);
    ("Maximum float32", `Quick, test_maximum_float32);
    ("Maximum 1D float32 with scalar", `Quick, test_maximum_scalar);
    ("Maximum inplace float32", `Quick, test_maximum_inplace);
    ("Minimum int32", `Quick, test_minimum_int32);
    ("Minimum 1D float32 with scalar", `Quick, test_minimum_scalar);
    ("Minimum inplace int32", `Quick, test_minimum_inplace);
    ("Equal float32", `Quick, test_equal_float32);
    ("Greater float32", `Quick, test_greater_float32);
    ("Greater equal int32", `Quick, test_greater_equal_int32);
    ("Less float32", `Quick, test_less_float32);
    ("Less equal int32", `Quick, test_less_equal_int32);
    ("Square float32", `Quick, test_square_float32);
    ("FMA float32", `Quick, test_fma);
    ("FMA inplace float32", `Quick, test_fma_inplace);
    ("Round float32", `Quick, test_round);
    ("Floor float32", `Quick, test_floor);
    ("Ceil float32", `Quick, test_ceil);
    ("Clip float32", `Quick, test_clip);
  ]

let trig_hyperbolic_tests =
  [
    ("Sine float32", `Quick, test_sin_float32);
    ("Cosine float32", `Quick, test_cos_float32);
    ("Tangent float32", `Quick, test_tan_float32);
    ("Arcsine float32", `Quick, test_asin_float32);
    ("Arccosine float32", `Quick, test_acos_float32);
    ("Arctangent float32", `Quick, test_atan_float32);
    ("Hyperbolic Sine float32", `Quick, test_sinh_float32);
    ("Hyperbolic Cosine float32", `Quick, test_cosh_float32);
    ("Hyperbolic Tangent float32", `Quick, test_tanh_float32);
    ("Inverse Hyperbolic Sine float32", `Quick, test_asinh_float32);
    ("Inverse Hyperbolic Cosine float32", `Quick, test_acosh_float32);
    ("Inverse Hyperbolic Tangent float32", `Quick, test_atanh_float32);
  ]

let broadcasting_tests =
  [
    ("Broadcasting: Add 1D to 2D", `Quick, test_broadcast_add_1d_to_2d);
    ("Broadcasting: Add 2x1 to 2x3", `Quick, test_broadcast_add_2x1_to_2x3);
    ("Broadcasting: Multiply (4,1) * (3,)", `Quick, test_broadcast_mul_4x1_3);
    ("Add scalar to 1D array", `Quick, test_add_scalar_to_1d);
  ]

let reduction_tests =
  [
    ("Sum of 2x2 array (all)", `Quick, test_sum_2x2_all);
    ("Sum of 2x3 array axis=0", `Quick, test_sum_2x3_axis0);
    ("Sum of 2x3 array axis=1 keepdims", `Quick, test_sum_2x3_axis1_keepdims);
    ("Product of 1D array int32", `Quick, test_prod_1d_int32);
    ("Product of 2x3 axis=1", `Quick, test_prod_2x3_axis1);
    ("Mean of 2x2 array (all)", `Quick, test_mean_2x2_all);
    ("Mean of 2x3 array axis=0 keepdims", `Quick, test_mean_2x3_axis0_keepdims);
    ("Max of 2x2 array (all)", `Quick, test_max_2x2_all);
    ("Max of 2x3 array axis=1", `Quick, test_max_2x3_axis1);
    ("Min of 2x2 array (all)", `Quick, test_min_2x2_all);
    ("Min of 2x3 array axis=0 keepdims", `Quick, test_min_2x3_axis0_keepdims);
    ("Variance of 1D array", `Quick, test_var_1d);
    ("Variance of 2x3 axis=1", `Quick, test_var_2x3_axis1);
    ("Standard Deviation of 1D array", `Quick, test_std_1d);
    ( "Standard Deviation of 2x3 axis=0 keepdims",
      `Quick,
      test_std_2x3_axis0_keepdims );
  ]

let bitwise_tests =
  [
    ("Bitwise AND int16", `Quick, test_bitwise_and);
    ("Bitwise OR int16", `Quick, test_bitwise_or);
    ("Bitwise XOR int16", `Quick, test_bitwise_xor);
    ("Invert int16", `Quick, test_invert);
  ]

let () =
  Printexc.record_backtrace true;
  Alcotest.run "Nx Operations"
    [
      ("Element-wise Operations", element_wise_tests);
      ("Trigonometric and Hyperbolic", trig_hyperbolic_tests);
      ("Broadcasting", broadcasting_tests);
      ("Reductions", reduction_tests);
      ("Bitwise Operations", bitwise_tests);
    ]
