(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sanity tests for Nx - quick smoke test for every API function *)

open Windtrap
open Test_nx_support

(* Test helper to create simple test data *)
let test_array = [| 1.; 2.; 3.; 4.; 5.; 6. |]
let shape_2x3 = [| 2; 3 |]

let creation_tests =
  [
    test "create" (fun () ->
        Nx.create Nx.float32 shape_2x3 test_array
        |> check_t "create" shape_2x3 test_array);
    test "init" (fun () ->
        let t =
          Nx.init Nx.float32 [| 2; 2 |] (fun indices ->
              float_of_int (indices.(0) + indices.(1)))
        in
        check_t "init" [| 2; 2 |] [| 0.; 1.; 1.; 2. |] t);
    test "empty" (fun () ->
        Nx.empty Nx.float32 shape_2x3 |> check_shape "empty shape" shape_2x3);
    test "full" (fun () ->
        Nx.full Nx.float32 shape_2x3 7.0
        |> check_t "full" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |]);
    test "ones" (fun () ->
        Nx.ones Nx.float32 shape_2x3
        |> check_t "ones" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test "zeros" (fun () ->
        Nx.zeros Nx.float32 shape_2x3
        |> check_t "zeros" shape_2x3 [| 0.; 0.; 0.; 0.; 0.; 0. |]);
    test "ones_like" (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.ones_like ref_t
        |> check_t "ones_like" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test "zeros_like" (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.zeros_like ref_t
        |> check_t "zeros_like" shape_2x3 [| 0.; 0.; 0.; 0.; 0.; 0. |]);
    test "empty_like" (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.empty_like ref_t |> check_shape "empty_like shape" shape_2x3);
    test "full_like" (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.full_like ref_t 9.0
        |> check_t "full_like" shape_2x3 [| 9.; 9.; 9.; 9.; 9.; 9. |]);
    test "scalar" (fun () ->
        Nx.scalar Nx.float32 42.0 |> check_t "scalar" [||] [| 42.0 |]);
    test "scalar_like" (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.scalar_like ref_t 5.0 |> check_t "scalar_like" [||] [| 5.0 |]);
    test "eye" (fun () ->
        Nx.eye Nx.float32 3
        |> check_t "eye" [| 3; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]);
    test "identity" (fun () ->
        Nx.identity Nx.float32 3
        |> check_t "identity" [| 3; 3 |]
             [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]);
    test "copy" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.copy t |> check_t "copy" shape_2x3 test_array);
    test "contiguous" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.contiguous t |> check_t "contiguous" shape_2x3 test_array);
  ]

let range_generation_tests =
  [
    test "arange" (fun () ->
        let t = Nx.arange Nx.int32 0 5 1 in
        check_t "arange" [| 5 |] [| 0l; 1l; 2l; 3l; 4l |] t);
    test "arange_f" (fun () ->
        Nx.arange_f Nx.float32 0.0 1.0 0.25
        |> check_t "arange_f" [| 4 |] [| 0.0; 0.25; 0.5; 0.75 |]);
    test "linspace" (fun () ->
        let t = Nx.linspace Nx.float32 0.0 1.0 5 in
        check_t ~eps:1e-6 "linspace" [| 5 |] [| 0.0; 0.25; 0.5; 0.75; 1.0 |] t);
    test "logspace" (fun () ->
        let t = Nx.logspace Nx.float32 0.0 2.0 3 in
        check_t ~eps:1e-4 "logspace" [| 3 |] [| 1.0; 10.0; 100.0 |] t);
    test "geomspace" (fun () ->
        let t = Nx.geomspace Nx.float32 1.0 100.0 3 in
        check_t ~eps:1e-4 "geomspace" [| 3 |] [| 1.0; 10.0; 100.0 |] t);
  ]

let property_access_tests =
  [
    test "data" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        let data = Nx.data t in
        equal ~msg:"data[0]" (float 1e-6) 1.0 (Nx_buffer.get data 0);
        equal ~msg:"data[5]" (float 1e-6) 6.0 (Nx_buffer.get data 5));
    test "shape" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"shape" (array int) shape_2x3 (Nx.shape t));
    test "dtype" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"dtype is float32" bool true (Nx.dtype t = Nx.float32));
    test "strides" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        let strides = Nx.strides t in
        equal ~msg:"strides length" int 2 (Array.length strides);
        equal ~msg:"stride 0" int 12 strides.(0);
        equal ~msg:"stride 1" int 4 strides.(1));
    test "stride" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"stride 0" int 12 (Nx.stride 0 t);
        equal ~msg:"stride 1" int 4 (Nx.stride 1 t));
    test "dims" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        let d = Nx.dims t in
        equal ~msg:"dims length" int 2 (Array.length d);
        equal ~msg:"dims[0]" int 2 d.(0);
        equal ~msg:"dims[1]" int 3 d.(1));
    test "dim" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"dim 0" int 2 (Nx.dim 0 t);
        equal ~msg:"dim 1" int 3 (Nx.dim 1 t));
    test "ndim" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"ndim" int 2 (Nx.ndim t));
    test "itemsize" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"itemsize" int 4 (Nx.itemsize t));
    test "size" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"size" int 6 (Nx.size t));
    test "numel" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"numel" int 6 (Nx.numel t));
    test "nbytes" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"nbytes" int 24 (Nx.nbytes t));
    test "offset" (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"offset" int 0 (Nx.offset t));
  ]

let data_manipulation_tests =
  [
    test "blit" (fun () ->
        let src = Nx.ones Nx.float32 shape_2x3 in
        let dst = Nx.zeros Nx.float32 shape_2x3 in
        Nx.blit src dst;
        check_t "blit" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |] dst);
    test "fill copy" (fun () ->
        let t = Nx.zeros Nx.float32 shape_2x3 in
        let filled = Nx.fill 5.0 t in
        check_t "fill copy" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] filled;
        check_t "fill leaves source" shape_2x3 [| 0.; 0.; 0.; 0.; 0.; 0. |] t);
    test "ifill" (fun () ->
        let t = Nx.zeros Nx.float32 shape_2x3 in
        ignore (Nx.ifill 5.0 t);
        check_t "ifill" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] t);
  ]

let element_wise_binary_tests =
  [
    test "add" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.add a b |> check_t "add" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test "add_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.add_s a 5.0 |> check_t "add_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test "radd_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.radd_s 5.0 a
        |> check_t "radd_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test "iadd" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.iadd a b in
        check_t "iadd" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] a);
    test "iadd_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.iadd_s a 5.0 in
        check_t "iadd_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |] a);
    test "sub" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.sub a b |> check_t "sub" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test "sub_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 10.0 in
        Nx.sub_s a 3.0 |> check_t "sub_s" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |]);
    test "rsub_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rsub_s 10.0 a
        |> check_t "rsub_s" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |]);
    test "isub" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.isub a b in
        check_t "isub" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
    test "isub_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 10.0 in
        let _ = Nx.isub_s a 3.0 in
        check_t "isub_s" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |] a);
    test "mul" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.mul a b |> check_t "mul" shape_2x3 [| 6.; 6.; 6.; 6.; 6.; 6. |]);
    test "mul_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 4.0 in
        Nx.mul_s a 3.0
        |> check_t "mul_s" shape_2x3 [| 12.; 12.; 12.; 12.; 12.; 12. |]);
    test "rmul_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 4.0 in
        Nx.rmul_s 3.0 a
        |> check_t "rmul_s" shape_2x3 [| 12.; 12.; 12.; 12.; 12.; 12. |]);
    test "imul" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.imul a b in
        check_t "imul" shape_2x3 [| 6.; 6.; 6.; 6.; 6.; 6. |] a);
    test "imul_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 4.0 in
        let _ = Nx.imul_s a 3.0 in
        check_t "imul_s" shape_2x3 [| 12.; 12.; 12.; 12.; 12.; 12. |] a);
    test "div" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 6.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.div a b |> check_t "div" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test "div_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 12.0 in
        Nx.div_s a 3.0 |> check_t "div_s" shape_2x3 [| 4.; 4.; 4.; 4.; 4.; 4. |]);
    test "rdiv_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.rdiv_s 6.0 a
        |> check_t "rdiv_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test "idiv" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 6.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.idiv a b in
        check_t "idiv" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
    test "idiv_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 12.0 in
        let _ = Nx.idiv_s a 3.0 in
        check_t "idiv_s" shape_2x3 [| 4.; 4.; 4.; 4.; 4.; 4. |] a);
    test "pow" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.pow a b |> check_t "pow" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test "pow_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.pow_s a 3.0 |> check_t "pow_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test "rpow_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rpow_s 2.0 a
        |> check_t "rpow_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test "ipow" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.ipow a b in
        check_t "ipow" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |] a);
    test "ipow_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.ipow_s a 3.0 in
        check_t "ipow_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |] a);
    test "mod" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.mod_ a b |> check_t "mod" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test "mod_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        Nx.mod_s a 3.0 |> check_t "mod_s" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test "rmod_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rmod_s 7.0 a
        |> check_t "rmod_s" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test "imod" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.imod a b in
        check_t "imod" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |] a);
    test "imod_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        let _ = Nx.imod_s a 3.0 in
        check_t "imod_s" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |] a);
    test "maximum" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.maximum a b
        |> check_t "maximum" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test "maximum_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.maximum_s a 5.0
        |> check_t "maximum_s" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test "rmaximum_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rmaximum_s 5.0 a
        |> check_t "rmaximum_s" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test "imaximum" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 5.0 in
        let _ = Nx.imaximum a b in
        check_t "imaximum" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] a);
    test "imaximum_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.imaximum_s a 5.0 in
        check_t "imaximum_s" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] a);
    test "minimum" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.minimum a b
        |> check_t "minimum" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test "minimum_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.minimum_s a 3.0
        |> check_t "minimum_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test "rminimum_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.rminimum_s 3.0 a
        |> check_t "rminimum_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test "iminimum" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.iminimum a b in
        check_t "iminimum" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
    test "iminimum_s" (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let _ = Nx.iminimum_s a 3.0 in
        check_t "iminimum_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
  ]

let comparison_tests =
  [
    test "equal" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 1. |] in
        Nx.equal a b |> check_t "equal" [| 3 |] [| true; false; true |]);
    test "not_equal" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 1. |] in
        Nx.not_equal a b |> check_t "not_equal" [| 3 |] [| false; true; false |]);
    test "greater" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 5.; 3.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.greater a b |> check_t "greater" [| 3 |] [| true; false; false |]);
    test "greater_equal" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 5.; 3.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.greater_equal a b
        |> check_t "greater_equal" [| 3 |] [| true; true; false |]);
    test "less" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 5. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.less a b |> check_t "less" [| 3 |] [| true; false; false |]);
    test "less_equal" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 5. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.less_equal a b
        |> check_t "less_equal" [| 3 |] [| true; true; false |]);
  ]

let element_wise_unary_tests =
  [
    test "neg" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; -2.; 3. |] in
        Nx.neg a |> check_t "neg" [| 3 |] [| -1.; 2.; -3. |]);
    test "abs" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| -3.; 0.; 5. |] in
        Nx.abs a |> check_t "abs" [| 3 |] [| 3.; 0.; 5. |]);
    test "sign" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| -5.; 0.; 3.; -0. |] in
        Nx.sign a |> check_t "sign" [| 4 |] [| -1.; 0.; 1.; 0. |]);
    test "square" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| -2.; 3.; 4. |] in
        Nx.square a |> check_t "square" [| 3 |] [| 4.; 9.; 16. |]);
    test "sqrt" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 4.; 9.; 16. |] in
        Nx.sqrt a |> check_t "sqrt" [| 3 |] [| 2.; 3.; 4. |]);
    test "rsqrt" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 4.; 16. |] in
        Nx.rsqrt a |> check_t "rsqrt" [| 3 |] [| 1.0; 0.5; 0.25 |]);
    test "recip" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 4. |] in
        Nx.recip a |> check_t "recip" [| 3 |] [| 1.0; 0.5; 0.25 |]);
    test "exp" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 2. |] in
        Nx.exp a
        |> check_t ~eps:1e-6 "exp" [| 3 |] [| 1.0; 2.718282; 7.389056 |]);
    test "exp2" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 3. |] in
        Nx.exp2 a |> check_t "exp2" [| 3 |] [| 1.; 2.; 8. |]);
    test "log" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2.718282 |] in
        Nx.log a |> check_t ~eps:1e-6 "log" [| 2 |] [| 0.0; 1.0 |]);
    test "log2" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 8. |] in
        Nx.log2 a |> check_t "log2" [| 3 |] [| 0.; 1.; 3. |]);
    test "sin" (fun () ->
        let pi = 3.14159265359 in
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; pi /. 2.; pi |] in
        Nx.sin a |> check_t ~eps:1e-6 "sin" [| 3 |] [| 0.0; 1.0; 0.0 |]);
    test "cos" (fun () ->
        let pi = 3.14159265359 in
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; pi /. 2.; pi |] in
        Nx.cos a |> check_t ~eps:1e-6 "cos" [| 3 |] [| 1.0; 0.0; -1.0 |]);
    test "tan" (fun () ->
        let pi = 3.14159265359 in
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; pi /. 4. |] in
        Nx.tan a |> check_t ~eps:1e-6 "tan" [| 2 |] [| 0.0; 1.0 |]);
    test "asin" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 0.5; 1. |] in
        Nx.asin a
        |> check_t ~eps:1e-6 "asin" [| 3 |] [| 0.0; 0.523599; 1.570796 |]);
    test "acos" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 0.5; 0. |] in
        Nx.acos a
        |> check_t ~eps:1e-6 "acos" [| 3 |] [| 0.0; 1.047198; 1.570796 |]);
    test "atan" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; -1. |] in
        Nx.atan a
        |> check_t ~eps:1e-6 "atan" [| 3 |] [| 0.0; 0.785398; -0.785398 |]);
    test "sinh" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; 1. |] in
        Nx.sinh a |> check_t ~eps:1e-6 "sinh" [| 2 |] [| 0.0; 1.175201 |]);
    test "cosh" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; 1. |] in
        Nx.cosh a |> check_t ~eps:1e-6 "cosh" [| 2 |] [| 1.0; 1.543081 |]);
    test "tanh" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; -1. |] in
        Nx.tanh a
        |> check_t ~eps:1e-6 "tanh" [| 3 |] [| 0.0; 0.761594; -0.761594 |]);
    test "asinh" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; 1. |] in
        Nx.asinh a |> check_t ~eps:1e-6 "asinh" [| 2 |] [| 0.0; 0.881374 |]);
    test "acosh" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
        Nx.acosh a |> check_t ~eps:1e-6 "acosh" [| 2 |] [| 0.0; 1.316958 |]);
    test "atanh" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 0.5; -0.5 |] in
        Nx.atanh a
        |> check_t ~eps:1e-6 "atanh" [| 3 |] [| 0.0; 0.549306; -0.549306 |]);
    test "round" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.round a |> check_t "round" [| 4 |] [| 3.; 4.; -3.; -4. |]);
    test "floor" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.floor a |> check_t "floor" [| 4 |] [| 3.; 3.; -4.; -4. |]);
    test "ceil" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.ceil a |> check_t "ceil" [| 4 |] [| 4.; 4.; -3.; -3. |]);
    test "trunc" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.trunc a |> check_t "trunc" [| 4 |] [| 3.; 3.; -3.; -3. |]);
    test "clip" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| -2.; 0.; 5.; 10.; 12. |] in
        Nx.clip ~min:2.0 ~max:8.0 a
        |> check_t "clip" [| 5 |] [| 2.0; 2.0; 5.0; 8.0; 8.0 |]);
    test "clamp" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| -2.; 0.; 5.; 10.; 12. |] in
        Nx.clamp ~min:2.0 ~max:8.0 a
        |> check_t "clamp" [| 5 |] [| 2.0; 2.0; 5.0; 8.0; 8.0 |]);
    test "lerp" (fun () ->
        let start_t = Nx.zeros Nx.float32 [| 3 |] in
        let end_t = Nx.full Nx.float32 [| 3 |] 10.0 in
        let weight = Nx.create Nx.float32 [| 3 |] [| 0.0; 0.5; 1.0 |] in
        Nx.lerp start_t end_t weight
        |> check_t "lerp" [| 3 |] [| 0.0; 5.0; 10.0 |]);
    test "lerp_scalar_weight" (fun () ->
        let start_t = Nx.zeros Nx.float32 [| 3 |] in
        let end_t = Nx.full Nx.float32 [| 3 |] 10.0 in
        Nx.lerp_scalar_weight start_t end_t 0.3
        |> check_t "lerp_scalar_weight" [| 3 |] [| 3.0; 3.0; 3.0 |]);
  ]

let bitwise_tests =
  [
    test "bitwise_and" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 7l; 12l; 15l |] in
        let b = Nx.create Nx.int32 [| 3 |] [| 3l; 10l; 7l |] in
        Nx.bitwise_and a b |> check_t "bitwise_and" [| 3 |] [| 3l; 8l; 7l |]);
    test "bitwise_or" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 1l; 4l; 8l |] in
        let b = Nx.create Nx.int32 [| 3 |] [| 2l; 2l; 7l |] in
        Nx.bitwise_or a b |> check_t "bitwise_or" [| 3 |] [| 3l; 6l; 15l |]);
    test "bitwise_xor" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 7l; 12l; 15l |] in
        let b = Nx.create Nx.int32 [| 3 |] [| 3l; 10l; 7l |] in
        Nx.bitwise_xor a b |> check_t "bitwise_xor" [| 3 |] [| 4l; 6l; 8l |]);
    test "bitwise_not" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 1l; -1l |] in
        Nx.bitwise_not a |> check_t "bitwise_not" [| 3 |] [| -1l; -2l; 0l |]);
    test "invert" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 1l; -1l |] in
        Nx.invert a |> check_t "invert" [| 3 |] [| -1l; -2l; 0l |]);
    test "lshift" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 4l |] in
        Nx.lshift a 2 |> check_t "lshift" [| 3 |] [| 4l; 8l; 16l |]);
    test "rshift" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 4l; 8l; 16l |] in
        Nx.rshift a 2 |> check_t "rshift" [| 3 |] [| 1l; 2l; 4l |]);
  ]

let logical_tests =
  [
    test "logical_and" (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 1l; 0l |] in
        let b = Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 1l |] in
        Nx.logical_and a b |> check_t "logical_and" [| 4 |] [| 0l; 0l; 1l; 0l |]);
    test "logical_or" (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 1l; 0l |] in
        let b = Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 1l |] in
        Nx.logical_or a b |> check_t "logical_or" [| 4 |] [| 0l; 1l; 1l; 1l |]);
    test "logical_xor" (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 1l; 0l |] in
        let b = Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 1l |] in
        Nx.logical_xor a b |> check_t "logical_xor" [| 4 |] [| 0l; 1l; 0l; 1l |]);
    test "logical_not" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 1l; 1l |] in
        Nx.logical_not a |> check_t "logical_not" [| 3 |] [| 1l; 0l; 0l |]);
  ]

let special_value_tests =
  [
    test "isnan" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.0; nan; 0.0 |] in
        Nx.isnan a |> check_t "isnan" [| 3 |] [| false; true; false |]);
    test "isinf" (fun () ->
        let a =
          Nx.create Nx.float32 [| 4 |] [| 1.0; infinity; neg_infinity; 0.0 |]
        in
        Nx.isinf a |> check_t "isinf" [| 4 |] [| false; true; true; false |]);
    test "isfinite" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.0; infinity; nan; 0.0 |] in
        Nx.isfinite a
        |> check_t "isfinite" [| 4 |] [| true; false; false; true |]);
  ]

let ternary_tests =
  [
    test "where" (fun () ->
        let cond =
          Nx.create Nx.bool [| 5 |] [| true; false; true; false; true |]
        in
        let x = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        let y = Nx.create Nx.float32 [| 5 |] [| 10.; 20.; 30.; 40.; 50. |] in
        Nx.where cond x y |> check_t "where" [| 5 |] [| 1.; 20.; 3.; 40.; 5. |]);
  ]

let reduction_tests =
  [
    test "sum" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        Nx.sum a |> check_t "sum" [||] [| 6.0 |]);
    test "prod" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
        Nx.prod a |> check_t "prod" [||] [| 24.0 |]);
    test "max" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 5.; 3.; 2.; 4. |] in
        Nx.max a |> check_t "max" [||] [| 5.0 |]);
    test "min" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 5.; 1.; 3.; 2.; 4. |] in
        Nx.min a |> check_t "min" [||] [| 1.0 |]);
    test "mean" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
        Nx.mean a |> check_t "mean" [||] [| 2.5 |]);
    test "var" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
        Nx.var a |> check_t "var" [||] [| 1.25 |]);
    test "std" (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 3.; 5.; 7. |] in
        Nx.std a |> check_t ~eps:1e-6 "std" [||] [| 2.236068 |]);
    test "all" (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 1l; 1l; 0l; 1l |] in
        Nx.all a |> check_t "all with zero" [||] [| false |];
        let c = Nx.create Nx.int32 [| 3 |] [| 1l; 1l; 1l |] in
        Nx.all c |> check_t "all without zero" [||] [| true |]);
    test "any" (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 0l |] in
        Nx.any a |> check_t "any with one" [||] [| true |];
        let c = Nx.create Nx.int32 [| 3 |] [| 0l; 0l; 0l |] in
        Nx.any c |> check_t "any all zeros" [||] [| false |]);
    test "array_equal" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let eq1 = Nx.array_equal a b in
        equal ~msg:"array_equal same" bool true (Nx.item [] eq1);
        let d = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 4. |] in
        let eq2 = Nx.array_equal a d in
        equal ~msg:"array_equal different" bool false (Nx.item [] eq2));
  ]

let shape_manipulation_tests =
  [
    test "reshape" (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.reshape [| 3; 2 |] a |> check_t "reshape" [| 3; 2 |] test_array);
    test "flatten" (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.flatten a |> check_t "flatten" [| 6 |] test_array);
    test "unflatten" (fun () ->
        let a = Nx.create Nx.float32 [| 6 |] test_array in
        Nx.unflatten 0 [| 2; 3 |] a |> check_t "unflatten" [| 2; 3 |] test_array);
    test "ravel" (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.ravel a |> check_t "ravel" [| 6 |] test_array);
    test "squeeze" (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3; 1 |] in
        Nx.squeeze a |> check_t "squeeze" [| 3 |] [| 1.; 1.; 1. |]);
    test "squeeze_axis" (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3; 1 |] in
        Nx.squeeze_axis 0 a
        |> check_t "squeeze_axis" [| 3; 1 |] [| 1.; 1.; 1. |]);
    test "unsqueeze" (fun () ->
        let a = Nx.ones Nx.float32 [| 3 |] in
        Nx.unsqueeze ~axes:[ 0; 2 ] a
        |> check_t "unsqueeze" [| 1; 3; 1 |] [| 1.; 1.; 1. |]);
    test "unsqueeze_axis" (fun () ->
        let a = Nx.ones Nx.float32 [| 3 |] in
        Nx.unsqueeze_axis 0 a
        |> check_t "unsqueeze_axis" [| 1; 3 |] [| 1.; 1.; 1. |]);
    test "expand_dims" (fun () ->
        let a = Nx.ones Nx.float32 [| 3 |] in
        Nx.expand_dims [ 0 ] a
        |> check_t "expand_dims" [| 1; 3 |] [| 1.; 1.; 1. |]);
    test "transpose" (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.transpose a
        |> check_t "transpose" [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |]);
    test "moveaxis" (fun () ->
        let a =
          Nx.create Nx.float32 [| 2; 3; 4 |]
            (Array.init 24 (fun i -> float_of_int i))
        in
        let b = Nx.moveaxis 0 2 a in
        check_shape "moveaxis shape" [| 3; 4; 2 |] b;
        (* Check a few values to ensure proper axis movement *)
        let expected =
          Nx.create Nx.float32 [| 3; 4; 2 |]
            [|
              0.;
              12.;
              1.;
              13.;
              2.;
              14.;
              3.;
              15.;
              4.;
              16.;
              5.;
              17.;
              6.;
              18.;
              7.;
              19.;
              8.;
              20.;
              9.;
              21.;
              10.;
              22.;
              11.;
              23.;
            |]
        in
        check_t "moveaxis values" [| 3; 4; 2 |] (Nx.to_array expected) b);
    test "swapaxes" (fun () ->
        let a =
          Nx.create Nx.float32 [| 2; 3; 4 |]
            (Array.init 24 (fun i -> float_of_int i))
        in
        let b = Nx.swapaxes 0 2 a in
        check_shape "swapaxes shape" [| 4; 3; 2 |] b;
        (* Check a few values to ensure proper axis swapping *)
        let expected =
          Nx.create Nx.float32 [| 4; 3; 2 |]
            [|
              0.;
              12.;
              4.;
              16.;
              8.;
              20.;
              1.;
              13.;
              5.;
              17.;
              9.;
              21.;
              2.;
              14.;
              6.;
              18.;
              10.;
              22.;
              3.;
              15.;
              7.;
              19.;
              11.;
              23.;
            |]
        in
        check_t "swapaxes values" [| 4; 3; 2 |] (Nx.to_array expected) b);
    test "flip" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        Nx.flip a |> check_t "flip" [| 5 |] [| 5.; 4.; 3.; 2.; 1. |]);
    test "roll" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        Nx.roll 2 a |> check_t "roll" [| 5 |] [| 4.; 5.; 1.; 2.; 3. |]);
    test "pad" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.pad [| (1, 1); (1, 1) |] 0.0 a in
        let expected =
          [| 0.; 0.; 0.; 0.; 0.; 1.; 2.; 0.; 0.; 3.; 4.; 0.; 0.; 0.; 0.; 0. |]
        in
        check_t "pad values" [| 4; 4 |] expected b);
    test "shrink" (fun () ->
        let a =
          Nx.create Nx.float32 [| 4; 4 |]
            (Array.init 16 (fun i -> float_of_int i))
        in
        let b = Nx.shrink [| (1, 3); (1, 3) |] a in
        check_t "shrink values" [| 2; 2 |] [| 5.; 6.; 9.; 10. |] b);
    test "expand" (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3 |] in
        let b = Nx.expand [| 2; -1 |] a in
        check_t "expand values" [| 2; 3 |] [| 1.; 1.; 1.; 1.; 1.; 1. |] b);
    test "broadcast_to" (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3 |] in
        Nx.broadcast_to [| 2; 3 |] a
        |> check_t "broadcast_to" [| 2; 3 |] [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test "broadcast_arrays" (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3 |] in
        let b = Nx.full Nx.float32 [| 2; 1 |] 2.0 in
        let cs = Nx.broadcast_arrays [ a; b ] in
        equal ~msg:"broadcast_arrays count" int 2 (List.length cs);
        List.iter
          (fun c -> check_shape "broadcast_arrays shape" [| 2; 3 |] c)
          cs);
    test "tile" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        Nx.tile [| 2; 1 |] a
        |> check_t "tile" [| 4; 2 |] [| 1.; 2.; 3.; 4.; 1.; 2.; 3.; 4. |]);
    test "repeat" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        Nx.repeat 2 a |> check_t "repeat" [| 6 |] [| 1.; 1.; 2.; 2.; 3.; 3. |]);
  ]

let array_combination_tests =
  [
    test "concatenate" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.create Nx.float32 [| 1; 3 |] [| 7.; 8.; 9. |] in
        let c = Nx.concatenate ~axis:0 [ a; b ] in
        check_t "concatenate values" [| 3; 3 |]
          [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
          c);
    test "stack" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
        let b = Nx.create Nx.float32 [| 2 |] [| 3.; 4. |] in
        let c = Nx.stack ~axis:0 [ a; b ] in
        check_t "stack values" [| 2; 2 |] [| 1.; 2.; 3.; 4. |] c);
    test "vstack" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.create Nx.float32 [| 1; 2 |] [| 5.; 6. |] in
        let c = Nx.vstack [ a; b ] in
        check_t "vstack values" [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] c);
    test "hstack" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.create Nx.float32 [| 2; 1 |] [| 5.; 6. |] in
        let c = Nx.hstack [ a; b ] in
        check_t "hstack values" [| 2; 3 |] [| 1.; 2.; 5.; 3.; 4.; 6. |] c);
    test "dstack" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.create Nx.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
        let c = Nx.dstack [ a; b ] in
        check_t "dstack values" [| 2; 2; 2 |]
          [| 1.; 5.; 2.; 6.; 3.; 7.; 4.; 8. |]
          c);
    test "array_split" (fun () ->
        let a = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let splits = Nx.array_split ~axis:0 (`Count 3) a in
        equal ~msg:"array_split count" int 3 (List.length splits);
        check_t "split 0 values" [| 2 |] [| 1.; 2. |] (List.nth splits 0);
        check_t "split 1 values" [| 2 |] [| 3.; 4. |] (List.nth splits 1);
        check_t "split 2 values" [| 2 |] [| 5.; 6. |] (List.nth splits 2));
    test "split" (fun () ->
        let a = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let splits = Nx.split ~axis:0 3 a in
        equal ~msg:"split count" int 3 (List.length splits);
        List.iter (fun s -> check_shape "split shape" [| 2 |] s) splits);
  ]

let type_conversion_tests =
  [
    test "cast" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.7; 2.3; 3.9 |] in
        let b = Nx.cast Nx.int32 a in
        check_t "cast values" [| 3 |] [| 1l; 2l; 3l |] b);
    test "astype" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 3.14; 2.71 |] in
        let b = Nx.astype Nx.int32 a in
        equal ~msg:"astype dtype" bool true (Nx.dtype b = Nx.int32));
    test "to_bigarray" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let ba = Nx.to_bigarray a in
        equal ~msg:"bigarray dims" int 1 (Bigarray.Genarray.num_dims ba);
        equal ~msg:"bigarray value" (float 1e-6) 2.0
          (Bigarray.Genarray.get ba [| 1 |]));
    test "of_bigarray" (fun () ->
        let ba =
          Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |]
        in
        Bigarray.Genarray.set ba [| 0 |] 4.0;
        Bigarray.Genarray.set ba [| 1 |] 5.0;
        Bigarray.Genarray.set ba [| 2 |] 6.0;
        Nx.of_bigarray ba |> check_t "of_bigarray" [| 3 |] [| 4.; 5.; 6. |]);
    test "to_array" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 7.; 8.; 9. |] in
        let arr = Nx.to_array a in
        equal ~msg:"to_array length" int 3 (Array.length arr);
        equal ~msg:"to_array value" (float 1e-6) 8.0 arr.(1));
  ]

let indexing_slicing_tests =
  [
    test "get" (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.get [ 0 ] a |> check_t "get row 0" [| 3 |] [| 1.; 2.; 3. |];
        Nx.get [ 1 ] a |> check_t "get row 1" [| 3 |] [| 4.; 5.; 6. |]);
    test "set" (fun () ->
        let a = Nx.zeros Nx.float32 shape_2x3 in
        let value = Nx.create Nx.float32 [| 3 |] [| 7.; 8.; 9. |] in
        Nx.set [ 1 ] a value;
        check_t "set" shape_2x3 [| 0.; 0.; 0.; 7.; 8.; 9. |] a);
    test "item" (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        equal ~msg:"item [0,0]" (float 1e-6) 1.0 (Nx.item [ 0; 0 ] a);
        equal ~msg:"item [1,2]" (float 1e-6) 6.0 (Nx.item [ 1; 2 ] a));
    test "set_item" (fun () ->
        let a = Nx.zeros Nx.float32 shape_2x3 in
        Nx.set_item [ 0; 1 ] 42.0 a;
        Nx.set_item [ 1; 2 ] 99.0 a;
        equal ~msg:"set_item [0,1]" (float 1e-6) 42.0 (Nx.item [ 0; 1 ] a);
        equal ~msg:"set_item [1,2]" (float 1e-6) 99.0 (Nx.item [ 1; 2 ] a));
    test "slice" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        Nx.slice [ Nx.R (1, 4) ] a |> check_t "slice" [| 3 |] [| 2.; 3.; 4. |]);
    test "set_slice" (fun () ->
        let a = Nx.zeros Nx.float32 [| 5 |] in
        let value = Nx.create Nx.float32 [| 2 |] [| 10.; 20. |] in
        Nx.set_slice [ Nx.R (2, 4) ] a value;
        check_t "set_slice" [| 5 |] [| 0.; 0.; 10.; 20.; 0. |] a);
  ]

let linear_algebra_tests =
  [
    test "dot" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
        Nx.dot a b |> check_t "dot" [||] [| 32.0 |]);
    test "matmul" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |] in
        Nx.matmul a b |> check_t "matmul" [| 2; 2 |] [| 14.; 32.; 32.; 77. |]);
  ]

let neural_network_tests =
  [
    test "relu" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| -2.; -1.; 0.; 1.; 2. |] in
        Nx.relu a |> check_t "relu" [| 5 |] [| 0.; 0.; 0.; 1.; 2. |]);
    test "sigmoid" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| -10.; 0.; 10. |] in
        Nx.sigmoid a
        |> check_t ~eps:1e-6 "sigmoid" [| 3 |] [| 0.0000454; 0.5; 0.9999546 |]);
    test "one_hot" (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 1l |] in
        let b = Nx.one_hot a ~num_classes:3 in
        check_t "one_hot values" [| 3; 3 |] [| 1; 0; 0; 0; 0; 1; 0; 1; 0 |] b);
    test "correlate1d" (fun () ->
        let x = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        let w = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
        let y = Nx.correlate1d x w in
        check_t "correlate1d values" [| 1; 1; 3 |] [| -2.; -2.; -2. |] y);
    test "correlate2d" (fun () ->
        let x = Nx.ones Nx.float32 [| 1; 1; 5; 5 |] in
        let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in
        let y = Nx.correlate2d x w in
        check_t ~eps:1e-6 "correlate2d values" [| 1; 1; 3; 3 |]
          [| 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9. |]
          y);
    test "convolve1d" (fun () ->
        let x = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        let w = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
        let y = Nx.convolve1d x w in
        (* NumPy convolve flips the kernel, so [1,0,-1] becomes [-1,0,1] *)
        (* Result: [1,2,3]·[-1,0,1] = 2, [2,3,4]·[-1,0,1] = 2, [3,4,5]·[-1,0,1] = 2 *)
        check_t "convolve1d values" [| 1; 1; 3 |] [| 2.; 2.; 2. |] y);
    test "convolve2d" (fun () ->
        let x = Nx.ones Nx.float32 [| 1; 1; 5; 5 |] in
        let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in
        let y = Nx.convolve2d x w in
        check_t ~eps:1e-6 "convolve2d values" [| 1; 1; 3; 3 |]
          [| 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9. |]
          y);
    test "avg_pool1d" (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
        in
        let y = Nx.avg_pool1d ~kernel_size:2 x in
        check_t "avg_pool1d values" [| 1; 1; 3 |] [| 1.5; 3.5; 5.5 |] y);
    test "avg_pool2d" (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 4; 4 |]
            (Array.init 16 (fun i -> float_of_int (i + 1)))
        in
        let y = Nx.avg_pool2d ~kernel_size:(2, 2) x in
        check_t "avg_pool2d values" [| 1; 1; 2; 2 |] [| 3.5; 5.5; 11.5; 13.5 |]
          y);
    test "max_pool1d" (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 6 |] [| 1.; 3.; 2.; 6.; 4.; 5. |]
        in
        let y, _ = Nx.max_pool1d ~kernel_size:2 x in
        check_t "max_pool1d values" [| 1; 1; 3 |] [| 3.; 6.; 5. |] y);
    test "max_pool2d" (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 4; 4 |]
            [|
              1.;
              3.;
              2.;
              4.;
              5.;
              7.;
              6.;
              8.;
              9.;
              11.;
              10.;
              12.;
              13.;
              15.;
              14.;
              16.;
            |]
        in
        let y, _ = Nx.max_pool2d ~kernel_size:(2, 2) x in
        check_t "max_pool2d values" [| 1; 1; 2; 2 |] [| 7.; 8.; 15.; 16. |] y);
    test "min_pool1d" (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 6 |] [| 4.; 2.; 3.; 1.; 6.; 5. |]
        in
        let y, _ = Nx.min_pool1d ~kernel_size:2 x in
        check_t "min_pool1d values" [| 1; 1; 3 |] [| 2.; 1.; 5. |] y);
    test "min_pool2d" (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 4; 4 |]
            [|
              1.;
              3.;
              2.;
              4.;
              5.;
              7.;
              6.;
              8.;
              9.;
              11.;
              10.;
              12.;
              13.;
              15.;
              14.;
              16.;
            |]
        in
        let y, _ = Nx.min_pool2d ~kernel_size:(2, 2) x in
        check_t "min_pool2d values" [| 1; 1; 2; 2 |] [| 1.; 2.; 9.; 10. |] y);
  ]

let random_tests =
  [
    test "rand" (fun () ->
        let t = Nx.rand Nx.float32 ~key:(Nx.Rng.key 0) shape_2x3 in
        check_shape "rand shape" shape_2x3 t;
        let vals = Nx.to_array t in
        Array.iter
          (fun v -> equal ~msg:"rand in range" bool true (v >= 0.0 && v < 1.0))
          vals);
    test "randn" (fun () ->
        let t = Nx.randn Nx.float32 ~key:(Nx.Rng.key 1) [| 100 |] in
        check_shape "randn shape" [| 100 |] t;
        (* Check that values are roughly normally distributed *)
        let vals = Nx.to_array t in
        let mean = Array.fold_left ( +. ) 0.0 vals /. 100.0 in
        equal ~msg:"randn mean" bool true (abs_float mean < 0.5));
    test "randint" (fun () ->
        let t = Nx.randint Nx.int32 ~key:(Nx.Rng.key 2) shape_2x3 0 ~high:10 in
        check_shape "randint shape" shape_2x3 t;
        (* Check all values are in range *)
        for i = 0 to 1 do
          for j = 0 to 2 do
            let v = Nx.item [ i; j ] t in
            equal ~msg:"randint in range" bool true (v >= 0l && v < 10l)
          done
        done);
  ]

let sorting_searching_tests =
  [
    test "sort" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
        let sorted, indices = Nx.sort a in
        check_t "sort values" [| 5 |] [| 1.; 1.; 3.; 4.; 5. |] sorted;
        check_shape "sort indices shape" [| 5 |] indices);
    test "argsort" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
        Nx.argsort a |> check_t "argsort" [| 5 |] [| 1l; 3l; 0l; 2l; 4l |]);
    test "argmax" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 5.; 2.; 4. |] in
        Nx.argmax a |> check_t "argmax" [||] [| 2l |]);
    test "argmin" (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 5.; 2.; 4. |] in
        Nx.argmin a |> check_t "argmin" [||] [| 1l |]);
  ]

let display_formatting_tests =
  [
    test "pp_data" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let str = Format.asprintf "%a" Nx.pp_data a in
        equal ~msg:"pp_data not empty" bool true (String.length str > 0);
        equal ~msg:"pp_data contains data" bool true (String.contains str '1'));
    test "data_to_string" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let str = Nx.data_to_string a in
        equal ~msg:"data_to_string not empty" bool true (String.length str > 0);
        equal ~msg:"data_to_string contains data" bool true
          (String.contains str '1'));
    test "print_data" (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        Nx.print_data a);
    test "pp" (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        let str = Format.asprintf "%a" Nx.pp a in
        equal ~msg:"pp not empty" bool true (String.length str > 0));
    test "to_string" (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        let str = Nx.to_string a in
        equal ~msg:"to_string not empty" bool true (String.length str > 0));
    test "print" (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        Nx.print a);
  ]

let higher_order_tests =
  [
    test "map" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.map (fun x -> Nx.mul_s x 2.0) a in
        check_t "map double" [| 2; 3 |] [| 2.; 4.; 6.; 8.; 10.; 12. |] b);
    test "map preserves shape" (fun () ->
        let a =
          Nx.create Nx.float32 [| 3; 2; 2 |]
            [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
        in
        let b = Nx.map (fun x -> Nx.add_s x 1.0) a in
        check_t "map values" [| 3; 2; 2 |]
          [| 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13. |]
          b);
    test "iter" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let sum = ref (Nx.scalar Nx.float32 0.0) in
        Nx.iter (fun x -> sum := Nx.add !sum x) a;
        equal ~msg:"iter sum" (float 0.01) 10.0 (Nx.item [] !sum));
    test "fold" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let sum =
          Nx.fold (fun acc x -> Nx.add acc x) (Nx.scalar Nx.float32 0.0) a
        in
        equal ~msg:"fold sum" (float 0.01) 21.0 (Nx.item [] sum));
    test "fold product" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let prod =
          Nx.fold (fun acc x -> Nx.mul acc x) (Nx.scalar Nx.float32 1.0) a
        in
        equal ~msg:"fold product" (float 0.01) 24.0 (Nx.item [] prod));
    test "fold max" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 5.; 3.; 2.; 6.; 4. |] in
        let max_val =
          Nx.fold
            (fun acc x -> Nx.maximum acc x)
            (Nx.scalar Nx.float32 neg_infinity)
            a
        in
        equal ~msg:"fold max" (float 0.01) 6.0 (Nx.item [] max_val));
    test "map_item" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.map_item (fun x -> x *. 2.0) a in
        check_t "map_item double" [| 2; 3 |] [| 2.; 4.; 6.; 8.; 10.; 12. |] b);
    test "iter_item" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let sum = ref 0.0 in
        Nx.iter_item (fun x -> sum := !sum +. x) a;
        equal ~msg:"iter_item sum" (float 0.01) 10.0 !sum);
    test "fold_item" (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let sum = Nx.fold_item (fun acc x -> acc +. x) 0.0 a in
        equal ~msg:"fold_item sum" (float 0.01) 21.0 sum);
  ]

let () =
  run "Nx Sanity"
    [
      group "Creation Functions" creation_tests;
      group "Range Generation" range_generation_tests;
      group "Property Access" property_access_tests;
      group "Data Manipulation" data_manipulation_tests;
      group "Element-wise Binary Operations" element_wise_binary_tests;
      group "Comparison Operations" comparison_tests;
      group "Element-wise Unary Operations" element_wise_unary_tests;
      group "Bitwise Operations" bitwise_tests;
      group "Logical Operations" logical_tests;
      group "Special Value Checks" special_value_tests;
      group "Ternary Operations" ternary_tests;
      group "Reduction Operations" reduction_tests;
      group "Shape Manipulation" shape_manipulation_tests;
      group "Array Combination" array_combination_tests;
      group "Type Conversion" type_conversion_tests;
      group "Indexing and Slicing" indexing_slicing_tests;
      group "Linear Algebra" linear_algebra_tests;
      group "Neural Network" neural_network_tests;
      group "Random Number Generation" random_tests;
      group "Sorting and Searching" sorting_searching_tests;
      group "Display and Formatting" display_formatting_tests;
      group "Higher-order Functions" higher_order_tests;
    ]
