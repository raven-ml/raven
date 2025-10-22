(* Sanity tests for Nx - quick smoke test for every API function *)

open Alcotest
open Test_nx_support

(* Test helper to create simple test data *)
let test_array = [| 1.; 2.; 3.; 4.; 5.; 6. |]
let shape_2x3 = [| 2; 3 |]

let creation_tests =
  [
    test_case "create" `Quick (fun () ->
        Nx.create Nx.float32 shape_2x3 test_array
        |> check_t "create" shape_2x3 test_array);
    test_case "init" `Quick (fun () ->
        let t =
          Nx.init Nx.float32 [| 2; 2 |] (fun indices ->
              float_of_int (indices.(0) + indices.(1)))
        in
        check_t "init" [| 2; 2 |] [| 0.; 1.; 1.; 2. |] t);
    test_case "empty" `Quick (fun () ->
        Nx.empty Nx.float32 shape_2x3 |> check_shape "empty shape" shape_2x3);
    test_case "full" `Quick (fun () ->
        Nx.full Nx.float32 shape_2x3 7.0
        |> check_t "full" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |]);
    test_case "ones" `Quick (fun () ->
        Nx.ones Nx.float32 shape_2x3
        |> check_t "ones" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test_case "zeros" `Quick (fun () ->
        Nx.zeros Nx.float32 shape_2x3
        |> check_t "zeros" shape_2x3 [| 0.; 0.; 0.; 0.; 0.; 0. |]);
    test_case "ones_like" `Quick (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.ones_like ref_t
        |> check_t "ones_like" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test_case "zeros_like" `Quick (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.zeros_like ref_t
        |> check_t "zeros_like" shape_2x3 [| 0.; 0.; 0.; 0.; 0.; 0. |]);
    test_case "empty_like" `Quick (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.empty_like ref_t |> check_shape "empty_like shape" shape_2x3);
    test_case "full_like" `Quick (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.full_like ref_t 9.0
        |> check_t "full_like" shape_2x3 [| 9.; 9.; 9.; 9.; 9.; 9. |]);
    test_case "scalar" `Quick (fun () ->
        Nx.scalar Nx.float32 42.0 |> check_t "scalar" [||] [| 42.0 |]);
    test_case "scalar_like" `Quick (fun () ->
        let ref_t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.scalar_like ref_t 5.0 |> check_t "scalar_like" [||] [| 5.0 |]);
    test_case "eye" `Quick (fun () ->
        Nx.eye Nx.float32 3
        |> check_t "eye" [| 3; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]);
    test_case "identity" `Quick (fun () ->
        Nx.identity Nx.float32 3
        |> check_t "identity" [| 3; 3 |]
             [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]);
    test_case "copy" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.copy t |> check_t "copy" shape_2x3 test_array);
    test_case "contiguous" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.contiguous t |> check_t "contiguous" shape_2x3 test_array);
  ]

let range_generation_tests =
  [
    test_case "arange" `Quick (fun () ->
        let t = Nx.arange Nx.int32 0 5 1 in
        check_t "arange" [| 5 |] [| 0l; 1l; 2l; 3l; 4l |] t);
    test_case "arange_f" `Quick (fun () ->
        Nx.arange_f Nx.float32 0.0 1.0 0.25
        |> check_t "arange_f" [| 4 |] [| 0.0; 0.25; 0.5; 0.75 |]);
    test_case "linspace" `Quick (fun () ->
        let t = Nx.linspace Nx.float32 0.0 1.0 5 in
        check_t ~eps:1e-6 "linspace" [| 5 |] [| 0.0; 0.25; 0.5; 0.75; 1.0 |] t);
    test_case "logspace" `Quick (fun () ->
        let t = Nx.logspace Nx.float32 0.0 2.0 3 in
        check_t ~eps:1e-4 "logspace" [| 3 |] [| 1.0; 10.0; 100.0 |] t);
    test_case "geomspace" `Quick (fun () ->
        let t = Nx.geomspace Nx.float32 1.0 100.0 3 in
        check_t ~eps:1e-4 "geomspace" [| 3 |] [| 1.0; 10.0; 100.0 |] t);
  ]

let property_access_tests =
  [
    test_case "data" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        let data = Nx.data t in
        check (float 1e-6) "data[0]" 1.0 data.{0};
        check (float 1e-6) "data[0]" 6.0 data.{5});
    test_case "shape" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check (array int) "shape" shape_2x3 (Nx.shape t));
    test_case "dtype" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check bool "dtype is float32" true (Nx.dtype t = Nx.float32));
    test_case "strides" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        let strides = Nx.strides t in
        check int "strides length" 2 (Array.length strides);
        check int "stride 0" 12 strides.(0);
        check int "stride 1" 4 strides.(1));
    test_case "stride" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "stride 0" 12 (Nx.stride 0 t);
        check int "stride 1" 4 (Nx.stride 1 t));
    test_case "dims" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        let d = Nx.dims t in
        check int "dims length" 2 (Array.length d);
        check int "dims[0]" 2 d.(0);
        check int "dims[1]" 3 d.(1));
    test_case "dim" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "dim 0" 2 (Nx.dim 0 t);
        check int "dim 1" 3 (Nx.dim 1 t));
    test_case "ndim" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "ndim" 2 (Nx.ndim t));
    test_case "itemsize" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "itemsize" 4 (Nx.itemsize t));
    test_case "size" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "size" 6 (Nx.size t));
    test_case "numel" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "numel" 6 (Nx.numel t));
    test_case "nbytes" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "nbytes" 24 (Nx.nbytes t));
    test_case "offset" `Quick (fun () ->
        let t = Nx.create Nx.float32 shape_2x3 test_array in
        check int "offset" 0 (Nx.offset t));
  ]

let data_manipulation_tests =
  [
    test_case "blit" `Quick (fun () ->
        let src = Nx.ones Nx.float32 shape_2x3 in
        let dst = Nx.zeros Nx.float32 shape_2x3 in
        Nx.blit src dst;
        check_t "blit" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |] dst);
    test_case "fill" `Quick (fun () ->
        let t = Nx.zeros Nx.float32 shape_2x3 in
        let _ = Nx.fill 5.0 t in
        check_t "fill" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] t);
  ]

let element_wise_binary_tests =
  [
    test_case "add" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.add a b |> check_t "add" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test_case "add_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.add_s a 5.0 |> check_t "add_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test_case "radd_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.radd_s 5.0 a
        |> check_t "radd_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test_case "iadd" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.iadd a b in
        check_t "iadd" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] a);
    test_case "iadd_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.iadd_s a 5.0 in
        check_t "iadd_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |] a);
    test_case "sub" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.sub a b |> check_t "sub" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test_case "sub_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 10.0 in
        Nx.sub_s a 3.0 |> check_t "sub_s" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |]);
    test_case "rsub_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rsub_s 10.0 a
        |> check_t "rsub_s" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |]);
    test_case "isub" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.isub a b in
        check_t "isub" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
    test_case "isub_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 10.0 in
        let _ = Nx.isub_s a 3.0 in
        check_t "isub_s" shape_2x3 [| 7.; 7.; 7.; 7.; 7.; 7. |] a);
    test_case "mul" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.mul a b |> check_t "mul" shape_2x3 [| 6.; 6.; 6.; 6.; 6.; 6. |]);
    test_case "mul_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 4.0 in
        Nx.mul_s a 3.0
        |> check_t "mul_s" shape_2x3 [| 12.; 12.; 12.; 12.; 12.; 12. |]);
    test_case "rmul_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 4.0 in
        Nx.rmul_s 3.0 a
        |> check_t "rmul_s" shape_2x3 [| 12.; 12.; 12.; 12.; 12.; 12. |]);
    test_case "imul" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.imul a b in
        check_t "imul" shape_2x3 [| 6.; 6.; 6.; 6.; 6.; 6. |] a);
    test_case "imul_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 4.0 in
        let _ = Nx.imul_s a 3.0 in
        check_t "imul_s" shape_2x3 [| 12.; 12.; 12.; 12.; 12.; 12. |] a);
    test_case "div" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 6.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.div a b |> check_t "div" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test_case "div_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 12.0 in
        Nx.div_s a 3.0 |> check_t "div_s" shape_2x3 [| 4.; 4.; 4.; 4.; 4.; 4. |]);
    test_case "rdiv_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.rdiv_s 6.0 a
        |> check_t "rdiv_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test_case "idiv" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 6.0 in
        let b = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.idiv a b in
        check_t "idiv" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
    test_case "idiv_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 12.0 in
        let _ = Nx.idiv_s a 3.0 in
        check_t "idiv_s" shape_2x3 [| 4.; 4.; 4.; 4.; 4.; 4. |] a);
    test_case "pow" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.pow a b |> check_t "pow" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test_case "pow_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        Nx.pow_s a 3.0 |> check_t "pow_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test_case "rpow_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rpow_s 2.0 a
        |> check_t "rpow_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |]);
    test_case "ipow" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.ipow a b in
        check_t "ipow" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |] a);
    test_case "ipow_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 2.0 in
        let _ = Nx.ipow_s a 3.0 in
        check_t "ipow_s" shape_2x3 [| 8.; 8.; 8.; 8.; 8.; 8. |] a);
    test_case "mod" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.mod_ a b |> check_t "mod" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test_case "mod_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        Nx.mod_s a 3.0 |> check_t "mod_s" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test_case "rmod_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rmod_s 7.0 a
        |> check_t "rmod_s" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test_case "imod" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.imod a b in
        check_t "imod" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |] a);
    test_case "imod_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 7.0 in
        let _ = Nx.imod_s a 3.0 in
        check_t "imod_s" shape_2x3 [| 1.; 1.; 1.; 1.; 1.; 1. |] a);
    test_case "maximum" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.maximum a b
        |> check_t "maximum" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test_case "maximum_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.maximum_s a 5.0
        |> check_t "maximum_s" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test_case "rmaximum_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        Nx.rmaximum_s 5.0 a
        |> check_t "rmaximum_s" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |]);
    test_case "imaximum" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 5.0 in
        let _ = Nx.imaximum a b in
        check_t "imaximum" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] a);
    test_case "imaximum_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.imaximum_s a 5.0 in
        check_t "imaximum_s" shape_2x3 [| 5.; 5.; 5.; 5.; 5.; 5. |] a);
    test_case "minimum" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 3.0 in
        let b = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.minimum a b
        |> check_t "minimum" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test_case "minimum_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.minimum_s a 3.0
        |> check_t "minimum_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test_case "rminimum_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        Nx.rminimum_s 3.0 a
        |> check_t "rminimum_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |]);
    test_case "iminimum" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let b = Nx.full Nx.float32 shape_2x3 3.0 in
        let _ = Nx.iminimum a b in
        check_t "iminimum" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
    test_case "iminimum_s" `Quick (fun () ->
        let a = Nx.full Nx.float32 shape_2x3 5.0 in
        let _ = Nx.iminimum_s a 3.0 in
        check_t "iminimum_s" shape_2x3 [| 3.; 3.; 3.; 3.; 3.; 3. |] a);
  ]

let comparison_tests =
  [
    test_case "equal" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 1. |] in
        Nx.equal a b |> check_t "equal" [| 3 |] [| true; false; true |]);
    test_case "not_equal" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 1. |] in
        Nx.not_equal a b |> check_t "not_equal" [| 3 |] [| false; true; false |]);
    test_case "greater" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 5.; 3.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.greater a b |> check_t "greater" [| 3 |] [| true; false; false |]);
    test_case "greater_equal" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 5.; 3.; 1. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.greater_equal a b
        |> check_t "greater_equal" [| 3 |] [| true; true; false |]);
    test_case "less" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 5. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.less a b |> check_t "less" [| 3 |] [| true; false; false |]);
    test_case "less_equal" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 5. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 3.; 3.; 2. |] in
        Nx.less_equal a b
        |> check_t "less_equal" [| 3 |] [| true; true; false |]);
  ]

let element_wise_unary_tests =
  [
    test_case "neg" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; -2.; 3. |] in
        Nx.neg a |> check_t "neg" [| 3 |] [| -1.; 2.; -3. |]);
    test_case "abs" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| -3.; 0.; 5. |] in
        Nx.abs a |> check_t "abs" [| 3 |] [| 3.; 0.; 5. |]);
    test_case "sign" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| -5.; 0.; 3.; -0. |] in
        Nx.sign a |> check_t "sign" [| 4 |] [| -1.; 0.; 1.; 0. |]);
    test_case "square" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| -2.; 3.; 4. |] in
        Nx.square a |> check_t "square" [| 3 |] [| 4.; 9.; 16. |]);
    test_case "sqrt" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 4.; 9.; 16. |] in
        Nx.sqrt a |> check_t "sqrt" [| 3 |] [| 2.; 3.; 4. |]);
    test_case "rsqrt" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 4.; 16. |] in
        Nx.rsqrt a |> check_t "rsqrt" [| 3 |] [| 1.0; 0.5; 0.25 |]);
    test_case "recip" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 4. |] in
        Nx.recip a |> check_t "recip" [| 3 |] [| 1.0; 0.5; 0.25 |]);
    test_case "exp" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 2. |] in
        Nx.exp a
        |> check_t ~eps:1e-6 "exp" [| 3 |] [| 1.0; 2.718282; 7.389056 |]);
    test_case "exp2" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 3. |] in
        Nx.exp2 a |> check_t "exp2" [| 3 |] [| 1.; 2.; 8. |]);
    test_case "log" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2.718282 |] in
        Nx.log a |> check_t ~eps:1e-6 "log" [| 2 |] [| 0.0; 1.0 |]);
    test_case "log2" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 8. |] in
        Nx.log2 a |> check_t "log2" [| 3 |] [| 0.; 1.; 3. |]);
    test_case "sin" `Quick (fun () ->
        let pi = 3.14159265359 in
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; pi /. 2.; pi |] in
        Nx.sin a |> check_t ~eps:1e-6 "sin" [| 3 |] [| 0.0; 1.0; 0.0 |]);
    test_case "cos" `Quick (fun () ->
        let pi = 3.14159265359 in
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; pi /. 2.; pi |] in
        Nx.cos a |> check_t ~eps:1e-6 "cos" [| 3 |] [| 1.0; 0.0; -1.0 |]);
    test_case "tan" `Quick (fun () ->
        let pi = 3.14159265359 in
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; pi /. 4. |] in
        Nx.tan a |> check_t ~eps:1e-6 "tan" [| 2 |] [| 0.0; 1.0 |]);
    test_case "asin" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 0.5; 1. |] in
        Nx.asin a
        |> check_t ~eps:1e-6 "asin" [| 3 |] [| 0.0; 0.523599; 1.570796 |]);
    test_case "acos" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 0.5; 0. |] in
        Nx.acos a
        |> check_t ~eps:1e-6 "acos" [| 3 |] [| 0.0; 1.047198; 1.570796 |]);
    test_case "atan" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; -1. |] in
        Nx.atan a
        |> check_t ~eps:1e-6 "atan" [| 3 |] [| 0.0; 0.785398; -0.785398 |]);
    test_case "sinh" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; 1. |] in
        Nx.sinh a |> check_t ~eps:1e-6 "sinh" [| 2 |] [| 0.0; 1.175201 |]);
    test_case "cosh" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; 1. |] in
        Nx.cosh a |> check_t ~eps:1e-6 "cosh" [| 2 |] [| 1.0; 1.543081 |]);
    test_case "tanh" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 1.; -1. |] in
        Nx.tanh a
        |> check_t ~eps:1e-6 "tanh" [| 3 |] [| 0.0; 0.761594; -0.761594 |]);
    test_case "asinh" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.; 1. |] in
        Nx.asinh a |> check_t ~eps:1e-6 "asinh" [| 2 |] [| 0.0; 0.881374 |]);
    test_case "acosh" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
        Nx.acosh a |> check_t ~eps:1e-6 "acosh" [| 2 |] [| 0.0; 1.316958 |]);
    test_case "atanh" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.; 0.5; -0.5 |] in
        Nx.atanh a
        |> check_t ~eps:1e-6 "atanh" [| 3 |] [| 0.0; 0.549306; -0.549306 |]);
    test_case "round" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.round a |> check_t "round" [| 4 |] [| 3.; 4.; -3.; -4. |]);
    test_case "floor" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.floor a |> check_t "floor" [| 4 |] [| 3.; 3.; -4.; -4. |]);
    test_case "ceil" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.ceil a |> check_t "ceil" [| 4 |] [| 4.; 4.; -3.; -3. |]);
    test_case "trunc" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 3.2; 3.7; -3.2; -3.7 |] in
        Nx.trunc a |> check_t "trunc" [| 4 |] [| 3.; 3.; -3.; -3. |]);
    test_case "clip" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| -2.; 0.; 5.; 10.; 12. |] in
        Nx.clip ~min:2.0 ~max:8.0 a
        |> check_t "clip" [| 5 |] [| 2.0; 2.0; 5.0; 8.0; 8.0 |]);
    test_case "clamp" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| -2.; 0.; 5.; 10.; 12. |] in
        Nx.clamp ~min:2.0 ~max:8.0 a
        |> check_t "clamp" [| 5 |] [| 2.0; 2.0; 5.0; 8.0; 8.0 |]);
    test_case "lerp" `Quick (fun () ->
        let start_t = Nx.zeros Nx.float32 [| 3 |] in
        let end_t = Nx.full Nx.float32 [| 3 |] 10.0 in
        let weight = Nx.create Nx.float32 [| 3 |] [| 0.0; 0.5; 1.0 |] in
        Nx.lerp start_t end_t weight
        |> check_t "lerp" [| 3 |] [| 0.0; 5.0; 10.0 |]);
    test_case "lerp_scalar_weight" `Quick (fun () ->
        let start_t = Nx.zeros Nx.float32 [| 3 |] in
        let end_t = Nx.full Nx.float32 [| 3 |] 10.0 in
        Nx.lerp_scalar_weight start_t end_t 0.3
        |> check_t "lerp_scalar_weight" [| 3 |] [| 3.0; 3.0; 3.0 |]);
  ]

let bitwise_tests =
  [
    test_case "bitwise_and" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 7l; 12l; 15l |] in
        let b = Nx.create Nx.int32 [| 3 |] [| 3l; 10l; 7l |] in
        Nx.bitwise_and a b |> check_t "bitwise_and" [| 3 |] [| 3l; 8l; 7l |]);
    test_case "bitwise_or" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 1l; 4l; 8l |] in
        let b = Nx.create Nx.int32 [| 3 |] [| 2l; 2l; 7l |] in
        Nx.bitwise_or a b |> check_t "bitwise_or" [| 3 |] [| 3l; 6l; 15l |]);
    test_case "bitwise_xor" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 7l; 12l; 15l |] in
        let b = Nx.create Nx.int32 [| 3 |] [| 3l; 10l; 7l |] in
        Nx.bitwise_xor a b |> check_t "bitwise_xor" [| 3 |] [| 4l; 6l; 8l |]);
    test_case "bitwise_not" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 1l; -1l |] in
        Nx.bitwise_not a |> check_t "bitwise_not" [| 3 |] [| -1l; -2l; 0l |]);
    test_case "invert" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 1l; -1l |] in
        Nx.invert a |> check_t "invert" [| 3 |] [| -1l; -2l; 0l |]);
    test_case "lshift" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 4l |] in
        Nx.lshift a 2 |> check_t "lshift" [| 3 |] [| 4l; 8l; 16l |]);
    test_case "rshift" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 4l; 8l; 16l |] in
        Nx.rshift a 2 |> check_t "rshift" [| 3 |] [| 1l; 2l; 4l |]);
  ]

let logical_tests =
  [
    test_case "logical_and" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 1l; 0l |] in
        let b = Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 1l |] in
        Nx.logical_and a b |> check_t "logical_and" [| 4 |] [| 0l; 0l; 1l; 0l |]);
    test_case "logical_or" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 1l; 0l |] in
        let b = Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 1l |] in
        Nx.logical_or a b |> check_t "logical_or" [| 4 |] [| 0l; 1l; 1l; 1l |]);
    test_case "logical_xor" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 1l; 0l |] in
        let b = Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 1l |] in
        Nx.logical_xor a b |> check_t "logical_xor" [| 4 |] [| 0l; 1l; 0l; 1l |]);
    test_case "logical_not" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 1l; 1l |] in
        Nx.logical_not a |> check_t "logical_not" [| 3 |] [| 1l; 0l; 0l |]);
  ]

let special_value_tests =
  [
    test_case "isnan" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.0; nan; 0.0 |] in
        Nx.isnan a |> check_t "isnan" [| 3 |] [| false; true; false |]);
    test_case "isinf" `Quick (fun () ->
        let a =
          Nx.create Nx.float32 [| 4 |] [| 1.0; infinity; neg_infinity; 0.0 |]
        in
        Nx.isinf a |> check_t "isinf" [| 4 |] [| false; true; true; false |]);
    test_case "isfinite" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.0; infinity; nan; 0.0 |] in
        Nx.isfinite a
        |> check_t "isfinite" [| 4 |] [| true; false; false; true |]);
  ]

let ternary_tests =
  [
    test_case "where" `Quick (fun () ->
        let cond =
          Nx.create Nx.bool [| 5 |] [| true; false; true; false; true |]
        in
        let x = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        let y = Nx.create Nx.float32 [| 5 |] [| 10.; 20.; 30.; 40.; 50. |] in
        Nx.where cond x y |> check_t "where" [| 5 |] [| 1.; 20.; 3.; 40.; 5. |]);
  ]

let reduction_tests =
  [
    test_case "sum" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        Nx.sum a |> check_t "sum" [||] [| 6.0 |]);
    test_case "prod" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
        Nx.prod a |> check_t "prod" [||] [| 24.0 |]);
    test_case "max" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 5.; 3.; 2.; 4. |] in
        Nx.max a |> check_t "max" [||] [| 5.0 |]);
    test_case "min" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 5.; 1.; 3.; 2.; 4. |] in
        Nx.min a |> check_t "min" [||] [| 1.0 |]);
    test_case "mean" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
        Nx.mean a |> check_t "mean" [||] [| 2.5 |]);
    test_case "var" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
        Nx.var a |> check_t "var" [||] [| 1.25 |]);
    test_case "std" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 4 |] [| 1.; 3.; 5.; 7. |] in
        Nx.std a |> check_t ~eps:1e-6 "std" [||] [| 2.236068 |]);
    test_case "all" `Quick (fun () ->
        let a = Nx.create Nx.int [| 4 |] [| 1; 1; 0; 1 |] in
        Nx.all a |> check_t "all with zero" [||] [| false |];
        let c = Nx.create Nx.int [| 3 |] [| 1; 1; 1 |] in
        Nx.all c |> check_t "all without zero" [||] [| true |]);
    test_case "any" `Quick (fun () ->
        let a = Nx.create Nx.int [| 4 |] [| 0; 0; 1; 0 |] in
        Nx.any a |> check_t "any with one" [||] [| true |];
        let c = Nx.create Nx.int [| 3 |] [| 0; 0; 0 |] in
        Nx.any c |> check_t "any all zeros" [||] [| false |]);
    test_case "array_equal" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let eq1 = Nx.array_equal a b in
        check bool "array_equal same" true (Nx.item [] eq1);
        let d = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 4. |] in
        let eq2 = Nx.array_equal a d in
        check bool "array_equal different" false (Nx.item [] eq2));
  ]

let shape_manipulation_tests =
  [
    test_case "reshape" `Quick (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.reshape [| 3; 2 |] a |> check_t "reshape" [| 3; 2 |] test_array);
    test_case "flatten" `Quick (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.flatten a |> check_t "flatten" [| 6 |] test_array);
    test_case "unflatten" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 6 |] test_array in
        Nx.unflatten 0 [| 2; 3 |] a |> check_t "unflatten" [| 2; 3 |] test_array);
    test_case "ravel" `Quick (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.ravel a |> check_t "ravel" [| 6 |] test_array);
    test_case "squeeze" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3; 1 |] in
        Nx.squeeze a |> check_t "squeeze" [| 3 |] [| 1.; 1.; 1. |]);
    test_case "squeeze_axis" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3; 1 |] in
        Nx.squeeze_axis 0 a
        |> check_t "squeeze_axis" [| 3; 1 |] [| 1.; 1.; 1. |]);
    test_case "unsqueeze" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 3 |] in
        Nx.unsqueeze ~axes:[ 0; 2 ] a
        |> check_t "unsqueeze" [| 1; 3; 1 |] [| 1.; 1.; 1. |]);
    test_case "unsqueeze_axis" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 3 |] in
        Nx.unsqueeze_axis 0 a
        |> check_t "unsqueeze_axis" [| 1; 3 |] [| 1.; 1.; 1. |]);
    test_case "expand_dims" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 3 |] in
        Nx.expand_dims [ 0 ] a
        |> check_t "expand_dims" [| 1; 3 |] [| 1.; 1.; 1. |]);
    test_case "transpose" `Quick (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.transpose a
        |> check_t "transpose" [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |]);
    test_case "moveaxis" `Quick (fun () ->
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
    test_case "swapaxes" `Quick (fun () ->
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
    test_case "flip" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        Nx.flip a |> check_t "flip" [| 5 |] [| 5.; 4.; 3.; 2.; 1. |]);
    test_case "roll" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        Nx.roll 2 a |> check_t "roll" [| 5 |] [| 4.; 5.; 1.; 2.; 3. |]);
    test_case "pad" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.pad [| (1, 1); (1, 1) |] 0.0 a in
        let expected =
          [| 0.; 0.; 0.; 0.; 0.; 1.; 2.; 0.; 0.; 3.; 4.; 0.; 0.; 0.; 0.; 0. |]
        in
        check_t "pad values" [| 4; 4 |] expected b);
    test_case "shrink" `Quick (fun () ->
        let a =
          Nx.create Nx.float32 [| 4; 4 |]
            (Array.init 16 (fun i -> float_of_int i))
        in
        let b = Nx.shrink [| (1, 3); (1, 3) |] a in
        check_t "shrink values" [| 2; 2 |] [| 5.; 6.; 9.; 10. |] b);
    test_case "expand" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3 |] in
        let b = Nx.expand [| 2; -1 |] a in
        check_t "expand values" [| 2; 3 |] [| 1.; 1.; 1.; 1.; 1.; 1. |] b);
    test_case "broadcast_to" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3 |] in
        Nx.broadcast_to [| 2; 3 |] a
        |> check_t "broadcast_to" [| 2; 3 |] [| 1.; 1.; 1.; 1.; 1.; 1. |]);
    test_case "broadcast_arrays" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 1; 3 |] in
        let b = Nx.full Nx.float32 [| 2; 1 |] 2.0 in
        let cs = Nx.broadcast_arrays [ a; b ] in
        check int "broadcast_arrays count" 2 (List.length cs);
        List.iter
          (fun c -> check_shape "broadcast_arrays shape" [| 2; 3 |] c)
          cs);
    test_case "tile" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        Nx.tile [| 2; 1 |] a
        |> check_t "tile" [| 4; 2 |] [| 1.; 2.; 3.; 4.; 1.; 2.; 3.; 4. |]);
    test_case "repeat" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        Nx.repeat 2 a |> check_t "repeat" [| 6 |] [| 1.; 1.; 2.; 2.; 3.; 3. |]);
  ]

let array_combination_tests =
  [
    test_case "concatenate" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.create Nx.float32 [| 1; 3 |] [| 7.; 8.; 9. |] in
        let c = Nx.concatenate ~axis:0 [ a; b ] in
        check_t "concatenate values" [| 3; 3 |]
          [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
          c);
    test_case "stack" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
        let b = Nx.create Nx.float32 [| 2 |] [| 3.; 4. |] in
        let c = Nx.stack ~axis:0 [ a; b ] in
        check_t "stack values" [| 2; 2 |] [| 1.; 2.; 3.; 4. |] c);
    test_case "vstack" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.create Nx.float32 [| 1; 2 |] [| 5.; 6. |] in
        let c = Nx.vstack [ a; b ] in
        check_t "vstack values" [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] c);
    test_case "hstack" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.create Nx.float32 [| 2; 1 |] [| 5.; 6. |] in
        let c = Nx.hstack [ a; b ] in
        check_t "hstack values" [| 2; 3 |] [| 1.; 2.; 5.; 3.; 4.; 6. |] c);
    test_case "dstack" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = Nx.create Nx.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
        let c = Nx.dstack [ a; b ] in
        check_t "dstack values" [| 2; 2; 2 |]
          [| 1.; 5.; 2.; 6.; 3.; 7.; 4.; 8. |]
          c);
    test_case "array_split" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let splits = Nx.array_split ~axis:0 (`Count 3) a in
        check int "array_split count" 3 (List.length splits);
        check_t "split 0 values" [| 2 |] [| 1.; 2. |] (List.nth splits 0);
        check_t "split 1 values" [| 2 |] [| 3.; 4. |] (List.nth splits 1);
        check_t "split 2 values" [| 2 |] [| 5.; 6. |] (List.nth splits 2));
    test_case "split" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let splits = Nx.split ~axis:0 3 a in
        check int "split count" 3 (List.length splits);
        List.iter (fun s -> check_shape "split shape" [| 2 |] s) splits);
  ]

let type_conversion_tests =
  [
    test_case "cast" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.7; 2.3; 3.9 |] in
        let b = Nx.cast Nx.int32 a in
        check_t "cast values" [| 3 |] [| 1l; 2l; 3l |] b);
    test_case "astype" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 3.14; 2.71 |] in
        let b = Nx.astype Nx.int32 a in
        check bool "astype dtype" true (Nx.dtype b = Nx.int32));
    test_case "to_bigarray" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let ba = Nx.to_bigarray a in
        check int "bigarray dims" 1 (Bigarray.Genarray.num_dims ba);
        check (float 1e-6) "bigarray value" 2.0
          (Bigarray.Genarray.get ba [| 1 |]));
    test_case "of_bigarray" `Quick (fun () ->
        let ba =
          Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |]
        in
        Bigarray.Genarray.set ba [| 0 |] 4.0;
        Bigarray.Genarray.set ba [| 1 |] 5.0;
        Bigarray.Genarray.set ba [| 2 |] 6.0;
        Nx.of_bigarray ba |> check_t "of_bigarray" [| 3 |] [| 4.; 5.; 6. |]);
    test_case "to_array" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 7.; 8.; 9. |] in
        let arr = Nx.to_array a in
        check int "to_array length" 3 (Array.length arr);
        check (float 1e-6) "to_array value" 8.0 arr.(1));
  ]

let indexing_slicing_tests =
  [
    test_case "get" `Quick (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        Nx.get [ 0 ] a |> check_t "get row 0" [| 3 |] [| 1.; 2.; 3. |];
        Nx.get [ 1 ] a |> check_t "get row 1" [| 3 |] [| 4.; 5.; 6. |]);
    test_case "set" `Quick (fun () ->
        let a = Nx.zeros Nx.float32 shape_2x3 in
        let value = Nx.create Nx.float32 [| 3 |] [| 7.; 8.; 9. |] in
        Nx.set [ 1 ] a value;
        check_t "set" shape_2x3 [| 0.; 0.; 0.; 7.; 8.; 9. |] a);
    test_case "item" `Quick (fun () ->
        let a = Nx.create Nx.float32 shape_2x3 test_array in
        check (float 1e-6) "item [0,0]" 1.0 (Nx.item [ 0; 0 ] a);
        check (float 1e-6) "item [1,2]" 6.0 (Nx.item [ 1; 2 ] a));
    test_case "set_item" `Quick (fun () ->
        let a = Nx.zeros Nx.float32 shape_2x3 in
        Nx.set_item [ 0; 1 ] 42.0 a;
        Nx.set_item [ 1; 2 ] 99.0 a;
        check (float 1e-6) "set_item [0,1]" 42.0 (Nx.item [ 0; 1 ] a);
        check (float 1e-6) "set_item [1,2]" 99.0 (Nx.item [ 1; 2 ] a));
    test_case "slice" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        Nx.slice [ Nx.R (1, 4) ] a |> check_t "slice" [| 3 |] [| 2.; 3.; 4. |]);
    test_case "set_slice" `Quick (fun () ->
        let a = Nx.zeros Nx.float32 [| 5 |] in
        let value = Nx.create Nx.float32 [| 2 |] [| 10.; 20. |] in
        Nx.set_slice [ Nx.R (2, 4) ] a value;
        check_t "set_slice" [| 5 |] [| 0.; 0.; 10.; 20.; 0. |] a);
    test_case "slice" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        Nx.slice [ Nx.R (1, 4) ] a |> check_t "slice" [| 3 |] [| 2.; 3.; 4. |]);
    test_case "set_slice" `Quick (fun () ->
        let a = Nx.zeros Nx.float32 [| 5 |] in
        let value = Nx.create Nx.float32 [| 2 |] [| 10.; 20. |] in
        Nx.set_slice [ Nx.R (2, 4) ] a value;
        check_t "set_slice" [| 5 |] [| 0.; 0.; 10.; 20.; 0. |] a);
  ]

let linear_algebra_tests =
  [
    test_case "dot" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
        let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
        Nx.dot a b |> check_t "dot" [||] [| 32.0 |]);
    test_case "matmul" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |] in
        Nx.matmul a b |> check_t "matmul" [| 2; 2 |] [| 14.; 32.; 32.; 77. |]);
  ]

let neural_network_tests =
  [
    test_case "relu" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| -2.; -1.; 0.; 1.; 2. |] in
        Nx.relu a |> check_t "relu" [| 5 |] [| 0.; 0.; 0.; 1.; 2. |]);
    test_case "sigmoid" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| -10.; 0.; 10. |] in
        Nx.sigmoid a
        |> check_t ~eps:1e-6 "sigmoid" [| 3 |] [| 0.0000454; 0.5; 0.9999546 |]);
    test_case "hardsigmoid" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| -3.; -1.; 0.; 1.; 3. |] in
        Nx.hard_sigmoid a
        |> check_t ~eps:1e-6 "hardsigmoid" [| 5 |]
             [| 0.0; 0.333333; 0.5; 0.666667; 1.0 |]);
    test_case "one_hot" `Quick (fun () ->
        let a = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 1l |] in
        let b = Nx.one_hot a ~num_classes:3 in
        check_t "one_hot values" [| 3; 3 |] [| 1; 0; 0; 0; 0; 1; 0; 1; 0 |] b);
    test_case "correlate1d" `Quick (fun () ->
        let x = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        let w = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
        let y = Nx.correlate1d x w in
        check_t "correlate1d values" [| 1; 1; 3 |] [| -2.; -2.; -2. |] y);
    test_case "correlate2d" `Quick (fun () ->
        let x = Nx.ones Nx.float32 [| 1; 1; 5; 5 |] in
        let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in
        let y = Nx.correlate2d x w in
        check_t ~eps:1e-6 "correlate2d values" [| 1; 1; 3; 3 |]
          [| 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9. |]
          y);
    test_case "convolve1d" `Quick (fun () ->
        let x = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        let w = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
        let y = Nx.convolve1d x w in
        (* NumPy convolve flips the kernel, so [1,0,-1] becomes [-1,0,1] *)
        (* Result: [1,2,3]·[-1,0,1] = 2, [2,3,4]·[-1,0,1] = 2, [3,4,5]·[-1,0,1] = 2 *)
        check_t "convolve1d values" [| 1; 1; 3 |] [| 2.; 2.; 2. |] y);
    test_case "convolve2d" `Quick (fun () ->
        let x = Nx.ones Nx.float32 [| 1; 1; 5; 5 |] in
        let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in
        let y = Nx.convolve2d x w in
        check_t ~eps:1e-6 "convolve2d values" [| 1; 1; 3; 3 |]
          [| 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9. |]
          y);
    test_case "avg_pool1d" `Quick (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
        in
        let y = Nx.avg_pool1d ~kernel_size:2 x in
        check_t "avg_pool1d values" [| 1; 1; 3 |] [| 1.5; 3.5; 5.5 |] y);
    test_case "avg_pool2d" `Quick (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 4; 4 |]
            (Array.init 16 (fun i -> float_of_int (i + 1)))
        in
        let y = Nx.avg_pool2d ~kernel_size:(2, 2) x in
        check_t "avg_pool2d values" [| 1; 1; 2; 2 |] [| 3.5; 5.5; 11.5; 13.5 |]
          y);
    test_case "max_pool1d" `Quick (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 6 |] [| 1.; 3.; 2.; 6.; 4.; 5. |]
        in
        let y, _ = Nx.max_pool1d ~kernel_size:2 x in
        check_t "max_pool1d values" [| 1; 1; 3 |] [| 3.; 6.; 5. |] y);
    test_case "max_pool2d" `Quick (fun () ->
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
    test_case "min_pool1d" `Quick (fun () ->
        let x =
          Nx.create Nx.float32 [| 1; 1; 6 |] [| 4.; 2.; 3.; 1.; 6.; 5. |]
        in
        let y, _ = Nx.min_pool1d ~kernel_size:2 x in
        check_t "min_pool1d values" [| 1; 1; 3 |] [| 2.; 1.; 5. |] y);
    test_case "min_pool2d" `Quick (fun () ->
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
    test_case "rand" `Quick (fun () ->
        let t = Nx.rand Nx.float32 shape_2x3 in
        check_shape "rand shape" shape_2x3 t;
        let vals = Nx.to_array t in
        Array.iter
          (fun v -> check bool "rand in range" true (v >= 0.0 && v < 1.0))
          vals);
    test_case "randn" `Quick (fun () ->
        let t = Nx.randn Nx.float32 [| 100 |] in
        check_shape "randn shape" [| 100 |] t;
        (* Check that values are roughly normally distributed *)
        let vals = Nx.to_array t in
        let mean = Array.fold_left ( +. ) 0.0 vals /. 100.0 in
        check bool "randn mean" true (abs_float mean < 0.5));
    test_case "randint" `Quick (fun () ->
        let t = Nx.randint Nx.int32 shape_2x3 0 ~high:10 in
        check_shape "randint shape" shape_2x3 t;
        (* Check all values are in range *)
        for i = 0 to 1 do
          for j = 0 to 2 do
            let v = Nx.item [ i; j ] t in
            check bool "randint in range" true (v >= 0l && v < 10l)
          done
        done);
  ]

let sorting_searching_tests =
  [
    test_case "sort" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
        let sorted, indices = Nx.sort a in
        check_t "sort values" [| 5 |] [| 1.; 1.; 3.; 4.; 5. |] sorted;
        check_shape "sort indices shape" [| 5 |] indices);
    test_case "argsort" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
        Nx.argsort a |> check_t "argsort" [| 5 |] [| 1l; 3l; 0l; 2l; 4l |]);
    test_case "argmax" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 5.; 2.; 4. |] in
        Nx.argmax a |> check_t "argmax" [||] [| 2l |]);
    test_case "argmin" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 5.; 2.; 4. |] in
        Nx.argmin a |> check_t "argmin" [||] [| 1l |]);
  ]

let display_formatting_tests =
  [
    test_case "pp_data" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let str = Format.asprintf "%a" Nx.pp_data a in
        check bool "pp_data not empty" true (String.length str > 0);
        check bool "pp_data contains data" true (String.contains str '1'));
    test_case "data_to_string" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let str = Nx.data_to_string a in
        check bool "data_to_string not empty" true (String.length str > 0);
        check bool "data_to_string contains data" true (String.contains str '1'));
    test_case "print_data" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        Nx.print_data a);
    test_case "pp" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        let str = Format.asprintf "%a" Nx.pp a in
        check bool "pp not empty" true (String.length str > 0));
    test_case "to_string" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        let str = Nx.to_string a in
        check bool "to_string not empty" true (String.length str > 0));
    test_case "print" `Quick (fun () ->
        let a = Nx.ones Nx.float32 [| 2; 2 |] in
        Nx.print a);
  ]

let higher_order_tests =
  [
    test_case "map" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.map (fun x -> Nx.mul_s x 2.0) a in
        check_t "map double" [| 2; 3 |] [| 2.; 4.; 6.; 8.; 10.; 12. |] b);
    test_case "map preserves shape" `Quick (fun () ->
        let a =
          Nx.create Nx.float32 [| 3; 2; 2 |]
            [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
        in
        let b = Nx.map (fun x -> Nx.add_s x 1.0) a in
        check_t "map values" [| 3; 2; 2 |]
          [| 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13. |]
          b);
    test_case "iter" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let sum = ref (Nx.scalar Nx.float32 0.0) in
        Nx.iter (fun x -> sum := Nx.add !sum x) a;
        check (float 0.01) "iter sum" 10.0 (Nx.item [] !sum));
    test_case "fold" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let sum =
          Nx.fold (fun acc x -> Nx.add acc x) (Nx.scalar Nx.float32 0.0) a
        in
        check (float 0.01) "fold sum" 21.0 (Nx.item [] sum));
    test_case "fold product" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let prod =
          Nx.fold (fun acc x -> Nx.mul acc x) (Nx.scalar Nx.float32 1.0) a
        in
        check (float 0.01) "fold product" 24.0 (Nx.item [] prod));
    test_case "fold max" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 5.; 3.; 2.; 6.; 4. |] in
        let max_val =
          Nx.fold
            (fun acc x -> Nx.maximum acc x)
            (Nx.scalar Nx.float32 neg_infinity)
            a
        in
        check (float 0.01) "fold max" 6.0 (Nx.item [] max_val));
    test_case "map_item" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let b = Nx.map_item (fun x -> x *. 2.0) a in
        check_t "map_item double" [| 2; 3 |] [| 2.; 4.; 6.; 8.; 10.; 12. |] b);
    test_case "iter_item" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let sum = ref 0.0 in
        Nx.iter_item (fun x -> sum := !sum +. x) a;
        check (float 0.01) "iter_item sum" 10.0 !sum);
    test_case "fold_item" `Quick (fun () ->
        let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let sum = Nx.fold_item (fun acc x -> acc +. x) 0.0 a in
        check (float 0.01) "fold_item sum" 21.0 sum);
  ]

let suite =
  [
    ("Sanity :: Creation Functions", creation_tests);
    ("Sanity :: Range Generation", range_generation_tests);
    ("Sanity :: Property Access", property_access_tests);
    ("Sanity :: Data Manipulation", data_manipulation_tests);
    ("Sanity :: Element-wise Binary Operations", element_wise_binary_tests);
    ("Sanity :: Comparison Operations", comparison_tests);
    ("Sanity :: Element-wise Unary Operations", element_wise_unary_tests);
    ("Sanity :: Bitwise Operations", bitwise_tests);
    ("Sanity :: Logical Operations", logical_tests);
    ("Sanity :: Special Value Checks", special_value_tests);
    ("Sanity :: Ternary Operations", ternary_tests);
    ("Sanity :: Reduction Operations", reduction_tests);
    ("Sanity :: Shape Manipulation", shape_manipulation_tests);
    ("Sanity :: Array Combination", array_combination_tests);
    ("Sanity :: Type Conversion", type_conversion_tests);
    ("Sanity :: Indexing and Slicing", indexing_slicing_tests);
    ("Sanity :: Linear Algebra", linear_algebra_tests);
    ("Sanity :: Neural Network", neural_network_tests);
    ("Sanity :: Random Number Generation", random_tests);
    ("Sanity :: Sorting and Searching", sorting_searching_tests);
    ("Sanity :: Display and Formatting", display_formatting_tests);
    ("Sanity :: Higher-order Functions", higher_order_tests);
  ]

let () = Alcotest.run "Nx Sanity" suite
