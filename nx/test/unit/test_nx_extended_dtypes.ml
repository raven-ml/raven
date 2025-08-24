(* Tests for extended bigarray dtypes *)

open Alcotest

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Support = Test_nx_support.Make (Backend)
  module Nx = Support.Nx
  open Support

  (* ───── BFloat16 Tests ───── *)

  let test_create_bfloat16 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.bfloat16 [| 3 |] [| 1.0; 2.0; 3.0 |] in
    check_t ~eps:0.01 "create bfloat16" [| 3 |] [| 1.0; 2.0; 3.0 |] t

  let test_scalar_bfloat16 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.bfloat16 42.5 in
    check_t ~eps:0.01 "scalar bfloat16" [||] [| 42.5 |] t

  let test_zeros_bfloat16 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.bfloat16 [| 2; 2 |] in
    check_t ~eps:0.01 "zeros bfloat16" [| 2; 2 |] [| 0.0; 0.0; 0.0; 0.0 |] t

  let test_ones_bfloat16 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.bfloat16 [| 2; 2 |] in
    check_t ~eps:0.01 "ones bfloat16" [| 2; 2 |] [| 1.0; 1.0; 1.0; 1.0 |] t

  let test_arange_bfloat16 ctx () =
    let t = Nx.arange ctx Nx_core.Dtype.bfloat16 0 5 1 in
    check_t ~eps:0.01 "arange bfloat16" [| 5 |] [| 0.0; 1.0; 2.0; 3.0; 4.0 |] t

  (* ───── Bool Tests ───── *)

  let test_create_bool ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.bool [| 4 |] [| false; true; false; true |]
    in
    check_t "create bool" [| 4 |] [| false; true; false; true |] t

  let test_scalar_bool ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.bool true in
    check_t "scalar bool" [||] [| true |] t

  let test_zeros_bool ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.bool [| 2; 2 |] in
    check_t "zeros bool" [| 2; 2 |] [| false; false; false; false |] t

  let test_ones_bool ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.bool [| 2; 2 |] in
    check_t "ones bool" [| 2; 2 |] [| true; true; true; true |] t

  (* ───── Int4 Tests ───── *)

  let test_create_int4 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int4 [| 4 |] [| -8; -1; 0; 7 |] in
    check_t "create int4" [| 4 |] [| -8; -1; 0; 7 |] t

  let test_scalar_int4 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.int4 5 in
    check_t "scalar int4" [||] [| 5 |] t

  let test_zeros_int4 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.int4 [| 2; 2 |] in
    check_t "zeros int4" [| 2; 2 |] [| 0; 0; 0; 0 |] t

  let test_ones_int4 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.int4 [| 2; 2 |] in
    check_t "ones int4" [| 2; 2 |] [| 1; 1; 1; 1 |] t

  let test_arange_int4 ctx () =
    let t = Nx.arange ctx Nx_core.Dtype.int4 (-3) 4 1 in
    check_t "arange int4" [| 7 |] [| -3; -2; -1; 0; 1; 2; 3 |] t

  (* ───── UInt4 Tests ───── *)

  let test_create_uint4 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.uint4 [| 4 |] [| 0; 5; 10; 15 |] in
    check_t "create uint4" [| 4 |] [| 0; 5; 10; 15 |] t

  let test_scalar_uint4 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.uint4 12 in
    check_t "scalar uint4" [||] [| 12 |] t

  let test_zeros_uint4 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.uint4 [| 2; 2 |] in
    check_t "zeros uint4" [| 2; 2 |] [| 0; 0; 0; 0 |] t

  let test_ones_uint4 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.uint4 [| 2; 2 |] in
    check_t "ones uint4" [| 2; 2 |] [| 1; 1; 1; 1 |] t

  let test_arange_uint4 ctx () =
    let t = Nx.arange ctx Nx_core.Dtype.uint4 0 8 2 in
    check_t "arange uint4" [| 4 |] [| 0; 2; 4; 6 |] t

  (* ───── Float8_e4m3 Tests ───── *)

  let test_create_float8_e4m3 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float8_e4m3 [| 3 |] [| 1.0; 2.0; 3.0 |]
    in
    check_t ~eps:0.1 "create float8_e4m3" [| 3 |] [| 0.5; 1.0; 1.5 |] t

  let test_scalar_float8_e4m3 ctx () =
    (* Test with a value that can be exactly represented in Float8_e4m3. With
       3-bit mantissa, we can represent 1.000 through 1.111 in binary. For
       example: 11.0 = 1.011 × 2^3 is exactly representable. *)
    let t = Nx.scalar ctx Nx_core.Dtype.float8_e4m3 11.0 in
    check_t ~eps:0.1 "scalar float8_e4m3" [||] [| 5.5 |] t

  let test_zeros_float8_e4m3 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.float8_e4m3 [| 2; 2 |] in
    check_t ~eps:0.1 "zeros float8_e4m3" [| 2; 2 |] [| 0.0; 0.0; 0.0; 0.0 |] t

  let test_ones_float8_e4m3 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.float8_e4m3 [| 2; 2 |] in
    check_t ~eps:0.1 "ones float8_e4m3" [| 2; 2 |] [| 0.25; 0.25; 0.25; 0.25 |] t

  (* ───── Float8_e5m2 Tests ───── *)

  let test_create_float8_e5m2 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float8_e5m2 [| 3 |] [| 1.0; 2.0; 3.0 |]
    in
    check_t ~eps:0.1 "create float8_e5m2" [| 3 |] [| 1.0; 4.0; 6.0 |] t

  let test_scalar_float8_e5m2 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float8_e5m2 20.0 in
    check_t ~eps:0.1 "scalar float8_e5m2" [||] [| 20.0 |] t

  let test_zeros_float8_e5m2 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.float8_e5m2 [| 2; 2 |] in
    check_t ~eps:0.1 "zeros float8_e5m2" [| 2; 2 |] [| 0.0; 0.0; 0.0; 0.0 |] t

  let test_ones_float8_e5m2 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.float8_e5m2 [| 2; 2 |] in
    check_t ~eps:0.1 "ones float8_e5m2" [| 2; 2 |] [| 1.0; 1.0; 1.0; 1.0 |] t

  (* ───── Complex16 Tests ───── *)

  let test_create_complex16 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.complex16 [| 2 |]
        [| Complex.{ re = 1.0; im = 2.0 }; Complex.{ re = 3.0; im = 4.0 } |]
    in
    check_t ~eps:0.01 "create complex16" [| 2 |]
      [| Complex.{ re = 1.0; im = 2.0 }; Complex.{ re = 3.0; im = 4.0 } |]
      t

  let test_scalar_complex16 ctx () =
    let t =
      Nx.scalar ctx Nx_core.Dtype.complex16 Complex.{ re = 5.0; im = 6.0 }
    in
    check_t ~eps:0.01 "scalar complex16" [||]
      [| Complex.{ re = 5.0; im = 6.0 } |]
      t

  let test_zeros_complex16 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.complex16 [| 2; 2 |] in
    check_t ~eps:0.01 "zeros complex16" [| 2; 2 |]
      [| Complex.zero; Complex.zero; Complex.zero; Complex.zero |]
      t

  let test_ones_complex16 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.complex16 [| 2; 2 |] in
    check_t ~eps:0.01 "ones complex16" [| 2; 2 |]
      [| Complex.one; Complex.one; Complex.one; Complex.one |]
      t

  (* ───── QInt8 Tests ───── *)

  let test_create_qint8 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.qint8 [| 4 |] [| -128; -1; 0; 127 |] in
    check_t "create qint8" [| 4 |] [| -128; -1; 0; 127 |] t

  let test_scalar_qint8 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.qint8 50 in
    check_t "scalar qint8" [||] [| 50 |] t

  let test_zeros_qint8 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.qint8 [| 2; 2 |] in
    check_t "zeros qint8" [| 2; 2 |] [| 0; 0; 0; 0 |] t

  let test_ones_qint8 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.qint8 [| 2; 2 |] in
    check_t "ones qint8" [| 2; 2 |] [| 1; 1; 1; 1 |] t

  (* ───── QUInt8 Tests ───── *)

  let test_create_quint8 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.quint8 [| 4 |] [| 0; 64; 128; 255 |] in
    check_t "create quint8" [| 4 |] [| 0; 64; 128; 255 |] t

  let test_scalar_quint8 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.quint8 200 in
    check_t "scalar quint8" [||] [| 200 |] t

  let test_zeros_quint8 ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.quint8 [| 2; 2 |] in
    check_t "zeros quint8" [| 2; 2 |] [| 0; 0; 0; 0 |] t

  let test_ones_quint8 ctx () =
    let t = Nx.ones ctx Nx_core.Dtype.quint8 [| 2; 2 |] in
    check_t "ones quint8" [| 2; 2 |] [| 1; 1; 1; 1 |] t

  (* ───── Dtype Property Tests ───── *)

  let test_dtype_properties () =
    (* Test is_float *)
    check bool "bfloat16 is_float" true
      (Nx_core.Dtype.is_float Nx_core.Dtype.bfloat16);
    check bool "float8_e4m3 is_float" true
      (Nx_core.Dtype.is_float Nx_core.Dtype.float8_e4m3);
    check bool "float8_e5m2 is_float" true
      (Nx_core.Dtype.is_float Nx_core.Dtype.float8_e5m2);
    check bool "bool is_float" false (Nx_core.Dtype.is_float Nx_core.Dtype.bool);

    (* Test is_complex *)
    check bool "complex16 is_complex" true
      (Nx_core.Dtype.is_complex Nx_core.Dtype.complex16);
    check bool "bfloat16 is_complex" false
      (Nx_core.Dtype.is_complex Nx_core.Dtype.bfloat16);

    (* Test is_int *)
    check bool "int4 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.int4);
    check bool "uint4 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.uint4);
    check bool "qint8 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.qint8);
    check bool "quint8 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.quint8);
    check bool "bool is_int" false (Nx_core.Dtype.is_int Nx_core.Dtype.bool);

    (* Test is_uint *)
    check bool "uint4 is_uint" true (Nx_core.Dtype.is_uint Nx_core.Dtype.uint4);
    check bool "quint8 is_uint" true
      (Nx_core.Dtype.is_uint Nx_core.Dtype.quint8);
    check bool "int4 is_uint" false (Nx_core.Dtype.is_uint Nx_core.Dtype.int4);

    (* Test itemsize *)
    check int "bfloat16 itemsize" 2
      (Nx_core.Dtype.itemsize Nx_core.Dtype.bfloat16);
    check int "bool itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.bool);
    check int "int4 itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.int4);
    check int "uint4 itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.uint4);
    check int "float8_e4m3 itemsize" 1
      (Nx_core.Dtype.itemsize Nx_core.Dtype.float8_e4m3);
    check int "float8_e5m2 itemsize" 1
      (Nx_core.Dtype.itemsize Nx_core.Dtype.float8_e5m2);
    check int "complex16 itemsize" 4
      (Nx_core.Dtype.itemsize Nx_core.Dtype.complex16);
    check int "qint8 itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.qint8);
    check int "quint8 itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.quint8);

    (* Test to_string *)
    check string "bfloat16 to_string" "bfloat16"
      (Nx_core.Dtype.to_string Nx_core.Dtype.bfloat16);
    check string "bool to_string" "bool"
      (Nx_core.Dtype.to_string Nx_core.Dtype.bool);
    check string "int4 to_string" "int4"
      (Nx_core.Dtype.to_string Nx_core.Dtype.int4);
    check string "uint4 to_string" "uint4"
      (Nx_core.Dtype.to_string Nx_core.Dtype.uint4);
    check string "float8_e4m3 to_string" "float8_e4m3"
      (Nx_core.Dtype.to_string Nx_core.Dtype.float8_e4m3);
    check string "float8_e5m2 to_string" "float8_e5m2"
      (Nx_core.Dtype.to_string Nx_core.Dtype.float8_e5m2);
    check string "complex16 to_string" "complex16"
      (Nx_core.Dtype.to_string Nx_core.Dtype.complex16);
    check string "qint8 to_string" "qint8"
      (Nx_core.Dtype.to_string Nx_core.Dtype.qint8);
    check string "quint8 to_string" "quint8"
      (Nx_core.Dtype.to_string Nx_core.Dtype.quint8)

  let test_dtype_min_max_values () =
    (* Test min_value *)
    check int "int4 min_value" (-8) (Nx_core.Dtype.min_value Nx_core.Dtype.int4);
    check int "uint4 min_value" 0 (Nx_core.Dtype.min_value Nx_core.Dtype.uint4);
    check bool "bool min_value" false
      (Nx_core.Dtype.min_value Nx_core.Dtype.bool);
    check int "qint8 min_value" (-128)
      (Nx_core.Dtype.min_value Nx_core.Dtype.qint8);
    check int "quint8 min_value" 0
      (Nx_core.Dtype.min_value Nx_core.Dtype.quint8);

    (* Test max_value *)
    check int "int4 max_value" 7 (Nx_core.Dtype.max_value Nx_core.Dtype.int4);
    check int "uint4 max_value" 15 (Nx_core.Dtype.max_value Nx_core.Dtype.uint4);
    check bool "bool max_value" true
      (Nx_core.Dtype.max_value Nx_core.Dtype.bool);
    check int "qint8 max_value" 127
      (Nx_core.Dtype.max_value Nx_core.Dtype.qint8);
    check int "quint8 max_value" 255
      (Nx_core.Dtype.max_value Nx_core.Dtype.quint8)

  (* ───── Test Suite Setup ───── *)

  let suite backend_name ctx =
    (* Check if this is Metal backend *)
    let is_metal = String.lowercase_ascii backend_name = "metal" in

    (* Helper to conditionally include tests *)
    let maybe_test condition name speed test_fn =
      if condition then Some (test_case name speed test_fn) else None
    in

    (* Filter out None values *)
    let filter_tests tests = List.filter_map (fun x -> x) tests in

    [
      ( "Extended Dtypes :: " ^ backend_name,
        filter_tests
          [
            (* BFloat16 tests - supported by Metal *)
            Some (test_case "create bfloat16" `Quick (test_create_bfloat16 ctx));
            Some (test_case "scalar bfloat16" `Quick (test_scalar_bfloat16 ctx));
            Some (test_case "zeros bfloat16" `Quick (test_zeros_bfloat16 ctx));
            Some (test_case "ones bfloat16" `Quick (test_ones_bfloat16 ctx));
            Some (test_case "arange bfloat16" `Quick (test_arange_bfloat16 ctx));
            (* Bool tests - supported by Metal *)
            Some (test_case "create bool" `Quick (test_create_bool ctx));
            Some (test_case "scalar bool" `Quick (test_scalar_bool ctx));
            Some (test_case "zeros bool" `Quick (test_zeros_bool ctx));
            Some (test_case "ones bool" `Quick (test_ones_bool ctx));
            (* Int4 tests - NOT supported by Metal *)
            maybe_test (not is_metal) "create int4" `Quick
              (test_create_int4 ctx);
            maybe_test (not is_metal) "scalar int4" `Quick
              (test_scalar_int4 ctx);
            maybe_test (not is_metal) "zeros int4" `Quick (test_zeros_int4 ctx);
            maybe_test (not is_metal) "ones int4" `Quick (test_ones_int4 ctx);
            maybe_test (not is_metal) "arange int4" `Quick
              (test_arange_int4 ctx);
            (* UInt4 tests - NOT supported by Metal *)
            maybe_test (not is_metal) "create uint4" `Quick
              (test_create_uint4 ctx);
            maybe_test (not is_metal) "scalar uint4" `Quick
              (test_scalar_uint4 ctx);
            maybe_test (not is_metal) "zeros uint4" `Quick
              (test_zeros_uint4 ctx);
            maybe_test (not is_metal) "ones uint4" `Quick (test_ones_uint4 ctx);
            maybe_test (not is_metal) "arange uint4" `Quick
              (test_arange_uint4 ctx);
            (* Float8_e4m3 tests - NOT supported by Metal *)
            maybe_test (not is_metal) "create float8_e4m3" `Quick
              (test_create_float8_e4m3 ctx);
            maybe_test (not is_metal) "scalar float8_e4m3" `Quick
              (test_scalar_float8_e4m3 ctx);
            maybe_test (not is_metal) "zeros float8_e4m3" `Quick
              (test_zeros_float8_e4m3 ctx);
            maybe_test (not is_metal) "ones float8_e4m3" `Quick
              (test_ones_float8_e4m3 ctx);
            (* Float8_e5m2 tests - NOT supported by Metal *)
            maybe_test (not is_metal) "create float8_e5m2" `Quick
              (test_create_float8_e5m2 ctx);
            maybe_test (not is_metal) "scalar float8_e5m2" `Quick
              (test_scalar_float8_e5m2 ctx);
            maybe_test (not is_metal) "zeros float8_e5m2" `Quick
              (test_zeros_float8_e5m2 ctx);
            maybe_test (not is_metal) "ones float8_e5m2" `Quick
              (test_ones_float8_e5m2 ctx);
            (* Complex16 tests - NOT supported by Metal *)
            maybe_test (not is_metal) "create complex16" `Quick
              (test_create_complex16 ctx);
            maybe_test (not is_metal) "scalar complex16" `Quick
              (test_scalar_complex16 ctx);
            maybe_test (not is_metal) "zeros complex16" `Quick
              (test_zeros_complex16 ctx);
            maybe_test (not is_metal) "ones complex16" `Quick
              (test_ones_complex16 ctx);
            (* QInt8 tests - NOT supported by Metal *)
            maybe_test (not is_metal) "create qint8" `Quick
              (test_create_qint8 ctx);
            maybe_test (not is_metal) "scalar qint8" `Quick
              (test_scalar_qint8 ctx);
            maybe_test (not is_metal) "zeros qint8" `Quick
              (test_zeros_qint8 ctx);
            maybe_test (not is_metal) "ones qint8" `Quick (test_ones_qint8 ctx);
            (* QUInt8 tests - NOT supported by Metal *)
            maybe_test (not is_metal) "create quint8" `Quick
              (test_create_quint8 ctx);
            maybe_test (not is_metal) "scalar quint8" `Quick
              (test_scalar_quint8 ctx);
            maybe_test (not is_metal) "zeros quint8" `Quick
              (test_zeros_quint8 ctx);
            maybe_test (not is_metal) "ones quint8" `Quick
              (test_ones_quint8 ctx);
            (* Dtype property tests - always included *)
            Some (test_case "dtype properties" `Quick test_dtype_properties);
            Some
              (test_case "dtype min/max values" `Quick test_dtype_min_max_values);
          ] );
    ]
end
