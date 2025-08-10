(* Comprehensive cumsum tests for Nx following the test plan *)

(* Cross-backend test modules *)
module Nx_native_frontend = Nx_core.Make_frontend (Nx_native)
module Nx_metal_frontend = Nx_core.Make_frontend (Nx_metal)
module Nx_c_frontend = Nx_core.Make_frontend (Nx_c)

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Support = Test_nx_support.Make (Backend)
  module Nx = Support.Nx
  open Support

  (* ───── Test Helpers ───── *)

  let test_cumsum_1d ~dtype ~data ~expected ~axis_opt ctx () =
    let input = Nx.create ctx dtype [| Array.length data |] data in
    let result = match axis_opt with
      | Some axis -> Nx.cumsum ~axis input
      | None -> Nx.cumsum input
    in
    check_t "cumsum 1d" [| Array.length expected |] expected result

  let test_cumsum_2d ~dtype ~shape ~data ~expected ~axis ctx () =
    let input = Nx.create ctx dtype shape data in
    let result = Nx.cumsum ~axis input in
    check_t "cumsum 2d" shape expected result

  let test_cumsum_3d ~dtype ~shape ~data ~expected ~axis ctx () =
    let input = Nx.create ctx dtype shape data in
    let result = Nx.cumsum ~axis input in
    check_t "cumsum 3d" shape expected result

  (* ───── 4.1 Basic Functionality Tests ───── *)

  let test_cumsum_1d_int32 ctx () =
    test_cumsum_1d ~dtype:Nx.Int32 
      ~data:[| 1l; 2l; 3l; 4l |] 
      ~expected:[| 1l; 3l; 6l; 10l |] 
      ~axis_opt:(Some 0) ctx ()

  let test_cumsum_1d_float32 ctx () =
    test_cumsum_1d ~dtype:Nx.Float32 
      ~data:[| 1.0; 2.0; 3.0; 4.0 |] 
      ~expected:[| 1.0; 3.0; 6.0; 10.0 |] 
      ~axis_opt:(Some 0) ctx ()

  let test_cumsum_1d_int64 ctx () =
    test_cumsum_1d ~dtype:Nx.Int64 
      ~data:[| 1L; 2L; 3L; 4L |] 
      ~expected:[| 1L; 3L; 6L; 10L |] 
      ~axis_opt:(Some 0) ctx ()

  let test_cumsum_1d_float64 ctx () =
    test_cumsum_1d ~dtype:Nx.Float64 
      ~data:[| 1.0; 2.0; 3.0; 4.0 |] 
      ~expected:[| 1.0; 3.0; 6.0; 10.0 |] 
      ~axis_opt:(Some 0) ctx ()

  let test_cumsum_2d_axis0 ctx () =
    test_cumsum_2d ~dtype:Nx.Int32 
      ~shape:[| 2; 3 |] 
      ~data:[| 1l; 2l; 3l; 4l; 5l; 6l |] 
      ~expected:[| 1l; 2l; 3l; 5l; 7l; 9l |] 
      ~axis:0 ctx ()

  let test_cumsum_2d_axis1 ctx () =
    test_cumsum_2d ~dtype:Nx.Int32 
      ~shape:[| 2; 3 |] 
      ~data:[| 1l; 2l; 3l; 4l; 5l; 6l |] 
      ~expected:[| 1l; 3l; 6l; 4l; 9l; 15l |] 
      ~axis:1 ctx ()

  let test_cumsum_3d_axis0 ctx () =
    test_cumsum_3d ~dtype:Nx.Int32 
      ~shape:[| 2; 2; 2 |] 
      ~data:[| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l |] 
      ~expected:[| 1l; 2l; 3l; 4l; 6l; 8l; 10l; 12l |] 
      ~axis:0 ctx ()

  let test_cumsum_3d_axis1 ctx () =
    test_cumsum_3d ~dtype:Nx.Int32 
      ~shape:[| 2; 2; 2 |] 
      ~data:[| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l |] 
      ~expected:[| 1l; 2l; 4l; 6l; 5l; 6l; 12l; 14l |] 
      ~axis:1 ctx ()

  let test_cumsum_3d_axis2 ctx () =
    test_cumsum_3d ~dtype:Nx.Int32 
      ~shape:[| 2; 2; 2 |] 
      ~data:[| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l |] 
      ~expected:[| 1l; 3l; 3l; 7l; 5l; 11l; 7l; 15l |] 
      ~axis:2 ctx ()

  (* ───── 4.2 Axis Parameter Tests ───── *)

  let test_cumsum_negative_axis ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    let result_neg1 = Nx.cumsum ~axis:(-1) input in
    let expected = [| 1l; 3l; 6l; 4l; 9l; 15l |] in
    check_t "negative axis -1" [| 2; 3 |] expected result_neg1

  let test_cumsum_negative_axis_2 ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    let result_neg2 = Nx.cumsum ~axis:(-2) input in
    let expected = [| 1l; 2l; 3l; 5l; 7l; 9l |] in
    check_t "negative axis -2" [| 2; 3 |] expected result_neg2

  let test_cumsum_out_of_bounds_positive ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    check_invalid_arg "out of bounds positive axis" 
      "cumsum: invalid axis 2 (out of bounds for tensor with rank 2)\nhint: axis must be in range [-2, 1]"
      (fun () -> Nx.cumsum ~axis:2 input)

  let test_cumsum_out_of_bounds_negative ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    check_invalid_arg "out of bounds negative axis" 
      "cumsum: invalid axis -3 (out of bounds for tensor with rank 2)\nhint: axis must be in range [-2, 1]"
      (fun () -> Nx.cumsum ~axis:(-3) input)

  let test_cumsum_no_axis_flatten ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    let result = Nx.cumsum input in
    let expected = [| 1l; 3l; 6l; 10l; 15l; 21l |] in
    check_t "cumsum no axis (flatten)" [| 2; 3 |] expected result

  (* ───── 4.3 Edge Case Tests ───── *)

  let test_cumsum_empty_1d ctx () =
    let input = Nx.create ctx Nx.Int32 [| 0 |] [||] in
    let result = Nx.cumsum ~axis:0 input in
    check_t "cumsum empty 1d" [| 0 |] [||] result

  let test_cumsum_empty_2d ctx () =
    let input = Nx.create ctx Nx.Int32 [| 0; 3 |] [||] in
    let result = Nx.cumsum ~axis:0 input in
    check_t "cumsum empty 2d axis 0" [| 0; 3 |] [||] result

  let test_cumsum_empty_2d_axis1 ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 0 |] [||] in
    let result = Nx.cumsum ~axis:1 input in
    check_t "cumsum empty 2d axis 1" [| 2; 0 |] [||] result

  let test_cumsum_scalar ctx () =
    let input = Nx.scalar ctx Nx.Int32 42l in
    let result = Nx.cumsum input in  (* No axis for scalar *)
    check_t "cumsum scalar" [||] [| 42l |] result

  let test_cumsum_single_element_1d ctx () =
    let input = Nx.create ctx Nx.Int32 [| 1 |] [| 5l |] in
    let result = Nx.cumsum ~axis:0 input in
    check_t "cumsum single element 1d" [| 1 |] [| 5l |] result

  let test_cumsum_single_element_2d_axis0 ctx () =
    let input = Nx.create ctx Nx.Int32 [| 1; 3 |] [| 1l; 2l; 3l |] in
    let result = Nx.cumsum ~axis:0 input in
    check_t "cumsum single element 2d axis 0" [| 1; 3 |] [| 1l; 2l; 3l |] result

  let test_cumsum_single_element_2d_axis1 ctx () =
    let input = Nx.create ctx Nx.Int32 [| 3; 1 |] [| 1l; 2l; 3l |] in
    let result = Nx.cumsum ~axis:1 input in
    check_t "cumsum single element 2d axis 1" [| 3; 1 |] [| 1l; 2l; 3l |] result

  (* ───── 4.4 Data Type Coverage Tests ───── *)

  (* Note: Only testing supported data types (Int32, Int64, Float32, Float64) *)
  (* The following data types are not yet implemented in the native backend: *)
  (* int8, uint8, int16, uint16, float16, complex32, complex64 *)

  let test_cumsum_supported_types ctx () =
    (* Test that all currently supported data types work *)
    let test_int32 = Nx.create ctx Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
    let result_int32 = Nx.cumsum ~axis:0 test_int32 in
    check_t "cumsum int32 supported" [| 3 |] [| 1l; 3l; 6l |] result_int32;

    let test_int64 = Nx.create ctx Nx.int64 [| 3 |] [| 1L; 2L; 3L |] in
    let result_int64 = Nx.cumsum ~axis:0 test_int64 in
    check_t "cumsum int64 supported" [| 3 |] [| 1L; 3L; 6L |] result_int64;

    let test_float32 = Nx.create ctx Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
    let result_float32 = Nx.cumsum ~axis:0 test_float32 in
    check_t "cumsum float32 supported" [| 3 |] [| 1.0; 3.0; 6.0 |] result_float32;

    let test_float64 = Nx.create ctx Nx.float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
    let result_float64 = Nx.cumsum ~axis:0 test_float64 in
    check_t "cumsum float64 supported" [| 3 |] [| 1.0; 3.0; 6.0 |] result_float64

  let test_cumsum_unsupported_types ctx () =
    (* Test that unsupported data types raise appropriate errors *)
    let test_int8 = Nx.create ctx Nx.int8 [| 3 |] [| 1; 2; 3 |] in
    check_failure "cumsum int8 unsupported" 
      "cumsum: data type not yet supported"
      (fun () -> ignore (Nx.cumsum ~axis:0 test_int8));

    let test_uint8 = Nx.create ctx Nx.uint8 [| 3 |] [| 1; 2; 3 |] in
    check_failure "cumsum uint8 unsupported" 
      "cumsum: data type not yet supported"
      (fun () -> ignore (Nx.cumsum ~axis:0 test_uint8))

  (* ───── Test Suite ───── *)

  let basic_functionality_tests _backend_name ctx =
    [
      ("cumsum 1d int32", `Quick, test_cumsum_1d_int32 ctx);
      ("cumsum 1d float32", `Quick, test_cumsum_1d_float32 ctx);
      ("cumsum 1d int64", `Quick, test_cumsum_1d_int64 ctx);
      ("cumsum 1d float64", `Quick, test_cumsum_1d_float64 ctx);
      ("cumsum 2d axis 0", `Quick, test_cumsum_2d_axis0 ctx);
      ("cumsum 2d axis 1", `Quick, test_cumsum_2d_axis1 ctx);
      ("cumsum 3d axis 0", `Quick, test_cumsum_3d_axis0 ctx);
      ("cumsum 3d axis 1", `Quick, test_cumsum_3d_axis1 ctx);
      ("cumsum 3d axis 2", `Quick, test_cumsum_3d_axis2 ctx);
    ]

  let axis_parameter_tests _backend_name ctx =
    [
      ("cumsum negative axis -1", `Quick, test_cumsum_negative_axis ctx);
      ("cumsum negative axis -2", `Quick, test_cumsum_negative_axis_2 ctx);
      ("cumsum out of bounds positive", `Quick, test_cumsum_out_of_bounds_positive ctx);
      ("cumsum out of bounds negative", `Quick, test_cumsum_out_of_bounds_negative ctx);
      ("cumsum no axis flatten", `Quick, test_cumsum_no_axis_flatten ctx);
    ]

  let edge_case_tests _backend_name ctx =
    [
      ("cumsum empty 1d", `Quick, test_cumsum_empty_1d ctx);
      ("cumsum empty 2d axis 0", `Quick, test_cumsum_empty_2d ctx);
      ("cumsum empty 2d axis 1", `Quick, test_cumsum_empty_2d_axis1 ctx);
      ("cumsum scalar", `Quick, test_cumsum_scalar ctx);
      ("cumsum single element 1d", `Quick, test_cumsum_single_element_1d ctx);
      ("cumsum single element 2d axis 0", `Quick, test_cumsum_single_element_2d_axis0 ctx);
      ("cumsum single element 2d axis 1", `Quick, test_cumsum_single_element_2d_axis1 ctx);
    ]

  let data_type_coverage_tests _backend_name ctx =
    [
      ("cumsum supported types", `Quick, test_cumsum_supported_types ctx);
      ("cumsum unsupported types", `Quick, test_cumsum_unsupported_types ctx);
    ]

  (* ───── 8.1 Integration Tests - Composition with Other Operations ───── *)

  let test_cumsum_with_reshape ctx () =
    let input = Nx.create ctx Nx.Int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    let cumsum_result = Nx.cumsum ~axis:0 input in
    let reshaped = Nx.reshape [| 2; 3 |] cumsum_result in
    let expected = [| 1l; 3l; 6l; 10l; 15l; 21l |] in
    check_t "cumsum then reshape" [| 2; 3 |] expected reshaped

  let test_cumsum_with_transpose ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    let cumsum_result = Nx.cumsum ~axis:1 input in
    let transposed = Nx.transpose cumsum_result in
    let expected = [| 1l; 4l; 3l; 9l; 6l; 15l |] in
    check_t "cumsum then transpose" [| 3; 2 |] expected transposed

  let test_cumsum_in_mathematical_expressions ctx () =
    let input = Nx.create ctx Nx.Int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
    let cumsum_result = Nx.cumsum ~axis:0 input in
    let doubled = Nx.mul cumsum_result (Nx.scalar ctx Nx.Int32 2l) in
    let expected = [| 2l; 6l; 12l; 20l |] in
    check_t "cumsum in math expression" [| 4 |] expected doubled

  let test_cumsum_broadcasting_compatibility ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    let cumsum_result = Nx.cumsum ~axis:0 input in
    let broadcast_add = Nx.add cumsum_result (Nx.create ctx Nx.Int32 [| 3 |] [| 10l; 20l; 30l |]) in
    let expected = [| 11l; 22l; 33l; 15l; 27l; 39l |] in
    check_t "cumsum broadcasting" [| 2; 3 |] expected broadcast_add

  let test_chained_cumsum_operations ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    let cumsum_axis0 = Nx.cumsum ~axis:0 input in
    let cumsum_axis1 = Nx.cumsum ~axis:1 cumsum_axis0 in
    let expected = [| 1l; 3l; 6l; 5l; 12l; 21l |] in
    check_t "chained cumsum operations" [| 2; 3 |] expected cumsum_axis1

  (* ───── 8.2 Comprehensive Error Handling Tests ───── *)

  let test_cumsum_invalid_axis_descriptive_error ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3; 4 |] (Array.init 24 (fun i -> Int32.of_int (i + 1))) in
    check_invalid_arg "descriptive error for invalid axis" 
      "cumsum: invalid axis 3 (out of bounds for tensor with rank 3)\nhint: axis must be in range [-3, 2]"
      (fun () -> Nx.cumsum ~axis:3 input)

  let test_cumsum_negative_axis_descriptive_error ctx () =
    let input = Nx.create ctx Nx.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
    check_invalid_arg "descriptive error for negative axis out of bounds" 
      "cumsum: invalid axis -4 (out of bounds for tensor with rank 2)\nhint: axis must be in range [-2, 1]"
      (fun () -> Nx.cumsum ~axis:(-4) input)

  let test_cumsum_error_propagation ctx () =
    (* Test that backend errors are properly propagated *)
    let input = Nx.create ctx Nx.Int32 [| 0 |] [||] in
    check_invalid_arg "error propagation from backend" 
      "cumsum: invalid axis 1 (out of bounds for tensor with rank 1)\nhint: axis must be in range [-1, 0]"
      (fun () -> Nx.cumsum ~axis:1 input)

  let test_cumsum_helpful_error_messages ctx () =
    let input = Nx.create ctx Nx.Int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |] in
    (* Test that error messages include helpful hints *)
    check_invalid_arg "helpful error message with hints" 
      "cumsum: invalid axis 5 (out of bounds for tensor with rank 1)\nhint: axis must be in range [-1, 0]"
      (fun () -> Nx.cumsum ~axis:5 input)

  let integration_tests _backend_name ctx =
    [
      ("cumsum with reshape", `Quick, test_cumsum_with_reshape ctx);
      ("cumsum with transpose", `Quick, test_cumsum_with_transpose ctx);
      ("cumsum in mathematical expressions", `Quick, test_cumsum_in_mathematical_expressions ctx);
      ("cumsum broadcasting compatibility", `Quick, test_cumsum_broadcasting_compatibility ctx);
      ("chained cumsum operations", `Quick, test_chained_cumsum_operations ctx);
    ]

  let error_handling_tests _backend_name ctx =
    [
      ("cumsum invalid axis descriptive error", `Quick, test_cumsum_invalid_axis_descriptive_error ctx);
      ("cumsum negative axis descriptive error", `Quick, test_cumsum_negative_axis_descriptive_error ctx);
      ("cumsum error propagation", `Quick, test_cumsum_error_propagation ctx);
      ("cumsum helpful error messages", `Quick, test_cumsum_helpful_error_messages ctx);
    ]

  let suite backend_name ctx =
    [
      ("Cumsum Basic Functionality", basic_functionality_tests backend_name ctx);
      ("Cumsum Axis Parameters", axis_parameter_tests backend_name ctx);
      ("Cumsum Edge Cases", edge_case_tests backend_name ctx);
      ("Cumsum Data Type Coverage", data_type_coverage_tests backend_name ctx);
      ("Cumsum Integration Tests", integration_tests backend_name ctx);
      ("Cumsum Error Handling", error_handling_tests backend_name ctx);
    ]
end

(* ───── Cross-Backend Validation Tests ───── *)

let test_backends_1d_int32 () =
  let ctx_native = Nx_native.create_context () in
  let ctx_metal = Nx_metal.create_context () in
  let ctx_c = Nx_c.create_context () in
  
  let input_native = Nx_native_frontend.create ctx_native Nx_native_frontend.Int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
  let input_metal = Nx_metal_frontend.create ctx_metal Nx_metal_frontend.Int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
  let input_c = Nx_c_frontend.create ctx_c Nx_c_frontend.Int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
  
  let result_native = Nx_native_frontend.cumsum ~axis:0 input_native in
  let result_metal = Nx_metal_frontend.cumsum ~axis:0 input_metal in
  let result_c = Nx_c_frontend.cumsum ~axis:0 input_c in
  
  let native_data = Nx_native_frontend.unsafe_to_array result_native in
  let metal_data = Nx_metal_frontend.unsafe_to_array result_metal in
  let c_data = Nx_c_frontend.unsafe_to_array result_c in
  
  if native_data <> metal_data then
    Alcotest.failf "Native vs Metal: arrays differ";
  if native_data <> c_data then
    Alcotest.failf "Native vs C: arrays differ"

let test_backends_2d_axis0 () =
  let ctx_native = Nx_native.create_context () in
  let ctx_metal = Nx_metal.create_context () in
  let ctx_c = Nx_c.create_context () in
  
  let input_native = Nx_native_frontend.create ctx_native Nx_native_frontend.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let input_metal = Nx_metal_frontend.create ctx_metal Nx_metal_frontend.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let input_c = Nx_c_frontend.create ctx_c Nx_c_frontend.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  
  let result_native = Nx_native_frontend.cumsum ~axis:0 input_native in
  let result_metal = Nx_metal_frontend.cumsum ~axis:0 input_metal in
  let result_c = Nx_c_frontend.cumsum ~axis:0 input_c in
  
  let native_data = Nx_native_frontend.unsafe_to_array result_native in
  let metal_data = Nx_metal_frontend.unsafe_to_array result_metal in
  let c_data = Nx_c_frontend.unsafe_to_array result_c in
  
  if native_data <> metal_data then
    Alcotest.failf "Native vs Metal: arrays differ";
  if native_data <> c_data then
    Alcotest.failf "Native vs C: arrays differ"

let test_backends_cumsum_with_reshape () =
  let ctx_native = Nx_native.create_context () in
  let ctx_metal = Nx_metal.create_context () in
  let ctx_c = Nx_c.create_context () in
  
  let input_native = Nx_native_frontend.create ctx_native Nx_native_frontend.Int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let input_metal = Nx_metal_frontend.create ctx_metal Nx_metal_frontend.Int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let input_c = Nx_c_frontend.create ctx_c Nx_c_frontend.Int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  
  let cumsum_native = Nx_native_frontend.cumsum ~axis:0 input_native in
  let cumsum_metal = Nx_metal_frontend.cumsum ~axis:0 input_metal in
  let cumsum_c = Nx_c_frontend.cumsum ~axis:0 input_c in
  
  let result_native = Nx_native_frontend.reshape [| 2; 3 |] cumsum_native in
  let result_metal = Nx_metal_frontend.reshape [| 2; 3 |] cumsum_metal in
  let result_c = Nx_c_frontend.reshape [| 2; 3 |] cumsum_c in
  
  let native_data = Nx_native_frontend.unsafe_to_array result_native in
  let metal_data = Nx_metal_frontend.unsafe_to_array result_metal in
  let c_data = Nx_c_frontend.unsafe_to_array result_c in
  
  if native_data <> metal_data then
    Alcotest.failf "Cumsum+reshape: Native vs Metal differ";
  if native_data <> c_data then
    Alcotest.failf "Cumsum+reshape: Native vs C differ"

let cross_backend_validation_tests =
  [
    ("backends 1d int32", `Quick, test_backends_1d_int32);
    ("backends 2d axis 0", `Quick, test_backends_2d_axis0);
    ("backends cumsum with reshape", `Quick, test_backends_cumsum_with_reshape);
  ]