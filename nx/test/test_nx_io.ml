open Alcotest

(* Helper functions *)
let temp_file prefix suffix = Filename.temp_file prefix suffix

let array_approx_equal ?(eps = 1e-6) a b =
  try
    let a_flat = Nx.flatten a in
    let b_flat = Nx.flatten b in
    let diff = Nx.sub a_flat b_flat in
    let abs_diff = Nx.abs diff in
    (* Get maximum value - reshape to scalar and extract *)
    let max_diff = Nx.max abs_diff ~axes:[ 0 ] ~keepdims:false in
    let max_val = Nx.item [] max_diff in
    max_val < eps
  with _ -> false

let check_array_approx msg ?(eps = 1e-6) expected actual =
  if not (array_approx_equal ~eps expected actual) then
    Alcotest.fail (Printf.sprintf "%s: arrays not approximately equal" msg)

(* Test NPY format *)
let test_npy_save_load_float32 () =
  let test_data = Nx.arange Nx.float32 0 12 1 |> Nx.reshape [| 3; 4 |] in
  let path = temp_file "test_npy_" ".npy" in

  (* Save the data *)
  Nx_io.save_npy path test_data;

  (* Load it back *)
  let loaded = Nx_io.load_npy path in
  let loaded_f32 = Nx_io.as_float32 loaded in

  (* Check shape and values *)
  check (array int) "loaded shape" [| 3; 4 |] (Nx.shape loaded_f32);
  check_array_approx "loaded values" test_data loaded_f32;

  (* Clean up *)
  Sys.remove path

let test_npy_save_load_int64 () =
  let test_data = Nx.arange Nx.int64 0 20 2 |> Nx.reshape [| 2; 5 |] in
  let path = temp_file "test_npy_" ".npy" in

  (* Save the data *)
  Nx_io.save_npy path test_data;

  (* Load it back *)
  let loaded = Nx_io.load_npy path in
  let loaded_i64 = Nx_io.as_int64 loaded in

  (* Check shape *)
  check (array int) "loaded shape" [| 2; 5 |] (Nx.shape loaded_i64);

  (* Check values *)
  for i = 0 to 1 do
    for j = 0 to 4 do
      let expected = (i * 10) + (j * 2) in
      let actual = Nx.item [ i; j ] loaded_i64 |> Int64.to_int in
      check int (Printf.sprintf "value at [%d, %d]" i j) expected actual
    done
  done;

  (* Clean up *)
  Sys.remove path

let test_npy_overwrite_protection () =
  let test_data = Nx.ones Nx.float32 [| 2; 2 |] in
  let path = temp_file "test_npy_" ".npy" in

  (* Save initial file *)
  Nx_io.save_npy path test_data;

  (* Try to save with overwrite=false - should fail *)
  let result = Nx_io.Safe.save_npy ~overwrite:false path test_data in
  check bool "overwrite protection" true (Result.is_error result);

  (* Save with overwrite=true - should succeed *)
  Nx_io.save_npy ~overwrite:true path test_data;

  (* Clean up *)
  Sys.remove path

(* Test NPZ format *)
let test_npz_save_load_multiple () =
  let weights = Nx.randn Nx.float32 [| 5; 3 |] in
  let bias = Nx.zeros Nx.float32 [| 3 |] in
  let scale = Nx.ones Nx.float64 [| 3 |] in
  let path = temp_file "test_npz_" ".npz" in

  (* Save multiple arrays *)
  Nx_io.save_npz path
    [
      ("weights", Nx_io.P weights);
      ("bias", Nx_io.P bias);
      ("scale", Nx_io.P scale);
    ];

  (* Load all back *)
  let archive = Nx_io.load_npz path in

  (* Check we got all arrays *)
  check int "number of arrays" 3 (Hashtbl.length archive);
  check bool "has weights" true (Hashtbl.mem archive "weights");
  check bool "has bias" true (Hashtbl.mem archive "bias");
  check bool "has scale" true (Hashtbl.mem archive "scale");

  (* Check shapes *)
  let loaded_weights = Hashtbl.find archive "weights" |> Nx_io.as_float32 in
  let loaded_bias = Hashtbl.find archive "bias" |> Nx_io.as_float32 in
  let loaded_scale = Hashtbl.find archive "scale" |> Nx_io.as_float64 in

  check (array int) "weights shape" [| 5; 3 |] (Nx.shape loaded_weights);
  check (array int) "bias shape" [| 3 |] (Nx.shape loaded_bias);
  check (array int) "scale shape" [| 3 |] (Nx.shape loaded_scale);

  (* Clean up *)
  Sys.remove path

let test_npz_load_member () =
  let array1 = Nx.arange Nx.float32 0 10 1 in
  let array2 = Nx.arange Nx.int32 10 20 1 in
  let array3 = Nx.ones Nx.float64 [| 2; 3 |] in
  let path = temp_file "test_npz_" ".npz" in

  (* Save arrays *)
  Nx_io.save_npz path
    [
      ("array1", Nx_io.P array1);
      ("array2", Nx_io.P array2);
      ("array3", Nx_io.P array3);
    ];

  (* Load specific members *)
  let loaded1 = Nx_io.load_npz_member ~name:"array1" path |> Nx_io.as_float32 in
  let loaded2 = Nx_io.load_npz_member ~name:"array2" path |> Nx_io.as_int32 in
  let loaded3 = Nx_io.load_npz_member ~name:"array3" path |> Nx_io.as_float64 in

  check (array int) "array1 shape" [| 10 |] (Nx.shape loaded1);
  check (array int) "array2 shape" [| 10 |] (Nx.shape loaded2);
  check (array int) "array3 shape" [| 2; 3 |] (Nx.shape loaded3);

  (* Test loading non-existent member *)
  let result = Nx_io.Safe.load_npz_member ~name:"nonexistent" path in
  check bool "missing member error" true (Result.is_error result);

  (* Clean up *)
  Sys.remove path

(* Test SafeTensors format *)
let test_safetensors_save_load () =
  let weights = Nx.randn Nx.float32 [| 10; 5 |] in
  let bias = Nx.zeros Nx.float32 [| 5 |] in
  let embeddings = Nx.randn Nx.float32 [| 100; 64 |] in
  let path = temp_file "test_safetensors_" ".safetensors" in

  (* Save tensors *)
  Nx_io.save_safetensor path
    [
      ("model.weights", Nx_io.P weights);
      ("model.bias", Nx_io.P bias);
      ("embeddings", Nx_io.P embeddings);
    ];

  (* Load back *)
  let archive = Nx_io.load_safetensor path in

  (* Check we got all tensors *)
  check int "number of tensors" 3 (Hashtbl.length archive);
  check bool "has weights" true (Hashtbl.mem archive "model.weights");
  check bool "has bias" true (Hashtbl.mem archive "model.bias");
  check bool "has embeddings" true (Hashtbl.mem archive "embeddings");

  (* Check shapes *)
  let loaded_weights =
    Hashtbl.find archive "model.weights" |> Nx_io.as_float32
  in
  let loaded_bias = Hashtbl.find archive "model.bias" |> Nx_io.as_float32 in
  let loaded_embeddings =
    Hashtbl.find archive "embeddings" |> Nx_io.as_float32
  in

  check (array int) "weights shape" [| 10; 5 |] (Nx.shape loaded_weights);
  check (array int) "bias shape" [| 5 |] (Nx.shape loaded_bias);
  check (array int) "embeddings shape" [| 100; 64 |]
    (Nx.shape loaded_embeddings);

  (* Check values are preserved *)
  check_array_approx "weights values" weights loaded_weights;
  check_array_approx "bias values" bias loaded_bias;
  check_array_approx "embeddings values" embeddings loaded_embeddings;

  (* Clean up *)
  Sys.remove path

let test_safetensors_different_dtypes () =
  let path = temp_file "test_safetensors_dtypes_" ".safetensors" in

  (* Create arrays of different types *)
  let f32_data = Nx.arange Nx.float32 0 10 1 in
  let f64_data = Nx.arange Nx.float64 10 20 1 in
  let i32_data = Nx.arange Nx.int32 20 30 1 in

  (* Save *)
  Nx_io.save_safetensor path
    [
      ("float32_array", Nx_io.P f32_data);
      ("float64_array", Nx_io.P f64_data);
      ("int32_array", Nx_io.P i32_data);
    ];

  (* Load and verify *)
  let archive = Nx_io.load_safetensor path in

  let loaded_f32 = Hashtbl.find archive "float32_array" |> Nx_io.as_float32 in
  let loaded_f64 = Hashtbl.find archive "float64_array" |> Nx_io.as_float64 in
  let loaded_i32 = Hashtbl.find archive "int32_array" |> Nx_io.as_int32 in

  check_array_approx "float32 values" f32_data loaded_f32;
  check_array_approx "float64 values" ~eps:1e-10 f64_data loaded_f64;

  (* Check int32 values *)
  for i = 0 to 9 do
    let expected = 20 + i in
    let actual = Nx.item [ i ] loaded_i32 |> Int32.to_int in
    check int (Printf.sprintf "int32 value at [%d]" i) expected actual
  done;

  (* Clean up *)
  Sys.remove path

(* Test dtype conversions *)
let test_dtype_conversions () =
  (* Create test data *)
  let original = Nx.arange Nx.float32 0 10 1 in
  let path = temp_file "test_dtype_" ".npy" in

  (* Save and load *)
  Nx_io.save_npy path original;
  let loaded = Nx_io.load_npy path in

  (* Test successful conversion *)
  let as_f32 = Nx_io.as_float32 loaded in
  check_array_approx "float32 conversion" original as_f32;

  (* Test failing conversion (wrong dtype) *)
  let should_fail () = ignore (Nx_io.as_int32 loaded) in
  check_raises "wrong dtype conversion" (Failure "Unsupported dtype")
    should_fail;

  (* Clean up *)
  Sys.remove path

(* Test edge cases *)
let test_empty_arrays () =
  (* Empty array *)
  let empty = Nx.zeros Nx.float32 [| 0 |] in
  let path = temp_file "test_empty_" ".npy" in

  Nx_io.save_npy path empty;
  let loaded = Nx_io.load_npy path in
  let loaded_f32 = Nx_io.as_float32 loaded in

  check (array int) "empty array shape" [| 0 |] (Nx.shape loaded_f32);

  (* Clean up *)
  Sys.remove path

let test_large_arrays () =
  (* Large array (but not too large for tests) *)
  let large = Nx.ones Nx.float32 [| 100; 100 |] in
  let path = temp_file "test_large_" ".npy" in

  Nx_io.save_npy path large;
  let loaded = Nx_io.load_npy path in
  let loaded_f32 = Nx_io.as_float32 loaded in

  check (array int) "large array shape" [| 100; 100 |] (Nx.shape loaded_f32);

  (* Verify all values are 1 - sum and check *)
  let sum = Nx.sum loaded_f32 ~axes:[ 0; 1 ] ~keepdims:false in
  let sum_val = Nx.item [] sum in
  check (float 1e-3) "large array sum" 10000.0 sum_val;

  (* Clean up *)
  Sys.remove path

let test_high_dimensional_arrays () =
  (* 5D array *)
  let high_dim =
    Nx.arange Nx.float32 0 120 1 |> Nx.reshape [| 2; 3; 4; 5; 1 |]
  in
  let path = temp_file "test_highdim_" ".npy" in

  Nx_io.save_npy path high_dim;
  let loaded = Nx_io.load_npy path in
  let loaded_f32 = Nx_io.as_float32 loaded in

  check (array int) "5D array shape" [| 2; 3; 4; 5; 1 |] (Nx.shape loaded_f32);
  check_array_approx "5D array values" high_dim loaded_f32;

  (* Clean up *)
  Sys.remove path

(* Test SafeTensors with float16 and bfloat16 round-trip *)
let test_safetensors_float16_roundtrip () =
  let test_data = Nx.full Nx.float16 [| 2; 3 |] 1.5 in
  let path = temp_file "test_safetensors_f16_" ".safetensors" in

  (* Save the data *)
  Nx_io.save_safetensor path [("test_f16", Nx_io.P test_data)];

  (* Load it back *)
  let archive = Nx_io.load_safetensor path in
  let loaded = Hashtbl.find archive "test_f16" |> Nx_io.as_float16 in

  (* Check shape and values *)
  check (array int) "float16 shape" [| 2; 3 |] (Nx.shape loaded);
  check_array_approx "float16 values" ~eps:1e-3 test_data loaded;

  (* Clean up *)
  Sys.remove path

let test_safetensors_bfloat16_roundtrip () =
  let test_data = Nx.full Nx.bfloat16 [| 2; 3 |] 1.5 in
  let path = temp_file "test_safetensors_bf16_" ".safetensors" in

  (* Save the data *)
  Nx_io.save_safetensor path [("test_bf16", Nx_io.P test_data)];

  (* Load it back *)
  let archive = Nx_io.load_safetensor path in
  let loaded = Hashtbl.find archive "test_bf16" |> Nx_io.as_bfloat16 in

  (* Check shape and values *)
  check (array int) "bfloat16 shape" [| 2; 3 |] (Nx.shape loaded);
  check_array_approx "bfloat16 values" ~eps:1e-3 test_data loaded;

  (* Clean up *)
  Sys.remove path

(* Test Safe module *)
let test_safe_module_error_handling () =
  (* Test file not found *)
  let result = Nx_io.Safe.load_npy "/nonexistent/file.npy" in
  check bool "file not found error" true (Result.is_error result);

  (* Test invalid path for save *)
  let data = Nx.ones Nx.float32 [| 2; 2 |] in
  let result = Nx_io.Safe.save_npy "/invalid/path/file.npy" data in
  check bool "invalid save path error" true (Result.is_error result);

  (* Test successful operation *)
  let path = temp_file "test_safe_" ".npy" in
  let result = Nx_io.Safe.save_npy path data in
  check bool "successful save" true (Result.is_ok result);

  let result = Nx_io.Safe.load_npy path in
  check bool "successful load" true (Result.is_ok result);

  (* Clean up *)
  Sys.remove path

let () =
  let open Alcotest in
  run "Nx_io comprehensive tests"
    [
      ( "npy",
        [
          test_case "Save/load float32" `Quick test_npy_save_load_float32;
          test_case "Save/load int64" `Quick test_npy_save_load_int64;
          test_case "Overwrite protection" `Quick test_npy_overwrite_protection;
        ] );
      ( "npz",
        [
          test_case "Save/load multiple arrays" `Quick
            test_npz_save_load_multiple;
          test_case "Load specific member" `Quick test_npz_load_member;
        ] );
      ( "safetensors",
        [
          test_case "Save/load tensors" `Quick test_safetensors_save_load;
          test_case "Different dtypes" `Quick test_safetensors_different_dtypes;
          test_case "Float16 round-trip" `Quick test_safetensors_float16_roundtrip;
          test_case "Bfloat16 round-trip" `Quick test_safetensors_bfloat16_roundtrip;
        ] );
      ( "dtype_conversions",
        [ test_case "Basic conversions" `Quick test_dtype_conversions ] );
      ( "edge_cases",
        [
          test_case "Empty arrays" `Quick test_empty_arrays;
          test_case "Large arrays" `Quick test_large_arrays;
          test_case "High dimensional arrays" `Quick
            test_high_dimensional_arrays;
        ] );
      ( "safe_module",
        [ test_case "Error handling" `Quick test_safe_module_error_handling ] );
    ]
