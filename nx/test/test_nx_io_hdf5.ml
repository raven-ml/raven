open Alcotest

let test_hdf5_available () =
  (* Just check if HDF5 is available *)
  let available = Nx_io.hdf5_available in
  Printf.printf "HDF5 support: %s\n"
    (if available then "available" else "not available");
  check bool "HDF5 availability check" true true

let test_save_load_single () =
  if not Nx_io.hdf5_available then skip ()
  else
    let test_data = Nx.arange Nx.float32 0 100 1 |> Nx.reshape [| 10; 10 |] in
    let path = "/tmp/test_single.h5" in

    (* Save the data *)
    Nx_io.save_h5 test_data ~path ~dataset:"test_data";

    (* Load it back *)
    let (Nx_io.P loaded) = Nx_io.load_h5 ~path ~dataset:"test_data" in
    let loaded_f32 = Nx_io.to_float32 (Nx_io.P loaded) in

    (* Check shape *)
    let shape = Nx.shape loaded_f32 in
    check (array int) "loaded shape" [| 10; 10 |] shape;

    (* Clean up *)
    Sys.remove path

let test_save_load_multiple () =
  if not Nx_io.hdf5_available then skip ()
  else
    let weights = Nx.randn Nx.float32 [| 5; 3 |] in
    let bias = Nx.zeros Nx.float32 [| 3 |] in
    let path = "/tmp/test_multiple.h5" in

    (* Save multiple arrays *)
    Nx_io.save_h5_all
      [ ("weights", Nx_io.P weights); ("bias", Nx_io.P bias) ]
      path;

    (* Load all back *)
    let archive = Nx_io.load_h5_all path in

    (* Check we got datasets *)
    check int "number of datasets" 2 (Hashtbl.length archive);
    check bool "has weights" true (Hashtbl.mem archive "weights");
    check bool "has bias" true (Hashtbl.mem archive "bias");

    (* Clean up *)
    Sys.remove path

let test_different_dtypes () =
  if not Nx_io.hdf5_available then skip ()
  else
    let path = "/tmp/test_dtypes.h5" in

    (* Test float32 *)
    let f32_data = Nx.ones Nx.float32 [| 2; 3 |] in
    Nx_io.save_h5 f32_data ~path ~dataset:"float32";
    let (Nx_io.P f32_loaded) = Nx_io.load_h5 ~path ~dataset:"float32" in
    let f32_conv = Nx_io.to_float32 (Nx_io.P f32_loaded) in
    check (array int) "float32 shape" [| 2; 3 |] (Nx.shape f32_conv);

    (* Test int32 *)
    let i32_data = Nx.ones Nx.int32 [| 3; 4 |] in
    Nx_io.save_h5 i32_data ~path ~dataset:"int32";
    let (Nx_io.P i32_loaded) = Nx_io.load_h5 ~path ~dataset:"int32" in
    let i32_conv = Nx_io.to_int32 (Nx_io.P i32_loaded) in
    check (array int) "int32 shape" [| 3; 4 |] (Nx.shape i32_conv);

    (* Clean up *)
    Sys.remove path

let () =
  let open Alcotest in
  run "Nx_io HDF5 tests"
    [
      ( "hdf5",
        [
          test_case "HDF5 availability" `Quick test_hdf5_available;
          test_case "Save/load single dataset" `Quick test_save_load_single;
          test_case "Save/load multiple datasets" `Quick test_save_load_multiple;
          test_case "Different dtypes" `Quick test_different_dtypes;
        ] );
    ]
