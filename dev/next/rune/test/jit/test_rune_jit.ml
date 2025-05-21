open Alcotest
open Rune_jit
module Jit = Rune_jit.Make (Metal_backend)

(* ───────────────── test helpers ───────────────── *)

let fresh_var_reset () = Var.counter := 0
let make_meta dtype shape = { Ir.dtype = Dtype.Any_Dtype dtype; shape }

(* Helper to get Bigarray from device buffer (remains the same) ... *)
let get_ba_from_any_buffer (type a b) any_dev_buf
    (expected_dtype : (a, b) Ir.Dtype.t) expected_len name =
  let (Jit.Any_Device_Buffer buf) = any_dev_buf in
  if Ir.Dtype.to_string buf.dtype <> Ir.Dtype.to_string expected_dtype then
    Alcotest.failf "%s: Dtype mismatch: device buffer has %s, expected %s" name
      (Ir.Dtype.to_string buf.dtype)
      (Ir.Dtype.to_string expected_dtype);

  let ba_kind : (a, b) Bigarray.kind =
    match expected_dtype with
    | Dtype.Float32 -> Bigarray.float32
    | Dtype.Int32 -> Bigarray.int32
    | Dtype.Uint8 -> Bigarray.int8_unsigned
    | Dtype.Bool -> Bigarray.int8_unsigned (* Assuming bools are 1 byte *)
    | Dtype.Unit ->
        Alcotest.failf "%s: Cannot create Bigarray for Unit dtype" name
  in
  let host_ba = Bigarray.Array1.create ba_kind Bigarray.c_layout expected_len in
  match Jit.copy_from_device ~src_buffer:buf ~host_dest_bigarray:host_ba with
  | Ok () -> host_ba
  | Error e -> Alcotest.failf "%s: Failed to copy from device: %s" name e

(* ───────────────────── end-to-end tests ───────────────────── *)

let test_e2e_add_f32 () =
  fresh_var_reset ();
  let var_a = Var.fresh () in
  let var_b = Var.fresh () in
  let var_c = Var.fresh () in
  let shape = [| 4 |] in
  let meta_a = make_meta Float32 shape in
  let meta_b = make_meta Float32 shape in
  let meta_c = make_meta Float32 shape in

  let graph_nodes =
    [
      Ir.Any_Node (Placeholder { out_var = var_a; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = var_b; dtype = Float32; shape });
      Ir.Any_Node
        (Add
           {
             in_a_var = var_a;
             in_b_var = var_b;
             out_var = var_c;
             dtype = Float32;
           });
    ]
  in
  let vars_metadata = Hashtbl.create 3 in
  Hashtbl.add vars_metadata var_a meta_a;
  Hashtbl.add vars_metadata var_b meta_b;
  Hashtbl.add vars_metadata var_c meta_c;

  let graph =
    {
      Ir.nodes = graph_nodes;
      vars_metadata;
      input_vars = [ var_a; var_b ];
      output_vars = [ var_c ];
    }
  in

  let data_a_arr = [| 1.0; 2.0; 3.0; 4.0 |] in
  let data_b_arr = [| 0.1; 0.2; 0.3; 0.4 |] in
  let expected_c_arr = [| 1.1; 2.2; 3.3; 4.4 |] in

  let data_a_ba =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout data_a_arr
  in
  let data_b_ba =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout data_b_arr
  in
  let expected_c_ba =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout expected_c_arr
  in

  match Jit.compile graph with
  | Error e -> Alcotest.failf "Compilation failed: %s" e
  | Ok exec -> (
      let dev_buf_a =
        Result.get_ok
          (Jit.allocate_buffer
             ~size_in_bytes:(Bigarray.Array1.size_in_bytes data_a_ba)
             ~dtype:Float32)
      in
      let dev_buf_b =
        Result.get_ok
          (Jit.allocate_buffer
             ~size_in_bytes:(Bigarray.Array1.size_in_bytes data_b_ba)
             ~dtype:Float32)
      in
      Result.get_ok
        (Jit.copy_to_device ~dest_buffer:dev_buf_a ~host_data:data_a_ba);
      Result.get_ok
        (Jit.copy_to_device ~dest_buffer:dev_buf_b ~host_data:data_b_ba);

      let inputs_map = Hashtbl.create 2 in
      Hashtbl.add inputs_map var_a (Jit.Any_Device_Buffer dev_buf_a);
      Hashtbl.add inputs_map var_b (Jit.Any_Device_Buffer dev_buf_b);

      match Jit.execute exec ~inputs:inputs_map ~outputs_vars:[ var_c ] with
      | Error e -> Alcotest.failf "Execution failed: %s" e
      | Ok outputs_map ->
          let dev_buf_c_any = Hashtbl.find outputs_map var_c in
          let result_c_ba =
            get_ba_from_any_buffer dev_buf_c_any Float32
              (Array.length expected_c_arr)
              "C"
          in
          check_f32_ba_approx "Add F32 result" expected_c_ba result_c_ba)
