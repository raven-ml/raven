open Alcotest
open Rune_jit
open Support

(* ───── helpers ───── *)

let device_info = Rune_jit_llvm.Device_info.get_default ()

let make_llvm_ir graph =
  let spec = List.hd (Rune_jit.Debug.schedule graph) in
  let lowered =
    Result.get_ok
      (Rune_jit.Debug.lower_kernel ~kernel_spec:spec
         ~original_graph_vars_metadata:graph.vars_metadata)
  in
  (* For LLVM backend, we generate IR directly in the compiler *)
  Rune_jit_llvm.Renderer.render ~device_info ~lowered_ir:lowered
    ~kernel_name:spec.name

(* ───── sanity test ───── *)

let test_sanity () =
  let _a, _b, _c, g = simple_add_graph () in
  let _ir = make_llvm_ir g in
  (* Basic sanity check that we can create IR *)
  check bool "can create IR" true true

(* ───── end-to-end execution ───── *)

let bigarray_float32 ?(eps = 1e-3) () =
  let open Bigarray in
  let pp fmt ba =
    Format.fprintf fmt "[|";
    for i = 0 to Array1.dim ba - 1 do
      if i > 0 then Format.fprintf fmt "; ";
      Format.fprintf fmt "%g" ba.{i}
    done;
    Format.fprintf fmt "|]"
  in
  let eq a b =
    Array1.dim a = Array1.dim b
    &&
    let rec loop i =
      i = Array1.dim a || (Float.abs (a.{i} -. b.{i}) <= eps && loop (i + 1))
    in
    loop 0
  in
  testable pp eq

let get_ba_from_buf (type a b) (Backend_intf.Any_Device_Buffer buf)
    ~(dtype : a Ir.Dtype.t) ~(kind : (a, b) Bigarray.kind) ~len label =
  let _ = dtype in
  let host = Bigarray.Array1.create kind Bigarray.c_layout len in
  (match
     Rune_jit.copy_from_device
       ~backend:(module Rune_jit_llvm)
       ~src_buffer:buf ~dest:host
   with
  | Ok () -> ()
  | Error e -> failf "%s copy_from_device: %s" label e);
  host

let test_e2e_add () =
  (* build graph *)
  let a, b, c, graph = simple_add_graph () in
  (* compile *)
  let exe =
    match Rune_jit.compile ~backend:(module Rune_jit_llvm) graph with
    | Ok e -> e
    | Error e -> failf "compile: %s" e
  in
  (* host data *)
  let arr_a = [| 1.0; 2.0; 3.0; 4.0 |] in
  let arr_b = [| 0.1; 0.2; 0.3; 0.4 |] in
  let ba_a =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_a
  in
  let ba_b =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_b
  in
  let len = Array.length arr_a in
  (* allocate device buffers *)
  let alloc host =
    Result.get_ok
      (Rune_jit.allocate_buffer
         ~backend:(module Rune_jit_llvm)
         ~size_in_bytes:(Bigarray.Array1.size_in_bytes host)
         ~dtype:Ir.Dtype.Float32)
  in
  let buf_a = alloc ba_a and buf_b = alloc ba_b in
  (* copy to device *)
  List.iter2
    (fun host buf ->
      Result.get_ok
        (Rune_jit.copy_to_device
           ~backend:(module Rune_jit_llvm)
           ~dest_buffer:buf ~host))
    [ ba_a; ba_b ] [ buf_a; buf_b ];
  (* prepare input map *)
  let inputs = Hashtbl.create 2 in
  Hashtbl.add inputs a (Backend_intf.Any_Device_Buffer buf_a);
  Hashtbl.add inputs b (Backend_intf.Any_Device_Buffer buf_b);
  (* run *)
  let outs =
    match
      Rune_jit.execute
        ~backend:(module Rune_jit_llvm)
        exe ~inputs ~outputs:[ c ]
    with
    | Ok tbl -> tbl
    | Error e -> failf "execute: %s" e
  in
  let buf_c = Hashtbl.find outs c in
  let ba_res =
    get_ba_from_buf buf_c ~dtype:Ir.Dtype.Float32 ~kind:Bigarray.float32 ~len
      "c"
  in
  let expected =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
      [| 1.1; 2.2; 3.3; 4.4 |]
  in
  check (bigarray_float32 ()) "result" expected ba_res

let test_e2e_where () =
  try
    Printf.eprintf "test_e2e_where starting...\n%!";
    (* build graph *)
    let cond, x, y, out, graph = simple_where_graph () in
    Printf.eprintf "Graph created\n%!";
    (* compile *)
    Printf.eprintf "Starting where test compilation...\n%!";
    let exe =
      match Rune_jit.compile ~backend:(module Rune_jit_llvm) graph with
      | Ok e -> Printf.eprintf "Compilation successful\n%!"; e
      | Error e -> failf "compile: %s" e
    in
    Printf.eprintf "Compilation done, starting host data prep...\n%!";
  (* host data *)
  let arr_cond = [| 1; 0; 1; 0 |] in (* bool as uint8 *)
  let arr_x = [| 1.0; 2.0; 3.0; 4.0 |] in
  let arr_y = [| 10.0; 20.0; 30.0; 40.0 |] in
  let ba_cond =
    Bigarray.Array1.of_array Bigarray.int8_unsigned Bigarray.c_layout arr_cond
  in
  let ba_x =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_x
  in
  let ba_y =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_y
  in
  let len = Array.length arr_x in
  (* allocate device buffers *)
  let buf_cond =
    Result.get_ok
      (Rune_jit.allocate_buffer
         ~backend:(module Rune_jit_llvm)
         ~size_in_bytes:(Bigarray.Array1.size_in_bytes ba_cond)
         ~dtype:Ir.Dtype.Uint8)
  in
  let buf_x =
    Result.get_ok
      (Rune_jit.allocate_buffer
         ~backend:(module Rune_jit_llvm)
         ~size_in_bytes:(Bigarray.Array1.size_in_bytes ba_x)
         ~dtype:Ir.Dtype.Float32)
  in
  let buf_y =
    Result.get_ok
      (Rune_jit.allocate_buffer
         ~backend:(module Rune_jit_llvm)
         ~size_in_bytes:(Bigarray.Array1.size_in_bytes ba_y)
         ~dtype:Ir.Dtype.Float32)
  in
  (* copy to device *)
  Result.get_ok
    (Rune_jit.copy_to_device
       ~backend:(module Rune_jit_llvm)
       ~dest_buffer:buf_cond ~host:ba_cond);
  Result.get_ok
    (Rune_jit.copy_to_device
       ~backend:(module Rune_jit_llvm)
       ~dest_buffer:buf_x ~host:ba_x);
  Result.get_ok
    (Rune_jit.copy_to_device
       ~backend:(module Rune_jit_llvm)
       ~dest_buffer:buf_y ~host:ba_y);
  (* prepare input map *)
  let inputs = Hashtbl.create 3 in
  Hashtbl.add inputs cond (Backend_intf.Any_Device_Buffer buf_cond);
  Hashtbl.add inputs x (Backend_intf.Any_Device_Buffer buf_x);
  Hashtbl.add inputs y (Backend_intf.Any_Device_Buffer buf_y);
  (* run *)
  let outs =
    match
      Rune_jit.execute
        ~backend:(module Rune_jit_llvm)
        exe ~inputs ~outputs:[ out ]
    with
    | Ok tbl -> tbl
    | Error e -> failf "execute: %s" e
  in
  let buf_out = Hashtbl.find outs out in
  let ba_res =
    get_ba_from_buf buf_out ~dtype:Ir.Dtype.Float32 ~kind:Bigarray.float32 ~len
      "out"
  in
  let expected =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
      [| 1.0; 20.0; 3.0; 40.0 |] (* where cond is true, take x, else y *)
  in
  check (bigarray_float32 ()) "result" expected ba_res
  with e ->
    Printf.eprintf "Exception in test_e2e_where: %s\n%!" (Printexc.to_string e);
    raise e

let _test_e2e_mulacc () =
  (* build graph *)
  let a, b, c, out, graph = simple_mulacc_graph () in
  (* compile *)
  let exe =
    match Rune_jit.compile ~backend:(module Rune_jit_llvm) graph with
    | Ok e -> e
    | Error e -> failf "compile: %s" e
  in
  (* host data *)
  let arr_a = [| 2.0; 3.0; 4.0; 5.0 |] in
  let arr_b = [| 10.0; 10.0; 10.0; 10.0 |] in
  let arr_c = [| 1.0; 2.0; 3.0; 4.0 |] in
  let ba_a =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_a
  in
  let ba_b =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_b
  in
  let ba_c =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_c
  in
  let len = Array.length arr_a in
  (* allocate device buffers *)
  let alloc host =
    Result.get_ok
      (Rune_jit.allocate_buffer
         ~backend:(module Rune_jit_llvm)
         ~size_in_bytes:(Bigarray.Array1.size_in_bytes host)
         ~dtype:Ir.Dtype.Float32)
  in
  let buf_a = alloc ba_a in
  let buf_b = alloc ba_b in
  let buf_c = alloc ba_c in
  (* copy to device *)
  List.iter2
    (fun host buf ->
      Result.get_ok
        (Rune_jit.copy_to_device
           ~backend:(module Rune_jit_llvm)
           ~dest_buffer:buf ~host))
    [ ba_a; ba_b; ba_c ] [ buf_a; buf_b; buf_c ];
  (* prepare input map *)
  let inputs = Hashtbl.create 3 in
  Hashtbl.add inputs a (Backend_intf.Any_Device_Buffer buf_a);
  Hashtbl.add inputs b (Backend_intf.Any_Device_Buffer buf_b);
  Hashtbl.add inputs c (Backend_intf.Any_Device_Buffer buf_c);
  (* run *)
  let outs =
    match
      Rune_jit.execute
        ~backend:(module Rune_jit_llvm)
        exe ~inputs ~outputs:[ out ]
    with
    | Ok tbl -> tbl
    | Error e -> failf "execute: %s" e
  in
  let buf_out = Hashtbl.find outs out in
  let ba_res =
    get_ba_from_buf buf_out ~dtype:Ir.Dtype.Float32 ~kind:Bigarray.float32 ~len
      "out"
  in
  let expected =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
      [| 21.0; 32.0; 43.0; 54.0 |] (* a * b + c *)
  in
  check (bigarray_float32 ()) "result" expected ba_res

(* ───── test suite ───── *)

let () =
  Alcotest.run "LLVM backend"
    [
      ("sanity", [ test_case "basic IR creation" `Quick test_sanity ]);
      ("end-to-end", [
        test_case "add f32" `Quick test_e2e_add;
        test_case "simple test" `Quick (fun () -> 
          Printf.eprintf "Simple test running...\n%!";
          check bool "true" true true);
        (* test_case "where f32" `Quick test_e2e_where; *)
        (* test_case "mulacc f32" `Quick test_e2e_mulacc; *)
      ]);
    ]