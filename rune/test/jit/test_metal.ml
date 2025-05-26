open Alcotest
open Rune_jit
open Support

(* ───── helpers ───── *)

let device_info = Rune_jit.Metal_backend.Device_info.get_default ()

let make_metal_source graph =
  let spec = List.hd (Rune_jit.Debug.schedule graph) in
  let lowered =
    Result.get_ok
      (Rune_jit.Debug.lower_kernel ~kernel_spec:spec
         ~original_graph_vars_metadata:graph.vars_metadata)
  in
  Rune_jit.Debug.render_metal ~device_info ~lowered_ir:lowered
    ~kernel_name:spec.name

(* pretty printer that truncates long strings *)
let pp_trunc ?(max = 400) fmt s =
  let len = String.length s in
  if len <= max then Format.pp_print_string fmt s
  else Format.fprintf fmt "%s...<%d more chars>" (String.sub s 0 max) (len - max)

let string_t = Alcotest.of_pp pp_trunc

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

(* ───── golden-source cases ───── *)

type golden_case = { name : string; graph : Ir.graph_t; golden : string }

let golden_cases =
  let _a, _b, _c, g_add = simple_add_graph () in
  let _cond, _x, _y, _out, g_where = simple_where_graph () in
  let _a, _b, _c, _out, g_mulacc = simple_mulacc_graph () in
  [
    { name = "add_float32"; graph = g_add; golden = "golden/add_f32.metal" };
    {
      name = "where_float32";
      graph = g_where;
      golden = "golden/where_f32.metal";
    };
    {
      name = "mulacc_float32";
      graph = g_mulacc;
      golden = "golden/mulacc_f32.metal";
    };
    (* add more entries here as you create new kernels *)
  ]

let test_golden { name; graph; golden } () =
  let got = make_metal_source graph in
  let exp = read_file golden in
  check string_t (name ^ " golden") exp got

(* ───── sanity test ───── *)

let count_occ sub s =
  let len_sub = String.length sub and len_s = String.length s in
  let rec loop i acc =
    if i > len_s - len_sub then acc
    else
      match String.index_from_opt s i sub.[0] with
      | None -> acc
      | Some j ->
          if j + len_sub <= len_s && String.equal (String.sub s j len_sub) sub
          then loop (j + 1) (acc + 1)
          else loop (j + 1) acc
  in
  loop 0 0

let test_sanity () =
  let _a, _b, _c, g = simple_add_graph () in
  let src = make_metal_source g in
  check int "three buffer params" 3 (count_occ "device float* v" src)

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
       ~backend:(module Rune_jit.Metal_backend)
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
    match Rune_jit.compile ~backend:(module Rune_jit.Metal_backend) graph with
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
         ~backend:(module Rune_jit.Metal_backend)
         ~size_in_bytes:(Bigarray.Array1.size_in_bytes host)
         ~dtype:Ir.Dtype.Float32)
  in
  let buf_a = alloc ba_a and buf_b = alloc ba_b in
  (* copy to device *)
  List.iter2
    (fun host buf ->
      Result.get_ok
        (Rune_jit.copy_to_device
           ~backend:(module Rune_jit.Metal_backend)
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
        ~backend:(module Rune_jit.Metal_backend)
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

(* ───── test suite ───── *)

let () =
  Alcotest.run "Metal backend"
    [
      ("sanity", [ test_case "param count" `Quick test_sanity ]);
      ( "golden",
        List.map (fun c -> test_case c.name `Quick (test_golden c)) golden_cases
      );
      ("end-to-end", [ test_case "add f32" `Quick test_e2e_add ]);
    ]
