open Alcotest
open Rune_jit (* brings Ir, Dtype, Metal_backend, … *)
module Metal_backend = Rune_jit.Metal_backend

(* ───────────── helpers ───────────── *)

let bigarray_float32 ?(epsilon = 1e-3) () =
  let open Bigarray in
  (* pretty-printer *)
  let pp fmt ba =
    Format.fprintf fmt "[|";
    for i = 0 to Array1.dim ba - 1 do
      if i > 0 then Format.fprintf fmt "; ";
      Format.fprintf fmt "%g" ba.{i}
    done;
    Format.fprintf fmt "|]"
  in
  (* equality with tolerance *)
  let equal a b =
    let len = Array1.dim a in
    len = Array1.dim b
    &&
    let rec loop i =
      i = len || (Float.abs (a.{i} -. b.{i}) <= epsilon && loop (i + 1))
    in
    loop 0
  in
  testable pp equal

let fresh_var_reset () = Var.counter := 0
let make_meta dtype shape = { Ir.dtype = Dtype.Any_Dtype dtype; shape }

(* Copy a device buffer into a freshly-allocated host big-array whose
 *kind witness* is given explicitly by the caller. *)
let get_ba_from_any_buffer (type a b) (Backend_intf.Any_Device_Buffer buf)
    ~(dtype : a Dtype.t) ~(kind : (a, b) Bigarray.kind) ~(len : int)
    ~(label : string) : (a, b, Bigarray.c_layout) Bigarray.Array1.t =
  let _ = dtype in
  let host = Bigarray.Array1.create kind Bigarray.c_layout len in
  (match
     Rune_jit.copy_from_device
       ~backend:(module Metal_backend)
       ~src_buffer:buf ~dest:host
   with
  | Ok () -> ()
  | Error e -> failf "%s: copy_from_device: %s" label e);
  host

(* ───────────── end-to-end test ───────────── *)

let test_e2e_add_f32 () =
  fresh_var_reset ();
  let a = Var.fresh () in
  let b = Var.fresh () in
  let c = Var.fresh () in
  let shape = [| 4 |] in
  let meta_f32 = make_meta Float32 shape in

  let nodes =
    [
      Ir.Any_Node (Placeholder { out_var = a; dtype = Float32; shape });
      Ir.Any_Node (Placeholder { out_var = b; dtype = Float32; shape });
      Ir.Any_Node
        (Ir.Binop
           { op = Ir.Add; a_var = a; b_var = b; out_var = c; dtype = Float32 });
    ]
  in
  let vars_meta = Hashtbl.create 3 in
  List.iter (fun v -> Hashtbl.add vars_meta v meta_f32) [ a; b; c ];

  let graph =
    {
      Ir.nodes;
      vars_metadata = vars_meta;
      input_vars = [ a; b ];
      output_vars = [ c ];
    }
  in

  let arr_a = [| 1.0; 2.0; 3.0; 4.0 |] in
  let arr_b = [| 0.1; 0.2; 0.3; 0.4 |] in
  let arr_c = [| 1.1; 2.2; 3.3; 4.4 |] in
  let ba_a =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_a
  in
  let ba_b =
    Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_b
  in

  match Rune_jit.compile ~backend:(module Metal_backend) graph with
  | Error e -> failf "compile: %s" e
  | Ok exe -> (
      let buf_a =
        Result.get_ok
          (Rune_jit.allocate_buffer
             ~backend:(module Metal_backend)
             ~size_in_bytes:(Bigarray.Array1.size_in_bytes ba_a)
             ~dtype:Float32)
      in
      let buf_b =
        Result.get_ok
          (Rune_jit.allocate_buffer
             ~backend:(module Metal_backend)
             ~size_in_bytes:(Bigarray.Array1.size_in_bytes ba_b)
             ~dtype:Float32)
      in
      Result.get_ok
        (Rune_jit.copy_to_device
           ~backend:(module Metal_backend)
           ~dest_buffer:buf_a ~host:ba_a);
      Result.get_ok
        (Rune_jit.copy_to_device
           ~backend:(module Metal_backend)
           ~dest_buffer:buf_b ~host:ba_b);

      let inputs = Hashtbl.create 2 in
      Hashtbl.add inputs a (Backend_intf.Any_Device_Buffer buf_a);
      Hashtbl.add inputs b (Backend_intf.Any_Device_Buffer buf_b);

      match
        Rune_jit.execute
          ~backend:(module Metal_backend)
          exe ~inputs ~outputs:[ c ]
      with
      | Error e -> failf "execute: %s" e
      | Ok outs ->
          let (Backend_intf.Any_Device_Buffer buf_c) = Hashtbl.find outs c in
          let ba_res =
            get_ba_from_any_buffer (Backend_intf.Any_Device_Buffer buf_c)
              ~dtype:Float32 ~kind:Bigarray.float32 ~len:(Array.length arr_c)
              ~label:"C"
          in
          let ba_c_expected =
            Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout arr_c
          in
          (check (bigarray_float32 ~epsilon:0.001 ()))
            "Add F32 result" ba_c_expected ba_res)

let () =
  Alcotest.run "Rune JIT"
    [
      ( "Metal backend",
        [
          Alcotest.test_case "Add Float32" `Quick test_e2e_add_f32;
          (* put more test_case entries here as you add new tests *)
        ] );
    ]
