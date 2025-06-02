open Alcotest
open Nx_core

(* Helper to compare tensors *)
let tensor_equal_float32 ?(eps = 1e-6) msg t1 t2 =
  let shape1 =
    Nx_metal.view t1 |> Lazy_view.shape |> fun s ->
    match Symbolic_shape.eval s with
    | Some arr -> arr
    | None -> failwith "symbolic shape"
  in
  let shape2 =
    Nx_metal.view t2 |> Lazy_view.shape |> fun s ->
    match Symbolic_shape.eval s with
    | Some arr -> arr
    | None -> failwith "symbolic shape"
  in
  check (array int) (msg ^ " shape") shape1 shape2;

  let data1 = Nx_metal.data t1 in
  let data2 = Nx_metal.data t2 in
  let n = Bigarray.Array1.dim data1 in

  for i = 0 to n - 1 do
    let v1 = Bigarray.Array1.get data1 i in
    let v2 = Bigarray.Array1.get data2 i in
    let diff = abs_float (v1 -. v2) in
    if diff > eps then
      failf "%s: values differ at index %d: %f vs %f (diff: %f)" msg i v1 v2
        diff
  done

(* Test context creation *)
let test_create_context () =
  let _ctx = Nx_metal.create_context () in
  (* If we get here without exception, context was created successfully *)
  ()

(* Test buffer allocation *)
let test_buffer_allocation () =
  let ctx = Nx_metal.create_context () in

  (* Test various sizes *)
  let sizes = [ 0; 1; 100; 1000; 10000 ] in
  List.iter
    (fun size ->
      let t = Nx_metal.op_buffer ctx Dtype.Float32 size in
      let actual_size =
        Nx_metal.view t |> Lazy_view.numel |> function
        | Symbolic_shape.Static n -> n
        | Symbolic_shape.Dynamic _ -> failwith "symbolic size"
      in
      check int (Printf.sprintf "buffer size %d" size) size actual_size)
    sizes

(* Test scalar creation *)
let test_const_scalar () =
  let ctx = Nx_metal.create_context () in

  (* Float scalar *)
  let t1 = Nx_metal.op_const_scalar ctx 42.0 Dtype.Float32 in
  let shape1 =
    Nx_metal.view t1 |> Lazy_view.shape |> fun s ->
    match Symbolic_shape.eval s with
    | Some arr -> arr
    | None -> failwith "symbolic shape"
  in
  check (array int) "scalar shape" [||] shape1;
  let data1 = Nx_metal.data t1 in
  check (float 0.001) "scalar value" 42.0 (Bigarray.Array1.get data1 0);

  (* Int scalar *)
  let t2 = Nx_metal.op_const_scalar ctx 42l Dtype.Int32 in
  let data2 = Nx_metal.data t2 in
  check int32 "int scalar value" 42l (Bigarray.Array1.get data2 0)

(* Test array creation *)
let test_const_array () =
  let ctx = Nx_metal.create_context () in

  (* Create bigarray *)
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 5 in
  for i = 0 to 4 do
    Bigarray.Array1.set ba i (float_of_int i)
  done;

  let t = Nx_metal.op_const_array ctx ba in
  let shape =
    Nx_metal.view t |> Lazy_view.shape |> fun s ->
    match Symbolic_shape.eval s with
    | Some arr -> arr
    | None -> failwith "symbolic shape"
  in
  check (array int) "array shape" [| 5 |] shape;

  let data = Nx_metal.data t in
  for i = 0 to 4 do
    check (float 0.001)
      (Printf.sprintf "array[%d]" i)
      (float_of_int i)
      (Bigarray.Array1.get data i)
  done

(* Test binary operations *)
let test_binary_ops () =
  let ctx = Nx_metal.create_context () in

  (* Create test arrays *)
  let ba1 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 5 in
  let ba2 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 5 in
  for i = 0 to 4 do
    Bigarray.Array1.set ba1 i (float_of_int (i + 1));
    Bigarray.Array1.set ba2 i (float_of_int (i + 2))
  done;

  let t1 = Nx_metal.op_const_array ctx ba1 in
  let t2 = Nx_metal.op_const_array ctx ba2 in

  (* Test add *)
  let sum = Nx_metal.op_add t1 t2 in
  let sum_data = Nx_metal.data sum in
  for i = 0 to 4 do
    check (float 0.001)
      (Printf.sprintf "add[%d]" i)
      (float_of_int ((2 * i) + 3))
      (Bigarray.Array1.get sum_data i)
  done;

  (* Test mul *)
  let prod = Nx_metal.op_mul t1 t2 in
  let prod_data = Nx_metal.data prod in
  for i = 0 to 4 do
    let expected = float_of_int ((i + 1) * (i + 2)) in
    check (float 0.001)
      (Printf.sprintf "mul[%d]" i)
      expected
      (Bigarray.Array1.get prod_data i)
  done

(* Test unary operations *)
let test_unary_ops () =
  let ctx = Nx_metal.create_context () in

  (* Create test array *)
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 5 in
  for i = 0 to 4 do
    Bigarray.Array1.set ba i (float_of_int (i + 1))
  done;

  let t = Nx_metal.op_const_array ctx ba in

  (* Test neg *)
  let neg = Nx_metal.op_neg t in
  let neg_data = Nx_metal.data neg in
  for i = 0 to 4 do
    check (float 0.001)
      (Printf.sprintf "neg[%d]" i)
      (-.float_of_int (i + 1))
      (Bigarray.Array1.get neg_data i)
  done;

  (* Test sqrt *)
  let sqrt_t = Nx_metal.op_sqrt t in
  let sqrt_data = Nx_metal.data sqrt_t in
  for i = 0 to 4 do
    check (float 0.001)
      (Printf.sprintf "sqrt[%d]" i)
      (sqrt (float_of_int (i + 1)))
      (Bigarray.Array1.get sqrt_data i)
  done

(* Test reduction operations *)
let test_reduce_ops () =
  let ctx = Nx_metal.create_context () in

  (* Create 2D test array *)
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 6 in
  for i = 0 to 5 do
    Bigarray.Array1.set ba i (float_of_int (i + 1))
  done;

  let t = Nx_metal.op_const_array ctx ba in
  let t = Nx_metal.op_reshape t (Symbolic_shape.of_ints [| 2; 3 |]) in

  (* Test sum along axis 0 *)
  let sum0 = Nx_metal.op_reduce_sum ~axes:[| 0 |] ~keepdims:false t in
  let sum0_data = Nx_metal.data sum0 in
  check (array int) "sum axis 0 shape" [| 3 |]
    ( Nx_metal.view sum0 |> Lazy_view.shape |> fun s ->
      match Symbolic_shape.eval s with
      | Some arr -> arr
      | None -> failwith "symbolic shape" );
  check (float 0.001) "sum0[0]" 5.0 (Bigarray.Array1.get sum0_data 0);
  check (float 0.001) "sum0[1]" 7.0 (Bigarray.Array1.get sum0_data 1);
  check (float 0.001) "sum0[2]" 9.0 (Bigarray.Array1.get sum0_data 2);

  (* Test sum along axis 1 *)
  let sum1 = Nx_metal.op_reduce_sum ~axes:[| 1 |] ~keepdims:false t in
  let sum1_data = Nx_metal.data sum1 in
  check (array int) "sum axis 1 shape" [| 2 |]
    ( Nx_metal.view sum1 |> Lazy_view.shape |> fun s ->
      match Symbolic_shape.eval s with
      | Some arr -> arr
      | None -> failwith "symbolic shape" );
  check (float 0.001) "sum1[0]" 6.0 (Bigarray.Array1.get sum1_data 0);
  check (float 0.001) "sum1[1]" 15.0 (Bigarray.Array1.get sum1_data 1)

(* Test reshape and view operations *)
let test_view_ops () =
  let ctx = Nx_metal.create_context () in

  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 12 in
  for i = 0 to 11 do
    Bigarray.Array1.set ba i (float_of_int i)
  done;

  let t = Nx_metal.op_const_array ctx ba in

  (* Test reshape *)
  let t2x6 = Nx_metal.op_reshape t (Symbolic_shape.of_ints [| 2; 6 |]) in
  check (array int) "reshape 2x6" [| 2; 6 |]
    ( Nx_metal.view t2x6 |> Lazy_view.shape |> fun s ->
      match Symbolic_shape.eval s with
      | Some arr -> arr
      | None -> failwith "symbolic shape" );

  let t3x4 = Nx_metal.op_reshape t (Symbolic_shape.of_ints [| 3; 4 |]) in
  check (array int) "reshape 3x4" [| 3; 4 |]
    ( Nx_metal.view t3x4 |> Lazy_view.shape |> fun s ->
      match Symbolic_shape.eval s with
      | Some arr -> arr
      | None -> failwith "symbolic shape" );

  (* Test permute *)
  let t4x3 = Nx_metal.op_reshape t (Symbolic_shape.of_ints [| 4; 3 |]) in
  let tp = Nx_metal.op_permute t4x3 [| 1; 0 |] in
  check (array int) "permute shape" [| 3; 4 |]
    ( Nx_metal.view tp |> Lazy_view.shape |> fun s ->
      match Symbolic_shape.eval s with
      | Some arr -> arr
      | None -> failwith "symbolic shape" )

(* Test copy and contiguous *)
let test_copy_contiguous () =
  let ctx = Nx_metal.create_context () in

  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 6 in
  for i = 0 to 5 do
    Bigarray.Array1.set ba i (float_of_int i)
  done;

  let t = Nx_metal.op_const_array ctx ba in

  (* Test copy *)
  let copy = Nx_metal.op_copy t in
  tensor_equal_float32 "copy" t copy;

  (* Test contiguous on already contiguous *)
  let cont = Nx_metal.op_contiguous t in
  tensor_equal_float32 "contiguous" t cont

(* Test comparison operations *)
let test_comparison_ops () =
  let ctx = Nx_metal.create_context () in

  let ba1 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 5 in
  let ba2 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 5 in
  for i = 0 to 4 do
    Bigarray.Array1.set ba1 i (float_of_int i);
    Bigarray.Array1.set ba2 i (float_of_int (i - 2))
  done;

  let t1 = Nx_metal.op_const_array ctx ba1 in
  let t2 = Nx_metal.op_const_array ctx ba2 in

  (* Test less than *)
  let lt = Nx_metal.op_cmplt t1 t2 in
  let lt_data = Nx_metal.data lt in
  check int "lt[0]" 0 (int_of_char (Char.chr (Bigarray.Array1.get lt_data 0)));
  (* 0 < -2 is false *)
  check int "lt[1]" 0 (int_of_char (Char.chr (Bigarray.Array1.get lt_data 1)));
  (* 1 < -1 is false *)
  check int "lt[2]" 0 (int_of_char (Char.chr (Bigarray.Array1.get lt_data 2)));
  (* 2 < 0 is false *)
  check int "lt[3]" 0 (int_of_char (Char.chr (Bigarray.Array1.get lt_data 3)));
  (* 3 < 1 is false *)
  check int "lt[4]" 0 (int_of_char (Char.chr (Bigarray.Array1.get lt_data 4)))
(* 4 < 2 is false *)

(* Test suite *)
let () =
  run "Nx_metal"
    [
      ("context", [ test_case "create" `Quick test_create_context ]);
      ( "buffer",
        [
          test_case "allocation" `Quick test_buffer_allocation;
          test_case "const_scalar" `Quick test_const_scalar;
          test_case "const_array" `Quick test_const_array;
        ] );
      ("binary_ops", [ test_case "basic" `Quick test_binary_ops ]);
      ("unary_ops", [ test_case "basic" `Quick test_unary_ops ]);
      ("reduce_ops", [ test_case "basic" `Quick test_reduce_ops ]);
      ("view_ops", [ test_case "reshape_permute" `Quick test_view_ops ]);
      ("memory", [ test_case "copy_contiguous" `Quick test_copy_contiguous ]);
      ("comparison", [ test_case "basic" `Quick test_comparison_ops ]);
    ]
