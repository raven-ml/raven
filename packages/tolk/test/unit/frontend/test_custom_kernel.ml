(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Kernels written in uops and run from the tensor graph. The kernels and the
   cases follow tinygrad's test/backend/test_custom_kernel.py. *)

open Windtrap
module Bound = Tolk_uop.Bound
module T = Tolk_frontend.Tensor
module Mv = Tolk_frontend.Movement
module El = Tolk_frontend.Elementwise
module Rd = Tolk_frontend.Reduce
module Op = Tolk_frontend.Op
module Creation = Tolk_frontend.Creation
module Run = Tolk_frontend.Run
module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype
module Ops = Tolk_uop.Ops
module Axis_type = Tolk_uop.Axis_type

let kernel_info ?opts_to_apply name =
  {
    U.name;
    applied_opts = [];
    opts_to_apply;
    estimates = None;
    beam = 0;
  }

let dims u = List.map (fun dim -> Bound.to_int (U.vmax dim)) (U.shape u)
let numel u = List.fold_left ( * ) 1 (dims u)
let flatten u = U.reshape ~src:u ~shape:(U.const_int (numel u))

let range ?(kind = Axis_type.Weak) size axis =
  U.range ~size:(U.const_int size) ~axis ~kind ()

let at ptr idxs = U.index ~ptr ~idxs ()
let add a b = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b
let mul a b = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b
let store dst value = U.store ~dst ~value ()

(* [ptr] once [value] is stored at [dst] and the [ends] loops are closed. *)
let set ?(ends = []) ptr dst value =
  U.after ~src:ptr ~deps:[ U.end_ ~value:(store dst value) ~ranges:ends ]

let const_like u x = U.const (Tolk_uop.Const.float (U.dtype u) x)

(* Kernels *)

let custom_arange_kernel = function
  | [ c ] ->
      let n = List.hd (dims c) in
      let i = range n 0 in
      U.sink
        ~kernel_info:(kernel_info (Printf.sprintf "custom_arange_%d" n))
        [
          U.end_
            ~value:(store (at c [ i ]) (U.cast ~src:i ~dtype:(U.dtype c)))
            ~ranges:[ i ];
        ]
  | _ -> assert false

let custom_eye_kernel = function
  | [ c ] ->
      let i = range (List.nth (dims c) 0) 0 in
      let j = range (List.nth (dims c) 1) 1 in
      let eq =
        U.alu_binary ~op:Ops.Cmpne
          ~lhs:(U.alu_binary ~op:Ops.Cmpne ~lhs:i ~rhs:j)
          ~rhs:(U.const_bool true)
      in
      U.sink
        ~kernel_info:(kernel_info (Printf.sprintf "custom_eye_%d" (numel c)))
        [
          U.end_
            ~value:(store (at c [ i; j ]) (U.cast ~src:eq ~dtype:(U.dtype c)))
            ~ranges:[ i; j ];
        ]
  | _ -> assert false

let custom_add_one_kernel = function
  | [ b; a ] ->
      let a = flatten a and b = flatten b in
      let i = range (numel a) 0 in
      U.sink
        ~kernel_info:(kernel_info (Printf.sprintf "add_one_%d" (numel a)))
        [
          U.end_
            ~value:(store (at b [ i ]) (add (at a [ i ]) (const_like a 1.0)))
            ~ranges:[ i ];
        ]
  | _ -> assert false

let custom_elementwise_add_kernel = function
  | [ c; a; b ] ->
      let c = flatten c and a = flatten a and b = flatten b in
      let i = range (numel c) 0 in
      U.sink
        ~kernel_info:
          (kernel_info (Printf.sprintf "custom_add_kernel_%d" (numel c)))
        [
          U.end_
            ~value:(store (at c [ i ]) (add (at a [ i ]) (at b [ i ])))
            ~ranges:[ i ];
        ]
  | _ -> assert false

let custom_elementwise_addmul_kernel = function
  | [ c; d; a; b ] ->
      let c = flatten c and d = flatten d in
      let a = flatten a and b = flatten b in
      let i = range (numel c) 0 in
      let store_c = store (at c [ i ]) (add (at a [ i ]) (at b [ i ])) in
      let store_d = store (at d [ i ]) (mul (at a [ i ]) (at b [ i ])) in
      U.sink
        ~kernel_info:
          (kernel_info (Printf.sprintf "custom_addmul_kernel_%d" (numel c)))
        [ U.end_ ~value:(U.group [ store_c; store_d ]) ~ranges:[ i ] ]
  | _ -> assert false

let custom_gemm = function
  | [ c; a; b ] ->
      let rows = List.nth (dims c) 0 and cols = List.nth (dims c) 1 in
      let inner = List.nth (dims a) 1 in
      let i = range rows 0 and j = range cols 1 in
      let k = range ~kind:Axis_type.Reduce inner 2 in
      let c = set c (at c [ i; j ]) (const_like c 0.0) in
      let acc =
        add
          (at (U.after ~src:c ~deps:[ k ]) [ i; j ])
          (mul (at a [ i; k ]) (at b [ k; j ]))
      in
      let writes = U.end_ ~value:(store (at c [ i; j ]) acc) ~ranges:[ k ] in
      U.sink
        ~kernel_info:
          (kernel_info ~opts_to_apply:[]
             (Printf.sprintf "custom_gemm_%d_%d_%d" rows cols inner))
        [ U.end_ ~value:writes ~ranges:[ i; j ] ]
  | _ -> assert false

let custom_sum = function
  | [ b; a ] ->
      let n = List.hd (dims a) in
      let i = range ~kind:Axis_type.Reduce n 0 in
      let zero = U.const_int 0 in
      let b = set b (at b [ zero ]) (U.zero_like (at b [ zero ])) in
      let acc = add (at (U.after ~src:b ~deps:[ i ]) [ zero ]) (at a [ i ]) in
      let b = set ~ends:[ i ] b (at b [ zero ]) acc in
      U.sink
        ~kernel_info:
          (kernel_info ~opts_to_apply:[] (Printf.sprintf "custom_sum_%d" n))
        [ b ]
  | _ -> assert false

(* Cases *)

let all_floats x t =
  Array.iter (fun v -> equal (float 1e-6) x v) (Run.to_float_array t)

let first = List.hd

let tests =
  group "custom kernel"
    [
      test "empty" (fun () ->
          let a = Creation.empty [ 1 ] in
          let fxn _ = U.sink ~kernel_info:(kernel_info "empty") [] in
          ignore (Run.realize (first (T.custom_kernel ~fxn [ a ]))));
      test "simple" (fun () ->
          let a = Creation.ones [ 16; 16 ] and b = Creation.ones [ 16; 16 ] in
          let c = Creation.empty [ 16; 16 ] in
          let fxn = custom_elementwise_add_kernel in
          all_floats 2.0 (first (T.custom_kernel ~fxn [ c; a; b ])));
      test "multioutput" (fun () ->
          let a = Creation.full [ 16; 16 ] (T.Sfloat 3.0) in
          let b = Creation.full [ 16; 16 ] (T.Sfloat 3.0) in
          let c = Creation.empty [ 16; 16 ] and d = Creation.empty [ 16; 16 ] in
          match
            T.custom_kernel ~fxn:custom_elementwise_addmul_kernel [ c; d; a; b ]
          with
          | c :: d :: _ ->
              Run.realize_many [ c; d ];
              all_floats 6.0 c;
              all_floats 9.0 d
          | _ -> assert false);
      test "arange" (fun () ->
          let tst = Creation.empty ~dtype:D.int32 [ 100 ] in
          equal (array int) (Array.init 100 Fun.id)
            (Run.to_int_array
               (first (T.custom_kernel ~fxn:custom_arange_kernel [ tst ]))));
      test "eye" (fun () ->
          let n = 64 in
          let tst = Creation.empty [ n; n ] in
          let got =
            Run.to_float_array
              (first (T.custom_kernel ~fxn:custom_eye_kernel [ tst ]))
          in
          Array.iteri
            (fun p v ->
              equal (float 1e-6) (if p / n = p mod n then 1.0 else 0.0) v)
            got);
      test "a source that is not contiguous is realized first" (fun () ->
          let a = Creation.ones [ 16; 16 ] in
          let tst = Creation.empty [ 16; 16 ] in
          let b = El.add a (T.f 1.0) in
          all_floats 3.0
            (first (T.custom_kernel ~fxn:custom_add_one_kernel [ tst; b ])));
      test "sum" (fun () ->
          let a = Run.of_float_array ~shape:[ 5 ] [| 1.; 2.; 3.; 4.; 5. |] in
          let tst = Creation.empty [ 1 ] in
          equal (float 1e-6) 15.0
            (Run.item_float
               (first (T.custom_kernel ~fxn:custom_sum [ tst; a ]))));
      test "sum of a computed source" (fun () ->
          let a =
            El.add
              (Run.of_float_array ~shape:[ 5 ] [| 1.; 2.; 3.; 4.; 5. |])
              (T.f 1.0)
          in
          let tst = Creation.empty [ 1 ] in
          equal (float 1e-6) 20.0
            (Run.item_float
               (first (T.custom_kernel ~fxn:custom_sum [ tst; a ]))));
      test "sum of ints" (fun () ->
          let a = Run.of_int_array ~shape:[ 5 ] [| 1; 2; 3; 4; 5 |] in
          let tst = Creation.empty ~dtype:D.int32 [ 1 ] in
          equal int 15
            (Run.item_int
               (first (T.custom_kernel ~fxn:custom_sum [ tst; a ]))));
      test "gemm accumulates under a serial inner range" (fun () ->
          let n = 16 in
          let data seed =
            Array.init (n * n) (fun p ->
                float_of_int (((p * seed) + 3) mod 11) -. 5.0)
          in
          let a = Run.of_float_array ~shape:[ n; n ] (data 7) in
          let b = Run.of_float_array ~shape:[ n; n ] (data 5) in
          let c = Creation.empty [ n; n ] in
          let got =
            Run.to_float_array
              (first (T.custom_kernel ~fxn:custom_gemm [ c; a; b ]))
          in
          let want = Run.to_float_array (Op.matmul a b) in
          Array.iteri (fun p w -> equal (float 1e-3) w got.(p)) want);
    ]

let call_info precompile : U.call_info =
  {grad_fxn = None; name = Some (U.Label "outputs"); precompile;
   precompile_backward = false; dtype = D.void; aux = None}

let explicit_call_outputs precompile () =
  List.iter (fun values ->
      let input = Run.of_float_array ~shape:[4] values in
      let formal = U.param_like (T.uop input) ~slot:1 in
      let outputs = U.call_with_outputs ~output_pos:[0; 2]
          ~values:[add formal (const_like formal 1.0); mul formal (const_like formal 2.0)]
          ~args:[T.uop input] ~info:(call_info precompile) () |> List.map T.of_uop in
      Run.realize_many outputs;
      equal (array (float 1e-6)) (Array.map (fun x -> x +. 1.0) values)
        (Run.to_float_array (List.nth outputs 0));
      equal (array (float 1e-6)) (Array.map (fun x -> x *. 2.0) values)
        (Run.to_float_array (List.nth outputs 1)))
    [[|1.; 2.; 3.; 4.|]; [|5.; 6.; 7.; 8.|]]

let nested_call_internal_allocation () =
  let input = Run.of_float_array ~shape:[4] [|1.; 2.; 3.; 4.|] in
  let formal = U.param_like (T.uop input) ~slot:0 in
  let temporary = U.alloc ~slot:(U.fresh_buffer_slot ()) ~dtype:D.float32
      ~shape:(U.const_int 4) () in
  let stored = U.after ~src:temporary ~deps:[store temporary (mul formal formal)] in
  let inner = U.call_with_outputs ~values:[add stored (const_like formal 1.0)]
      ~args:[formal] ~info:(call_info false) () in
  let output = List.hd (U.call_with_outputs ~values:inner ~args:[T.uop input]
      ~info:(call_info true) ()) |> T.of_uop in
  equal (array (float 1e-6)) [|2.; 5.; 10.; 17.|] (Run.to_float_array output)

let symbolic_call_outputs precompile () =
  List.iter (fun size ->
      let input = Run.of_float_array ~shape:[4] [|1.; 2.; 3.; 4.|] in
      let formal = U.param_like (T.uop input) ~slot:0 in
      let variable = U.variable ~name:"output_size" ~min_val:1 ~max_val:4 () in
      let bound = U.bind ~var:variable ~value:(U.const_int size) in
      let extent = U.param_like bound ~slot:1 in
      let value = U.shrink ~src:formal ~offset:(U.const_int 0) ~size:extent in
      let outputs = U.call_with_outputs ~values:[add value (const_like value 1.0)]
          ~args:[T.uop input; bound] ~info:(call_info precompile) () in
      let output = T.of_uop (List.hd outputs) in
      ignore (Run.realize output);
      let view = U.shrink ~src:(T.uop output) ~offset:(U.const_int 0)
          ~size:(U.const_int size) |> T.of_uop in
      equal (array (float 1e-6)) (Array.init size (fun i -> float_of_int (i + 2)))
        (Run.to_float_array view)) [2; 4]

let moved_input precompile movement () =
  let input = Run.of_float_array ~shape:[32] (Array.init 32 float_of_int) in
  let expected = Run.to_float_array (El.add (movement input) (T.f 1.0)) in
  let argument =
    if precompile then T.of_uop (U.param_like (T.uop input) ~slot:0)
    else input
  in
  let view = movement argument in
  let output = Creation.empty ~dtype:(T.dtype input) (T.shape view) in
  let result = first (T.custom_kernel ~fxn:custom_add_one_kernel [output; view]) in
  let result =
    if precompile then
      first (U.call_with_outputs ~values:[T.uop result] ~args:[T.uop input]
        ~info:(call_info true) ()) |> T.of_uop
    else result
  in
  equal (array (float 1e-6)) expected (Run.to_float_array result)

let input_tests =
  let movements = [
    "reshape", (fun t -> Mv.reshape t [16; 2]);
    "permute", (fun t -> Mv.permute (Mv.reshape t [4; 8]) [1; 0]);
    "double permute", (fun t ->
      Mv.permute (Mv.permute (Mv.reshape t [4; 8]) [1; 0]) [1; 0]);
    "prefix slice", (fun t -> Mv.shrink t [0, 4]);
    "padded slice", (fun t -> Mv.pad (Mv.shrink t [0, 4]) [0, 4]);
    "flip", (fun t -> Mv.flip t [0]);
    "offset slice", (fun t -> Mv.shrink t [4, 8]);
    "matrix slice", (fun t -> Mv.shrink (Mv.reshape t [4; 8]) [0, 4; 2, 6]);
    "expanded slice", (fun t ->
      Mv.expand (Mv.shrink (Mv.reshape t [16; 2]) [0, 16; 0, 1]) [16; 2]);
  ] in
  group "moved custom inputs" (List.concat_map (fun (name, movement) ->
    [test (name ^ " buffer") (moved_input false movement);
     test (name ^ " parameter") (moved_input true movement)]) movements)

let storage_tests = group "custom storage" [
  test "duplicate arguments update shared storage" (fun () ->
    let x = Run.of_float_array ~shape:[4] [|0.; 1.; 2.; 3.|] in
    let output = first (T.custom_kernel ~fxn:custom_add_one_kernel [x; x]) in
    equal (array (float 1e-6)) [|1.; 2.; 3.; 4.|] (Run.to_float_array output));
  test "unused arguments preserve parameter positions" (fun () ->
    let unused = Run.of_float_array ~shape:[4] [|100.; 200.; 300.; 400.|] in
    let input = Run.of_float_array ~shape:[4] [|1.; 2.; 3.; 4.|] in
    let output = Creation.empty [4] in
    let fxn = function
      | [dst; unused; src] ->
          ignore unused;
          custom_add_one_kernel [dst; src]
      | _ -> assert false
    in
    equal (array (float 1e-6)) [|2.; 3.; 4.; 5.|]
      (Run.to_float_array (first (T.custom_kernel ~fxn [output; unused; input]))));
  test "sharded custom inputs use per-device shapes" (fun () ->
    let devices = ["CPU:0"; "CPU:1"] in
    let input = Run.of_float_array ~shape:[4; 4]
        (Array.init 16 float_of_int) |> Creation.clone ~device:(U.Single "CPU")
        |> Creation.shard ~devices ~axis:0 in
    let output = Creation.empty ~device:(U.Multi devices) [2; 4] in
    let output = T.of_uop (U.unshard ~src:(T.uop output) ~axes:[0] ()) in
    let output = first (T.custom_kernel ~fxn:custom_add_one_kernel [output; input])
        |> Creation.clone ~device:(U.Single "CPU") in
    equal (array (float 1e-6)) (Array.init 16 (fun i -> float_of_int (i + 1)))
      (Run.to_float_array output));
  test "assignment reads a reshaped custom output" (fun () ->
    let input = Run.of_float_array ~shape:[4] [|0.; 1.; 2.; 3.|] in
    let output = Creation.empty [4] in
    let result = first (T.custom_kernel ~fxn:custom_add_one_kernel [output; input]) in
    let destination = Run.of_float_array ~shape:[2; 2] [|9.; 9.; 9.; 9.|] in
    ignore (Op.assign destination (Mv.reshape result [2; 2]));
    equal (array (float 1e-6)) [|1.; 2.; 3.; 4.|] (Run.to_float_array destination));
  test "a slice of allocated custom output executes its pending call" (fun () ->
    let input = Run.of_float_array ~shape:[6] [|1.; 2.; 3.; 4.; 5.; 6.|] in
    let output = Run.of_float_array ~shape:[6] (Array.make 6 0.) in
    let result = first (T.custom_kernel ~fxn:custom_add_one_kernel [output; input]) in
    let view = Mv.reshape (Mv.shrink result [1, 4]) [3] in
    is_true ~msg:"a pending call cannot be bypassed by resolving its storage view"
      (Option.is_none (Run.buffer_of_node (T.uop view)));
    equal (array (float 1e-6)) [|3.; 4.; 5.|] (Run.to_float_array view);
    equal (array (float 1e-6)) [|2.; 3.; 4.; 5.; 6.; 7.|] (Run.to_float_array output));
  test "invalid partial stores preserve uncovered reads" (fun () ->
    let input = Run.of_float_array ~shape:[4] [|10.; 20.; 30.; 40.|] in
    let src = T.uop input in
    let prefix = U.shrink ~src ~offset:(U.const_int 0) ~size:(U.const_int 2) in
    let result = U.after ~src ~deps:[store prefix (U.invalid ())] |> T.of_uop in
    equal (array (float 1e-6)) [|10.; 20.; 30.; 40.|] (Run.to_float_array result));
  test "invalid stores retain neighboring writes" (fun () ->
    let output = Creation.empty [4] |> T.uop in
    let result = U.after ~src:output
        ~deps:[store output (const_like output 5.0); store output (U.invalid ())]
        |> T.of_uop in
    all_floats 5.0 result);
]

let () = run "Tolk_frontend_custom_kernel"
    [tests; input_tests; storage_tests; group "explicit call outputs"
      [test "inline calls write outputs at explicit positions" (explicit_call_outputs false);
       test "precompiled calls write outputs at explicit positions" (explicit_call_outputs true);
       test "nested calls resolve anonymous internal storage" nested_call_internal_allocation;
       test "inline outputs keep symbolic extents" (symbolic_call_outputs false);
       test "precompiled outputs keep symbolic extents" (symbolic_call_outputs true)]]
