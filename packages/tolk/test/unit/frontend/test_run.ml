(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Numeric end-to-end tests: build a tensor expression, realize it on the CPU
   backend, and assert the computed values against expectations derived from
   tinygrad. *)

open Windtrap
module T = Tolk_frontend.Tensor
module Mv = Tolk_frontend.Movement
module El = Tolk_frontend.Elementwise
module Rd = Tolk_frontend.Reduce
module Op = Tolk_frontend.Op
module Dt = Tolk_frontend.Dtype_ops
module Creation = Tolk_frontend.Creation
module Run = Tolk_frontend.Run
module U = Tolk_uop.Uop

let fa ~shape data = Run.of_float_array ~shape data
let vec data = Run.of_float_array ~shape:[ Array.length data ] data

let close a b = Float.abs (a -. b) < 1e-4

let check_floats expected t =
  let got = Run.to_float_array t in
  equal int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      if not (close e got.(i)) then
        failf "element %d: expected %g, got %g" i e got.(i))
    expected

let check_ints expected t =
  equal (array int) expected (Run.to_int_array t)

let count_kernels t =
  let sink = U.sink [ U.contiguous ~src:(T.uop t) () ] in
  List.length
    (List.filter
       (fun u -> U.op u = Tolk_uop.Ops.Call)
       (U.toposort (Tolk.Rangeify.get_kernel_graph sink)))

let elementwise_tests =
  group "elementwise"
    [
      test "add" (fun () ->
          check_floats [| 11.; 22.; 33. |]
            (El.add (vec [| 1.; 2.; 3. |]) (vec [| 10.; 20.; 30. |])));
      test "mul then relu" (fun () ->
          check_floats [| 0.; 2.; 0.; 4. |]
            (El.relu (vec [| -1.; 2.; -3.; 4. |])));
      test "exp" (fun () ->
          check_floats [| 1.; 2.718282; 7.389056 |] (El.exp (vec [| 0.; 1.; 2. |])));
      test "pow" (fun () ->
          check_floats [| 4.; 9. |] (El.pow (vec [| 2.; 3. |]) (T.f 2.0)));
      test "broadcast add scalar" (fun () ->
          check_floats [| 6.; 7.; 8. |] (El.add (vec [| 1.; 2.; 3. |]) (T.f 5.0)));
      test "contiguous preserves values" (fun () ->
          check_floats [| 2.; 4.; 6. |]
            (El.contiguous (El.add (vec [| 1.; 2.; 3. |]) (vec [| 1.; 2.; 3. |]))));
    ]

let reduce_tests =
  group "reduce"
    [
      test "weak reductions preserve values beyond int32" (fun () ->
          let large = 1 lsl 40 in
          let input = Mv.expand (Mv.reshape (T.i large) [ 1 ]) [ 3 ] in
          let check expected tensor =
            let bytes = Run.data tensor in
            equal int 8 (Bytes.length bytes);
            equal int64 (Int64.of_int expected) (Bytes.get_int64_le bytes 0)
          in
          check (3 * large) (Rd.sum input);
          check large (Rd.max input);
          check large (Rd.prod (Mv.reshape (T.i large) [ 1 ])));
      test "weak integer mean preserves values beyond int32" (fun () ->
          let large = 1 lsl 40 in
          let input = Mv.expand (Mv.reshape (T.i large) [ 1 ]) [ 3 ] in
          check_floats [| float_of_int large |] (Op.mean input));
      test "sum all" (fun () ->
          check_floats [| 21. |] (Rd.sum (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
      test "sum axis 0" (fun () ->
          check_floats [| 5.; 7.; 9. |]
            (Rd.sum ~axis:[ 0 ] (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
      test "mean" (fun () ->
          check_floats [| 2.5 |] (Op.mean (vec [| 1.; 2.; 3.; 4. |])));
      test "max" (fun () -> check_floats [| 5. |] (Rd.max (vec [| 1.; 5.; 3. |])));
      test "argmax and argmin take the first of equal extremes" (fun () ->
          check_ints [| 1 |] (Op.argmax (vec [| 1.; 3.; 3.; 0. |]));
          check_ints [| 0 |] (Op.argmin (vec [| 0.; 3.; 3.; 0. |])));
    ]

let matmul_tests =
  group "matmul"
    [
      test "float matmul" (fun () ->
          check_floats [| 4.; 5.; 10.; 11. |]
            (Op.matmul
               (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])
               (fa ~shape:[ 3; 2 ] [| 1.; 0.; 0.; 1.; 1.; 1. |])));
      test "int matmul" (fun () ->
          check_ints [| 1; 2; 3; 4 |]
            (Op.matmul
               (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 2; 3; 4 |])
               (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 0; 0; 1 |])));
    ]

let scan_tests =
  group "scan"
    [
      test "cumsum 1d" (fun () ->
          check_floats [| 1.; 3.; 6.; 10. |] (Op.cumsum (vec [| 1.; 2.; 3.; 4. |])));
      test "cumsum 2d axis 1" (fun () ->
          check_floats [| 1.; 3.; 6.; 4.; 9.; 15. |]
            (Op.cumsum ~axis:1 (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
    ]

let logspace_tests =
  group "logspace"
    [
      test "softmax" (fun () ->
          check_floats [| 0.0900306; 0.2447284; 0.6652409 |]
            (Op.softmax (vec [| 1.; 2.; 3. |])));
      test "logsumexp" (fun () ->
          check_floats [| 3.4076059 |] (Op.logsumexp (vec [| 1.; 2.; 3. |])));
      (* A sink is a key of value zero with no column: its logit joins the
         shift and the normaliser. It is finite, so a row with every key masked
         divides 0 by 1. *)
      test "a sink keeps a fully masked softmax row at zero" (fun () ->
          let scores = fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 1.; 2.; 3. |] in
          let seen =
            El.gt (fa ~shape:[ 2; 3 ] [| 1.; 1.; 1.; 0.; 0.; 0. |]) (T.f 0.5)
          in
          let sink = fa ~shape:[ 1; 1 ] [| 0. |] in
          let scores = El.where seen scores (T.f Float.neg_infinity) in
          let top = El.maximum (Rd.max ~axis:[ 1 ] ~keepdim:true scores) sink in
          let e = El.exp (El.sub scores top) in
          let total =
            El.add (Rd.sum ~axis:[ 1 ] ~keepdim:true e) (El.exp (El.sub sink top))
          in
          check_floats
            [| 0.087144; 0.236883; 0.643914; 0.; 0.; 0. |]
            (El.div e total));
    ]

let getitem_tests =
  let base () = fa ~shape:[ 3; 4 ] (Array.init 12 float_of_int) in
  group "getitem"
    [
      test "slice" (fun () ->
          check_floats [| 5.; 6.; 9.; 10. |]
            (Op.getitem (base ())
               [ Mv.R (Some 1, Some 3, None); Mv.R (Some 1, Some 3, None) ]));
      test "int index" (fun () ->
          check_floats [| 4.; 5.; 6.; 7. |] (Op.getitem (base ()) [ Mv.I 1 ]));
      test "strided" (fun () ->
          check_floats [| 0.; 2. |]
            (Op.getitem (base ()) [ Mv.I 0; Mv.R (None, None, Some 2) ]));
      test "tensor index" (fun () ->
          check_floats [| 8.; 9.; 10.; 11.; 0.; 1.; 2.; 3. |]
            (Op.getitem (base ()) [ Mv.T (Run.of_int_array ~shape:[ 2 ] [| 2; 0 |]) ]));
    ]

let large_gather_tests =
  let rows = 65_536 in
  group "large gather"
    [
      test "gather over 65536 rows schedules to one kernel" (fun () ->
          let param slot dtype dims =
            T.of_uop
              (U.param ~slot ~dtype ~shape:(T.shape_uop dims)
                 ~device:(U.Single "CPU") ())
          in
          let table = param 0 Tolk_uop.Dtype.float32 [ rows; 4 ] in
          let index = param 1 Tolk_uop.Dtype.int32 [ 8; 4 ] in
          equal int 1 (count_kernels (Op.gather table ~dim:0 index)));
      test "gather over 65536 rows reads the indexed rows" (fun () ->
          let table =
            fa ~shape:[ rows; 2 ]
              (Array.init (rows * 2) (fun i -> float_of_int (i mod 1000)))
          in
          let index =
            Run.of_int_array ~shape:[ 3; 2 ]
              [| 0; 65_535; 40_000; 1; 123; 32_768 |]
          in
          check_floats [| 0.; 71.; 0.; 3.; 246.; 537. |]
            (Op.gather table ~dim:0 index));
    ]

let conv_tests =
  group "conv"
    [
      test "conv2d 3x3 with 2x2 kernel" (fun () ->
          let x = fa ~shape:[ 1; 1; 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))) in
          let w = fa ~shape:[ 1; 1; 2; 2 ] [| 1.; 0.; 0.; 1. |] in
          check_floats [| 6.; 8.; 12.; 14. |] (Op.conv2d x w));
    ]

let select_tests =
  let mask ~shape bits = Dt.bool (Run.of_int_array ~shape bits) in
  group "select"
    [
      test "masked_select 2d with fill" (fun () ->
          check_floats [| 0.; 2.; 4.; 8.; -1.; -1. |]
            (Op.masked_select ~fill_value:(T.Sfloat (-1.))
               (fa ~shape:[ 3; 3 ] (Array.init 9 float_of_int))
               (mask ~shape:[ 3; 3 ] [| 1; 0; 1; 0; 1; 0; 0; 0; 1 |])
               ~size:6));
      test "masked_select 1d" (fun () ->
          check_floats [| 1.; 3.; 5. |]
            (Op.masked_select
               (vec [| 1.; 2.; 3.; 4.; 5. |])
               (mask ~shape:[ 5 ] [| 1; 0; 1; 0; 1 |])
               ~size:3));
      test "masked_select truncates overflow" (fun () ->
          check_floats [| 1.; 3. |]
            (Op.masked_select
               (vec [| 1.; 2.; 3.; 4.; 5. |])
               (mask ~shape:[ 5 ] [| 1; 0; 1; 0; 1 |])
               ~size:2));
      test "nonzero 1d" (fun () ->
          check_ints [| 0; 2; 4 |]
            (Op.nonzero ~fill_value:(T.Sint (-1))
               (Run.of_int_array ~shape:[ 5 ] [| 1; 0; 2; 0; 3 |])
               ~size:3));
      test "nonzero 2d" (fun () ->
          check_ints [| 0; 0; 1; 1 |]
            (Op.nonzero
               (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 0; 0; 2 |])
               ~size:2));
      test "nonzero pads with fill" (fun () ->
          check_ints [| 0; 4; -1; -1 |]
            (Op.nonzero ~fill_value:(T.Sint (-1))
               (Run.of_int_array ~shape:[ 5 ] [| 1; 0; 0; 0; 5 |])
               ~size:4));
    ]

let dynamic_select_tests =
  let mask ~shape bits = Dt.bool (Run.of_int_array ~shape bits) in
  group "dynamic_select"
    [
      test "masked_select dynamic size" (fun () ->
          check_floats [| 1.; 3.; 5. |]
            (Run.masked_select
               (vec [| 1.; 2.; 3.; 4.; 5. |])
               (mask ~shape:[ 5 ] [| 1; 0; 1; 0; 1 |])));
      test "nonzero dynamic size" (fun () ->
          check_ints [| 0; 2; 4 |]
            (Run.nonzero (Run.of_int_array ~shape:[ 5 ] [| 1; 0; 2; 0; 3 |])));
      test "list-style advanced index" (fun () ->
          check_floats [| 8.; 9.; 10.; 11.; 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |]
            (Op.getitem
               (fa ~shape:[ 3; 4 ] (Array.init 12 float_of_int))
               [ Mv.T (Run.of_int_array ~shape:[ 3 ] [| 2; 0; 1 |]) ]));
    ]

let scatter_tests =
  let fi ~shape data = Run.of_int_array ~shape data in
  let src10 = fa ~shape:[ 2; 5 ] (Array.init 10 (fun i -> float_of_int (i + 1))) in
  let zeros35 = fa ~shape:[ 3; 5 ] (Array.make 15 0.) in
  let idx0 () = fi ~shape:[ 2; 5 ] (Array.make 10 0) in
  group "scatter"
    [
      test "scatter along dim 0" (fun () ->
          check_floats
            [| 1.; 0.; 0.; 4.; 0.; 0.; 2.; 0.; 0.; 0.; 0.; 0.; 3.; 0.; 0. |]
            (Op.scatter zeros35 ~dim:0 (fi ~shape:[ 1; 4 ] [| 0; 1; 2; 0 |]) src10));
      test "scatter along dim 1" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 0.; 0.; 6.; 7.; 0.; 0.; 8.; 0.; 0.; 0.; 0.; 0. |]
            (Op.scatter zeros35 ~dim:1
               (fi ~shape:[ 2; 3 ] [| 0; 1; 2; 0; 1; 4 |])
               (fa ~shape:[ 2; 5 ] (Array.init 10 (fun i -> float_of_int (i + 1))))));
      test "scatter_reduce sum" (fun () ->
          check_floats [| 8.; 10.; 12.; 14.; 16. |]
            (Op.scatter_reduce (fa ~shape:[ 1; 5 ] (Array.make 5 1.)) ~dim:0
               (idx0 ()) src10 ~reduce:`Sum ()));
      test "scatter_reduce prod" (fun () ->
          check_floats [| 6.; 14.; 24.; 36.; 50. |]
            (Op.scatter_reduce (fa ~shape:[ 1; 5 ] (Array.make 5 1.)) ~dim:0
               (idx0 ()) src10 ~reduce:`Prod ()));
      test "scatter_reduce amax" (fun () ->
          check_floats [| 6.; 20.; 8.; 9.; 10. |]
            (Op.scatter_reduce
               (fa ~shape:[ 1; 5 ] [| -10.; 20.; 0.; 5.; 10. |])
               ~dim:0 (idx0 ()) src10 ~reduce:`Amax ()));
      test "scatter_reduce amin" (fun () ->
          check_floats [| -10.; 2.; 0.; 4.; 5. |]
            (Op.scatter_reduce
               (fa ~shape:[ 1; 5 ] [| -10.; 20.; 0.; 5.; 10. |])
               ~dim:0 (idx0 ()) src10 ~reduce:`Amin ()));
      test "scatter_reduce mean excluding self" (fun () ->
          check_floats [| 3.5; 4.5; 5.5; 6.5; 7.5 |]
            (Op.scatter_reduce (fa ~shape:[ 1; 5 ] (Array.make 5 1.)) ~dim:0
               (idx0 ()) src10 ~reduce:`Mean ~include_self:false ()));
    ]

let scatter_indexed_tests =
  let fi ~shape data = Run.of_int_array ~shape data in
  let iota shape =
    let n = List.fold_left ( * ) 1 shape in
    fa ~shape (Array.init n (fun i -> float_of_int (i + 1)))
  in
  let zeros shape = fa ~shape (Array.make (List.fold_left ( * ) 1 shape) 0.) in
  let indexed ?(unique = false) mode t ~dim index src =
    let device = List.find_map T.device [ t; index; src ] in
    Op.scatter_indexed (Creation.clone ?device t) ~dim index src ~mode ~unique
  in
  let same_as_reference name ~dim t index src =
    [
      test (name ^ ", set") (fun () ->
          check_floats
            (Run.to_float_array (Op.scatter (t ()) ~dim (index ()) (src ())))
            (indexed `Set (t ()) ~dim (index ()) (src ())));
      test (name ^ ", add") (fun () ->
          check_floats
            (Run.to_float_array
               (Op.scatter_reduce (t ()) ~dim (index ()) (src ()) ~reduce:`Sum
                  ()))
            (indexed `Add (t ()) ~dim (index ()) (src ())));
    ]
  in
  group "scatter_indexed"
    (same_as_reference "rows with a repeated index" ~dim:0
       (fun () -> iota [ 4; 3 ])
       (fun () -> fi ~shape:[ 3; 3 ] [| 2; 0; 1; 2; 3; 1; 0; 0; 1 |])
       (fun () -> iota [ 3; 3 ])
    @ same_as_reference "every update in one lane aims at one cell" ~dim:1
        (fun () -> iota [ 2; 4 ])
        (fun () -> fi ~shape:[ 2; 3 ] [| 1; 1; 1; 3; 3; 3 |])
        (fun () -> iota [ 2; 3 ])
    @ same_as_reference "middle axis" ~dim:1
        (fun () -> iota [ 2; 4; 3 ])
        (fun () ->
          fi ~shape:[ 2; 2; 3 ] [| 3; 0; 1; 3; 2; 1; 0; 0; 0; 1; 2; 3 |])
        (fun () -> iota [ 2; 2; 3 ])
    @ [
        test "an index outside the axis writes nothing" (fun () ->
            let index () = fi ~shape:[ 4; 1 ] [| -1; 4; 1; -7 |] in
            check_floats [| 0.; 0.; 5.; 6.; 0.; 0.; 0.; 0. |]
              (indexed `Set (zeros [ 4; 2 ]) ~dim:0 (index ()) (iota [ 4; 2 ]));
            check_floats [| 1.; 2.; 8.; 10.; 5.; 6.; 7.; 8. |]
              (indexed `Add (iota [ 4; 2 ]) ~dim:0 (index ()) (iota [ 4; 2 ])));
        test "an index broadcast off the axis is read once per row" (fun () ->
            let index = fi ~shape:[ 3; 1 ] [| 2; 0; 2 |] in
            let broadcast = Mv.expand index [ 3; 2 ] in
            check_floats [| 3.; 4.; 0.; 0.; 5.; 6.; 0.; 0. |]
              (indexed `Set (zeros [ 4; 2 ]) ~dim:0 broadcast (iota [ 3; 2 ]));
            check_floats [| 3.; 4.; 0.; 0.; 6.; 8.; 0.; 0. |]
              (indexed `Add (zeros [ 4; 2 ]) ~dim:0 broadcast (iota [ 3; 2 ])));
        test "unique indices" (fun () ->
            check_floats [| 0.; 0.; 1.; 2.; 0.; 0.; 3.; 4. |]
              (indexed ~unique:true `Set (zeros [ 4; 2 ]) ~dim:0
                 (fi ~shape:[ 2; 1 ] [| 1; 3 |])
                 (iota [ 2; 2 ])));
        test "unique indices broken at one row leave every other row exact"
          (fun () ->
            let got =
              Run.to_float_array
                (indexed ~unique:true `Set (zeros [ 4; 3 ]) ~dim:0
                   (fi ~shape:[ 4; 1 ] [| 1; 3; 0; 3 |])
                   (iota [ 4; 3 ]))
            in
            let expected = [| 7.; 8.; 9.; 1.; 2.; 3.; 0.; 0.; 0. |] in
            Array.iteri
              (fun i e ->
                is_true ~msg:(Printf.sprintf "element %d" i) (close e got.(i)))
              expected;
            for j = 0 to 2 do
              let v = got.(9 + j) in
              is_true
                ~msg:(Printf.sprintf "the repeated row holds one update at %d" j)
                (close v (float_of_int (4 + j)) || close v (float_of_int (10 + j)))
            done);
        test "a cloned destination keeps its value" (fun () ->
            let t = iota [ 3 ] in
            check_floats [| 1.; 9.; 3. |]
              (indexed `Set t ~dim:0 (fi ~shape:[ 1 ] [| 1 |])
                 (fa ~shape:[ 1 ] [| 9. |]));
            check_floats [| 1.; 2.; 3. |] t);
        test "the write lands in the destination's storage" (fun () ->
            let t = iota [ 3 ] in
            let written =
              Run.realize
                (Op.scatter_indexed t ~dim:0 (fi ~shape:[ 1 ] [| 1 |])
                   (fa ~shape:[ 1 ] [| 9. |])
                   ~mode:`Set ~unique:false)
            in
            check_floats [| 1.; 9.; 3. |] written;
            check_floats [| 1.; 9.; 3. |] t);
        test "a destination that is not storage is refused" (fun () ->
            raises (Invalid_argument "Op.scatter_indexed: self must be storage")
              (fun () ->
                Op.scatter_indexed
                  (El.add (iota [ 3 ]) (iota [ 3 ]))
                  ~dim:0 (fi ~shape:[ 1 ] [| 1 |])
                  (fa ~shape:[ 1 ] [| 9. |])
                  ~mode:`Set ~unique:false));
        test "constant destination and source" (fun () ->
            check_floats [| 0.; 0.; 2.5; 2.5; 0.; 0. |]
              (indexed `Add
                 (Creation.full ~buffer:false [ 3; 2 ] (T.Sfloat 0.0))
                 ~dim:0
                 (fi ~shape:[ 1; 2 ] [| 1; 1 |])
                 (Creation.full ~buffer:false [ 1; 2 ] (T.Sfloat 2.5))));
        test "a source that is a view of the destination" (fun () ->
            let x = iota [ 2; 3 ] in
            check_floats [| 2.; 2.; 1.; 4.; 5.; 6. |]
              (indexed `Set x ~dim:1
                 (fi ~shape:[ 2; 2 ] [| 2; 0; 1; 1 |])
                 (Mv.shrink x [ (0, 2); (0, 2) ])));
        test "int payload" (fun () ->
            check_ints [| 7; 0; 12 |]
              (indexed `Add
                 (Run.of_int_array ~shape:[ 3 ] [| 0; 0; 0 |])
                 ~dim:0
                 (fi ~shape:[ 3 ] [| 2; 0; 2 |])
                 (Run.of_int_array ~shape:[ 3 ] [| 5; 7; 7 |])));
        test "2048 updates schedule to a fill and a write" (fun () ->
            let param slot dtype dims =
              T.of_uop
                (U.param ~slot ~dtype ~shape:(T.shape_uop dims)
                   ~device:(U.Single "CPU") ())
            in
            let t = param 0 Tolk_uop.Dtype.float32 [ 4096; 8 ] in
            let index = param 1 Tolk_uop.Dtype.int32 [ 2048; 8 ] in
            let src = param 2 Tolk_uop.Dtype.float32 [ 2048; 8 ] in
            let written = T.uop (indexed `Set t ~dim:0 index src) in
            let graph = Tolk.Rangeify.get_kernel_graph (U.sink [ written ]) in
            let calls =
              List.filter
                (fun u -> U.op u = Tolk_uop.Ops.Call)
                (U.toposort graph)
            in
            equal int 2 (List.length calls));
      ])

let clone_tests =
  let param device =
    T.of_uop
      (U.param ~slot:0 ~dtype:Tolk_uop.Dtype.float32 ~shape:(T.shape_uop [ 4 ])
         ~device:(U.Single device) ())
  in
  let copies t =
    List.length
      (List.filter
         (fun u -> U.op u = Tolk_uop.Ops.Copy)
         (U.toposort (T.uop t)))
  in
  group "clone"
    [
      test "a clone on another device copies its source across" (fun () ->
          let t = Creation.clone ~device:(U.Single "METAL") (param "CPU") in
          equal int 1 (copies t);
          is_true ~msg:"placed on the device asked for"
            (T.device t = Some (U.Single "METAL")));
      test "a clone on its source's device copies nothing across" (fun () ->
          equal int 0 (copies (Creation.clone (param "CPU")));
          equal int 0
            (copies (Creation.clone ~device:(U.Single "CPU") (param "CPU"))));
      test "empty storage is placed on the device asked for" (fun () ->
          is_true ~msg:"device"
            (T.device (Creation.empty ~device:(U.Single "CPU") [ 2; 2 ])
            = Some (U.Single "CPU")));
    ]

let sort_tests =
  group "sort"
    [
      test "sort ascending values and indices" (fun () ->
          let v, i = Op.sort (vec [| 3.; 1.; 2.; 5.; 4. |]) in
          check_floats [| 1.; 2.; 3.; 4.; 5. |] v;
          check_ints [| 1; 2; 0; 4; 3 |] i);
      test "sort descending" (fun () ->
          let v, i = Op.sort ~descending:true (vec [| 3.; 1.; 2.; 5.; 4. |]) in
          check_floats [| 5.; 4.; 3.; 2.; 1. |] v;
          check_ints [| 3; 4; 0; 2; 1 |] i);
      test "argsort" (fun () ->
          check_ints [| 1; 2; 0; 4; 3 |] (Op.argsort (vec [| 3.; 1.; 2.; 5.; 4. |])));
      test "sort 2d axis 1" (fun () ->
          let v, i = Op.sort ~dim:1 (fa ~shape:[ 2; 3 ] [| 3.; 1.; 2.; 6.; 5.; 4. |]) in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] v;
          check_ints [| 1; 2; 0; 2; 1; 0 |] i);
      test "sort keeps ties stable" (fun () ->
          let v, i = Op.sort (vec [| 2.; 1.; 2.; 1. |]) in
          check_floats [| 1.; 1.; 2.; 2. |] v;
          check_ints [| 1; 3; 0; 2 |] i);
      test "sort length not a power of two" (fun () ->
          let _, i = Op.sort (vec [| 8.; 3.; 5.; 1.; 7.; 2.; 6.; 4. |]) in
          check_ints [| 3; 5; 1; 7; 2; 6; 4; 0 |] i);
      test "sort int input" (fun () ->
          let v, i = Op.sort (Run.of_int_array ~shape:[ 5 ] [| 3; 1; 2; 5; 4 |]) in
          check_ints [| 1; 2; 3; 4; 5 |] v;
          check_ints [| 1; 2; 0; 4; 3 |] i);
      test "topk largest" (fun () ->
          let v, i = Op.topk (vec [| 1.; 5.; 3.; 4.; 2. |]) 3 in
          check_floats [| 5.; 4.; 3. |] v;
          check_ints [| 1; 3; 2 |] i);
      test "topk smallest" (fun () ->
          let v, i = Op.topk ~largest:false (vec [| 1.; 5.; 3.; 4.; 2. |]) 2 in
          check_floats [| 1.; 2. |] v;
          check_ints [| 0; 4 |] i);
    ]

(* Values may be infinite; [close] is NaN on two infinities of the same
   sign, so compare those for equality instead. *)
let check_floats_inf expected t =
  let got = Run.to_float_array t in
  equal int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      let ok =
        if Float.is_finite e then close e got.(i) else e = got.(i)
      in
      if not ok then failf "element %d: expected %g, got %g" i e got.(i))
    expected

let stack_tests =
  group "stack"
    [
      test "stack along new leading axis" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 5.; 6.; 7.; 8. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] (Mv.stack a [ b ]));
      test "stack along inner axis" (fun () ->
          let a = vec [| 1.; 2. |] and b = vec [| 3.; 4. |] in
          check_floats [| 1.; 3.; 2.; 4. |] (Mv.stack ~dim:1 a [ b ]));
    ]

(* [cat] takes one of two lowerings: equal extents on the joined axis stack
   and merge the new axis, anything else pads and sums. Both are covered on
   every axis, since only the equal-extent case reaches the stack. *)
let cat_tests =
  group "cat"
    [
      test "equal extents along axis 0" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 5.; 6.; 7.; 8. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] (Op.cat a [ b ]));
      test "equal extents along axis 1" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 5.; 6.; 7.; 8. |] in
          check_floats
            [| 1.; 2.; 5.; 6.; 3.; 4.; 7.; 8. |]
            (Op.cat ~dim:1 a [ b ]));
      test "equal extents, three operands" (fun () ->
          let a = vec [| 1.; 2. |] and b = vec [| 3.; 4. |] in
          let c = vec [| 5.; 6. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] (Op.cat a [ b; c ]));
      test "unequal extents along axis 0" (fun () ->
          let a = fa ~shape:[ 1; 2 ] [| 1.; 2. |] in
          let b = fa ~shape:[ 2; 2 ] [| 3.; 4.; 5.; 6. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] (Op.cat a [ b ]));
      test "unequal extents along axis 1" (fun () ->
          let a = fa ~shape:[ 2; 1 ] [| 1.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 2.; 3.; 5.; 6. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] (Op.cat ~dim:1 a [ b ]));
      test "a single operand is the identity" (fun () ->
          check_floats [| 1.; 2.; 3. |] (Op.cat (vec [| 1.; 2.; 3. |]) []));
      test "negative axis counts from the end" (fun () ->
          let a = fa ~shape:[ 2; 1 ] [| 1.; 3. |] in
          let b = fa ~shape:[ 2; 1 ] [| 2.; 4. |] in
          check_floats [| 1.; 2.; 3.; 4. |] (Op.cat ~dim:(-1) a [ b ]));
      test "booleans keep their values" (fun () ->
          let b v = Dt.bool (vec v) in
          check_ints [| 1; 0; 0; 1 |]
            (Dt.int (Op.cat (b [| 1.; 0. |]) [ b [| 0.; 1. |] ])));
      test "operands are promoted to a common dtype" (fun () ->
          let a = Dt.int (vec [| 1.; 2. |]) and b = vec [| 3.5; 4.5 |] in
          check_floats [| 1.; 2.; 3.5; 4.5 |] (Op.cat a [ b ]));
      test "shapes must match off the joined axis" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 3 ] (Array.make 6 0.) in
          raises
            (Invalid_argument
               "Op.cat: shapes must match off the concatenated axis")
            (fun () -> ignore (Op.cat a [ b ])));
    ]

let triu_tests =
  group "triu"
    [
      test "triu main diagonal" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 0.; 5.; 6.; 0.; 0.; 9. |]
            (Op.triu (fa ~shape:[ 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))))));
      test "triu positive diagonal" (fun () ->
          check_floats
            [| 0.; 2.; 3.; 0.; 0.; 6.; 0.; 0.; 0. |]
            (Op.triu ~diagonal:1
               (fa ~shape:[ 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))))));
      test "triu negative diagonal" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 4.; 5.; 6.; 0.; 8.; 9. |]
            (Op.triu ~diagonal:(-1)
               (fa ~shape:[ 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))))));
      test "full neg-infinity causal mask" (fun () ->
          let mask =
            Op.triu ~diagonal:1
              (Creation.full [ 2; 2 ] (T.Sfloat Float.neg_infinity))
          in
          check_floats_inf
            [| 1.; Float.neg_infinity; 3.; 4. |]
            (El.add (fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |]) mask));
    ]

let assign_tests =
  group "assign"
    [
      test "assign whole tensor in place" (fun () ->
          let t = vec [| 1.; 2.; 3. |] in
          ignore (Run.realize t);
          ignore (Run.realize (Op.assign t (vec [| 4.; 5.; 6. |])));
          check_floats [| 4.; 5.; 6. |] t);
      test "assign broadcasts the value" (fun () ->
          let t = fa ~shape:[ 2; 3 ] (Array.make 6 0.) in
          ignore (Run.realize (Op.assign t (vec [| 1.; 2.; 3. |])));
          check_floats [| 1.; 2.; 3.; 1.; 2.; 3. |] t);
      test "assign dtype mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Op.assign (vec [| 1. |]) (Run.of_int_array ~shape:[ 1 ] [| 1 |])));
      test "assign to shrunk view then read back (kv cache)" (fun () ->
          (* cache is (2, bsz=1, ctx=4, heads=2, head_dim=3); write positions
             1..2 with the stack of xk and xv, as a transformer kv-cache
             update does, then read the whole cache and a keys slice back. *)
          let cache = fa ~shape:[ 2; 1; 4; 2; 3 ] (Array.make 48 0.) in
          ignore (Run.realize cache);
          let xk =
            fa ~shape:[ 1; 2; 2; 3 ] (Array.init 12 (fun i -> float_of_int (i + 1)))
          in
          let xv =
            fa ~shape:[ 1; 2; 2; 3 ]
              (Array.init 12 (fun i -> float_of_int (i + 13)))
          in
          let view =
            Op.getitem cache
              [ Mv.All; Mv.All; Mv.R (Some 1, Some 3, None); Mv.All; Mv.All ]
          in
          ignore (Run.realize (Op.assign view (Mv.stack xk [ xv ])));
          let expected = Array.make 48 0. in
          for pos = 0 to 1 do
            for h = 0 to 1 do
              for c = 0 to 2 do
                let src = (((pos * 2) + h) * 3) + c in
                expected.((((1 + pos) * 2 + h) * 3) + c) <-
                  float_of_int (src + 1);
                expected.((((4 + 1 + pos) * 2 + h) * 3) + c) <-
                  float_of_int (src + 13)
              done
            done
          done;
          check_floats expected cache;
          (* keys = cache[0][:, :3, :, :]: position 0 still zero, then xk. *)
          let keys =
            Op.getitem
              (Op.getitem cache [ Mv.I 0 ])
              [ Mv.All; Mv.R (None, Some 3, None); Mv.All; Mv.All ]
          in
          let expected_keys = Array.make 18 0. in
          Array.blit expected 6 expected_keys 6 12;
          check_floats expected_keys keys);
      test "sequential view assigns accumulate" (fun () ->
          let cache = fa ~shape:[ 2; 1; 4; 2; 3 ] (Array.make 48 0.) in
          ignore (Run.realize cache);
          let step start v =
            let xk = fa ~shape:[ 1; 1; 2; 3 ] (Array.make 6 v) in
            let xv = fa ~shape:[ 1; 1; 2; 3 ] (Array.make 6 (v +. 0.5)) in
            let view =
              Op.getitem cache
                [
                  Mv.All; Mv.All;
                  Mv.R (Some start, Some (start + 1), None);
                  Mv.All; Mv.All;
                ]
            in
            ignore (Run.realize (Op.assign view (Mv.stack xk [ xv ])))
          in
          step 0 1.;
          step 1 2.;
          let expected = Array.make 48 0. in
          for h = 0 to 5 do
            expected.(h) <- 1.;
            expected.(6 + h) <- 2.;
            expected.(24 + h) <- 1.5;
            expected.(30 + h) <- 2.5
          done;
          check_floats expected cache);
    ]

let attention_tests =
  let q () = fa ~shape:[ 2; 3 ] [| 0.1; 0.2; 0.3; -0.1; 0.4; 0.5 |] in
  let k () = fa ~shape:[ 2; 3 ] [| 0.5; 0.1; -0.2; 0.3; 0.9; 0.4 |] in
  let v () = fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  group "attention"
    [
      test "sdpa matches reference" (fun () ->
          check_floats
            [| 2.638171; 3.638171; 4.638171; 2.774017; 3.774017; 4.774017 |]
            (Op.scaled_dot_product_attention (q ()) (k ()) (v ())));
      test "sdpa causal matches reference" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 2.774017; 3.774017; 4.774017 |]
            (Op.scaled_dot_product_attention ~is_causal:true (q ()) (k ())
               (v ())));
      test "sdpa additive neg-infinity mask equals causal" (fun () ->
          let mask =
            Op.triu ~diagonal:1
              (Creation.full [ 2; 2 ] (T.Sfloat Float.neg_infinity))
          in
          check_floats
            (Run.to_float_array
               (Op.scaled_dot_product_attention ~is_causal:true (q ()) (k ())
                  (v ())))
            (Op.scaled_dot_product_attention ~attn_mask:mask (q ()) (k ())
               (v ())));
      test "sdpa boolean mask equals causal" (fun () ->
          let mask = Dt.bool (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 0; 1; 1 |]) in
          check_floats
            (Run.to_float_array
               (Op.scaled_dot_product_attention ~is_causal:true (q ()) (k ())
                  (v ())))
            (Op.scaled_dot_product_attention ~attn_mask:mask (q ()) (k ())
               (v ())));
      test "sdpa rejects mask with is_causal" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Op.scaled_dot_product_attention ~is_causal:true
                ~attn_mask:(fa ~shape:[ 2; 2 ] (Array.make 4 0.))
                (q ()) (k ()) (v ())));
      test "layernorm matches reference" (fun () ->
          check_floats
            [| -1.341635; -0.447212; 0.447212; 1.341635 |]
            (Op.layernorm (vec [| 1.; 2.; 3.; 4. |])));
    ]

let gpt2_getitem_tests =
  group "gpt2_getitem"
    [
      test "integer index in the middle of the rank" (fun () ->
          (* xqkv[:, :, i, :, :] on a (1, 2, 3, 2, 2) tensor. *)
          let data = Array.init 24 float_of_int in
          let t = fa ~shape:[ 1; 2; 3; 2; 2 ] data in
          let expected i =
            Array.init 8 (fun j ->
                let s = j / 4 and h = j mod 4 / 2 and d = j mod 2 in
                data.((((s * 3) + i) * 2 + h) * 2 + d))
          in
          check_floats (expected 1)
            (Op.getitem t [ Mv.All; Mv.All; Mv.I 1; Mv.All; Mv.All ]);
          check_floats (expected 2)
            (Op.getitem t [ Mv.All; Mv.All; Mv.I 2; Mv.All; Mv.All ]));
      test "negative integer index selects the last row" (fun () ->
          (* logits[:, -1, :] on a (2, 3, 4) tensor. *)
          let data = Array.init 24 float_of_int in
          let t = fa ~shape:[ 2; 3; 4 ] data in
          check_floats
            (Array.init 8 (fun j -> data.((j / 4 * 3 + 2) * 4 + (j mod 4))))
            (Op.getitem t [ Mv.All; Mv.I (-1); Mv.All ]));
      test "leading index then open-ended slice" (fun () ->
          (* cache_kv[0][:, :2, :, :] on a (2, 1, 3, 2, 2) tensor. *)
          let data = Array.init 24 float_of_int in
          let t = fa ~shape:[ 2; 1; 3; 2; 2 ] data in
          check_floats
            (Array.sub data 0 8)
            (Op.getitem
               (Op.getitem t [ Mv.I 0 ])
               [ Mv.All; Mv.R (None, Some 2, None); Mv.All; Mv.All ]));
    ]

(* Symbolic shapes: a shape dimension may be an expression over a bound
   variable. One traced graph then serves every bound value; the value is
   passed to the kernel at launch. *)


let bound_var name ~max_val value =
  let var = U.variable ~name ~min_val:0 ~max_val () in
  U.bind ~var ~value:(U.const_int value)

let plus1 u = U.O.(u + U.const_int 1)

let symbolic_tests =
  group "symbolic"
    [
      test "shrink + sum matches concrete for several bind values" (fun () ->
          let data = Array.init 8 (fun i -> float_of_int (i + 1)) in
          List.iter
            (fun pos ->
              let t = vec data in
              let bound = bound_var "pos" ~max_val:6 pos in
              let s =
                Mv.symbolic_shrink t [ Some (U.const_int 0, plus1 bound) ]
              in
              let expected =
                Array.fold_left ( +. ) 0. (Array.sub data 0 (pos + 1))
              in
              check_floats [| expected |] (Rd.sum s))
            [ 2; 5; 0 ]);
      test "attention with symbolic key/value length matches concrete"
        (fun () ->
          let qdata = [| 0.1; 0.2; 0.3; -0.1 |] in
          let kdata = Array.init 24 (fun i -> float_of_int (i mod 5) /. 5.) in
          let vdata = Array.init 24 (fun i -> float_of_int (i + 1)) in
          List.iter
            (fun pos ->
              let bound = bound_var "apos" ~max_val:4 pos in
              let sym_bounds = [ Some (U.const_int 0, plus1 bound); None ] in
              let keys =
                Mv.symbolic_shrink (fa ~shape:[ 6; 4 ] kdata) sym_bounds
              in
              let values =
                Mv.symbolic_shrink (fa ~shape:[ 6; 4 ] vdata) sym_bounds
              in
              let attend q k v =
                Op.matmul
                  (Op.softmax ~axis:(-1)
                     (Op.matmul q (Mv.transpose ~dim0:(-2) ~dim1:(-1) k)))
                  v
              in
              let got =
                attend (fa ~shape:[ 1; 4 ] qdata) keys values
              in
              let concrete_bounds = [ (0, pos + 1); (0, 4) ] in
              let expected =
                attend
                  (fa ~shape:[ 1; 4 ] qdata)
                  (Mv.shrink (fa ~shape:[ 6; 4 ] kdata) concrete_bounds)
                  (Mv.shrink (fa ~shape:[ 6; 4 ] vdata) concrete_bounds)
              in
              check_floats (Run.to_float_array expected) got)
            [ 1; 3 ]);
      test "kv-cache: assign at symbolic position, read symbolic prefix"
        (fun () ->
          (* cache is (ctx=4, width=2); each decode step writes one row at a
             symbolic position and reads back rows [0, pos+1). *)
          let cache = fa ~shape:[ 4; 2 ] (Array.make 8 0.) in
          ignore (Run.realize cache);
          let step pos value expected_sum =
            let bound = bound_var "cpos" ~max_val:3 pos in
            let row =
              fa ~shape:[ 1; 2 ] [| value; value +. 0.5 |]
            in
            let view =
              Mv.symbolic_shrink cache [ Some (bound, plus1 bound); None ]
            in
            ignore (Run.realize (Op.assign view row));
            let keys =
              Mv.symbolic_shrink cache
                [ Some (U.const_int 0, plus1 bound); None ]
            in
            check_floats [| expected_sum |] (Rd.sum keys)
          in
          step 0 1. 2.5;
          step 1 2. 7.;
          step 2 3. 13.5;
          check_floats [| 1.; 1.5; 2.; 2.5; 3.; 3.5; 0.; 0. |] cache);
      test "one schedule serves every bind value" (fun () ->
          (* The bound value is stripped from the schedule cache key, so the
             second value must reuse the first schedule instead of scheduling
             (and later compiling) a new kernel. *)
          let lowered = ref 0 in
          let get_kernel_graph sink =
            incr lowered;
            Tolk.Rangeify.get_kernel_graph sink
          in
          let schedule pos =
            let data = Array.init 8 (fun i -> float_of_int (i + 1)) in
            let t = vec data in
            let bound = bound_var "spos" ~max_val:6 pos in
            let s =
              Mv.symbolic_shrink t [ Some (U.const_int 0, plus1 bound) ]
            in
            let out = Rd.sum (El.mul s (T.f 2.5)) in
            let sink =
              U.sink [ U.contiguous ~src:(T.uop out) () ]
            in
            let call, _ = Tolk.Callify.transform_to_call sink in
            ignore
              (U.graph_rewrite ~enter_calls:true
                 (fun node ->
                   Tolk.Schedule.lower_sink_to_linear ~get_kernel_graph node)
                 call)
          in
          schedule 2;
          equal int ~msg:"first value schedules" 1 !lowered;
          schedule 5;
          equal int ~msg:"second value hits the schedule cache" 1 !lowered);
    ]

(* A contiguous view of a realized buffer realizes as an alias of that buffer —
   the source memory viewed at the slice offset, with no copy — mirroring the
   reference, where such a view resolves lazily to [base.view(offset)]. *)
let aliasing_tests =
  let module Device = Tolk.Device in
  let buffer_of t =
    match Run.buffer_of_node (T.uop t) with
    | Some b -> b
    | None -> fail "tensor has no backing buffer after realize"
  in
  group "aliasing"
    [
      test "contiguous slice aliases its source buffer" (fun () ->
          let x = vec [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
          ignore (Run.realize x);
          let xbuf = buffer_of x in
          let y = Op.getitem x [ Mv.R (Some 2, Some 6, None) ] in
          ignore (Run.realize y);
          let ybuf = buffer_of y in
          (* Same underlying allocation, viewed at the slice's byte offset (2
             elements * 4 bytes), as a distinct view — not a fresh buffer. *)
          equal int
            ~msg:"slice shares the source allocation"
            (Device.Buffer.base_id xbuf) (Device.Buffer.base_id ybuf);
          equal int ~msg:"view starts at the slice offset" (2 * 4)
            (Device.Buffer.offset ybuf);
          if Device.Buffer.id xbuf = Device.Buffer.id ybuf then
            fail "slice must be a distinct view, not the base buffer itself";
          check_floats [| 2.; 3.; 4.; 5. |] y);
      test "explicit contiguous on a slice still aliases" (fun () ->
          let x = vec [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
          ignore (Run.realize x);
          let xbuf = buffer_of x in
          let y = El.contiguous (Op.getitem x [ Mv.R (Some 3, Some 7, None) ]) in
          ignore (Run.realize y);
          let ybuf = buffer_of y in
          equal int
            ~msg:"contiguous view shares the source allocation"
            (Device.Buffer.base_id xbuf) (Device.Buffer.base_id ybuf);
          equal int ~msg:"view starts at the slice offset" (3 * 4)
            (Device.Buffer.offset ybuf);
          check_floats [| 3.; 4.; 5.; 6. |] y);
      test "a shrunk trailing axis is not a contiguous range" (fun () ->
          let x = fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
          check_floats [| 1.; 2.; 4.; 5. |]
            (El.contiguous (Mv.shrink x [ (0, 2); (0, 2) ]));
          check_floats [| 2.; 3.; 5.; 6. |]
            (El.contiguous (Mv.shrink x [ (0, 2); (1, 3) ])));
      test "a shrunk leading axis is a contiguous range" (fun () ->
          let x = fa ~shape:[ 3; 2 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
          check_floats [| 3.; 4.; 5.; 6. |]
            (El.contiguous (Mv.shrink x [ (1, 3); (0, 2) ])));
    ]

(* An expression whose graph folds to a pure constant owns no storage and is
   placed on no device. Reading it must materialize it rather than fail. *)

let constant_tests =
  group "constant"
    [
      test "broadcast constant reads back" (fun () ->
          check_floats [| 3.; 3.; 3.; 3. |]
            (Creation.full ~buffer:false [ 4 ] (T.Sfloat 3.0)));
      test "expression folding to a constant reads back" (fun () ->
          check_floats [| 0.; 0.; 0. |]
            (El.mul (vec [| 1.; 2.; 3. |]) (T.f 0.0)));
      test "constant scalar reads back" (fun () ->
          equal float_exact 5.0 (Run.item_float (T.f 5.0)));
      test "constant realizes into a buffer only once" (fun () ->
          let t = Creation.full ~buffer:false [ 2; 3 ] (T.Sfloat 1.5) in
          check_floats [| 1.5; 1.5; 1.5; 1.5; 1.5; 1.5 |] t;
          check_floats [| 1.5; 1.5; 1.5; 1.5; 1.5; 1.5 |] t);
    ]

let numerical_edge_tests =
  let int64s values =
    let bytes = Bytes.create (Array.length values * 8) in
    Array.iteri (fun i n -> Bytes.set_int64_le bytes (8 * i) n) values;
    Run.of_bytes ~dtype:Tolk_uop.Dtype.int64 ~shape:[ Array.length values ] bytes
  in
  let check_int64s values tensor =
    let bytes = Run.data tensor in
    equal int (8 * Array.length values) (Bytes.length bytes);
    Array.iteri (fun i n -> equal int64 n (Bytes.get_int64_le bytes (8 * i))) values
  in
  let check expected t =
    let actual = Run.to_float_array t in
    equal int (Array.length expected) (Array.length actual);
    Array.iteri
      (fun i x ->
        let y = actual.(i) in
        if Float.is_nan x then is_true (Float.is_nan y)
        else if Float.is_infinite x then equal float_exact x y
        else if not (Float.abs (x -. y) <= 1e-4 *. max 1. (Float.abs x)) then
          failf "element %d: expected %g, got %g" i x y)
      expected
  in
  group "numerical edges"
    [
      test "int64 scan, sort and scatter retain full-width identities" (fun () ->
          let lo = Int64.min_int and hi = Int64.max_int in
          let ascending = [| lo; Int64.succ lo; Int64.add lo 2L |] in
          check_int64s ascending (fst (Op.cummax (int64s ascending)));
          let unordered = [| hi; Int64.sub hi 2L; Int64.pred hi |] in
          check_int64s [| Int64.sub hi 2L; Int64.pred hi; hi |]
            (fst (Op.sort (int64s unordered)));
          check_int64s [| hi; Int64.sub hi 2L; hi |]
            (Op.scatter_reduce (int64s [| hi; hi; hi |]) ~dim:0
               (Run.of_int_array ~shape:[ 1 ] [| 1 |])
               (int64s [| Int64.sub hi 2L |]) ~reduce:`Amin ~include_self:false ()));
      test "small-dtype arange accumulates before narrowing" (fun () ->
          let module D = Tolk_uop.Dtype in
          List.iter
            (fun dtype ->
              List.iter
                (fun (start, stop, step) ->
                  let expected =
                    Dt.float (Dt.cast (Op.arange ~stop ~step start) dtype)
                    |> Run.to_float_array
                  in
                  check expected (Dt.float (Op.arange ~dtype ~stop ~step start)))
                ([ (0, 10, 3); (0, 200, 1) ]
                 @ if D.is_fp8 dtype then [] else [ (0, 2560, 1) ]))
            [ D.float16; D.bfloat16; D.fp8e4m3; D.fp8e5m2 ]);
      test "pad_to preserves nonzero fill" (fun () ->
          let input = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          check [| 1.; 2.; -7.; 3.; 4.; -7.; -7.; -7.; -7. |]
            (Op.pad_to ~value:(T.Sfloat (-7.)) input [ Some 3; Some 3 ]);
          check [| 1.; 2.; 0.; 3.; 4.; 0. |]
            (Op.pad_to input [ None; Some 3 ]);
          is_true
            (T.uop (Op.pad_to ~value:(T.Sfloat (-7.)) input [ None; None ])
             == T.uop input));
      test "asinh handles large negative inputs" (fun () ->
          let values = [| -10000.; -1.; 0.; 1.; 10000. |] in
          check (Array.map Float.asinh values) (El.asinh (vec values)));
      test "logaddexp handles infinities" (fun () ->
          check [| neg_infinity; infinity; infinity; 0.; nan |]
            (El.logaddexp
               (vec [| neg_infinity; infinity; infinity; neg_infinity; nan |])
               (vec [| neg_infinity; infinity; 0.; 0.; 1. |])));
      test "logsumexp handles infinite rows" (fun () ->
          check [| neg_infinity; infinity; Float.log 3. |]
            (Op.logsumexp ~axis:1
               (fa ~shape:[ 3; 2 ]
                  [| neg_infinity; neg_infinity; infinity; 0.; 0.; Float.log 2. |])));
      test "logcumsumexp handles infinite prefixes" (fun () ->
          check [| neg_infinity; neg_infinity; 0.; infinity |]
            (Op.logcumsumexp (vec [| neg_infinity; neg_infinity; 0.; infinity |])));
      test "bounded activations saturate at large inputs" (fun () ->
          let input = vec [| 1e20; infinity; neg_infinity; 0. |] in
          check [| 6.; 6.; 0.; 0. |] (El.relu6 input);
          check [| 1.; 1.; 0.; 0.5 |] (El.hardsigmoid input));
      test "integer variance retains fractional squares" (fun () ->
          check [| 0.25 |]
            (Op.var ~correction:0 (Run.of_int_array ~shape:[ 2 ] [| 0; 1 |])));
      test "weak promotion preserves padding" (fun () ->
          let padded = Mv.pad (Mv.reshape (T.i 3) [ 1 ]) [ (1, 1) ] in
          check [| 1.; 4.; 1. |] (El.add padded (vec [| 1.; 1.; 1. |])));
      test "integer max pooling pads with the dtype minimum" (fun () ->
          let input = Run.of_int_array ~shape:[ 1; 1; 2; 2 ] [| -4; -3; -2; -1 |] in
          check_ints [| -4; -3; -3; -2; -1; -1; -2; -1; -1 |]
            (Op.max_pool2d ~stride:[ 1; 1 ] ~padding:[ 1 ] input));
      test "int64 max pooling preserves a full-width padding value" (fun () ->
          let data = Bytes.create 8 in
          Bytes.set_int64_le data 0 Int64.min_int;
          let input = Run.of_bytes ~dtype:Tolk_uop.Dtype.int64 ~shape:[ 1; 1; 1; 1 ] data in
          let out = Op.max_pool2d ~stride:[ 1; 1 ] ~padding:[ 1 ] input in
          let actual = Run.data out in
          equal int 32 (Bytes.length actual);
          for i = 0 to 3 do equal int64 Int64.min_int (Bytes.get_int64_le actual (8 * i)) done);
    ]

let lifetime_tests =
  group "lifetime"
    [
      test "empty host inputs need no native allocation" (fun () ->
          let floats = Run.of_float_array ~shape:[ 0; 3 ] [||] in
          let ints = Run.of_int_array ~shape:[ 2; 0 ] [||] in
          Run.realize_many [ floats; ints ];
          equal int 0 (Array.length (Run.to_float_array floats));
          equal int 0 (Array.length (Run.to_int_array ints));
          equal int 0 (Bytes.length (Run.data floats));
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Run.of_float_array ~shape:[ 3 ] [||]));
      test "unreachable input and realized storage are collectible" (fun () ->
          let nodes = Stdlib.Weak.create 2 and buffers = Stdlib.Weak.create 2 in
          let[@inline never] populate () =
            let input = vec [| 1.; 2.; 3. |] in
            let output = Run.realize (El.add input (T.f 1.)) in
            List.iteri
              (fun i tensor ->
                let node = U.buf_uop (T.uop tensor) in
                Stdlib.Weak.set nodes i (Some node);
                Stdlib.Weak.set buffers i (Run.buffer_of_node node))
              [ input; output ]
          in
          populate ();
          Gc.full_major ();
          Gc.full_major ();
          for i = 0 to 1 do
            is_false ~msg:"unreachable node retained" (Stdlib.Weak.check nodes i);
            is_false ~msg:"unreachable buffer retained" (Stdlib.Weak.check buffers i)
          done);
      test "live slice retains its backing storage" (fun () ->
          let[@inline never] make_view () =
            let input = vec [| 1.; 2.; 3.; 4. |] in
            Mv.shrink input [ (1, 3) ]
          in
          let view = make_view () in
          Gc.full_major ();
          check_floats [| 2.; 3. |] view);
    ]

let () =
  run "Tolk_frontend_run"
    [
      aliasing_tests;
      lifetime_tests;
      numerical_edge_tests;
      constant_tests;
      elementwise_tests;
      select_tests;
      dynamic_select_tests;
      scatter_tests;
      scatter_indexed_tests;
      clone_tests;
      sort_tests;
      reduce_tests;
      matmul_tests;
      scan_tests;
      logspace_tests;
      getitem_tests;
      gpt2_getitem_tests;
      large_gather_tests;
      conv_tests;
      stack_tests;
      cat_tests;
      triu_tests;
      assign_tests;
      attention_tests;
      symbolic_tests;
    ]
