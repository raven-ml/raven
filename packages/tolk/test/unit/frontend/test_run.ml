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
module Run = Tolk_frontend.Run

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
      test "sum all" (fun () ->
          check_floats [| 21. |] (Rd.sum (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
      test "sum axis 0" (fun () ->
          check_floats [| 5.; 7.; 9. |]
            (Rd.sum ~axis:[ 0 ] (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
      test "mean" (fun () ->
          check_floats [| 2.5 |] (Op.mean (vec [| 1.; 2.; 3.; 4. |])));
      test "max" (fun () -> check_floats [| 5. |] (Rd.max (vec [| 1.; 5.; 3. |])));
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

let () =
  run "Tolk_frontend_run"
    [
      elementwise_tests;
      select_tests;
      dynamic_select_tests;
      scatter_tests;
      sort_tests;
      reduce_tests;
      matmul_tests;
      scan_tests;
      logspace_tests;
      getitem_tests;
      conv_tests;
    ]
