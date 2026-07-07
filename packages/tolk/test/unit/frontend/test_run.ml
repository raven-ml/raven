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
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] (Op.stack a [ b ]));
      test "stack along inner axis" (fun () ->
          let a = vec [| 1.; 2. |] and b = vec [| 3.; 4. |] in
          check_floats [| 1.; 3.; 2.; 4. |] (Op.stack ~dim:1 a [ b ]));
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
          ignore (Run.realize (Op.assign view (Op.stack xk [ xv ])));
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
            ignore (Run.realize (Op.assign view (Op.stack xk [ xv ])))
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
      gpt2_getitem_tests;
      conv_tests;
      stack_tests;
      triu_tests;
      assign_tests;
      attention_tests;
    ]
