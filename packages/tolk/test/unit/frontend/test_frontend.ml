(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Graph-shape tests for the tolk tensor frontend. Each test constructs a
   tensor expression and asserts the op, dtype, and shape of the resulting IR
   node. Expectations are derived from tinygrad's tensor frontend. *)

open Windtrap
module T = Tolk_frontend.Tensor
module Mv = Tolk_frontend.Movement
module El = Tolk_frontend.Elementwise
module Cr = Tolk_frontend.Creation
module Rd = Tolk_frontend.Reduce
module Dt = Tolk_frontend.Dtype_ops
module Op = Tolk_frontend.Op
module U = Tolk_uop.Uop
module Ops = Tolk_uop.Ops
module D = Tolk_uop.Dtype

let op t = U.op (T.uop t)
let src t k = (U.src (T.uop t)).(k)
let has_op t o = Ops.equal (op t) o
let is_dtype t d = D.equal (T.dtype t) d
let shape = T.shape

(* Test tensors *)

let ones_f sh = Cr.ones sh
let ones_i sh = Cr.ones ~dtype:D.Val.int32 sh

(* Creation *)

let creation_tests =
  group "creation"
    [
      test "zeros shape and dtype" (fun () ->
          let t = Cr.zeros [ 2; 3 ] in
          equal (list int) [ 2; 3 ] (shape t);
          is_true (is_dtype t D.float32));
      test "ones int dtype" (fun () ->
          let t = ones_i [ 4 ] in
          equal (list int) [ 4 ] (shape t);
          is_true (is_dtype t D.int32));
      test "full materializes a buffer by default" (fun () ->
          let t = Cr.full [ 2; 2 ] (T.Sint 7) in
          (* After(Reshape(Buffer), Store(...)) *)
          is_true (has_op t Ops.After);
          is_true (Ops.equal (U.op (src t 0)) Ops.Reshape));
      test "full ~buffer:false is a broadcast const" (fun () ->
          let t = Cr.full ~buffer:false [ 2; 2 ] (T.Sint 7) in
          (* Expand(Reshape(Const)) *)
          is_true (has_op t Ops.Expand);
          is_true (Ops.equal (U.op (src t 0)) Ops.Reshape));
      test "scalar const has empty shape" (fun () ->
          equal (list int) [] (shape (T.f 3.0)));
      test "full_like matches shape and dtype" (fun () ->
          let t = ones_i [ 3; 1 ] in
          let l = Cr.full_like t (T.Sint 0) in
          equal (list int) [ 3; 1 ] (shape l);
          is_true (is_dtype l D.int32));
    ]

(* Movement *)

let movement_tests =
  group "movement"
    [
      test "reshape" (fun () ->
          equal (list int) [ 6 ] (shape (Mv.reshape (ones_f [ 2; 3 ]) [ 6 ])));
      test "reshape infers -1" (fun () ->
          equal (list int) [ 2; 3 ]
            (shape (Mv.reshape (ones_f [ 6 ]) [ 2; -1 ])));
      test "reshape to same shape is identity node" (fun () ->
          let t = ones_f [ 2; 3 ] in
          is_true (U.equal (T.uop (Mv.reshape t [ 2; 3 ])) (T.uop t)));
      test "reshape size mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Mv.reshape (ones_f [ 2; 3 ]) [ 5 ]));
      test "expand broadcasts size-1 axis" (fun () ->
          equal (list int) [ 4; 3 ] (shape (Mv.expand (ones_f [ 1; 3 ]) [ 4; 3 ])));
      test "expand keeps -1 axis" (fun () ->
          equal (list int) [ 4; 3 ]
            (shape (Mv.expand (ones_f [ 1; 3 ]) [ 4; -1 ])));
      test "permute reorders" (fun () ->
          equal (list int) [ 5; 2; 3 ]
            (shape (Mv.permute (ones_f [ 2; 3; 5 ]) [ 2; 0; 1 ])));
      test "permute negative axes" (fun () ->
          equal (list int) [ 5; 3; 2 ]
            (shape (Mv.permute (ones_f [ 2; 3; 5 ]) [ -1; 1; 0 ])));
      test "flip preserves shape" (fun () ->
          let t = Mv.flip (ones_f [ 2; 3 ]) [ 0 ] in
          equal (list int) [ 2; 3 ] (shape t);
          is_true (has_op t Ops.Flip));
      test "pad grows dims" (fun () ->
          equal (list int) [ 5; 3 ]
            (shape (Mv.pad (ones_f [ 2; 3 ]) [ (1, 2); (0, 0) ])));
      test "shrink trims dims" (fun () ->
          equal (list int) [ 1; 2 ]
            (shape (Mv.shrink (ones_f [ 3; 3 ]) [ (1, 2); (0, 2) ])));
      test "squeeze removes size-1" (fun () ->
          equal (list int) [ 2; 3 ]
            (shape (Mv.squeeze (ones_f [ 2; 1; 3; 1 ]))));
      test "squeeze dim keeps non-unit" (fun () ->
          equal (list int) [ 2; 3 ] (shape (Mv.squeeze ~dim:0 (ones_f [ 2; 3 ]))));
      test "unsqueeze inserts axis" (fun () ->
          equal (list int) [ 2; 1; 3 ]
            (shape (Mv.unsqueeze (ones_f [ 2; 3 ]) 1)));
      test "transpose swaps axes 0 and 1 by default" (fun () ->
          equal (list int) [ 3; 2; 5 ]
            (shape (Mv.transpose (ones_f [ 2; 3; 5 ]))));
      test "transpose named dims" (fun () ->
          equal (list int) [ 2; 5; 3 ]
            (shape (Mv.transpose ~dim0:1 ~dim1:2 (ones_f [ 2; 3; 5 ]))));
      test "flatten collapses" (fun () ->
          equal (list int) [ 24 ] (shape (Mv.flatten (ones_f [ 2; 3; 4 ]))));
      test "flatten range" (fun () ->
          equal (list int) [ 2; 12 ]
            (shape (Mv.flatten ~start_dim:1 (ones_f [ 2; 3; 4 ]))));
      test "unflatten splits" (fun () ->
          equal (list int) [ 2; 2; 3 ]
            (shape (Mv.unflatten (ones_f [ 2; 6 ]) 1 [ 2; 3 ])));
      test "repeat tiles" (fun () ->
          equal (list int) [ 4; 6 ] (shape (Mv.repeat (ones_f [ 2; 3 ]) [ 2; 2 ])));
      test "unfold 1d" (fun () ->
          equal (list int) [ 4; 2 ]
            (shape (Mv.unfold (ones_f [ 8 ]) 0 ~size:2 ~step:2)));
      test "unfold last axis" (fun () ->
          equal (list int) [ 3; 3; 1; 2 ]
            (shape (Mv.unfold (ones_f [ 3; 3; 3 ]) (-1) ~size:2 ~step:3)));
      test "unfold middle axis" (fun () ->
          equal (list int) [ 4; 4; 5; 3 ]
            (shape (Mv.unfold (ones_f [ 4; 10; 5 ]) 1 ~size:3 ~step:2)));
      test "unfold size too large raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Mv.unfold (ones_f [ 4 ]) 0 ~size:5 ~step:1));
      test "split even chunks" (fun () ->
          equal (list (list int)) [ [ 2; 4 ]; [ 2; 4 ]; [ 2; 4 ] ]
            (List.map shape (Mv.split (ones_f [ 6; 4 ]) 2)));
      test "split remainder chunk" (fun () ->
          equal (list (list int)) [ [ 2 ]; [ 2 ]; [ 1 ] ]
            (List.map shape (Mv.split (ones_f [ 5 ]) 2)));
      test "split along dim" (fun () ->
          equal (list (list int)) [ [ 2; 3 ]; [ 2; 3 ] ]
            (List.map shape (Mv.split ~dim:1 (ones_f [ 2; 6 ]) 3)));
    ]

(* Broadcasting and promotion *)

let broadcast_tests =
  group "broadcast"
    [
      test "broadcast_shape aligns from the right" (fun () ->
          equal (list int) [ 3; 4 ] (T.broadcast_shape [ [ 3; 1 ]; [ 1; 4 ] ]));
      test "broadcast_shape with scalar" (fun () ->
          equal (list int) [ 2; 3 ] (T.broadcast_shape [ []; [ 2; 3 ] ]));
      test "broadcast_shape zero propagates" (fun () ->
          equal (list int) [ 0 ] (T.broadcast_shape [ [ 0 ]; [ 1 ] ]));
      test "add broadcasts" (fun () ->
          equal (list int) [ 3; 4 ]
            (shape (El.add (ones_f [ 3; 1 ]) (ones_f [ 1; 4 ]))));
      test "int + float promotes to float" (fun () ->
          let t = El.add (ones_i [ 3 ]) (ones_f [ 3 ]) in
          is_true (is_dtype t D.float32));
      test "scalar operand broadcasts" (fun () ->
          equal (list int) [ 2; 2 ] (shape (El.mul (ones_f [ 2; 2 ]) (T.f 2.0))));
    ]

(* Elementwise structure *)

let elementwise_tests =
  group "elementwise"
    [
      test "sub is add of neg" (fun () ->
          let t = El.sub (ones_f [ 4 ]) (ones_f [ 4 ]) in
          is_true (has_op t Ops.Add);
          is_true (Ops.equal (U.op (src t 1)) Ops.Mul));
      test "div is mul of reciprocal" (fun () ->
          let t = El.div (ones_f [ 4 ]) (ones_f [ 4 ]) in
          is_true (has_op t Ops.Mul);
          is_true (Ops.equal (U.op (src t 1)) Ops.Reciprocal));
      test "neg is mul" (fun () -> is_true (has_op (El.neg (ones_f [ 4 ])) Ops.Mul));
      test "comparison yields bool" (fun () ->
          is_true (is_dtype (El.lt (ones_f [ 3 ]) (ones_f [ 3 ])) D.bool));
      test "eq yields bool" (fun () ->
          is_true (is_dtype (El.eq (ones_f [ 3 ]) (ones_f [ 3 ])) D.bool));
      test "relu is where" (fun () ->
          is_true (has_op (El.relu (ones_f [ 4 ])) Ops.Where));
      test "int floordiv uses floordiv op" (fun () ->
          is_true (has_op (El.floordiv (ones_i [ 4 ]) (ones_i [ 4 ])) Ops.Floordiv));
      test "int mod uses floormod op" (fun () ->
          is_true (has_op (El.mod_ (ones_i [ 4 ]) (ones_i [ 4 ])) Ops.Floormod));
      test "float div is float" (fun () ->
          is_true (is_dtype (El.div (ones_i [ 4 ]) (ones_i [ 4 ])) D.float32));
      test "sqrt promotes int to float" (fun () ->
          is_true (is_dtype (El.sqrt (ones_i [ 4 ])) D.float32));
      test "where shape and dtype from branches" (fun () ->
          let cond = El.lt (ones_f [ 3 ]) (ones_f [ 3 ]) in
          let t = El.where cond (ones_f [ 3 ]) (ones_f [ 3 ]) in
          equal (list int) [ 3 ] (shape t);
          is_true (is_dtype t D.float32));
      test "where broadcasts branches" (fun () ->
          let cond = El.lt (ones_f [ 3; 1 ]) (ones_f [ 3; 1 ]) in
          equal (list int) [ 3; 4 ]
            (shape (El.where cond (ones_f [ 3; 1 ]) (ones_f [ 1; 4 ]))));
    ]

(* Reduce *)

let reduce_tests =
  group "reduce"
    [
      test "sum all reduces to scalar" (fun () ->
          equal (list int) [] (shape (Rd.sum (ones_f [ 2; 3 ]))));
      test "sum axis removes axis" (fun () ->
          equal (list int) [ 2; 4 ]
            (shape (Rd.sum ~axis:[ 1 ] (ones_f [ 2; 3; 4 ]))));
      test "sum keepdim keeps axis" (fun () ->
          equal (list int) [ 2; 1; 4 ]
            (shape (Rd.sum ~axis:[ 1 ] ~keepdim:true (ones_f [ 2; 3; 4 ]))));
      test "sum negative axis" (fun () ->
          equal (list int) [ 2; 3 ]
            (shape (Rd.sum ~axis:[ -1 ] (ones_f [ 2; 3; 4 ]))));
      test "sum float32 stays float32" (fun () ->
          is_true (is_dtype (Rd.sum (ones_f [ 4 ])) D.float32));
      test "sum int32 stays int32" (fun () ->
          is_true (is_dtype (Rd.sum (ones_i [ 4 ])) D.int32));
      test "sum int8 widens to int32" (fun () ->
          is_true
            (is_dtype (Rd.sum (Cr.ones ~dtype:D.Val.int8 [ 4 ])) D.int32));
      test "max reduces" (fun () ->
          equal (list int) [ 2 ] (shape (Rd.max ~axis:[ 1 ] (ones_f [ 2; 3 ]))));
      test "min reduces" (fun () ->
          equal (list int) [ 2 ] (shape (Rd.min ~axis:[ 1 ] (ones_f [ 2; 3 ]))));
      test "any yields bool" (fun () ->
          is_true (is_dtype (Rd.any (ones_i [ 4 ])) D.bool));
      test "reduce size-1 axis is a no-op reduce" (fun () ->
          (* reducing an axis of size 1 removes it without a Reduce node *)
          let t = Rd.sum ~axis:[ 1 ] (ones_f [ 2; 1; 3 ]) in
          equal (list int) [ 2; 3 ] (shape t));
    ]

(* dtype ops *)

let dtype_tests =
  group "dtype"
    [
      test "cast changes dtype" (fun () ->
          is_true (is_dtype (Dt.int (ones_f [ 3 ])) D.int32));
      test "cast to same dtype is identity" (fun () ->
          let t = ones_f [ 3 ] in
          is_true (U.equal (T.uop (Dt.float t)) (T.uop t)));
      test "is_floating_point" (fun () ->
          is_true (Dt.is_floating_point (ones_f [ 3 ]));
          is_true (not (Dt.is_floating_point (ones_i [ 3 ]))));
    ]

(* Composed ops *)

let op_tests =
  group "op"
    [
      test "mean removes axis" (fun () ->
          equal (list int) [ 2; 4 ] (shape (Op.mean ~axis:[ 1 ] (ones_f [ 2; 3; 4 ]))));
      test "mean of int is float" (fun () ->
          is_true (is_dtype (Op.mean (ones_i [ 4 ])) D.float32));
      test "mean all is scalar" (fun () ->
          equal (list int) [] (shape (Op.mean (ones_f [ 2; 3 ]))));
      test "var reduces axis" (fun () ->
          equal (list int) [ 2 ] (shape (Op.var ~axis:[ 1 ] (ones_f [ 2; 3 ]))));
      test "std reduces axis" (fun () ->
          equal (list int) [ 2 ] (shape (Op.std ~axis:[ 1 ] (ones_f [ 2; 3 ]))));
      test "cat dim 0" (fun () ->
          equal (list int) [ 6; 3 ]
            (shape (Op.cat ~dim:0 (ones_f [ 2; 3 ]) [ ones_f [ 4; 3 ] ])));
      test "cat dim 1" (fun () ->
          equal (list int) [ 2; 8 ]
            (shape (Op.cat ~dim:1 (ones_f [ 2; 3 ]) [ ones_f [ 2; 5 ] ])));
      test "stack adds axis" (fun () ->
          equal (list int) [ 2; 2; 3 ]
            (shape (Op.stack ~dim:1 (ones_f [ 2; 3 ]) [ ones_f [ 2; 3 ] ])));
      test "dot 2d x 2d" (fun () ->
          equal (list int) [ 2; 4 ] (shape (Op.dot (ones_f [ 2; 3 ]) (ones_f [ 3; 4 ]))));
      test "dot 1d x 1d is scalar" (fun () ->
          equal (list int) [] (shape (Op.dot (ones_f [ 3 ]) (ones_f [ 3 ]))));
      test "dot batched" (fun () ->
          equal (list int) [ 5; 2; 4 ]
            (shape (Op.dot (ones_f [ 5; 2; 3 ]) (ones_f [ 3; 4 ]))));
      test "matmul" (fun () ->
          equal (list int) [ 2; 4 ] (shape (Op.matmul (ones_f [ 2; 3 ]) (ones_f [ 3; 4 ]))));
      test "dot int stays int" (fun () ->
          is_true (is_dtype (Op.dot (ones_i [ 2; 3 ]) (ones_i [ 3; 4 ])) D.int32));
      test "dot mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.dot (ones_f [ 2; 3 ]) (ones_f [ 4; 5 ])));
    ]

(* Pooling / windowing *)

let pool_tests =
  group "pool"
    [
      test "asymmetric window shape matches tinygrad" (fun () ->
          (* 1x1x6x7 input, 2x3 kernel, stride 2, dilation 2 -> (b,c,o0,o1,k0,k1) *)
          let t = ones_f [ 1; 1; 6; 7 ] in
          let p = Mv.pool t ~k:[ 2; 3 ] ~stride:[ 2; 2 ] ~dilation:[ 2; 2 ] () in
          equal (list int) [ 1; 1; 2; 2; 2; 3 ] (shape p));
      test "default stride and dilation" (fun () ->
          let p = Mv.pool (ones_f [ 1; 1; 4; 4 ]) ~k:[ 2; 2 ] () in
          equal (list int) [ 1; 1; 3; 3; 2; 2 ] (shape p));
      test "stride equal to kernel is non-overlapping" (fun () ->
          let p = Mv.pool (ones_f [ 1; 1; 4; 4 ]) ~k:[ 2; 2 ] ~stride:[ 2; 2 ] () in
          equal (list int) [ 1; 1; 2; 2; 2; 2 ] (shape p));
      test "scalar stride broadcasts" (fun () ->
          let p = Mv.pool (ones_f [ 1; 1; 4; 4 ]) ~k:[ 2; 2 ] ~stride:[ 2 ] () in
          equal (list int) [ 1; 1; 2; 2; 2; 2 ] (shape p));
      test "kernel larger than input raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Mv.pool (ones_f [ 1; 1; 2; 2 ]) ~k:[ 3; 3 ] ()));
      test "shrink_to keeps None axes" (fun () ->
          equal (list int) [ 2; 2 ]
            (shape (Mv.shrink_to (ones_f [ 2; 3 ]) [ None; Some 2 ])));
      test "pad_to grows to target" (fun () ->
          equal (list int) [ 2; 5 ]
            (shape (Mv.pad_to (ones_f [ 2; 3 ]) [ None; Some 5 ])));
    ]

(* Scans and ranges *)

let scan_tests =
  group "scan"
    [
      test "cumsum preserves shape" (fun () ->
          equal (list int) [ 2; 3 ] (shape (Op.cumsum ~axis:1 (ones_f [ 2; 3 ]))));
      test "cumsum axis 0" (fun () ->
          equal (list int) [ 4 ] (shape (Op.cumsum (ones_f [ 4 ]))));
      test "cumprod preserves shape" (fun () ->
          equal (list int) [ 2; 3 ] (shape (Op.cumprod ~axis:0 (ones_f [ 2; 3 ]))));
      test "arange length" (fun () ->
          equal (list int) [ 5 ] (shape (Op.arange 5)));
      test "arange start stop step" (fun () ->
          equal (list int) [ 3 ] (shape (Op.arange ~stop:10 ~step:2 5)));
      test "arange negative step" (fun () ->
          equal (list int) [ 5 ] (shape (Op.arange ~stop:0 ~step:(-1) 5)));
      test "arange empty" (fun () ->
          equal (list int) [ 0 ] (shape (Op.arange ~stop:0 5)));
      test "arange dtype default int" (fun () ->
          is_true (is_dtype (Op.arange 5) D.int32));
      test "pad_constant grows" (fun () ->
          equal (list int) [ 6 ]
            (shape (Op.pad_constant (ones_f [ 3 ]) [ Some (1, 2) ] (T.Sfloat 0.0))));
      test "pad_constant negative shrinks" (fun () ->
          equal (list int) [ 3 ]
            (shape (Op.pad_constant (ones_f [ 4 ]) [ Some (-1, 0) ] (T.Sint 0))));
      test "cummax returns values and indices shapes" (fun () ->
          let v, i = Op.cummax (ones_f [ 7 ]) in
          equal (list int) [ 7 ] (shape v);
          equal (list int) [ 7 ] (shape i));
      test "cummax indices are int" (fun () ->
          let _, i = Op.cummax (ones_f [ 5 ]) in
          is_true (is_dtype i D.int32));
      test "cummax 2d axis" (fun () ->
          let v, _ = Op.cummax ~axis:1 (ones_f [ 2; 3 ]) in
          equal (list int) [ 2; 3 ] (shape v));
      test "cummin returns values and indices shapes" (fun () ->
          let v, i = Op.cummin (ones_f [ 7 ]) in
          equal (list int) [ 7 ] (shape v);
          equal (list int) [ 7 ] (shape i));
      test "triu preserves shape" (fun () ->
          equal (list int) [ 3; 3 ] (shape (Op.triu (ones_f [ 3; 3 ]))));
      test "tril preserves shape" (fun () ->
          equal (list int) [ 3; 3 ] (shape (Op.tril (ones_f [ 3; 3 ]))));
      test "triu is where" (fun () ->
          is_true (has_op (Op.triu (ones_f [ 3; 3 ])) Ops.Where));
      test "gather takes index shape" (fun () ->
          equal (list int) [ 2; 2 ]
            (shape (Op.gather (ones_f [ 2; 2 ]) ~dim:1 (ones_i [ 2; 2 ]))));
      test "gather smaller index along non-dim axis" (fun () ->
          equal (list int) [ 3; 5 ]
            (shape
               (Op.gather (ones_f [ 4; 5 ]) ~dim:0
                  (Cr.zeros ~dtype:D.Val.int32 [ 3; 5 ]))));
      test "gather preserves dtype" (fun () ->
          is_true
            (is_dtype (Op.gather (ones_i [ 2; 2 ]) ~dim:1 (ones_i [ 2; 2 ])) D.int32));
      test "gather ndim mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.gather (ones_f [ 2; 2 ]) ~dim:0 (ones_i [ 2 ])));
      test "one_hot adds class axis" (fun () ->
          equal (list int) [ 3; 4 ] (shape (Op.one_hot (Op.arange 3) 4)));
      test "argmax over all is scalar int" (fun () ->
          let t = Op.argmax (ones_f [ 2; 3 ]) in
          equal (list int) [] (shape t);
          is_true (is_dtype t D.int32));
      test "argmax axis removes it" (fun () ->
          equal (list int) [ 2; 4 ] (shape (Op.argmax ~axis:1 (ones_f [ 2; 3; 4 ]))));
      test "argmax keepdim" (fun () ->
          equal (list int) [ 2; 1; 4 ]
            (shape (Op.argmax ~axis:1 ~keepdim:true (ones_f [ 2; 3; 4 ]))));
      test "argmin axis" (fun () ->
          equal (list int) [ 2; 3 ] (shape (Op.argmin ~axis:(-1) (ones_f [ 2; 3; 4 ]))));
    ]

(* Convolution and pooling wrappers *)

let conv_tests =
  group "conv"
    [
      test "conv basic" (fun () ->
          equal (list int) [ 1; 1; 2; 2 ]
            (shape (Op.conv2d (ones_f [ 1; 1; 3; 3 ]) (ones_f [ 1; 1; 2; 2 ]))));
      test "conv multichannel" (fun () ->
          equal (list int) [ 2; 5; 6; 6 ]
            (shape (Op.conv2d (ones_f [ 2; 3; 8; 8 ]) (ones_f [ 5; 3; 3; 3 ]))));
      test "conv stride and padding" (fun () ->
          equal (list int) [ 2; 5; 4; 4 ]
            (shape
               (Op.conv2d ~stride:[ 2 ] ~padding:[ 1 ] (ones_f [ 2; 3; 8; 8 ])
                  (ones_f [ 5; 3; 3; 3 ]))));
      test "conv groups" (fun () ->
          equal (list int) [ 1; 4; 6; 6 ]
            (shape
               (Op.conv2d ~groups:2 (ones_f [ 1; 4; 8; 8 ])
                  (ones_f [ 4; 2; 3; 3 ]))));
      test "conv asymmetric grouped stride<>dilation" (fun () ->
          (* groups=2, kernel 2x3, stride (2,1), dilation (1,2); traced against
             tinygrad: pool (1,4,3,5,2,3) -> ... -> final (1,6,3,5) *)
          equal (list int) [ 1; 6; 3; 5 ]
            (shape
               (Op.conv2d ~groups:2 ~stride:[ 2; 1 ] ~dilation:[ 1; 2 ]
                  (ones_f [ 1; 4; 7; 9 ]) (ones_f [ 6; 2; 2; 3 ]))));
      test "conv bias" (fun () ->
          equal (list int) [ 1; 2; 2; 2 ]
            (shape
               (Op.conv2d ~bias:(ones_f [ 2 ]) (ones_f [ 1; 1; 3; 3 ])
                  (ones_f [ 2; 1; 2; 2 ]))));
      test "conv shape mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.conv2d (ones_f [ 1; 3; 8; 8 ]) (ones_f [ 5; 4; 3; 3 ])));
      test "avg_pool2d default" (fun () ->
          equal (list int) [ 1; 1; 2; 2 ] (shape (Op.avg_pool2d (ones_f [ 1; 1; 4; 4 ]))));
      test "max_pool2d default" (fun () ->
          equal (list int) [ 1; 1; 2; 2 ] (shape (Op.max_pool2d (ones_f [ 1; 1; 4; 4 ]))));
      test "avg_pool2d stride and padding" (fun () ->
          equal (list int) [ 1; 3; 4; 4 ]
            (shape
               (Op.avg_pool2d ~kernel_size:[ 3; 3 ] ~stride:[ 2 ] ~padding:[ 1 ]
                  (ones_f [ 1; 3; 8; 8 ]))));
      test "max_pool2d on int raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.max_pool2d (ones_i [ 1; 1; 4; 4 ])));
    ]

(* Elementwise remainder *)

let elementwise2_tests =
  group "elementwise2"
    [
      test "pow float stays float" (fun () ->
          is_true (has_op (El.pow (ones_f [ 4 ]) (T.f 2.0)) Ops.Pow);
          is_true (is_dtype (El.pow (ones_f [ 4 ]) (T.f 2.0)) D.float32));
      test "pow int base float exp rounds to int" (fun () ->
          let t = El.pow (ones_i [ 4 ]) (T.f 2.0) in
          is_true (is_dtype t D.int32));
      test "pow int base int exp is int" (fun () ->
          is_true (is_dtype (El.pow (ones_i [ 4 ]) (T.i 2)) D.int32));
      test "cdiv int uses cdiv op" (fun () ->
          is_true (has_op (El.cdiv (ones_i [ 4 ]) (ones_i [ 4 ])) Ops.Cdiv));
      test "cdiv float truncates quotient" (fun () ->
          is_true (has_op (El.cdiv (ones_f [ 4 ]) (ones_f [ 4 ])) Ops.Trunc));
      test "fmod int uses cmod op" (fun () ->
          is_true (has_op (El.fmod (ones_i [ 4 ]) (ones_i [ 4 ])) Ops.Cmod));
      test "fmod float stays float" (fun () ->
          is_true (is_dtype (El.fmod (ones_f [ 4 ]) (ones_f [ 4 ])) D.float32));
      test "lshift uses shl op" (fun () ->
          is_true (has_op (El.lshift (ones_i [ 4 ]) (T.i 2)) Ops.Shl));
      test "rshift uses shr op" (fun () ->
          is_true (has_op (El.rshift (ones_i [ 4 ]) (T.i 2)) Ops.Shr));
      test "round preserves float" (fun () ->
          let t = El.round (ones_f [ 4 ]) in
          equal (list int) [ 4 ] (shape t);
          is_true (is_dtype t D.float32));
      test "clamp both bounds" (fun () ->
          equal (list int) [ 4 ]
            (shape (El.clamp ~min:(T.f 0.0) ~max:(T.f 1.0) (ones_f [ 4 ]))));
      test "clamp min only is a where" (fun () ->
          is_true (has_op (El.clamp ~min:(T.f 0.0) (ones_f [ 4 ])) Ops.Where));
      test "clamp no bounds raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> El.clamp (ones_f [ 4 ])));
      test "clip aliases clamp" (fun () ->
          equal (list int) [ 4 ] (shape (El.clip ~max:(T.f 1.0) (ones_f [ 4 ]))));
      test "copysign broadcasts and keeps float" (fun () ->
          let t = El.copysign (ones_f [ 3; 1 ]) (ones_f [ 1; 4 ]) in
          equal (list int) [ 3; 4 ] (shape t);
          is_true (is_dtype t D.float32));
      test "logaddexp preserves shape" (fun () ->
          equal (list int) [ 4 ] (shape (El.logaddexp (ones_f [ 4 ]) (ones_f [ 4 ]))));
      test "lerp interpolates" (fun () ->
          equal (list int) [ 4 ]
            (shape (El.lerp (ones_f [ 4 ]) (ones_f [ 4 ]) (T.f 0.5))));
      test "isnan is bool" (fun () ->
          is_true (is_dtype (El.isnan (ones_f [ 4 ])) D.bool));
      test "isinf is bool" (fun () ->
          is_true (is_dtype (El.isinf (ones_f [ 4 ])) D.bool));
      test "isfinite is bool" (fun () ->
          is_true (is_dtype (El.isfinite (ones_f [ 4 ])) D.bool));
      test "isclose is bool" (fun () ->
          is_true (is_dtype (El.isclose (ones_f [ 4 ]) (ones_f [ 4 ])) D.bool));
      test "erf preserves float" (fun () ->
          is_true (is_dtype (El.erf (ones_f [ 4 ])) D.float32));
      test "log10 promotes int to float" (fun () ->
          is_true (is_dtype (El.log10 (ones_i [ 4 ])) D.float32));
      test "trig inverses preserve shape" (fun () ->
          equal (list int) [ 4 ] (shape (El.asin (ones_f [ 4 ])));
          equal (list int) [ 4 ] (shape (El.acos (ones_f [ 4 ])));
          equal (list int) [ 4 ] (shape (El.atan (ones_f [ 4 ])));
          equal (list int) [ 4 ] (shape (El.tan (ones_f [ 4 ]))));
      test "hyperbolic preserve shape" (fun () ->
          equal (list int) [ 4 ] (shape (El.sinh (ones_f [ 4 ])));
          equal (list int) [ 4 ] (shape (El.cosh (ones_f [ 4 ])));
          equal (list int) [ 4 ] (shape (El.atanh (ones_f [ 4 ])));
          equal (list int) [ 4 ] (shape (El.asinh (ones_f [ 4 ])));
          equal (list int) [ 4 ] (shape (El.acosh (ones_f [ 4 ]))));
      test "activation zoo preserves shape and float" (fun () ->
          List.iter
            (fun f -> is_true (is_dtype (f (ones_f [ 4 ])) D.float32))
            [
              El.relu6; El.hardswish;
              (fun t -> El.hardsigmoid t);
              (fun t -> El.hardtanh t);
              (fun t -> El.leaky_relu t);
              El.quick_gelu; El.gelu; El.swish; El.silu;
              (fun t -> El.elu t);
              (fun t -> El.celu t);
              (fun t -> El.selu t);
              (fun t -> El.softplus t);
              El.mish; El.logsigmoid; El.softsign;
            ]);
    ]

(* Indexing *)

let index_tests =
  let base () = ones_i [ 2; 3; 4 ] in
  let ar = Op.arange in
  group "index"
    [
      test "int index drops axis" (fun () ->
          equal (list int) [ 3; 4 ] (shape (Op.getitem (base ()) [ Mv.I 1 ])));
      test "two int indices" (fun () ->
          equal (list int) [ 4 ] (shape (Op.getitem (base ()) [ Mv.I 1; Mv.I 2 ])));
      test "full int index is scalar" (fun () ->
          equal (list int) []
            (shape (Op.getitem (base ()) [ Mv.I 1; Mv.I 2; Mv.I 3 ])));
      test "negative int index" (fun () ->
          equal (list int) [ 3; 4 ] (shape (Op.getitem (base ()) [ Mv.I (-1) ])));
      test "int out of bounds raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.getitem (base ()) [ Mv.I 2 ]));
      test "slice and strided slice" (fun () ->
          equal (list int) [ 2; 2; 4 ]
            (shape
               (Op.getitem (base ())
                  [ Mv.R (Some 0, Some 2, None); Mv.R (None, None, Some 2) ])));
      test "negative step reverses without changing shape" (fun () ->
          equal (list int) [ 2; 3; 4 ]
            (shape (Op.getitem (base ()) [ Mv.All; Mv.R (None, None, Some (-1)) ])));
      test "slice keeps single-width axis" (fun () ->
          equal (list int) [ 2; 1; 4 ]
            (shape (Op.getitem (base ()) [ Mv.All; Mv.R (Some 1, Some 2, None) ])));
      test "slice clamps out-of-range stop" (fun () ->
          equal (list int) [ 2; 3; 4 ]
            (shape (Op.getitem (base ()) [ Mv.R (Some 0, Some 100, None) ])));
      test "strided on every axis" (fun () ->
          equal (list int) [ 1; 2; 2 ]
            (shape
               (Op.getitem (base ())
                  [
                    Mv.R (None, None, Some 2);
                    Mv.R (None, None, Some 2);
                    Mv.R (None, None, Some 2);
                  ])));
      test "mixed int slice reversed" (fun () ->
          equal (list int) [ 3; 4 ]
            (shape
               (Op.getitem (base ())
                  [ Mv.I 1; Mv.All; Mv.R (None, None, Some (-1)) ])));
      test "ellipsis fills leading axes" (fun () ->
          equal (list int) [ 2; 3 ]
            (shape (Op.getitem (base ()) [ Mv.Ellipsis; Mv.I 0 ])));
      test "ellipsis then new axis" (fun () ->
          equal (list int) [ 2; 3; 4; 1 ]
            (shape (Op.getitem (base ()) [ Mv.Ellipsis; Mv.New ])));
      test "leading new axis" (fun () ->
          equal (list int) [ 1; 2; 3; 4 ]
            (shape (Op.getitem (base ()) [ Mv.New ])));
      test "interior new axis" (fun () ->
          equal (list int) [ 2; 1; 3; 4 ]
            (shape (Op.getitem (base ()) [ Mv.All; Mv.New ])));
      test "two ellipses raise" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.getitem (base ()) [ Mv.Ellipsis; Mv.Ellipsis; Mv.I 0 ]));
      test "too many indices raise" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.getitem (base ()) [ Mv.I 0; Mv.I 0; Mv.I 0; Mv.I 0 ]));
      (* Advanced (tensor) indexing *)
      test "single tensor index keeps trailing axes" (fun () ->
          equal (list int) [ 2; 3; 4 ]
            (shape (Op.getitem (base ()) [ Mv.T (ar 2) ])));
      test "2d tensor index adds axes" (fun () ->
          equal (list int) [ 2; 1; 3; 4 ]
            (shape
               (Op.getitem (base ()) [ Mv.T (Mv.reshape (ar 2) [ 2; 1 ]) ])));
      test "tensor index after slice" (fun () ->
          equal (list int) [ 2; 3; 4 ]
            (shape (Op.getitem (base ()) [ Mv.All; Mv.T (ar 3) ])));
      test "two consecutive tensor indices broadcast" (fun () ->
          equal (list int) [ 2; 4 ]
            (shape (Op.getitem (base ()) [ Mv.T (ar 2); Mv.T (ar 2) ])));
      test "three consecutive tensor indices" (fun () ->
          equal (list int) [ 2 ]
            (shape
               (Op.getitem (base ())
                  [ Mv.T (ar 2); Mv.T (ar 2); Mv.T (ar 2) ])));
      test "non-consecutive tensor indices" (fun () ->
          equal (list int) [ 2; 3 ]
            (shape (Op.getitem (base ()) [ Mv.T (ar 2); Mv.All; Mv.T (ar 2) ])));
      test "consecutive tensor indices with leading axis" (fun () ->
          equal (list int) [ 2; 3 ]
            (shape (Op.getitem (base ()) [ Mv.All; Mv.T (ar 3); Mv.T (ar 3) ])));
      test "broadcast tensor indices" (fun () ->
          equal (list int) [ 2; 3; 4 ]
            (shape
               (Op.getitem (base ())
                  [
                    Mv.T (Mv.reshape (ar 2) [ 2; 1 ]);
                    Mv.T (Mv.reshape (ar 3) [ 1; 3 ]);
                  ])));
      test "negative tensor index" (fun () ->
          equal (list int) [ 2; 3; 4 ]
            (shape
               (Op.getitem (base ()) [ Mv.T (El.sub (ar 2) (T.i 2)) ])));
      test "advanced getitem preserves dtype" (fun () ->
          is_true (is_dtype (Op.getitem (base ()) [ Mv.T (ar 2) ]) D.int32));
      test "float tensor index raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.getitem (ones_f [ 3 ]) [ Mv.T (Cr.ones [ 2 ]) ]));
    ]

(* Log-space reductions *)

let logspace_tests =
  group "logspace"
    [
      test "softmax preserves shape and dtype" (fun () ->
          let t = Op.softmax (ones_f [ 2; 3; 4 ]) in
          equal (list int) [ 2; 3; 4 ] (shape t);
          is_true (is_dtype t D.float32));
      test "softmax axis 0" (fun () ->
          equal (list int) [ 2; 3; 4 ] (shape (Op.softmax ~axis:0 (ones_f [ 2; 3; 4 ]))));
      test "log_softmax preserves shape" (fun () ->
          equal (list int) [ 2; 3; 4 ] (shape (Op.log_softmax (ones_f [ 2; 3; 4 ]))));
      test "logsumexp all is scalar" (fun () ->
          equal (list int) [] (shape (Op.logsumexp (ones_f [ 2; 3; 4 ]))));
      test "logsumexp axis removes it" (fun () ->
          equal (list int) [ 2; 4 ]
            (shape (Op.logsumexp ~axis:1 (ones_f [ 2; 3; 4 ]))));
      test "logsumexp keepdim" (fun () ->
          equal (list int) [ 2; 1; 4 ]
            (shape (Op.logsumexp ~axis:1 ~keepdim:true (ones_f [ 2; 3; 4 ]))));
      test "logcumsumexp preserves shape" (fun () ->
          equal (list int) [ 2; 3; 4 ]
            (shape (Op.logcumsumexp ~axis:1 (ones_f [ 2; 3; 4 ]))));
      test "logcumsumexp default axis" (fun () ->
          equal (list int) [ 2; 3 ] (shape (Op.logcumsumexp (ones_f [ 2; 3 ]))));
    ]

(* Creation remainder *)

let creation2_tests =
  group "creation2"
    [
      test "eye square" (fun () ->
          let t = Op.eye 3 in
          equal (list int) [ 3; 3 ] (shape t);
          is_true (is_dtype t D.float32));
      test "eye rectangular" (fun () ->
          equal (list int) [ 2; 4 ] (shape (Op.eye ~m:4 2)));
      test "eye int dtype" (fun () ->
          is_true (is_dtype (Op.eye ~dtype:D.Val.int32 3) D.int32));
      test "eye negative raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.eye (-1)));
      test "linspace length" (fun () ->
          let t = Op.linspace 0.0 10.0 5 in
          equal (list int) [ 5 ] (shape t);
          is_true (is_dtype t D.float32));
      test "linspace single" (fun () ->
          equal (list int) [ 1 ] (shape (Op.linspace 0.0 10.0 1)));
      test "linspace empty" (fun () ->
          equal (list int) [ 0 ] (shape (Op.linspace 0.0 10.0 0)));
      test "linspace int dtype" (fun () ->
          is_true (is_dtype (Op.linspace ~dtype:D.Val.int32 0.0 10.0 5) D.int32));
      test "linspace negative steps raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.linspace 0.0 10.0 (-1)));
    ]

(* Padding modes and masked fill *)

let pad_mode_tests =
  let base () = ones_f [ 1; 1; 3; 3 ] in
  group "pad_modes"
    [
      test "constant mode matches pad_constant" (fun () ->
          equal (list int) [ 1; 1; 4; 6 ]
            (shape (Op.pad (base ()) [ None; None; Some (0, 1); Some (1, 2) ])));
      test "reflect grows dims" (fun () ->
          equal (list int) [ 1; 1; 4; 6 ]
            (shape
               (Op.pad ~mode:Op.Reflect (base ())
                  [ None; None; Some (0, 1); Some (1, 2) ])));
      test "replicate grows dims" (fun () ->
          equal (list int) [ 1; 1; 4; 6 ]
            (shape
               (Op.pad ~mode:Op.Replicate (base ())
                  [ None; None; Some (0, 1); Some (1, 2) ])));
      test "circular grows dims" (fun () ->
          equal (list int) [ 1; 1; 5; 5 ]
            (shape
               (Op.pad ~mode:Op.Circular (base ())
                  [ None; None; Some (1, 1); Some (1, 1) ])));
      test "reflect both sides" (fun () ->
          equal (list int) [ 1; 1; 5; 5 ]
            (shape
               (Op.pad ~mode:Op.Reflect (base ())
                  [ None; None; Some (1, 1); Some (2, 0) ])));
      test "replicate negative shrinks" (fun () ->
          equal (list int) [ 1; 1; 3; 3 ]
            (shape
               (Op.pad ~mode:Op.Replicate (base ())
                  [ None; None; Some (0, 0); Some (1, -1) ])));
      test "reflect too large raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Op.pad ~mode:Op.Reflect (base ())
                [ None; None; Some (0, 0); Some (3, 0) ]));
      test "circular wrap too large raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Op.pad ~mode:Op.Circular (base ())
                [ None; None; Some (0, 0); Some (4, 0) ]));
      test "masked_fill preserves shape" (fun () ->
          equal (list int) [ 4 ]
            (shape (El.masked_fill (ones_f [ 4 ]) (El.gt (ones_f [ 4 ]) (T.f 0.0)) (T.f (-1.0)))));
    ]

(* Shape is memoised per node, so a deeply shared (diamond) graph computes its
   shape in linear time. Without the memo this doubles at every level and this
   test would not terminate. *)
let shape_memo_tests =
  group "shape_memo"
    [
      test "deep diamond shape is cheap" (fun () ->
          let x = ref (ones_f [ 4 ]) in
          for _ = 1 to 40 do
            x := El.add !x !x
          done;
          equal (list int) [ 4 ] (shape !x));
    ]

(* Scatter *)

let scatter_tests =
  let zi sh = Cr.zeros ~dtype:D.Val.int32 sh in
  group "scatter"
    [
      test "scatter preserves self shape" (fun () ->
          equal (list int) [ 3; 5 ]
            (shape (Op.scatter (Cr.zeros [ 3; 5 ]) ~dim:0 (zi [ 1; 4 ]) (ones_f [ 2; 5 ]))));
      test "scatter keeps dtype" (fun () ->
          is_true
            (is_dtype
               (Op.scatter (ones_i [ 2; 3 ]) ~dim:1 (zi [ 2; 2 ]) (ones_i [ 2; 3 ]))
               D.int32));
      test "scatter_reduce sum shape" (fun () ->
          equal (list int) [ 1; 5 ]
            (shape
               (Op.scatter_reduce (ones_f [ 1; 5 ]) ~dim:0 (zi [ 2; 5 ])
                  (ones_f [ 2; 5 ]) ~reduce:`Sum ())));
      test "scatter rank mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.scatter (Cr.zeros [ 3; 5 ]) ~dim:0 (zi [ 4 ]) (ones_f [ 2; 5 ])));
      test "scatter dtype mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.scatter (Cr.zeros [ 3; 5 ]) ~dim:0 (zi [ 1; 4 ]) (ones_i [ 2; 5 ])));
    ]

(* Boolean selection *)

let select_tests =
  let boolmask sh = El.gt (ones_f sh) (T.f 0.0) in
  group "select"
    [
      test "masked_select fixed size" (fun () ->
          let t = Op.masked_select (ones_f [ 3; 3 ]) (boolmask [ 3; 3 ]) ~size:5 in
          equal (list int) [ 5 ] (shape t);
          is_true (is_dtype t D.float32));
      test "masked_select broadcasts mask" (fun () ->
          equal (list int) [ 4 ]
            (shape (Op.masked_select (ones_f [ 2; 3 ]) (boolmask [ 3 ]) ~size:4)));
      test "masked_select non-bool mask raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.masked_select (ones_f [ 3 ]) (ones_f [ 3 ]) ~size:2));
      test "nonzero shape is size by rank" (fun () ->
          let t = Op.nonzero (ones_i [ 2; 3 ]) ~size:4 in
          equal (list int) [ 4; 2 ] (shape t);
          is_true (is_dtype t D.int32));
      test "nonzero 1d" (fun () ->
          equal (list int) [ 3; 1 ] (shape (Op.nonzero (ones_i [ 5 ]) ~size:3)));
    ]

(* Sorting *)

let sort_tests =
  group "sort"
    [
      test "sort returns values and indices shapes" (fun () ->
          let v, i = Op.sort (ones_f [ 5 ]) in
          equal (list int) [ 5 ] (shape v);
          equal (list int) [ 5 ] (shape i));
      test "sort indices are int" (fun () ->
          let _, i = Op.sort (ones_f [ 5 ]) in
          is_true (is_dtype i D.int32));
      test "sort values keep dtype" (fun () ->
          let v, _ = Op.sort (ones_i [ 5 ]) in
          is_true (is_dtype v D.int32));
      test "sort 3d along axis" (fun () ->
          let v, i = Op.sort ~dim:1 (ones_f [ 2; 3; 4 ]) in
          equal (list int) [ 2; 3; 4 ] (shape v);
          equal (list int) [ 2; 3; 4 ] (shape i));
      test "sort singleton axis is trivial" (fun () ->
          let v, i = Op.sort (ones_f [ 1 ]) in
          equal (list int) [ 1 ] (shape v);
          is_true (is_dtype i D.int32));
      test "argsort shape" (fun () ->
          equal (list int) [ 2; 3 ] (shape (Op.argsort (ones_f [ 2; 3 ]))));
      test "topk trims the axis" (fun () ->
          let v, i = Op.topk (ones_f [ 6 ]) 2 in
          equal (list int) [ 2 ] (shape v);
          equal (list int) [ 2 ] (shape i));
      test "topk k too large raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.topk (ones_f [ 3 ]) 5));
    ]

let () =
  run "Tolk_frontend"
    [
      creation_tests;
      creation2_tests;
      shape_memo_tests;
      scatter_tests;
      select_tests;
      sort_tests;
      index_tests;
      logspace_tests;
      pad_mode_tests;
      movement_tests;
      broadcast_tests;
      elementwise_tests;
      reduce_tests;
      dtype_tests;
      elementwise2_tests;
      op_tests;
      pool_tests;
      scan_tests;
      conv_tests;
    ]
