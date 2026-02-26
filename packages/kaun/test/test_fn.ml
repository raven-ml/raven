(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Fn = Kaun.Fn

let flatten_f32 t =
  Rune.to_array (Rune.reshape [| -1 |] (Rune.cast Rune.float32 t))

let check_shape msg expected t = equal ~msg (array int) expected (Rune.shape t)

let check_values msg expected t =
  let actual = flatten_f32 t in
  let n = Array.length expected in
  if Array.length actual <> n then
    failf "%s: expected %d elements, got %d" msg n (Array.length actual);
  for i = 0 to n - 1 do
    equal
      ~msg:(Printf.sprintf "%s[%d]" msg i)
      (float 1e-4) expected.(i) actual.(i)
  done

(* conv1d *)

let test_conv1d_basic () =
  let x = Rune.create Rune.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let w = Rune.create Rune.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in
  let result = Fn.conv1d x w in
  check_shape "conv1d basic shape" [| 1; 1; 3 |] result;
  check_values "conv1d basic" [| 6.; 9.; 12. |] result

let test_conv1d_same_padding () =
  let x = Rune.create Rune.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let w = Rune.create Rune.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in
  let result = Fn.conv1d ~padding:`Same x w in
  check_shape "conv1d same shape" [| 1; 1; 5 |] result;
  check_values "conv1d same" [| 3.; 6.; 9.; 12.; 9. |] result

let test_conv1d_stride () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 8 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let w = Rune.create Rune.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in
  let result = Fn.conv1d ~stride:2 x w in
  check_shape "conv1d stride shape" [| 1; 1; 3 |] result;
  check_values "conv1d stride" [| 6.; 12.; 18. |] result

let test_conv1d_dilation () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 7 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7. |]
  in
  let w = Rune.create Rune.float32 [| 1; 1; 3 |] [| 1.; 0.; 1. |] in
  let result = Fn.conv1d ~dilation:2 x w in
  check_shape "conv1d dilation shape" [| 1; 1; 3 |] result;
  (* kernel [1;0;1] with dilation=2 picks (i, i+2, i+4): 1+5=6, 2+6=8, 3+7=10 *)
  check_values "conv1d dilation" [| 6.; 8.; 10. |] result

let test_conv1d_bias () =
  let x = Rune.create Rune.float32 [| 1; 1; 3 |] [| 1.; 2.; 3. |] in
  let w = Rune.create Rune.float32 [| 1; 1; 2 |] [| 1.; 1. |] in
  let bias = Rune.create Rune.float32 [| 1 |] [| 10. |] in
  let result = Fn.conv1d ~bias x w in
  check_values "conv1d bias" [| 13.; 15. |] result

let test_conv1d_groups () =
  let x = Rune.create Rune.float32 [| 1; 4; 4 |] (Array.init 16 float_of_int) in
  let w =
    Rune.create Rune.float32 [| 2; 2; 2 |] [| 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1. |]
  in
  let result = Fn.conv1d ~groups:2 x w in
  check_shape "conv1d groups shape" [| 1; 2; 3 |] result

(* conv2d *)

let test_conv2d_basic () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
  in
  let w = Rune.create Rune.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in
  let result = Fn.conv2d x w in
  check_shape "conv2d basic shape" [| 1; 1; 2; 2 |] result;
  check_values "conv2d basic" [| 45.; 54.; 81.; 90. |] result

let test_conv2d_same_padding () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 3; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let w = Rune.create Rune.float32 [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |] in
  let result = Fn.conv2d ~padding:`Same x w in
  check_shape "conv2d same shape" [| 1; 1; 3; 3 |] result

let test_conv2d_stride () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 5; 5 |] (Array.init 25 float_of_int)
  in
  let w = Rune.create Rune.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in
  let result = Fn.conv2d ~stride:(2, 2) x w in
  check_shape "conv2d stride shape" [| 1; 1; 2; 2 |] result;
  check_values "conv2d stride" [| 54.; 72.; 144.; 162. |] result

let test_conv2d_dilation () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 5; 5 |] (Array.init 25 float_of_int)
  in
  let w =
    Rune.create Rune.float32 [| 1; 1; 3; 3 |]
      [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1. |]
  in
  let result = Fn.conv2d ~dilation:(2, 2) x w in
  check_shape "conv2d dilation shape" [| 1; 1; 1; 1 |] result;
  check_values "conv2d dilation" [| 24. |] result

let test_conv2d_multi_channel () =
  let x =
    Rune.create Rune.float32 [| 1; 3; 4; 4 |] (Array.init 48 float_of_int)
  in
  let w = Rune.create Rune.float32 [| 2; 3; 3; 3 |] (Array.make 54 1.0) in
  let result = Fn.conv2d x w in
  check_shape "conv2d multi-channel shape" [| 1; 2; 2; 2 |] result

let test_conv2d_groups () =
  let x =
    Rune.create Rune.float32 [| 1; 4; 6; 6 |] (Array.init 144 float_of_int)
  in
  let w = Rune.create Rune.float32 [| 4; 2; 2; 2 |] (Array.make 32 1.0) in
  let result = Fn.conv2d ~groups:2 x w in
  check_shape "conv2d groups shape" [| 1; 4; 5; 5 |] result

let test_conv2d_bias () =
  let x = Rune.create Rune.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in
  let w = Rune.create Rune.float32 [| 2; 1; 2; 2 |] (Array.make 8 1.0) in
  let bias = Rune.create Rune.float32 [| 2 |] [| 10.; 20. |] in
  let result = Fn.conv2d ~bias x w in
  check_shape "conv2d bias shape" [| 1; 2; 2; 2 |] result;
  (* Each 2x2 window of ones with all-ones kernel = 4.0, + bias *)
  check_values "conv2d bias" [| 14.; 14.; 14.; 14.; 24.; 24.; 24.; 24. |] result

(* max_pool1d *)

let test_max_pool1d_basic () =
  let x = Rune.create Rune.float32 [| 1; 1; 6 |] [| 1.; 3.; 2.; 5.; 4.; 6. |] in
  let result = Fn.max_pool1d ~kernel_size:2 ~stride:2 x in
  check_shape "max_pool1d shape" [| 1; 1; 3 |] result;
  check_values "max_pool1d" [| 3.; 5.; 6. |] result

let test_max_pool1d_same_padding () =
  let x = Rune.create Rune.float32 [| 1; 1; 5 |] [| 1.; 3.; 2.; 5.; 4. |] in
  let result = Fn.max_pool1d ~kernel_size:3 ~stride:1 ~padding:`Same x in
  check_shape "max_pool1d same shape" [| 1; 1; 5 |] result

(* max_pool2d *)

let test_max_pool2d_basic () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
  in
  let result = Fn.max_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) x in
  check_shape "max_pool2d shape" [| 1; 1; 2; 2 |] result;
  check_values "max_pool2d" [| 5.; 7.; 13.; 15. |] result

let test_max_pool2d_stride1 () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 3; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let result = Fn.max_pool2d ~kernel_size:(2, 2) ~stride:(1, 1) x in
  check_shape "max_pool2d stride1 shape" [| 1; 1; 2; 2 |] result;
  check_values "max_pool2d stride1" [| 5.; 6.; 8.; 9. |] result

(* avg_pool1d *)

let test_avg_pool1d_basic () =
  let x = Rune.create Rune.float32 [| 1; 1; 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Fn.avg_pool1d ~kernel_size:2 ~stride:2 x in
  check_shape "avg_pool1d shape" [| 1; 1; 3 |] result;
  check_values "avg_pool1d" [| 1.5; 3.5; 5.5 |] result

(* avg_pool2d *)

let test_avg_pool2d_basic () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
  in
  let result = Fn.avg_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) x in
  check_shape "avg_pool2d shape" [| 1; 1; 2; 2 |] result;
  check_values "avg_pool2d" [| 2.5; 4.5; 10.5; 12.5 |] result

let test_avg_pool2d_same_padding () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 3; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let result =
    Fn.avg_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) ~padding:`Same x
  in
  check_shape "avg_pool2d same shape" [| 1; 1; 2; 2 |] result

(* Gradient tests *)

let eps = 1e-4

let check_rune ~eps msg expected actual =
  let xs = flatten_f32 expected in
  let ys = flatten_f32 actual in
  let n = Array.length xs in
  if Array.length ys <> n then
    failf "%s: shape mismatch: expected %d elts, got %d" msg n (Array.length ys);
  for i = 0 to n - 1 do
    equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float eps) xs.(i) ys.(i)
  done

let test_grad_conv2d () =
  (* conv2d is correlation (no kernel flip), unlike the old Nx convolve2d *)
  let x =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i + 1)))
  in
  let w = Rune.create Rune.float32 [| 1; 1; 2; 2 |] [| 1.; 0.; 0.; 1. |] in
  (* grad w.r.t. input: sum(conv2d(x, w)) → each input pixel's grad is how many
     output windows include it, weighted by the kernel value at that position.
     For a 2x2 kernel [1,0;0,1] on 4x4 input with Valid padding → 3x3 output.
     JAX: jax.grad(lambda x: jnp.sum(jax.lax.conv(x, w, (1,1), 'VALID')))(x) *)
  let f_x x = Rune.sum (Fn.conv2d x w) in
  let grad_x = Rune.grad f_x x in
  let expected_x =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |]
      [| 1.; 1.; 1.; 0.; 1.; 2.; 2.; 1.; 1.; 2.; 2.; 1.; 0.; 1.; 1.; 1. |]
  in
  check_rune ~eps "conv2d dx" expected_x grad_x;
  (* grad w.r.t. kernel *)
  let f_w w = Rune.sum (Fn.conv2d x w) in
  let grad_w = Rune.grad f_w w in
  (* For correlation: dL/dw[i,j] = sum of x values at positions covered by
     w[i,j] across all output windows. w[0,0] covers x[0..2,0..2], w[0,1] covers
     x[0..2,1..3], etc. *)
  let expected_w =
    Rune.create Rune.float32 [| 1; 1; 2; 2 |] [| 54.; 63.; 90.; 99. |]
  in
  check_rune ~eps "conv2d dw" expected_w grad_w

let test_grad_avg_pool2d () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i + 1)))
  in
  (* Non-overlapping 2x2 avg pool: each output = mean of 4 inputs. grad of
     sum(avg_pool) = 0.25 everywhere (each input contributes to exactly one
     output, scaled by 1/4) *)
  let f x = Rune.sum (Fn.avg_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) x) in
  let grad_x = Rune.grad f x in
  let expected = Rune.full Rune.float32 [| 1; 1; 4; 4 |] 0.25 in
  check_rune ~eps "avg_pool2d dx" expected grad_x

let test_grad_avg_pool2d_overlapping () =
  let x =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i + 1)))
  in
  (* Overlapping 2x2 avg pool with stride 1: 3x3 output. Each output window
     contributes 0.25 per input pixel it covers. Corner pixels appear in 1
     window, edge in 2, interior in 4. *)
  let f x = Rune.sum (Fn.avg_pool2d ~kernel_size:(2, 2) ~stride:(1, 1) x) in
  let grad_x = Rune.grad f x in
  let expected =
    Rune.create Rune.float32 [| 1; 1; 4; 4 |]
      [|
        0.25;
        0.5;
        0.5;
        0.25;
        0.5;
        1.0;
        1.0;
        0.5;
        0.5;
        1.0;
        1.0;
        0.5;
        0.25;
        0.5;
        0.5;
        0.25;
      |]
  in
  check_rune ~eps "avg_pool2d overlapping dx" expected grad_x

let () =
  run "Kaun.Fn"
    [
      group "conv1d"
        [
          test "basic" test_conv1d_basic;
          test "same padding" test_conv1d_same_padding;
          test "stride" test_conv1d_stride;
          test "dilation" test_conv1d_dilation;
          test "bias" test_conv1d_bias;
          test "groups" test_conv1d_groups;
        ];
      group "conv2d"
        [
          test "basic" test_conv2d_basic;
          test "same padding" test_conv2d_same_padding;
          test "stride" test_conv2d_stride;
          test "dilation" test_conv2d_dilation;
          test "multi-channel" test_conv2d_multi_channel;
          test "groups" test_conv2d_groups;
          test "bias" test_conv2d_bias;
        ];
      group "max_pool"
        [
          test "1d basic" test_max_pool1d_basic;
          test "1d same padding" test_max_pool1d_same_padding;
          test "2d basic" test_max_pool2d_basic;
          test "2d stride 1" test_max_pool2d_stride1;
        ];
      group "avg_pool"
        [
          test "1d basic" test_avg_pool1d_basic;
          test "2d basic" test_avg_pool2d_basic;
          test "2d same padding" test_avg_pool2d_same_padding;
        ];
      group "gradients"
        [
          test "conv2d" test_grad_conv2d;
          test "avg_pool2d" test_grad_avg_pool2d;
          test "avg_pool2d overlapping" test_grad_avg_pool2d_overlapping;
        ];
    ]
