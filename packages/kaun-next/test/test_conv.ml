(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun_next

(* Float64 instances for gradient checking; Conv's traversals are dtype-generic,
   so the instance is just a type pin. Tensor64 treats a bare tensor as a
   one-leaf parameter tree, for gradients with respect to an input. *)

module Conv64 = struct
  type t = Nx.float64_elt Conv.params

  let map = Conv.map
  let map2 = Conv.map2
  let iter = Conv.iter
end

module Tensor64 = struct
  type t = (float, Nx.float64_elt) Nx.t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) x = f x

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) x = f x
end

let grads_ok = function Ok () -> () | Error m -> fail m
let shape_is ?msg expected t = equal ?msg (array int) expected (Nx.shape t)

let values_are ?msg ~tol expected t =
  equal ?msg (array (float tol)) expected (Nx.to_array t)

let image shape values = Nx.create Nx.float32 shape values
let arange_image shape n = image shape (Array.init n float_of_int)

(* Conv *)

let test_conv_init_shapes () =
  Nx.Rng.run ~seed:1 @@ fun () ->
  let p = Conv.init ~in_channels:3 ~out_channels:5 ~kernel_size:(2, 4) in
  shape_is ~msg:"w shape" [| 5; 3; 2; 4 |] p.Conv.w;
  match p.Conv.b with
  | None -> fail "init should create a bias"
  | Some b -> shape_is ~msg:"b shape" [| 5 |] b

let test_conv_identity_kernel () =
  (* A 3x3 kernel with a single centre tap and `Same padding reproduces the
     input exactly: the zero padding only meets zero filter taps. *)
  let p =
    {
      Conv.w = image [| 1; 1; 3; 3 |] [| 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0. |];
      b = None;
    }
  in
  let x = arange_image [| 1; 1; 3; 4 |] 12 in
  values_are ~msg:"identity kernel is the identity" ~tol:1e-6 (Nx.to_array x)
    (Conv.apply ~padding:`Same p x)

let test_conv_window_sums () =
  (* A 2x2 kernel of ones sums each window; the bias then shifts every output of
     its channel. *)
  let p =
    {
      Conv.w = image [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |];
      b = Some (image [| 1 |] [| 10. |]);
    }
  in
  let x = arange_image [| 1; 1; 3; 3 |] 9 in
  values_are ~msg:"window sums plus bias" ~tol:1e-6 [| 18.; 22.; 30.; 34. |]
    (Conv.apply p x)

let test_conv_edge_detector_row () =
  (* Cross-correlation with [1, -1]: y(i) = x(i) - x(i + 1), so y is the negated
     horizontal gradient. No kernel flip. *)
  let p = { Conv.w = image [| 1; 1; 1; 2 |] [| 1.; -1. |]; b = None } in
  let x = image [| 1; 1; 1; 4 |] [| 1.; 3.; 6.; 10. |] in
  let y = Conv.apply p x in
  shape_is ~msg:"valid output shrinks by kw - 1" [| 1; 1; 1; 3 |] y;
  values_are ~msg:"x(i) - x(i+1)" ~tol:1e-6 [| -2.; -3.; -4. |] y

let test_conv_mixes_channels () =
  (* 1x1 filters reduce convolution to a per-pixel channel mix. *)
  let p =
    { Conv.w = image [| 2; 2; 1; 1 |] [| 2.; 3.; 10.; 100. |]; b = None }
  in
  (* Channel 0 is [1; 2], channel 1 is [3; 4]. *)
  let x = image [| 1; 2; 1; 2 |] [| 1.; 2.; 3.; 4. |] in
  values_are ~msg:"co0 = 2*c0 + 3*c1; co1 = 10*c0 + 100*c1" ~tol:1e-6
    [| 11.; 16.; 310.; 420. |] (Conv.apply p x)

let test_conv_stride () =
  let p = { Conv.w = image [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |]; b = None } in
  let x = arange_image [| 1; 1; 4; 4 |] 16 in
  let y = Conv.apply ~stride:(2, 2) p x in
  shape_is ~msg:"stride 2 halves the output" [| 1; 1; 2; 2 |] y;
  values_are ~msg:"disjoint window sums" ~tol:1e-6 [| 10.; 18.; 42.; 50. |] y

let test_conv_shape_contracts () =
  Nx.Rng.run ~seed:2 @@ fun () ->
  let p = Conv.init ~in_channels:3 ~out_channels:4 ~kernel_size:(3, 3) in
  let x = Nx.zeros Nx.float32 [| 2; 3; 8; 7 |] in
  shape_is ~msg:"valid, stride 1" [| 2; 4; 6; 5 |] (Conv.apply p x);
  shape_is ~msg:"same, stride 1 preserves the spatial size" [| 2; 4; 8; 7 |]
    (Conv.apply ~padding:`Same p x);
  shape_is ~msg:"valid, stride 2" [| 2; 4; 3; 3 |]
    (Conv.apply ~stride:(2, 2) p x);
  shape_is ~msg:"same, stride 2 is the ceiling division" [| 2; 4; 4; 4 |]
    (Conv.apply ~stride:(2, 2) ~padding:`Same p x)

let test_conv_custom_inits_and_fans () =
  let fans = ref None in
  let w_init ~fan_in ~fan_out dtype shape =
    fans := Some (fan_in, fan_out);
    Init.constant 2.0 ~fan_in ~fan_out dtype shape
  in
  let p =
    Conv.make ~w_init ~bias_init:(Init.constant 1.0) ~in_channels:2
      ~out_channels:3 ~kernel_size:(2, 2) Nx.float32
  in
  values_are ~msg:"w_init" ~tol:0.0 (Array.make 24 2.0) p.Conv.w;
  (match p.Conv.b with
  | None -> fail "make should create a bias by default"
  | Some b -> values_are ~msg:"bias_init" ~tol:0.0 (Array.make 3 1.0) b);
  match !fans with
  | None -> fail "w_init was never applied"
  | Some (fan_in, fan_out) ->
      equal ~msg:"fan_in is in_channels * kh * kw" int 8 fan_in;
      equal ~msg:"fan_out is out_channels * kh * kw" int 12 fan_out

let test_conv_no_bias () =
  Nx.Rng.run ~seed:3 @@ fun () ->
  let p =
    Conv.make ~bias:false ~in_channels:2 ~out_channels:2 ~kernel_size:(2, 2)
      Nx.float32
  in
  is_true ~msg:"no bias parameter" (p.Conv.b = None);
  equal ~msg:"names without bias" (list string) [ "w" ] (Conv.names p)

let test_conv_names () =
  Nx.Rng.run ~seed:4 @@ fun () ->
  let p = Conv.init ~in_channels:2 ~out_channels:2 ~kernel_size:(2, 2) in
  equal ~msg:"with bias" (list string) [ "w"; "b" ] (Conv.names p)

let test_conv_gradients () =
  Nx.Rng.run ~seed:5 @@ fun () ->
  let x = Nx.randn Nx.float64 [| 2; 2; 4; 4 |] in
  let loss p =
    let y = Conv.apply p x in
    Nx.sum (Nx.mul y y)
  in
  let p =
    Conv.make ~in_channels:2 ~out_channels:2 ~kernel_size:(2, 2) Nx.float64
  in
  grads_ok (Rune_next.check_grads (module Conv64) loss p);
  let no_bias =
    Conv.make ~bias:false ~in_channels:2 ~out_channels:2 ~kernel_size:(3, 3)
      Nx.float64
  in
  let strided_loss p =
    let y = Conv.apply ~stride:(2, 2) ~padding:`Same p x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune_next.check_grads (module Conv64) strided_loss no_bias)

let test_conv_input_gradients () =
  Nx.Rng.run ~seed:6 @@ fun () ->
  let p =
    Conv.make ~in_channels:2 ~out_channels:3 ~kernel_size:(2, 2) Nx.float64
  in
  let loss x =
    let y = Conv.apply ~padding:`Same p x in
    Nx.sum (Nx.mul y y)
  in
  let x = Nx.randn Nx.float64 [| 1; 2; 3; 3 |] in
  grads_ok (Rune_next.check_grads (module Tensor64) loss x)

let test_conv_rejects_bad_geometry () =
  raises_invalid_arg
    "Conv.make: channels and kernel size must be positive, got in_channels=0 \
     out_channels=4 kernel_size=(3, 3)" (fun () ->
      Conv.make ~in_channels:0 ~out_channels:4 ~kernel_size:(3, 3) Nx.float32);
  raises_invalid_arg
    "Conv.make: channels and kernel size must be positive, got in_channels=2 \
     out_channels=2 kernel_size=(3, 0)" (fun () ->
      Conv.make ~in_channels:2 ~out_channels:2 ~kernel_size:(3, 0) Nx.float32)

let test_conv_rejects_bad_input () =
  Nx.Rng.run ~seed:7 @@ fun () ->
  let p = Conv.init ~in_channels:2 ~out_channels:2 ~kernel_size:(3, 3) in
  raises_invalid_arg
    "Conv.apply: input must be [batch; channels; height; width], got rank 3"
    (fun () -> Conv.apply p (Nx.zeros Nx.float32 [| 2; 4; 4 |]));
  raises_invalid_arg "Conv.apply: input has 3 channels but the layer expects 2"
    (fun () -> Conv.apply p (Nx.zeros Nx.float32 [| 1; 3; 4; 4 |]));
  raises_invalid_arg "Conv.apply: stride must be positive, got (0, 1)"
    (fun () ->
      Conv.apply ~stride:(0, 1) p (Nx.zeros Nx.float32 [| 1; 2; 4; 4 |]));
  raises_invalid_arg "Conv.apply: kernel (3, 3) does not fit input (2, 5)"
    (fun () -> Conv.apply p (Nx.zeros Nx.float32 [| 1; 2; 2; 5 |]))

(* Pool *)

let test_max_pool_analytic () =
  let x = arange_image [| 1; 1; 4; 4 |] 16 in
  let y = Pool.max_pool2d ~kernel_size:(2, 2) x in
  shape_is ~msg:"stride defaults to kernel_size" [| 1; 1; 2; 2 |] y;
  values_are ~msg:"window maxima" ~tol:0.0 [| 5.; 7.; 13.; 15. |] y

let test_avg_pool_analytic () =
  let x = arange_image [| 1; 1; 4; 4 |] 16 in
  values_are ~msg:"window means" ~tol:1e-6 [| 2.5; 4.5; 10.5; 12.5 |]
    (Pool.avg_pool2d ~kernel_size:(2, 2) x)

let test_pool_overlapping_stride () =
  let x = arange_image [| 1; 1; 3; 3 |] 9 in
  values_are ~msg:"unit stride overlaps the windows" ~tol:0.0
    [| 4.; 5.; 7.; 8. |]
    (Pool.max_pool2d ~kernel_size:(2, 2) ~stride:(1, 1) x);
  values_are ~msg:"unit stride window means" ~tol:1e-6 [| 2.; 3.; 5.; 6. |]
    (Pool.avg_pool2d ~kernel_size:(2, 2) ~stride:(1, 1) x)

let test_pool_rectangular_kernel () =
  let x = image [| 1; 1; 1; 4 |] [| 3.; 1.; 4.; 1. |] in
  values_are ~msg:"1x2 maxima" ~tol:0.0 [| 3.; 4. |]
    (Pool.max_pool2d ~kernel_size:(1, 2) x);
  values_are ~msg:"1x2 means" ~tol:1e-6 [| 2.; 2.5 |]
    (Pool.avg_pool2d ~kernel_size:(1, 2) x)

let test_pool_shape_contracts () =
  let x = Nx.zeros Nx.float32 [| 2; 3; 5; 6 |] in
  shape_is ~msg:"leading axes are preserved" [| 2; 3; 2; 3 |]
    (Pool.max_pool2d ~kernel_size:(2, 2) x);
  shape_is ~msg:"a bare image pools too" [| 2; 2 |]
    (Pool.avg_pool2d ~kernel_size:(2, 2) (Nx.zeros Nx.float32 [| 4; 4 |]))

let test_max_pool_gradient_routes_to_max () =
  let x = Nx.create Nx.float64 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int) in
  let loss x = Nx.sum (Pool.max_pool2d ~kernel_size:(2, 2) x) in
  let g = Rune_next.grad (module Tensor64) loss x in
  let expected = Array.make 16 0.0 in
  List.iter (fun i -> expected.(i) <- 1.0) [ 5; 7; 13; 15 ];
  values_are ~msg:"gradient is 1 at each window maximum" ~tol:1e-12 expected g

let test_pool_gradients () =
  Nx.Rng.run ~seed:8 @@ fun () ->
  let x = Nx.randn Nx.float64 [| 1; 2; 4; 4 |] in
  let max_loss x =
    let y = Pool.max_pool2d ~kernel_size:(2, 2) ~stride:(1, 1) x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune_next.check_grads (module Tensor64) max_loss x);
  let avg_loss x =
    let y = Pool.avg_pool2d ~kernel_size:(2, 2) x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune_next.check_grads (module Tensor64) avg_loss x)

let test_pool_rejects_bad_input () =
  let x = Nx.zeros Nx.float32 [| 1; 1; 2; 5 |] in
  raises_invalid_arg "Pool.max_pool2d: kernel_size must be positive, got (0, 2)"
    (fun () -> Pool.max_pool2d ~kernel_size:(0, 2) x);
  raises_invalid_arg "Pool.avg_pool2d: stride must be positive, got (1, 0)"
    (fun () -> Pool.avg_pool2d ~kernel_size:(2, 2) ~stride:(1, 0) x);
  raises_invalid_arg
    "Pool.max_pool2d: input must have at least 2 axes, got rank 1" (fun () ->
      Pool.max_pool2d ~kernel_size:(1, 1) (Nx.zeros Nx.float32 [| 4 |]));
  raises_invalid_arg "Pool.max_pool2d: kernel (3, 3) does not fit input (2, 5)"
    (fun () -> Pool.max_pool2d ~kernel_size:(3, 3) x)

let () =
  run "kaun-next conv"
    [
      group "conv"
        [
          test "init produces the documented shapes" test_conv_init_shapes;
          test "an identity kernel is the identity" test_conv_identity_kernel;
          test "a kernel of ones sums each window" test_conv_window_sums;
          test "correlates without flipping the kernel"
            test_conv_edge_detector_row;
          test "1x1 filters mix channels per pixel" test_conv_mixes_channels;
          test "stride subsamples the windows" test_conv_stride;
          test "output shapes follow stride and padding"
            test_conv_shape_contracts;
          test "make respects w_init, bias_init and the conv fans"
            test_conv_custom_inits_and_fans;
          test "bias:false drops the bias parameter" test_conv_no_bias;
          test "names follow traversal order" test_conv_names;
          test "parameter gradients agree with finite differences"
            test_conv_gradients;
          test "input gradients agree with finite differences"
            test_conv_input_gradients;
          test "make rejects non-positive geometry"
            test_conv_rejects_bad_geometry;
          test "apply rejects invalid inputs" test_conv_rejects_bad_input;
        ];
      group "pooling"
        [
          test "max pool takes each window maximum" test_max_pool_analytic;
          test "avg pool takes each window mean" test_avg_pool_analytic;
          test "stride overrides the window step" test_pool_overlapping_stride;
          test "rectangular kernels pool each axis independently"
            test_pool_rectangular_kernel;
          test "leading axes are preserved" test_pool_shape_contracts;
          test "max pool routes the gradient to the maximum"
            test_max_pool_gradient_routes_to_max;
          test "gradients agree with finite differences" test_pool_gradients;
          test "invalid inputs are rejected" test_pool_rejects_bad_input;
        ];
    ]
