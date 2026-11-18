(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

(* Conv's walk is dtype-generic, so each instance is a type pin: float64 for
   gradient checking. *)

let conv64 = Nx.Ptree.instantiate (module Conv)

let paths p =
  let conv = Nx.Ptree.instantiate (module Conv) in
  List.rev
    (Nx.Ptree.fold conv
       (fun path _ acc -> Nx.Ptree.Path.to_string path :: acc)
       p [])

let grads_ok = function Ok () -> () | Error m -> fail m
let shape_is ?msg expected t = equal ?msg (array int) expected (Nx.shape t)

let values_are ?msg ~tol expected t =
  let ft = if tol = 0. then float_exact else float tol in
  equal ?msg (array ft) expected (Nx.to_array t)

let image shape values = Nx.create Nx.float32 shape values
let arange_image shape n = image shape (Array.init n float_of_int)

(* Goldens

   golden/conv.npz holds PyTorch's convolutions of inputs, weights and
   biases that are multiples of 1/4, so that every sum is exact in float64
   and each result is the exact one. Activations are channels last and
   weights in PyTorch's layout, moved as an importer moves them. *)

let golden = lazy (Nx_io.load_npz "golden/conv.npz")

let entry name =
  match Nx_io.Archive.find name (Lazy.force golden) with
  | Some (Nx.P x) -> Nx.cast Nx.float64 x
  | None -> failwith ("golden/conv.npz has no " ^ name)

let moved name = Nx.moveaxis 1 (-1) (entry name)

let exact ~msg want got =
  equal ~msg:(msg ^ ", shape") (array int) (Nx.shape want) (Nx.shape got);
  equal ~msg (array float_exact) (Nx.to_array want) (Nx.to_array got)

(* Whisper's stem, two 3-tap convolutions, the second at stride 2, both
   padded by one each side: at an even extent and an odd one, which split a
   stride-dependent padding differently. *)
let test_stem () =
  List.iter
    (fun t ->
      let e = Printf.sprintf "stem%d." t in
      let c1 = { Conv.w = moved (e ^ "w1"); b = Some (entry (e ^ "b1")) } in
      let c2 = { Conv.w = moved (e ^ "w2"); b = Some (entry (e ^ "b2")) } in
      let padding = [| (1, 1) |] in
      let y1 = Conv.apply ~padding c1 (entry (e ^ "x")) in
      exact ~msg:(Printf.sprintf "%d frames, conv1" t) (entry (e ^ "y1")) y1;
      let y2 = Conv.apply ~stride:[| 2 |] ~padding c2 (entry (e ^ "y1")) in
      exact ~msg:(Printf.sprintf "%d frames, conv2" t) (entry (e ^ "y2")) y2)
    [ 3000; 3001 ]

let test_grouped () =
  let p = { Conv.w = moved "grouped.w"; b = Some (entry "grouped.b") } in
  exact ~msg:"two groups, dilation (2, 1), stride (2, 1)" (entry "grouped.y")
    (Conv.apply ~stride:[| 2; 1 |] ~padding:[| (1, 2); (0, 1) |]
       ~dilation:[| 2; 1 |] ~groups:2 p (entry "grouped.x"))

let test_volume () =
  let p = { Conv.w = moved "volume.w"; b = None } in
  exact ~msg:"3-D, stride (1, 2, 1)" (entry "volume.y")
    (Conv.apply ~stride:[| 1; 2; 1 |]
       ~padding:[| (1, 1); (0, 0); (1, 1) |]
       p (entry "volume.x"))

(* SAM 2's upscaling, a 2x2 transpose at stride 2, and a transpose that
   PyTorch states with padding 1 and output padding 1: the pair (1, 0). *)
let test_upscale () =
  let p =
    { Conv_transpose.w = moved "upscale.w"; b = Some (entry "upscale.b") }
  in
  exact ~msg:"k2 s2" (entry "upscale.y")
    (Conv_transpose.apply ~stride:[| 2; 2 |] p (entry "upscale.x"))

let test_output_padding () =
  let p = { Conv_transpose.w = moved "outpad.w"; b = Some (entry "outpad.b") } in
  exact ~msg:"k3 s2, padding (1, 1 - 1)" (entry "outpad.y")
    (Conv_transpose.apply ~stride:[| 2 |] ~padding:[| (1, 0) |] p
       (entry "outpad.x"))

(* Conv_transpose *)

let transpose64 = Nx.Ptree.instantiate (module Conv_transpose)

let test_transpose_shapes () =
  Nx.Rng.with_key (Nx.Rng.key 9) @@ fun () ->
  let fans = ref [] in
  let w_init key ~fan_in ~fan_out dtype shape =
    fans := (fan_in, fan_out) :: !fans;
    Init.glorot_uniform key ~fan_in ~fan_out dtype shape
  in
  let p =
    Conv_transpose.make ~w_init ~key:(Nx.Rng.next_key ()) ~in_channels:4
      ~out_channels:2 ~kernel_size:[| 2; 3 |] Nx.float32
  in
  shape_is ~msg:"w is [in; k...; out]" [| 4; 2; 3; 2 |] p.Conv_transpose.w;
  (match p.Conv_transpose.b with
  | None -> fail "make should create a bias"
  | Some b -> shape_is ~msg:"b is [out]" [| 2 |] b);
  (* One output reads in_channels at each tap; one input feeds
     out_channels. *)
  equal ~msg:"(fan_in, fan_out)" (list (pair int int)) [ (24, 12) ] !fans;
  let y = Conv_transpose.apply ~stride:[| 2; 2 |] p (Nx.zeros Nx.float32 [| 1; 3; 3; 4 |]) in
  shape_is ~msg:"(t - 1) stride + k" [| 1; 6; 7; 2 |] y;
  let none =
    Conv_transpose.make ~bias:false ~key:(Nx.Rng.next_key ()) ~in_channels:4
      ~out_channels:2 ~kernel_size:[| 2 |] Nx.float32
  in
  equal ~msg:"paths without bias" (list string) [ "w" ]
    (List.rev
       (Nx.Ptree.fold
          (Nx.Ptree.instantiate (module Conv_transpose))
          (fun path _ acc -> Nx.Ptree.Path.to_string path :: acc)
          none []))

let test_transpose_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 10) @@ fun () ->
  let x = Nx.Rng.normal Nx.float64 [| 2; 3; 3; 2 |] in
  let p =
    Conv_transpose.make ~key:(Nx.Rng.next_key ()) ~in_channels:2
      ~out_channels:3 ~kernel_size:[| 2; 2 |] Nx.float64
  in
  let loss p =
    let y = Conv_transpose.apply ~stride:[| 2; 1 |] ~padding:[| (0, 1); (1, 0) |] p x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads transpose64 loss p)

let test_transpose_refusals () =
  let make ~in_channels ~out_channels kernel_size () =
    ignore
      (Conv_transpose.make ~key:(Nx.Rng.next_key ()) ~in_channels ~out_channels
         ~kernel_size Nx.float32)
  in
  raises
    (Invalid_argument
       "Conv_transpose.make: channels and kernel extents must be positive, \
        got in_channels=2 out_channels=0 kernel_size=[|2|]")
    (make ~in_channels:2 ~out_channels:0 [| 2 |]);
  raises (Invalid_argument "Conv_transpose.make: kernel_size has no extent")
    (make ~in_channels:2 ~out_channels:2 [||]);
  let p =
    Conv_transpose.init ~key:(Nx.Rng.next_key ()) ~in_channels:2
      ~out_channels:2 ~kernel_size:[| 2 |]
  in
  raises_match (Exn.invalid_arg ~sub:"Nx.conv_transpose: ") (fun () ->
      Conv_transpose.apply p (Nx.zeros Nx.float32 [| 1; 4; 3 |]))

(* Conv *)

let test_conv_init_shapes () =
  Nx.Rng.with_key (Nx.Rng.key 1) @@ fun () ->
  let p =
    Conv.init ~key:(Nx.Rng.next_key ()) ~in_channels:3 ~out_channels:5
      ~kernel_size:[| 2; 4 |]
  in
  shape_is ~msg:"w is [out; k...; in]" [| 5; 2; 4; 3 |] p.Conv.w;
  (match p.Conv.b with
  | None -> fail "init should create a bias"
  | Some b -> shape_is ~msg:"b shape" [| 5 |] b);
  let g =
    Conv.make ~key:(Nx.Rng.next_key ()) ~groups:2 ~in_channels:4
      ~out_channels:6 ~kernel_size:[| 3 |] Nx.float32
  in
  shape_is ~msg:"grouped w is [out; k...; in / groups]" [| 6; 3; 2 |] g.Conv.w

let test_conv_window_sums () =
  (* A 2x2 kernel of ones sums each window; the bias then shifts every output
     of its channel. *)
  let p =
    {
      Conv.w = image [| 1; 2; 2; 1 |] [| 1.; 1.; 1.; 1. |];
      b = Some (image [| 1 |] [| 10. |]);
    }
  in
  let x = arange_image [| 1; 3; 3; 1 |] 9 in
  values_are ~msg:"window sums plus bias" ~tol:0. [| 18.; 22.; 30.; 34. |]
    (Conv.apply p x)

let test_conv_correlates () =
  (* y(i) = x(i) - x(i + 1): the kernel is not flipped. *)
  let p = { Conv.w = image [| 1; 2; 1 |] [| 1.; -1. |]; b = None } in
  let x = image [| 1; 4; 1 |] [| 1.; 3.; 6.; 10. |] in
  let y = Conv.apply p x in
  shape_is ~msg:"unpadded output shrinks by k - 1" [| 1; 3; 1 |] y;
  values_are ~msg:"x(i) - x(i+1)" ~tol:0. [| -2.; -3.; -4. |] y

let test_conv_mixes_channels () =
  (* 1x1 filters reduce convolution to a per-pixel channel mix: channels
     last, pixel 0 is (1, 3) and pixel 1 is (2, 4). *)
  let p = { Conv.w = image [| 2; 1; 1; 2 |] [| 2.; 3.; 10.; 100. |]; b = None } in
  let x = image [| 1; 1; 2; 2 |] [| 1.; 3.; 2.; 4. |] in
  values_are ~msg:"co0 = 2 c0 + 3 c1; co1 = 10 c0 + 100 c1" ~tol:0.
    [| 11.; 310.; 16.; 420. |] (Conv.apply p x)

(* At float16 a sum of 2048 and 1 rounds to 2048: added to the bias of
   -2048 before that rounding, as [apply] does, it gives 1, where a sum
   rounded first would give 0. *)
let test_conv_bias_rounds_once () =
  let f16 s vs = Nx.cast Nx.float16 (Nx.create Nx.float32 s vs) in
  let p = { Conv.w = f16 [| 1; 1; 2 |] [| 1.; 1. |]; b = Some (f16 [| 1 |] [| -2048. |]) } in
  values_are ~msg:"one rounding" ~tol:0. [| 1. |]
    (Nx.cast Nx.float32 (Conv.apply p (f16 [| 1; 1; 2 |] [| 2048.; 1. |])))

let test_conv_custom_inits_and_fans () =
  let fans = ref [] in
  let w_init key ~fan_in ~fan_out dtype shape =
    fans := (fan_in, fan_out) :: !fans;
    Init.constant 2.0 key ~fan_in ~fan_out dtype shape
  in
  let p =
    Conv.make ~key:(Nx.Rng.next_key ()) ~w_init ~bias_init:(Init.constant 1.0)
      ~in_channels:2 ~out_channels:3 ~kernel_size:[| 2; 2 |] Nx.float32
  in
  values_are ~msg:"w_init" ~tol:0.0 (Array.make 24 2.0) p.Conv.w;
  (match p.Conv.b with
  | None -> fail "make should create a bias by default"
  | Some b -> values_are ~msg:"bias_init" ~tol:0.0 (Array.make 3 1.0) b);
  ignore
    (Conv.make ~key:(Nx.Rng.next_key ()) ~w_init ~groups:4 ~in_channels:8
       ~out_channels:12 ~kernel_size:[| 3; 3 |] Nx.float32);
  (* One output reads in_channels / groups channels of each tap; one input
     feeds out_channels / groups outputs at each. *)
  equal ~msg:"(fan_in, fan_out), ungrouped then four groups"
    (list (pair int int))
    [ (8, 12); (18, 27) ]
    (List.rev !fans)

let test_conv_no_bias () =
  Nx.Rng.with_key (Nx.Rng.key 3) @@ fun () ->
  let p =
    Conv.make ~key:(Nx.Rng.next_key ()) ~bias:false ~in_channels:2
      ~out_channels:2 ~kernel_size:[| 2; 2 |] Nx.float32
  in
  equal ~msg:"paths without bias" (list string) [ "w" ] (paths p)

let test_conv_names () =
  Nx.Rng.with_key (Nx.Rng.key 4) @@ fun () ->
  let p =
    Conv.init ~key:(Nx.Rng.next_key ()) ~in_channels:2 ~out_channels:2
      ~kernel_size:[| 2; 2 |]
  in
  equal ~msg:"with bias" (list string) [ "w"; "b" ] (paths p)

let test_conv_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 5) @@ fun () ->
  let x = Nx.Rng.normal Nx.float64 [| 2; 4; 4; 2 |] in
  let loss p =
    let y = Conv.apply p x in
    Nx.sum (Nx.mul y y)
  in
  let p =
    Conv.make ~key:(Nx.Rng.next_key ()) ~in_channels:2 ~out_channels:2
      ~kernel_size:[| 2; 2 |] Nx.float64
  in
  grads_ok (Rune.check_grads conv64 loss p);
  let grouped =
    Conv.make ~key:(Nx.Rng.next_key ()) ~groups:2 ~in_channels:2
      ~out_channels:4 ~kernel_size:[| 3; 3 |] Nx.float64
  in
  let strided_loss p =
    let y =
      Conv.apply ~stride:[| 2; 2 |] ~padding:[| (1, 1); (0, 1) |] ~groups:2 p x
    in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads conv64 strided_loss grouped)

let test_conv_input_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 6) @@ fun () ->
  let p =
    Conv.make ~key:(Nx.Rng.next_key ()) ~in_channels:2 ~out_channels:3
      ~kernel_size:[| 3 |] Nx.float64
  in
  let loss x =
    let y = Conv.apply ~padding:[| (1, 1) |] ~dilation:[| 2 |] p x in
    Nx.sum (Nx.mul y y)
  in
  let x = Nx.Rng.normal Nx.float64 [| 1; 7; 2 |] in
  grads_ok (Rune.check_grads Nx.Ptree.tensor loss x)

let test_conv_rejects_bad_geometry () =
  let make ?groups ~in_channels ~out_channels kernel_size () =
    ignore
      (Conv.make ~key:(Nx.Rng.next_key ()) ?groups ~in_channels ~out_channels
         ~kernel_size Nx.float32)
  in
  raises
    (Invalid_argument
       "Conv.make: channels and kernel extents must be positive, got \
        in_channels=0 out_channels=4 kernel_size=[|3; 3|]")
    (make ~in_channels:0 ~out_channels:4 [| 3; 3 |]);
  raises
    (Invalid_argument
       "Conv.make: channels and kernel extents must be positive, got \
        in_channels=2 out_channels=2 kernel_size=[|3; 0|]")
    (make ~in_channels:2 ~out_channels:2 [| 3; 0 |]);
  raises (Invalid_argument "Conv.make: kernel_size has no extent")
    (make ~in_channels:2 ~out_channels:2 [||]);
  raises
    (Invalid_argument
       "Conv.make: 3 groups do not divide in_channels=6 and out_channels=4")
    (make ~groups:3 ~in_channels:6 ~out_channels:4 [| 3 |])

let test_conv_rejects_bad_input () =
  Nx.Rng.with_key (Nx.Rng.key 7) @@ fun () ->
  let p =
    Conv.init ~key:(Nx.Rng.next_key ()) ~in_channels:2 ~out_channels:2
      ~kernel_size:[| 3; 3 |]
  in
  let rejects f = raises_match (Exn.invalid_arg ~sub:"Nx.conv: ") f in
  rejects (fun () -> Conv.apply p (Nx.zeros Nx.float32 [| 4; 4; 2 |]));
  rejects (fun () -> Conv.apply p (Nx.zeros Nx.float32 [| 1; 4; 4; 3 |]));
  rejects (fun () ->
      Conv.apply ~stride:[| 0; 1 |] p (Nx.zeros Nx.float32 [| 1; 4; 4; 2 |]))

(* Pool *)

let test_pool_goldens () =
  exact ~msg:"3x3 max at stride 2, padded by 1" (entry "maxpool.y")
    (Pool.max ~window:[| 3; 3 |] ~stride:[| 2; 2 |]
       ~padding:[| (1, 1); (1, 1) |] (entry "maxpool.x"));
  exact ~msg:"2x2 mean at stride 2, padded by 1, padding counted"
    (entry "avgpool.y")
    (Pool.avg ~window:[| 2; 2 |] ~stride:[| 2; 2 |]
       ~padding:[| (1, 1); (1, 1) |] (entry "avgpool.x"));
  exact ~msg:"3-D mean" (entry "avgpool3.y")
    (Pool.avg ~window:[| 2; 2; 2 |] ~stride:[| 1; 2; 2 |] (entry "avgpool3.x"))

let test_max_pool_analytic () =
  let x = arange_image [| 1; 4; 4; 1 |] 16 in
  let y = Pool.max ~window:[| 2; 2 |] x in
  shape_is ~msg:"stride defaults to the window" [| 1; 2; 2; 1 |] y;
  values_are ~msg:"window maxima" ~tol:0.0 [| 5.; 7.; 13.; 15. |] y;
  (* Channels last: each channel pools alone. *)
  let x = image [| 1; 4; 2 |] [| 1.; 8.; 3.; 6.; 2.; 7.; 4.; 5. |] in
  values_are ~msg:"two channels" ~tol:0.0 [| 3.; 8.; 4.; 7. |]
    (Pool.max ~window:[| 2 |] x)

let test_avg_pool_analytic () =
  let x = arange_image [| 1; 4; 4; 1 |] 16 in
  values_are ~msg:"window means" ~tol:0. [| 2.5; 4.5; 10.5; 12.5 |]
    (Pool.avg ~window:[| 2; 2 |] x);
  let x = arange_image [| 1; 3; 3; 1 |] 9 in
  values_are ~msg:"unit stride overlaps the windows" ~tol:0.
    [| 2.; 3.; 5.; 6. |]
    (Pool.avg ~window:[| 2; 2 |] ~stride:[| 1; 1 |] x)

(* A window in padding alone holds the dtype's least value: -inf, and
   -448 in float8_e4m3fn, which has no infinity. A window that meets [x]
   takes [x]'s greatest, below zero here. *)
let test_max_pool_padding () =
  let x = image [| 1; 2; 1 |] [| -3.; -2. |] in
  let padding = [| (2, 1) |] in
  values_are ~msg:"float32" ~tol:0. [| Float.neg_infinity; -3.; -2.; -2. |]
    (Pool.max ~window:[| 2 |] ~stride:[| 1 |] ~padding x);
  values_are ~msg:"float8_e4m3fn" ~tol:0. [| -448.; -3.; -2.; -2. |]
    (Nx.cast Nx.float32
       (Pool.max ~window:[| 2 |] ~stride:[| 1 |] ~padding
          (Nx.cast Nx.float8_e4m3fn x)))

(* Padding counts as zeros in each mean; dividing by the mean of ones
   leaves the mean of [x]'s elements alone. A larger [hi] keeps a last,
   partial window. *)
let test_avg_pool_padding () =
  let x = image [| 1; 5; 1 |] [| 2.; 4.; 6.; 8.; 10. |] in
  let padding = [| (0, 1) |] and window = [| 2 |] in
  let y = Pool.avg ~window ~padding x in
  values_are ~msg:"padding counted" ~tol:0. [| 3.; 7.; 5. |] y;
  values_are ~msg:"x's elements alone" ~tol:0. [| 3.; 7.; 10. |]
    (Nx.div y (Pool.avg ~window ~padding (Nx.ones_like x)));
  shape_is ~msg:"no partial window without it" [| 1; 2; 1 |]
    (Pool.avg ~window x)

let test_pool_shapes () =
  let x = Nx.zeros Nx.float32 [| 2; 5; 6; 3 |] in
  shape_is ~msg:"batch and channels are kept" [| 2; 2; 3; 3 |]
    (Pool.max ~window:[| 2; 2 |] x);
  shape_is ~msg:"1-D" [| 2; 3; 4 |]
    (Pool.avg ~window:[| 3 |] ~stride:[| 2 |] (Nx.zeros Nx.float32 [| 2; 7; 4 |]))

let test_max_pool_gradient_routes_to_max () =
  let x = Nx.create Nx.float64 [| 1; 4; 4; 1 |] (Array.init 16 float_of_int) in
  let loss x = Nx.sum (Pool.max ~window:[| 2; 2 |] x) in
  let g = Rune.grad Nx.Ptree.tensor loss x in
  let expected = Array.make 16 0.0 in
  List.iter (fun i -> expected.(i) <- 1.0) [ 5; 7; 13; 15 ];
  values_are ~msg:"gradient is 1 at each window maximum" ~tol:0. expected g

let test_pool_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 8) @@ fun () ->
  let x = Nx.Rng.normal Nx.float64 [| 1; 4; 4; 2 |] in
  let max_loss x =
    let y =
      Pool.max ~window:[| 3; 2 |] ~stride:[| 2; 1 |]
        ~padding:[| (1, 1); (0, 1) |] x
    in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads Nx.Ptree.tensor max_loss x);
  let avg_loss x =
    let y = Pool.avg ~window:[| 2; 2 |] ~padding:[| (1, 0); (0, 1) |] x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads Nx.Ptree.tensor avg_loss x)

let test_pool_rejects_bad_input () =
  let x = Nx.zeros Nx.float32 [| 1; 2; 5; 1 |] in
  raises (Invalid_argument "Pool.max: window has no extent") (fun () ->
      Pool.max ~window:[||] x);
  raises (Invalid_argument "Pool.max: a window extent below 1 in [|0; 2|]")
    (fun () -> Pool.max ~window:[| 0; 2 |] x);
  raises (Invalid_argument "Pool.avg: a stride below 1 in [|1; 0|]")
    (fun () -> Pool.avg ~window:[| 2; 2 |] ~stride:[| 1; 0 |] x);
  raises (Invalid_argument "Pool.avg: 1 strides for 2 window extents")
    (fun () -> Pool.avg ~window:[| 2; 2 |] ~stride:[| 1 |] x);
  raises (Invalid_argument "Pool.max: a negative padding") (fun () ->
      Pool.max ~window:[| 1; 1 |] ~padding:[| (0, 0); (-1, 0) |] x);
  raises
    (Invalid_argument "Pool.max: 2 window extents take an input of 4 axes, got 3")
    (fun () -> Pool.max ~window:[| 1; 1 |] (Nx.zeros Nx.float32 [| 2; 5; 1 |]));
  raises
    (Invalid_argument
       "Pool.max: no window of [|3; 3|] fits the input's spatial axes [|2; 5|]")
    (fun () -> Pool.max ~window:[| 3; 3 |] x)

let () =
  exit
    (run "kaun conv"
       [
         group "goldens"
           [
             test "Whisper's stem at 3000 and 3001 frames" test_stem;
             test "a grouped, dilated 2-D convolution" test_grouped;
             test "a 3-D convolution" test_volume;
             test "SAM 2's 2x2 upscaling" test_upscale;
             test "an output padding is a smaller hi" test_output_padding;
           ];
         group "conv"
           [
             test "init and make produce the documented shapes"
               test_conv_init_shapes;
             test "a kernel of ones sums each window" test_conv_window_sums;
             test "correlates without flipping the kernel" test_conv_correlates;
             test "1x1 filters mix channels per pixel" test_conv_mixes_channels;
             test "the bias joins before the one rounding"
               test_conv_bias_rounds_once;
             test "make respects w_init, bias_init and the fans per group"
               test_conv_custom_inits_and_fans;
             test "bias:false drops the bias parameter" test_conv_no_bias;
             test "names follow walk order" test_conv_names;
             test "parameter gradients agree with finite differences"
               test_conv_gradients;
             test "input gradients agree with finite differences"
               test_conv_input_gradients;
             test "make rejects invalid geometry" test_conv_rejects_bad_geometry;
             test "apply rejects invalid inputs" test_conv_rejects_bad_input;
           ];
         group "conv_transpose"
           [
             test "make's shapes and fans, apply's extents"
               test_transpose_shapes;
             test "parameter gradients agree with finite differences"
               test_transpose_gradients;
             test "refusals" test_transpose_refusals;
           ];
         group "pooling"
           [
             test "PyTorch's pools, 2-D and 3-D" test_pool_goldens;
             test "max takes each window's greatest" test_max_pool_analytic;
             test "avg takes each window's mean" test_avg_pool_analytic;
             test "max pads with the dtype's least value"
               test_max_pool_padding;
             test "avg counts the padding" test_avg_pool_padding;
             test "batch and channels are kept" test_pool_shapes;
             test "max routes the gradient to the maximum"
               test_max_pool_gradient_routes_to_max;
             test "gradients agree with finite differences" test_pool_gradients;
             test "invalid inputs are rejected" test_pool_rejects_bad_input;
           ];
       ])
