(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

(* A tensor's shape and its values widened to float64, which every float dtype
   widens to exactly: equal bits, NaN equal to NaN. *)
let tensor = pair (array int) (array float_exact)
let bits t = (Nx.shape t, Nx.to_array (Nx.cast Nx.float64 t))
let floats shape values = Nx.create Nx.float32 shape values

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let float_dtypes =
  [
    Dtype Nx.float16;
    Dtype Nx.bfloat16;
    Dtype Nx.float32;
    Dtype Nx.float64;
    Dtype Nx.float8_e4m3fn;
    Dtype Nx.float8_e5m2;
    Dtype Nx.float4_e2m1fn;
  ]

let dtype_name (Dtype dt) = Nx.Dtype.name dt
let pp_dtype ppf d = Format.pp_print_string ppf (dtype_name d)

(* A witness of placements at one brand. *)
let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
let software = lazy (Nx.devices [ Result.get_ok (Rig.software "kaun-conv") ])

let visits p =
  List.map
    (Format.asprintf "%a" Nx.Ptree.pp_visit)
    (Nx.Ptree.visits (Nx.Ptree.instantiate (module Conv)) p)

(* Types *)

let test_shapes () =
  let p =
    Conv.make ~groups:2 ~key:(Nx.Rng.key 1) ~in_channels:4 ~out_channels:6
      ~kernel_size:[| 2; 3 |] Nx.float32
  in
  equal ~msg:"w" (array int) [| 6; 2; 3; 2 |] (Nx.shape p.w);
  equal ~msg:"b" (array int) [| 6 |] (Nx.shape (require_some p.b))

let test_no_bias () =
  let p =
    Conv.make ~bias:false ~key:(Nx.Rng.key 2) ~in_channels:2 ~out_channels:3
      ~kernel_size:[| 2 |] Nx.float32
  in
  is_none ~pp:Nx.pp p.b

(* Constructors *)

(* Fans at groups 2 of 4 inputs and 6 outputs over 2 x 3 taps: (4 / 2) * 6 and
   (6 / 2) * 6. *)
let test_make_draws () =
  let key = Nx.Rng.key 3 in
  let p =
    Conv.make ~bias_init:Init.glorot_uniform ~groups:2 ~key ~in_channels:4
      ~out_channels:6 ~kernel_size:[| 2; 3 |] Nx.float32
  in
  let keys = Nx.Rng.split key in
  let glorot key shape =
    Init.glorot_uniform key ~fan_in:12 ~fan_out:18 Nx.float32 shape
  in
  equal ~msg:"w" tensor (bits (glorot keys.(0) [| 6; 2; 3; 2 |])) (bits p.w);
  equal ~msg:"b" tensor
    (bits (glorot keys.(1) [| 6 |]))
    (bits (require_some p.b))

let test_zero_bias () =
  let p =
    Conv.make ~key:(Nx.Rng.key 3) ~in_channels:4 ~out_channels:2
      ~kernel_size:[| 3 |] Nx.float32
  in
  equal tensor (bits (Nx.zeros Nx.float32 [| 2 |])) (bits (require_some p.b))

(* One output reads 8 / 4 channels at each of 9 taps; one input feeds 12 / 4
   outputs at each. *)
let test_make_initializers () =
  let fans = ref [] in
  let init name v key ~fan_in ~fan_out dtype shape =
    fans := (name, fan_in, fan_out) :: !fans;
    Init.constant v key ~fan_in ~fan_out dtype shape
  in
  let p =
    Conv.make ~w_init:(init "w" 2.) ~bias_init:(init "b" 1.) ~groups:4
      ~key:(Nx.Rng.key 4) ~in_channels:8 ~out_channels:12
      ~kernel_size:[| 3; 3 |] Nx.float32
  in
  equal ~msg:"fans"
    (list (triple string int int))
    [ ("w", 18, 27); ("b", 18, 27) ]
    (List.rev !fans);
  equal ~msg:"w" tensor
    (bits (Nx.full Nx.float32 [| 12; 3; 3; 2 |] 2.))
    (bits p.w);
  equal ~msg:"b" tensor
    (bits (Nx.full Nx.float32 [| 12 |] 1.))
    (bits (require_some p.b))

let test_make_placement () =
  let (module D : Nx.Devices) = Lazy.force software in
  let p =
    Conv.make
      ~key:(Nx.Rng.place D.on (Nx.Rng.key 6))
      ~in_channels:2 ~out_channels:2 ~kernel_size:[| 3 |] Nx.float32
  in
  equal ~msg:"w" (placement ()) D.on (Nx.placement p.w);
  equal ~msg:"b" (placement ()) D.on (Nx.placement (require_some p.b))

type geometry = {
  id : string;
  in_channels : int;
  out_channels : int;
  kernel_size : int array;
  groups : int;
}

let bad_geometries =
  let row id ?(groups = 1) in_channels out_channels kernel_size =
    { id; in_channels; out_channels; kernel_size; groups }
  in
  [
    row "an empty kernel_size" 2 2 [||];
    row "no in_channels" 0 4 [| 3; 3 |];
    row "no out_channels" 2 0 [| 3 |];
    row "a kernel extent of 0" 2 2 [| 3; 0 |];
    row "a negative kernel extent" 2 2 [| -1 |];
    row "groups 0" ~groups:0 2 2 [| 3 |];
    row "negative groups" ~groups:(-1) 2 2 [| 3 |];
    row "groups not dividing in_channels" ~groups:4 6 4 [| 3 |];
    row "groups not dividing out_channels" ~groups:3 6 4 [| 3 |];
  ]

let make_refuses g =
  raises_match Exn.invalid_arg (fun () ->
      Conv.make ~groups:g.groups ~key:(Nx.Rng.key 7) ~in_channels:g.in_channels
        ~out_channels:g.out_channels ~kernel_size:g.kernel_size Nx.float32)

let test_init () =
  let key = Nx.Rng.key 8 in
  let make =
    Conv.make ~key ~in_channels:3 ~out_channels:2 ~kernel_size:[| 2; 2 |]
      Nx.float32
  in
  let init =
    Conv.init ~key ~in_channels:3 ~out_channels:2 ~kernel_size:[| 2; 2 |]
  in
  equal ~msg:"w" tensor (bits make.w) (bits init.w);
  equal ~msg:"b" tensor
    (bits (require_some make.b))
    (bits (require_some init.b))

(* Applying *)

type case = {
  dtype : dtype;
  batch : int;
  spatial : int array;
  kernel : int array;
  stride : int array;
  dilation : int array;
  padding : (int * int) array;
  groups : int;
  per_group : int * int;
  bias : bool;
  specials : bool;
  seed : int;
}

let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a))

let pairs a =
  let pair (l, h) = Printf.sprintf "(%d, %d)" l h in
  String.concat "; " (Array.to_list (Array.map pair a))

let pp_case ppf c =
  Format.fprintf ppf
    "%a, batch %d, spatial [|%s|], kernel [|%s|], stride [|%s|], dilation \
     [|%s|], padding [|%s|], groups %d, (in, out) per group (%d, %d), bias %b, \
     specials %b, seed %d"
    pp_dtype c.dtype c.batch (ints c.spatial) (ints c.kernel) (ints c.stride)
    (ints c.dilation) (pairs c.padding) c.groups (fst c.per_group)
    (snd c.per_group) c.bias c.specials c.seed

(* Padding pairs from -1 to 2 a side: hi below lo, a negative hi that cuts. *)
let case =
  let open Gen in
  let* k = int_range 1 3 in
  let axes g = array ~size:(constant k) g in
  let+ dtype = of_list ~pp:pp_dtype float_dtypes
  and+ batch = frequency [ (1, constant 0); (3, int_range 1 2) ]
  and+ spatial = axes (int_range 0 5)
  and+ kernel = axes (int_range 1 3)
  and+ stride = axes (int_range 1 2)
  and+ dilation = axes (int_range 1 2)
  and+ padding = axes (pair (int_range 0 2) (int_range (-1) 2))
  and+ groups = int_range 1 3
  and+ per_group = pair (int_range 1 3) (int_range 1 2)
  and+ bias = bool
  and+ specials = bool
  and+ seed = int_range 0 1_000_000 in
  {
    dtype;
    batch;
    spatial;
    kernel;
    stride;
    dilation;
    padding;
    groups;
    per_group;
    bias;
    specials;
    seed;
  }

(* A kernel of 3 over 1 element: no window fits. *)
let no_window =
  {
    dtype = Dtype Nx.float32;
    batch = 1;
    spatial = [| 1 |];
    kernel = [| 3 |];
    stride = [| 1 |];
    dilation = [| 1 |];
    padding = [| (0, 0) |];
    groups = 1;
    per_group = (1, 1);
    bias = true;
    specials = false;
    seed = 0;
  }

(* Multiples of 1/4 scaled by 2^-4 to 2^4, which the narrow dtypes round; with
   [specials], one element in eight NaN, an infinity or -0. *)
let element rng ~specials =
  if specials && Random.State.int rng 8 = 0 then
    [| Float.nan; Float.infinity; Float.neg_infinity; -0. |].(Random.State.int
                                                                rng 4)
  else
    Float.ldexp
      (Float.of_int (Random.State.int rng 9 - 4) /. 4.)
      (Random.State.int rng 9 - 4)

let draw (type b) (dt : (float, b) Nx.dtype) rng ~specials shape =
  Nx.cast dt (Nx.init Nx.float64 shape (fun _ -> element rng ~specials))

(* Per dtype, a case whose result holds a finite non-zero element, so that the
   comparison reads more than NaN and zeros. *)
let cover_finite dt values =
  let read v = Float.is_finite v && v <> 0. in
  List.iter
    (fun d ->
      cover
        (dtype_name d ^ ": a finite non-zero element")
        (dtype_name d = Nx.Dtype.name dt && Array.exists read values))
    float_dtypes

let law_apply_at (type b) (dt : (float, b) Nx.dtype) c =
  let k = Array.length c.spatial in
  let cg, og = c.per_group in
  cover "1 spatial axis" (k = 1);
  cover "2 spatial axes" (k = 2);
  cover "3 spatial axes" (k = 3);
  cover "groups" (c.groups > 1 && cg > 1);
  cover "depthwise" (c.groups > 1 && cg = 1);
  cover "stride 2" (Array.mem 2 c.stride);
  cover "dilation 2" (Array.mem 2 c.dilation);
  cover "hi below lo" (Array.exists (fun (lo, hi) -> hi < lo) c.padding);
  cover "a negative hi" (Array.exists (fun (_, hi) -> hi < 0) c.padding);
  cover "no bias" (not c.bias);
  cover "an empty batch" (c.batch = 0);
  cover "an empty spatial axis" (Array.mem 0 c.spatial);
  cover "NaN, infinities and -0." c.specials;
  let rng = Random.State.make [| c.seed |] in
  let draw shape = draw dt rng ~specials:c.specials shape in
  let o = c.groups * og in
  let x =
    draw (Array.concat [ [| c.batch |]; c.spatial; [| c.groups * cg |] ])
  in
  let w = draw (Array.concat [ [| o |]; c.kernel; [| cg |] ]) in
  let b = if c.bias then Some (draw [| o |]) else None in
  let y =
    Conv.apply ~stride:c.stride ~padding:c.padding ~dilation:c.dilation
      ~groups:c.groups { Conv.w; b } x
  in
  let expected =
    bits
      (Nx.conv ?init:b ~stride:c.stride ~padding:c.padding ~dilation:c.dilation
         ~groups:c.groups x w)
  in
  cover "no window fits" (Array.mem 0 (Array.sub (Nx.shape y) 1 k));
  cover_finite dt (snd expected);
  equal tensor expected (bits y)

let law_apply c = match c.dtype with Dtype dt -> law_apply_at dt c

(* y = 10 + the sum of each 2 x 2 window of 0 .. 8. *)
let test_window_sums () =
  let p =
    {
      Conv.w = floats [| 1; 2; 2; 1 |] [| 1.; 1.; 1.; 1. |];
      b = Some (floats [| 1 |] [| 10. |]);
    }
  in
  let x = floats [| 1; 3; 3; 1 |] (Array.init 9 Float.of_int) in
  equal tensor
    (bits (floats [| 1; 2; 2; 1 |] [| 18.; 22.; 30.; 34. |]))
    (bits (Conv.apply p x))

(* y(i) = x(i) - x(i + 1): the kernel [1; -1] read in order. *)
let test_not_flipped () =
  let p = { Conv.w = floats [| 1; 2; 1 |] [| 1.; -1. |]; b = None } in
  let x = floats [| 1; 4; 1 |] [| 1.; 3.; 6.; 10. |] in
  equal tensor
    (bits (floats [| 1; 3; 1 |] [| -2.; -3.; -4. |]))
    (bits (Conv.apply p x))

(* [big] is 2^p for p the significand's bits, where [big + 1] rounds to [big]:
   rounding the sum before adding the bias [-big] gives 0, one rounding gives
   1. *)
let bias_rounding_rows =
  [
    (Dtype Nx.float16, 2048.);
    (Dtype Nx.bfloat16, 256.);
    (Dtype Nx.float8_e4m3fn, 16.);
    (Dtype Nx.float8_e5m2, 8.);
    (Dtype Nx.float4_e2m1fn, 4.);
  ]

let bias_rounds_once_at (type b) (dt : (float, b) Nx.dtype) big =
  let narrow shape values = Nx.cast dt (floats shape values) in
  let p =
    {
      Conv.w = narrow [| 1; 1; 2 |] [| 1.; 1. |];
      b = Some (narrow [| 1 |] [| -.big |]);
    }
  in
  equal tensor
    (bits (floats [| 1; 1; 1 |] [| 1. |]))
    (bits (Conv.apply p (narrow [| 1; 1; 2 |] [| big; 1. |])))

let bias_rounds_once (d, big) =
  match d with Dtype dt -> bias_rounds_once_at dt big

let law_odd_extent (half, s) =
  let k = (2 * half) + 1 in
  cover "a filter of 1" (k = 1);
  cover "an axis shorter than the filter" (s < k);
  cover "an empty axis" (s = 0);
  let p = { Conv.w = Nx.zeros Nx.float32 [| 1; k; 1 |]; b = None } in
  let y =
    Conv.apply
      ~padding:[| (k / 2, k / 2) |]
      p
      (Nx.zeros Nx.float32 [| 1; s; 1 |])
  in
  equal int s (Nx.shape y).(1)

let law_ceil_padding (k, stride, dilation, s) =
  cover "a stride that does not divide the extent" (s mod stride <> 0);
  cover "an even filter" (k mod 2 = 0);
  cover "dilation 2" (dilation = 2);
  let p = { Conv.w = Nx.zeros Nx.float32 [| 1; k; 1 |]; b = None } in
  let padding =
    Nx.ceil_padding ~stride:[| stride |] ~dilation:[| dilation |]
      ~kernel_size:[| k |] [| s |]
  in
  let y =
    Conv.apply ~stride:[| stride |] ~dilation:[| dilation |] ~padding p
      (Nx.zeros Nx.float32 [| 1; s; 1 |])
  in
  equal int ((s + stride - 1) / stride) (Nx.shape y).(1)

(* golden/conv.npz holds PyTorch's convolutions of inputs, weights and biases
   that are multiples of 1/4, so that every sum is exact in float64 and each
   result is the exact one. Weights are in PyTorch's layout, [| out; in /
   groups; k… |], moved as the preamble's importer moves them. *)
let golden = lazy (Nx_io.load_npz "golden/conv.npz")

let entry name =
  match Nx_io.Archive.find name (Lazy.force golden) with
  | Some (Nx.P x) -> Nx.cast Nx.float64 x
  | None -> failwith ("golden/conv.npz has no " ^ name)

let moved name = Nx.moveaxis 1 (-1) (entry name)

(* Whisper's stem: two 3-tap convolutions padded by one a side, the second at
   stride 2, at an even extent and an odd one. A padding that depends on the
   stride splits them differently: the old `Same split was off by up to 20 at
   3000 frames and matched at 3001. *)
let whisper_stem frames =
  let e = Printf.sprintf "stem%d." frames in
  let c1 = { Conv.w = moved (e ^ "w1"); b = Some (entry (e ^ "b1")) } in
  let c2 = { Conv.w = moved (e ^ "w2"); b = Some (entry (e ^ "b2")) } in
  let padding = [| (1, 1) |] in
  equal ~msg:"conv1" tensor
    (bits (entry (e ^ "y1")))
    (bits (Conv.apply ~padding c1 (entry (e ^ "x"))));
  equal ~msg:"conv2" tensor
    (bits (entry (e ^ "y2")))
    (bits (Conv.apply ~stride:[| 2 |] ~padding c2 (entry (e ^ "y1"))))

let ok = function Ok () -> () | Error m -> fail m
let squares y = Nx.sum (Nx.mul y y)

let test_parameter_gradients () =
  let conv = Nx.Ptree.instantiate (module Conv) in
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float64 [| 2; 4; 4; 2 |] in
  let p =
    Conv.make ~key:(Nx.Rng.key 10) ~in_channels:2 ~out_channels:2
      ~kernel_size:[| 2; 2 |] Nx.float64
  in
  ok (Rune.check_grads conv (fun p -> squares (Conv.apply p x)) p);
  let grouped =
    Conv.make ~key:(Nx.Rng.key 11) ~groups:2 ~in_channels:2 ~out_channels:4
      ~kernel_size:[| 3; 3 |] Nx.float64
  in
  let strided p =
    squares
      (Conv.apply ~stride:[| 2; 2 |]
         ~padding:[| (1, 1); (0, 1) |]
         ~groups:2 p x)
  in
  ok (Rune.check_grads conv strided grouped)

let test_input_gradients () =
  let p =
    Conv.make ~key:(Nx.Rng.key 12) ~in_channels:2 ~out_channels:3
      ~kernel_size:[| 3 |] Nx.float64
  in
  let loss x =
    squares (Conv.apply ~padding:[| (1, 1) |] ~dilation:[| 2 |] p x)
  in
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 13) Nx.float64 [| 1; 7; 2 |] in
  ok (Rune.check_grads Nx.Ptree.tensor loss x)

type bad_input = {
  input : string;
  layer_groups : int;
  in_channels : int;
  kernel_size : int array;
  x : int array;
  stride : int array option;
}

let bad_inputs =
  let row input ?(layer_groups = 1) ?stride in_channels kernel_size x =
    { input; layer_groups; in_channels; kernel_size; x; stride }
  in
  [
    row "x of two axes" 2 [| 3 |] [| 4; 2 |];
    row "x of another rank than w" 2 [| 3; 3 |] [| 1; 4; 2 |];
    row "x's channels other than w's" 2 [| 3 |] [| 1; 4; 3 |];
    row "a stride below 1" ~stride:[| 0 |] 2 [| 3 |] [| 1; 4; 2 |];
    row "groups other than the layer's" ~layer_groups:2 4 [| 3 |] [| 1; 5; 4 |];
  ]

let apply_refuses r =
  let p =
    Conv.make ~groups:r.layer_groups ~key:(Nx.Rng.key 14)
      ~in_channels:r.in_channels ~out_channels:r.in_channels
      ~kernel_size:r.kernel_size Nx.float32
  in
  raises_match (Exn.invalid_arg ~sub:"Nx.conv: ") (fun () ->
      Conv.apply ?stride:r.stride p (Nx.zeros Nx.float32 r.x))

(* Structure *)

let walk_rows =
  [
    ("with a bias", true, [ "w: a leaf"; "b: Some"; "b: a leaf" ]);
    ("without a bias", false, [ "w: a leaf"; "b: None" ]);
  ]

let walk_visits (_, bias, expected) =
  let p =
    Conv.make ~bias ~key:(Nx.Rng.key 15) ~in_channels:2 ~out_channels:2
      ~kernel_size:[| 3 |] Nx.float32
  in
  equal (list string) expected (visits p)

let test_cast () =
  let p =
    Conv.make ~key:(Nx.Rng.key 16) ~in_channels:2 ~out_channels:2
      ~kernel_size:[| 3 |] Nx.float32
  in
  let q = Nx.Ptree.cast (module Conv) Nx.bfloat16 p in
  equal ~msg:"w" tensor (bits (Nx.cast Nx.bfloat16 p.w)) (bits q.w);
  equal ~msg:"b" tensor
    (bits (Nx.cast Nx.bfloat16 (require_some p.b)))
    (bits (require_some q.b))

let () =
  exit
    (run "kaun conv"
       [
         group "t"
           [
             test "w is [| out; k…; in / groups |] and b is [| out |]"
               test_shapes;
             test "b is None for a layer built without a bias" test_no_bias;
           ];
         group "make"
           [
             test
               "draws Glorot-uniform w from the first key Nx.Rng.split makes \
                of key, and b from the second"
               test_make_draws;
             test "defaults to a zero bias" test_zero_bias;
             test "applies w_init and bias_init with the fans of one group"
               test_make_initializers;
             test "lies where its key lies" test_make_placement;
             cases "raises Invalid_argument on a bad geometry"
               ~id:(fun g -> g.id)
               bad_geometries make_refuses;
           ];
         group "init" [ test "is make at float32" test_init ];
         group "apply"
           [
             prop ~count:600 ~examples:[ no_window ]
               "is Nx.conv of x by w with b as its init, bit for bit"
               (Gen.with_pp pp_case case) law_apply;
             test "sums each window, then adds the bias" test_window_sums;
             test "does not flip the kernel" test_not_flipped;
             cases "adds the bias before the one rounding"
               ~id:(fun (d, _) -> dtype_name d)
               bias_rounding_rows bias_rounds_once;
             prop "keeps an extent under (k / 2, k / 2) for an odd k"
               Gen.(pair (int_range 0 3) (int_range 0 6))
               law_odd_extent;
             prop "keeps ⌈s / stride⌉ windows under Nx.ceil_padding"
               Gen.(
                 quad (int_range 1 4) (int_range 1 3) (int_range 1 2)
                   (int_range 1 9))
               law_ceil_padding;
             cases "reads Whisper's stem from PyTorch's layout"
               ~id:(Printf.sprintf "%d frames")
               [ 3000; 3001 ] whisper_stem;
             test "differentiates in the parameters" test_parameter_gradients;
             test "differentiates in the input" test_input_gradients;
             cases "raises as Nx.conv does"
               ~id:(fun r -> r.input)
               bad_inputs apply_refuses;
           ];
         group "walk"
           [
             cases "visits w, then b, reporting whether b is present"
               ~id:(fun (id, _, _) -> id)
               walk_rows walk_visits;
             test "Nx.Ptree.cast converts each parameter as Nx.cast" test_cast;
           ];
       ])
