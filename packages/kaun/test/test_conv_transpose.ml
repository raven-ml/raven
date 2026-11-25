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

let software =
  lazy (Nx.devices [ Result.get_ok (Rig.software "kaun-conv-transpose") ])

let visits p =
  List.map
    (Format.asprintf "%a" Nx.Ptree.pp_visit)
    (Nx.Ptree.visits (Nx.Ptree.instantiate (module Conv_transpose)) p)

(* Types *)

let test_shapes () =
  let p =
    Conv_transpose.make ~key:(Nx.Rng.key 1) ~in_channels:4 ~out_channels:2
      ~kernel_size:[| 2; 3 |] Nx.float32
  in
  equal ~msg:"w" (array int) [| 4; 2; 3; 2 |] (Nx.shape p.w);
  equal ~msg:"b" (array int) [| 2 |] (Nx.shape (require_some p.b))

let test_no_bias () =
  let p =
    Conv_transpose.make ~bias:false ~key:(Nx.Rng.key 2) ~in_channels:2
      ~out_channels:3 ~kernel_size:[| 2 |] Nx.float32
  in
  is_none ~pp:Nx.pp p.b

(* Constructors *)

(* Fans of 4 inputs and 2 outputs over 2 x 3 taps: 4 * 6 and 2 * 6. *)
let test_make_draws () =
  let key = Nx.Rng.key 3 in
  let p =
    Conv_transpose.make ~bias_init:Init.glorot_uniform ~key ~in_channels:4
      ~out_channels:2 ~kernel_size:[| 2; 3 |] Nx.float32
  in
  let keys = Nx.Rng.split key in
  let glorot key shape =
    Init.glorot_uniform key ~fan_in:24 ~fan_out:12 Nx.float32 shape
  in
  equal ~msg:"w" tensor (bits (glorot keys.(0) [| 4; 2; 3; 2 |])) (bits p.w);
  equal ~msg:"b" tensor
    (bits (glorot keys.(1) [| 2 |]))
    (bits (require_some p.b))

let test_zero_bias () =
  let p =
    Conv_transpose.make ~key:(Nx.Rng.key 3) ~in_channels:4 ~out_channels:2
      ~kernel_size:[| 3 |] Nx.float32
  in
  equal tensor (bits (Nx.zeros Nx.float32 [| 2 |])) (bits (require_some p.b))

(* One output reads 3 channels at each of 4 taps; one input feeds 5 outputs at
   each. *)
let test_make_initializers () =
  let fans = ref [] in
  let init name v key ~fan_in ~fan_out dtype shape =
    fans := (name, fan_in, fan_out) :: !fans;
    Init.constant v key ~fan_in ~fan_out dtype shape
  in
  let p =
    Conv_transpose.make ~w_init:(init "w" 2.) ~bias_init:(init "b" 1.)
      ~key:(Nx.Rng.key 4) ~in_channels:3 ~out_channels:5 ~kernel_size:[| 2; 2 |]
      Nx.float32
  in
  equal ~msg:"fans"
    (list (triple string int int))
    [ ("w", 12, 20); ("b", 12, 20) ]
    (List.rev !fans);
  equal ~msg:"w" tensor
    (bits (Nx.full Nx.float32 [| 3; 2; 2; 5 |] 2.))
    (bits p.w);
  equal ~msg:"b" tensor
    (bits (Nx.full Nx.float32 [| 5 |] 1.))
    (bits (require_some p.b))

let test_make_placement () =
  let (module D : Nx.Devices) = Lazy.force software in
  let p =
    Conv_transpose.make
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
}

let bad_geometries =
  let row id in_channels out_channels kernel_size =
    { id; in_channels; out_channels; kernel_size }
  in
  [
    row "an empty kernel_size" 2 2 [||];
    row "no in_channels" 0 4 [| 3; 3 |];
    row "no out_channels" 2 0 [| 3 |];
    row "a kernel extent of 0" 2 2 [| 3; 0 |];
    row "a negative kernel extent" 2 2 [| -1 |];
  ]

let make_refuses g =
  raises_match Exn.invalid_arg (fun () ->
      Conv_transpose.make ~key:(Nx.Rng.key 7) ~in_channels:g.in_channels
        ~out_channels:g.out_channels ~kernel_size:g.kernel_size Nx.float32)

let test_init () =
  let key = Nx.Rng.key 8 in
  let make =
    Conv_transpose.make ~key ~in_channels:3 ~out_channels:2
      ~kernel_size:[| 2; 2 |] Nx.float32
  in
  let init =
    Conv_transpose.init ~key ~in_channels:3 ~out_channels:2
      ~kernel_size:[| 2; 2 |]
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
  channels : int * int;
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
     [|%s|], padding [|%s|], (in, out) channels (%d, %d), bias %b, specials \
     %b, seed %d"
    pp_dtype c.dtype c.batch (ints c.spatial) (ints c.kernel) (ints c.stride)
    (ints c.dilation) (pairs c.padding) (fst c.channels) (snd c.channels) c.bias
    c.specials c.seed

(* Padding pairs from -1 to 2 a side: hi below lo, a negative hi. *)
let case =
  let open Gen in
  let* k = int_range 1 3 in
  let axes g = array ~size:(constant k) g in
  let+ dtype = of_list ~pp:pp_dtype float_dtypes
  and+ batch = frequency [ (1, constant 0); (3, int_range 1 2) ]
  and+ spatial = axes (int_range 0 4)
  and+ kernel = axes (int_range 1 3)
  and+ stride = axes (int_range 1 2)
  and+ dilation = axes (int_range 1 2)
  and+ padding = axes (pair (int_range 0 2) (int_range (-1) 2))
  and+ channels = pair (int_range 1 3) (int_range 1 3)
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
    channels;
    bias;
    specials;
    seed;
  }

(* A one-tap kernel over one element padded (2, 2): s = 1 - 4 is negative, and
   no input has a convolution of x's shape. *)
let refused =
  {
    dtype = Dtype Nx.float32;
    batch = 1;
    spatial = [| 1 |];
    kernel = [| 1 |];
    stride = [| 1 |];
    dilation = [| 1 |];
    padding = [| (2, 2) |];
    channels = (1, 1);
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

(* A call's result, or the message of the [Invalid_argument] it raises. *)
let outcome f =
  match f () with y -> Ok (bits y) | exception Invalid_argument m -> Error m

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
  let inputs, outputs = c.channels in
  cover "1 spatial axis" (k = 1);
  cover "2 spatial axes" (k = 2);
  cover "3 spatial axes" (k = 3);
  cover "stride 2" (Array.mem 2 c.stride);
  cover "dilation 2" (Array.mem 2 c.dilation);
  cover "hi below lo" (Array.exists (fun (lo, hi) -> hi < lo) c.padding);
  cover "a negative hi" (Array.exists (fun (_, hi) -> hi < 0) c.padding);
  cover "a one-element axis" (Array.mem 1 c.spatial);
  cover "no bias" (not c.bias);
  cover "an empty batch" (c.batch = 0);
  cover "an empty spatial axis" (Array.mem 0 c.spatial);
  cover "NaN, infinities and -0." c.specials;
  let rng = Random.State.make [| c.seed |] in
  let draw shape = draw dt rng ~specials:c.specials shape in
  let x = draw (Array.concat [ [| c.batch |]; c.spatial; [| inputs |] ]) in
  let w = draw (Array.concat [ [| inputs |]; c.kernel; [| outputs |] ]) in
  let b = if c.bias then Some (draw [| outputs |]) else None in
  let expected =
    outcome (fun () ->
        Nx.conv_transpose ?init:b ~stride:c.stride ~padding:c.padding
          ~dilation:c.dilation x w)
  in
  cover "a result" (Result.is_ok expected);
  cover "a refusal" (Result.is_error expected);
  cover "an empty result axis"
    (match expected with
    | Ok (shape, _) -> Array.mem 0 (Array.sub shape 1 k)
    | Error _ -> false);
  cover_finite dt (match expected with Ok (_, v) -> v | Error _ -> [||]);
  equal (result tensor string) expected
    (outcome (fun () ->
         Conv_transpose.apply ~stride:c.stride ~padding:c.padding
           ~dilation:c.dilation { Conv_transpose.w; b } x))

let law_apply c = match c.dtype with Dtype dt -> law_apply_at dt c

type extent = {
  claim : string;
  t : int;
  stride : int;
  kernel : int;
  dilation : int;
  padding : int * int;
  s : int;
}

(* s = (t - 1) stride + e - lo - hi, e = dilation (kernel - 1) + 1. *)
let extents =
  let row claim ~t ~stride ~kernel ~dilation ~padding s =
    { claim; t; stride; kernel; dilation; padding; s }
  in
  [
    row "unit stride" ~t:4 ~stride:1 ~kernel:3 ~dilation:1 ~padding:(0, 0) 6;
    row "stride 2" ~t:3 ~stride:2 ~kernel:2 ~dilation:1 ~padding:(0, 0) 6;
    row "dilation 2" ~t:2 ~stride:1 ~kernel:3 ~dilation:2 ~padding:(0, 0) 6;
    row "padding (1, 1)" ~t:3 ~stride:2 ~kernel:3 ~dilation:1 ~padding:(1, 1) 5;
    row "an hi below lo lengthens" ~t:3 ~stride:2 ~kernel:3 ~dilation:1
      ~padding:(1, 0) 6;
    row "a negative hi" ~t:3 ~stride:2 ~kernel:3 ~dilation:1 ~padding:(1, -1) 7;
  ]

let extent_is e =
  let p =
    { Conv_transpose.w = Nx.zeros Nx.float32 [| 1; e.kernel; 1 |]; b = None }
  in
  let y =
    Conv_transpose.apply ~stride:[| e.stride |] ~dilation:[| e.dilation |]
      ~padding:[| e.padding |] p
      (Nx.zeros Nx.float32 [| 1; e.t; 1 |])
  in
  equal int e.s (Nx.shape y).(1)

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
      Conv_transpose.w = narrow [| 2; 1; 1 |] [| 1.; 1. |];
      b = Some (narrow [| 1 |] [| -.big |]);
    }
  in
  equal tensor
    (bits (floats [| 1; 1; 1 |] [| 1. |]))
    (bits (Conv_transpose.apply p (narrow [| 1; 1; 2 |] [| big; 1. |])))

let bias_rounds_once (d, big) =
  match d with Dtype dt -> bias_rounds_once_at dt big

(* golden/conv.npz holds PyTorch's transposed convolutions of inputs, weights
   and biases that are multiples of 1/4, so that every sum is exact in float64
   and each result is the exact one. Weights are in PyTorch's layout, [| in;
   out; k… |], moved as the preamble says. *)
let golden = lazy (Nx_io.load_npz "golden/conv.npz")

let entry name =
  match Nx_io.Archive.find name (Lazy.force golden) with
  | Some (Nx.P x) -> Nx.cast Nx.float64 x
  | None -> failwith ("golden/conv.npz has no " ^ name)

let moved name = Nx.moveaxis 1 (-1) (entry name)

(* SAM 2's upscaling: a 2 x 2 transpose at stride 2. *)
let test_upscale () =
  let p =
    { Conv_transpose.w = moved "upscale.w"; b = Some (entry "upscale.b") }
  in
  equal tensor
    (bits (entry "upscale.y"))
    (bits (Conv_transpose.apply ~stride:[| 2; 2 |] p (entry "upscale.x")))

(* PyTorch's padding 1 and output padding 1, at kernel 3 and stride 2. *)
let test_output_padding () =
  let p =
    { Conv_transpose.w = moved "outpad.w"; b = Some (entry "outpad.b") }
  in
  equal tensor
    (bits (entry "outpad.y"))
    (bits
       (Conv_transpose.apply ~stride:[| 2 |]
          ~padding:[| (1, 1 - 1) |]
          p (entry "outpad.x")))

let ok = function Ok () -> () | Error m -> fail m
let squares y = Nx.sum (Nx.mul y y)

let test_parameter_gradients () =
  let transpose = Nx.Ptree.instantiate (module Conv_transpose) in
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float64 [| 2; 3; 3; 2 |] in
  let p =
    Conv_transpose.make ~key:(Nx.Rng.key 10) ~in_channels:2 ~out_channels:3
      ~kernel_size:[| 2; 2 |] Nx.float64
  in
  let loss p =
    squares
      (Conv_transpose.apply ~stride:[| 2; 1 |]
         ~padding:[| (0, 1); (1, 0) |]
         p x)
  in
  ok (Rune.check_grads transpose loss p)

let test_input_gradients () =
  let p =
    Conv_transpose.make ~key:(Nx.Rng.key 11) ~in_channels:2 ~out_channels:3
      ~kernel_size:[| 3 |] Nx.float64
  in
  let loss x =
    squares
      (Conv_transpose.apply ~stride:[| 2 |]
         ~padding:[| (1, 0) |]
         ~dilation:[| 2 |] p x)
  in
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 12) Nx.float64 [| 1; 4; 2 |] in
  ok (Rune.check_grads Nx.Ptree.tensor loss x)

type bad_input = {
  input : string;
  kernel_size : int array;
  x : int array;
  stride : int array option;
}

let bad_inputs =
  let row input ?stride kernel_size x = { input; kernel_size; x; stride } in
  [
    row "x of two axes" [| 3 |] [| 4; 2 |];
    row "x of another rank than w" [| 3; 3 |] [| 1; 4; 2 |];
    row "x's channels other than w's" [| 3 |] [| 1; 4; 3 |];
    row "a stride below 1" ~stride:[| 0 |] [| 3 |] [| 1; 4; 2 |];
  ]

let apply_refuses r =
  let p =
    Conv_transpose.make ~key:(Nx.Rng.key 13) ~in_channels:2 ~out_channels:2
      ~kernel_size:r.kernel_size Nx.float32
  in
  raises_match (Exn.invalid_arg ~sub:"Nx.conv_transpose: ") (fun () ->
      Conv_transpose.apply ?stride:r.stride p (Nx.zeros Nx.float32 r.x))

(* Structure *)

let walk_rows =
  [
    ("with a bias", true, [ "w: a leaf"; "b: Some"; "b: a leaf" ]);
    ("without a bias", false, [ "w: a leaf"; "b: None" ]);
  ]

let walk_visits (_, bias, expected) =
  let p =
    Conv_transpose.make ~bias ~key:(Nx.Rng.key 14) ~in_channels:2
      ~out_channels:2 ~kernel_size:[| 3 |] Nx.float32
  in
  equal (list string) expected (visits p)

let test_cast () =
  let p =
    Conv_transpose.make ~key:(Nx.Rng.key 15) ~in_channels:2 ~out_channels:2
      ~kernel_size:[| 3 |] Nx.float32
  in
  let q = Nx.Ptree.cast (module Conv_transpose) Nx.bfloat16 p in
  equal ~msg:"w" tensor (bits (Nx.cast Nx.bfloat16 p.w)) (bits q.w);
  equal ~msg:"b" tensor
    (bits (Nx.cast Nx.bfloat16 (require_some p.b)))
    (bits (require_some q.b))

let () =
  exit
    (run "kaun conv transpose"
       [
         group "t"
           [
             test "w is [| in; k…; out |] and b is [| out |]" test_shapes;
             test "b is None for a layer built without a bias" test_no_bias;
           ];
         group "make"
           [
             test
               "draws Glorot-uniform w from the first key Nx.Rng.split makes \
                of key, and b from the second"
               test_make_draws;
             test "defaults to a zero bias" test_zero_bias;
             test
               "applies w_init and bias_init with the fans in · taps and out · \
                taps"
               test_make_initializers;
             test "lies where its key lies" test_make_placement;
             cases "raises Invalid_argument on a bad geometry"
               ~id:(fun g -> g.id)
               bad_geometries make_refuses;
           ];
         group "init" [ test "is make at float32" test_init ];
         group "apply"
           [
             prop ~count:600 ~examples:[ refused ]
               "is Nx.conv_transpose of x by w with b as its init, bit for bit"
               (Gen.with_pp pp_case case) law_apply;
             cases "gives extents s = (t - 1) stride + e - lo - hi"
               ~id:(fun e -> e.claim)
               extents extent_is;
             cases "adds the bias before the one rounding"
               ~id:(fun (d, _) -> dtype_name d)
               bias_rounding_rows bias_rounds_once;
             test "reads SAM 2's upscaling from PyTorch's layout" test_upscale;
             test "takes an output padding a beside p as the padding (p, p - a)"
               test_output_padding;
             test "differentiates in the parameters" test_parameter_gradients;
             test "differentiates in the input" test_input_gradients;
             cases "raises as Nx.conv_transpose does"
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
