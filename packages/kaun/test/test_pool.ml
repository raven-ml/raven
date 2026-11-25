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

let dtype_name (Dtype dt) = Nx.Dtype.name dt
let pp_dtype ppf d = Format.pp_print_string ppf (dtype_name d)

(* Each float dtype, with max's padding, its least value: -∞ where the format
   has an infinity, else its least finite value, -448 for E4M3 and -6 for E2M1
   (OCP 8-bit floating point and microscaling formats); and the values beyond
   integers [x] may hold: -0., NaN where the format has one (all but E2M1), and
   the infinities where it has them. *)
type format = { dtype : dtype; least : float; specials : float array }

let formats =
  let format dtype least specials = { dtype; least; specials } in
  let all = [| -0.; Float.nan; Float.infinity; Float.neg_infinity |] in
  let neg_inf = Float.neg_infinity in
  [
    format (Dtype Nx.float16) neg_inf all;
    format (Dtype Nx.bfloat16) neg_inf all;
    format (Dtype Nx.float32) neg_inf all;
    format (Dtype Nx.float64) neg_inf all;
    format (Dtype Nx.float8_e4m3fn) (-448.) [| -0.; Float.nan |];
    format (Dtype Nx.float8_e5m2) neg_inf all;
    format (Dtype Nx.float4_e2m1fn) (-6.) [| -0. |];
  ]

(* C-order index arithmetic over flat arrays. *)
let unflat shape f =
  let idx = Array.make (Array.length shape) 0 in
  let r = ref f in
  for d = Array.length shape - 1 downto 0 do
    idx.(d) <- !r mod shape.(d);
    r := !r / shape.(d)
  done;
  idx

let flat shape idx =
  let f = ref 0 in
  Array.iteri (fun d i -> f := (!f * shape.(d)) + i) idx;
  !f

(* The number of windows of extent [w] along an axis of extent [s] padded by
   [(lo, hi)]: [(s + lo + hi - w) / stride + 1], none where the padded axis is
   shorter. *)
let windows_along ~s ~lo ~hi ~w ~stride =
  let len = s + lo + hi in
  if len < w then 0 else ((len - w) / stride) + 1

(* The preamble's windows, placed as Nx.conv places a kernel: the window at [(b,
   t…, c)] reads [x[b, t · stride + q - lo…, c]] at its offsets [q…], [None]
   where that lies in the padding. *)
let windows ~window ~stride ~padding x =
  let xs = Nx.shape x in
  let k = Array.length window in
  let spatial = Array.sub xs 1 k in
  let extent d =
    let lo, hi = padding.(d) in
    windows_along ~s:spatial.(d) ~lo ~hi ~w:window.(d) ~stride:stride.(d)
  in
  let out =
    Array.concat [ [| xs.(0) |]; Array.init k extent; [| xs.(k + 1) |] ]
  in
  let xv = Nx.to_array (Nx.cast Nx.float64 x) in
  let taps = Array.fold_left ( * ) 1 window in
  let elements f =
    let idx = unflat out f in
    List.init taps (fun t ->
        let q = unflat window t in
        let at =
          Array.init k (fun d ->
              (idx.(d + 1) * stride.(d)) + q.(d) - fst padding.(d))
        in
        if not (Array.for_all2 (fun a s -> a >= 0 && a < s) at spatial) then
          None
        else
          let at = Array.concat [ [| idx.(0) |]; at; [| idx.(k + 1) |] ] in
          Some xv.(flat xs at))
  in
  (out, Array.init (Array.fold_left ( * ) 1 out) elements)

type case = {
  format : format;
  batch : int;
  spatial : int array;
  channels : int;
  window : int array;
  stride : int array option;
  padding : (int * int) array option;
  specials : bool;
  seed : int;
}

let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a))

let pairs a =
  let pair (l, h) = Printf.sprintf "(%d, %d)" l h in
  String.concat "; " (Array.to_list (Array.map pair a))

let pp_case ppf c =
  let opt f = function None -> "default" | Some a -> "[|" ^ f a ^ "|]" in
  Format.fprintf ppf
    "%a, batch %d, spatial [|%s|], channels %d, window [|%s|], stride %s, \
     padding %s, specials %b, seed %d"
    pp_dtype c.format.dtype c.batch (ints c.spatial) c.channels (ints c.window)
    (opt ints c.stride) (opt pairs c.padding) c.specials c.seed

let case =
  let open Gen in
  let* k = int_range 1 3 in
  let axes g = array ~size:(constant k) g in
  let+ format = of_list formats
  and+ batch = frequency [ (1, constant 0); (3, int_range 1 2) ]
  and+ spatial = axes (int_range 0 5)
  and+ channels = int_range 1 3
  and+ window = axes (int_range 1 3)
  and+ stride = option (axes (int_range 1 3))
  and+ padding = option (axes (pair (int_range 0 2) (int_range 0 2)))
  and+ specials = bool
  and+ seed = int_range 0 1_000_000 in
  { format; batch; spatial; channels; window; stride; padding; specials; seed }

(* Integers from -3 to 3, exact in every float dtype, so each window's sum is
   exact in a float32 accumulator; with [specials], one element in eight drawn
   from the format's [specials]. *)
let element rng (f : format) ~specials =
  if specials && Random.State.int rng 8 = 0 then
    f.specials.(Random.State.int rng (Array.length f.specials))
  else Float.of_int (Random.State.int rng 7 - 3)

(* An empty axis padded (1, 1) under a window of 2: one window, in padding
   alone. *)
let empty_axis =
  {
    format =
      { dtype = Dtype Nx.float32; least = Float.neg_infinity; specials = [||] };
    batch = 1;
    spatial = [| 0 |];
    channels = 1;
    window = [| 2 |];
    stride = None;
    padding = Some [| (1, 1) |];
    specials = false;
    seed = 0;
  }

let stride c = Option.value c.stride ~default:c.window

let padding c =
  Option.value c.padding ~default:(Array.make (Array.length c.window) (0, 0))

(* [Pool] refuses an input where no window fits. *)
let fits c =
  Array.for_all2
    (fun s (w, (lo, hi)) -> s + lo + hi >= w)
    c.spatial
    (Array.combine c.window (padding c))

(* An axis whose [hi] keeps a window that meets [x] and that no window without
   [hi] has. *)
let partial_window c =
  let stride = stride c in
  Array.exists Fun.id
    (Array.mapi
       (fun d (lo, hi) ->
         let s = c.spatial.(d) and w = c.window.(d) and stride = stride.(d) in
         let n = windows_along ~s ~lo ~hi ~w ~stride in
         n > windows_along ~s ~lo ~hi:0 ~w ~stride
         && ((n - 1) * stride) - lo < s)
       (padding c))

let cover_case c =
  let k = Array.length c.window in
  cover "1 spatial axis" (k = 1);
  cover "2 spatial axes" (k = 2);
  cover "3 spatial axes" (k = 3);
  cover "stride defaults to the window" (Option.is_none c.stride);
  cover "overlapping windows"
    (Array.exists2 (fun s w -> s < w) (stride c) c.window);
  cover "padding defaults to none" (Option.is_none c.padding);
  cover "padding" (Array.exists (fun (l, h) -> l > 0 || h > 0) (padding c));
  cover "a last, partial window" (partial_window c);
  cover "an empty batch" (c.batch = 0);
  cover "an empty spatial axis" (Array.mem 0 c.spatial);
  cover "NaN, infinities and -0." c.specials

(* Per dtype, a case whose result holds a finite non-zero element of [x]'s
   windows, so that the comparison reads more than NaN, padding and zeros. *)
let cover_finite c result =
  let read v = Float.is_finite v && v <> 0. && v <> c.format.least in
  List.iter
    (fun f ->
      let name = dtype_name f.dtype in
      cover
        (name ^ ": a finite non-zero element")
        (name = dtype_name c.format.dtype && Array.exists read result))
    formats

let draw_x (type b) (dt : (float, b) Nx.dtype) c =
  let rng = Random.State.make [| c.seed |] in
  let shape = Array.concat [ [| c.batch |]; c.spatial; [| c.channels |] ] in
  Nx.cast dt
    (Nx.init Nx.float64 shape (fun _ ->
         element rng c.format ~specials:c.specials))

(* max *)

let law_max_at (type b) (dt : (float, b) Nx.dtype) c =
  let x = draw_x dt c in
  let shape, windows =
    windows ~window:c.window ~stride:(stride c) ~padding:(padding c) x
  in
  cover "a window in padding alone"
    (Array.exists (List.for_all Option.is_none) windows);
  (* pool.mli: max is Nx.max over each window's elements, as Nx.maximum orders
     them: a NaN gives NaN, and -0. orders below +0., as Float.max. *)
  let greatest w =
    List.fold_left
      (fun m v -> Float.max m (Option.value v ~default:c.format.least))
      Float.neg_infinity w
  in
  let expected = Array.map greatest windows in
  cover_finite c expected;
  equal tensor (shape, expected)
    (bits (Pool.max ~window:c.window ?stride:c.stride ?padding:c.padding x))

let law_max c =
  assume (fits c);
  cover_case c;
  match c.format.dtype with Dtype dt -> law_max_at dt c

(* x = [-3; -2] padded (2, 1), windows of 2 at stride 1: the first lies in
   padding alone, the last meets x. *)
let max_padding f =
  match f.dtype with
  | Dtype dt ->
      let x = Nx.cast dt (floats [| 1; 2; 1 |] [| -3.; -2. |]) in
      equal tensor
        ([| 1; 4; 1 |], [| f.least; -3.; -2.; -2. |])
        (bits
           (Pool.max ~window:[| 2 |] ~stride:[| 1 |] ~padding:[| (2, 1) |] x))

let ok = function Ok () -> () | Error m -> fail m
let squares y = Nx.sum (Nx.mul y y)

let test_max_differentiable () =
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float64 [| 1; 4; 4; 2 |] in
  let loss x =
    squares
      (Pool.max ~window:[| 3; 2 |] ~stride:[| 2; 1 |]
         ~padding:[| (1, 1); (0, 1) |]
         x)
  in
  ok (Rune.check_grads Nx.Ptree.tensor loss x)

type refusal = {
  input : string;
  window : int array;
  stride : int array option;
  padding : (int * int) array option;
  x : int array;
}

let refusals =
  let row input ?stride ?padding ?(x = [| 1; 2; 5; 1 |]) window =
    { input; window; stride; padding; x }
  in
  [
    row "an empty window" [||];
    row "x without K + 2 axes" ~x:[| 2; 5; 1 |] [| 1; 1 |];
    row "a stride of other than K entries" ~stride:[| 1 |] [| 2; 2 |];
    row "a padding of other than K entries" ~padding:[| (0, 0) |] [| 2; 2 |];
    row "a window entry below 1" [| 0; 2 |];
    row "a stride entry below 1" ~stride:[| 1; 0 |] [| 2; 2 |];
    row "a negative padding" ~padding:[| (0, 0); (-1, 0) |] [| 1; 1 |];
    row "no window fits" [| 3; 3 |];
  ]

let max_refuses r =
  raises_match Exn.invalid_arg (fun () ->
      Pool.max ~window:r.window ?stride:r.stride ?padding:r.padding
        (Nx.zeros Nx.float32 r.x))

(* avg *)

let law_avg_at (type b) (dt : (float, b) Nx.dtype) c =
  let x = draw_x dt c in
  let shape, windows =
    windows ~window:c.window ~stride:(stride c) ~padding:(padding c) x
  in
  let size = Float.of_int (Array.fold_left ( * ) 1 c.window) in
  let mean w =
    List.fold_left (fun s v -> s +. Option.value v ~default:0.) 0. w /. size
  in
  (* pool.mli: avg is Nx.mean over each window's elements, which adds them from
     +0 and rounds once. The sum is exact, so the quotient rounded to float64,
     then to [dt], is the quotient rounded once to [dt]: float64 holds more than
     twice [dt]'s significand bits. *)
  let expected =
    Nx.cast dt (Nx.create Nx.float64 shape (Array.map mean windows))
  in
  cover_finite c (snd (bits expected));
  equal tensor (bits expected)
    (bits (Pool.avg ~window:c.window ?stride:c.stride ?padding:c.padding x))

let law_avg c =
  assume (fits c);
  cover_case c;
  match c.format.dtype with Dtype dt -> law_avg_at dt c

(* Windows of 2 over [2; 4; 6; 8; 10] padded (0, 1): the last window holds 10
   and one zero of padding. *)
let test_mean_of_elements () =
  let x = floats [| 1; 5; 1 |] [| 2.; 4.; 6.; 8.; 10. |] in
  let avg x = Pool.avg ~window:[| 2 |] ~padding:[| (0, 1) |] x in
  equal tensor
    ([| 1; 3; 1 |], [| 3.; 7.; 10. |])
    (bits (Nx.div (avg x) (avg (Nx.ones_like x))))

let test_avg_differentiable () =
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 2) Nx.float64 [| 1; 4; 4; 2 |] in
  let loss x =
    squares (Pool.avg ~window:[| 2; 2 |] ~padding:[| (1, 0); (0, 1) |] x)
  in
  ok (Rune.check_grads Nx.Ptree.tensor loss x)

let avg_refuses r =
  raises_match Exn.invalid_arg (fun () ->
      Pool.avg ~window:r.window ?stride:r.stride ?padding:r.padding
        (Nx.zeros Nx.float32 r.x))

(* Windows *)

let law_ceil_padding (w, stride, s) =
  cover "a stride that does not divide the extent" (s mod stride <> 0);
  cover "a window longer than the extent" (w > s);
  let padding =
    Nx.ceil_padding ~stride:[| stride |] ~kernel_size:[| w |] [| s |]
  in
  let y =
    Pool.max ~window:[| w |] ~stride:[| stride |] ~padding
      (Nx.zeros Nx.float32 [| 1; s; 1 |])
  in
  equal int ((s + stride - 1) / stride) (Nx.shape y).(1)

let () =
  let case = Gen.with_pp pp_case case in
  exit
    (run "kaun pool"
       [
         group "windows"
           [
             prop "keep ⌈s / stride⌉ windows under Nx.ceil_padding"
               Gen.(triple (int_range 1 4) (int_range 1 3) (int_range 1 6))
               law_ceil_padding;
           ];
         group "max"
           [
             prop ~count:500 ~examples:[ empty_axis ]
               "is the greatest element of each window" case law_max;
             cases "pads with -∞, or the dtype's least finite value"
               ~id:(fun f -> dtype_name f.dtype)
               formats max_padding;
             test "differentiates through Rune" test_max_differentiable;
             cases "raises Invalid_argument on a bad window or input"
               ~id:(fun r -> r.input)
               refusals max_refuses;
           ];
         group "avg"
           [
             prop ~count:500 ~examples:[ empty_axis ]
               "is each window's sum divided by the product of window, padding \
                counted as zeros"
               case law_avg;
             test
               "divided by avg ~padding (ones_like x) is the mean of x's \
                elements alone"
               test_mean_of_elements;
             test "differentiates through Rune" test_avg_differentiable;
             cases "raises as max" ~id:(fun r -> r.input) refusals avg_refuses;
           ];
       ])
