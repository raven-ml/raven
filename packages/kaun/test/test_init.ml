(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Init = Kaun.Init

let key () = Nx.Rng.key 7
let values = Nx.to_array
let dtype_name dt = Format.asprintf "%a" Nx.Dtype.pp dt

let initializers () =
  [
    ("zeros", Init.zeros);
    ("ones", Init.ones);
    ("constant", Init.constant 3.5);
    ("uniform", Init.uniform ~scale:2.);
    ("normal", Init.normal ~stddev:2.);
    ( "variance_scaling `Normal",
      Init.variance_scaling ~scale:1. ~mode:`Fan_in ~distribution:`Normal );
    ( "variance_scaling `Truncated_normal",
      Init.variance_scaling ~scale:1. ~mode:`Fan_in
        ~distribution:`Truncated_normal );
    ( "variance_scaling `Uniform",
      Init.variance_scaling ~scale:1. ~mode:`Fan_in ~distribution:`Uniform );
    ("glorot_uniform", Init.glorot_uniform);
    ("glorot_normal", Init.glorot_normal);
    ("he_uniform", Init.he_uniform);
    ("he_normal", Init.he_normal);
    ("lecun_uniform", Init.lecun_uniform);
    ("lecun_normal", Init.lecun_normal);
  ]

let max_abs a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a

(* The statistical tests draw this many float64 values from one fixed key, so
   they are deterministic. *)
let draws = 100_000

(* The Kolmogorov-Smirnov distance of the draws [x] from the distribution
   function [cdf]: the largest gap between [cdf] and the draws' empirical
   distribution function. *)
let ks_distance cdf x =
  let x = Array.copy x in
  Array.sort Float.compare x;
  let n = float_of_int (Array.length x) in
  let d = ref 0. in
  Array.iteri
    (fun i v ->
      let f = cdf v in
      d :=
        Float.max !d
          (Float.max
             (f -. (float_of_int i /. n))
             ((float_of_int (i + 1) /. n) -. f)))
    x;
  !d

(* By the Dvoretzky-Kiefer-Wolfowitz inequality with Massart's constant, the
   distance of [n] independent draws exceeds [sqrt (ln (2 / a) / (2 n))] with
   probability at most [a]: 0.0103 at [n = 10^5] and [a = 10^-9]. A uniform and
   a normal of one variance are 0.057 apart, a normal and one truncated at 2
   standard deviations 0.017, and a normal and one of 6% another scale (an [n]
   of 4 for 4.5) 0.014: each such mistake is beyond it. *)
let ks_bound = sqrt (log 2e9 /. (2. *. float_of_int draws))

let from_distribution cdf x =
  at_most float_exact ~than:ks_bound (ks_distance cdf x)

let gauss_cdf x = 0.5 *. (1. +. Float.erf (x /. sqrt 2.))

(* t *)

let shape_name s =
  "[|" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "|]"

let has_dtype_and_shape (name, shape) =
  let check (type b) (dt : (float, b) Nx.dtype) =
    let init = List.assoc name (initializers ()) in
    let t = init (key ()) ~fan_in:3 ~fan_out:5 dt shape in
    equal string (dtype_name dt) (dtype_name (Nx.dtype t));
    equal (array int) shape (Nx.shape t)
  in
  check Nx.float64;
  check Nx.float32;
  check Nx.float16;
  check Nx.bfloat16;
  check Nx.float8_e4m3fn;
  check Nx.float8_e5m2;
  check Nx.float4_e2m1fn

let dtype_and_shape_rows =
  List.concat_map
    (fun (name, _) ->
      List.map (fun s -> (name, s)) [ [||]; [| 0; 3 |]; [| 2; 3; 4 |] ])
    (initializers ())

let one_key_one_draw name =
  let init = List.assoc name (initializers ()) in
  let draw () =
    values (init (key ()) ~fan_in:3 ~fan_out:5 Nx.float32 [| 4; 5 |])
  in
  equal (array float_exact) (draw ()) (draw ())

let software =
  fixture (fun () ->
      Nx.devices [ Result.get_ok (Rig.software "kaun-test-init") ])

let placement () =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)

let lies_at_the_key name =
  let module S = (val software ()) in
  let init = List.assoc name (initializers ()) in
  let t =
    init (Nx.Rng.place S.on (key ())) ~fan_in:3 ~fan_out:5 Nx.float32 [| 4; 5 |]
  in
  equal (placement ()) S.on (Nx.placement t)

(* One program draws through the uniform, normal and truncated normal paths; the
   key is its input, so each key's draw must be eager's from that key. Rune's
   Numerics: a draw, a product and a difference are exact. *)
let compiled_draws_eager () =
  Compiled.here ();
  let draw key =
    let keys = Nx.Rng.split ~n:3 key in
    let at (init : (Nx.float32_elt, _) Init.t) k =
      Nx.flatten (init k ~fan_in:3 ~fan_out:5 Nx.float32 [| 4; 5 |])
    in
    Nx.concatenate ~axis:0
      [
        at Init.glorot_uniform keys.(0);
        at (Init.normal ~stddev:2.) keys.(1);
        at Init.he_normal keys.(2);
      ]
  in
  let compiled = Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns tensor) draw in
  List.iter
    (fun seed ->
      equal
        ~msg:(Printf.sprintf "key %d" seed)
        (array float_exact)
        (values (draw (Nx.Rng.key seed)))
        (values (compiled (Nx.Rng.key seed))))
    [ 1; 2 ]

(* Constant *)

let constant_rows =
  [
    ("zeros", Init.zeros, 0.);
    ("ones", Init.ones, 1.);
    ("constant 3.5", Init.constant 3.5, 3.5);
    ("constant -0.", Init.constant (-0.), -0.);
    ("constant infinity", Init.constant infinity, infinity);
    ("constant nan", Init.constant nan, nan);
  ]

let fills (_, (init : (Nx.float32_elt, _) Init.t), v) =
  let t = init (key ()) ~fan_in:(-1) ~fan_out:0 Nx.float32 [| 3; 4 |] in
  equal (array float_exact) (Array.make 12 v) (values t)

(* Random *)

let uniform_draws () =
  let scale = 0.3 in
  let x =
    values
      (Init.uniform ~scale (key ()) ~fan_in:0 ~fan_out:0 Nx.float64 [| draws |])
  in
  at_least float_exact ~than:0. (Array.fold_left Float.min infinity x);
  less float_exact ~than:scale (Array.fold_left Float.max neg_infinity x);
  from_distribution (fun v -> v /. scale) x

let normal_draws () =
  let stddev = 0.3 in
  let x =
    values
      (Init.normal ~stddev (key ()) ~fan_in:0 ~fan_out:0 Nx.float64 [| draws |])
  in
  from_distribution (fun v -> gauss_cdf (v /. stddev)) x

let refuses_scale (scale, message) =
  raises (Invalid_argument message) (fun () -> Init.uniform ~scale)

let refuses_stddev (stddev, message) =
  raises (Invalid_argument message) (fun () -> Init.normal ~stddev)

(* Variance scaling *)

let distributions = [ `Normal; `Truncated_normal; `Uniform ]

let distribution_name = function
  | `Normal -> "`Normal"
  | `Truncated_normal -> "`Truncated_normal"
  | `Uniform -> "`Uniform"

let scaled ~mode distribution ~fan_in ~fan_out =
  let init = Init.variance_scaling ~scale:1.7 ~mode ~distribution in
  values (init (key ()) ~fan_in ~fan_out Nx.float32 [| 4; 5 |])

(* [`Fan_avg] at two equal fans divides by that fan, so each mode is the average
   mode at the fan it reads. The statistics below fix the average mode. *)
let modes_read_their_fan distribution =
  equal ~msg:"`Fan_in" (array float_exact)
    (scaled ~mode:`Fan_avg distribution ~fan_in:3 ~fan_out:3)
    (scaled ~mode:`Fan_in distribution ~fan_in:3 ~fan_out:5);
  equal ~msg:"`Fan_out" (array float_exact)
    (scaled ~mode:`Fan_avg distribution ~fan_in:5 ~fan_out:5)
    (scaled ~mode:`Fan_out distribution ~fan_in:3 ~fan_out:5)

(* The standard deviation of the standard normal truncated to [-2, 2]: with Z =
   erf (sqrt 2) and p = exp (-2) / sqrt (2 pi), its variance is 1 - 4p/Z =
   0.77374130354992. *)
let truncated_stddev = 0.87962566103423978

(* The distribution function of [distribution] at variance [v]. *)
let cdf distribution v x =
  match distribution with
  | `Normal -> gauss_cdf (x /. sqrt v)
  | `Uniform ->
      let limit = sqrt (3. *. v) in
      Float.min 1. (Float.max 0. ((x +. limit) /. (2. *. limit)))
  | `Truncated_normal ->
      let z = x *. truncated_stddev /. sqrt v in
      let lo = gauss_cdf (-2.) and hi = gauss_cdf 2. in
      Float.min 1. (Float.max 0. ((gauss_cdf z -. lo) /. (hi -. lo)))

(* At fans 3 and 6 the average mode's n is 4.5, which an integer average misses.
   The shape is unrelated to the fans. *)
let variance_is_scale_over_n distribution =
  let scale = 1.7 and n = 4.5 in
  let init = Init.variance_scaling ~scale ~mode:`Fan_avg ~distribution in
  let x =
    values (init (key ()) ~fan_in:3 ~fan_out:6 Nx.float64 [| 250; 400 |])
  in
  let v = scale /. n in
  from_distribution (cdf distribution v) x;
  match distribution with
  | `Normal -> ()
  | `Uniform -> at_most float_exact ~than:(sqrt (3. *. v)) (max_abs x)
  | `Truncated_normal ->
      at_most float_exact ~than:(2. *. sqrt v /. truncated_stddev) (max_abs x)

let fan_rows =
  [
    (0, 4, "fans must be positive, got fan_in=0 fan_out=4");
    (-1, 4, "fans must be positive, got fan_in=-1 fan_out=4");
    (3, 0, "fans must be positive, got fan_in=3 fan_out=0");
    (3, -1, "fans must be positive, got fan_in=3 fan_out=-1");
  ]

let refuses_fans (fan_in, fan_out, message) =
  let init =
    Init.variance_scaling ~scale:1. ~mode:`Fan_in ~distribution:`Normal
  in
  raises (Invalid_argument message) (fun () ->
      init (key ()) ~fan_in ~fan_out Nx.float32 [| 3 |])

(* Named families *)

let families =
  [
    ("glorot_uniform", Init.glorot_uniform, 1., `Fan_avg, `Uniform);
    ("glorot_normal", Init.glorot_normal, 1., `Fan_avg, `Truncated_normal);
    ("he_uniform", Init.he_uniform, 2., `Fan_in, `Uniform);
    ("he_normal", Init.he_normal, 2., `Fan_in, `Truncated_normal);
    ("lecun_uniform", Init.lecun_uniform, 1., `Fan_in, `Uniform);
    ("lecun_normal", Init.lecun_normal, 1., `Fan_in, `Truncated_normal);
  ]

let is_its_instance
    (_, (family : (Nx.float32_elt, _) Init.t), scale, mode, distribution) =
  let draw (init : (Nx.float32_elt, _) Init.t) =
    values (init (key ()) ~fan_in:3 ~fan_out:5 Nx.float32 [| 4; 5 |])
  in
  equal (array float_exact)
    (draw (Init.variance_scaling ~scale ~mode ~distribution))
    (draw family)

let () =
  exit
    (run "kaun init"
       [
         group "t"
           [
             cases "has the requested dtype and shape"
               ~id:(fun (name, s) -> name ^ " " ^ shape_name s)
               dtype_and_shape_rows has_dtype_and_shape;
             cases "lies where its key lies" ~id:Fun.id
               (List.map fst (initializers ()))
               lies_at_the_key;
             cases "one key gives one draw" ~id:Fun.id
               [
                 "uniform";
                 "normal";
                 "variance_scaling `Normal";
                 "variance_scaling `Truncated_normal";
                 "variance_scaling `Uniform";
               ]
               one_key_one_draw;
             test "compiles on a traced key to eager's draw from that key"
               compiled_draws_eager;
           ];
         group "constant"
           [
             cases "fills with its value and ignores the fans"
               ~id:(fun (name, _, _) -> name)
               constant_rows fills;
           ];
         group "uniform"
           [
             test "samples U[0, scale)" uniform_draws;
             (* The value prints as the shortest decimal that reads back, the
                same on every system: the C library's [%g] wrote 1e+06, 1e+006
                under mingw-w64. *)
             cases "raises on a scale that is negative or not finite" ~id:snd
               [
                 (-1., "scale must be finite and >= 0, got -1");
                 (-1e6, "scale must be finite and >= 0, got -1000000");
                 (Float.nan, "scale must be finite and >= 0, got nan");
                 (Float.infinity, "scale must be finite and >= 0, got inf");
               ]
               refuses_scale;
           ];
         group "normal"
           [
             test "samples N(0, stddev^2)" normal_draws;
             cases "raises on a stddev that is negative or not finite" ~id:snd
               [
                 (-0.1, "stddev must be finite and >= 0, got -0.1");
                 (-4.58e-72, "stddev must be finite and >= 0, got -4.58e-72");
                 (Float.nan, "stddev must be finite and >= 0, got nan");
                 (Float.infinity, "stddev must be finite and >= 0, got inf");
               ]
               refuses_stddev;
           ];
         group "variance_scaling"
           [
             cases "divides scale by the fan its mode reads"
               ~id:distribution_name distributions modes_read_their_fan;
             cases
               "samples its distribution at variance scale / n, within its \
                bound"
               ~id:distribution_name distributions variance_is_scale_over_n;
             cases "raises on a scale that is negative or not finite" ~id:snd
               [
                 (-1., "scale must be finite and >= 0, got -1");
                 (Float.nan, "scale must be finite and >= 0, got nan");
                 (Float.infinity, "scale must be finite and >= 0, got inf");
               ]
               (fun (scale, message) ->
                 raises (Invalid_argument message) (fun () ->
                     Init.variance_scaling ~scale ~mode:`Fan_in
                       ~distribution:`Uniform));
             cases "raises when applied to a fan that is not positive"
               ~id:(fun (_, _, message) -> message)
               fan_rows refuses_fans;
           ];
         group "named families"
           [
             cases "each is its variance_scaling instance"
               ~id:(fun (name, _, _, _, _) -> name)
               families is_its_instance;
           ];
       ])
