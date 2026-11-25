(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtypes =
  [
    Dtype Nx.float64;
    Dtype Nx.float32;
    Dtype Nx.float16;
    Dtype Nx.bfloat16;
    Dtype Nx.float8_e4m3fn;
    Dtype Nx.float8_e5m2;
    Dtype Nx.float4_e2m1fn;
  ]

let each_dtype f =
  List.iter (function Dtype dt -> f (Nx.Dtype.name dt) (Dtype dt)) dtypes

let values x = Nx.to_array (Nx.cast Nx.float64 x)

let pos_of rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  Nx.create Nx.int64 [| batch; seq |]
    (Array.map Int64.of_int (Array.concat (Array.to_list rows)))

let round32 v = Int32.float_of_bits (Int32.bits_of_float v)
let u dt = (Nx.Dtype.float_format dt).epsilon /. 2.
let u32 = u Nx.float32
let u64 = u Nx.float64

(* The least positive subnormal: the most a rounding loses near zero. *)
let tiny dt =
  let f = Nx.Dtype.float_format dt in
  f.min_normal *. f.epsilon

let pp_array pp ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ") pp)
    (Array.to_list a)

let pp_floats = pp_array (fun ppf v -> Format.fprintf ppf "%h" v)

(* An expected value and the error its derivation allows, against an actual
   value: equal when both are one float, NaN included, or within the
   allowance. *)
let within =
  Testable.make
    ~pp:(fun ppf (v, tol) ->
      if tol = 0. then Format.fprintf ppf "%h" v
      else Format.fprintf ppf "%h +- %.3g" v tol)
    ~equal:(fun (v, tol) (v', _) ->
      Float.equal v v' || Float.abs (v -. v') <= tol)

(* Rotary embeddings *)

(* Two rotations at positions [p] and [p'] and the same two shifted by [t]. *)
type shift = {
  freqs : float array;
  q : float array;
  k : float array;
  p : int;
  p' : int;
  t : int;
}

let shifts =
  let open Gen in
  with_pp (fun ppf s ->
      Format.fprintf ppf "@[frequencies %a,@ q %a,@ k %a,@ %d and %d by %d@]"
        pp_floats s.freqs pp_floats s.q pp_floats s.k s.p s.p' s.t)
  @@ let* pairs = int_range 1 4 in
     let* freqs = array ~size:(constant pairs) (float_range 0. 1.5) in
     let* q = array ~size:(constant (2 * pairs)) (float_range (-2.) 2.) in
     let* k = array ~size:(constant (2 * pairs)) (float_range (-2.) 2.) in
     let* p = int_range 0 63 in
     let* p' = int_range 0 63 in
     let+ t = int_range (-Int.min p p') (63 - Int.max p p') in
     { freqs; q; k; p; p'; t }

(* Each rotated feature is within [d] of the exact rotation's times the sum of
   its pair's magnitudes, [d] the error of a float32 cosine or sine: [u32], and
   [u64] for the angle and for the float64 cosine. A pair's product then errs by
   at most [4d A B] to first order, [A] and [B] the pairs' magnitude sums, and
   the two products compared by [8d] of their sum; [9d] covers the second order
   and the float64 sums. Near zero each float64 rounding loses half a subnormal
   more, times a factor of magnitude at most 4: [64 h] subnormals bound them
   over the [h] pairs. *)
let relative_positions =
  prop
    "the dot product of a rotated query and key depends on their positions \
     only through the difference"
    shifts (fun s ->
      cover "a negative shift" (s.t < 0);
      let h = Array.length s.freqs in
      let t = Rope.of_frequencies ~context:64 s.freqs in
      let rotate p x =
        Rope.apply t ~pos:(pos_of [| [| p |] |])
          (Nx.create Nx.float64 [| 1; 1; 1; 2 * h |] x)
      in
      let dot p p' =
        Nx.item [] (Nx.sum (Nx.mul (rotate p s.q) (rotate p' s.k)))
      in
      let angle = 64. *. Array.fold_left Float.max 0. s.freqs in
      let d = u32 +. ((angle +. 1.) *. u64) in
      let sum = ref 0. in
      for i = 0 to h - 1 do
        let a = Float.abs s.q.(i) +. Float.abs s.q.(i + h)
        and b = Float.abs s.k.(i) +. Float.abs s.k.(i + h) in
        sum := !sum +. (a *. b)
      done;
      equal within
        ( dot s.p s.p',
          (9. *. d *. !sum) +. (float_of_int (64 * h) *. tiny Nx.float64) )
        (dot (s.p + s.t) (s.p' + s.t), 0.))

(* of_frequencies *)

let frequencies_round_trip =
  let gen = Gen.(pair (int_range 1 4) (array ~size:(int_range 1 6) float)) in
  prop "the schedule's frequencies are f, in pair order" gen
    (fun (context, f) ->
      Law.round_trip (array float_exact) pass
        (Rope.of_frequencies ~context)
        Rope.frequencies f)

(* 25fc22708d: a float32 product of position 100003 and frequency 0.1 is off by
   about 1e-3 radians. Rotating [[1; 0]] reads the cosine and sine
   themselves. *)
let angles_rounded_once () =
  let t = Rope.of_frequencies ~context:100004 [| 0.1 |] in
  let x = Nx.create Nx.float32 [| 1; 1; 1; 2 |] [| 1.; 0. |] in
  let a = 100003. *. 0.1 in
  equal (array float_exact)
    [| round32 (cos a); round32 (sin a) |]
    (values (Rope.apply t ~pos:(pos_of [| [| 100003 |] |]) x))

let of_frequencies_refuses =
  cases
    "raises if f is empty or holds a non-finite number, or context is not \
     positive"
    ~id:(fun (id, _, _, _) -> id)
    [
      ("no frequency", 16, [||], "no frequency");
      ("NaN", 16, [| 1.; Float.nan |], "not finite");
      ("infinity", 16, [| Float.infinity |], "not finite");
      ("negative infinity", 16, [| 0.5; Float.neg_infinity |], "not finite");
      ("context 0", 0, [| 1. |], "context");
      ("context -1", -1, [| 1. |], "context");
    ]
    (fun (_, context, f, sub) ->
      raises_match (Exn.invalid_arg ~sub) (fun () ->
          Rope.of_frequencies ~context f))

(* make *)

let standard =
  let gen =
    Gen.(
      with_pp (fun ppf (theta, h) ->
          Format.fprintf ppf "theta %h, head_dim %d" theta h)
      @@ pair (float_range 1e-3 1e7) (map (fun n -> 2 * n) (int_range 1 8)))
  in
  prop "pair i has frequency theta ** (-2 i / head_dim)"
    ~examples:[ (1000., 6); (10000., 2) ]
    gen
    (fun (theta, head_dim) ->
      equal (array float_exact)
        (Array.init (head_dim / 2) (fun i ->
             theta ** (-2. *. float_of_int i /. float_of_int head_dim)))
        (Rope.frequencies (Rope.make ~theta ~head_dim ~context:1 ())))

let default_theta () =
  equal (array float_exact)
    (Rope.frequencies (Rope.make ~theta:10000. ~head_dim:8 ~context:1 ()))
    (Rope.frequencies (Rope.make ~head_dim:8 ~context:1 ()))

let make_refuses =
  cases
    "raises if head_dim is not positive and even, theta is not positive, or \
     context is not positive"
    ~id:(fun (id, _, _, _, _) -> id)
    [
      ("head_dim 3", 3, 10000., 4, "head_dim");
      ("head_dim 0", 0, 10000., 4, "head_dim");
      ("head_dim -2", -2, 10000., 4, "head_dim");
      ("theta 0", 4, 0., 4, "theta");
      ("theta -1", 4, -1., 4, "theta");
      ("theta nan, head_dim 2", 2, Float.nan, 4, "theta");
      ("theta nan, head_dim 4", 4, Float.nan, 4, "theta");
      ("context 0", 4, 10000., 0, "context");
    ]
    (fun (_, head_dim, theta, context, sub) ->
      raises_match (Exn.invalid_arg ~sub) (fun () ->
          Rope.make ~theta ~head_dim ~context ()))

(* llama3 *)

(* Meta's [apply_scaling] (llama-models), evaluated in float64 on the standard
   frequencies of a head of 8 at theta 500000:

   {v wavelen = 2 * pi / freq if wavelen < high_freq_wavelen: freq elif wavelen
   > low_freq_wavelen: freq / factor else: smooth = (old_context_len / wavelen -
   low_freq_factor) / (high_freq_factor - low_freq_factor) (1 - smooth) * freq /
   factor + smooth * freq v}

   Pairs 0 and 1 are kept, pair 2 is blended, pair 3 is divided by the factor.
   The two evaluations differ by the order of a few float64 roundings at
   most. *)
let llama3_bands =
  cases
    "low frequencies are divided by factor, high ones kept, and a linear blend \
     between"
    ~id:fst
    [
      ( "Llama 3.1, factor 8",
        ( 8.,
          [|
            1.0;
            0.03760603093086393;
            0.0005248461609929547;
            6.647869871181235e-06;
          |] ) );
      ( "Llama 3.2, factor 32",
        ( 32.,
          [|
            1.0;
            0.03760603093086393;
            0.00042955679655936815;
            1.6619674677953088e-06;
          |] ) );
    ]
    (fun (_, (factor, expected)) ->
      equal
        (array (float_rel ~rel:1e-15 ~abs:0.))
        expected
        (Rope.frequencies
           (Rope.llama3 ~theta:500000. ~head_dim:8 ~factor ~low_freq_factor:1.
              ~high_freq_factor:4. ~original_context:8192 ~context:1)))

let llama3_refuses =
  let llama3 ?(theta = 500000.) ?(head_dim = 8) ?(factor = 8.) ?(low = 1.)
      ?(high = 4.) ?(original = 8192) ?(context = 4) () =
    Rope.llama3 ~theta ~head_dim ~factor ~low_freq_factor:low
      ~high_freq_factor:high ~original_context:original ~context
  in
  cases
    "raises on a non-positive argument, or if high_freq_factor <= \
     low_freq_factor"
    ~id:fst
    [
      ("theta 0", ("theta", fun () -> llama3 ~theta:0. ()));
      ("theta nan", ("theta", fun () -> llama3 ~theta:Float.nan ()));
      ("factor nan", ("factor", fun () -> llama3 ~factor:Float.nan ()));
      ( "low_freq_factor nan",
        ("low_freq_factor", fun () -> llama3 ~low:Float.nan ()) );
      ( "high_freq_factor nan",
        ("high_freq_factor", fun () -> llama3 ~high:Float.nan ()) );
      ("head_dim 0", ("head_dim", fun () -> llama3 ~head_dim:0 ()));
      ("factor 0", ("factor", fun () -> llama3 ~factor:0. ()));
      ("low_freq_factor 0", ("low_freq_factor", fun () -> llama3 ~low:0. ()));
      ("high_freq_factor 0", ("high_freq_factor", fun () -> llama3 ~high:0. ()));
      ( "high_freq_factor equal to low_freq_factor",
        ("high_freq_factor", fun () -> llama3 ~high:1. ()) );
      ( "original_context 0",
        ("original_context", fun () -> llama3 ~original:0 ()) );
      ("context 0", ("context", fun () -> llama3 ~context:0 ()));
    ]
    (fun (_, (sub, make)) -> raises_match (Exn.invalid_arg ~sub) make)

(* yarn *)

(* gpt-oss's schedule on a head of 8. Its ramp runs from pair 1.01 to pair 2.17,
   so pairs 0 and 1 are kept, pair 2 is blended and pair 3 is divided by the
   factor. *)
let gpt_oss_yarn () =
  Rope.yarn ~theta:150000. ~head_dim:8 ~factor:32. ~beta_fast:32. ~beta_slow:1.
    ~original_context:4096 ~context:4096

(* transformers 5.17.0's [GptOssRotaryEmbedding] forms these in float32: a power
   and the blend's four steps, each within [u32] of 6e-8, put it within 3e-7 of
   the float64 values. *)
let yarn_ramp () =
  equal
    (array (float_rel ~rel:1e-6 ~abs:0.))
    [| 1.0; 0.05081327259540558; 0.0004564839182421565; 4.099978468730114e-06 |]
    (Rope.frequencies (gpt_oss_yarn ()))

(* transformers 5.17.0 scales its cosines and sines by 0.1 ln 32 + 1: divided by
   that, its rotated values are ours. Its float32 frequencies are a unit in the
   last place from ours, which at position 3000 moves pair 1's angle of 152
   radians by 1e-5, and so an element of a pair of norm at most 4.6 by 4.6e-5;
   the float32 roundings on both sides add under 1e-6. *)
let yarn_rotation () =
  let x =
    Nx.create Nx.float32 [| 1; 1; 3; 8 |]
      (Array.init 24 (fun i -> (float_of_int i -. 10.) /. 4.))
  in
  let concentration = (0.1 *. log 32.) +. 1. in
  let reference =
    [|
      -3.366434097290039;
      -3.0297906398773193;
      -2.6931471824645996;
      -2.35650372505188;
      -2.01986026763916;
      -1.6832170486450195;
      -1.3465735912322998;
      -1.00993013381958;
      0.4546450674533844;
      -0.5796743035316467;
      -0.003073443192988634;
      0.33660888671875;
      0.836617112159729;
      0.8928972482681274;
      1.3465700149536133;
      1.6832239627838135;
      -2.708630323410034;
      -3.864203691482544;
      -3.41951847076416;
      2.975733995437622;
      -2.841836452484131;
      2.0817830562591553;
      3.4466328620910645;
      4.413298606872559;
    |]
  in
  let y = Rope.apply (gpt_oss_yarn ()) ~pos:(pos_of [| [| 0; 5; 3000 |] |]) x in
  equal
    (array (float 5e-5))
    (Array.map (fun v -> v /. concentration) reference)
    (values y)

(* The .mli's formula in float64 on the standard frequencies of a head of 8 at
   theta 10000 and factor 4 over 4096 positions:

   {v pair turns = head_dim * ln (original_context / (turns * 2 pi)) / (2 ln
   theta) low = max (pair beta_fast) 0 high = min (pair beta_slow) (head_dim -
   1) ramp i = clamp to [0, 1] of (i - low) / (high - low) f' i = ramp i * f i /
   factor + (1 - ramp i) * f i v}

   At beta_fast 1000, [pair 1000] is -0.186: clamped to 0, pair 0 keeps its
   frequency, which the ramp from -0.186 would blend. At beta_slow 1e-6, [pair
   1e-6] is 8.81: clamped to 7, pair 2's ramp is 0.121 where 8.81 would give
   0.092. *)
let yarn_clamp =
  cases "the ramp's ends are clamped to [0, head_dim - 1]" ~id:fst
    [
      ( "beta_fast 1000, below pair 0",
        ( (1000., 1.),
          [| 1.0; 0.07334925346696261; 0.004669850693392522; 0.00025 |] ) );
      ( "beta_slow 1e-6, past pair 7",
        ( (32., 1e-6),
          [| 1.0; 0.1; 0.009089386460908656; 0.0007771509168726924 |] ) );
    ]
    (fun (_, ((beta_fast, beta_slow), expected)) ->
      equal
        (array (float_rel ~rel:1e-15 ~abs:0.))
        expected
        (Rope.frequencies
           (Rope.yarn ~theta:10000. ~head_dim:8 ~factor:4. ~beta_fast ~beta_slow
              ~original_context:4096 ~context:1)))

let yarn_refuses =
  let yarn ?(theta = 150000.) ?(head_dim = 8) ?(factor = 32.) ?(fast = 32.)
      ?(slow = 1.) ?(original = 4096) ?(context = 4) () =
    Rope.yarn ~theta ~head_dim ~factor ~beta_fast:fast ~beta_slow:slow
      ~original_context:original ~context
  in
  cases
    "raises if head_dim is not positive and even, theta, beta_slow, \
     original_context or context is not positive, factor is below 1, or \
     beta_fast <= beta_slow"
    ~id:fst
    [
      ("head_dim 3", ("head_dim", fun () -> yarn ~head_dim:3 ()));
      ("head_dim 0", ("head_dim", fun () -> yarn ~head_dim:0 ()));
      ("theta 0", ("theta", fun () -> yarn ~theta:0. ()));
      ("theta nan", ("theta", fun () -> yarn ~theta:Float.nan ()));
      ("beta_slow nan", ("beta_slow", fun () -> yarn ~slow:Float.nan ()));
      ("beta_fast nan", ("beta_fast", fun () -> yarn ~fast:Float.nan ()));
      ("factor nan", ("factor", fun () -> yarn ~factor:Float.nan ()));
      ("beta_slow 0", ("beta_slow", fun () -> yarn ~fast:1. ~slow:0. ()));
      ("original_context 0", ("original_context", fun () -> yarn ~original:0 ()));
      ("context 0", ("context", fun () -> yarn ~context:0 ()));
      ("factor 0.5", ("factor", fun () -> yarn ~factor:0.5 ()));
      ("beta_fast equal to beta_slow", ("beta_fast", fun () -> yarn ~fast:1. ()));
      ("beta_fast below beta_slow", ("beta_fast", fun () -> yarn ~fast:0.5 ()));
    ]
    (fun (_, (sub, make)) -> raises_match (Exn.invalid_arg ~sub) make)

(* place *)

let two =
  fixture (fun () ->
      Nx.devices
        [
          Result.get_ok (Rig.software "kaun-rope-0");
          Result.get_ok (Rig.software "kaun-rope-1");
        ])

(* A rotation reads the tables beside [x], which lies whole on each device:
   placed whole, they need no move, and the rotation lies where [x] does, with
   the host's values. *)
let place_whole () =
  let (module Two : Nx.Devices) = two () in
  let t = Rope.make ~head_dim:4 ~context:8 () in
  let x = Nx.create Nx.float32 [| 1; 1; 2; 4 |] (Array.init 8 float_of_int) in
  let pos = pos_of [| [| 3; 7 |] |] in
  let y =
    Rope.apply
      (Rope.place (Two.split ~axis:0) t)
      ~pos:(Nx.place Two.on pos) (Nx.place Two.on x)
  in
  let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal in
  equal
    (pair placement (array float_exact))
    (Two.on, values (Rope.apply t ~pos x))
    (Nx.placement y, values y)

(* frequencies *)

let frequencies_copy () =
  let t = Rope.of_frequencies ~context:1 [| 1.; 0.5 |] in
  (Rope.frequencies t).(0) <- 7.;
  equal (array float_exact) [| 1.; 0.5 |] (Rope.frequencies t)

(* context *)

let context_given () =
  equal (list int) [ 1; 7; 9; 11 ]
    [
      Rope.context (Rope.of_frequencies ~context:1 [| 1. |]);
      Rope.context (Rope.make ~head_dim:4 ~context:7 ());
      Rope.context
        (Rope.llama3 ~theta:500000. ~head_dim:8 ~factor:8. ~low_freq_factor:1.
           ~high_freq_factor:4. ~original_context:8192 ~context:9);
      Rope.context
        (Rope.yarn ~theta:150000. ~head_dim:8 ~factor:32. ~beta_fast:32.
           ~beta_slow:1. ~original_context:4096 ~context:11);
    ]

(* apply *)

(* A schedule and an input: [x] of shape [[| batch; heads; seq; head_dim |]] and
   positions of shape [[| batch; seq |]], or [[| 1; seq |]] when [shared]. *)
type rotation = {
  freqs : float array;
  context : int;
  batch : int;
  heads : int;
  seq : int;
  shared : bool;
  pos : int64 array;
  xs : float array;
}

let pp_rotation ppf r =
  Format.fprintf ppf
    "@[frequencies %a,@ context %d,@ x [%d; %d; %d; %d] %a,@ %s positions %a@]"
    pp_floats r.freqs r.context r.batch r.heads r.seq
    (2 * Array.length r.freqs)
    pp_floats r.xs
    (if r.shared then "shared" else "per-row")
    (pp_array (fun ppf p -> Format.fprintf ppf "%Ld" p))
    r.pos

(* An extent of x, 0 one draw in twelve. *)
let extent n = Gen.(frequency [ (1, constant 0); (11, int_range 1 n) ])

(* Elements of every magnitude, signed zeros, infinities and NaN. *)
let element =
  Gen.(
    frequency
      [
        (5, float_range (-2.) 2.);
        (1, float);
        (1, of_list [ -0.; Float.infinity; Float.neg_infinity; Float.nan ]);
      ])

let rotations =
  let open Gen in
  with_pp pp_rotation
  @@ let* pairs = int_range 1 4 in
     let* freqs = array ~size:(constant pairs) (float_range (-1.5) 1.5) in
     let* context = int_range 1 16 in
     let* batch = extent 2 in
     let* heads = extent 2 in
     let* seq = extent 3 in
     let* shared = bool in
     let rows = if shared then 1 else batch in
     let* pos =
       array
         ~size:(constant (rows * seq))
         (map Int64.of_int (int_range 0 (context - 1)))
     in
     let+ xs =
       array ~size:(constant (batch * heads * seq * 2 * pairs)) element
     in
     { freqs; context; batch; heads; seq; shared; pos; xs }

(* The rotation the .mli states, each step stored at [dt]: the angle in float64,
   its cosine and sine rounded to float32, zero outside [[0, context)], then
   cast to [dt], each product and the sum rounded to [dt]. Below float64 a
   product of two values of at most 24 bits is exact in float64, and a sum
   rounded to float64 and then to a format of [p <= 24] bits is the sum rounded
   once, since [53 >= 2p + 2]; at float64 the float64 steps are [dt]'s own. *)
let rotation dt r ~xs =
  let at = Nx.Dtype.of_float dt in
  let h = Array.length r.freqs in
  let y = Array.make (Array.length xs) 0. in
  for b = 0 to r.batch - 1 do
    for hd = 0 to r.heads - 1 do
      for s = 0 to r.seq - 1 do
        let p = r.pos.(((if r.shared then 0 else b) * r.seq) + s) in
        let base = ((((b * r.heads) + hd) * r.seq) + s) * 2 * h in
        for i = 0 to h - 1 do
          let a = Int64.to_float p *. r.freqs.(i) in
          let c, sn =
            if p < 0L || p >= Int64.of_int r.context then (0., 0.)
            else (at (round32 (cos a)), at (round32 (sin a)))
          in
          let x1 = xs.(base + i) and x2 = xs.(base + i + h) in
          y.(base + i) <- at (at (x1 *. c) -. at (x2 *. sn));
          y.(base + i + h) <- at (at (x2 *. c) +. at (x1 *. sn))
        done
      done
    done
  done;
  y

(* At bfloat16 a long position turns by its float64 angle, whose cosine and sine
   are cast to bfloat16 once. *)
let long_position =
  {
    freqs = [| 1. |];
    context = 100004;
    batch = 1;
    heads = 1;
    seq = 1;
    shared = false;
    pos = [| 100003L |];
    xs = [| 1.; 0. |];
  }

let shared_positions =
  {
    freqs = [| 1.; 0.25 |];
    context = 8;
    batch = 2;
    heads = 1;
    seq = 2;
    shared = true;
    pos = [| 3L; 7L |];
    xs = Array.init 16 (fun i -> float_of_int (i - 8) /. 4.);
  }

let rotates =
  prop
    "rotates feature i with feature i + head_dim / 2 by pos * frequency.(i), \
     with the tables' cosines and sines cast to x's dtype"
    ~examples:[ long_position; shared_positions ] rotations (fun r ->
      cover "positions shared by the batch" (r.shared && r.batch > 1);
      cover "one pair" (Array.length r.freqs = 1);
      cover "the last position" (Array.mem (Int64.of_int (r.context - 1)) r.pos);
      cover "a zero-size x" (r.batch * r.heads * r.seq = 0);
      let t = Rope.of_frequencies ~context:r.context r.freqs in
      let shape = [| r.batch; r.heads; r.seq; 2 * Array.length r.freqs |] in
      let pos =
        Nx.create Nx.int64 [| (if r.shared then 1 else r.batch); r.seq |] r.pos
      in
      each_dtype (fun msg (Dtype dt) ->
          let x = Nx.cast dt (Nx.create Nx.float64 shape r.xs) in
          equal ~msg (array float_exact)
            (rotation dt r ~xs:(values x))
            (values (Rope.apply t ~pos x))))

(* Signed zeros compare as zero: [x(i) cos 0 - x(i + h) sin 0] is [+0.] at [x(i)
   = -0.] when [x(i + h)] is negative. *)
let position_zero =
  let element =
    Gen.(frequency [ (4, float_range (-4.) 4.); (1, constant (-0.)) ])
  in
  let gen =
    Gen.(
      with_pp (fun ppf (h, xs) ->
          Format.fprintf ppf "%d pairs, x %a" h pp_floats xs)
      @@ let* h = int_range 1 4 in
         let+ xs = array ~size:(constant (4 * h)) element in
         (h, xs))
  in
  prop "position 0 is the identity" gen (fun (h, xs) ->
      cover "a signed zero" (Array.exists (fun v -> 1. /. v < 0. && v = 0.) xs);
      let t = Rope.make ~head_dim:(2 * h) ~context:1 () in
      let zero v = v +. 0. in
      each_dtype (fun msg (Dtype dt) ->
          let x = Nx.cast dt (Nx.create Nx.float64 [| 1; 1; 2; 2 * h |] xs) in
          let y = Rope.apply t ~pos:(pos_of [| [| 0; 0 |] |]) x in
          equal ~msg (array float_exact)
            (Array.map zero (values x))
            (Array.map zero (values y))))

(* One program, the positions its input. *)
let outside_context =
  let in_range = Gen.(map Int64.of_int (int_range 0 7)) in
  let outside = Gen.of_list [ -1L; 8L; 9L; Int64.max_int; Int64.min_int ] in
  let rows =
    Gen.(
      with_pp (pp_array (fun ppf p -> Format.fprintf ppf "%Ld" p))
      @@ array ~size:(constant 5) (frequency [ (2, in_range); (1, outside) ]))
  in
  let program =
    lazy
      (let t = Rope.make ~head_dim:4 ~context:8 () in
       let x =
         Nx.create Nx.float32 [| 1; 1; 5; 4 |]
           (Array.init 20 (fun i -> float_of_int (i - 7) /. 2.))
       in
       (t, x, Rune.jit' (fun pos -> Rope.apply t ~pos x)))
  in
  prop
    "a position outside [0, context) gives zero features on finite x, eagerly \
     and under Rune.jit alike"
    ~examples:
      [
        [| 3L; -1L; 7L; 8L; Int64.max_int |];
        [| 0L; Int64.min_int; 9L; 5L; 2L |];
      ]
    rows
    (fun row ->
      let outside p = p < 0L || p >= 8L in
      cover "a position outside" (Array.exists outside row);
      cover "a position inside" (Array.exists (fun p -> not (outside p)) row);
      Compiled.here ();
      let t, x, compiled = Lazy.force program in
      let r =
        {
          freqs = Rope.frequencies t;
          context = 8;
          batch = 1;
          heads = 1;
          seq = 5;
          shared = false;
          pos = row;
          xs = values x;
        }
      in
      let pos = Nx.create Nx.int64 [| 1; 5 |] row in
      let expected = rotation Nx.float32 r ~xs:r.xs in
      equal ~msg:"eagerly" (array float_exact) expected
        (values (Rope.apply t ~pos x));
      equal ~msg:"under Rune.jit" (array float_exact) expected
        (values (compiled pos)))

let gradients () =
  let t = Rope.make ~head_dim:4 ~context:10 () in
  let pos = pos_of [| [| 2; 5; 9 |] |] in
  let w =
    Nx.create Nx.float64 [| 1; 2; 3; 4 |]
      (Array.init 24 (fun i -> float_of_int ((i * 7 mod 11) - 5) /. 3.))
  in
  let x =
    Nx.create Nx.float64 [| 1; 2; 3; 4 |]
      (Array.init 24 (fun i -> float_of_int ((i * 5 mod 13) - 6) /. 4.))
  in
  equal (result unit string) (Ok ())
    (Rune.check_grads Nx.Ptree.tensor
       (fun x -> Nx.sum (Nx.mul w (Rope.apply t ~pos x)))
       x)

let apply_refuses =
  cases
    "raises if x is not of rank 4, its last axis is not twice the pair count, \
     or pos has another shape"
    ~id:(fun (id, _, _) -> id)
    [
      ("x of rank 3", [| 1; 3; 4 |], [| 1; 3 |]);
      ("x of rank 5", [| 1; 1; 1; 3; 4 |], [| 1; 3 |]);
      ("a last axis of 6", [| 1; 1; 3; 6 |], [| 1; 3 |]);
      ("pos of 2 positions for 3", [| 2; 1; 3; 4 |], [| 1; 2 |]);
      ("pos of 3 rows for 2", [| 2; 1; 3; 4 |], [| 3; 3 |]);
      ("pos of rank 1", [| 1; 1; 3; 4 |], [| 3 |]);
    ]
    (fun (_, shape, pos) ->
      let t = Rope.make ~head_dim:4 ~context:4 () in
      raises_match (Exn.invalid_arg ~sub:"Rope.apply") (fun () ->
          Rope.apply t ~pos:(Nx.zeros Nx.int64 pos) (Nx.zeros Nx.float32 shape)))

let () =
  exit
    (run "kaun rope"
       [
         group "rotary embeddings" [ relative_positions ];
         group "of_frequencies"
           [
             frequencies_round_trip;
             test "angles are computed in float64 and rounded once to float32"
               angles_rounded_once;
             of_frequencies_refuses;
           ];
         group "make"
           [
             standard;
             test "theta defaults to 10000." default_theta;
             make_refuses;
           ];
         group "llama3" [ llama3_bands; llama3_refuses ];
         group "yarn"
           [
             test "frequencies agree with transformers' gpt-oss schedule"
               yarn_ramp;
             test "rotations agree with transformers' gpt-oss rotation"
               yarn_rotation;
             yarn_clamp;
             yarn_refuses;
           ];
         group "place"
           [ test "the cosines and sines lie whole on each device" place_whole ];
         group "frequencies" [ test "is a copy" frequencies_copy ];
         group "context"
           [ test "is the context the schedule was made with" context_given ];
         group "apply"
           [
             rotates;
             position_zero;
             outside_context;
             test "gradients in x agree with central differences" gradients;
             apply_refuses;
           ];
       ])
