(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module R = Alda.Resample
module K = Alda.Kernel

(* torchaudio's resample and torch's antialiased interpolate, from
   golden/resample.npz (contrib/alda/test/gen/resample.py) *)
let golden = lazy (Nx_io.load_npz "golden/resample.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/resample.npz has no %s" key

let floats a = Nx.to_array (Nx.cast Nx.float64 a)

let bits a =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (floats a)

let equal_bits want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal (array int64) (bits want) (bits got)

(* torchaudio's default kernel: a Hann-windowed sinc of rolloff 0.99 and six
   zero crossings, read as zeros past the ends. *)
let torchaudio =
  K.Sinc { cutoff = 0.495; half_width = 6. /. 0.99; window = K.Hann }

let rates i =
  let r =
    Array.map Int64.to_int
      (Nx.to_array (entry Nx.int64 (Printf.sprintf "ta.%d.rates" i)))
  in
  (r.(0), r.(1))

let ta i =
  let src, dst = rates i in
  R.v ~grid:`Points ~edge:`Zero ~kernel:torchaudio ~by:(dst, src)

let rate_name i =
  let src, dst = rates i in
  Printf.sprintf "%d to %d" src dst

(* In float64 both compute the same weights at float64: within 1e-13. *)
let torchaudio64 =
  cases "torchaudio's resample in float64" ~id:rate_name [ 0; 1; 2; 3 ]
    (fun i ->
      let key s = Printf.sprintf "ta.%d.f64.%s" i s in
      let want = entry Nx.float64 (key "y") in
      let got = Alda.resample (ta i) (entry Nx.float64 (key "x")) in
      equal (array int) (Nx.shape want) (Nx.shape got);
      equal (array (float 1e-13)) (floats want) (floats got))

(* [magnitudes ~by w x start count] is, for outputs [start] to [start + count -
   1] on points of the reduced ratio [by], [Σ |wᵢ| |x(i)|] over the [T] weights
   [w] of its phase ({!Alda.Resample.weights}), the samples outside [x] zero,
   and [T] the most weights a phase has. *)
let magnitudes ~by:(l, m) w x start count =
  let n = Array.length x in
  let taps = Array.fold_left (fun t (_, w) -> Int.max t (Array.length w)) 0 w in
  let sums =
    Array.init count (fun j ->
        let u = start + j in
        let off, w = w.(u mod l) in
        let base = (u * m / l) + off in
        let s = ref 0. in
        Array.iteri
          (fun q v ->
            let i = base + q in
            if i >= 0 && i < n then s := !s +. Float.abs (v *. x.(i)))
          w;
        !s)
  in
  (taps, sums)

(* [within ~bound want got] checks each output against its own bound. *)
let within ~bound want got =
  let worst = ref 0. in
  Array.iteri
    (fun j w -> worst := Float.max !worst (Float.abs (got.(j) -. w) /. bound j))
    want;
  less float_exact ~than:1. !worst

let u32 = Float.ldexp 1. (-24)

(* In float32 alda rounds its float64 weights once and sums [T] products, after
   the samples were rounded to float32: within [(T + 2) u Σ |wᵢ xᵢ|] of the
   float64 result, which is torchaudio's to 1e-13. torchaudio computes its
   kernel's positions in float32, which costs it up to 2e-5 at 22.05 kHz's 320
   phases. *)
let torchaudio32 =
  cases "float32 is within its rounding of float64" ~id:rate_name [ 0; 1; 2; 3 ]
    (fun i ->
      let key s = Printf.sprintf "ta.%d.%s" i s in
      let want = floats (entry Nx.float64 (key "f64.y")) in
      let got =
        floats (Alda.resample (ta i) (entry Nx.float32 (key "f32.x")))
      in
      let src, dst = rates i in
      let rec gcd a b = if b = 0 then a else gcd b (a mod b) in
      let g = gcd src dst in
      let taps, sums =
        magnitudes
          ~by:(dst / g, src / g)
          (R.weights (ta i))
          (floats (entry Nx.float64 (key "f64.x")))
          0 (Array.length want)
      in
      within
        ~bound:(fun j -> Float.of_int (taps + 2) *. u32 *. sums.(j))
        want got)

(* torch's antialiased resize is resampling on cells, renormalised, along each
   axis in turn: Keys' cubic of -0.5 for bicubic. *)
let images =
  let rows =
    List.concat_map
      (fun j -> [ (j, "bilinear", K.Triangle); (j, "bicubic", K.Cubic (-0.5)) ])
      [ 0; 1; 2; 3 ]
  in
  cases "torch's antialiased interpolate as two calls"
    ~id:(fun (j, mode, _) -> Printf.sprintf "%d %s" j mode)
    rows
    (fun (j, mode, kernel) ->
      let x = entry Nx.float64 (Printf.sprintf "img.%d.x" j) in
      let want = entry Nx.float64 (Printf.sprintf "img.%d.%s.y" j mode) in
      let along axis n x =
        Alda.resample ~axis
          (R.v ~grid:`Cells ~edge:`Renormalise ~kernel ~by:(n, Nx.dim axis x))
          x
      in
      let got = along 1 (Nx.dim 1 want) (along 0 (Nx.dim 0 want) x) in
      equal (array int) (Nx.shape want) (Nx.shape got);
      equal (array (float 1e-14)) (floats want) (floats got))

(* Streams *)

(* swr's kernel to 16 kHz: its cut and length as resample_init computes them, at
   float64 from the raw rates in that order. *)
let swr ~from =
  let factor = Float.min (16_000. *. 0.97 /. Float.of_int from) 1. in
  let taps = Float.to_int (Float.ceil (32. /. factor)) in
  let taps = taps + (taps land 1) in
  let s = Float.max 1. (Float.of_int from /. 16_000.) in
  K.Sinc
    {
      cutoff = factor *. s /. 2.;
      half_width = Float.of_int taps /. 2. /. s;
      window = K.Kaiser 9.;
    }

type case = {
  from : int;
  kernel : string;
  edge : [ Alda.edge | `Renormalise ];
  grid : [ `Points | `Cells ];
  n : int;
  split : [ `By of int | `Random ];
  channels : bool;
  seed : int;
}

let edge_name = function
  | `Zero -> "Zero"
  | `Replicate -> "Replicate"
  | `Symmetric_whole -> "Symmetric_whole"
  | `Symmetric_half -> "Symmetric_half"
  | `Antisymmetric_whole -> "Antisymmetric_whole"
  | `Renormalise -> "Renormalise"

let pp_case ppf c =
  Format.fprintf ppf "{%d to 16000; %s; %s; %s; n %d; %s; %s; seed %d}" c.from
    c.kernel (edge_name c.edge)
    (match c.grid with `Points -> "points" | `Cells -> "cells")
    c.n
    (match c.split with
    | `By k -> Printf.sprintf "by %d" k
    | `Random -> "random")
    (if c.channels then "[n; 2] along 0" else "[n]")
    c.seed

let case =
  let open Gen in
  with_pp pp_case
  @@
  let* from = of_list [ 48_000; 44_100; 8_000 ] in
  let* kernel = of_list [ "swr"; "torchaudio"; "triangle" ] in
  let* edge =
    of_list
      [
        `Zero;
        `Replicate;
        `Symmetric_whole;
        `Symmetric_half;
        `Antisymmetric_whole;
        `Renormalise;
      ]
  in
  let* grid = of_list [ `Points; `Cells ] in
  let* n = frequency [ (1, int_range 0 5); (4, int_range 0 2500) ] in
  let* split = of_list [ `By 480; `By 441; `By 333; `By 1; `Random ] in
  let* channels = bool in
  let+ seed = int_range 0 1_000_000 in
  { from; kernel; edge; grid; n; split; channels; seed }

let resampling c =
  let kernel =
    match c.kernel with
    | "swr" -> swr ~from:c.from
    | "torchaudio" -> torchaudio
    | _ -> K.Triangle
  in
  R.v ~grid:c.grid ~edge:c.edge ~kernel ~by:(16_000, c.from)

let signal c =
  let r = Random.State.make [| c.seed |] in
  Nx.init Nx.float32
    (if c.channels then [| c.n; 2 |] else [| c.n |])
    (fun _ -> Random.State.float r 2. -. 1.)

let axis c = if c.channels then 0 else -1

let chunks c =
  let r = Random.State.make [| c.seed; 1 |] in
  let rec go left =
    if left = 0 then [ 0 ]
    else
      let k =
        match c.split with
        | `By k -> Int.min k left
        | `Random -> Int.min left (Random.State.int r 700)
      in
      k :: go (left - k)
  in
  go c.n

let piece c x a b =
  if c.channels then Nx.slice [ Nx.R (a, b); Nx.A ] x
  else Nx.slice [ Nx.R (a, b) ] x

let stream_law =
  prop "every split gives the whole call's bits" ~count:200 case (fun c ->
      cover ~share:0.1 "Renormalise" (c.edge = `Renormalise);
      cover ~share:0.05 "a signal shorter than the kernel" (c.n > 0 && c.n < 40);
      let r = resampling c and x = signal c in
      let whole =
        match Alda.resample ~axis:(axis c) r x with
        | y -> y
        | exception Invalid_argument _ ->
            (* an extension of no sample *)
            assume false;
            assert false
      in
      let s, outs, _ =
        List.fold_left
          (fun (s, outs, at) k ->
            let s, y = R.feed ~axis:(axis c) r s (piece c x at (at + k)) in
            (s, y :: outs, at + k))
          (R.start, [], 0) (chunks c)
      in
      cover ~share:0.3 "outputs from feeds"
        (List.exists (fun y -> Nx.dim (axis c) y > 0) outs);
      let last = R.finish ~axis:(axis c) r s in
      equal_bits whole (Nx.concatenate ~axis:(axis c) (List.rev (last :: outs))))

(* After its first feed a stream of chunks of [m] samples keeps its state's
   shapes, so Rune.scan carries it: the scan's outputs are the eager feeds'. *)
let in_scan =
  test "a stream in Rune.scan after a first feed" (fun () ->
      let r =
        R.v ~grid:`Points ~edge:`Symmetric_whole ~kernel:(swr ~from:48_000)
          ~by:(1, 3)
      in
      let rng = Random.State.make [| 9 |] in
      let x =
        Nx.init Nx.float32
          [| 20 * 480 |]
          (fun _ -> Random.State.float rng 2. -. 1.)
      in
      let chunk i = Nx.slice [ Nx.R (480 * i, 480 * (i + 1)) ] x in
      let s0, y0 = R.feed r R.start (chunk 0) in
      less int ~than:160 (Nx.dim 0 y0);
      let st = Nx.Ptree.instantiate (module R.State) in
      let rest =
        Nx.reshape [| 19; 480 |] (Nx.slice [ Nx.R (480, 20 * 480) ] x)
      in
      let s, ys =
        Rune.scan st Nx.Ptree.tensor Nx.Ptree.tensor
          (fun s c -> R.feed r s c)
          ~init:s0 rest
      in
      equal (array int) [| 19; 160 |] (Nx.shape ys);
      let eager =
        let s = ref s0 and out = ref [] in
        for i = 1 to 19 do
          let s', y = R.feed r !s (chunk i) in
          s := s';
          out := y :: !out
        done;
        (!s, Nx.concatenate ~axis:0 (List.rev !out))
      in
      equal_bits (snd eager) (Nx.reshape [| 19 * 160 |] ys);
      equal_bits
        (Nx.concatenate ~axis:0 [ y0; snd eager; R.finish r (fst eager) ])
        (Nx.concatenate ~axis:0
           [ y0; Nx.reshape [| 19 * 160 |] ys; R.finish r s ]))

(* Kernels *)

(* libsoxr's 20-bit quality at 16 kHz: a pass band to 0.9137 of the Nyquist
   frequency, a stop band from it, 120.4 dB (20 bits of 6.02 dB). The taps'
   gain, read at 512 times as many frequencies as taps, stays within δ. *)
let soxr_spec =
  test "lowpass meets libsoxr's 20-bit specification" (fun () ->
      let rate = 16_000.
      and pass = 0.9137 *. 8000.
      and stop = 8000.
      and reject = 120.4 in
      let k = K.lowpass ~rate ~pass ~stop ~reject in
      let t = K.taps Nx.float64 k in
      let len = Nx.dim 0 t in
      let n = ref 1 in
      while !n < 512 * len do
        n := 2 * !n
      done;
      let g =
        Nx.to_array
          (Nx.magnitude Nx.float64 (Nx.rfft Nx.complex128 ~s:[ !n ] t))
      in
      let delta = 10. ** (-.reject /. 20.) in
      let pass_err = ref 0. and stop_gain = ref 0. in
      Array.iteri
        (fun i g ->
          let f = Float.of_int i *. rate /. Float.of_int !n in
          if f <= pass then
            pass_err := Float.max !pass_err (Float.abs (g -. 1.));
          if f >= stop then stop_gain := Float.max !stop_gain g)
        g;
      less float_exact ~than:delta !pass_err;
      less float_exact ~than:delta !stop_gain;
      (* Kaiser's estimate alone falls short of it *)
      match k with
      | K.Sinc { half_width; _ } ->
          let dw = 2. *. Float.pi *. (stop -. pass) /. rate in
          less float_exact ~than:half_width
            (Float.ceil ((reject -. 7.95) /. (4.57 *. dw)))
      | _ -> fail "not a sinc")

(* [amplitude t f] is the zero-phase gain of the odd symmetric taps [t] at [f]
   cycles per sample, summed here in OCaml floats. *)
let amplitude t f =
  let h = (Array.length t - 1) / 2 in
  let s = ref 0. in
  Array.iteri
    (fun k v ->
      s := !s +. (v *. Float.cos (2. *. Float.pi *. f *. Float.of_int (k - h))))
    t;
  !s

(* [largest t ~lo ~hi err] is the largest [err] of the gain over [[lo, hi]]:
   both edges, a grid of 256 points a tap, and each grid maximum refined by
   ternary search. *)
let largest t ~lo ~hi err =
  let e f = err (amplitude t f) in
  let m =
    Int.max 2
      (256 * Array.length t * Float.to_int (Float.ceil (2. *. (hi -. lo))))
  in
  let f i = lo +. ((hi -. lo) *. Float.of_int i /. Float.of_int m) in
  let g = Array.init (m + 1) (fun i -> e (f i)) in
  let worst = ref (Float.max (e lo) (e hi)) in
  for i = 0 to m do
    worst := Float.max !worst g.(i);
    if (i = 0 || g.(i) >= g.(i - 1)) && (i = m || g.(i) >= g.(i + 1)) then begin
      let a = ref (f (Int.max 0 (i - 1))) and b = ref (f (Int.min m (i + 1))) in
      for _ = 1 to 60 do
        let c = !a +. ((!b -. !a) /. 3.) and d = !b -. ((!b -. !a) /. 3.) in
        if e c > e d then b := d else a := c
      done;
      worst := Float.max !worst (e ((!a +. !b) /. 2.))
    end
  done;
  !worst

type spec = { pass : float; stop : float; reject : float }

let pp_spec ppf s =
  Format.fprintf ppf "{rate 1000; pass %g; stop %g; reject %g}" s.pass s.stop
    s.reject

let spec =
  let open Gen in
  with_pp pp_spec
  @@
  let* pass = float_range 20. 420. in
  let* width = float_range 25. 80. in
  let+ reject = float_range 20. 100. in
  { pass; stop = Float.min 500. (pass +. width); reject }

(* The taps meet the specification at the band edges and between grid points:
   the checker's own search, in OCaml floats, may differ from nx's in the last
   bits, so it allows a billionth of δ. *)
let meets s =
  let rate = 1000. in
  let k = K.lowpass ~rate ~pass:s.pass ~stop:s.stop ~reject:s.reject in
  let t = Nx.to_array (K.taps Nx.float64 k) in
  let delta = (10. ** (-.s.reject /. 20.)) *. (1. +. 1e-9) in
  less float_exact ~than:delta
    (largest t ~lo:0. ~hi:(s.pass /. rate) (fun g -> Float.abs (g -. 1.)));
  less float_exact ~than:delta
    (largest t ~lo:(s.stop /. rate) ~hi:0.5 Float.abs)

let lowpass_specs =
  cases "lowpass meets specifications its grid alone missed"
    ~id:(Format.asprintf "%a" pp_spec)
    [
      { pass = 170.641; stop = 220.41; reject = 42.0757 };
      { pass = 378.01; stop = 429.219; reject = 51.7404 };
    ]
    meets

(* Specifications whose worst error lies between the points of a 64-point grid:
   there the grid reads 0.99973 δ and 0.99987 δ at the half-widths 86 and 47,
   whose true worst errors are 1.0005 δ and 1.00015 δ. A check that read the
   grid and the edges alone would deliver them. *)
let between_points =
  cases "lowpass meets specifications whose peaks fall between grid points"
    ~id:(Format.asprintf "%a" pp_spec)
    [
      { pass = 290.739357; stop = 324.029921; reject = 89.794841 };
      { pass = 211.246761; stop = 242.640288; reject = 49.968051 };
    ]
    meets

(* A transition of 1.5 at a rate of 1000 and 80 dB takes over 3000 taps: the
   check's memory is linear in them. Its gain is read here on a transform's grid
   and at the edges. *)
let long_lowpass =
  test "lowpass of thousands of taps" (fun () ->
      let rate = 1000. and pass = 400. and stop = 401.5 and reject = 80. in
      let k = K.lowpass ~rate ~pass ~stop ~reject in
      let t = K.taps Nx.float64 k in
      let len = Nx.dim 0 t in
      less int ~than:len 3000;
      let n = ref 1 in
      while !n < 16 * len do
        n := 2 * !n
      done;
      let g =
        Nx.to_array
          (Nx.magnitude Nx.float64 (Nx.rfft Nx.complex128 ~s:[ !n ] t))
      in
      let delta = (10. ** (-.reject /. 20.)) *. (1. +. 1e-9) in
      let tt = Nx.to_array t in
      let worst = ref 0. in
      Array.iteri
        (fun i g ->
          let f = Float.of_int i *. rate /. Float.of_int !n in
          if f <= pass then worst := Float.max !worst (Float.abs (g -. 1.));
          if f >= stop then worst := Float.max !worst g)
        g;
      worst :=
        Float.max !worst
          (Float.max
             (Float.abs (amplitude tt (pass /. rate) -. 1.))
             (Float.abs (amplitude tt (stop /. rate))));
      less float_exact ~than:delta !worst)

let lowpass_prop = prop "lowpass meets its specification" ~count:80 spec meets

(* [at k t] is [k] at the float64 position [t]. *)
let at k t = Nx.item [ 0 ] (K.eval k (Nx.create Nx.float64 [| 1 |] [| t |]))

let kernel_values =
  test "kernels are their formulas" (fun () ->
      equal float_exact 0.75 (at K.Triangle 0.25);
      equal float_exact 0. (at K.Triangle 1.);
      equal float_exact 1. (at K.Box (-0.5));
      equal float_exact 0. (at K.Box 0.5);
      let a = -0.5 in
      let keys x =
        let x = Float.abs x in
        if x <= 1. then
          ((a +. 2.) *. x *. x *. x) -. ((a +. 3.) *. x *. x) +. 1.
        else
          (a *. x *. x *. x)
          -. (5. *. a *. x *. x)
          +. (8. *. a *. x)
          -. (4. *. a)
      in
      List.iter
        (fun t -> equal (float 1e-15) (keys t) (at (K.Cubic a) t))
        [ 0.; 0.3; -1.; 1.7 ];
      let k = K.Sinc { cutoff = 0.25; half_width = 8.; window = K.Lanczos } in
      let sinc x =
        if x = 0. then 1. else Float.sin (Float.pi *. x) /. (Float.pi *. x)
      in
      List.iter
        (fun t ->
          equal (float 1e-15)
            (0.5 *. sinc (0.5 *. t) *. sinc (t /. 8.))
            (at k t))
        [ 0.; 1.5; -7.25 ];
      equal float_exact 0. (at k 8.))

(* Renormalised weights sum to one; away from the edges they are the
   kernel's. *)
let weights_sum =
  test "renormalised weights sum to one" (fun () ->
      let r =
        R.v ~grid:`Points ~edge:`Renormalise ~kernel:torchaudio ~by:(160, 441)
      in
      let w = R.weights r in
      equal int 160 (Array.length w);
      Array.iter
        (fun (_, w) -> equal (float 1e-15) 1. (Array.fold_left ( +. ) 0. w))
        w;
      equal int 160 (R.count r 441);
      equal int 1 (R.count r 1))

(* Whisper's upstream resampling, ffmpeg's swr at its defaults, as a call site:
   the kernel, phase 0's normalisation, a whole-sample mirror of the kernel's
   taps before the signal and a half-sample mirror after it. *)
let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* [swr_parts ~from x] is swr's resampling, its ratio, its extended signal, its
   first output and phase 0's sum. *)
let swr_parts ~from x =
  let g = gcd from 16_000 in
  let l = 16_000 / g and m = from / g in
  let factor = Float.min (16_000. *. 0.97 /. Float.of_int from) 1. in
  let taps = Float.to_int (Float.ceil (32. /. factor)) in
  let taps = taps + (taps land 1) in
  let s = Float.max 1. (Float.of_int m /. Float.of_int l) in
  let kernel =
    K.Sinc
      {
        cutoff = factor *. s /. 2.;
        half_width = Float.of_int taps /. 2. /. s;
        window = K.Kaiser 9.;
      }
  in
  let r = R.v ~grid:`Points ~edge:`Zero ~kernel ~by:(l, m) in
  let n = Nx.dim 0 x in
  (* Zeros before the mirror up to a multiple of [m] samples, so that the
     signal's first output is an output. *)
  let before = taps + ((m - (taps mod m)) mod m) in
  let e =
    Nx.extend `Symmetric_half
      (0, (Int.min n taps + 1) / 2)
      (Nx.pad
         [| (before - taps, 0) |]
         0.
         (Nx.extend `Symmetric_whole (taps, 0) x))
  in
  let w0 = Array.fold_left ( +. ) 0. (snd (R.weights r).(0)) in
  (r, (l, m), e, before * l / m, w0)

let swr_16k ~from x =
  let r, _, e, first, w0 = swr_parts ~from x in
  Nx.div_s
    (Nx.slice
       [ Nx.R (first, first + R.count r (Nx.dim 0 x)) ]
       (Alda.resample r e))
    w0

let swr_golden = lazy (Nx_io.load_npz "golden/swr.npz")

let swr_entry dt key =
  match Nx_io.Archive.find key (Lazy.force swr_golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/swr.npz has no %s" key

(* PyAV's swr on three signals, from golden/swr.npz
   (contrib/alda/test/gen/swr.py). swr computes in float32: it rounds each
   weight, divided by phase 0's sum, to float32 and adds an output's [T]
   products in float32, then the result is a float32. So its output is within
   [(T + 1) u Σ |wᵢ xᵢ| + u |y|] of the exact composition, which alda computes
   at float64 on the same samples. *)
let swr =
  cases "Whisper's swr call site is PyAV's" ~id:string_of_int [ 0; 1; 2 ]
    (fun i ->
      let key s = Printf.sprintf "%d.%s" i s in
      let from =
        Int64.to_int (Nx.item [ 0 ] (swr_entry Nx.int64 (key "rate")))
      in
      let x = Nx.cast Nx.float64 (swr_entry Nx.float32 (key "x")) in
      let want = floats (swr_entry Nx.float32 (key "y")) in
      let got = floats (swr_16k ~from x) in
      equal int (Array.length want) (Array.length got);
      let r, by, e, first, w0 = swr_parts ~from x in
      let taps, sums =
        magnitudes ~by (R.weights r) (floats e) first (Array.length want)
      in
      within
        ~bound:(fun j ->
          u32
          *. ((Float.of_int (taps + 1) *. sums.(j) /. w0) +. Float.abs want.(j)))
        want got)

(* Kernels at positions in data *)

(* [f32 x] is [x] rounded to float32. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let pp_window ppf = function
  | K.Kaiser b -> Format.fprintf ppf "Kaiser %.17g" b
  | K.Hann -> Format.pp_print_string ppf "Hann"
  | K.Lanczos -> Format.pp_print_string ppf "Lanczos"

let pp_kernel ppf = function
  | K.Box -> Format.pp_print_string ppf "Box"
  | K.Triangle -> Format.pp_print_string ppf "Triangle"
  | K.Cubic a -> Format.fprintf ppf "Cubic %.17g" a
  | K.Sinc { cutoff; half_width; window } ->
      Format.fprintf ppf "Sinc {cutoff %.17g; half_width %.17g; %a}" cutoff
        half_width pp_window window

(* Every kernel, half-widths whole or float32 fractions. *)
let kernel =
  let open Gen in
  let window =
    one_of
      [
        map (fun b -> K.Kaiser b) (float_range 0. 20.);
        constant K.Hann;
        constant K.Lanczos;
      ]
  in
  let sinc =
    let* cutoff = float_range 0.05 0.5 in
    let* half_width =
      one_of [ map Float.of_int (int_range 1 16); map f32 (float_range 1. 16.) ]
    in
    let+ window = window in
    K.Sinc { cutoff; half_width; window }
  in
  with_pp pp_kernel
  @@ frequency
       [
         (1, constant K.Box);
         (1, constant K.Triangle);
         (1, map (fun a -> K.Cubic a) (float_range (-1.) 0.));
         (3, sinc);
       ]

(* The positions where [k] or one of its first three derivatives jumps. *)
let breaks = function
  | K.Box -> [ -0.5; 0.5 ]
  | K.Triangle -> [ -1.; 0.; 1. ]
  | K.Cubic _ -> [ -2.; -1.; 0.; 1.; 2. ]
  | K.Sinc { half_width = h; _ } -> [ -.h; h ]

type point = { k : K.t; tau : float }

let pp_point ppf p = Format.fprintf ppf "{%a at %.17g}" pp_kernel p.k p.tau

(* A kernel and a float32 position in its support or within [1] of it, a break a
   quarter of the time. *)
let point =
  let open Gen in
  with_pp pp_point
  @@
  let* k = kernel in
  let h = K.half_width k in
  let+ tau =
    frequency
      [
        (3, map f32 (float_range (-.h -. 1.) (h +. 1.))); (1, of_list (breaks k));
      ]
  in
  { k; tau }

let in_float32 =
  prop "eval in float32 is eval in float64 within float32's rounding" ~count:300
    point (fun p ->
      cover ~share:0.1 "Kaiser"
        (match p.k with
        | K.Sinc { window = K.Kaiser _; _ } -> true
        | _ -> false);
      let got =
        Nx.item [ 0 ] (K.eval p.k (Nx.create Nx.float32 [| 1 |] [| p.tau |]))
      in
      equal (float 1e-5) (at p.k p.tau) got)

(* Traced, positions lie where the compiled program runs; its arithmetic may
   round differently. *)
let compiled =
  test "eval compiled is eval" (fun () ->
      let tau = Nx.linspace Nx.float32 (-5.) 5. 101 in
      List.iter
        (fun k ->
          let want = Nx.to_array (K.eval k tau) in
          let got = Nx.to_array (Rune.jit' (K.eval k) tau) in
          equal
            ~msg:(Format.asprintf "%a" pp_kernel k)
            (array (float 1e-6))
            want got)
        [
          K.Box;
          K.Triangle;
          K.Cubic (-0.5);
          K.Sinc { cutoff = 0.4; half_width = 4.; window = K.Kaiser 8.6 };
          K.Sinc { cutoff = 0.5; half_width = 3.; window = K.Lanczos };
        ])

(* The derivative against a one-sided difference of second order, [δ] to the
   right, whose error is about [δ² |k'''| / 3]: where a break lies in [(τ, τ +
   2δ]] the difference reads across it, so the case is dropped. *)
let delta = 1e-4

let derivative_examples =
  let kaiser =
    K.Sinc { cutoff = 0.4; half_width = 4.; window = K.Kaiser 8.6 }
  in
  [
    { k = K.Triangle; tau = 0. };
    { k = K.Triangle; tau = -1. };
    { k = K.Box; tau = -0.5 };
    { k = K.Cubic (-0.5); tau = 0. };
    { k = kaiser; tau = -4. };
    { k = kaiser; tau = 0. };
  ]

let derivative =
  prop "eval's derivative is the one from the right" ~count:300
    ~examples:derivative_examples point (fun p ->
      let k = p.k and t = p.tau in
      assume
        (not
           (List.exists (fun b -> t < b && b <= t +. (2. *. delta)) (breaks k)));
      cover ~share:0.1 "at a break" (List.mem t (breaks k));
      let fd =
        ((-3. *. at k t)
        +. (4. *. at k (t +. delta))
        -. at k (t +. (2. *. delta)))
        /. (2. *. delta)
      in
      let x = Nx.create Nx.float64 [| 1 |] [| t |] in
      let _, forward = Rune.jvp' (K.eval k) x (Nx.ones_like x) in
      let reverse = Rune.grad' (fun x -> Nx.sum (K.eval k x)) x in
      let close = float_rel ~rel:1e-4 ~abs:1e-5 in
      equal ~msg:"forward" close fd (Nx.item [ 0 ] forward);
      equal ~msg:"reverse" close fd (Nx.item [ 0 ] reverse))

(* The derivatives at [-h], where a Kaiser-windowed sinc jumps from [0], against
   one-sided differences of second order [δ] to the right: the first's error is
   about [δ² |k‴| / 3], the second's [11 δ² |k⁗| / 12]. *)
let at_edge =
  cases "a Kaiser sinc's first two derivatives at -h are the right's"
    ~id:(Format.asprintf "%a" pp_kernel)
    [
      K.Sinc { cutoff = 0.4; half_width = 4.; window = K.Kaiser 8.6 };
      K.Sinc { cutoff = 0.25; half_width = 8.; window = K.Kaiser 20. };
      K.Sinc { cutoff = 0.5; half_width = 2.; window = K.Kaiser 0. };
      K.Sinc { cutoff = 0.1; half_width = 16.; window = K.Kaiser 40. };
    ]
    (fun k ->
      let h = K.half_width k in
      let t = -.h and d = 1e-3 in
      let f i = at k (t +. (Float.of_int i *. d)) in
      let fd1 = ((-3. *. f 0) +. (4. *. f 1) -. f 2) /. (2. *. d) in
      let fd2 =
        ((2. *. f 0) -. (5. *. f 1) +. (4. *. f 2) -. f 3) /. (d *. d)
      in
      let x = Nx.create Nx.float64 [| 1 |] [| t |] in
      let one = Nx.ones_like x in
      let d1 y = snd (Rune.jvp' (K.eval k) y one) in
      let d2 = snd (Rune.jvp' d1 x one) in
      let g y = Rune.grad' (fun y -> Nx.sum (K.eval k y)) y in
      let gg = Rune.grad' (fun y -> Nx.sum (g y)) x in
      let close = float_rel ~rel:1e-4 ~abs:1e-6 in
      equal ~msg:"first" close fd1 (Nx.item [ 0 ] (d1 x));
      equal ~msg:"second, forward" close fd2 (Nx.item [ 0 ] d2);
      equal ~msg:"second, reverse" close fd2 (Nx.item [ 0 ] gg))

(* Kaiser's window as Bessel functions, [I₀(a) / I₀(β) = (i0e a / i0e β) e^(a -
   β)] for [a = β √(1 - r²)], at float64. *)
let bessel beta r =
  let a = beta *. Float.sqrt (Float.max 0. (1. -. (r *. r))) in
  let i0e v = Nx.item [] (Nx.i0e (Nx.scalar Nx.float64 v)) in
  i0e a /. i0e beta *. Float.exp (a -. beta)

(* The series and the Bessel form each round: the series' [K ≤ β + 14] steps of
   Horner over positive terms about [2K u], its coefficients and truncation a
   few [u]; the Bessel form's [e^(a - β)] about [β u] from [a]'s and [a - β]'s
   roundings, [i0e] a few [u]. So [(3β + 100) u] bounds their difference, and
   one more [u] the product with the sinc. A Kaiser of [β = 0] is [1], so it
   gives the sinc alone. *)
let as_bessel =
  let open Gen in
  let gen =
    let* beta = frequency [ (4, float_range 0. 20.); (1, constant 40.) ] in
    let* h = map Float.of_int (int_range 1 16) in
    let+ tau = frequency [ (4, float_range (-.h) h); (1, constant (-.h)) ] in
    (beta, h, f32 (Float.min tau (Float.pred h)))
  in
  prop "Kaiser's window is its Bessel form" ~count:300 gen
    (fun (beta, h, tau) ->
      cover ~share:0.1 "beta 40" (beta = 40.);
      cover ~share:0.1 "at -h" (tau = -.h);
      let sinc w =
        K.Sinc { cutoff = 0.3; half_width = h; window = K.Kaiser w }
      in
      let want = at (sinc 0.) tau *. bessel beta (tau /. h) in
      let u = epsilon_float /. 2. in
      let bound = ((3. *. beta) +. 101.) *. u *. Float.abs want in
      less float_exact ~than:(bound +. Float.min_float)
        (Float.abs (at (sinc beta) tau -. want)))

type ratio = { kernel : K.t; up : int; down : int; grid : [ `Points | `Cells ] }

let pp_ratio ppf r =
  Format.fprintf ppf "{%a; by (%d, %d); %s}" pp_kernel r.kernel r.up r.down
    (match r.grid with `Points -> "points" | `Cells -> "cells")

let ratio =
  let open Gen in
  with_pp pp_ratio
  @@
  let* kernel = kernel in
  let* up = int_range 1 40 in
  let* down = int_range 1 40 in
  let+ grid = of_list [ `Points; `Cells ] in
  { kernel; up; down; grid }

(* [fdiv a b] is [⌊a / b⌋] for [b > 0]. *)
let fdiv a b = if a >= 0 then a / b else -((-a + b - 1) / b)

(* Each weight is [k(τ) / s] at [τ = (c_j - i) / s], both rounded once from
   their exact values. *)
let weights_bits =
  prop "resampling's float64 weights are eval's bits" ~count:200 ratio (fun c ->
      let g = gcd c.up c.down in
      let l = c.up / g and m = c.down / g in
      cover ~share:0.3 "shrinking" (m > l);
      let r = R.v ~grid:c.grid ~edge:`Zero ~kernel:c.kernel ~by:(l, m) in
      let num j =
        match c.grid with `Points -> j * m | `Cells -> (((2 * j) + 1) * m) - l
      in
      let den = match c.grid with `Points -> l | `Cells -> 2 * l in
      let s, sl, sm =
        if m > l then (Float.of_int m /. Float.of_int l, l, m) else (1., 1, 1)
      in
      Array.iteri
        (fun j (off, w) ->
          let first = fdiv (num j) den + off in
          let tau =
            Array.init (Array.length w) (fun i ->
                Float.of_int ((num j - ((first + i) * den)) * sl)
                /. Float.of_int (den * sm))
          in
          let v =
            Nx.to_array
              (K.eval c.kernel
                 (Nx.create Nx.float64 [| Array.length tau |] tau))
          in
          equal (array int64)
            (Array.map (fun v -> Int64.bits_of_float (v /. s)) v)
            (Array.map Int64.bits_of_float w))
        (R.weights r))

let errors =
  group "misuse"
    [
      test "a ratio of zero" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.Resample.v") (fun () ->
              R.v ~grid:`Points ~edge:`Zero ~kernel:K.Box ~by:(0, 1)));
      test "a stop band below the pass band" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.Kernel.lowpass") (fun () ->
              K.lowpass ~rate:16_000. ~pass:4000. ~stop:3000. ~reject:60.));
      test "finish before a feed" (fun () ->
          let r = R.v ~grid:`Points ~edge:`Zero ~kernel:K.Box ~by:(1, 2) in
          raises_match (Exn.invalid_arg ~sub:"Alda.Resample.finish") (fun () ->
              R.finish r
                (R.start
                  : ((float, Nx.float32_elt, Nx.host) Nx.t, Nx.host) R.State.t)));
    ]

(* A phase reads the inputs [i] with [-s h ≤ c_j - i < s h], so it holds at most
   [⌈2 s h⌉] weights. *)
type phases = { k : K.t; l : int; m : int; cells : bool }

let pp_phases ppf c =
  Format.fprintf ppf "{%a; by (%d, %d); %s}" pp_kernel c.k c.l c.m
    (if c.cells then "cells" else "points")

let phase_size =
  let case =
    let open Gen in
    with_pp pp_phases
    @@
    let* k = kernel in
    let+ l = int_range 1 40 and+ m = int_range 1 40 and+ cells = bool in
    { k; l; m; cells }
  in
  prop "a phase holds at most ⌈2 s h⌉ weights" ~count:300 case (fun c ->
      cover ~share:0.3 "shrinking" (c.m > c.l);
      let grid = if c.cells then `Cells else `Points in
      let r = R.v ~grid ~edge:`Zero ~kernel:c.k ~by:(c.l, c.m) in
      let s = Float.max 1. (Float.of_int c.m /. Float.of_int c.l) in
      let most = Float.to_int (Float.ceil (2. *. s *. K.half_width c.k)) in
      Array.iteri
        (fun j (_, w) ->
          at_most
            ~msg:(Printf.sprintf "phase %d" j)
            int ~than:most (Array.length w))
        (R.weights r))

let () =
  exit
    (run "alda.resample"
       [
         torchaudio64;
         torchaudio32;
         images;
         stream_law;
         in_scan;
         swr;
         soxr_spec;
         lowpass_specs;
         between_points;
         long_lowpass;
         lowpass_prop;
         kernel_values;
         in_float32;
         compiled;
         derivative;
         at_edge;
         as_bessel;
         weights_bits;
         weights_sum;
         phase_size;
         errors;
       ])
