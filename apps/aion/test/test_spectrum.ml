(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectrum codec's laws on small codecs of Spectrum.init and on decoders
   whose output is known, and its agreement with upstream's code on a designed
   few-channel codec, eager and compiled. The preparation's rules hold exactly
   where float32 computes them exactly, its formulas within nx's error bounds,
   and the network's values within the rounding ensemble's spread (Spread). *)

open Windtrap
open Aion

let f32 = Bounds.f32
let ints a = Nx.to_array (Nx.cast Nx.int64 a) |> Array.map Int64.to_int
let floats a = Nx.to_array (Nx.cast Nx.float64 a)
let bools a = Nx.to_array a
let tensor shape a = Nx.create Nx.float32 shape a
let float32_max = Int32.float_of_bits 0x7f7fffffl
let u = ldexp 1. (-24)

(* AION-1's layout at a few channels and 64 grid wavelengths: 2 latent
   vectors. *)
let grid_length = 64
let grid_at k = f32 (3500. +. (0.8 *. float_of_int k))

let arch : Spectrum.arch =
  {
    encoder = [ (4, 1); (4, 1); (4, 2); (8, 1) ];
    decoder = [ (4, 1); (4, 1); (4, 1) ];
    output = 1;
    grid = grid_length;
  }

let random seed = Spectrum.init ~key:(Nx.Rng.key seed) arch

(* A decoder whose weights are all 0 but the bias [(y, z)] of its last
   transposed convolution, layer 10: it emits the signal [y] and the mask logit
   [z] everywhere. *)
let constant (y, z) =
  let bias = tensor [| 2 |] [| y; z |] in
  Nx.Ptree.map
    (Nx.Ptree.instantiate (module Spectrum))
    (fun p a ->
      match Nx.Ptree.Path.segments p with
      | [ Field "decoder"; Index 10; Field "b" ] -> Nx.cast (Nx.dtype a) bias
      | Field ("decoder" | "post_quant") :: _ -> Nx.zeros_like a
      | _ -> a)
    (random 0)

(* Spectra *)

let spectrum ?mask ?ivar flux wavelength =
  let b = Array.length flux and m = Array.length flux.(0) in
  let t a = tensor [| b; m |] (Array.concat (Array.to_list a)) in
  let fill v = Array.map (Array.map (fun _ -> v)) flux in
  let mask = Option.value mask ~default:(fill false) in
  let ivar = Option.value ivar ~default:(fill 1.) in
  Spectrum.
    {
      flux = t flux;
      ivar = t ivar;
      mask = Nx.create Nx.bool [| b; m |] (Array.concat (Array.to_list mask));
      wavelength = t wavelength;
    }

let flat b m v = Array.make b (Array.make m v)

(* Rows over part of the grid's range, wavelengths increasing, some repeated,
   with NaN, infinite, negative and masked values. *)
let rows_of ~infinities =
  Gen.(
    let* b = int_range 1 3 in
    let* m = int_range 2 12 in
    let row g = array ~size:(constant m) g in
    let specials =
      if infinities then [ Float.nan; Float.infinity; Float.neg_infinity; 0. ]
      else [ Float.nan; 0. ]
    in
    let flux =
      frequency [ (8, map f32 (float_range (-20.) 60.)); (1, of_list specials) ]
    in
    let ivar =
      frequency
        [
          (6, map f32 (float_range 0. 150.)); (1, of_list [ Float.nan; -1.; 0. ]);
        ]
    in
    (* Free wavelengths, or the grid's own at increasing indices. *)
    let scan start next steps =
      let acc = ref start in
      Array.map
        (fun d ->
          let w = !acc in
          acc := next !acc d;
          w)
        steps
    in
    let free =
      let* s = map f32 (float_range 3450. 3520.) in
      let+ steps =
        row (frequency [ (5, map f32 (float_range 0.1 6.)); (1, constant 0.) ])
      in
      scan s (fun w d -> f32 (w +. d)) steps
    in
    let on_grid =
      let* j = int_range (-5) 40 in
      let+ steps = row (int_range 0 3) in
      Array.map grid_at (scan j ( + ) steps)
    in
    let+ fs = array ~size:(constant b) (row flux)
    and+ vs = array ~size:(constant b) (row ivar)
    and+ ms =
      array ~size:(constant b)
        (row (frequency [ (4, constant false); (1, constant true) ]))
    and+ ws = array ~size:(constant b) (one_of [ free; on_grid ]) in
    (fs, vs, ms, ws))

let rows = rows_of ~infinities:false

(* A flux as preparation reads it: NaN as 0, infinities as float32's largest
   finite values. *)
let fix_flux x =
  if Float.is_nan x then 0.
  else Float.max (-.float32_max) (Float.min float32_max x)

let fix_ivar x = if Float.is_nan x then 0. else Float.max 0. (Float.min 100. x)

(* A row's infinities after its first become NaN: two of one sign overflow
   float32's sum, where nx's bounds do not hold. *)
let one_infinity row =
  let seen = ref false in
  Array.map
    (fun x ->
      if Float.abs x <> Float.infinity then x
      else if !seen then Float.nan
      else begin
        seen := true;
        x
      end)
    row

let within ~what bound expected got =
  if not (Float.abs (got -. expected) <= bound) then
    failf "%s: %.9g, expected %.9g within %.3g" what got expected bound

(* Preparation *)

(* One row's normalisation in float64, and its bound: nx's sum bound γ(β(n))
   Σ|x| (β and γ as Nx_kernel's section on bounds states them), then a rounding
   each for the division, the floor's 0.1, the addition of 1 and log10's 3. *)
let norm64 flux mask =
  let s = ref 0. and a = ref 0. and c = ref 1. in
  Array.iteri
    (fun i x ->
      if not mask.(i) then begin
        s := !s +. fix_flux x;
        a := !a +. Float.abs (fix_flux x);
        c := !c +. 1.
      end)
    flux;
  let n = Array.length flux in
  let k = float_of_int ((((5 * min n 256) + 3) / 4) + ((n + 127) / 128)) in
  let gamma = k *. u /. (1. -. (k *. u)) in
  let mean = !s /. !c in
  let e_mean =
    (((!a *. gamma) +. (Float.abs !s *. u)) /. !c) +. (Float.abs mean *. u)
  in
  let w = Float.max mean 0.1 +. 1. in
  let e_w = e_mean +. (0.1 *. u) +. (w *. u) in
  ( Float.log10 w,
    (e_w /. (w *. log 10.)) +. (Float.abs (Float.log10 w) *. 3. *. u) )

let normalisation =
  prop "the normalisation is log10 (1 + max mean 0.1)"
    (rows_of ~infinities:true) (fun (fs, vs, ms, ws) ->
      let fs = Array.map one_infinity fs in
      let x = spectrum ~mask:ms ~ivar:vs fs ws in
      let got = floats (Spectrum.normalisation x) in
      Array.iteri
        (fun i f ->
          cover "NaN flux" (Array.exists Float.is_nan f);
          cover "infinite flux"
            (Array.exists (fun x -> Float.abs x = Float.infinity) f);
          cover "masked pixel" (Array.exists Fun.id ms.(i));
          let want, bound = norm64 f ms.(i) in
          cover "mean below 0.1" (want = Float.log10 1.1);
          within ~what:"normalisation" bound want got.(i))
        fs)

(* A float16 spectrum clamps an infinite flux to float16's largest finite value,
   which float32's would overflow. *)
let narrow_infinity =
  test "an infinite flux is its dtype's largest finite value" (fun () ->
      let row f =
        let x =
          spectrum [| [| f; 1.; 2. |] |] [| [| 3600.; 3601.; 3602. |] |]
        in
        let h t = Nx.cast Nx.float16 t in
        Spectrum.normalisation
          {
            x with
            flux = h x.flux;
            ivar = h x.ivar;
            wavelength = h x.wavelength;
          }
      in
      let max16 = (Nx.Dtype.float_format Nx.float16).max_finite in
      equal (array float_exact)
        (floats (row max16))
        (floats (row Float.infinity)))

(* The rules: inputs equal after a rule prepare equal. *)
let prepared_equal t a b =
  let pa, ka = Spectrum.prepare t a and pb, kb = Spectrum.prepare t b in
  equal (array int) (ints ka) (ints kb);
  equal (array float_exact) (floats pa) (floats pb)

let rules =
  let t = random 1 in
  group "rules"
    [
      prop "NaN fluxes prepare as 0, infinities as float32's largest"
        (rows_of ~infinities:true) (fun (fs, vs, ms, ws) ->
          let fs = Array.map one_infinity fs in
          cover "NaN flux" (Array.exists (Array.exists Float.is_nan) fs);
          cover "infinite flux"
            (Array.exists
               (Array.exists (fun x -> Float.abs x = Float.infinity))
               fs);
          prepared_equal t
            (spectrum ~mask:ms ~ivar:vs fs ws)
            (spectrum ~mask:ms ~ivar:vs (Array.map (Array.map fix_flux) fs) ws));
      prop "inverse variances prepare NaN as 0, clamped to [0, 100]" rows
        (fun (fs, vs, ms, ws) ->
          cover "NaN inverse variance"
            (Array.exists (Array.exists Float.is_nan) vs);
          cover "an inverse variance above 100"
            (Array.exists (Array.exists (fun v -> v > 100.)) vs);
          prepared_equal t
            (spectrum ~mask:ms ~ivar:vs fs ws)
            (spectrum ~mask:ms ~ivar:(Array.map (Array.map fix_ivar) vs) fs ws));
      prop "the first token counts the levels below the normalisation" rows
        (fun (fs, vs, ms, ws) ->
          let x = spectrum ~mask:ms ~ivar:vs fs ws in
          let levels = floats (Spectrum.levels t) in
          let count y =
            min 1023
              (Array.fold_left (fun n l -> if l < y then n + 1 else n) 0 levels)
          in
          equal (array int)
            (Array.map count (floats (Spectrum.normalisation x)))
            (ints (Nx.slice [ A; I 0 ] (Spectrum.encode t x))));
    ]

(* Upstream's interpolation, in float64: the interval [x(i), x(i+1)] with [i]
   the number of wavelengths below the key, less one, kept in [0, m - 2]; 0
   outside [x(0), x(m - 1)]. *)
let interval x key =
  let m = Array.length x in
  let below = Array.fold_left (fun n v -> if v < key then n + 1 else n) 0 x in
  max 0 (min (m - 2) (below - 1))

let interp x y key =
  let m = Array.length x in
  if key < x.(0) || key > x.(m - 1) then 0.
  else
    let i = interval x key in
    let dx = x.(i) -. x.(i + 1) in
    let slope = if dx = 0. then 0. else (y.(i) -. y.(i + 1)) /. dx in
    y.(i) +. ((key -. x.(i)) *. slope)

(* The scale [s] of the token value [w], and its bound: pow's 4 floats, the
   subtraction's and 0.1's rounding. *)
let scale_of w =
  let s = Float.max ((10. ** w) -. 1.) 0.1 in
  (s, (9. *. u *. (10. ** w)) +. (u *. s) +. (0.1 *. u))

(* Against float64 with the pixels' values exact. [prepare]'s signal at a key in
   [x(i), x(i+1)] is [asinh (y(i) + (key - x(i)) · slope)], with [y(j) = (f(j) /
   s - 1) · 0.2]. Its bound sums the larger of [y(i)]'s and [y(i+1)]'s (s's
   relative bound and 2 roundings of [0.2 |f| / s], one of [0.2 |f / s - 1|]),
   the step's 6 roundings of [|ynew - y(i)|] ([key - x(i)] and [x(i) - x(i+1)]
   are exact), the sum's 2 and asinh's 3. [decode]'s flux at a key is [(ynew +
   1) · s] on the line through the grid's fluxes [f(j)]; its bound sums 8
   roundings of [|F - f(i)|], the sum's, the addition's and the product's [6 u
   |F| + 4 u s], and the grid fluxes' own rounding, twice. *)
let all_good = lazy (constant (0., -1.))

let interpolation =
  let t = random 3 in
  group "interpolation"
    [
      prop "prepare reads each grid wavelength on the line through its pixels"
        rows (fun (fs, vs, ms, ws) ->
          let x = spectrum ~mask:ms ~ivar:vs fs ws in
          let p, code = Spectrum.prepare t x in
          let p = floats p and code = ints code in
          let levels = floats (Spectrum.levels t) in
          Array.iteri
            (fun r w ->
              let s, e_s = scale_of levels.(code.(r)) in
              let f = Array.map fix_flux fs.(r) in
              let y = Array.map (fun f -> ((f /. s) -. 1.) *. 0.2) f in
              let e =
                Array.map
                  (fun f ->
                    (0.2 *. Float.abs f /. s *. ((e_s /. s) +. (2. *. u)))
                    +. (0.2 *. u *. Float.abs ((f /. s) -. 1.)))
                  f
              in
              let m = Array.length w in
              cover "a key between pixels of unequal signals"
                (Array.exists
                   (fun k ->
                     let key = grid_at k in
                     key > w.(0)
                     && key < w.(m - 1)
                     && y.(interval w key) <> y.(interval w key + 1))
                   (Array.init grid_length Fun.id));
              for k = 0 to grid_length - 1 do
                let key = grid_at k and got = p.(((r * grid_length) + k) * 2) in
                if key < w.(0) || key > w.(m - 1) then equal float_exact 0. got
                else
                  let i = interval w key in
                  let ynew = interp w y key in
                  let want = Float.asinh ynew in
                  let bound =
                    Float.max e.(i) e.(i + 1)
                    +. (6. *. u *. Float.abs (ynew -. y.(i)))
                    +. (2. *. u *. Float.abs ynew)
                    +. (3. *. u *. Float.abs want)
                  in
                  within
                    ~what:(Printf.sprintf "row %d at %g" r key)
                    bound want got
              done)
            ws);
      prop "decode reads a wavelength on the line through the grid's flux"
        Gen.(
          let* tokens = array ~size:(constant 3) (int_range 0 1023) in
          let+ keys =
            array ~size:(int_range 1 9)
              (frequency
                 [
                   (4, map f32 (float_range 3490. 3560.));
                   (1, map grid_at (int_range 0 (grid_length - 1)));
                 ])
          in
          (tokens, keys))
        (fun (tokens, keys) ->
          let k =
            Nx.create Nx.int64 [| 1; 3 |] (Array.map Int64.of_int tokens)
          in
          let gx = Array.init grid_length grid_at in
          let fg = floats (Spectrum.decode t k).flux in
          let n = Array.length keys in
          let d = Spectrum.decode ~wavelength:(tensor [| 1; n |] keys) t k in
          let flux = floats d.flux and mask = bools d.mask in
          (* A decoder whose every pixel is good, so that a bad pixel shows the
             rule for wavelengths outside the grid. *)
          let good =
            bools
              (Spectrum.decode
                 ~wavelength:(tensor [| 1; n |] keys)
                 (Lazy.force all_good) k)
                .mask
          in
          let s, _ = scale_of (floats (Spectrum.levels t)).(tokens.(0)) in
          Array.iteri
            (fun j key ->
              cover "a key between grid wavelengths"
                (key > gx.(0)
                && key < gx.(grid_length - 1)
                && not (Array.mem key gx));
              cover "a key below the grid" (key < gx.(0));
              cover "a key above the grid" (key > gx.(grid_length - 1));
              let outside = key < gx.(0) || key > gx.(grid_length - 1) in
              equal bool outside good.(j);
              if outside then begin
                equal float_exact 0. flux.(j);
                equal bool true mask.(j)
              end
              else
                let i = interval gx key in
                let want = interp gx fg key in
                let bound =
                  (8. *. u *. Float.abs (want -. fg.(i)))
                  +. (6. *. u *. Float.abs want)
                  +. (4. *. u *. s)
                  +. 2. *. u
                     *. Float.max (Float.abs fg.(i)) (Float.abs fg.(i + 1))
                in
                within ~what:(Printf.sprintf "at %g" key) bound want flux.(j))
            keys);
    ]

(* A row of one flux [f] and inverse variance [v] has one signal and one noise,
   which resampling keeps at each grid wavelength in the row's range, exactly,
   and 0 outside. Each is its formula of the token's value within s's bound
   carried through the division, then the roundings of the subtraction, the
   products and asinh's 2 floats. *)
let constant_rows =
  let t = random 2 in
  prop "a row of one flux prepares to its formula inside its range, 0 outside"
    Gen.(
      let* f = map f32 (float_range (-20.) 60.) in
      let* v = map f32 (float_range 0. 150.) in
      let* j = int_range (-5) 40 in
      let+ m = int_range 2 12 in
      (f, v, j, m))
    (fun (f, v, j, m) ->
      let ws = Array.init m (fun i -> grid_at (j + (2 * i))) in
      let x =
        spectrum ~ivar:[| Array.make m v |] [| Array.make m f |] [| ws |]
      in
      let p, code = Spectrum.prepare t x in
      let p = floats p in
      let s, e_s = scale_of (floats (Spectrum.levels t)).((ints code).(0)) in
      let signal = Float.asinh (((f /. s) -. 1.) *. 0.2) in
      let e_signal =
        (0.2 *. Float.abs f /. s *. ((e_s /. s) +. (2. *. u)))
        +. (0.2 *. u *. Float.abs ((f /. s) -. 1.))
        +. (3. *. u *. Float.abs signal)
      in
      let r = Float.sqrt (Float.min 100. v) /. s in
      let noise = Float.asinh (r *. 0.2) in
      let e_noise =
        (0.2 *. r *. ((e_s /. s) +. (3. *. u))) +. (3. *. u *. Float.abs noise)
      in
      let inside k = grid_at k >= ws.(0) && grid_at k <= ws.(m - 1) in
      let first = lazy (List.find inside (List.init grid_length Fun.id)) in
      cover "keys below the row" (j > 0);
      cover "keys above the row" (ws.(m - 1) < grid_at (grid_length - 1));
      for k = 0 to grid_length - 1 do
        let sg = p.(2 * k) and ns = p.((2 * k) + 1) in
        if inside k then begin
          within ~what:(Printf.sprintf "signal at %d" k) e_signal signal sg;
          within ~what:(Printf.sprintf "noise at %d" k) e_noise noise ns;
          equal float_exact p.(2 * Lazy.force first) sg;
          equal float_exact p.((2 * Lazy.force first) + 1) ns
        end
        else begin
          equal float_exact 0. sg;
          equal float_exact 0. ns
        end
      done)

(* At the grid's first wavelength, a row whose first two pixels share it reads
   the first pixel's value, where upstream's slope is NaN: as the row without
   the second pixel, masked so that the normalisations agree. *)
let repeated_first =
  test "a repeated first wavelength reads its first pixel" (fun () ->
      let t = random 5 in
      let g0 = grid_at 0 and g1 = grid_at 9 in
      let repeated =
        spectrum
          ~mask:[| [| false; true; false |] |]
          [| [| 2.; 30.; 4. |] |]
          [| [| g0; g0; g1 |] |]
      in
      let single = spectrum [| [| 2.; 4. |] |] [| [| g0; g1 |] |] in
      let first x =
        floats (Nx.slice [ I 0; I 0 ] (fst (Spectrum.prepare t x)))
      in
      equal (array float_exact) (first single) (first repeated))

(* Padding *)

let pad =
  test "pad fills each field with its pad value to the modality's length"
    (fun () ->
      equal int 7808 (Spectrum.length Modality.Desi);
      equal int 4800 (Spectrum.length Modality.Sdss);
      let x = spectrum [| [| 1.; 2. |] |] [| [| 3600.; 3601. |] |] in
      let p = Spectrum.pad Modality.Sdss x in
      let from1 a = Nx.to_array (Nx.slice [ I 0; R (1, 4) ] a) in
      equal int 4800 (Nx.dim 1 p.flux);
      equal (array float_exact) [| 2.; 0.; 0. |]
        (from1 (Nx.cast Nx.float64 p.flux));
      equal (array float_exact) [| 1.; 0.; 0. |]
        (from1 (Nx.cast Nx.float64 p.ivar));
      equal (array bool) [| false; true; true |] (from1 p.mask);
      equal (array float_exact)
        [| 3601.; 99999.; 99999. |]
        (from1 (Nx.cast Nx.float64 p.wavelength)))

(* Rows with a length each, the pixels past it holding Multimodal Universe's -1
   wavelengths or NaN, which an encode that read them would refuse. *)
let with_lengths =
  Gen.(
    let* fs, vs, ms, ws = rows in
    let m = Array.length ws.(0) in
    let+ ls =
      array
        ~size:(constant (Array.length ws))
        (frequency [ (2, int_range 1 m); (1, of_list [ 1; m ]) ])
    and+ fill = of_list [ -1.; Float.nan ] in
    let cut r l = Array.mapi (fun j x -> if j < l then x else fill) r in
    ( Array.mapi (fun i f -> cut f ls.(i)) fs,
      vs,
      ms,
      Array.mapi (fun i w -> cut w ls.(i)) ws,
      ls ))

let lengths =
  let t = random 9 in
  prop "a row encodes as its first lengths.(i) pixels alone" with_lengths
    (fun (fs, vs, ms, ws, ls) ->
      let m = Array.length ws.(0) in
      cover "no pixel cut" (Array.exists (( = ) m) ls);
      cover "one pixel kept" (Array.exists (( = ) 1) ls);
      cover "rows of two lengths" (Array.exists (fun l -> l <> ls.(0)) ls);
      let x = spectrum ~mask:ms ~ivar:vs fs ws in
      let lengths =
        Nx.create Nx.int64 [| Array.length ls |] (Array.map Int64.of_int ls)
      in
      let got = Spectrum.encode t (Spectrum.pad ~lengths Modality.Sdss x) in
      Array.iteri
        (fun i l ->
          let row a = [| Array.sub a.(i) 0 l |] in
          let alone =
            spectrum ~mask:(row ms) ~ivar:(row vs) (row fs) (row ws)
          in
          equal (array int)
            (ints (Spectrum.encode t (Spectrum.pad Modality.Sdss alone)))
            (ints (Nx.slice [ R (i, i + 1) ] got)))
        ls)

(* Encoding and decoding *)

let shapes =
  test "encode gives 1 + L / 32 tokens; decode reads the grid or wavelengths"
    (fun () ->
      let t = random 3 in
      let w = Array.init 50 (fun i -> f32 (3480. +. (0.9 *. float_of_int i))) in
      let x = spectrum (flat 3 50 2.) (Array.make 3 w) in
      let k = Spectrum.encode t (Spectrum.pad Modality.Desi x) in
      equal (array int) [| 3; 3 |] (Nx.shape k);
      let d = Spectrum.decode t k in
      equal (array int) [| 3; grid_length |] (Nx.shape d.flux);
      equal (array int) [| 3; grid_length |] (Nx.shape d.mask);
      equal (array float_exact)
        (Array.init grid_length grid_at)
        (floats (Nx.slice [ I 1; A ] d.wavelength));
      let w =
        tensor [| 3; 5 |]
          (Array.init 15 (fun i -> f32 (3500. +. (5. *. float_of_int i))))
      in
      let d = Spectrum.decode ~wavelength:w t k in
      equal (array int) [| 3; 5 |] (Nx.shape d.flux);
      equal (array float_exact) (floats w) (floats d.wavelength))

(* The decoder's [(y, z)] gives the flux [(y + 1) s] and a pixel bad where
   sigmoid z > 0.5, a logit of 0 a good one; the inverse variance decodes as 1.
   A zero signal decodes to [s] itself, within its bound (scale_of). *)
let decoders =
  lazy
    (List.map
       (fun yz -> (yz, constant yz))
       [ (0., 0.); (0.25, 2.); (-0.5, -3.) ])

let decoded =
  prop "decode gives the flux (y + 1) s, bad where sigmoid z > 0.5"
    Gen.(pair (int_range 0 1023) (int_range 0 2))
    (fun (code, which) ->
      let ds = Lazy.force decoders in
      let zero = snd (List.hd ds) and (y, z), t = List.nth ds which in
      let k =
        Nx.create Nx.int64 [| 1; 3 |] [| Int64.of_int code; 3L; 1000L |]
      in
      let s, e = scale_of (floats (Spectrum.levels zero)).(code) in
      cover "at the floor" (s = 0.1);
      cover "a bad pixel" (z > 0.);
      let s0 = floats (Spectrum.decode zero k).flux in
      Array.iter (within ~what:"scale" e s) s0;
      let d = Spectrum.decode t k in
      Array.iteri
        (fun i f -> equal float_exact (f32 ((y +. 1.) *. s0.(i))) f)
        (floats d.flux);
      Array.iter (equal bool (z > 0.)) (bools d.mask);
      Array.iter (equal float_exact 1.) (floats d.ivar))

(* Upstream's network. upstream/spectrum_codec.safetensors is a few-channel
   codec of AION-1's layout with random weights, GRN's included, its encoder's
   last layer norm scaled so that the pre-quantizer norm's lane variance sits
   near its eps; upstream/spectrum_golden.safetensors holds what upstream's code
   computes with it in float64 on 16 short spectra, unpadded (its base Spectrum
   class) and padded as DESI and SDSS spectra, and the spread of its rounding
   ensemble at float32's scale. Every value lies within [τ ŝ] of the reference,
   and every decision, a token's bit, the normalisation's token or a pixel's
   mask, is the reference's but within [τ ŝ] of its boundary. *)

let upstream_codec = lazy (Spectrum.of_archive (Fixture.load "spectrum_codec"))
let golden = Fixture.load "spectrum_golden"

let variants =
  [
    ("base", Fun.id);
    ("desi", Spectrum.pad Modality.Desi);
    ("sdss", Spectrum.pad Modality.Sdss);
  ]

(* The LFQ's input, [b; n; 10], from the reference's [b; 10; n]. *)
let lfq x = floats (Nx.transpose ~axes:[ 0; 2; 1 ] x)
let signs mask = Array.map (fun b -> if b then 1. else -1.) (bools mask)
let edges t = Array.sub (floats (Spectrum.levels t)) 0 1023

let bits k =
  floats (Lfq.decode Nx.float64 ~bits:10 (Nx.slice [ A; R (1, Nx.dim 1 k) ] k))

let upstream =
  let part kind name p =
    Fixture.get golden Nx.float64 (kind ^ "." ^ name ^ "." ^ p)
  in
  let v = part "f64" and s = part "s" in
  group "upstream's network"
    [
      cases "the LFQ's input is upstream's" ~id:fst variants (fun (name, pad) ->
          let h, _ =
            Spectrum.latent
              (Lazy.force upstream_codec)
              (pad (Fixture.spectra golden))
          in
          Spread.holds "LFQ input"
            (Spread.values ~n:(Fixture.ensemble golden)
               ~reference:(lfq (v name "pre"))
               ~spread:(lfq (s name "pre"))
               (floats h)));
      cases "the normalisation is upstream's" ~id:fst variants
        (fun (name, pad) ->
          Spread.holds "normalisation"
            (Spread.values ~n:(Fixture.ensemble golden)
               ~reference:(floats (v name "norm_pre"))
               ~spread:(floats (s name "norm_pre"))
               (floats (Spectrum.normalisation (pad (Fixture.spectra golden))))));
      cases "encode gives upstream's tokens" ~id:fst variants
        (fun (name, pad) ->
          let t = Lazy.force upstream_codec in
          let k = Spectrum.encode t (pad (Fixture.spectra golden)) in
          Spread.holds "normalisation token"
            (Spread.levels ~n:(Fixture.ensemble golden) ~edges:(edges t)
               ~reference:(floats (v name "norm_pre"))
               ~spread:(floats (s name "norm_pre"))
               (ints (Nx.slice [ A; I 0 ] k)));
          Spread.holds "LFQ bits"
            (Spread.signs ~n:(Fixture.ensemble golden)
               ~reference:(lfq (v name "pre"))
               ~spread:(lfq (s name "pre"))
               (bits k)));
      cases "decode gives upstream's spectrum" ~id:fst variants
        (fun (name, _) ->
          let k = Fixture.get golden Nx.int64 ("f64." ^ name ^ ".tokens") in
          let d = Spectrum.decode (Lazy.force upstream_codec) k in
          Spread.holds "flux"
            (Spread.values ~n:(Fixture.ensemble golden)
               ~reference:(floats (v name "flux"))
               ~spread:(floats (s name "flux"))
               (floats d.flux));
          Spread.holds "mask"
            (Spread.signs ~n:(Fixture.ensemble golden)
               ~reference:(floats (v name "logit"))
               ~spread:(floats (s name "logit"))
               (signs d.mask)));
    ]

(* Compiled codecs: the codec an argument of Rune.jit, on the host or on a set
   over a software device, which has no eager kernels and compiles for this
   machine's CPU. Eager and compiled are float32 computations of one exact
   value, each within τ ŝ of it, so they differ by at most 2 τ ŝ: each decoded
   flux, and each LFQ bit and normalisation token but where eager's value lies
   within 2 τ ŝ of its boundary; a mask decides as upstream's within τ ŝ. A
   row's values do not depend on the rows beside it, so a batch of drawn rows
   keeps each row's spread. *)

let spectra () = Nx.Ptree.instantiate (module Spectrum.Spectra)
let codec () = Nx.Ptree.instantiate (module Spectrum)

let compiled_fns () =
  let open Nx.Ptree in
  ( Rune.jit (codec () @-> spectra () @-> returns tensor) Spectrum.encode,
    Rune.jit
      (codec () @-> tensor @-> returns (spectra ()))
      (fun t k -> Spectrum.decode t k) )

let encode_c, decode_c = compiled_fns ()

module Soft = (val Nx.devices [ Result.get_ok (Rig.software "aion-spectrum") ])

let encode_s, decode_s = compiled_fns ()

(* [t]'s compiled encode and decode on the host, or on Soft with [t] placed
   there, their results on the host. *)
let compiled_on ~soft t =
  if not soft then (encode_c t, decode_c t)
  else
    let s = Nx.Ptree.place (module Spectrum) Soft.on t in
    let host x = Nx.place Nx.Host.on x in
    ( (fun x ->
        host (encode_s s (Nx.Ptree.place (module Spectrum.Spectra) Soft.on x))),
      fun k ->
        let d = decode_s s (Nx.place Soft.on k) in
        {
          Spectrum.flux = host d.flux;
          ivar = host d.ivar;
          mask = host d.mask;
          wavelength = host d.wavelength;
        } )

(* Batches of one row and of four, drawn with repeats. *)
let compiled =
  prop ~count:50 "compiled computes as eager within 2 τ ŝ, on either set"
    Gen.(
      triple (of_list variants)
        (list ~size:(of_list [ 1; 4 ]) (int_range 0 15))
        bool)
    (fun ((name, pad), rows, soft) ->
      cover "padded" (name <> "base");
      cover "unpadded" (name = "base");
      cover "a batch of 4" (List.length rows = 4);
      cover "on the host" (not soft);
      cover "on a set without eager kernels" soft;
      let n = Fixture.ensemble golden and t = Lazy.force upstream_codec in
      let encode_c, decode_c = compiled_on ~soft t in
      let part kind p =
        Fixture.take rows
          (Fixture.get golden Nx.float64 (kind ^ "." ^ name ^ "." ^ p))
      in
      let twice a = Array.map (fun s -> 2. *. s) a in
      let x0 = Fixture.spectra golden in
      let x =
        pad
          {
            Spectrum.flux = Fixture.take rows x0.flux;
            ivar = Fixture.take rows x0.ivar;
            mask = Fixture.take rows x0.mask;
            wavelength = Fixture.take rows x0.wavelength;
          }
      in
      let k = encode_c x in
      Spread.holds "normalisation token"
        (Spread.levels ~n ~edges:(edges t)
           ~reference:(floats (Spectrum.normalisation x))
           ~spread:(twice (floats (part "s" "norm_pre")))
           (ints (Nx.slice [ A; I 0 ] k)));
      Spread.holds "LFQ bits"
        (Spread.signs ~n
           ~reference:(floats (fst (Spectrum.latent t x)))
           ~spread:(twice (lfq (part "s" "pre")))
           (bits k));
      let k =
        Fixture.take rows
          (Fixture.get golden Nx.int64 ("f64." ^ name ^ ".tokens"))
      in
      let d = Spectrum.decode t k and dc = decode_c k in
      Spread.holds "flux"
        (Spread.values ~n ~reference:(floats d.flux)
           ~spread:(twice (floats (part "s" "flux")))
           (floats dc.flux));
      Spread.holds "mask"
        (Spread.signs ~n
           ~reference:(floats (part "f64" "logit"))
           ~spread:(floats (part "s" "logit"))
           (signs dc.mask)))

(* Refusals *)

(* A call refused eagerly, and compiled on the host at the call: a shape's
   refusal at the trace, a value's by Nx.check. Each input fails one check. *)
type call =
  | Encode of (Nx.host Nx.float32_t, Nx.host) Spectrum.spectrum
  | Pad of int64 * (Nx.host Nx.float32_t, Nx.host) Spectrum.spectrum
  | Decode of Nx.host Nx.int64_t

let pad_c =
  Rune.jit
    Nx.Ptree.(codec () @-> spectra () @-> tensor @-> returns tensor)
    (fun t x lengths ->
      Spectrum.encode t (Spectrum.pad ~lengths Modality.Sdss x))

let refusals =
  let tokens a = Nx.create Nx.int64 [| 1; Array.length a |] a in
  let row4 w = spectrum (flat 1 4 1.) [| w |] in
  let short = spectrum [| [| 1.; 2. |] |] [| [| 3600.; 3601. |] |] in
  let long =
    spectrum [| Array.make 4801 1. |] [| Array.init 4801 float_of_int |]
  in
  let token k =
    Printf.sprintf "Aion.Spectrum.decode: token %d at [0; 1] outside [0, 1024)"
      k
  in
  cases "encode, pad and decode refuse, eagerly and compiled"
    ~id:(fun (name, _, _) -> name)
    [
      ( "a row of one pixel",
        "rows of 1 pixel",
        Encode (spectrum [| [| 1. |] |] [| [| 3600. |] |]) );
      ( "fields of different shapes",
        "fields not of one 2-d shape",
        Encode { short with ivar = Nx.ones Nx.float32 [| 1; 3 |] } );
      ( "wavelengths that decrease",
        "decrease",
        Encode (row4 [| 3600.; 3601.; -1.; -1. |]) );
      ( "a NaN wavelength",
        "NaN",
        Encode (row4 [| 3600.; Float.nan; 3602.; 3603. |]) );
      ("a length of 0", "outside [1, 2]", Pad (0L, short));
      ("a length past the row", "outside [1, 2]", Pad (3L, short));
      ("a row longer than the modality's", "at most 4800", Pad (4801L, long));
      ( "tokens of another count",
        "tokens of shape",
        Decode (tokens [| 0L; 0L; 0L; 0L |]) );
      ("a negative token", token (-1), Decode (tokens [| 0L; -1L; 0L |]));
      ("a token past 1023", token 1024, Decode (tokens [| 0L; 1024L; 0L |]));
    ]
    (fun (_, sub, call) ->
      let t = random 5 in
      let refuses f = raises_match (Exn.invalid_arg ~sub) f in
      match call with
      | Encode x ->
          refuses (fun () -> Spectrum.encode t x);
          refuses (fun () -> encode_c t x)
      | Pad (l, x) ->
          let lengths = Nx.create Nx.int64 [| 1 |] [| l |] in
          refuses (fun () -> Spectrum.pad ~lengths Modality.Sdss x);
          refuses (fun () -> pad_c t x lengths)
      | Decode k ->
          refuses (fun () -> Spectrum.decode t k);
          refuses (fun () -> decode_c t k))

let of_archive =
  let without_block () =
    let a = Fixture.load "spectrum_codec" in
    Nx_io.Archive.of_list
      (List.filter_map
         (fun n ->
           if String.starts_with ~prefix:"encoder.stages.2.0." n then None
           else Option.map (fun p -> (n, p)) (Nx_io.Archive.find n a))
         (Nx_io.Archive.names a))
  and one_weight () =
    Nx_io.Archive.of_list
      [ ("pre_quant_norm.weight", Nx.P (Nx.ones Nx.float32 [| 8 |])) ]
  in
  cases "of_archive refuses a missing weight"
    ~id:(fun (name, _, _) -> name)
    [
      ("a missing block", "encoder.stages.2.1.", without_block);
      ("a missing weight", "encoder.norm.", one_weight);
    ]
    (fun (_, sub, a) ->
      raises_match (Exn.failure ~sub) (fun () -> Spectrum.of_archive (a ())))

let () =
  exit
    (run "aion.spectrum"
       [
         group "prepare"
           [
             normalisation;
             narrow_infinity;
             rules;
             interpolation;
             constant_rows;
             repeated_first;
           ];
         group "pad" [ pad; lengths ];
         group "decode" [ shapes; decoded ];
         upstream;
         group "compiled" [ compiled ];
         group "refusals" [ refusals; of_archive ];
       ])
