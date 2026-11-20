(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectrum codec's laws on small codecs of Spectrum.init, a latent grid of
   64 wavelengths, and decoders whose output is known; and its agreement with
   upstream's own code on a designed few-channel codec, eager and compiled. The
   preparation's rules hold exactly where float32 computes them exactly, its
   formulas within nx's error bounds, and the network's values within the
   rounding ensemble's spread (Spread). *)

open Windtrap
open Aion

let f32 = Bounds.f32
let ints a = Nx.to_array (Nx.cast Nx.int64 a) |> Array.map Int64.to_int
let floats a = Nx.to_array (Nx.cast Nx.float64 a)
let bools a = Nx.to_array a
let tensor shape a = Nx.create Nx.float32 shape a
let float32_max = Int32.float_of_bits 0x7f7fffffl
let u = ldexp 1. (-24)

(* Small codecs: AION-1's layout at a few channels, two blocks in the encoder's
   third stage, and 64 grid wavelengths, so 2 latent vectors. *)

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

(* The last decoder stage's transposed convolution, layer 10 of the decoder:
   each stage is a layer norm, a resampling and a block, the first's two
   swapped. *)
let last_unpatch = 10

(* A decoder whose weights are all 0 but the last transposed convolution's bias
   [(y, z)]: it emits the signal [y] and the mask logit [z] everywhere, its
   blocks then the identity. *)
let constant (y, z) =
  let bias = tensor [| 2 |] [| y; z |] in
  Nx.Ptree.map
    (Nx.Ptree.instantiate (module Spectrum))
    (fun p a ->
      match Nx.Ptree.Path.segments p with
      | [ Field "decoder"; Index i; Field "b" ] when i = last_unpatch ->
          Nx.cast (Nx.dtype a) bias
      | Field ("decoder" | "post_quant") :: _ -> Nx.zeros_like a
      | _ -> a)
    (random 0)

(* Spectra *)

let spectrum ?mask ?ivar flux wavelength =
  let b = Array.length flux and m = Array.length flux.(0) in
  let t a = tensor [| b; m |] (Array.concat (Array.to_list a)) in
  let mask =
    match mask with
    | Some k -> k
    | None -> Array.map (Array.map (fun _ -> false)) flux
  in
  let ivar =
    match ivar with
    | Some v -> v
    | None -> Array.map (Array.map (fun _ -> 1.)) flux
  in
  Spectrum.
    {
      flux = t flux;
      ivar = t ivar;
      mask = Nx.create Nx.bool [| b; m |] (Array.concat (Array.to_list mask));
      wavelength = t wavelength;
    }

(* Rows of [m] pixels over part of the grid's range, wavelengths increasing,
   some repeated, with fluxes, inverse variances and masks that break the
   preparation: NaN, infinities, negative inverse variances, masked pixels. *)
let rows_of ?(b = Gen.int_range 1 3) ?(m = Gen.int_range 2 12) ~infinities () =
  Gen.(
    let* b = b in
    let* m = m in
    let row g = array ~size:(constant m) g in
    let flux =
      frequency
        [
          (8, map f32 (float_range (-20.) 60.));
          ( 1,
            of_list
              (if infinities then
                 [ Float.nan; Float.infinity; Float.neg_infinity; 0. ]
               else [ Float.nan; 0. ]) );
        ]
    in
    let ivar =
      frequency
        [
          (6, map f32 (float_range 0. 150.)); (1, of_list [ Float.nan; -1.; 0. ]);
        ]
    in
    (* Free wavelengths, or the grid's own at increasing indices, so that keys
       meet them. *)
    let free =
      let* s = map f32 (float_range 3450. 3520.) in
      let+ steps =
        row (frequency [ (5, map f32 (float_range 0.1 6.)); (1, constant 0.) ])
      in
      let acc = ref s in
      Array.map
        (fun d ->
          let w = !acc in
          acc := f32 (!acc +. d);
          w)
        steps
    in
    let on_grid =
      let* j = int_range (-5) 40 in
      let+ steps = row (int_range 0 3) in
      let acc = ref j in
      Array.map
        (fun d ->
          let w = grid_at !acc in
          acc := !acc + d;
          w)
        steps
    in
    let wave = one_of [ free; on_grid ] in
    let+ fs = array ~size:(constant b) (row flux)
    and+ vs = array ~size:(constant b) (row ivar)
    and+ ms =
      array ~size:(constant b)
        (row (frequency [ (4, constant false); (1, constant true) ]))
    and+ ws = array ~size:(constant b) wave in
    (fs, vs, ms, ws))

(* A flux as preparation reads it: NaN as 0, infinities as float32's largest
   finite values. *)
let fix_flux x =
  if Float.is_nan x then 0.
  else Float.max (-.float32_max) (Float.min float32_max x)

(* The normalisation of one row, in float64. *)
let norm64 flux mask =
  let s = ref 0. and c = ref 1. in
  Array.iteri
    (fun i x ->
      if not mask.(i) then begin
        s := !s +. fix_flux x;
        c := !c +. 1.
      end)
    flux;
  Float.log10 (1. +. Float.max (!s /. !c) 0.1)

let within ~what bound expected got =
  if not (Float.abs (got -. expected) <= bound) then
    failf "%s: %.9g, expected %.9g within %.3g" what got expected bound

let rows = rows_of ~infinities:false ()

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

(* nx's sum bound: a sum of [n] terms within γ(β(n)) Σ|x|, β and γ as
   Nx_kernel's section on bounds states them. *)
let sum_bound n =
  let beta = (((5 * min n 256) + 3) / 4) + ((n + 127) / 128) in
  let k = float_of_int beta in
  k *. u /. (1. -. (k *. u))

(* The normalisation's distance from [norm64]: the sum's, then a rounding each
   for the division, the floor's 0.1, the addition of 1 and log10's 3. *)
let norm_bound flux mask =
  let s = ref 0. and a = ref 0. and c = ref 1. in
  Array.iteri
    (fun i x ->
      if not mask.(i) then begin
        s := !s +. fix_flux x;
        a := !a +. Float.abs (fix_flux x);
        c := !c +. 1.
      end)
    flux;
  let mean = !s /. !c in
  let e_mean =
    (((!a *. sum_bound (Array.length flux)) +. (Float.abs !s *. u)) /. !c)
    +. (Float.abs mean *. u)
  in
  let w = Float.max mean 0.1 +. 1. in
  let e_w = e_mean +. (0.1 *. u) +. (w *. u) in
  (e_w /. (w *. log 10.)) +. (Float.abs (Float.log10 w) *. 3. *. u)

let normalisation =
  prop "the normalisation is log10 (1 + max mean 0.1)"
    (rows_of ~infinities:true ()) (fun (fs, vs, ms, ws) ->
      let fs = Array.map one_infinity fs in
      let x = spectrum ~mask:ms ~ivar:vs fs ws in
      let got = floats (Spectrum.normalisation x) in
      Array.iteri
        (fun i f ->
          cover "NaN flux" (Array.exists Float.is_nan f);
          cover "infinite flux"
            (Array.exists (fun x -> Float.abs x = Float.infinity) f);
          cover "masked pixel" (Array.exists Fun.id ms.(i));
          cover "mean below 0.1" (norm64 f ms.(i) = Float.log10 1.1);
          within ~what:"normalisation"
            (norm_bound f ms.(i))
            (norm64 f ms.(i))
            got.(i))
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

(* Preparation, by its rules: equal inputs after the rule prepare equal. *)

let prepared_equal t a b =
  let pa, ka = Spectrum.prepare t a and pb, kb = Spectrum.prepare t b in
  equal (array int) (ints ka) (ints kb);
  equal (array float_exact) (floats pa) (floats pb)

let fix_ivar x = if Float.is_nan x then 0. else Float.max 0. (Float.min 100. x)

let rules =
  let t = random 1 in
  group "preparation's rules"
    [
      prop "NaN fluxes prepare as 0, infinities as float32's largest"
        (rows_of ~infinities:true ()) (fun (fs, vs, ms, ws) ->
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
          let v = floats (Spectrum.normalisation x) in
          let levels = floats (Spectrum.levels t) in
          let count y =
            min 1023
              (Array.fold_left (fun n l -> if l < y then n + 1 else n) 0 levels)
          in
          equal (array int) (Array.map count v)
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

(* Interpolation between pixels of unequal values, against float64 with the
   pixels' values exact. [prepare]'s signal at a key in [x(i), x(i+1)] is
   [asinh (y(i) + (key - x(i)) · slope)], [y(j) = (f(j) / s - 1) · 0.2]. Its
   bound sums:
   - the larger of [y(i)]'s and [y(i+1)]'s: [0.2 |f| / s] times [s]'s relative
     bound (pow's 4 floats, the subtraction's and 0.1's rounding, [e_s / s])
     plus the division's and product's 2 roundings, and [0.2 u |f / s - 1|]
     for the subtraction;
   - the step's: the slope's division and subtraction and the product, 6
     roundings of [|ynew - y(i)|] ([key - x(i)] and [x(i) - x(i+1)] are exact,
     differences of nearby floats);
   - the sum's and asinh's: [2 u |ynew|] and asinh's 2 floats, [3 u |asinh|].
   [decode ~wavelength]'s flux at a key, [F] on the line through the grid's
   fluxes [f(j) = (y(j) + 1) · s], is [(ynew + 1) · s]. Its bound sums:
   - the step's 6 roundings of [|ynew - y(i)| s = |F - f(i)|], counted as 8
     for the rounding of the [f(j)] the line runs through;
   - the sum's, the addition of 1's and the product's roundings of
     [|ynew| s <= |F| + s] and of [|F|]: [6 u |F| + 4 u s];
   - the grid fluxes' own rounding, [u max (|f(i)|, |f(i+1)|)], twice. *)
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
          let* c = int_range 0 1023 in
          let* a = int_range 0 1023 in
          let* b = int_range 0 1023 in
          let+ keys =
            array ~size:(int_range 1 9)
              (frequency
                 [
                   (4, map f32 (float_range 3490. 3560.));
                   (1, map grid_at (int_range 0 (grid_length - 1)));
                 ])
          in
          (c, a, b, keys))
        (fun (c, a, b, keys) ->
          let k =
            Nx.create Nx.int64 [| 1; 3 |] (Array.map Int64.of_int [| c; a; b |])
          in
          let gx = Array.init grid_length grid_at in
          let fg = floats (Spectrum.decode t k).flux in
          let n = Array.length keys in
          let d = Spectrum.decode ~wavelength:(tensor [| 1; n |] keys) t k in
          let flux = floats d.flux and mask = bools d.mask in
          let s, _ = scale_of (floats (Spectrum.levels t)).(c) in
          Array.iteri
            (fun j key ->
              cover "a key between grid wavelengths"
                (key > gx.(0)
                && key < gx.(grid_length - 1)
                && not (Array.mem key gx));
              cover "a key outside the grid" (key < gx.(0));
              if key < gx.(0) || key > gx.(grid_length - 1) then begin
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
   and 0 outside. Each is its formula of the token's value [w] within s's bound
   (pow's 4 floats, the subtraction's and 0.1's rounding) carried through the
   division, then the roundings of the subtraction, the products and asinh's 2
   floats. *)
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
      let w =
        (floats (Spectrum.levels t)).(List.hd (Array.to_list (ints code)))
      in
      let s, e_s = scale_of w in
      let q = Float.abs f /. s in
      let signal = Float.asinh (((f /. s) -. 1.) *. 0.2) in
      let e_signal =
        (0.2 *. q *. ((e_s /. s) +. (2. *. u)))
        +. (0.2 *. u *. Float.abs ((f /. s) -. 1.))
        +. (3. *. u *. Float.abs signal)
      in
      let r = Float.sqrt (Float.min 100. v) /. s in
      let noise = Float.asinh (r *. 0.2) in
      let e_noise =
        (0.2 *. r *. ((e_s /. s) +. (3. *. u))) +. (3. *. u *. Float.abs noise)
      in
      let inside k = grid_at k >= ws.(0) && grid_at k <= ws.(m - 1) in
      cover "keys below the row" (j > 0);
      cover "keys above the row" (ws.(m - 1) < grid_at (grid_length - 1));
      for k = 0 to grid_length - 1 do
        let sg = p.(2 * k) and ns = p.((2 * k) + 1) in
        if inside k then begin
          within ~what:(Printf.sprintf "signal at %d" k) e_signal signal sg;
          within ~what:(Printf.sprintf "noise at %d" k) e_noise noise ns;
          let first = List.find inside (List.init grid_length Fun.id) in
          equal float_exact p.(2 * first) sg;
          equal float_exact p.((2 * first) + 1) ns
        end
        else begin
          equal float_exact 0. sg;
          equal float_exact 0. ns
        end
      done)

(* Padding *)

let pad =
  group "pad"
    [
      test "pads each field with its pad value" (fun () ->
          let x = spectrum [| [| 1.; 2. |] |] [| [| 3600.; 3601. |] |] in
          let p = Spectrum.pad Modality.Sdss x in
          equal int 4800 (Nx.dim 1 p.flux);
          equal (array float_exact) [| 2.; 0.; 0. |]
            (floats (Nx.slice [ I 0; R (1, 4) ] p.flux));
          equal (array float_exact) [| 0.; 0. |]
            (floats (Nx.slice [ I 0; R (2, 4) ] p.ivar));
          equal (array bool) [| false; true; true |]
            (bools (Nx.slice [ I 0; R (1, 4) ] p.mask));
          equal (array float_exact)
            [| 3601.; 99999.; 99999. |]
            (floats (Nx.slice [ I 0; R (1, 4) ] p.wavelength)));
      test "lengths" (fun () ->
          equal int 7808 (Spectrum.length Modality.Desi);
          equal int 4800 (Spectrum.length Modality.Sdss));
      (* AION-1's F.pad with a negative width cuts the row. *)
      test "refuses a row longer than the modality's" (fun () ->
          let x =
            spectrum [| Array.make 4801 1. |] [| Array.init 4801 float_of_int |]
          in
          raises_match (Exn.invalid_arg ~sub:"at most 4800") (fun () ->
              Spectrum.pad Modality.Sdss x));
    ]

(* Lengths *)

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
    let cut r l v = Array.mapi (fun j x -> if j < l then x else v) r in
    let ws = Array.mapi (fun i w -> cut w ls.(i) fill) ws in
    let fs = Array.mapi (fun i f -> cut f ls.(i) fill) fs in
    (fs, vs, ms, ws, ls))

let lengths =
  let t = random 9 in
  group "lengths"
    [
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
            ls);
      test "pad refuses a length outside [1, m]" (fun () ->
          let x = spectrum [| [| 1.; 2. |] |] [| [| 3600.; 3601. |] |] in
          List.iter
            (fun l ->
              raises_match (Exn.invalid_arg ~sub:"outside [1, 2]") (fun () ->
                  Spectrum.pad
                    ~lengths:(Nx.create Nx.int64 [| 1 |] [| l |])
                    Modality.Sdss x))
            [ 0L; 3L ]);
    ]

(* Encoding and decoding *)

let flat b m v = Array.make b (Array.make m v)

let waves b m =
  Array.make b (Array.init m (fun i -> f32 (3480. +. (0.9 *. float_of_int i))))

let shapes =
  let t = random 3 in
  group "shapes"
    [
      test "encode: one normalisation token, then one per 32 grid wavelengths"
        (fun () ->
          let k =
            Spectrum.encode t
              (Spectrum.pad Modality.Desi
                 (spectrum (flat 3 50 2.) (waves 3 50)))
          in
          equal (array int) [| 3; 3 |] (Nx.shape k);
          Array.iter
            (fun v ->
              at_least int ~than:0 v;
              less int ~than:1024 v)
            (ints k));
      test "decode: the grid without wavelengths" (fun () ->
          let d =
            Spectrum.decode t
              (Nx.create Nx.int64 [| 2; 3 |] [| 5L; 0L; 1023L; 9L; 512L; 7L |])
          in
          equal (array int) [| 2; grid_length |] (Nx.shape d.flux);
          equal (array int) [| 2; grid_length |] (Nx.shape d.mask);
          equal (array float_exact)
            (Array.init grid_length grid_at)
            (floats (Nx.slice [ I 1; A ] d.wavelength)));
      test "decode: the given wavelengths" (fun () ->
          let w =
            tensor [| 2; 5 |]
              (Array.init 10 (fun i -> f32 (3500. +. (5. *. float_of_int i))))
          in
          let d =
            Spectrum.decode ~wavelength:w t (Nx.zeros Nx.int64 [| 2; 3 |])
          in
          equal (array int) [| 2; 5 |] (Nx.shape d.flux);
          equal (array float_exact) (floats w) (floats d.wavelength));
    ]

(* The decoder's [(y, z)] gives the flux [(y + 1) s] and a pixel bad where
   sigmoid z > 0.5; a logit of 0 is a good pixel. A zero signal decodes to [s]
   itself, max (10^w - 1) 0.1 of the token's value [w], within pow's 4 floats,
   the subtraction's rounding and 0.1's. *)
let scale =
  let t = constant (0., 0.) in
  prop "a zero signal decodes to the scale max (10^w - 1) 0.1"
    Gen.(int_range 0 1023)
    (fun code ->
      let w = (floats (Spectrum.levels t)).(code) in
      let s = Float.max ((10. ** w) -. 1.) 0.1 in
      cover "at the floor" ((10. ** w) -. 1. < 0.1);
      let k =
        Nx.create Nx.int64 [| 1; 3 |] [| Int64.of_int code; 3L; 1000L |]
      in
      let d = Spectrum.decode t k in
      let e = (9. *. u *. (10. ** w)) +. (u *. s) +. (0.1 *. u) in
      Array.iter (within ~what:"flux" e s) (floats d.flux);
      Array.iter (equal bool false) (bools d.mask))

let decoded =
  let k = Nx.create Nx.int64 [| 1; 3 |] [| 700L; 3L; 1000L |] in
  let s = lazy (floats (Spectrum.decode (constant (0., 0.)) k).flux).(0) in
  cases "decode applies the normalisation and the mask's rule"
    ~id:(fun (y, z, _, _) -> Printf.sprintf "y %g, z %g" y z)
    [ (0.25, 2., 1.25, true); (-0.5, -3., 0.5, false) ]
    (fun (y, z, factor, bad) ->
      let d = Spectrum.decode (constant (y, z)) k in
      Array.iter
        (fun f -> equal float_exact (f32 (factor *. Lazy.force s)) f)
        (floats d.flux);
      Array.iter (equal bool bad) (bools d.mask);
      Array.iter (equal float_exact 1.) (floats d.ivar))

(* Upstream's network. upstream/spectrum_codec.safetensors is a few-channel
   codec of AION-1's layout with random weights, GRN's included, its encoder's
   last layer norm scaled so that the pre-quantizer norm's lane variance sits
   near its eps; upstream/spectrum_golden.safetensors holds what upstream's
   PyTorch code computes with it in float64 on 16 short spectra, unpadded (its
   base Spectrum class) and padded as DESI and SDSS spectra, and the spread of
   its rounding ensemble at float32's scale ([Spread]). The generator is named
   in the dune file. Every value lies within [τ ŝ] of the float64 reference, and
   every decision, a token's bit, the normalisation's token or a pixel's mask,
   is the reference's but within [τ ŝ] of its boundary. *)

let fixture name = Nx_io.load_safetensors ("upstream/" ^ name ^ ".safetensors")
let upstream_codec = lazy (Spectrum.of_archive (fixture "spectrum_codec"))
let golden = lazy (fixture "spectrum_golden")

let get dt name =
  match Nx_io.Archive.find name (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let f64s name = floats (get Nx.float64 name)
let n () = (ints (get Nx.int64 "n")).(0)

let input () =
  Spectrum.
    {
      flux = get Nx.float32 "input.flux";
      ivar = get Nx.float32 "input.ivar";
      mask = Nx.not_equal_s (get Nx.uint8 "input.mask") 0;
      wavelength = get Nx.float32 "input.lambda";
    }

let variants =
  [
    ("base", Fun.id);
    ("desi", Spectrum.pad Modality.Desi);
    ("sdss", Spectrum.pad Modality.Sdss);
  ]

let holds name (c : Spread.check) =
  if c.failures <> [] then failf "%a" (Spread.pp name) c

(* The LFQ's input, [b; n; 10], from the reference's [b; 10; n]. *)
let lfq name = floats (Nx.transpose ~axes:[ 0; 2; 1 ] (get Nx.float64 name))

let upstream =
  let v name part = "f64." ^ name ^ "." ^ part
  and s name part = "s." ^ name ^ "." ^ part in
  group "upstream's network"
    [
      cases "the LFQ's input is upstream's" ~id:fst variants (fun (name, pad) ->
          let h, _ =
            Spectrum.latent (Lazy.force upstream_codec) (pad (input ()))
          in
          holds "LFQ input"
            (Spread.values ~n:(n ())
               ~reference:(lfq (v name "pre"))
               ~spread:(lfq (s name "pre"))
               (floats h)));
      cases "the normalisation is upstream's" ~id:fst variants
        (fun (name, pad) ->
          holds "normalisation"
            (Spread.values ~n:(n ())
               ~reference:(f64s (v name "norm_pre"))
               ~spread:(f64s (s name "norm_pre"))
               (floats (Spectrum.normalisation (pad (input ()))))));
      cases "encode gives upstream's tokens" ~id:fst variants
        (fun (name, pad) ->
          let t = Lazy.force upstream_codec in
          let k = Spectrum.encode t (pad (input ())) in
          let edges = Array.sub (floats (Spectrum.levels t)) 0 1023 in
          holds "normalisation token"
            (Spread.levels ~n:(n ()) ~edges
               ~reference:(f64s (v name "norm_pre"))
               ~spread:(f64s (s name "norm_pre"))
               (ints (Nx.slice [ A; I 0 ] k)));
          let bits =
            Lfq.decode Nx.float64 ~bits:10 (Nx.slice [ A; R (1, Nx.dim 1 k) ] k)
          in
          holds "LFQ bits"
            (Spread.signs ~n:(n ())
               ~reference:(lfq (v name "pre"))
               ~spread:(lfq (s name "pre"))
               (floats bits)));
      cases "decode gives upstream's spectrum" ~id:fst variants
        (fun (name, _) ->
          let k = get Nx.int64 (v name "tokens") in
          let d = Spectrum.decode (Lazy.force upstream_codec) k in
          holds "flux"
            (Spread.values ~n:(n ())
               ~reference:(f64s (v name "flux"))
               ~spread:(f64s (s name "flux"))
               (floats d.flux));
          holds "mask"
            (Spread.signs ~n:(n ())
               ~reference:(f64s (v name "logit"))
               ~spread:(f64s (s name "logit"))
               (Array.map (fun b -> if b then 1. else -1.) (bools d.mask))));
    ]

(* Departures from AION-1's code *)

let divergences =
  group "divergences"
    [
      test "decode: a wavelength outside the grid is bad, of flux 0" (fun () ->
          let t = constant (0., -1.) in
          let w =
            tensor [| 1; 4 |]
              [| 3000.; 3500.; grid_at (grid_length - 1); 99999. |]
          in
          let d =
            Spectrum.decode ~wavelength:w t
              (Nx.create Nx.int64 [| 1; 3 |] [| 700L; 0L; 0L |])
          in
          let f = floats d.flux in
          equal (array bool) [| true; false; false; true |] (bools d.mask);
          equal float_exact 0. f.(0);
          equal float_exact 0. f.(3);
          greater float_exact ~than:0. f.(1));
      test "encode: refuses wavelengths that decrease" (fun () ->
          let t = random 5 in
          let x = spectrum (flat 1 4 1.) [| [| 3600.; 3601.; -1.; -1. |] |] in
          raises_match (Exn.invalid_arg ~sub:"decrease") (fun () ->
              Spectrum.encode t (Spectrum.pad Modality.Sdss x)));
      test "encode: refuses a NaN wavelength" (fun () ->
          let t = random 5 in
          let x = spectrum (flat 1 3 1.) [| [| 3600.; Float.nan; 3602. |] |] in
          raises_match (Exn.invalid_arg ~sub:"NaN") (fun () ->
              Spectrum.encode t (Spectrum.pad Modality.Sdss x)));
      (* At the grid's first wavelength, a row whose first two pixels share it
         reads the first pixel's value, where upstream's slope is NaN: as the
         row without the second pixel, masked so that the normalisations
         agree. *)
      test "prepare: a repeated first wavelength reads its value" (fun () ->
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
          equal (array float_exact) (first single) (first repeated));
      cases "decode: refuses a token outside [0, 1024)" ~id:string_of_int
        [ -1; 1024 ] (fun k ->
          let t = random 6 in
          raises
            (Invalid_argument
               (Printf.sprintf
                  "Aion.Spectrum.decode: token %d at [0; 1] outside [0, 1024)" k))
            (fun () ->
              Spectrum.decode t
                (Nx.create Nx.int64 [| 1; 3 |] [| 0L; Int64.of_int k; 0L |])));
    ]

let refusals =
  group "refusals"
    [
      test "encode: rows of one pixel" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"rows of 1 pixel") (fun () ->
              Spectrum.encode (random 7)
                (spectrum [| [| 1. |] |] [| [| 3600. |] |])));
      test "encode: fields of different shapes" (fun () ->
          let x = spectrum (flat 1 4 1.) (waves 1 4) in
          raises_match (Exn.invalid_arg ~sub:"fields not of one 2-d shape")
            (fun () ->
              Spectrum.encode (random 7)
                { x with Spectrum.ivar = Nx.ones Nx.float32 [| 1; 3 |] }));
      test "decode: tokens of the wrong count" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Spectrum.decode (random 7) (Nx.zeros Nx.int64 [| 1; 4 |])));
      test "of_archive: a missing block leaves a weight unread" (fun () ->
          let a = fixture "spectrum_codec" in
          let kept =
            List.filter_map
              (fun n ->
                if String.starts_with ~prefix:"encoder.stages.2.0." n then None
                else Option.map (fun p -> (n, p)) (Nx_io.Archive.find n a))
              (Nx_io.Archive.names a)
          in
          raises_match (Exn.failure ~sub:"encoder.stages.2.1.") (fun () ->
              Spectrum.of_archive (Nx_io.Archive.of_list kept)));
      test "of_archive: a missing weight" (fun () ->
          let a =
            Nx_io.Archive.of_list
              [ ("pre_quant_norm.weight", Nx.P (Nx.ones Nx.float32 [| 8 |])) ]
          in
          raises_match (Exn.failure ~sub:"encoder.norm.") (fun () ->
              Spectrum.of_archive a));
    ]

(* Compiled codecs: the codec an argument of Rune.jit, a program per shape.
   Eager and compiled are float32 computations of one exact value, each within
   τ ŝ of it (the checks above), ŝ the fixture's ensemble spread, so they
   differ by at most 2 τ ŝ: each decoded flux, and each LFQ bit and
   normalisation token but where eager's value lies within 2 τ ŝ of its
   boundary; a mask decides as upstream's within τ ŝ, as eager's does. A row's
   values do not depend on the rows beside it, so a batch of drawn rows keeps
   each row's spread. *)

let spectra () = Nx.Ptree.instantiate (module Spectrum.Spectra)
let codec () = Nx.Ptree.instantiate (module Spectrum)

let compiled_fns () =
  let open Nx.Ptree in
  ( Rune.jit (codec () @-> spectra () @-> returns tensor) Spectrum.encode,
    Rune.jit
      (codec () @-> tensor @-> returns (spectra ()))
      (fun t k -> Spectrum.decode t k) )

let encode_c, decode_c = compiled_fns ()

let take rows x =
  let rows = Array.of_list (List.map Int64.of_int rows) in
  Nx.take ~axis:0 (Nx.create Nx.int64 [| Array.length rows |] rows) x

let take_rows rows (x : _ Spectrum.spectrum) : _ Spectrum.spectrum =
  {
    flux = take rows x.flux;
    ivar = take rows x.ivar;
    mask = take rows x.mask;
    wavelength = take rows x.wavelength;
  }

(* Batches of one row and of four, drawn with repeats: each variant then
   traces two encode programs, whatever the seed. *)
let compiled =
  prop ~count:50 "compiled computes as eager, within 2 τ ŝ"
    Gen.(
      pair (of_list variants) (list ~size:(of_list [ 1; 4 ]) (int_range 0 15)))
    (fun ((name, pad), rows) ->
      cover "padded" (name <> "base");
      cover "unpadded" (name = "base");
      cover "a batch of 4" (List.length rows = 4);
      let n = n () in
      let t = Lazy.force upstream_codec in
      let twice a = Array.map (fun s -> 2. *. s) a in
      let part p = take rows (get Nx.float64 ("s." ^ name ^ "." ^ p)) in
      let x = pad (take_rows rows (input ())) in
      let k = encode_c t x in
      let edges = Array.sub (floats (Spectrum.levels t)) 0 1023 in
      holds "normalisation token"
        (Spread.levels ~n ~edges
           ~reference:(floats (Spectrum.normalisation x))
           ~spread:(twice (floats (part "norm_pre")))
           (ints (Nx.slice [ A; I 0 ] k)));
      holds "LFQ bits"
        (Spread.signs ~n
           ~reference:(floats (fst (Spectrum.latent t x)))
           ~spread:(twice (floats (Nx.transpose ~axes:[ 0; 2; 1 ] (part "pre"))))
           (floats
              (Lfq.decode Nx.float64 ~bits:10
                 (Nx.slice [ A; R (1, Nx.dim 1 k) ] k))));
      let k = take rows (get Nx.int64 ("f64." ^ name ^ ".tokens")) in
      let d = Spectrum.decode t k and dc = decode_c t k in
      holds "flux"
        (Spread.values ~n ~reference:(floats d.flux)
           ~spread:(twice (floats (part "flux")))
           (floats dc.flux));
      let logit = take rows (get Nx.float64 ("f64." ^ name ^ ".logit")) in
      holds "mask"
        (Spread.signs ~n ~reference:(floats logit)
           ~spread:(floats (part "logit"))
           (Array.map (fun b -> if b then 1. else -1.) (bools dc.mask))))

(* A set over a software device, without kernels: it computes only under
   Rune.jit, on this machine's CPU. *)
module Soft = (val Nx.devices [ Result.get_ok (Rig.software "aion-spectrum") ])

let encode_s, decode_s = compiled_fns ()

let compiled_soft =
  test "a compiled codec computes on a set without eager kernels" (fun () ->
      (* Its programs compile for the host's CPU there too, so they give the
         host's bits. *)
      let t = Lazy.force upstream_codec in
      let soft = Nx.Ptree.place (module Spectrum) Soft.on t in
      let x = Spectrum.pad Modality.Desi (input ()) in
      let on_soft x = Nx.Ptree.place (module Spectrum.Spectra) Soft.on x in
      let host x = Nx.place Nx.Host.on x in
      let k = encode_c t x in
      equal (array int) (ints k) (ints (host (encode_s soft (on_soft x))));
      let d = decode_c t k and ds = decode_s soft (Nx.place Soft.on k) in
      equal (array float_exact) (floats d.flux) (floats (host ds.flux));
      equal (array bool) (bools d.mask) (bools (host ds.mask)))

(* On the host a compiled call raises a refusal at the call, as eagerly: a
   shape's at the trace, a value's by Nx.check. Each input fails one check. *)
let compiled_refusals =
  let t = lazy (random 5) in
  let refuses sub f = raises_match (Exn.invalid_arg ~sub) f in
  let encode_padded =
    Rune.jit
      Nx.Ptree.(codec () @-> spectra () @-> tensor @-> returns tensor)
      (fun t x lengths ->
        Spectrum.encode t (Spectrum.pad ~lengths Modality.Sdss x))
  in
  group "compiled refusals"
    [
      test "wavelengths that decrease" (fun () ->
          let x = spectrum (flat 1 4 1.) [| [| 3600.; 3601.; -1.; -1. |] |] in
          refuses "decrease" (fun () -> encode_c (Lazy.force t) x));
      test "a length outside [1, m]" (fun () ->
          let x = spectrum [| [| 1.; 2. |] |] [| [| 3600.; 3601. |] |] in
          refuses "outside [1, 2]" (fun () ->
              encode_padded (Lazy.force t) x
                (Nx.create Nx.int64 [| 1 |] [| 3L |])));
      test "a row longer than the modality's" (fun () ->
          let x =
            spectrum [| Array.make 4801 1. |] [| Array.init 4801 float_of_int |]
          in
          refuses "at most 4800" (fun () ->
              encode_padded (Lazy.force t) x
                (Nx.create Nx.int64 [| 1 |] [| 4801L |])));
      test "a token outside [0, 1024)" (fun () ->
          raises
            (Invalid_argument
               "Aion.Spectrum.decode: token 1024 at [0; 1] outside [0, 1024)")
            (fun () ->
              decode_c (Lazy.force t)
                (Nx.create Nx.int64 [| 1; 3 |] [| 0L; 1024L; 0L |])));
    ]

let () =
  exit
    (run "aion.spectrum"
       [
         normalisation;
         narrow_infinity;
         rules;
         constant_rows;
         pad;
         lengths;
         shapes;
         scale;
         decoded;
         interpolation;
         upstream;
         group "compiled" [ compiled; compiled_soft; compiled_refusals ];
         divergences;
         refusals;
       ])
