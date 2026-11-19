(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectrum codec's laws on small codecs: random weights of AION-1's layout
   at a few channels, a latent grid of 64 wavelengths, and hand-built decoders
   whose output is known; and its agreement with upstream's own code on a
   designed few-channel codec. The preparation's tolerances are the running
   error bounds of Shadow, from nx.mli's float32 bounds, against expected values
   computed in float64 from the formulas of Spectrum's preamble; the network's
   are the rounding ensemble's (Spread). *)

open Windtrap
open Aion

let f32 = Bounds.f32
let ints a = Nx.to_array (Nx.cast Nx.int64 a) |> Array.map Int64.to_int
let floats a = Nx.to_array (Nx.cast Nx.float64 a)
let bools a = Nx.to_array a
let tensor shape a = Nx.create Nx.float32 shape a
let float32_max = Int32.float_of_bits 0x7f7fffffl
let u = ldexp 1. (-24)

(* Small codecs *)

let grid_length = 64
let grid_at k = f32 (3500. +. (0.8 *. float_of_int k))
let enc_dims = [| 4; 4; 4; 8 |]
let dec_dims = [| 4; 4; 4; 2 |]

(* Every weight of a codec of [enc_dims] and [dec_dims], one block a stage but
   two in the encoder's third, named as AION-1's file names them, with values
   from [init name shape]. *)
let archive init =
  let w name shape = (name, Nx.P (init name shape)) in
  let ln name d =
    [ w (name ^ ".weight") [| d |]; w (name ^ ".bias") [| d |] ]
  in
  let lin name o i =
    [ w (name ^ ".weight") [| o; i |]; w (name ^ ".bias") [| o |] ]
  in
  let conv name o i k =
    [ w (name ^ ".weight") [| o; i; k |]; w (name ^ ".bias") [| o |] ]
  in
  let convt name i o k =
    [ w (name ^ ".weight") [| i; o; k |]; w (name ^ ".bias") [| o |] ]
  in
  let block name d =
    List.concat
      [
        conv (name ^ ".dwconv") d 1 7;
        ln (name ^ ".norm") d;
        lin (name ^ ".pwconv1") (4 * d) d;
        [
          w (name ^ ".grn.gamma") [| 1; 1; 4 * d |];
          w (name ^ ".grn.beta") [| 1; 1; 4 * d |];
        ];
        lin (name ^ ".pwconv2") d (4 * d);
      ]
  in
  let enc i =
    let s = Printf.sprintf "encoder.downsample_layers.%d" i in
    let d = enc_dims.(i) in
    (if i = 0 then conv (s ^ ".0") d 2 4 @ ln (s ^ ".1") d
     else ln (s ^ ".0") enc_dims.(i - 1) @ conv (s ^ ".1") d enc_dims.(i - 1) 2)
    @ block (Printf.sprintf "encoder.stages.%d.0" i) d
    @ if i = 2 then block "encoder.stages.2.1" d else []
  in
  let dec i =
    let s = Printf.sprintf "decoder.upsample_layers.%d" i in
    let d = dec_dims.(i) in
    (if i = 0 then convt (s ^ ".0") 8 d 2 @ ln (s ^ ".1") d
     else
       ln (s ^ ".0") dec_dims.(i - 1)
       @ convt (s ^ ".1") dec_dims.(i - 1) d (if i = 3 then 4 else 2))
    @ block (Printf.sprintf "decoder.stages.%d.0" i) d
  in
  let grid = Array.init grid_length grid_at in
  let levels =
    Array.init 1024 (fun k -> f32 (-1. +. (6. *. float_of_int k /. 1023.)))
  in
  Nx_io.Archive.of_list
    (List.concat
       [
         List.concat_map enc [ 0; 1; 2; 3 ];
         ln "encoder.norm" 8;
         ln "pre_quant_norm" 8;
         conv "quant_conv" 10 8 1;
         conv "post_quant_conv" 8 10 1;
         List.concat_map dec [ 0; 1; 2; 3 ];
         [
           ("latent_grid._wavelength", Nx.P (tensor [| 1; grid_length |] grid));
           ("normalization_quantizer.buckets", Nx.P (tensor [| 1024 |] levels));
         ];
       ])

let numel shape = Array.fold_left ( * ) 1 shape

(* Weights uniform in [-0.5, 0.5], layer norms' scales about 1. *)
let random seed =
  let st = Random.State.make [| seed |] in
  let draw name shape =
    let centre =
      if
        String.ends_with ~suffix:"norm.weight" name
        || (String.ends_with ~suffix:".1.weight" name && Array.length shape = 1)
        || (String.ends_with ~suffix:".0.weight" name && Array.length shape = 1)
      then 1.
      else 0.
    in
    tensor shape
      (Array.init (numel shape) (fun _ ->
           f32 (centre +. Random.State.float st 1. -. 0.5)))
  in
  Spectrum.of_archive (archive draw)

(* A decoder whose weights are all 0 but the last transposed convolution's bias
   [(y, z)]: it emits the signal [y] and the mask logit [z] everywhere. *)
let constant (y, z) =
  let init name shape =
    if name = "decoder.upsample_layers.3.1.bias" then tensor shape [| y; z |]
    else if
      String.starts_with ~prefix:"decoder" name
      || name = "post_quant_conv.weight"
      || name = "post_quant_conv.bias"
    then Nx.zeros Nx.float32 shape
    else tensor shape (Array.make (numel shape) 0.25)
  in
  Spectrum.of_archive (archive init)

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
let rows_of ~infinities =
  Gen.(
    let* b = int_range 1 3 in
    let* m = int_range 2 12 in
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

(* The normalisation of one row, in float64. *)
let norm64 flux mask =
  let fix x =
    if Float.is_nan x then 0.
    else Float.max (-.float32_max) (Float.min float32_max x)
  in
  let s = ref 0. and c = ref 1. in
  Array.iteri
    (fun i x ->
      if not mask.(i) then begin
        s := !s +. fix x;
        c := !c +. 1.
      end)
    flux;
  Float.log10 (1. +. Float.max (!s /. !c) 0.1)

let within ~what bound expected got =
  if not (Float.abs (got -. expected) <= bound) then
    failf "%s: %.9g, expected %.9g within %.3g" what got expected bound

let rows = rows_of ~infinities:false

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

let normalisation =
  prop "the normalisation is log10 (1 + max mean 0.1)"
    (rows_of ~infinities:true) (fun (fs, vs, ms, ws) ->
      let fs = Array.map one_infinity fs in
      let x = spectrum ~mask:ms ~ivar:vs fs ws in
      Array.iteri
        (fun i f ->
          cover "NaN flux" (Array.exists Float.is_nan f);
          cover "infinite flux"
            (Array.exists (fun x -> Float.abs x = Float.infinity) f);
          cover "masked pixel" (Array.exists Fun.id ms.(i));
          cover "mean below 0.1" (norm64 f ms.(i) = Float.log10 1.1))
        fs;
      let s = Shadow.normalisation x in
      let got = floats s.x and e = floats s.e in
      Array.iteri
        (fun i f ->
          within ~what:"normalisation" e.(i) (norm64 f ms.(i)) got.(i))
        fs)

let scale =
  prop "the scale is max (10^v - 1) 0.1"
    Gen.(map f32 (float_range (-1.) 5.))
    (fun v ->
      cover "at the floor" ((10. ** v) -. 1. < 0.1);
      let s = Shadow.scale (tensor [| 1 |] [| v |]) in
      within ~what:"scale"
        (floats s.e).(0)
        (Float.max ((10. ** v) -. 1.) 0.1)
        (floats s.x).(0))

(* Upstream's interpolation, in float64: the interval [x(i), x(i+1)] with i the
   number of wavelengths below the key, less one, kept in [0, m-2]. *)
let interp x y key =
  let m = Array.length x in
  if key < x.(0) || key > x.(m - 1) then 0.
  else begin
    let below = Array.fold_left (fun n v -> if v < key then n + 1 else n) 0 x in
    let i = max 0 (min (m - 2) (below - 1)) in
    let dx = x.(i) -. x.(i + 1) in
    let slope = if dx = 0. then 0. else (y.(i) -. y.(i + 1)) /. dx in
    y.(i) +. ((key -. x.(i)) *. slope)
  end

(* Five roundings of the interval's step and one of the sum. *)
let interp_bound x y key got =
  let m = Array.length x in
  if key < x.(0) || key > x.(m - 1) then 0.
  else
    let below = Array.fold_left (fun n v -> if v < key then n + 1 else n) 0 x in
    let i = max 0 (min (m - 2) (below - 1)) in
    (6. *. u *. Float.abs (got -. y.(i))) +. (2. *. u *. Float.abs got)

let resample =
  prop "resampling interpolates between the pixels around each key" rows
    (fun (fs, _, _, ws) ->
      let fs =
        Array.map (Array.map (fun v -> if Float.is_finite v then v else 1.)) fs
      in
      let x =
        tensor
          [| Array.length ws; Array.length ws.(0) |]
          (Array.concat (Array.to_list ws))
      in
      let y = tensor (Nx.shape x) (Array.concat (Array.to_list fs)) in
      let keys = Array.init grid_length grid_at in
      let got =
        floats (Spectrum.resample ~x ~y (tensor [| grid_length |] keys))
      in
      Array.iteri
        (fun r w ->
          let m = Array.length w in
          cover "a key equals a wavelength"
            (Array.exists (fun k -> Array.mem k w) keys);
          cover "repeated wavelengths"
            (Array.exists
               (fun i -> i > 0 && w.(i) = w.(i - 1))
               (Array.init m Fun.id));
          cover "keys beyond the last wavelength"
            (keys.(grid_length - 1) > w.(m - 1));
          Array.iteri
            (fun k key ->
              let g = got.((r * grid_length) + k) in
              within
                ~what:(Printf.sprintf "row %d at %g" r key)
                (interp_bound w fs.(r) key g)
                (interp w fs.(r) key)
                g)
            keys)
        ws)

(* Decoding resamples from the grid: one wavelength axis, keys of [b; k]. *)
let resample_from_grid =
  prop "resampling from the grid reads each row's keys"
    Gen.(array ~size:(int_range 1 9) (map f32 (float_range 3400. 3580.)))
    (fun keys ->
      let gx = Array.init grid_length grid_at in
      let gy = Array.init grid_length (fun k -> f32 (sin (float_of_int k))) in
      Array.iter
        (fun k ->
          cover "outside the grid" (k < gx.(0) || k > gx.(grid_length - 1)))
        keys;
      let y = tensor [| 2; grid_length |] (Array.append gy gy) in
      let n = Array.length keys in
      let k2 = tensor [| 2; n |] (Array.append keys keys) in
      let got =
        floats (Spectrum.resample ~x:(tensor [| grid_length |] gx) ~y k2)
      in
      Array.iteri
        (fun i key ->
          let g = got.(n + i) in
          within
            ~what:(Printf.sprintf "at %g" key)
            (interp_bound gx gy key g) (interp gx gy key) g)
        keys)

(* The prepared input in float64: signal and noise of the token's value, on the
   grid, through asinh. *)
let prepared t (fs, vs, ws) code =
  let v = (floats (Spectrum.levels t)).(code) in
  let s = Float.max ((10. ** v) -. 1.) 0.1 in
  let fix x =
    if Float.is_nan x then 0.
    else Float.max (-.float32_max) (Float.min float32_max x)
  in
  let signal = Array.map (fun f -> ((fix f /. s) -. 1.) *. 0.2) fs in
  let noise =
    Array.map
      (fun v -> Float.sqrt (Float.max 0. (Float.min 100. (fix v))) /. s *. 0.2)
      vs
  in
  Array.init grid_length (fun k ->
      let key = grid_at k in
      (Float.asinh (interp ws signal key), Float.asinh (interp ws noise key)))

let prepare =
  let t = random 1 in
  prop "preparation follows the preamble's formula" rows
    (fun (fs, vs, ms, ws) ->
      let x = spectrum ~mask:ms ~ivar:vs fs ws in
      let a, code = Shadow.prepare t x in
      let got = floats a.x and e = floats a.e and code = ints code in
      Array.iteri
        (fun r w ->
          cover "pads after the row"
            (w.(Array.length w - 1) < grid_at (grid_length - 1));
          let expected = prepared t (fs.(r), vs.(r), w) code.(r) in
          Array.iteri
            (fun k (sg, ns) ->
              let at c = (((r * grid_length) + k) * 2) + c in
              within ~what:"signal" e.(at 0) sg got.(at 0);
              within ~what:"noise" e.(at 1) ns got.(at 1))
            expected)
        ws)

let norm_token =
  let t = random 2 in
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
        (ints (Nx.slice [ A; I 0 ] (Spectrum.encode_unpadded t x))))

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
      (* DIVERGENCES: upstream's F.pad with a negative width cuts the row. *)
      test "encode refuses a row longer than the modality's" (fun () ->
          let x =
            spectrum [| Array.make 4801 1. |] [| Array.init 4801 float_of_int |]
          in
          raises_match (Exn.invalid_arg ~sub:"at most 4800") (fun () ->
              Spectrum.encode (random 8) Modality.Sdss x));
    ]

(* Lengths *)

(* Rows with a length each, the pixels past it holding Multimodal Universe's -1
   wavelengths or NaN, which an encode that read them would refuse. *)
let with_lengths =
  Gen.(
    let* fs, vs, ms, ws = rows in
    let m = Array.length ws.(0) in
    let+ ls =
      array ~size:(constant (Array.length ws))
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
          cover "rows of two lengths"
            (Array.exists (fun l -> l <> ls.(0)) ls);
          let x = spectrum ~mask:ms ~ivar:vs fs ws in
          let lengths =
            Nx.create Nx.int64 [| Array.length ls |] (Array.map Int64.of_int ls)
          in
          let got = Spectrum.encode ~lengths t Modality.Sdss x in
          Array.iteri
            (fun i l ->
              let row a = [| Array.sub a.(i) 0 l |] in
              let alone =
                spectrum ~mask:(row ms) ~ivar:(row vs) (row fs) (row ws)
              in
              equal (array int)
                (ints (Spectrum.encode t Modality.Sdss alone))
                (ints (Nx.slice [ R (i, i + 1) ] got)))
            ls);
      test "encode refuses a length outside [1, m]" (fun () ->
          let x = spectrum [| [| 1.; 2. |] |] [| [| 3600.; 3601. |] |] in
          List.iter
            (fun l ->
              raises_match (Exn.invalid_arg ~sub:"outside [1, 2]") (fun () ->
                  Spectrum.encode
                    ~lengths:(Nx.create Nx.int64 [| 1 |] [| l |])
                    t Modality.Sdss x))
            [ 0L; 3L ]);
    ]

(* Tokens *)

let codes =
  group "LFQ"
    [
      cases "value c sets bit 9 - c where positive" ~id:string_of_int
        [ 0; 1; 5; 9 ] (fun c ->
          let h = Array.init 10 (fun i -> if i = c then 0.5 else -0.5) in
          equal (array int)
            [| 1 lsl (9 - c) |]
            (ints (Spectrum.codes (tensor [| 1; 10 |] h))));
      test "a value of 0 or NaN clears its bit" (fun () ->
          let h = [| 0.; -0.; Float.nan; 1.; 1.; 1.; 1.; 1.; 1.; 1. |] in
          equal (array int) [| 127 |]
            (ints (Spectrum.codes (tensor [| 1; 10 |] h))));
      prop "a code's bits are its values' signs"
        Gen.(array ~size:(constant 10) (map f32 (float_range (-2.) 2.)))
        (fun h ->
          let k = ints (Spectrum.codes (tensor [| 1; 10 |] h)) in
          let expected =
            Array.fold_left ( + ) 0
              (Array.mapi (fun c v -> if v > 0. then 1 lsl (9 - c) else 0) h)
          in
          equal (array int) [| expected |] k);
      test "values invert codes on every token" (fun () ->
          let k = Nx.arange Nx.int64 0 1024 1 in
          let v = Spectrum.values Nx.float32 k in
          equal (array int) (ints k) (ints (Spectrum.codes v));
          equal (list float_exact) [ -1.; 1. ]
            (List.sort_uniq compare (Array.to_list (floats v))));
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
            Spectrum.encode t Modality.Desi
              (spectrum (flat 3 50 2.) (waves 3 50))
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
   sigmoid z > 0.5; a logit of 0 is a good pixel. *)
let decoded =
  cases "decode applies the normalisation and the mask's rule"
    ~id:(fun (y, z, _, _) -> Printf.sprintf "y %g, z %g" y z)
    [ (0., 0., 1., false); (0.25, 2., 1.25, true); (-0.5, -3., 0.5, false) ]
    (fun (y, z, factor, bad) ->
      let t = constant (y, z) in
      let code = 700 in
      let v = (floats (Spectrum.levels t)).(code) in
      let s = (floats (Spectrum.scale (tensor [| 1 |] [| v |]))).(0) in
      let k =
        Nx.create Nx.int64 [| 1; 3 |] [| Int64.of_int code; 3L; 1000L |]
      in
      let d = Spectrum.decode t k in
      Array.iter
        (fun f -> equal float_exact (f32 (factor *. s)) f)
        (floats d.flux);
      Array.iter (equal bool bad) (bools d.mask);
      Array.iter (equal float_exact 1.) (floats d.ivar))

(* Upstream's network. upstream/spectrum_codec.safetensors is a few-channel
   codec of AION-1's layout with random weights, GRN's included, its encoder's
   last layer norm scaled so that the pre-quantizer norm's lane variance sits
   near its eps; upstream/spectrum_golden.safetensors holds what upstream's
   PyTorch code computes with it in float64 on 16 short spectra, unpadded (its
   base Spectrum class) and padded as DESI and SDSS spectra, and the spread of
   its rounding ensemble at float32's scale ([Spread]); and one GRN whose
   channels' mean norm is near its eps. The generator is named in the dune file.
   Every value lies within [τ ŝ] of the float64 reference, and every decision, a
   token's bit, the normalisation's token or a pixel's mask, is the reference's
   but within [τ ŝ] of its boundary. *)

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
  let encode name =
    match name with
    | "base" -> Spectrum.encode_unpadded
    | "desi" -> fun t -> Spectrum.encode t Modality.Desi
    | _ -> fun t -> Spectrum.encode t Modality.Sdss
  in
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
      cases "encode gives upstream's tokens" ~id:fst variants (fun (name, _) ->
          let t = Lazy.force upstream_codec in
          let k = encode name t (input ()) in
          let edges = Array.sub (floats (Spectrum.levels t)) 0 1023 in
          holds "normalisation token"
            (Spread.levels ~n:(n ()) ~edges
               ~reference:(f64s (v name "norm_pre"))
               ~spread:(f64s (s name "norm_pre"))
               (ints (Nx.slice [ A; I 0 ] k)));
          let bits =
            Spectrum.values Nx.float64 (Nx.slice [ A; R (1, Nx.dim 1 k) ] k)
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
      test "GRN is upstream's where its eps matters" (fun () ->
          let y =
            Spectrum.grn
              ~gamma:(get Nx.float32 "grn.gamma")
              ~beta:(get Nx.float32 "grn.beta")
              (get Nx.float32 "grn.x")
          in
          holds "GRN"
            (Spread.values ~n:(n ()) ~reference:(f64s "grn.ref")
               ~spread:(f64s "grn.s") (floats y)));
    ]

(* Departures from upstream, each in DIVERGENCES.md *)

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
              Spectrum.encode t Modality.Sdss x));
      test "encode: refuses a NaN wavelength" (fun () ->
          let t = random 5 in
          let x = spectrum (flat 1 3 1.) [| [| 3600.; Float.nan; 3602. |] |] in
          raises_match (Exn.invalid_arg ~sub:"NaN") (fun () ->
              Spectrum.encode t Modality.Sdss x));
      test "resample: a repeated first wavelength reads its value" (fun () ->
          let x = tensor [| 1; 3 |] [| 3500.; 3500.; 3510. |] in
          let y = tensor [| 1; 3 |] [| 2.; 3.; 4. |] in
          let got =
            floats (Spectrum.resample ~x ~y (tensor [| 1 |] [| 3500. |]))
          in
          equal (array float_exact) [| 2. |] got);
      cases "decode: refuses a token outside [0, 1024)" ~id:string_of_int
        [ -1; 1024 ] (fun k ->
          let t = random 6 in
          raises_match (Exn.invalid_arg ~sub:"outside [0, 1024)") (fun () ->
              Spectrum.decode t
                (Nx.create Nx.int64 [| 1; 3 |] [| 0L; Int64.of_int k; 0L |])));
    ]

let refusals =
  group "refusals"
    [
      test "encode_unpadded: rows of one pixel" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Spectrum.encode_unpadded (random 7)
                (spectrum [| [| 1. |] |] [| [| 3600. |] |])));
      test "encode: fields of different shapes" (fun () ->
          let x = spectrum (flat 1 4 1.) (waves 1 4) in
          raises_match Exn.invalid_arg (fun () ->
              Spectrum.encode (random 7) Modality.Sdss
                { x with Spectrum.ivar = Nx.ones Nx.float32 [| 1; 3 |] }));
      test "decode: tokens of the wrong count" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Spectrum.decode (random 7) (Nx.zeros Nx.int64 [| 1; 4 |])));
      test "of_archive: a missing block leaves a weight unread" (fun () ->
          let a = archive (fun _ shape -> Nx.zeros Nx.float32 shape) in
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
          raises_match (Exn.failure ~sub:"no weight") (fun () ->
              Spectrum.of_archive a));
    ]

let () =
  exit
    (run "aion.spectrum"
       [
         normalisation;
         scale;
         resample;
         resample_from_grid;
         prepare;
         norm_token;
         pad;
         lengths;
         codes;
         shapes;
         decoded;
         upstream;
         divergences;
         refusals;
       ])
