(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The image codec on a narrow network of random weights, named as upstream's
   state dict names them: the steps' formulas, shapes and the laws that tie
   bands together. The full network's agreement with upstream is
   test/weights/test_image_weights.ml, outside runtest. *)

open Windtrap
open Aion

let u32 = 0x1p-24

(* A narrow codec: [p] projected channels, widths [d], [2d] and [4d], and
   squeeze-excite widths [h]. Each tensor is drawn from its own key, at a scale
   that keeps activations of order one. *)
let narrow ?(p = 3) ?(d = 2) ?(h = 3) seed =
  let entries = ref [] and k = ref seed in
  let draw shape scale =
    incr k;
    Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key !k) Nx.float32 shape) scale
  in
  let add name x = entries := (name, Nx.P x) :: !entries in
  let conv name o i kk =
    let fan = Float.of_int (i * kk * kk) in
    add (name ^ ".weight") (draw [| o; i; kk; kk |] (1. /. Float.sqrt fan));
    add (name ^ ".bias") (draw [| o |] 0.1)
  in
  let ru name c =
    conv (name ^ ".net.0.conv") c c 3;
    conv (name ^ ".net.2") c c 1;
    conv (name ^ ".net.4.to_k") 1 c 1;
    conv (name ^ ".net.4.net.0") h c 1;
    conv (name ^ ".net.4.net.2") c h 1
  in
  let rus name c =
    List.iter (fun i -> ru (Printf.sprintf "%s.%d" name i) c) [ 0; 1; 2; 3 ]
  in
  let d1 = 2 * d and d2 = 4 * d in
  conv "model.conv_in.conv" d p 7;
  ru "model.encoder_layers.0" d;
  conv "model.encoder_layers.1.conv" d1 d 3;
  rus "model.encoder_layers.2" d1;
  conv "model.encoder_layers.3.conv" d2 d1 3;
  rus "model.encoder_layers.4" d2;
  conv "model.encoder_layers.5.conv" d2 d2 7;
  rus "model.encoder_layers.6" d2;
  add "model.encoder_layers.7.1.weight" (Nx.add_s (draw [| d2 |] 0.1) 1.);
  add "model.encoder_layers.7.1.bias" (draw [| d2 |] 0.1);
  conv "pre_quant_proj" 5 d2 1;
  conv "post_quant_proj" d2 5 1;
  rus "model.decoder_layers.0" d2;
  conv "model.decoder_layers.1.conv" d2 d2 3;
  rus "model.decoder_layers.2" d2;
  conv "model.decoder_layers.3.net.0" (4 * d1) d2 1;
  rus "model.decoder_layers.4" d1;
  conv "model.decoder_layers.5.net.0" (4 * d) d1 1;
  ru "model.decoder_layers.6" d;
  conv "model.conv_out.conv" p d 3;
  add "subsample_in.weight" (draw [| p; 9 |] (1. /. 3.));
  add "subsample_in.bias" (draw [| p |] 0.1);
  add "subsample_out.weight"
    (draw [| 9; p |] (1. /. Float.sqrt (Float.of_int p)));
  add "subsample_out.bias" (draw [| 9 |] 0.1);
  add "_quantizer.levels" (Nx.create Nx.int32 [| 5 |] [| 7l; 5l; 5l; 5l; 5l |]);
  Nx_io.Archive.of_list !entries

let archive = narrow 7
let codec () = Image.of_archive archive

let entry name =
  Nx_io.Archive.float
    ~shape:
      (match Nx_io.Archive.find name archive with
      | Some (Nx.P x) -> Nx.shape x
      | None -> [||])
    Nx.float64 name archive

(* Distinct bands, in any order: one, all nine, or any set. *)
let bands_gen =
  Gen.(
    let any =
      let* keep = array ~size:(constant 9) bool in
      let* order = array ~size:(constant 9) (int_range 0 1000) in
      let bs = List.filteri (fun i _ -> keep.(i)) Image.bands in
      let bs = if bs = [] then [ Image.Des_g ] else bs in
      let ranked = List.map (fun b -> (order.(Image.channel b), b)) bs in
      constant (List.map snd (List.sort compare ranked))
    in
    frequency
      [
        (1, map (fun b -> [ b ]) (of_list Image.bands));
        (1, constant Image.bands);
        (2, any);
      ])

(* Images whose pixels pass their bands' limits, tie them, sit near zero, or are
   zero, at sizes whose crop starts at even and odd offsets. *)
let images_gen =
  Gen.(
    with_pp
      (fun ppf (bs, h, w, seed) ->
        Format.fprintf ppf "[%s] %d × %d, seed %d"
          (String.concat "; " (List.map Image.name bs))
          h w seed)
      (let* bs = bands_gen in
       let* h = of_list [ 96; 97; 100; 131 ] in
       let* w = of_list [ 96; 99; 120 ] in
       let+ seed = int_range 0 1_000_000 in
       (bs, h, w, seed)))

(* The first five elements where [a] and [b] differ in bits, as (index, a,
   b). *)
let differences a b =
  let a = Nx.to_array a and b = Nx.to_array b in
  let found = ref [] in
  Array.iteri
    (fun i x ->
      if
        List.length !found < 5
        && Int64.bits_of_float x <> Int64.bits_of_float b.(i)
      then found := (i, x, b.(i)) :: !found)
    a;
  List.rev !found

let same a b =
  equal (list (triple int float_exact float_exact)) [] (differences a b)

let images bs h w seed =
  let c = List.length bs in
  let n = Nx.Rng.normal ~key:(Nx.Rng.key seed) Nx.float32 [| 2; c; h; w |] in
  let scale =
    Nx.create Nx.float32 [| 1; c; 1; 1 |]
      (Array.of_list (List.map (fun b -> 0.8 *. Image.limit b) bs))
  in
  let x = Nx.mul n scale in
  (* A few exact values: the limits, their negations, zero and tiny ones. *)
  let special =
    Nx.Rng.uniform ~key:(Nx.Rng.key (seed + 1)) Nx.float32 [| 2; c; h; w |]
  in
  let lim = Nx.broadcast_to [| 2; c; h; w |] (Nx.div_s scale 0.8) in
  let x = Nx.where (Nx.less_s special 0.02) lim x in
  let x =
    Nx.where
      (Nx.logical_and
         (Nx.greater_equal_s special 0.02)
         (Nx.less_s special 0.04))
      (Nx.neg lim) x
  in
  let x =
    Nx.where
      (Nx.logical_and
         (Nx.greater_equal_s special 0.04)
         (Nx.less_s special 0.06))
      (Nx.mul_s x 1e-4) x
  in
  Nx.where
    (Nx.logical_and (Nx.greater_equal_s special 0.06) (Nx.less_s special 0.07))
    (Nx.zeros_like x) x

(* HSC pixels in nanomaggies: 10^((22.5 - 27) / 2.5) of their value. *)
let to_nmgy b = if Image.channel b < 5 then 10. ** -1.8 else 1.
let compress x = 10. *. 0.01 *. Float.asinh (x /. 0.01)

(* The preamble's preparation of pixel [(i, j)] of band [b], exactly. *)
let prepared b v =
  let m = Image.limit b in
  compress (Float.max (-.m) (Float.min m v) *. to_nmgy b)

(* [f b v i] for each pixel [v] of band [b] that the crop keeps, [i] its index
   in the cropped batch, of two images: the first five [Some]. *)
let each_pixel xs bs h w f =
  let c = List.length bs in
  let r0 = (h - 96) / 2 and c0 = (w - 96) / 2 in
  let found = ref [] and count = ref 0 in
  List.iteri
    (fun ci b ->
      for n = 0 to 1 do
        for i = 0 to 95 do
          for j = 0 to 95 do
            let v = xs.((((((n * c) + ci) * h) + i + r0) * w) + j + c0) in
            match f b v ((((((n * c) + ci) * 96) + i) * 96) + j) with
            | Some r when !count < 5 ->
                incr count;
                found := r :: !found
            | _ -> ()
          done
        done
      done)
    bs;
  List.rev !found

let prepare_formula =
  prop "prepare crops, clamps, converts to nanomaggies and compresses"
    images_gen (fun (bs, h, w, seed) ->
      let x = images bs h w seed in
      let p = Image.prepare bs x in
      equal (array int) [| 2; List.length bs; 96; 96 |] (Nx.shape p);
      let r0 = (h - 96) / 2 and c0 = (w - 96) / 2 in
      cover "an odd crop offset" (r0 mod 2 = 1 || c0 mod 2 = 1);
      cover "an HSC band" (List.exists (fun b -> Image.channel b < 5) bs);
      cover "a Legacy Survey band"
        (List.exists (fun b -> Image.channel b >= 5) bs);
      let ps = Nx.to_array p in
      let each_pixel = each_pixel (Nx.to_array x) in
      (* Each step rounds once, the constants 0.01 and 10^-1.8 once each, and
         asinh within 2 ulps of its rounded input: within 16 float32 roundings
         of the exact value, relatively. *)
      let clamped = ref false in
      let refuted =
        each_pixel bs h w (fun b v i ->
            if Float.abs v > Image.limit b then clamped := true;
            let want = prepared b v in
            if Float.abs (ps.(i) -. want) > 16. *. u32 *. Float.abs want then
              Some (want, ps.(i))
            else None)
      in
      cover "a clamped pixel" !clamped;
      equal (list (pair float_exact float_exact)) [] refuted)

(* A band's preparation is its own: a list prepares as each band alone. *)
let per_band =
  prop "each band is prepared by its own survey" images_gen
    (fun (bs, h, w, seed) ->
      let x = images bs h w seed in
      let mixed =
        List.exists (fun b -> Image.channel b < 5) bs
        && List.exists (fun b -> Image.channel b >= 5) bs
      in
      cover "HSC and Legacy Survey bands together" mixed;
      let alone =
        List.mapi
          (fun i b -> Image.prepare [ b ] (Nx.slice [ A; R (i, i + 1) ] x))
          bs
      in
      same (Nx.concatenate ~axis:1 alone) (Image.prepare bs x))

(* The nine channels of prepared bands, zeros elsewhere. *)
let nine bs p =
  let zero = Nx.zeros Nx.float32 [| Nx.dim 0 p; 1; 96; 96 |] in
  Nx.concatenate ~axis:1
    (List.map
       (fun b ->
         let rec find i = function
           | [] -> zero
           | b' :: rest ->
               if b' = b then Nx.slice [ A; R (i, i + 1) ] p
               else find (i + 1) rest
         in
         find 0 bs)
       Image.bands)

let restores =
  prop "restore inverts the conversion and compression of prepare" images_gen
    (fun (bs, h, w, seed) ->
      let x = images bs h w seed in
      let p = Image.prepare bs x in
      let r = Nx.to_array (Image.restore bs (nine bs p)) in
      (* asinh then sinh of up to 8 (500 HSC units, 7.9 nanomaggies, over 0.01):
         each step's relative rounding grows by at most the sinh's condition
         number, below 9, over some 20 roundings. *)
      let refuted =
        each_pixel (Nx.to_array x) bs h w (fun b v i ->
            let m = Image.limit b in
            let want = Float.max (-.m) (Float.min m v) in
            if Float.abs (r.(i) -. want) > 256. *. u32 *. Float.abs want then
              Some (want, r.(i))
            else None)
      in
      equal (list (pair float_exact float_exact)) [] refuted)

(* The projection in: each pixel's nine channels by subsample_in's weights, plus
   its bias, times √(9 / c). *)
let embeds =
  prop "embed projects the nine channels, scaled by √(9 / c)" images_gen
    (fun (bs, h, w, seed) ->
      let c = List.length bs in
      cover "one band" (c = 1);
      cover "nine bands" (c = 9);
      let p = Image.prepare bs (images bs h w seed) in
      let e = Image.embed (codec ()) bs p in
      let m = Nx.dim 0 (entry "subsample_in.bias") in
      equal (array int) [| 2; 96; 96; m |] (Nx.shape e);
      let x9 = Nx.cast Nx.float64 (Nx.moveaxis 1 3 (nine bs p)) in
      let wt = Nx.transpose (entry "subsample_in.weight") in
      let s = Float.sqrt (9. /. Float.of_int c) in
      let want =
        Nx.mul_s (Nx.add (Nx.matmul x9 wt) (entry "subsample_in.bias")) s
      in
      let mag =
        Nx.mul_s
          (Nx.add
             (Nx.matmul (Nx.abs x9) (Nx.abs wt))
             (Nx.abs (entry "subsample_in.bias")))
          s
      in
      (* Nine products and a bias, each rounding once, then the scale: within 16
         roundings of the magnitude of the terms. *)
      let got = Nx.to_array (Nx.cast Nx.float64 e) in
      let want = Nx.to_array want and mag = Nx.to_array mag in
      let refuted = ref [] in
      Array.iteri
        (fun i g ->
          if
            Float.abs (g -. want.(i)) > 16. *. u32 *. mag.(i)
            && List.length !refuted < 5
          then refuted := (want.(i), g) :: !refuted)
        got;
      equal (list (pair float_exact float_exact)) [] !refuted)

let x0 () = images [ Image.Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y ] 100 120 3

let shapes =
  group "shapes"
    [
      test "latent, encode and decode" (fun () ->
          let bs = Image.[ Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y ] in
          let z = Image.latent (codec ()) bs (x0 ()) in
          equal (array int) [| 2; 576; 5 |] (Nx.shape z);
          let k = Image.encode (codec ()) bs (x0 ()) in
          equal (array int) [| 2; 576 |] (Nx.shape k);
          equal (array int64)
            (Nx.to_array (Fsq.encode (Fsq.v [ 7; 5; 5; 5; 5 ]) z))
            (Nx.to_array k);
          let d = Image.decode (codec ()) Image.[ Des_r; Hsc_y ] k in
          equal (array int) [| 2; 2; 96; 96 |] (Nx.shape d));
      test "encode leaves its input as it was" (fun () ->
          let bs = Image.[ Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y ] in
          let x = x0 () in
          let before = Nx.copy x in
          ignore (Image.encode (codec ()) bs x);
          same before x);
    ]

(* Decoding all nine bands gives each band as decoding it alone does: no band
   takes another's survey's unit. *)
(* AION-1 trained on CUDA, where PyTorch divides by a Python number as a
   product with the float32 reciprocal: [x / 0.01] is [x * 100] and [y / 10]
   is [y * 0.1f]. The values here are those where the product and the
   division differ in float32. *)
let reciprocals =
  let grid = Nx.mul_s (Nx.arange_f Nx.float32 1. 4097. 1.) 1.37e-3 in
  let pick f g =
    let a = Nx.to_array (f grid) and b = Nx.to_array (g grid) in
    let xs = Nx.to_array grid in
    let found = ref [] in
    Array.iteri
      (fun i x ->
        if a.(i) <> b.(i) && List.length !found < 8 then found := x :: !found)
      xs;
    Nx.create Nx.float32
      [| 1; 1; 1; List.length !found |]
      (Array.of_list !found)
  in
  let tile x =
    Nx.broadcast_to [| 1; 1; 96; 96 |]
      (Nx.pad [| (0, 0); (0, 0); (0, 0); (0, 96 - Nx.dim 3 x) |] 0. x)
  in
  group "divisions by constants are products with float32 reciprocals"
    [
      test "prepare" (fun () ->
          let x =
            tile (pick (fun x -> Nx.div_s x 0.01) (fun x -> Nx.mul_s x 100.))
          in
          let want =
            Nx.mul_s (Nx.mul_s (Nx.asinh (Nx.mul_s x 100.)) 0.01) 10.
          in
          same want (Image.prepare [ Image.Des_g ] x));
      test "restore" (fun () ->
          let y =
            tile
              (pick
                 (fun y -> Nx.div_s y 10.)
                 (fun y -> Nx.mul_s y (Bounds.f32 0.1)))
          in
          let want =
            Nx.mul_s
              (Nx.sinh (Nx.mul_s (Nx.mul_s y (Bounds.f32 0.1)) 100.))
              0.01
          in
          let nine = Nx.concatenate ~axis:1 (List.init 9 (fun _ -> y)) in
          same want (Image.restore [ Image.Des_g ] nine));
    ]

(* Upstream's network. upstream/image_codec.safetensors is a narrow codec of
   AION-1's layout (widths 2, 4 and 8, three projected channels) with random
   weights, designed so that each constant matters: the squeeze-excite gates
   open, the final layer norm's lane variance near its eps, two bands of nine.
   upstream/image_golden.safetensors holds what upstream's PyTorch code computes
   with it in float64 on a Legacy Survey and an HSC image of two bands, and the
   spread of its rounding ensemble at float32's scale and at float64's
   ([Spread]); the generator is named in the dune file. At both dtypes every
   latent and decoded value lies within [τ ŝ] of the float64 reference, and
   every digit of a token is the reference's but where the reference's bounded
   value lies within [τ ŝ] of a rounding boundary. *)

let fixture name = Nx_io.load_safetensors ("upstream/" ^ name ^ ".safetensors")
let upstream_codec = lazy (Image.of_archive (fixture "image_codec"))
let golden = lazy (fixture "image_golden")

let golden_get dt name =
  match Nx_io.Archive.find name (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let floats t = Nx.to_array (Nx.cast Nx.float64 t)
let ensemble_n () = Int64.to_int (Nx.to_array (golden_get Nx.int64 "n")).(0)

let upstream_sets =
  [ ("ls", Image.[ Des_g; Des_z ]); ("hsc", Image.[ Hsc_i; Hsc_y ]) ]

let holds name (c : Spread.check) =
  if c.failures <> [] then failf "%a" (Spread.pp name) c

(* A token's numbers rounded, [q + 3] for the integer [q] each is rounded to:
   the count of the edges below it of the widest level, seven. *)
let image_levels = [ 7; 5; 5; 5; 5 ]
let digit_edges = [| -2.5; -1.5; -0.5; 0.5; 1.5; 2.5 |]

let digits k =
  Array.concat
    (List.map
       (fun k ->
         let k = Int64.to_int k in
         Array.of_list
           (List.rev
              (snd
                 (List.fold_left
                    (fun (k, ds) l -> (k / l, ((k mod l) - (l / 2) + 3) :: ds))
                    (k, []) image_levels))))
       (Array.to_list (Nx.to_array k)))

(* The two checks: the codec in float32 against the ensemble at float32's scale,
   and cast to float64 against the ensemble at float64's. *)
let rows =
  List.concat_map
    (fun (precision, scale) ->
      List.map (fun (name, bs) -> (precision, scale, name, bs)) upstream_sets)
    [ (`F32, "order"); (`F64, "order64") ]

let row_id (precision, _, name, _) =
  (match precision with `F32 -> "float32 " | `F64 -> "float64 ") ^ name

let codec64 = lazy (Image.cast Nx.float64 (Lazy.force upstream_codec))

let latent_at precision bs x =
  match precision with
  | `F32 -> floats (Image.latent (Lazy.force upstream_codec) bs x)
  | `F64 -> floats (Image.latent (Lazy.force codec64) bs (Nx.cast Nx.float64 x))

let encode_at precision bs x =
  match precision with
  | `F32 -> Image.encode (Lazy.force upstream_codec) bs x
  | `F64 -> Image.encode (Lazy.force codec64) bs (Nx.cast Nx.float64 x)

let decode_at precision bs k =
  match precision with
  | `F32 -> floats (Image.decode (Lazy.force upstream_codec) bs k)
  | `F64 -> floats (Image.decode (Lazy.force codec64) bs k)

let upstream_network =
  let input name = golden_get Nx.float32 ("input." ^ name) in
  let ref name part =
    floats (golden_get Nx.float64 ("f64." ^ name ^ "." ^ part))
  in
  let spread scale name part =
    floats (golden_get Nx.float32 ("s." ^ scale ^ "." ^ name ^ "." ^ part))
  in
  (* The reference's [1; 5; 576] as the codec's [1; 576; 5]. *)
  let cells a =
    floats (Nx.moveaxis 1 2 (Nx.create Nx.float64 [| 1; 5; 576 |] a))
  in
  group "upstream's network"
    [
      cases "the latent is upstream's" ~id:row_id rows
        (fun (precision, scale, name, bs) ->
          holds "latent"
            (Spread.values ~n:(ensemble_n ())
               ~reference:(cells (ref name "pre"))
               ~spread:(cells (spread scale name "pre"))
               (latent_at precision bs (input name))));
      cases "encode gives upstream's tokens" ~id:row_id rows
        (fun (precision, scale, name, bs) ->
          holds "token digits"
            (Spread.levels ~n:(ensemble_n ()) ~edges:digit_edges
               ~reference:(ref name "bounded")
               ~spread:(spread scale name "bounded")
               (digits (encode_at precision bs (input name)))));
      cases "decode gives upstream's images" ~id:row_id rows
        (fun (precision, scale, name, bs) ->
          let k = golden_get Nx.int64 ("f64." ^ name ^ ".tokens") in
          holds "decoded"
            (Spread.values ~n:(ensemble_n ()) ~reference:(ref name "decoded")
               ~spread:(spread scale name "decoded")
               (decode_at precision [ List.hd bs ] k)));
    ]

let decodes_per_band =
  test "each decoded band is in its own unit" (fun () ->
      let k =
        Image.encode (codec ())
          Image.[ Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y ]
          (x0 ())
      in
      let all = Image.decode (codec ()) Image.bands k in
      let alone =
        List.map (fun b -> Image.decode (codec ()) [ b ] k) Image.bands
      in
      same (Nx.concatenate ~axis:1 alone) all)

let refusals =
  let x = Nx.zeros Nx.float32 [| 1; 2; 96; 96 |] in
  let bad name f = test name (fun () -> raises_match Exn.invalid_arg f) in
  group "refusals"
    [
      bad "no band" (fun () ->
          Image.prepare [] (Nx.zeros Nx.float32 [| 1; 0; 96; 96 |]));
      bad "a band twice" (fun () -> Image.prepare Image.[ Des_g; Des_g ] x);
      bad "fewer bands than the array's" (fun () ->
          Image.prepare Image.[ Des_g ] x);
      bad "an image under 96 rows" (fun () ->
          Image.prepare Image.[ Des_g ] (Nx.zeros Nx.float32 [| 1; 1; 95; 96 |]));
      bad "an image under 96 columns" (fun () ->
          Image.prepare Image.[ Des_g ] (Nx.zeros Nx.float32 [| 1; 1; 96; 95 |]));
      bad "a batch of rank 3" (fun () ->
          Image.prepare Image.[ Des_g ] (Nx.zeros Nx.float32 [| 1; 96; 96 |]));
      bad "tokens not [n; 576]" (fun () ->
          Image.decode (codec ())
            Image.[ Des_g ]
            (Nx.zeros Nx.int64 [| 1; 575 |]));
      bad "decode with a band twice" (fun () ->
          Image.decode (codec ())
            Image.[ Des_g; Des_g ]
            (Nx.zeros Nx.int64 [| 1; 576 |]));
    ]

let bands =
  test "bands, channels, names and units" (fun () ->
      equal (list int)
        [ 0; 1; 2; 3; 4; 5; 6; 7; 8 ]
        (List.map Image.channel Image.bands);
      equal (list string)
        [
          "HSC-G";
          "HSC-R";
          "HSC-I";
          "HSC-Z";
          "HSC-Y";
          "DES-G";
          "DES-R";
          "DES-I";
          "DES-Z";
        ]
        (List.map Image.name Image.bands);
      equal (list float_exact)
        [ 80.; 110.; 200.; 330.; 500.; 6.; 15.; 20.; 25. ]
        (List.map Image.limit Image.bands);
      let nmgy = Ymir.Unit.nano Ymir.Units.maggy in
      List.iter
        (fun b ->
          let want =
            if Image.channel b < 5 then Ymir.Units.zero_point "27" else nmgy
          in
          equal string (Ymir.Unit.to_string want)
            (Ymir.Unit.to_string (Image.unit b)))
        Image.bands)

(* Compiled codecs. A compiled codec meets upstream as the eager one does, at
   both rungs, and refuses at the call on a set without eager kernels, where a
   check in its program would raise only at a later wait. *)

module Soft = (val Nx.devices [ Result.get_ok (Rig.software "aion-image") ])

let compiled =
  let c32 = lazy (Image.jit (Lazy.force upstream_codec)) in
  let c64 = lazy (Image.jit (Lazy.force codec64)) in
  let input name = golden_get Nx.float32 ("input." ^ name) in
  let ref name part =
    floats (golden_get Nx.float64 ("f64." ^ name ^ "." ^ part))
  in
  let spread scale name part =
    floats (golden_get Nx.float32 ("s." ^ scale ^ "." ^ name ^ "." ^ part))
  in
  let encode rung bs x =
    match rung with
    | `F32 -> Image.encode (Lazy.force c32) bs x
    | `F64 -> Image.encode (Lazy.force c64) bs (Nx.cast Nx.float64 x)
  in
  let decode rung bs k =
    match rung with
    | `F32 -> floats (Image.decode (Lazy.force c32) bs k)
    | `F64 -> floats (Image.decode (Lazy.force c64) bs k)
  in
  let soft =
    lazy (Image.jit (Image.place Soft.on (Lazy.force upstream_codec)))
  in
  group "compiled"
    [
      cases "a compiled codec encodes upstream's tokens" ~id:row_id rows
        (fun (rung, scale, name, bs) ->
          holds "token digits"
            (Spread.levels ~n:(ensemble_n ()) ~edges:digit_edges
               ~reference:(ref name "bounded")
               ~spread:(spread scale name "bounded")
               (digits (encode rung bs (input name)))));
      cases "a compiled codec decodes upstream's images" ~id:row_id rows
        (fun (rung, scale, name, bs) ->
          let k = golden_get Nx.int64 ("f64." ^ name ^ ".tokens") in
          holds "decoded"
            (Spread.values ~n:(ensemble_n ()) ~reference:(ref name "decoded")
               ~spread:(spread scale name "decoded")
               (decode rung [ List.hd bs ] k)));
      test "a compiled codec computes on a set without eager kernels"
        (fun () ->
          (* Its programs compile for the host's CPU there too, so they give the
             host's bits. *)
          let bs = Image.[ Hsc_i; Hsc_y ] in
          let x = input "hsc" in
          let k = Image.encode (Lazy.force c32) bs x in
          let ks = Image.encode (Lazy.force soft) bs (Nx.place Soft.on x) in
          equal (array int64) (Nx.to_array k)
            (Nx.to_array (Nx.place Nx.Host.on ks));
          equal (array float_exact)
            (floats (Image.decode (Lazy.force c32) bs k))
            (floats
               (Nx.place Nx.Host.on
                  (Image.decode (Lazy.force soft) bs (Nx.place Soft.on k)))));
      test "a compiled codec refuses a code outside [0, 4375) at the call"
        (fun () ->
          let k =
            Nx.create Nx.int64 [| 1; 576 |]
              (Array.init 576 (fun i -> if i = 7 then 4375L else 0L))
          in
          raises_match (Exn.invalid_arg ~sub:"code 4375 at [0; 7]") (fun () ->
              Image.decode (Lazy.force soft)
                Image.[ Des_g ]
                (Nx.place Soft.on k)));
    ]

let () =
  exit
    (run "aion.image"
       [
         group "prepare"
           [ prepare_formula; per_band; restores; reciprocals; bands ];
         group "embed" [ embeds ];
         upstream_network;
         compiled;
         shapes;
         group "decode" [ decodes_per_band ];
         refusals;
       ])
