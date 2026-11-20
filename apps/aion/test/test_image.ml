(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The image codec on a narrow network of Image.init: the steps' formulas,
   shapes and the laws that tie bands together; and on a narrow upstream
   network, its agreement with upstream's code, eager and compiled. AION-1's
   network's agreement with upstream is parity.exe image, outside runtest. *)

open Windtrap
open Aion

let u32 = 0x1p-24
let floats t = Nx.to_array (Nx.cast Nx.float64 t)

(* A narrow codec: three projected channels, widths 2, 4 and 8. *)
let narrow = lazy (Image.init ~key:(Nx.Rng.key 7) { projected = 3; width = 2 })
let codec () = Lazy.force narrow
let hsc = Image.[ Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y ]
let is_hsc b = Image.channel b < 5

(* Distinct bands, in any order: one, all nine, or any set. *)
let bands_gen =
  Gen.(
    let any =
      let* keep = array ~size:(constant 9) bool in
      let+ order = array ~size:(constant 9) (int_range 0 1000) in
      let bs = List.filteri (fun i _ -> keep.(i)) Image.bands in
      let bs = if bs = [] then [ Image.Des_g ] else bs in
      let ranked = List.map (fun b -> (order.(Image.channel b), b)) bs in
      List.map snd (List.sort compare ranked)
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
let same a b =
  let a = Nx.to_array a and b = Nx.to_array b in
  let found = ref [] in
  Array.iteri
    (fun i x ->
      if
        List.length !found < 5
        && Int64.bits_of_float x <> Int64.bits_of_float b.(i)
      then found := (i, x, b.(i)) :: !found)
    a;
  equal (list (triple int float_exact float_exact)) [] (List.rev !found)

(* Two images of bands [bs], normal draws at 0.8 of each band's limit, a few
   pixels set to the limit, its negation, a tiny value or zero. *)
let images bs h w seed =
  let c = List.length bs and shape = [| 2; List.length bs; h; w |] in
  let n = Nx.Rng.normal ~key:(Nx.Rng.key seed) Nx.float32 shape in
  let lim =
    Nx.create Nx.float32 [| 1; c; 1; 1 |]
      (Array.of_list (List.map Image.limit bs))
  in
  let x = Nx.mul n (Nx.mul_s lim 0.8) in
  let lim = Nx.broadcast_to shape lim in
  let special = Nx.Rng.uniform ~key:(Nx.Rng.key (seed + 1)) Nx.float32 shape in
  let between a b =
    Nx.logical_and (Nx.greater_equal_s special a) (Nx.less_s special b)
  in
  let x = Nx.where (Nx.less_s special 0.02) lim x in
  let x = Nx.where (between 0.02 0.04) (Nx.neg lim) x in
  let x = Nx.where (between 0.04 0.06) (Nx.mul_s x 1e-4) x in
  Nx.where (between 0.06 0.07) (Nx.zeros_like x) x

(* The preamble's preparation of a pixel [v] of band [b], exactly: HSC pixels
   are 10^((22.5 - 27) / 2.5) nanomaggies. *)
let prepared b v =
  let m = Image.limit b in
  let nmgy = if is_hsc b then 10. ** -1.8 else 1. in
  let x = Float.max (-.m) (Float.min m v) *. nmgy in
  10. *. 0.01 *. Float.asinh (x /. 0.01)

(* [f b v i] for each pixel [v] of band [b] that the crop keeps, [i] its index
   in the cropped batch of two images: the first five [Some]. *)
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

(* Preparation *)

let prepare_formula =
  prop "prepare crops, clamps, converts to nanomaggies and compresses"
    images_gen (fun (bs, h, w, seed) ->
      let x = images bs h w seed in
      let p = Image.prepare bs x in
      equal (array int) [| 2; List.length bs; 96; 96 |] (Nx.shape p);
      cover "an odd crop offset"
        ((h - 96) / 2 mod 2 = 1 || (w - 96) / 2 mod 2 = 1);
      cover "an HSC band" (List.exists is_hsc bs);
      cover "a Legacy Survey band" (List.exists (fun b -> not (is_hsc b)) bs);
      let ps = Nx.to_array p in
      (* Each step rounds once, the constants 0.01 and 10^-1.8 once each, and
         asinh within 2 ulps of its rounded input: within 16 float32 roundings
         of the exact value, relatively. *)
      let clamped = ref false in
      let refuted =
        each_pixel (Nx.to_array x) bs h w (fun b v i ->
            if Float.abs v > Image.limit b then clamped := true;
            let want = prepared b v in
            if Float.abs (ps.(i) -. want) > 16. *. u32 *. Float.abs want then
              Some (want, ps.(i))
            else None)
      in
      cover "a clamped pixel" !clamped;
      equal (list (pair float_exact float_exact)) [] refuted)

let per_band =
  prop "a list of bands prepares as each band alone" images_gen
    (fun (bs, h, w, seed) ->
      let x = images bs h w seed in
      cover "HSC and Legacy Survey bands together"
        (List.exists is_hsc bs && List.exists (fun b -> not (is_hsc b)) bs);
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
         match List.find_index (( = ) b) bs with
         | Some i -> Nx.slice [ A; R (i, i + 1) ] p
         | None -> zero)
       Image.bands)

let restores =
  prop "restore inverts the conversion and compression of prepare" images_gen
    (fun (bs, h, w, seed) ->
      let x = images bs h w seed in
      let r = Nx.to_array (Image.restore bs (nine bs (Image.prepare bs x))) in
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

(* AION-1 trained on CUDA, where PyTorch divides by a Python number as a product
   with the float32 reciprocal: [x / 0.01] is [x * 100] and [y / 10] is [y *
   0.1f]. The values here are those where the product and the division differ in
   float32. *)
let reciprocals =
  test "divisions by constants are products with float32 reciprocals" (fun () ->
      let grid = Nx.mul_s (Nx.arange_f Nx.float32 1. 4097. 1.) 1.37e-3 in
      let pick f g =
        let a = Nx.to_array (f grid) and b = Nx.to_array (g grid) in
        let found = ref [] in
        Array.iteri
          (fun i x ->
            if a.(i) <> b.(i) && List.length !found < 8 then
              found := x :: !found)
          (Nx.to_array grid);
        let x =
          Nx.create Nx.float32
            [| 1; 1; 1; List.length !found |]
            (Array.of_list !found)
        in
        Nx.broadcast_to [| 1; 1; 96; 96 |]
          (Nx.pad [| (0, 0); (0, 0); (0, 0); (0, 96 - Nx.dim 3 x) |] 0. x)
      in
      let x = pick (fun x -> Nx.div_s x 0.01) (fun x -> Nx.mul_s x 100.) in
      same
        (Nx.mul_s (Nx.mul_s (Nx.asinh (Nx.mul_s x 100.)) 0.01) 10.)
        (Image.prepare [ Image.Des_g ] x);
      let tenth = Bounds.f32 0.1 in
      let y = pick (fun y -> Nx.div_s y 10.) (fun y -> Nx.mul_s y tenth) in
      same
        (Nx.mul_s (Nx.sinh (Nx.mul_s (Nx.mul_s y tenth) 100.)) 0.01)
        (Image.restore [ Image.Des_g ]
           (Nx.concatenate ~axis:1 (List.init 9 (fun _ -> y)))))

let bands =
  test "bands, channels, names, limits and units" (fun () ->
      equal (list int) (List.init 9 Fun.id) (List.map Image.channel Image.bands);
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
          let want = if is_hsc b then Ymir.Units.zero_point "27" else nmgy in
          equal string (Ymir.Unit.to_string want)
            (Ymir.Unit.to_string (Image.unit b)))
        Image.bands)

(* Upstream's network. upstream/image_codec.safetensors is a narrow codec of
   AION-1's layout (widths 2, 4 and 8, three projected channels) with random
   weights, designed so that each constant matters: the squeeze-excite gates
   open, the final layer norm's lane variance near its eps, two bands of nine.
   upstream/image_golden.safetensors holds what upstream's code computes with it
   in float64 on a Legacy Survey and an HSC image of two bands, and the spread
   of its rounding ensemble at float32's scale and at float64's (Spread). At
   both dtypes every latent and decoded value lies within [τ ŝ] of the float64
   reference, and every digit of a token is the reference's but where the
   reference's bounded value lies within [τ ŝ] of a rounding boundary. *)

let upstream_archive = lazy (Fixture.load "image_codec")
let upstream_codec = lazy (Image.of_archive (Lazy.force upstream_archive))

let codec64 =
  lazy (Nx.Ptree.cast (module Image) Nx.float64 (Lazy.force upstream_codec))

let golden = Fixture.load "image_golden"

(* An entry of the upstream codec's file, in float64. *)
let entry name =
  let a = Lazy.force upstream_archive in
  let shape =
    match Nx_io.Archive.find name a with
    | Some (Nx.P x) -> Nx.shape x
    | None -> [||]
  in
  Nx_io.Archive.float ~shape Nx.float64 name a

(* Each pixel's nine channels by subsample_in's weights, plus its bias, times
   √(9 / c): nine products and a bias, each rounding once, then the scale, are
   within 16 roundings of the magnitude of the terms. *)
let embeds =
  prop "embed projects the nine channels, scaled by √(9 / c)" images_gen
    (fun (bs, h, w, seed) ->
      let c = List.length bs in
      cover "one band" (c = 1);
      cover "nine bands" (c = 9);
      let p = Image.prepare bs (images bs h w seed) in
      let e = Image.embed (Lazy.force upstream_codec) bs p in
      let bias = entry "subsample_in.bias" in
      equal (array int) [| 2; 96; 96; Nx.dim 0 bias |] (Nx.shape e);
      let x9 = Nx.cast Nx.float64 (Nx.moveaxis 1 3 (nine bs p)) in
      let wt = Nx.transpose (entry "subsample_in.weight") in
      let s = Float.sqrt (9. /. Float.of_int c) in
      let want = floats (Nx.mul_s (Nx.add (Nx.matmul x9 wt) bias) s) in
      let mag =
        floats
          (Nx.mul_s
             (Nx.add (Nx.matmul (Nx.abs x9) (Nx.abs wt)) (Nx.abs bias))
             s)
      in
      let refuted = ref [] in
      Array.iteri
        (fun i g ->
          if
            Float.abs (g -. want.(i)) > 16. *. u32 *. mag.(i)
            && List.length !refuted < 5
          then refuted := (want.(i), g) :: !refuted)
        (floats e);
      equal (list (pair float_exact float_exact)) [] !refuted)

let x0 () = images hsc 100 120 3

let shapes =
  test "latent, encode and decode shapes; encode quantizes the latent"
    (fun () ->
      let z = Image.latent (codec ()) hsc (x0 ()) in
      equal (array int) [| 2; 576; 5 |] (Nx.shape z);
      let k = Image.encode (codec ()) hsc (x0 ()) in
      equal (array int) [| 2; 576 |] (Nx.shape k);
      equal (array int64)
        (Nx.to_array (Fsq.encode (Fsq.v [ 7; 5; 5; 5; 5 ]) z))
        (Nx.to_array k);
      let d = Image.decode (codec ()) Image.[ Des_r; Hsc_y ] k in
      equal (array int) [| 2; 2; 96; 96 |] (Nx.shape d))

let decodes_per_band =
  test "each decoded band is in its own unit" (fun () ->
      let k = Image.encode (codec ()) hsc (x0 ()) in
      let all = Image.decode (codec ()) Image.bands k in
      let alone =
        List.map (fun b -> Image.decode (codec ()) [ b ] k) Image.bands
      in
      same (Nx.concatenate ~axis:1 alone) all)

let digits k = Decisions.digits (Array.map Int64.to_int (Nx.to_array k))

(* The reference's [1; 5; 576] as the codec's [1; 576; 5]. *)
let cells a = floats (Nx.moveaxis 1 2 (Nx.create Nx.float64 [| 1; 5; 576 |] a))

(* The codec in float32 against the ensemble at float32's scale, and cast to
   float64 against the ensemble at float64's. *)
let sets = [ ("ls", Image.[ Des_g; Des_z ]); ("hsc", Image.[ Hsc_i; Hsc_y ]) ]

let rows =
  List.concat_map
    (fun p -> List.map (fun (name, bs) -> (p, name, bs)) sets)
    [ `F32; `F64 ]

let scale = function `F32 -> "order" | `F64 -> "order64"
let row_id (p, name, _) = scale p ^ " " ^ name

let upstream =
  let input name = Fixture.get golden Nx.float32 ("input." ^ name) in
  let ref name part =
    floats (Fixture.get golden Nx.float64 ("f64." ^ name ^ "." ^ part))
  in
  let spread p name part =
    floats
      (Fixture.get golden Nx.float32 ("s." ^ scale p ^ "." ^ name ^ "." ^ part))
  in
  let at p f32 f64 x =
    match p with
    | `F32 -> f32 (Lazy.force upstream_codec) x
    | `F64 -> f64 (Lazy.force codec64) (Nx.cast Nx.float64 x)
  in
  group "upstream's network"
    [
      cases "the latent is upstream's" ~id:row_id rows (fun (p, name, bs) ->
          let latent t x = floats (Image.latent t bs x) in
          Spread.holds "latent"
            (Spread.values ~n:(Fixture.ensemble golden)
               ~reference:(cells (ref name "pre"))
               ~spread:(cells (spread p name "pre"))
               (at p latent latent (input name))));
      cases "encode gives upstream's tokens" ~id:row_id rows
        (fun (p, name, bs) ->
          let encode t x = digits (Image.encode t bs x) in
          Spread.holds "token digits"
            (Spread.levels ~n:(Fixture.ensemble golden)
               ~edges:Decisions.digit_edges ~reference:(ref name "bounded")
               ~spread:(spread p name "bounded")
               (at p encode encode (input name))));
      cases "decode gives upstream's images" ~id:row_id rows
        (fun (p, name, bs) ->
          let k = Fixture.get golden Nx.int64 ("f64." ^ name ^ ".tokens") in
          let one = [ List.hd bs ] in
          let decoded =
            match p with
            | `F32 -> floats (Image.decode (Lazy.force upstream_codec) one k)
            | `F64 -> floats (Image.decode (Lazy.force codec64) one k)
          in
          Spread.holds "decoded"
            (Spread.values ~n:(Fixture.ensemble golden)
               ~reference:(ref name "decoded") ~spread:(spread p name "decoded")
               decoded));
    ]

(* Compiled codecs: the codec an argument of Rune.jit, a program per list of
   bands, dtype, shape and placement, on the host or on a set over a software
   device, which has no eager kernels and compiles for this machine's CPU. Eager
   and compiled are computations of one exact value at one dtype, each within τ
   ŝ of it, so they differ by at most 2 τ ŝ: each latent and decoded value, and
   each digit of a token but where eager's bounded value lies within 2 τ ŝ of a
   rounding boundary. The inputs are the fixture's two images, so the law runs
   on each at each dtype, two of the four on each set. *)

let signature () =
  Nx.Ptree.(
    instantiate (module Image)
    @-> list (v Image.walk_band)
    @-> tensor @-> returns tensor)

let compiled_fns () =
  ( Rune.jit (signature ()) Image.latent,
    Rune.jit (signature ()) Image.encode,
    Rune.jit (signature ()) Image.decode )

let c32 = compiled_fns ()
let c64 = compiled_fns ()
let _, encode_c, decode_c = c32
let embed_c = Rune.jit (signature ()) Image.embed

module Soft = (val Nx.devices [ Result.get_ok (Rig.software "aion-image") ])

let s32 = compiled_fns ()
let s64 = compiled_fns ()

(* Compiled functions of a codec on the host, from ones on Soft. *)
let on_soft (latent, encode, decode) =
  let host x = Nx.place Nx.Host.on x in
  let call f t bs x =
    host (f (Nx.Ptree.place (module Image) Soft.on t) bs (Nx.place Soft.on x))
  in
  (call latent, call encode, call decode)

let quantizer = Fsq.v [ 7; 5; 5; 5; 5 ]

let check (latent_c, encode_c, decode_c) ~spread t bs x k =
  let n = Fixture.ensemble golden in
  let z = Image.latent t bs x in
  Spread.holds "latent"
    (Spread.values ~n ~reference:(floats z)
       ~spread:(cells (spread "pre"))
       (floats (latent_c t bs x)));
  Spread.holds "token digits"
    (Spread.levels ~n ~edges:Decisions.digit_edges
       ~reference:(floats (Fsq.bound quantizer z))
       ~spread:(spread "bounded")
       (digits (encode_c t bs x)));
  let one = [ List.hd bs ] in
  Spread.holds "decoded"
    (Spread.values ~n
       ~reference:(floats (Image.decode t one k))
       ~spread:(spread "decoded")
       (floats (decode_c t one k)))

let compiled =
  cases "compiled computes as eager within 2 τ ŝ, on either set"
    ~id:(fun ((p, name, _), soft) ->
      row_id (p, name, ()) ^ if soft then " on a set without kernels" else "")
    (List.mapi (fun i row -> (row, i = 1 || i = 2)) rows)
    (fun ((p, name, bs), soft) ->
      let spread part =
        let key = "s." ^ scale p ^ "." ^ name ^ "." ^ part in
        Array.map
          (fun s -> 2. *. s)
          (floats (Fixture.get golden Nx.float32 key))
      in
      let x = Fixture.get golden Nx.float32 ("input." ^ name) in
      let k = Fixture.get golden Nx.int64 ("f64." ^ name ^ ".tokens") in
      let t32 = Lazy.force upstream_codec and t64 = Lazy.force codec64 in
      let x64 = Nx.cast Nx.float64 x in
      match (p, soft) with
      | `F32, false -> check c32 ~spread t32 bs x k
      | `F32, true -> check (on_soft s32) ~spread t32 bs x k
      | `F64, false -> check c64 ~spread t64 bs x64 k
      | `F64, true -> check (on_soft s64) ~spread t64 bs x64 k)

(* Refusals *)

(* A call refused eagerly, and compiled on the host at the call: a shape's
   refusal at the trace, a value's by Nx.check. A prepare's refusal is encode's
   compiled. *)
type call =
  | Prepare of Image.band list * Nx.host Nx.float32_t
  | Embed of Image.band list * Nx.host Nx.float32_t
  | Decode of Image.band list * Nx.host Nx.int64_t

let refusals =
  let x s = Nx.zeros Nx.float32 s in
  let k = Nx.zeros Nx.int64 [| 1; 576 |] in
  let code =
    Nx.create Nx.int64 [| 1; 576 |]
      (Array.init 576 (fun i -> if i = 7 then 4375L else 0L))
  in
  cases "prepare, embed and decode refuse, eagerly and compiled"
    ~id:(fun (name, _, _) -> name)
    [
      ("no band", "no band", Prepare ([], x [| 1; 0; 96; 96 |]));
      ( "a band twice",
        "band DES-G twice",
        Prepare (Image.[ Des_g; Des_g ], x [| 1; 2; 96; 96 |]) );
      ( "fewer bands than the array's",
        "2 bands in the array, 1 listed",
        Prepare (Image.[ Des_g ], x [| 1; 2; 96; 96 |]) );
      ( "an image under 96 rows",
        "at least 96 × 96",
        Prepare (Image.[ Des_g ], x [| 1; 1; 95; 96 |]) );
      ( "an image under 96 columns",
        "at least 96 × 96",
        Prepare (Image.[ Des_g ], x [| 1; 1; 96; 95 |]) );
      ( "a batch of rank 3",
        "[n; c; h; w] expected",
        Prepare (Image.[ Des_g ], x [| 1; 96; 96 |]) );
      ( "embed: fewer bands than the array's",
        "Aion.Image.embed: 3 bands in the array, 1 listed",
        Embed (Image.[ Hsc_g ], x [| 1; 3; 96; 96 |]) );
      ( "tokens not [n; 576]",
        "[n; 576] expected",
        Decode (Image.[ Des_g ], Nx.zeros Nx.int64 [| 1; 575 |]) );
      ( "decode with a band twice",
        "band DES-G twice",
        Decode (Image.[ Des_g; Des_g ], k) );
      ( "a code outside [0, 4375)",
        "code 4375 at [0; 7]",
        Decode (Image.[ Des_g ], code) );
    ]
    (fun (_, sub, call) ->
      let t = codec () in
      let refuses f = raises_match (Exn.invalid_arg ~sub) f in
      match call with
      | Prepare (bs, x) ->
          refuses (fun () -> Image.prepare bs x);
          refuses (fun () -> encode_c t bs x)
      | Embed (bs, x) ->
          refuses (fun () -> Image.embed t bs x);
          refuses (fun () -> embed_c t bs x)
      | Decode (bs, k) ->
          refuses (fun () -> Image.decode t bs k);
          refuses (fun () -> decode_c t bs k))

let unread =
  test "of_archive refuses an entry it does not read" (fun () ->
      let extra = "model.encoder_layers.8.weight" in
      let a =
        Nx_io.Archive.union
          [
            Lazy.force upstream_archive;
            Nx_io.Archive.of_list
              [ (extra, Nx.P (Nx.zeros Nx.float32 [| 4 |])) ];
          ]
      in
      raises_match
        (Exn.failure ~sub:(extra ^ ": an entry in the archive, never read"))
        (fun () -> Image.of_archive a))

let () =
  exit
    (run "aion.image"
       [
         group "prepare"
           [ prepare_formula; per_band; restores; reciprocals; bands ];
         group "embed" [ embeds ];
         group "codec" [ shapes; decodes_per_band ];
         upstream;
         group "compiled" [ compiled ];
         group "refusals" [ refusals; unread ];
       ])
