(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Bands *)

type band =
  | Hsc_g
  | Hsc_r
  | Hsc_i
  | Hsc_z
  | Hsc_y
  | Des_g
  | Des_r
  | Des_i
  | Des_z

let bands = [ Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y; Des_g; Des_r; Des_i; Des_z ]

(* Upstream's BAND_TO_INDEX (aion/codecs/preprocessing/band_to_index.py). *)
let channel = function
  | Hsc_g -> 0
  | Hsc_r -> 1
  | Hsc_i -> 2
  | Hsc_z -> 3
  | Hsc_y -> 4
  | Des_g -> 5
  | Des_r -> 6
  | Des_i -> 7
  | Des_z -> 8

let channels = List.length bands

let name = function
  | Hsc_g -> "HSC-G"
  | Hsc_r -> "HSC-R"
  | Hsc_i -> "HSC-I"
  | Hsc_z -> "HSC-Z"
  | Hsc_y -> "HSC-Y"
  | Des_g -> "DES-G"
  | Des_r -> "DES-R"
  | Des_i -> "DES-I"
  | Des_z -> "DES-Z"

(* The codec works in nanomaggies, the Legacy Survey's unit. HSC's coadds are on
   AB zero point 27, which upstream rescales by 10^((27 - 22.5) / 2.5)
   (aion/codecs/preprocessing/image.py:93-107). *)
let nanomaggy = Ymir.Unit.nano Ymir.Units.maggy
let hsc_unit = Ymir.Units.zero_point "27"

let unit = function
  | Hsc_g | Hsc_r | Hsc_i | Hsc_z | Hsc_y -> hsc_unit
  | Des_g | Des_r | Des_i | Des_z -> nanomaggy

(* Upstream's BAND_CENTER_MAX. *)
let limit = function
  | Hsc_g -> 80.
  | Hsc_r -> 110.
  | Hsc_i -> 200.
  | Hsc_z -> 330.
  | Hsc_y -> 500.
  | Des_g -> 6.
  | Des_r -> 15.
  | Des_i -> 20.
  | Des_z -> 25.

(* The crop, the token grid, the range compression's scale and gain, and the
   encoder's final layer norm's epsilon (codecs/image/config.json and
   aion/codecs/modules/magvit.py). *)
let side = 96
let grid = 24
let range = 0.01
let gain = 10.
let eps = 1e-5

(* Layers *)

type ('b, 'd) conv = ((float, 'b, 'd) Nx.t, 'd) Kaun.Conv.t

(* A residual unit, [x + excite (elu (c1 (elu (c3 x))))], whose global-context
   squeeze-excite scores the cells with [to_k] and gates the channels with [gate
   (leaky_relu (hidden ctx))]. *)
type ('b, 'd) excite = {
  to_k : ('b, 'd) conv;
  hidden : ('b, 'd) conv;
  gate : ('b, 'd) conv;
}

type ('b, 'd) residual = {
  c3 : ('b, 'd) conv;
  c1 : ('b, 'd) conv;
  excite : ('b, 'd) excite;
}

type ('b, 'd) encoder = {
  conv_in : ('b, 'd) conv;
  unit0 : ('b, 'd) residual;
  down1 : ('b, 'd) conv;
  units1 : ('b, 'd) residual list;
  down2 : ('b, 'd) conv;
  units2 : ('b, 'd) residual list;
  conv7 : ('b, 'd) conv;
  units3 : ('b, 'd) residual list;
  norm : ((float, 'b, 'd) Nx.t, 'd) Kaun.Layer_norm.t;
}

(* The decoder, in the order it runs: upstream's decoder_layers 0 to 6, then
   conv_out. *)
type ('b, 'd) decoder = {
  units3' : ('b, 'd) residual list;
  conv3 : ('b, 'd) conv;
  units2' : ('b, 'd) residual list;
  up2 : ('b, 'd) conv;
  units1' : ('b, 'd) residual list;
  up1 : ('b, 'd) conv;
  unit0' : ('b, 'd) residual;
  conv_out : ('b, 'd) conv;
}

(* A compiled codec's programs ({!jit}): encode and decode without their
   checks, for a list of bands. *)
type ('b, 'd) programs = {
  p_encode : band list -> (float, 'b, 'd) Nx.t -> 'd Nx.int64_t;
  p_decode : band list -> 'd Nx.int64_t -> (float, 'b, 'd) Nx.t;
}

(* [proj_in] and [proj_out] are the subsampled linear maps between the nine
   channels and the network's, as 1 × 1 convolutions. *)
type ('b, 'd) t = {
  proj_in : ('b, 'd) conv;
  encoder : ('b, 'd) encoder;
  pre : ('b, 'd) conv;
  quantizer : Fsq.t;
  post : ('b, 'd) conv;
  decoder : ('b, 'd) decoder;
  proj_out : ('b, 'd) conv;
  programs : ('b, 'd) programs option;
}

(* Loading *)

let failure fmt = Format.kasprintf failwith fmt

let float ?shape a name = Nx_io.Archive.float ?shape Nx.float32 name a

let outputs name o want =
  if o <> want then
    failure "Aion.Image.of_archive: %s.weight: %d outputs, %d expected" name o
      want

(* The convolution [name] reading [i] channels with a [k] × [k] kernel, its
   PyTorch weight [[o; i; k; k]] moved channels last, and its [o]. *)
let conv a name ~i ~k =
  let w = float a (name ^ ".weight") in
  match Nx.shape w with
  | [| o; i'; kh; kw |] when i' = i && kh = k && kw = k ->
      let b = float ~shape:[| o |] a (name ^ ".bias") in
      ({ Kaun.Conv.w = Nx.copy (Nx.moveaxis 1 (-1) w); b = Some b }, o)
  | s ->
      failure
        "Aion.Image.of_archive: %s.weight: shape %a, [o; %d; %d; %d] expected"
        name Nx.pp_shape s i k k

(* A convolution of [o] outputs. *)
let conv_to a name ~i ~k ~o =
  let c, o' = conv a name ~i ~k in
  outputs name o' o;
  c

let residual_of a name d =
  let net = name ^ ".net.4." in
  let hidden, h = conv a (net ^ "net.0") ~i:d ~k:1 in
  let excite =
    {
      to_k = conv_to a (net ^ "to_k") ~i:d ~k:1 ~o:1;
      hidden;
      gate = conv_to a (net ^ "net.2") ~i:h ~k:1 ~o:d;
    }
  in
  {
    c3 = conv_to a (name ^ ".net.0.conv") ~i:d ~k:3 ~o:d;
    c1 = conv_to a (name ^ ".net.2") ~i:d ~k:1 ~o:d;
    excite;
  }

(* Each run of residual units holds upstream's num_consecutive, 4. *)
let residuals_of a name d =
  List.init 4 (fun i -> residual_of a (Printf.sprintf "%s.%d" name i) d)

(* A subsampled linear map's weight [[o; i]] and bias [[o]], as a 1 × 1
   convolution. *)
let linear a name ~i =
  let w = float a (name ^ ".weight") in
  match Nx.shape w with
  | [| o; i' |] when i' = i ->
      let b = float ~shape:[| o |] a (name ^ ".bias") in
      ({ Kaun.Conv.w = Nx.reshape [| o; 1; 1; i |] w; b = Some b }, o)
  | s ->
      failure "Aion.Image.of_archive: %s.weight: shape %a, [o; %d] expected"
        name
        Nx.pp_shape s i

let levels_of a =
  let l = Nx_io.Archive.tensor Nx.int32 "_quantizer.levels" a in
  if Nx.ndim l <> 1 then
    failure "Aion.Image.of_archive: _quantizer.levels: shape %a, [l] expected"
      Nx.pp_shape (Nx.shape l);
  List.map Int32.to_int (Array.to_list (Nx.to_array l))

let read a =
  let enc = "model.encoder_layers." and dec = "model.decoder_layers." in
  let proj_in, p = linear a "subsample_in" ~i:channels in
  let conv_in, d = conv a "model.conv_in.conv" ~i:p ~k:7 in
  let down1, d1 = conv a (enc ^ "1.conv") ~i:d ~k:3 in
  let down2, d2 = conv a (enc ^ "3.conv") ~i:d1 ~k:3 in
  let norm =
    let v suffix = float ~shape:[| d2 |] a (enc ^ "7.1." ^ suffix) in
    { Kaun.Layer_norm.gamma = v "weight"; beta = Some (v "bias") }
  in
  let encoder =
    {
      conv_in;
      unit0 = residual_of a (enc ^ "0") d;
      down1;
      units1 = residuals_of a (enc ^ "2") d1;
      down2;
      units2 = residuals_of a (enc ^ "4") d2;
      conv7 = conv_to a (enc ^ "5.conv") ~i:d2 ~k:7 ~o:d2;
      units3 = residuals_of a (enc ^ "6") d2;
      norm;
    }
  in
  let quantizer = Fsq.v (levels_of a) in
  let e = List.length (Fsq.levels quantizer) in
  let decoder =
    {
      units3' = residuals_of a (dec ^ "0") d2;
      conv3 = conv_to a (dec ^ "1.conv") ~i:d2 ~k:3 ~o:d2;
      units2' = residuals_of a (dec ^ "2") d2;
      up2 = conv_to a (dec ^ "3.net.0") ~i:d2 ~k:1 ~o:(4 * d1);
      units1' = residuals_of a (dec ^ "4") d1;
      up1 = conv_to a (dec ^ "5.net.0") ~i:d1 ~k:1 ~o:(4 * d);
      unit0' = residual_of a (dec ^ "6") d;
      conv_out = conv_to a "model.conv_out.conv" ~i:d ~k:3 ~o:p;
    }
  in
  let proj_out, nine = linear a "subsample_out" ~i:p in
  outputs "subsample_out" nine channels;
  {
    proj_in;
    encoder;
    pre = conv_to a "pre_quant_proj" ~i:d2 ~k:1 ~o:e;
    quantizer;
    post = conv_to a "post_quant_proj" ~i:e ~k:1 ~o:d2;
    decoder;
    proj_out;
    programs = None;
  }

let of_archive a = Nx_io.Archive.import read a

(* Placing *)

(* [t] with [f] applied to each of its tensors. *)
let map f t =
  let c (c : _ conv) : _ conv = { w = f c.w; b = Option.map f c.b } in
  let r r =
    let e = r.excite in
    {
      c3 = c r.c3;
      c1 = c r.c1;
      excite = { to_k = c e.to_k; hidden = c e.hidden; gate = c e.gate };
    }
  in
  let rs = List.map r in
  let e = t.encoder and d = t.decoder in
  let encoder =
    {
      conv_in = c e.conv_in;
      unit0 = r e.unit0;
      down1 = c e.down1;
      units1 = rs e.units1;
      down2 = c e.down2;
      units2 = rs e.units2;
      conv7 = c e.conv7;
      units3 = rs e.units3;
      norm = { gamma = f e.norm.gamma; beta = Option.map f e.norm.beta };
    }
  in
  let decoder =
    {
      units3' = rs d.units3';
      conv3 = c d.conv3;
      units2' = rs d.units2';
      up2 = c d.up2;
      units1' = rs d.units1';
      up1 = c d.up1;
      unit0' = r d.unit0';
      conv_out = c d.conv_out;
    }
  in
  {
    proj_in = c t.proj_in;
    encoder;
    pre = c t.pre;
    quantizer = t.quantizer;
    post = c t.post;
    decoder;
    proj_out = c t.proj_out;
    programs = None;
  }

let dtype t = Nx.dtype t.proj_in.w

(* Checks *)

let invalid fmt = Format.kasprintf invalid_arg fmt

let check_bands fn bs =
  if bs = [] then invalid "Aion.%s: no band" fn;
  let rec twice = function
    | [] -> ()
    | b :: rest ->
        if List.mem b rest then invalid "Aion.%s: band %s twice" fn (name b);
        twice rest
  in
  twice bs

let check_images fn bs x =
  check_bands fn bs;
  match Nx.shape x with
  | [| _; c; h; w |] ->
      if c <> List.length bs then
        invalid "Aion.%s: %d bands in the array, %d listed" fn c
          (List.length bs);
      if h < side || w < side then
        invalid "Aion.%s: images of %d × %d, at least %d × %d expected" fn h w
          side side
  | s -> invalid "Aion.%s: shape %a, [n; c; h; w] expected" fn Nx.pp_shape s

(* Preparing *)

let to_nanomaggies b x =
  Ymir.Quantity.value nanomaggy (Ymir.Quantity.v (unit b) x)

let of_nanomaggies b x =
  Ymir.Quantity.value (unit b) (Ymir.Quantity.v nanomaggy x)

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Whether [dt] is float64, where the codec computes as upstream's float64
   reference does, rather than as AION-1 trained in float32. *)
let wide (type b) (dt : (float, b) Nx.dtype) =
  match dt with Nx.Dtype.Float64 -> true | _ -> false

(* A division by a constant as AION-1 trained it: on CUDA, PyTorch divides a
   tensor by a Python number as a product with the float32 quotient of 1 by the
   number rounded to float32 (aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu,
   the same from torch 2.4, AION-1's floor, to 2.9; a double reciprocal only
   after the weights were published). For 0.01 that is 100 exactly, for 10 the
   float32 0.1. In float64, upstream's reference divides. *)
let div_const x c =
  if wide (Nx.dtype x) then Nx.div_s x c else Nx.mul_s x (f32 (1. /. f32 c))

(* Upstream's _range_compress and its inverse, in its order of operations. *)
let compress x = Nx.mul_s (Nx.mul_s (Nx.asinh (div_const x range)) range) gain
let expand y = Nx.mul_s (Nx.sinh (div_const (div_const y gain) range)) range
let band_at i x = Nx.slice [ A; R (i, i + 1) ] x

let prepare bs x =
  check_images "Image.prepare" bs x;
  let h = Nx.dim 2 x and w = Nx.dim 3 x in
  let r0 = (h - side) / 2 and c0 = (w - side) / 2 in
  let x = Nx.slice [ A; A; R (r0, r0 + side); R (c0, c0 + side) ] x in
  let each i b =
    let m = limit b in
    to_nanomaggies b (Nx.clamp ~min:(-.m) ~max:m (band_at i x))
  in
  compress (Nx.concatenate ~axis:1 (List.mapi each bs))

(* Encoding *)

let apply ?(stride = 1) ?(pad = 0) c x =
  Kaun.Conv.apply ~stride:[| stride; stride |]
    ~padding:[| (pad, pad); (pad, pad) |]
    c x

(* A softmax over the cells weighs their average, which gates each channel. *)
let excite e x =
  let n = Nx.dim 0 x and cells = Nx.dim 1 x * Nx.dim 2 x and c = Nx.dim 3 x in
  let k = Nx.softmax (Nx.reshape [| n; 1; cells |] (apply e.to_k x)) in
  let ctx = Nx.matmul k (Nx.reshape [| n; cells; c |] x) in
  let ctx = Nx.reshape [| n; 1; 1; c |] ctx in
  let hidden = Kaun.Fn.leaky_relu ~negative_slope:0.1 (apply e.hidden ctx) in
  Nx.mul (Nx.sigmoid (apply e.gate hidden)) x

let residual r x =
  let y = Kaun.Fn.elu (apply r.c1 (Kaun.Fn.elu (apply ~pad:1 r.c3 x))) in
  Nx.add (excite r.excite y) x

let residuals rs x = List.fold_left (fun x r -> residual r x) x rs

(* The prepared bands in their channels, zeros elsewhere, channels last. *)
let nine bs p =
  let p = Nx.moveaxis 1 3 p in
  let zero =
    Nx.zeros_at (Nx.beside p) (Nx.dtype p) [| Nx.dim 0 p; side; side; 1 |]
  in
  let at b =
    let rec find i = function
      | [] -> zero
      | b' :: rest ->
          if b' = b then Nx.slice [ A; A; A; R (i, i + 1) ] p
          else find (i + 1) rest
    in
    find 0 bs
  in
  Nx.concatenate ~axis:3 (List.map at bands)

let embed t bs p =
  check_bands "Image.embed" bs;
  (* PyTorch's [(9 / c) ** 0.5]: a quotient rounded, then its square root, as
     its pow takes at the exponent 0.5. *)
  let q = Float.of_int channels /. Float.of_int (List.length bs) in
  let scale =
    if wide (Nx.dtype p) then Float.sqrt q else f32 (Float.sqrt (f32 q))
  in
  Nx.mul_s (apply t.proj_in (nine bs p)) scale

let latent t bs x =
  let e = t.encoder in
  let h = embed t bs (prepare bs x) in
  let h = residual e.unit0 (apply ~pad:3 e.conv_in h) in
  let h = residuals e.units1 (apply ~stride:2 ~pad:1 e.down1 h) in
  let h = residuals e.units2 (apply ~stride:2 ~pad:1 e.down2 h) in
  let h = residuals e.units3 (apply ~pad:3 e.conv7 h) in
  let z = apply t.pre (Kaun.Layer_norm.apply ~eps e.norm h) in
  Nx.reshape [| Nx.dim 0 z; grid * grid; Nx.dim 3 z |] z

let encode_raw t bs x = Fsq.encode t.quantizer (latent t bs x)

(* Decoding *)

(* A 2 × upsampling: a 1 × 1 convolution to four times the channels, SiLU, then
   each cell's channels [(c, p1, p2)] spread over a 2 × 2 block. *)
let upsample c x =
  Nx.rearrange
    ~sizes:[ ("p1", 2); ("p2", 2) ]
    (Nx.Pattern.v "b h w (c p1 p2) -> b (h p1) (w p2) c")
    (Kaun.Fn.silu (apply c x))

let restore bs y =
  check_bands "Image.restore" bs;
  let y = expand y in
  Nx.concatenate ~axis:1
    (List.map (fun b -> of_nanomaggies b (band_at (channel b) y)) bs)

let check_decode bs k =
  check_bands "Image.decode" bs;
  match Nx.shape k with
  | [| _; c |] when c = grid * grid -> ()
  | s ->
      invalid "Aion.Image.decode: shape %a, [n; %d] expected" Nx.pp_shape s
        (grid * grid)

let decode_raw t bs k =
  let n = Nx.dim 0 k in
  let d = t.decoder in
  let z = Fsq.decode (dtype t) t.quantizer k in
  let h = apply t.post (Nx.reshape [| n; grid; grid; Nx.dim 2 z |] z) in
  let h = residuals d.units3' h in
  let h = residuals d.units2' (apply ~pad:1 d.conv3 h) in
  let h = residuals d.units1' (upsample d.up2 h) in
  let h = residual d.unit0' (upsample d.up1 h) in
  let y = apply t.proj_out (apply ~pad:1 d.conv_out h) in
  restore bs (Nx.moveaxis 3 1 y)

(* Entry points: the checks, then the computation. A compiled codec checks the
   codes it decodes on the host, so that a refusal comes before its program
   runs; its programs check them again, and the latents' NaN, which only the
   computation finds. *)

let encode t bs x =
  match t.programs with
  | None -> encode_raw t bs x
  | Some p ->
      check_images "Image.prepare" bs x;
      p.p_encode bs x

let decode t bs k =
  check_decode bs k;
  match t.programs with
  | None -> decode_raw t bs k
  | Some p ->
      (* Fsq.decode of a host copy refuses a code as the program would. *)
      ignore (Fsq.decode Nx.float32 t.quantizer (Nx.place Nx.Host.on k));
      p.p_decode bs k

(* Compiling *)

let bands_s () =
  Nx.Ptree.v (fun c bs ->
      let open Nx.Ptree.Walk in
      list (fun c b -> case c (name b); b) c bs)

let jit t =
  let t = { t with programs = None } in
  let open Nx.Ptree in
  let programs =
    {
      p_encode =
        Rune.jit (bands_s () @-> tensor @-> returns tensor) (encode_raw t);
      p_decode =
        Rune.jit (bands_s () @-> tensor @-> returns tensor) (decode_raw t);
    }
  in
  { t with programs = Some programs }

let compiled t = Option.is_some t.programs

let place p t =
  let t' = map (Nx.place p) t in
  if compiled t then jit t' else t'

let cast dt t =
  let t' = map (Nx.cast dt) t in
  if compiled t then jit t' else t'
