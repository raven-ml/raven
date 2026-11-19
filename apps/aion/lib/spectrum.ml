(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The constants of the codec's config.json on the Hub (polymathic-ai/aion-base,
   codecs/spectrum): clip_ivar, input_scaling, embedding_dim and
   codebook_size. *)
let ivar_max = 100.
let input_scaling = 0.2
let bits = 10
let token_count = 1 lsl bits

(* Upstream's constants: the normalisation's floor, the pad values of
   aion/codecs/preprocessing/spectrum.py, and the epsilons of its layer norms
   (aion/codecs/modules/utils.py, torch's default for the pre-quantizer one) and
   of its GRN. *)
let norm_floor = 0.1
let pad_wavelength = 99999.
let ln_eps = 1e-6
let pre_norm_eps = 1e-5
let grn_eps = 1e-6

(* The largest finite float32, torch.nan_to_num's replacement for +inf in
   float32. *)
let float32_max = Int32.float_of_bits 0x7f7fffffl

(* Spectra *)

type ('b, 'd) spectrum = {
  flux : (float, 'b, 'd) Nx.t;
  ivar : (float, 'b, 'd) Nx.t;
  mask : 'd Nx.bool_t;
  wavelength : (float, 'b, 'd) Nx.t;
}

(* Upstream's pad_length of DESISpectrum and SDSSSpectrum
   (aion/modalities.py). *)
let length : Modality.spectrum -> int = function Desi -> 7808 | Sdss -> 4800

let shape_of fn x =
  let s = Nx.shape x.flux in
  let same t = Nx.shape t = s in
  if
    Array.length s <> 2
    || not (same x.ivar && Nx.shape x.mask = s && same x.wavelength)
  then
    Refuse.invalid fn "fields not of one 2-d shape";
  (s.(0), s.(1))

let pad modality x =
  let n = length modality in
  let _, m = shape_of "Spectrum.encode" x in
  if m > n then
    Refuse.invalid "Spectrum.encode" "rows of %d pixels, at most %d" m n;
  let p = [| (0, 0); (0, n - m) |] in
  {
    flux = Nx.pad p 0. x.flux;
    ivar = Nx.pad p 0. x.ivar;
    mask = Nx.pad p true x.mask;
    wavelength = Nx.pad p pad_wavelength x.wavelength;
  }

(* Codecs *)

type ('b, 'd) block = {
  dw : ((float, 'b, 'd) Nx.t, 'd) Kaun.Conv.t;
  norm : ((float, 'b, 'd) Nx.t, 'd) Kaun.Layer_norm.t;
  up : ((float, 'b, 'd) Nx.t, 'd) Kaun.Linear.t;
  gamma : (float, 'b, 'd) Nx.t;
  beta : (float, 'b, 'd) Nx.t;
  down : ((float, 'b, 'd) Nx.t, 'd) Kaun.Linear.t;
}

(* A [Patch] is a convolution whose stride is its kernel's extent, as every
   down-sampling one here is, and an [Unpatch] its transpose. *)
type ('b, 'd) layer =
  | Patch of ((float, 'b, 'd) Nx.t, 'd) Kaun.Conv.t
  | Unpatch of ((float, 'b, 'd) Nx.t, 'd) Kaun.Conv_transpose.t
  | Norm of ((float, 'b, 'd) Nx.t, 'd) Kaun.Layer_norm.t
  | Block of ('b, 'd) block

(* A compiled codec's programs ({!jit}): encode and decode without their
   checks. *)
type ('b, 'd) programs = {
  p_encode :
    'd Nx.int64_t option ->
    Modality.spectrum ->
    ('b, 'd) spectrum ->
    'd Nx.int64_t;
  p_decode : (float, 'b, 'd) Nx.t option -> 'd Nx.int64_t -> ('b, 'd) spectrum;
}

(* [norm] is the normalisation token's codebook, as a one-channel scalar codec:
   upstream's ScalarLinearQuantizer clamps torch.bucketize over all 1024 values
   to 1023, which is Scalar's count over the first 1023 wherever the values are
   in order, as the stored grid is. *)
type ('b, 'd) t = {
  encoder : ('b, 'd) layer list;
  pre_norm : ((float, 'b, 'd) Nx.t, 'd) Kaun.Layer_norm.t;
  quant : ((float, 'b, 'd) Nx.t, 'd) Kaun.Linear.t;
  post_quant : ((float, 'b, 'd) Nx.t, 'd) Kaun.Linear.t;
  decoder : ('b, 'd) layer list;
  grid : (float, 'b, 'd) Nx.t;
  norm : 'd Scalar.t;
  programs : ('b, 'd) programs option;
}

let grid t = t.grid
let levels t = Nx.reshape [| -1 |] (Scalar.edges t.norm)

(* [t] with [f] applied to each weight, as [map_layer] to a layer's, and
   [norm] to its normalisation codebook. *)
let map_layer f = function
  | Patch c -> Patch { w = f c.w; b = Option.map f c.b }
  | Unpatch c -> Unpatch { w = f c.w; b = Option.map f c.b }
  | Norm n -> Norm { gamma = f n.gamma; beta = Option.map f n.beta }
  | Block b ->
      let linear (l : _ Kaun.Linear.t) : _ Kaun.Linear.t =
        { w = f l.w; b = Option.map f l.b }
      in
      Block
        {
          dw = { w = f b.dw.w; b = Option.map f b.dw.b };
          norm = { gamma = f b.norm.gamma; beta = Option.map f b.norm.beta };
          up = linear b.up;
          gamma = f b.gamma;
          beta = f b.beta;
          down = linear b.down;
        }

let map f norm t =
  let linear (l : _ Kaun.Linear.t) : _ Kaun.Linear.t =
    { w = f l.w; b = Option.map f l.b }
  in
  {
    encoder = List.map (map_layer f) t.encoder;
    pre_norm = { gamma = f t.pre_norm.gamma; beta = Option.map f t.pre_norm.beta };
    quant = linear t.quant;
    post_quant = linear t.post_quant;
    decoder = List.map (map_layer f) t.decoder;
    grid = f t.grid;
    norm = norm t.norm;
    programs = None;
  }


(* Reading the weights. Upstream's modules (aion/codecs/modules/convnext.py)
   store a convolution [out; in / groups; k], a transposed one [in; out; k] and
   a linear layer [out; in]; kaun takes the channels last and a linear layer
   [out; in] too. The encoder's stage [i] is [downsample_layers.i], a
   convolution then a layer norm for the stem and a layer norm then a
   convolution after it, then the blocks [stages.i.j]; a last layer norm closes
   it. The decoder's stages are [upsample_layers.i] then [stages.i], and nothing
   closes it. *)

(* An archive and the names read from it, so that a weight no layer reads, as in
   a file missing a block, is refused. *)
type 'd reader = {
  archive : 'd Nx_io.Archive.t;
  read : (string, unit) Hashtbl.t;
}

let find a name =
  Hashtbl.replace a.read name ();
  Nx_io.Archive.find name a.archive

let weight a name =
  match find a name with
  | None -> Refuse.failure "Spectrum" "no weight %s" name
  | Some p -> (
      try Nx.unpack Nx.float32 p
      with Invalid_argument _ ->
        Refuse.failure "Spectrum" "%s is not float32" name)

let has a name = Option.is_some (Nx_io.Archive.find name a.archive)

let conv a name : _ Kaun.Conv.t =
  {
    w = Nx.moveaxis 1 (-1) (weight a (name ^ ".weight"));
    b = Some (weight a (name ^ ".bias"));
  }

let conv_t a name : _ Kaun.Conv_transpose.t =
  {
    w = Nx.moveaxis 1 (-1) (weight a (name ^ ".weight"));
    b = Some (weight a (name ^ ".bias"));
  }

let linear a name : _ Kaun.Linear.t =
  let w = weight a (name ^ ".weight") in
  (* A 1x1 convolution, [out; in; 1], is a linear layer. *)
  let w = if Nx.ndim w = 3 then Nx.squeeze ~axes:[ 2 ] w else w in
  { w; b = Some (weight a (name ^ ".bias")) }

let layer_norm a name : _ Kaun.Layer_norm.t =
  { gamma = weight a (name ^ ".weight"); beta = Some (weight a (name ^ ".bias")) }

let block a name =
  let v key = Nx.reshape [| -1 |] (weight a (name ^ ".grn." ^ key)) in
  Block
    {
      dw = conv a (name ^ ".dwconv");
      norm = layer_norm a (name ^ ".norm");
      up = linear a (name ^ ".pwconv1");
      gamma = v "gamma";
      beta = v "beta";
      down = linear a (name ^ ".pwconv2");
    }

(* The indices [0, 1, …] for which [name i] is in [a]. *)
let count a name =
  let rec go i = if has a (name i) then go (i + 1) else i in
  go 0

let stage a net i =
  let s = Printf.sprintf "%s.stages.%d" net i in
  List.init
    (count a (fun j -> Printf.sprintf "%s.%d.dwconv.weight" s j))
    (fun j -> block a (Printf.sprintf "%s.%d" s j))

let network a net resize sample =
  let sampling i = Printf.sprintf "%s.%s_layers.%d" net sample i in
  let stages = count a (fun i -> sampling i ^ ".0.weight") in
  List.concat
    (List.init stages (fun i ->
         let l = sampling i in
         let first, second =
           if i = 0 then (resize (l ^ ".0"), Norm (layer_norm a (l ^ ".1")))
           else (Norm (layer_norm a (l ^ ".0")), resize (l ^ ".1"))
         in
         first :: second :: stage a net i))

let stride (c : _ Kaun.Conv.t) = Nx.dim 1 c.w

(* The grid wavelengths per latent vector: the product of the encoder's
   strides. *)
let reduction encoder =
  List.fold_left
    (fun r -> function Patch c -> r * stride c | _ -> r)
    1 encoder

(* Upstream's buffers that restate what the weights fix, checked where present:
   the LFQ's bit weights (aion/codecs/quantizers/ lookup_free_quantization.py,
   [mask]), and the grid's length and first wavelength. The grid's resolution is
   read and left: the grid holds its wavelengths. *)
let buffers a grid =
  let ints name =
    Option.map
      (fun p -> Array.map Int64.to_int (Nx.to_array (Nx.unpack Nx.int64 p)))
      (find a name)
  in
  let floats name =
    Option.map (fun p -> Nx.to_array (Nx.unpack Nx.float32 p)) (find a name)
  in
  let expected = Array.init bits (fun c -> 1 lsl (bits - 1 - c)) in
  (match ints "_quantizer._quantizer.mask" with
  | Some m when m <> expected ->
      Refuse.failure "Spectrum" "LFQ bit weights other than 2^(9-c)"
  | _ -> ());
  (match ints "latent_grid.length" with
  | Some [| n |] when n <> Nx.dim 0 grid ->
      Refuse.failure "Spectrum" "a grid length of %d" n
  | _ -> ());
  (match floats "latent_grid.lambda_min" with
  | Some [| w |] when w <> Nx.item [ 0 ] grid ->
      Refuse.failure "Spectrum" "a grid starting at %g" (Nx.item [ 0 ] grid)
  | _ -> ());
  ignore (find a "latent_grid.resolution")

let of_archive archive =
  let a = { archive; read = Hashtbl.create 512 } in
  let encoder =
    network a "encoder" (fun n -> Patch (conv a n)) "downsample"
    @ [ Norm (layer_norm a "encoder.norm") ]
  in
  let decoder =
    network a "decoder" (fun n -> Unpatch (conv_t a n)) "upsample"
  in
  let grid = Nx.reshape [| -1 |] (weight a "latent_grid._wavelength") in
  let levels =
    Nx.reshape [| 1; -1 |] (weight a "normalization_quantizer.buckets")
  in
  let r = reduction encoder in
  let l = Nx.dim 0 grid in
  if l = 0 || l mod r <> 0 then
    Refuse.failure "Spectrum"
      "a latent grid of %d wavelengths, a positive multiple of %d expected" l r;
  let t =
    {
      encoder;
      pre_norm = layer_norm a "pre_quant_norm";
      quant = linear a "quant_conv";
      post_quant = linear a "post_quant_conv";
      decoder;
      grid;
      norm = Scalar.v Identity levels;
      programs = None;
    }
  in
  buffers a grid;
  (match
     List.filter
       (fun n -> not (Hashtbl.mem a.read n))
       (Nx_io.Archive.names archive)
   with
  | [] -> ()
  | n :: _ as unread ->
      Refuse.failure "Spectrum"
        "%d weights never read, as %s: a layer is missing or extra"
        (List.length unread) n);
  t

let load ?cache_dir ?offline () =
  of_archive (Hub.codec ?cache_dir ?offline "spectrum")

(* Preparation *)

(* torch.nan_to_num: NaN to 0 and the infinities to the extreme floats. *)
let nan_to_num (type b d) (x : (float, b, d) Nx.t) : (float, b, d) Nx.t =
  let big =
    match Nx.dtype x with
    | Nx.Dtype.Float64 -> Float.max_float
    | _ -> float32_max
  in
  Nx.where (Nx.isnan x) (Nx.zeros_like x) (Nx.clamp ~min:(-.big) ~max:big x)

(* The normalisation, before its token: log10 (1 + max mean 0.1), the mean
   dividing the unmasked fluxes' sum by one more than their count, as upstream
   does (aion/codecs/spectrum.py, _encode). *)
let normalisation x =
  let valid = Nx.logical_not x.mask in
  let flux = Nx.where valid (nan_to_num x.flux) (Nx.zeros_like x.flux) in
  let sum = Nx.sum ~axes:[ 1 ] flux in
  let count =
    Nx.add_s (Nx.sum ~axes:[ 1 ] (Nx.cast (Nx.dtype flux) valid)) 1.
  in
  Nx.log10 (Nx.add_s (Nx.maximum_s (Nx.div sum count) norm_floor) 1.)

let scale v = Nx.maximum_s (Nx.sub_s (Nx.rpow_s 10. v) 1.) norm_floor

(* The search sorted ~side:`Left of each row of [xs] for the keys [keys]: [xs]
   is [b; m] and [keys] 1-d, or [xs] is 1-d and [keys] [b; k]. *)
let positions xs keys =
  if Nx.ndim xs = 1 then Nx.searchsorted ~side:`Left xs keys
  else
    Nx.stack
      (List.init (Nx.dim 0 xs) (fun i ->
           Nx.searchsorted ~side:`Left (Nx.get [ i ] xs) keys))

(* Upstream's interp1d (aion/codecs/modules/spectrum.py): [y] sampled at [x],
   read at [xnew], linearly within the sample's interval [x(i), x(i+1)] that
   holds it, [y(i) + (xnew - x(i)) · (y(i) - y(i+1)) / (x(i) - x(i+1))], and 0
   outside [x(0), x(m-1)]. [x] and [y] are [b; m], or [x] is the 1-d grid;
   [xnew] is the grid, or [b; k]. A key equal to [x(i)] takes the interval
   ending there, the first for [x(0)]. An interval of zero width, between
   repeated wavelengths as pads are, is read only by a key equal to [x(0)] where
   [x(1) = x(0)]; its slope is 0 where upstream's is NaN. *)
let resample ~x ~y xnew =
  let m = Nx.dim (-1) x in
  let i =
    Nx.clamp ~min:0L
      ~max:(Int64.of_int (m - 2))
      (Nx.sub_s (positions x xnew) 1L)
  in
  let lo t = Nx.slice [ Nx.A; Nx.R (0, m - 1) ] t
  and hi t = Nx.slice [ Nx.A; Nx.R (1, m) ] t in
  let lo1 t = Nx.slice [ Nx.R (0, m - 1) ] t
  and hi1 t = Nx.slice [ Nx.R (1, m) ] t in
  let dx =
    if Nx.ndim x = 1 then Nx.sub (lo1 x) (hi1 x) else Nx.sub (lo x) (hi x)
  in
  let slope = Nx.div (Nx.sub (lo y) (hi y)) dx in
  let slope = Nx.where (Nx.equal_s dx 0.) (Nx.zeros_like slope) slope in
  let gather t =
    if Nx.ndim t = 1 then Nx.take i t
    else
      Nx.take_along_axis ~axis:1 i
        (Nx.broadcast_to [| Nx.dim 0 i; Nx.dim 1 t |] t)
  in
  let xi = gather x in
  let ynew = Nx.add (gather y) (Nx.mul (Nx.sub xnew xi) (gather slope)) in
  let first, last =
    if Nx.ndim x = 1 then
      (Nx.slice [ Nx.R (0, 1) ] x, Nx.slice [ Nx.R (m - 1, m) ] x)
    else (Nx.slice [ Nx.A; Nx.R (0, 1) ] x, Nx.slice [ Nx.A; Nx.R (m - 1, m) ] x)
  in
  let outside = Nx.logical_or (Nx.less xnew first) (Nx.greater xnew last) in
  Nx.where outside (Nx.zeros_like ynew) ynew

let sorted w =
  let m = Nx.dim 1 w in
  let lo = Nx.slice [ Nx.A; Nx.R (0, m - 1) ] w in
  let ok = Nx.less_equal lo (Nx.slice [ Nx.A; Nx.R (1, m) ] w) in
  Nx.check Nx.Ptree.tensor ok lo (fun i _ ->
      Refuse.exn "Spectrum.encode"
        "row %d's wavelengths decrease or are NaN between pixels %d and %d"
        i.(0) i.(1)
        (i.(1) + 1))

(* The checks of a spectrum's preparation: rows of two pixels or more, their
   wavelengths sorted. *)
let check_prepare x =
  let _, m = shape_of "Spectrum.encode" x in
  if m < 2 then
    Refuse.invalid "Spectrum.encode" "rows of %d pixel" m;
  sorted x.wavelength

let prepare_raw t x =
  let code = Scalar.encode t.norm (normalisation x) in
  let v = Nx.cast (Nx.dtype x.flux) (Scalar.decode_raw t.norm code) in
  let s = Nx.reshape [| -1; 1 |] (scale v) in
  let signal =
    Nx.mul_s (Nx.sub_s (Nx.div (nan_to_num x.flux) s) 1.) input_scaling
  in
  let ivar = Nx.clamp ~min:0. ~max:ivar_max (nan_to_num x.ivar) in
  let noise = Nx.mul_s (Nx.div (Nx.sqrt ivar) s) input_scaling in
  let on_grid y = resample ~x:x.wavelength ~y t.grid in
  (Nx.asinh (Nx.stack ~axis:(-1) [ on_grid signal; on_grid noise ]), code)

let prepare t x =
  check_prepare x;
  prepare_raw t x

(* Networks *)

(* Global response normalisation (Woo et al., ConvNeXt V2, 2023, eq. 1-3) over
   [b; n; c]: each channel's L2 norm over the [n] positions, divided by the
   channels' mean norm, as aion/codecs/modules/utils.py computes it. *)
let grn ~gamma ~beta x =
  let g = Nx.sqrt (Nx.sum ~axes:[ 1 ] ~keepdims:true (Nx.mul x x)) in
  let n = Nx.div g (Nx.add_s (Nx.mean ~axes:[ 2 ] ~keepdims:true g) grn_eps) in
  Nx.add (Nx.add (Nx.mul gamma (Nx.mul x n)) beta) x

(* A ConvNeXt V2 block: a depthwise convolution of extent 7, then per position a
   layer norm, a linear layer to four times the channels, GELU, GRN and a linear
   layer back, added to the block's input. *)
let block p x =
  let k = Nx.dim 1 p.dw.w in
  let y =
    Kaun.Conv.apply ~padding:[| (k / 2, k / 2) |] ~groups:(Nx.dim (-1) x) p.dw x
  in
  let y = Kaun.Layer_norm.apply ~eps:ln_eps p.norm y in
  let y = Kaun.Fn.gelu (Kaun.Linear.apply p.up y) in
  let y = grn ~gamma:p.gamma ~beta:p.beta y in
  Nx.add x (Kaun.Linear.apply p.down y)

let layer l x =
  match l with
  | Patch c -> Kaun.Conv.apply ~stride:[| stride c |] c x
  | Unpatch c -> Kaun.Conv_transpose.apply ~stride:[| Nx.dim 1 c.w |] c x
  | Norm n -> Kaun.Layer_norm.apply ~eps:ln_eps n x
  | Block p -> block p x

let latent_raw t x =
  let input, code = prepare_raw t x in
  let h = List.fold_left (fun h l -> layer l h) input t.encoder in
  let h = Kaun.Layer_norm.apply ~eps:pre_norm_eps t.pre_norm h in
  (Kaun.Linear.apply t.quant h, code)

let latent t x =
  check_prepare x;
  latent_raw t x

(* Tokens *)

(* Bit [bits - 1 - c] for value [c], so value 0 is the most significant, as
   upstream's LFQ masks with 2^(arange(bits - 1, -1, -1)). *)
let weights at =
  Nx.place at
    (Nx.create Nx.int64 [| bits |]
       (Array.init bits (fun c -> Int64.shift_left 1L (bits - 1 - c))))

(* A value of exactly 0 gives a clear bit, as upstream's [x > 0]. *)
let codes h =
  Nx.sum ~axes:[ -1 ]
    (Nx.mul (Nx.cast Nx.int64 (Nx.greater_s h 0.)) (weights (Nx.beside h)))

let values dt k =
  let k = Nx.unsqueeze ~axes:[ -1 ] k in
  let set = Nx.not_equal_s (Nx.bitwise_and k (weights (Nx.beside k))) 0L in
  let one = Nx.ones_at (Nx.beside k) dt [||] in
  Nx.where set one (Nx.neg one)

let encode_unpadded_raw t x =
  let h, code = latent_raw t x in
  Nx.concatenate ~axis:1 [ Nx.reshape [| -1; 1 |] code; codes h ]

let encode_unpadded t x =
  check_prepare x;
  encode_unpadded_raw t x

(* Row [i]'s pixels at and past [lengths.(i)] become pads, so the row reads as
   its first [lengths.(i)] pixels padded, and the pixels cut are never read. *)
let trim lengths x =
  let b, m = shape_of "Spectrum.encode" x in
  let past =
    Nx.greater_equal
      (Nx.reshape [| 1; m |] (Nx.arange_at (Nx.beside lengths) Nx.int64 0 m 1))
      (Nx.reshape [| b; 1 |] lengths)
  in
  let fill v y = Nx.where past (Nx.full_like y v) y in
  {
    flux = fill 0. x.flux;
    ivar = fill 0. x.ivar;
    mask = Nx.logical_or past x.mask;
    wavelength = fill pad_wavelength x.wavelength;
  }

let check_lengths x lengths =
  let b, m = shape_of "Spectrum.encode" x in
  if Nx.shape lengths <> [| b |] then
    Refuse.invalid "Spectrum.encode" "lengths not of shape [%d]" b;
  let inside =
    Nx.logical_and
      (Nx.greater_equal_s lengths 1L)
      (Nx.less_equal_s lengths (Int64.of_int m))
  in
  Nx.check Nx.Ptree.tensor inside lengths (fun i l ->
      Refuse.exn "Spectrum.encode" "length %Ld of row %d outside [1, %d]"
        (Nx.item [] l) i.(0) m)

(* The checks of [encode], by shape and by {!Nx.check} of the lengths and of
   the padded rows' wavelengths, as [prepare] reads them. *)
let check_encode ?lengths modality x =
  ignore (shape_of "Spectrum.encode" x);
  let w = x.wavelength in
  let mask = Nx.zeros_at (Nx.beside w) Nx.bool (Nx.shape w) in
  let x = { flux = w; ivar = w; mask; wavelength = w } in
  let x =
    match lengths with
    | None -> x
    | Some l ->
        check_lengths x l;
        trim l x
  in
  check_prepare (pad modality x)

let encode_raw ?lengths t modality x =
  let x = match lengths with None -> x | Some l -> trim l x in
  encode_unpadded_raw t (pad modality x)

(* Decoding *)

let check_decode ?wavelength t k =
  let n = Nx.dim 0 t.grid / reduction t.encoder in
  let s = Nx.shape k in
  if Array.length s <> 2 || s.(1) <> n + 1 then
    Refuse.invalid "Spectrum.decode" "tokens of shape %a, [b; %d] expected"
      Nx.pp_shape s (n + 1);
  let b = s.(0) in
  (match wavelength with
  | Some w when Nx.ndim w <> 2 || Nx.dim 0 w <> b ->
      Refuse.invalid "Spectrum.decode" "wavelengths not of %d rows" b
  | _ -> ());
  let inside =
    Nx.logical_and (Nx.greater_equal_s k 0L)
      (Nx.less_s k (Int64.of_int token_count))
  in
  Nx.check Nx.Ptree.tensor inside k (fun i k ->
      Refuse.exn "Spectrum.decode" "token %Ld at [%d; %d] outside [0, %d)"
        (Nx.item [] k) i.(0) i.(1) token_count)

let decode_raw ?wavelength t k =
  let l = Nx.dim 0 t.grid in
  let n = l / reduction t.encoder in
  let b = Nx.dim 0 k in
  let dt = Nx.dtype t.grid in
  let v =
    Nx.cast dt (Scalar.decode_raw t.norm (Nx.slice [ Nx.A; Nx.R (0, 1) ] k))
  in
  let z = values dt (Nx.slice [ Nx.A; Nx.R (1, n + 1) ] k) in
  let h = Kaun.Linear.apply t.post_quant z in
  let out = List.fold_left (fun h l -> layer l h) h t.decoder in
  let y = Nx.squeeze ~axes:[ 2 ] (Nx.slice [ Nx.A; Nx.A; Nx.R (0, 1) ] out)
  and logit =
    Nx.squeeze ~axes:[ 2 ] (Nx.slice [ Nx.A; Nx.A; Nx.R (1, 2) ] out)
  in
  let y, logit, wavelength, outside =
    match wavelength with
    | None ->
        let w = Nx.broadcast_to [| b; l |] t.grid in
        (y, logit, w, Nx.zeros_at (Nx.beside k) Nx.bool [| b; l |])
    | Some w ->
        let on y = resample ~x:t.grid ~y w in
        let first = Nx.slice [ Nx.R (0, 1) ] t.grid
        and last = Nx.slice [ Nx.R (l - 1, l) ] t.grid in
        let outside = Nx.logical_or (Nx.less w first) (Nx.greater w last) in
        (on y, on logit, w, outside)
  in
  let flux = Nx.mul (Nx.add_s y 1.) (scale v) in
  let bad = Nx.greater_s (Nx.sigmoid logit) 0.5 in
  {
    flux = Nx.where outside (Nx.zeros_like flux) flux;
    ivar = Nx.ones_like flux;
    mask = Nx.logical_or outside bad;
    wavelength;
  }

(* Entry points: the checks, then the computation. A compiled codec checks
   copies on the host, so that its refusals come before its programs run, and
   its programs check nothing. *)

let host x = Nx.place Nx.Host.on x

let encode ?lengths t modality x =
  match t.programs with
  | None ->
      check_encode ?lengths modality x;
      encode_raw ?lengths t modality x
  | Some p ->
      ignore (shape_of "Spectrum.encode" x);
      let w = host x.wavelength in
      let mask = Nx.zeros Nx.bool (Nx.shape w) in
      check_encode
        ?lengths:(Option.map host lengths)
        modality
        { flux = w; ivar = w; mask; wavelength = w };
      p.p_encode lengths modality x

let decode ?wavelength t k =
  match t.programs with
  | None ->
      check_decode ?wavelength t k;
      decode_raw ?wavelength t k
  | Some p ->
      check_decode ?wavelength:(Option.map host wavelength) t (host k);
      p.p_decode wavelength k

(* Compiling *)

let spectrum_s () =
  Nx.Ptree.v (fun c x ->
      let open Nx.Ptree.Walk in
      let flux = field c "flux" tensor x.flux in
      let ivar = field c "ivar" tensor x.ivar in
      let mask = field c "mask" tensor x.mask in
      let wavelength = field c "wavelength" tensor x.wavelength in
      { flux; ivar; mask; wavelength })

let modality_s () =
  Nx.Ptree.v (fun c (m : Modality.spectrum) ->
      Nx.Ptree.Walk.case c (Modality.key (Spectrum m));
      m)

let jit t =
  let t = { t with programs = None } in
  let open Nx.Ptree in
  let encode lengths m x = encode_raw ?lengths t m x in
  let decode wavelength k = decode_raw ?wavelength t k in
  let programs =
    {
      p_encode =
        Rune.jit
          (option tensor @-> modality_s () @-> spectrum_s () @-> returns tensor)
          encode;
      p_decode =
        Rune.jit (option tensor @-> tensor @-> returns (spectrum_s ())) decode;
    }
  in
  { t with programs = Some programs }

let compiled t = Option.is_some t.programs

let place p t =
  let t' = map (Nx.place p) (Scalar.place p) t in
  if compiled t then jit t' else t'

let cast dt t =
  let t' = map (Nx.cast dt) Fun.id t in
  if compiled t then jit t' else t'
