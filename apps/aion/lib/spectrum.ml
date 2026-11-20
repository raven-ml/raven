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

(* Spectra *)

type ('a, 'd) spectrum = {
  flux : 'a;
  ivar : 'a;
  mask : 'd Nx.bool_t;
  wavelength : 'a;
}
[@@deriving ptree]

module Spectra = struct
  type ('a, 'd) t = ('a, 'd) spectrum

  let walk = walk_spectrum
end

(* Upstream's pad_length of DESISpectrum and SDSSSpectrum
   (aion/modalities.py). *)
let length : Modality.spectrum -> int = function Desi -> 7808 | Sdss -> 4800

let invalid fmt = Format.kasprintf invalid_arg fmt
let refusal fmt = Printf.ksprintf (fun m -> Invalid_argument m) fmt

let shape_of fn x =
  let s = Nx.shape x.flux in
  let same t = Nx.shape t = s in
  if
    Array.length s <> 2
    || not (same x.ivar && Nx.shape x.mask = s && same x.wavelength)
  then
    invalid "Aion.%s: fields not of one 2-d shape" fn;
  (s.(0), s.(1))

(* Row [i]'s pixels at and past [lengths.(i)] become pads, so the row reads as
   its first [lengths.(i)] pixels padded, and the pixels cut are never read. *)
let trim lengths b m x =
  if Nx.shape lengths <> [| b |] then
    invalid "Aion.Spectrum.pad: lengths not of shape [%d]" b;
  let inside =
    Nx.logical_and
      (Nx.greater_equal_s lengths 1L)
      (Nx.less_equal_s lengths (Int64.of_int m))
  in
  Nx.check Nx.Ptree.tensor inside lengths (fun i l ->
      refusal "Aion.Spectrum.pad: length %Ld of row %d outside [1, %d]"
        (Nx.item [] l) i.(0) m);
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

let pad ?lengths modality x =
  let n = length modality in
  let b, m = shape_of "Spectrum.pad" x in
  if m > n then invalid "Aion.Spectrum.pad: rows of %d pixels, at most %d" m n;
  let x = match lengths with None -> x | Some l -> trim l b m x in
  let p = [| (0, 0); (0, n - m) |] in
  {
    flux = Nx.pad p 0. x.flux;
    ivar = Nx.pad p 0. x.ivar;
    mask = Nx.pad p true x.mask;
    wavelength = Nx.pad p pad_wavelength x.wavelength;
  }

(* Codecs *)

type ('a, 'd) block = {
  dw : ('a, 'd) Kaun.Conv.t;
  norm : ('a, 'd) Kaun.Layer_norm.t;
  up : ('a, 'd) Kaun.Linear.t;
  gamma : 'a;
  beta : 'a;
  down : ('a, 'd) Kaun.Linear.t;
}
[@@deriving ptree]

(* A [Patch] is a convolution whose stride is its kernel's extent, as every
   down-sampling one here is, and an [Unpatch] its transpose. *)
type ('a, 'd) layer =
  | Patch of ('a, 'd) Kaun.Conv.t
  | Unpatch of ('a, 'd) Kaun.Conv_transpose.t
  | Norm of ('a, 'd) Kaun.Layer_norm.t
  | Block of ('a, 'd) block
[@@deriving ptree]

type arch = {
  encoder : (int * int) list;
  decoder : (int * int) list;
  output : int;
  grid : int;
}

(* [norm] is the normalisation token's codebook, as a one-channel scalar codec
   of fixed edges: upstream's ScalarLinearQuantizer clamps torch.bucketize over
   all 1024 values to 1023, which is Scalar's count over the first 1023 wherever
   the values are in order, as the stored grid is. *)
type ('a, 'd) t = {
  encoder : ('a, 'd) layer list;
  pre_norm : ('a, 'd) Kaun.Layer_norm.t;
  quant : ('a, 'd) Kaun.Linear.t;
  post_quant : ('a, 'd) Kaun.Linear.t;
  decoder : ('a, 'd) layer list;
  grid : 'a;
  norm : 'd Scalar.t;
}
[@@deriving ptree]

let grid t = t.grid
let levels t = Nx.reshape [| -1 |] (Scalar.edges t.norm)

(* Reading the weights. Upstream's modules (aion/codecs/modules/convnext.py)
   store a convolution [out; in / groups; k], a transposed one [in; out; k] and
   a linear layer [out; in]; kaun takes the channels last and a linear layer
   [out; in] too. The encoder's stage [i] is [downsample_layers.i], a
   convolution then a layer norm for the stem and a layer norm then a
   convolution after it, then the blocks [stages.i.j]; a last layer norm closes
   it. The decoder's stages are [upsample_layers.i] then [stages.i], and nothing
   closes it. *)

let weight a name = Nx_io.Archive.tensor Nx.float32 name a

(* Whether [a] holds [name]: a probe through [names], which [import] does not
   note as a read. *)
let has a name = List.mem name (Nx_io.Archive.names a)

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

(* A network of [n] stages, as convnext.py lays them out: stage [i]'s resampling
   then a layer norm for the first, a layer norm then the resampling for the
   others, then its blocks. *)
let stages n ~resize ~norm ~blocks =
  List.concat
    (List.init n (fun i ->
         let first, second =
           if i = 0 then (resize 0, Norm (norm 0)) else (Norm (norm i), resize i)
         in
         first :: second :: blocks i))

let network a net resize sample =
  let sampling i = Printf.sprintf "%s.%s_layers.%d" net sample i in
  stages
    (count a (fun i -> sampling i ^ ".0.weight"))
    ~resize:(fun i -> resize (sampling i ^ if i = 0 then ".0" else ".1"))
    ~norm:(fun i -> layer_norm a (sampling i ^ if i = 0 then ".1" else ".0"))
    ~blocks:(stage a net)

let stride (c : _ Kaun.Conv.t) = Nx.dim 1 c.w

(* The grid wavelengths per latent vector: the product of the encoder's
   strides. *)
let reduction encoder =
  List.fold_left
    (fun r -> function Patch c -> r * stride c | _ -> r)
    1 encoder

let failure fmt = Printf.ksprintf failwith fmt

(* A file missing a block leaves the next block's weights unread, so [import]
   refuses it. *)
let read a =
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
    failure
      "Aion.Spectrum.of_archive: a latent grid of %d wavelengths, a positive \
       multiple of %d expected"
      l r;
  {
    encoder;
    pre_norm = layer_norm a "pre_quant_norm";
    quant = linear a "quant_conv";
    post_quant = linear a "post_quant_conv";
    decoder;
    grid;
    norm = Scalar.v Identity levels;
  }

(* Upstream's buffers restate what the weights and the grid hold: the LFQ's bit
   weights (aion/codecs/quantizers/lookup_free_quantization.py, [mask]), and the
   grid's length, first wavelength and spacing. They are not read. *)
let restated =
  [
    "_quantizer._quantizer.mask";
    "latent_grid.length";
    "latent_grid.lambda_min";
    "latent_grid.resolution";
  ]

let of_archive a =
  Nx_io.Archive.import ~unread:(fun n -> List.mem n restated) read a

(* Initialising *)

(* The stem's stride and each later stage's, as AION-1's; the decoder's last
   stage takes the stem's. *)
let stem = 4
let step = 2

(* The latent grid of AION-1's file, [3500 + 0.8 k] rounded to float32, and the
   normalisation's codebook, an even grid from -1 to 5. *)
let grid_start = 3500.
let grid_spacing = 0.8
let level_min = -1.
let level_max = 5.

(* The decoder's last stage has two channels, the signal and the mask logit. *)
let init ~key (a : arch) =
  let enc = Array.of_list a.encoder in
  let dec = Array.of_list (a.decoder @ [ (2, a.output) ]) in
  let k = Array.length enc in
  let positive = Array.for_all (fun (w, n) -> w > 0 && n >= 0) in
  if k = 0 || Array.length dec <> k then
    invalid_arg
      "Aion.Spectrum.init: an encoder of no stage, or a decoder of another \
       number";
  if not (positive enc && positive dec) then
    invalid_arg "Aion.Spectrum.init: a width not positive or blocks negative";
  let r = stem * (1 lsl (k - 1)) in
  if a.grid <= 0 || a.grid mod r <> 0 then
    invalid
      "Aion.Spectrum.init: a grid of %d wavelengths, a positive multiple of %d \
       expected"
      a.grid r;
  let at = Nx.Rng.placement key in
  let n = ref 0 in
  let key () =
    incr n;
    Nx.Rng.fold_in key !n
  in
  let f32 = Nx.float32 in
  let ln d = Kaun.Layer_norm.init ~dim:d at in
  let linear i o = Kaun.Linear.init ~key:(key ()) ~inputs:i ~outputs:o in
  (* ConvNeXt V2's GRN starts as the identity, gamma and beta 0. *)
  let block d =
    Block
      {
        dw =
          Kaun.Conv.make ~key:(key ()) ~groups:d ~in_channels:d ~out_channels:d
            ~kernel_size:[| 7 |] f32;
        norm = ln d;
        up = linear d (4 * d);
        gamma = Nx.zeros_at at f32 [| 4 * d |];
        beta = Nx.zeros_at at f32 [| 4 * d |];
        down = linear (4 * d) d;
      }
  in
  (* Stage [i] at width [w.(i)]: its layer norm at the width it reads. *)
  let net w resize =
    stages k ~resize
      ~norm:(fun i -> ln (fst w.(max 0 (i - 1))))
      ~blocks:(fun i -> List.init (snd w.(i)) (fun _ -> block (fst w.(i))))
  in
  let width = fst enc.(k - 1) in
  let encoder =
    net enc (fun i ->
        let input = if i = 0 then 2 else fst enc.(i - 1) in
        Patch
          (Kaun.Conv.init ~key:(key ()) ~in_channels:input
             ~out_channels:(fst enc.(i))
             ~kernel_size:[| (if i = 0 then stem else step) |]))
    @ [ Norm (ln width) ]
  in
  let decoder =
    net dec (fun i ->
        let input = if i = 0 then width else fst dec.(i - 1) in
        Unpatch
          (Kaun.Conv_transpose.init ~key:(key ()) ~in_channels:input
             ~out_channels:(fst dec.(i))
             ~kernel_size:[| (if i = k - 1 then stem else step) |]))
  in
  let k64 = Nx.arange_f_at at Nx.float64 0. (Float.of_int a.grid) 1. in
  let grid = Nx.cast f32 (Nx.add_s (Nx.mul_s k64 grid_spacing) grid_start) in
  let levels =
    Nx.reshape [| 1; token_count |]
      (Nx.linspace_at at f32 level_min level_max token_count)
  in
  {
    encoder;
    pre_norm = ln width;
    quant = linear width bits;
    post_quant = linear bits width;
    decoder;
    grid;
    norm = Scalar.v Identity levels;
  }

(* Preparation *)

(* torch.nan_to_num: NaN to 0 and the infinities to the dtype's largest finite
   floats. *)
let nan_to_num x =
  let big = (Nx.Dtype.float_format (Nx.dtype x)).max_finite in
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

(* Upstream's interp1d (aion/codecs/modules/spectrum.py): [y] sampled at [x],
   read at [xnew], linearly within the sample's interval [x(i), x(i+1)] that
   holds it, [y(i) + (xnew - x(i)) · (y(i) - y(i+1)) / (x(i) - x(i+1))], and 0
   outside [x(0), x(m-1)]. Each is [b; m] or [1; m] and [xnew] [b; k] or [1; k],
   rows broadcast. A key equal to [x(i)] takes the interval ending there, the
   first for [x(0)]. An interval of zero width, between repeated wavelengths as
   pads are, is read only by a key equal to [x(0)] where [x(1) = x(0)]; its
   slope is 0 where upstream's is NaN. *)
let resample ~x ~y xnew =
  let m = Nx.dim 1 x in
  let i =
    Nx.clamp ~min:0L
      ~max:(Int64.of_int (m - 2))
      (Nx.sub_s (Nx.searchsorted_along_axis ~axis:1 ~side:`Left x xnew) 1L)
  in
  let lo t = Nx.slice [ A; R (0, m - 1) ] t
  and hi t = Nx.slice [ A; R (1, m) ] t in
  let dx = Nx.sub (lo x) (hi x) in
  let slope = Nx.div (Nx.sub (lo y) (hi y)) dx in
  let slope = Nx.where (Nx.equal_s dx 0.) (Nx.zeros_like slope) slope in
  let gather t =
    Nx.take_along_axis ~axis:1 i
      (Nx.broadcast_to [| Nx.dim 0 i; Nx.dim 1 t |] t)
  in
  let ynew =
    Nx.add (gather y) (Nx.mul (Nx.sub xnew (gather x)) (gather slope))
  in
  let first = Nx.slice [ A; R (0, 1) ] x
  and last = Nx.slice [ A; R (m - 1, m) ] x in
  let outside = Nx.logical_or (Nx.less xnew first) (Nx.greater xnew last) in
  Nx.where outside (Nx.zeros_like ynew) ynew

let sorted w =
  let m = Nx.dim 1 w in
  let lo = Nx.slice [ Nx.A; Nx.R (0, m - 1) ] w in
  let ok = Nx.less_equal lo (Nx.slice [ Nx.A; Nx.R (1, m) ] w) in
  Nx.check Nx.Ptree.tensor ok lo (fun i _ ->
      refusal
        "Aion.Spectrum.encode: row %d's wavelengths decrease or are NaN \
         between pixels %d and %d"
        i.(0) i.(1)
        (i.(1) + 1))

(* Rows of two pixels or more, their wavelengths sorted: the checks, then the
   computation. *)
let prepare t x =
  let _, m = shape_of "Spectrum.encode" x in
  if m < 2 then invalid "Aion.Spectrum.encode: rows of %d pixel" m;
  sorted x.wavelength;
  let code = Scalar.encode t.norm (normalisation x) in
  let v = Nx.cast (Nx.dtype x.flux) (Scalar.decode t.norm code) in
  let s = Nx.reshape [| -1; 1 |] (scale v) in
  let signal =
    Nx.mul_s (Nx.sub_s (Nx.div (nan_to_num x.flux) s) 1.) input_scaling
  in
  let ivar = Nx.clamp ~min:0. ~max:ivar_max (nan_to_num x.ivar) in
  let noise = Nx.mul_s (Nx.div (Nx.sqrt ivar) s) input_scaling in
  let grid = Nx.reshape [| 1; -1 |] t.grid in
  let on_grid y = resample ~x:x.wavelength ~y grid in
  (Nx.asinh (Nx.stack ~axis:(-1) [ on_grid signal; on_grid noise ]), code)

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

let latent t x =
  let input, code = prepare t x in
  let h = List.fold_left (fun h l -> layer l h) input t.encoder in
  let h = Kaun.Layer_norm.apply ~eps:pre_norm_eps t.pre_norm h in
  (Kaun.Linear.apply t.quant h, code)

(* Tokens *)

let encode t x =
  let h, code = latent t x in
  Nx.concatenate ~axis:1 [ Nx.reshape [| -1; 1 |] code; Lfq.encode h ]

(* Decoding *)

(* The tokens are checked here, so that a refusal names a token's position in
   [k]; Scalar.decode and Lfq.decode then find them in range. *)
let decode ?wavelength t k =
  let l = Nx.dim 0 t.grid in
  let n = l / reduction t.encoder in
  let s = Nx.shape k in
  if Array.length s <> 2 || s.(1) <> n + 1 then
    invalid "Aion.Spectrum.decode: tokens of shape %a, [b; %d] expected"
      Nx.pp_shape s (n + 1);
  let b = s.(0) in
  (match wavelength with
  | Some w when Nx.ndim w <> 2 || Nx.dim 0 w <> b ->
      invalid "Aion.Spectrum.decode: wavelengths not of %d rows" b
  | _ -> ());
  let inside =
    Nx.logical_and (Nx.greater_equal_s k 0L)
      (Nx.less_s k (Int64.of_int token_count))
  in
  Nx.check Nx.Ptree.tensor inside k (fun i k ->
      refusal "Aion.Spectrum.decode: token %Ld at [%d; %d] outside [0, %d)"
        (Nx.item [] k) i.(0) i.(1) token_count);
  let dt = Nx.dtype t.grid in
  let v =
    Nx.cast dt (Scalar.decode t.norm (Nx.slice [ Nx.A; Nx.R (0, 1) ] k))
  in
  let z = Lfq.decode dt ~bits (Nx.slice [ Nx.A; Nx.R (1, n + 1) ] k) in
  let h = Kaun.Linear.apply t.post_quant z in
  let out = List.fold_left (fun h l -> layer l h) h t.decoder in
  let y = Nx.slice [ A; A; I 0 ] out and logit = Nx.slice [ A; A; I 1 ] out in
  let y, logit, wavelength, outside =
    match wavelength with
    | None ->
        let w = Nx.broadcast_to [| b; l |] t.grid in
        (y, logit, w, Nx.zeros_at (Nx.beside k) Nx.bool [| b; l |])
    | Some w ->
        let on y = resample ~x:(Nx.reshape [| 1; l |] t.grid) ~y w in
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
