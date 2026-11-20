(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's image codec: tokens for galaxy images.

    The image codec turns a galaxy's cutout, one image for each {e band} (a
    filter of a survey's camera), into 576 tokens, one for each cell of a 24 ×
    24 grid over its centre, and tokens back into images. One codec serves Hyper
    Suprime-Cam (HSC) and Legacy Survey images, so a model sees both surveys'
    bands in one space: nine bands in all, each with a fixed {e channel} of that
    space.

    An image batch is an array [[n; c; h; w]] of [n] cutouts of [c] bands,
    listed by a band list in that order, each image [h] rows of [w] pixels, and
    [h] and [w] at least 96. A pixel is in its band's {!unit}.

    Encoding first {e prepares} each image: it keeps the central 96 × 96 pixels,
    starting at row [(h - 96) / 2] and column [(w - 96) / 2], clamps each pixel
    to [±limit b] of its band [b], converts it to nanomaggies, and compresses
    its range by [x ↦ 10 · 0.01 · asinh (x / 0.01)], which is nearly linear
    below 0.01 and logarithmic above ({!prepare}). The {e embedding} ({!embed})
    puts the prepared bands in their channels of nine, zeros elsewhere, and a
    linear map projects each pixel's nine channels to the network's channels,
    scaled by [√(9 / c)] so that fewer bands give the same magnitude. The
    encoder of MAGVIT-v2 (Yu et al., "Language Model Beats Diffusion: Tokenizer
    is Key to Visual Generation", 2023), residual convolutions, reduces 96 × 96
    to 24 × 24, and a final convolution gives each cell five numbers, its
    {e latent}. {!Fsq} with the levels [[7; 5; 5; 5; 5]] makes each latent a
    token below 4375.

    Decoding inverts each step but the crop and the clamp: tokens become
    latents, a decoder rebuilds the network's channels at 96 × 96, a linear map
    gives the nine channels, and each asked-for band takes its channel, its
    range expanded by [y ↦ 0.01 sinh (y / (10 · 0.01))], in its unit.

    A codec is a structure ({!walk}) of weights on a device set:
    [Nx.Ptree.place (module Image)] moves it and [Nx.Ptree.cast (module Image)]
    gives it another float dtype. It computes with images and tokens of its set
    at its weights' float dtype, float32 as AION-1 stores them. In float32 it
    computes in the order AION-1 trained with: a division by a constant is a
    product with the constant's float32 reciprocal, and the quantizer's and the
    projection's constants are PyTorch's float32 ones. In float64 it computes as
    upstream's code does in float64, each constant the exact value rounded once,
    for a reference computation.

    Compiled by [Rune.jit] with the codec as an argument, as in
    [Rune.jit Nx.Ptree.(instantiate (module Image) @-> list (v walk_band) @->
     tensor @-> returns tensor) Image.encode], a codec computes to the accuracy
    [Rune.jit] states: its sums may combine their terms in another order than
    eager ones, so its results differ in their last bits, and a token whose
    value lies that close to a boundary may differ. A program is traced per list
    of bands and per shape, dtype and placement of the arguments, the codec's
    included. *)

(** {1:bands Bands} *)

(** The type for bands. The Legacy Survey's are those of the Dark Energy Camera
    (DECam), as its southern images were taken. *)
type band =
  | Hsc_g  (** HSC g, channel 0, ["HSC-G"]. *)
  | Hsc_r  (** HSC r, channel 1, ["HSC-R"]. *)
  | Hsc_i  (** HSC i, channel 2, ["HSC-I"]. *)
  | Hsc_z  (** HSC z, channel 3, ["HSC-Z"]. *)
  | Hsc_y  (** HSC y, channel 4, ["HSC-Y"]. *)
  | Des_g  (** Legacy Survey g, channel 5, ["DES-G"]. *)
  | Des_r  (** Legacy Survey r, channel 6, ["DES-R"]. *)
  | Des_i  (** Legacy Survey i, channel 7, ["DES-I"]. *)
  | Des_z  (** Legacy Survey z, channel 8, ["DES-Z"]. *)

val walk_band : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> band -> band
(** [walk_band c b] reports [b] as a case, so that [Rune.jit] keys a program on
    a list of bands. *)

val bands : band list
(** [bands] is the nine bands in channel order. *)

val channel : band -> int
(** [channel b] is [b]'s channel. *)

val name : band -> string
(** [name b] is [b]'s name in AION-1's code and files, as ["HSC-G"]. *)

val unit : band -> Ymir.Unit.t
(** [unit b] is the unit of [b]'s pixels: flux on AB zero point 27
    ({!Ymir.Units.zero_point}[ "27"]) for HSC, the nanomaggy for the Legacy
    Survey. *)

val limit : band -> float
(** [limit b] is the magnitude past which a pixel of [b] is clamped, in [b]'s
    unit: 80, 110, 200, 330 and 500 for HSC g, r, i, z and y; 6, 15, 20 and 25
    for the Legacy Survey's g, r, i and z. *)

(** {1:codecs Codecs} *)

type ('a, 'd) t
(** The type for image codecs whose weights are ['a] on the device set ['d]:
    [((float, 'b, 'd) Nx.t, 'd) t] computes at the float dtype ['b]. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c t] walks [t]'s weights, named as its layers are, and reports the
    quantizer's levels and the number of its residual units. *)

val of_archive : 'd Nx_io.Archive.t -> ('d Nx.float32_t, 'd) t
(** [of_archive a] is the codec of the weights in [a], named and shaped as in
    upstream's [ImageCodec] state dict, PyTorch's convolution weights
    [[out; in; kh; kw]]. The channel widths are read from the weights, so a
    narrower network of the same layers loads too; the quantizer's levels are
    [_quantizer.levels].

    AION-1's codec is [of_archive (Hub.codec Image)].

    Raises [Failure] naming the entry if [a] lacks one, if its shape disagrees
    with the others', or if [a] holds an entry the codec does not read. *)

type arch = {
  projected : int;  (** The network's channels at each pixel: 54. *)
  width : int;
      (** The first stage's width [d], 128: the second's is [2d], the third's
          [4d]. *)
}
(** The type for codec architectures. A residual unit's squeeze-excite has half
    the unit's width, at least 1, and the quantizer AION-1's levels. *)

val init : key:'d Nx.Rng.key -> arch -> ('d Nx.float32_t, 'd) t
(** [init ~key a] is a codec of architecture [a] at [key]'s set, with kaun's
    initialisers: Glorot-uniform convolutions with zero biases, drawn from keys
    folded from [key], and an identity layer norm.

    Raises [Invalid_argument] if a field of [a] is not positive. *)

(** {1:images Images and tokens}

    Each function takes its bands as a band list, and raises [Invalid_argument]
    if the list is empty or repeats a band, or if its array's band axis does not
    have one entry per band. *)

val side : int
(** [side] is [96], the side in pixels of a prepared image ({!prepare}) and of
    a decoded one ({!decode}). *)

val prepare : band list -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [prepare bs x] is the prepared images of [x], [[n; c; 96; 96]].

    Raises [Invalid_argument] if [x] is not [[n; c; h; w]] with [h] and [w] at
    least 96. *)

val embed :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  band list ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [embed t bs p] is the prepared images [p], [[n; c; 96; 96]], each band in
    its channel of nine, zeros elsewhere, projected to the network's channels
    and scaled by [√(9 / c)]: [[n; 96; 96; m]], channels last.

    Raises [Invalid_argument] if [p] is not [[n; c; h; w]] with [h] and [w] at
    least 96. *)

val latent :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  band list ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [latent t bs x] is the latents of [x], [[n; 576; 5]], the cells in row-major
    order: the encoder's output of {!embed}[ t bs (]{!prepare}[ bs x)].

    Raises as {!prepare} does. *)

val encode :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  band list ->
  (float, 'b, 'd) Nx.t ->
  'd Nx.int64_t
(** [encode t bs x] is the tokens of [x], [[n; 576]]: {!Fsq.encode} of
    {!latent}[ t bs x].

    Raises as {!latent} and {!Fsq.encode} do. *)

val restore : band list -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [restore bs y] is the bands [bs] of the nine compressed channels [y],
    [[n; 9; 96; 96]], expanded and in their units: [[n; c; 96; 96]]. It inverts
    {!prepare}'s compression and conversion. *)

val decode :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  band list ->
  'd Nx.int64_t ->
  (float, 'b, 'd) Nx.t
(** [decode t bs k] is the images of the tokens [k], [[n; 576]], in the bands
    [bs]: [[n; c; 96; 96]], {!restore} of the decoder's nine channels.

    Raises [Invalid_argument] if [k] is not [[n; 576]], and as {!Fsq.decode}
    does. *)
