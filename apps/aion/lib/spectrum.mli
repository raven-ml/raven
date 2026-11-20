(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's spectrum codec: tokens for spectra.

    A {e spectrum} is a batch of [b] rows of [m] {e pixels}, each pixel a flux,
    its inverse variance [ivar], a mask, true where the pixel is bad, and a
    wavelength in ångströms, non-decreasing along each row. The codec turns a
    row into 1 + [n] {e tokens}, integers in [\[0, 1024)], and tokens back into
    a spectrum.

    Encoding first prepares a row:
    + NaN fluxes become [0] and infinite ones the largest finite float of their
      dtype and sign; NaN inverse variances become [0], and each is clamped to
      [[0, 100]].
    + The {e normalisation} is the sum of the unmasked fluxes over one more than
      their count, at least [0.1], as [log10 (1 + mean)]. Its token is the
      number of the codebook's 1024 values {!levels} below it, at most 1023: an
      even grid from -1 to 5. The normalisation becomes its token's value [v],
      the least codebook value at or above it, and [s = max (10{^v} - 1) 0.1].
    + The {e signal} [(flux / s - 1) · 0.2] and the {e noise} [√ivar / s · 0.2]
      are resampled onto the codec's {e latent grid} of [L] wavelengths
      ({!grid}): linearly between the two pixels around each grid wavelength,
      the lower pixel strictly below it, or the first two pixels at the first
      pixel's wavelength; [0] below the row's first wavelength and above its
      last. Between two pixels of one wavelength the slope is [0].
    + The two resampled rows go through [asinh]: the network's input
      ({!prepare}).

    A ConvNeXt V2 network over the latent grid, the {e encoder}, then makes [n]
    latent vectors of 10 values ({!latent}), [n = L / 32] for AION-1's weights,
    and each becomes a token by {!Lfq.encode}: its bit [9 - c] is set where
    value [c] is positive, value 0 giving the most significant bit. The
    normalisation's token comes first.

    Decoding reverses this: each token's bits give a vector of [±1]
    ({!Lfq.decode}), a second network, the {e decoder}, makes two values on each
    latent wavelength, a signal [y] and a mask logit [z], and the flux is
    [(y + 1) · s] and the pixel bad where [sigmoid z], computed in float32,
    exceeds [0.5]. The decoder was trained to emit [flux / s - 1], without the
    encoder's [asinh] and factor [0.2]. The inverse variance is not encoded and
    decodes as [1].

    Each spectrum modality ({!Modality.spectrum}) has a length, and AION-1's
    weights were trained on spectra padded to it ({!pad}) with pixels of flux
    [0], inverse variance [0], mask [true] and wavelength [99999]. A pad's
    wavelength is far above the latent grid, so on the grid's wavelengths above
    a row's last pixel the signal tends to the first pad's, [-0.2], and the
    noise to [0]. {!encode} encodes rows as given: rows that are not padded give
    tokens the weights never saw, as upstream's base spectrum class does.

    A codec is a structure ({!walk}) of weights and a latent grid on a device
    set, and a fixed normalisation codebook: [Nx.Ptree.place (module Spectrum)]
    moves it and [Nx.Ptree.cast (module Spectrum)] casts its weights and grid to
    another float dtype, as float64 for a reference computation, keeping the
    codebook float32, its values compared exactly with the normalisation at that
    dtype. It computes with spectra and tokens of its set at its weights' dtype,
    float32 as AION-1 stores them.

    Compiled by [Rune.jit] with the codec as an argument, as in
    [Rune.jit Nx.Ptree.(instantiate (module Spectrum) @-> nest (module Spectra)
     tensor @-> returns tensor) Spectrum.encode], a codec computes to the
    accuracy [Rune.jit] states: its sums may combine their terms in another
    order than eager ones, so its results differ in their last bits, and a token
    whose value lies that close to a boundary may differ. *)

(** {1:spectra Spectra} *)

type ('a, 'd) spectrum = {
  flux : 'a;  (** Flux, [[|b; m|]]. *)
  ivar : 'a;  (** Inverse variance of the flux, [[|b; m|]]. *)
  mask : 'd Nx.bool_t;  (** [true] where a pixel is bad, [[|b; m|]]. *)
  wavelength : 'a;
      (** Wavelength in ångströms, [[|b; m|]], non-decreasing along a row. *)
}
(** The type for spectra whose values are ['a] on the device set ['d]:
    [((float, 'b, 'd) Nx.t, 'd) spectrum] holds floats of dtype ['b]. *)

module Spectra : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) spectrum
(** Spectra as a structure: its walk walks the flux, inverse variance and
    wavelength at the parameter's positions, and the mask as a fixed array. *)

val length : Modality.spectrum -> int
(** [length m] is the number of pixels of [m]'s padded spectra: [7808] for
    {!Modality.Desi}, [4800] for {!Modality.Sdss}. *)

val pad :
  ?lengths:'d Nx.int64_t ->
  Modality.spectrum ->
  ((float, 'b, 'd) Nx.t, 'd) spectrum ->
  ((float, 'b, 'd) Nx.t, 'd) spectrum
(** [pad ~lengths m x] is each row of [x] padded to {!length}[ m] pixels.
    [lengths], [[|b|]], is each row's number of pixels, all of them by default:
    row [i] is its first [lengths.(i)] pixels, padded, and the pixels past them
    are not read. So spectra of different lengths encode in one batch.

    Raises [Invalid_argument] if [x]'s fields are not of one 2-d shape, its rows
    are longer than {!length}[ m], or [lengths] is not [[|b|]] or holds a length
    outside [[1, w]], [w] the rows' width. *)

(** {1:codecs Codecs} *)

type ('a, 'd) t
(** The type for spectrum codecs whose weights are ['a] on the device set ['d]:
    [((float, 'b, 'd) Nx.t, 'd) t] computes at the float dtype ['b]. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c t] walks [t]'s weights and latent grid, reporting each network's
    layers by kind, and its normalisation codebook as a fixed array. *)

val of_archive : 'd Nx_io.Archive.t -> ('d Nx.float32_t, 'd) t
(** [of_archive a] is the codec whose weights are [a]'s, named as in AION-1's
    file [codecs/spectrum/model.safetensors]. The number of stages of each
    network, of blocks in each stage and of channels in each layer are those the
    weights have. AION-1's codec is [of_archive (Hub.codec (Spectrum Desi))].

    Raises [Failure] if a weight is missing or is not float32, if [a] holds a
    tensor the codec does not read, as when a stage lacks one of its blocks, or
    if the grid's length is not a positive multiple of the encoder's reduction,
    32 for AION-1. The file's restatements of the LFQ's bit weights and of the
    grid's length, first wavelength and spacing are not read. *)

type arch = {
  encoder : (int * int) list;
      (** Each encoder stage's width and number of blocks:
          [[(64, 3); (128, 3); (256, 27); (512, 3)]]. *)
  decoder : (int * int) list;
      (** Each decoder stage's but the last: [[(256, 3); (128, 9); (64, 3)]]. *)
  output : int;
      (** The decoder's last stage's number of blocks, 3, at its two channels:
          the signal and the mask logit. *)
  grid : int;  (** The latent grid's length [L]: [8704]. *)
}
(** The type for codec architectures. The encoder's first stage reduces the grid
    by 4 and each later one by 2, the decoder's last stage expands it by 4 and
    each earlier one by 2. *)

val init : key:'d Nx.Rng.key -> arch -> ('d Nx.float32_t, 'd) t
(** [init ~key a] is a codec of architecture [a] at [key]'s set, with kaun's
    initialisers, drawn from keys folded from [key]: Glorot-uniform weights,
    zero biases, identity layer norms, and GRNs of zero [gamma] and [beta], as
    ConvNeXt V2 starts them. Its grid is AION-1's first [L] wavelengths and its
    normalisation codebook AION-1's even grid of 1024 values from -1 to 5.

    Raises [Invalid_argument] if the encoder has no stage or the decoder not one
    stage fewer before its last, a width is not positive or a number of blocks
    negative, or [L] is not a positive multiple of the encoder's reduction. *)

val grid : ('a, 'd) t -> 'a
(** [grid t] is [t]'s latent grid, [[|L|]] wavelengths in ångströms:
    [3500 + 0.8 k] for [k] below [8704], rounded to float32, in AION-1's file.
*)

val levels : ('a, 'd) t -> 'd Nx.float32_t
(** [levels t] is the codebook of [t]'s normalisation token, [[|1024|]]. *)

(** {1:steps Steps} *)

val normalisation : ((float, 'b, 'd) Nx.t, 'd) spectrum -> (float, 'b, 'd) Nx.t
(** [normalisation x] is each row's normalisation, [[|b|]], before its token. *)

val prepare :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  ((float, 'b, 'd) Nx.t, 'd) spectrum ->
  (float, 'b, 'd) Nx.t * 'd Nx.int64_t
(** [prepare t x] is the network's input, [[|b; L; 2|]], the signal then the
    noise at each grid wavelength, and the normalisation's token, [[|b|]].

    Raises [Invalid_argument] if [x]'s fields are not of one 2-d shape, its rows
    have fewer than two pixels, or a row's wavelengths decrease or are NaN. *)

val latent :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  ((float, 'b, 'd) Nx.t, 'd) spectrum ->
  (float, 'b, 'd) Nx.t * 'd Nx.int64_t
(** [latent t x] is the encoder's latent vectors, [[|b; n; 10|]], and the
    normalisation's token, [[|b|]], of {!prepare}[ t x].

    Raises as {!prepare} does. *)

(** {1:tokens Tokens} *)

val encode :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  ((float, 'b, 'd) Nx.t, 'd) spectrum ->
  'd Nx.int64_t
(** [encode t x] is the tokens of each row of the spectra [x], [[|b; 1 + n|]],
    the normalisation's first, then {!Lfq.encode} of {!latent}[ t x]. For
    AION-1's tokens of a modality [m], encode {!pad}[ m x].

    Raises as {!prepare} does. *)

val decode :
  ?wavelength:(float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t, 'd) t ->
  'd Nx.int64_t ->
  ((float, 'b, 'd) Nx.t, 'd) spectrum
(** [decode ~wavelength t k] is the spectrum of the tokens [k], [[|b; 1 + n|]],
    on the wavelengths [wavelength], [[|b; m|]], or on {!grid}[ t] for each row
    without it. A wavelength between two of the grid's takes the linear
    interpolation of the decoder's two values there. A pixel at a wavelength
    outside the grid's range has flux [0] and is bad.

    Raises [Invalid_argument] if [k] is not [[|b; 1 + n|]], a token is outside
    [\[0, 1024)] ({!Scalar.decode}, {!Lfq.decode}), or [wavelength] is not 2-d
    with [b] rows. *)
