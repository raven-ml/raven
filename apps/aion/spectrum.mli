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
    + NaN fluxes become [0] and infinite ones the largest float32 of their sign;
      NaN inverse variances become [0], and each is clamped to [[0, 100]].
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
    + The two resampled rows go through [asinh].

    A ConvNeXt V2 network over the latent grid, the {e encoder}, then makes [n]
    latent vectors of 10 values, [n = L / 32] for AION-1's weights, and each
    becomes a token whose bit [9 - c] is set where value [c] is positive: value
    0 gives the most significant bit. The normalisation's token comes first.

    Decoding reverses this: each token's bits give a vector of [±1], a second
    network, the {e decoder}, makes two values on each latent wavelength, a
    signal [y] and a mask logit [z], and the flux is [(y + 1) · s] and the pixel
    bad where [sigmoid z], computed in float32, exceeds [0.5]. The decoder was
    trained to emit [flux / s - 1], without the encoder's [asinh] and factor
    [0.2]. The inverse variance is not encoded and decodes as [1].

    Each spectrum modality ({!Modality.spectrum}) has a length, and encoding
    first pads each row to it with pixels of flux [0], inverse variance [0],
    mask [true] and wavelength [99999]: AION-1's weights were trained on padded
    spectra. A pad's wavelength is far above the latent grid, so on the grid's
    wavelengths above a row's last pixel the signal tends to the first pad's,
    [-0.2], and the noise to [0].

    A codec lies on a device set, as its weights do, and computes with spectra
    and tokens of that set: {!place} moves it. It computes at its weights' float
    dtype, float32 as AION-1 stores them, with spectra of that dtype: {!cast}
    gives it another, as float64 for a reference computation. *)

(** {1:spectra Spectra} *)

type ('b, 'd) spectrum = {
  flux : (float, 'b, 'd) Nx.t;  (** Flux, [[|b; m|]]. *)
  ivar : (float, 'b, 'd) Nx.t;  (** Inverse variance of the flux, [[|b; m|]]. *)
  mask : 'd Nx.bool_t;  (** [true] where a pixel is bad, [[|b; m|]]. *)
  wavelength : (float, 'b, 'd) Nx.t;
      (** Wavelength in ångströms, [[|b; m|]], non-decreasing along a row. *)
}
(** The type for spectra of float dtype ['b] on the device set ['d]. *)

val length : Modality.spectrum -> int
(** [length m] is the number of pixels of [m]'s padded spectra: [7808] for
    {!Modality.Desi}, [4800] for {!Modality.Sdss}. *)

(** {1:codecs Codecs} *)

type ('b, 'd) t
(** The type for spectrum codecs computing at float dtype ['b] on the device set
    ['d]. *)

val of_archive : 'd Nx_io.Archive.t -> (Nx.float32_elt, 'd) t
(** [of_archive a] is the codec whose weights are [a]'s, named as in AION-1's
    file [codecs/spectrum/model.safetensors]. The number of stages of each
    network, of blocks in each stage and of channels in each layer are those the
    weights have.

    Raises [Failure] if a weight is missing or is not float32, if [a] holds a
    tensor the codec does not read, as when a stage lacks one of its blocks, if
    the LFQ's stored bit weights or the grid's stored length or first wavelength
    disagree with the codec's, or if the grid's length is not a positive
    multiple of the encoder's reduction, 32 for AION-1. *)

val load :
  ?cache_dir:string -> ?offline:bool -> unit -> (Nx.float32_elt, Nx.host) t
(** [load ()] is {!of_archive} of the spectrum codec's file in {!Hub.codec}, to
    which [cache_dir] and [offline] are passed.

    Raises [Failure] if the file cannot be fetched or read, or as {!of_archive}.
*)

val grid : ('b, 'd) t -> (float, 'b, 'd) Nx.t
(** [grid t] is [t]'s latent grid, [[|L|]] wavelengths in ångströms:
    [3500 + 0.8 k] for [k] below [8704], rounded to float32, in AION-1's file.
*)

val levels : ('b, 'd) t -> 'd Nx.float32_t
(** [levels t] is the codebook of [t]'s normalisation token, [[|1024|]]. *)

val place : 'e Nx.Placement.t -> ('b, 'd) t -> ('b, 'e) t
(** [place p t] is [t] with its weights at [p], as {!Nx.place}.

    Raises as {!Nx.place} does. *)

val cast : (float, 'c) Nx.dtype -> ('b, 'd) t -> ('c, 'd) t
(** [cast dt t] is [t] with its weights and grid cast to [dt], as {!Nx.cast}:
    exactly, for a dtype as wide as float32. Its normalisation codebook stays
    float32, its values compared exactly with the normalisation at [dt]. *)

(** {1:tokens Tokens} *)

val encode :
  ?lengths:'d Nx.int64_t ->
  ('b, 'd) t ->
  Modality.spectrum ->
  ('b, 'd) spectrum ->
  'd Nx.int64_t
(** [encode ~lengths t m x] is the tokens of each row of the spectra [x] of the
    modality [m], [[|b; 1 + n|]], the normalisation's first, padded to
    {!length}[ m]. [lengths], [[|b|]], is each row's number of pixels, all of
    them by default: row [i] is encoded as its first [lengths.(i)] pixels alone,
    padded, and the pixels past them are not read. So spectra of different
    lengths encode in one batch.

    Raises [Invalid_argument] if [x]'s fields are not of one 2-d shape, its rows
    are longer than {!length}[ m], [lengths] is not [[|b|]] or holds a length
    outside [\[1, w\]], [w] the rows' width, or a row's wavelengths decrease or
    are NaN. *)

val decode :
  ?wavelength:(float, 'b, 'd) Nx.t ->
  ('b, 'd) t ->
  'd Nx.int64_t ->
  ('b, 'd) spectrum
(** [decode ~wavelength t k] is the spectrum of the tokens [k], [[|b; 1 + n|]],
    on the wavelengths [wavelength], [[|b; m|]], or on {!grid}[ t] for each row
    without it. A wavelength between two of the grid's takes the linear
    interpolation of the decoder's two values there. A pixel at a wavelength
    outside the grid's range has flux [0] and is bad.

    Raises [Invalid_argument] if [k] is not [[|b; 1 + n|]], a token is outside
    [\[0, 1024)], or [wavelength] is not 2-d with [b] rows. *)

(**/**)

(* For aion's tests: steps of [encode] and [decode]. [pad m x] pads as [encode]
   does; [encode_unpadded t x] encodes [x] as given, as upstream's base Spectrum
   class, which its goldens use, does; [latent t x] is the LFQ's input, [[|b; n;
   10|]], and the normalisation's token. *)

val pad : Modality.spectrum -> ('b, 'd) spectrum -> ('b, 'd) spectrum
val encode_unpadded : ('b, 'd) t -> ('b, 'd) spectrum -> 'd Nx.int64_t
val nan_to_num : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
val normalisation : ('b, 'd) spectrum -> (float, 'b, 'd) Nx.t
val scale : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t

val resample :
  x:(float, 'b, 'd) Nx.t ->
  y:(float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t

val prepare :
  ('b, 'd) t -> ('b, 'd) spectrum -> (float, 'b, 'd) Nx.t * 'd Nx.int64_t

val grn :
  gamma:(float, 'b, 'd) Nx.t ->
  beta:(float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t

val latent :
  ('b, 'd) t -> ('b, 'd) spectrum -> (float, 'b, 'd) Nx.t * 'd Nx.int64_t

val codes : (float, 'b, 'd) Nx.t -> 'd Nx.int64_t
val values : (float, 'b) Nx.dtype -> 'd Nx.int64_t -> (float, 'b, 'd) Nx.t
