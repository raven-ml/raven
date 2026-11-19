(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's codecs, chosen by modality: data to tokens and back.

    A run picks its modalities when it starts, so the data of one modality is a
    value of {!data}, whatever its codec, and a set of codecs ({!t}) holds one
    codec for each modality it was loaded for. One image codec serves both
    surveys' images. Everything lies on the host. *)

(** {1:data Data} *)

(** The type for one modality's data of [b] objects. *)
type data =
  | Image of { bands : Image.band list; flux : Nx.host Nx.float32_t }
      (** Cutouts, [[|b; c; h; w|]], the [c] bands in the order of [bands],
          each pixel in its band's {!Image.unit}. Bands of the Hyper
          Suprime-Cam make {!Modality.Image_hsc}, the Legacy Survey's
          {!Modality.Image}. *)
  | Spectrum of {
      modality : Modality.spectrum;
      spectrum : (Nx.float32_elt, Nx.host) Spectrum.spectrum;
      lengths : Nx.host Nx.int64_t option;
          (** Each row's number of pixels ({!Spectrum.encode}). *)
    }  (** Spectra of one survey. *)
  | Scalar of { modality : Modality.scalar; values : Nx.host Nx.float32_t }
      (** Values, [[|b|]], or [[|b; 55|]] for a Gaia XP spectrum. *)

val modality : data -> Modality.t
(** [modality d] is [d]'s modality.

    Raises [Invalid_argument] for an image whose bands are of both surveys or of
    none. *)

val bands : Modality.t -> Image.band list
(** [bands m] is the bands of the survey of the image modality [m], in channel
    order: the Hyper Suprime-Cam's five for {!Modality.Image_hsc}, the Legacy
    Survey's four for any other. *)

(** {1:codecs Codecs} *)

type t
(** The type for sets of codecs. *)

val load : ?cache_dir:string -> ?offline:bool -> Modality.t list -> t
(** [load ms] is the codecs of the modalities [ms], each read by {!Hub.codec}
    with [cache_dir] and [offline]: aion-base's files, which aion-large's
    repository holds too, byte for byte.

    Raises [Invalid_argument] as {!Hub.file}, before any file is fetched, and
    [Failure] as {!Hub.codec} and each codec's [of_archive]. *)

val jit : t -> t
(** [jit t] is [t] with each codec compiled: {!Image.jit}, {!Spectrum.jit} and
    {!Scalar.jit}, whose programs live as long as the result. *)

val scalar : t -> Modality.scalar -> Nx.host Scalar.t
(** [scalar t m] is [m]'s codec.

    Raises [Invalid_argument] if [t] was not loaded for [Scalar m]. *)

(** {1:tokens Tokens} *)

val encode : t -> data -> Nx.host Tokens.t
(** [encode t d] is the tokens of [d]. A scalar modality is absent for an object
    one of whose values is NaN or infinite, all its tokens with it, unless that
    value's codebook holds NaN edges: such a value carries no measurement, and a
    codebook would give it a token of the distribution's edge. A codebook with
    NaN edges, as seven of HSC's, was trained with NaN as a value, and every
    value it encodes keeps its token ({!Scalar.encode}). Every other token is
    present.

    Raises [Invalid_argument] if [t] was not loaded for [d]'s modality, and as
    that codec's encode. *)

val decode : t -> Modality.t -> Nx.host Nx.int64_t -> data
(** [decode t m k] is the data of [m]'s tokens [k], [[|b; n|]], or [[|b|]] for a
    modality of one token: images in every band of their survey, in channel
    order; spectra on the codec's latent grid ({!Spectrum.grid}); a scalar's
    values, each the upper end of its token's bin ({!Scalar.decode}).

    Raises [Invalid_argument] if [t] was not loaded for [m], and as that
    codec's decode. *)
