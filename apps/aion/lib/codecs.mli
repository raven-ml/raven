(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's codecs, chosen by modality: data to tokens and back.

    A run picks its modalities when it starts, so the data of one modality is a
    value of {!data}, whatever its codec, and a set of codecs ({!t}) holds one
    codec for each modality it was loaded for. One image codec serves both
    surveys' images.

    A set of codecs is a structure ({!walk}), and so is data ({!walk_data}): a
    caller compiles {!encode} and {!decode} with [Rune.jit], the codecs an
    argument, as in
    [Rune.jit Nx.Ptree.(instantiate (module Codecs) @-> v walk_data @-> returns
     (v Tokens.walk)) Codecs.encode]. Each then computes as its codec does
    compiled. *)

(** {1:data Data} *)

(** The type for one modality's data of [b] objects on the device set ['d]. *)
type 'd data =
  | Image of { bands : Image.band list; flux : 'd Nx.float32_t }
      (** Cutouts, [[|b; c; h; w|]], the [c] bands in the order of [bands], each
          pixel in its band's {!Image.unit}. Bands of the Hyper Suprime-Cam make
          {!Modality.Image_hsc}, the Legacy Survey's {!Modality.Image}. *)
  | Spectrum of {
      modality : Modality.spectrum;
      spectrum : ('d Nx.float32_t, 'd) Spectrum.spectrum;
      lengths : 'd Nx.int64_t option;
          (** Each row's number of pixels ({!Spectrum.pad}). *)
    }  (** Spectra of one survey. *)
  | Scalar of { modality : Modality.scalar; values : 'd Nx.float32_t }
      (** Values, [[|b|]], or [[|b; 55|]] for a Gaia XP spectrum. *)

val walk_data : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd data -> 'e data
(** [walk_data c d] reports [d]'s case, its bands or its modality, and whether
    its lengths are given, and walks its arrays as fixed ones. *)

val modality : 'd data -> Modality.t
(** [modality d] is [d]'s modality.

    Raises [Invalid_argument] for an image whose bands are of both surveys or of
    none. *)

val bands : Modality.t -> Image.band list
(** [bands m] is the bands of the survey of the image modality [m], in channel
    order: the Hyper Suprime-Cam's five for {!Modality.Image_hsc}, the Legacy
    Survey's four for any other. *)

(** {1:codecs Codecs} *)

type ('a, 'd) t
(** The type for sets of codecs of weights ['a] on the device set ['d]. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c t] walks [t]'s codecs, reporting which it holds and the modality of
    each scalar codec. *)

val load :
  ?cache_dir:string ->
  ?offline:bool ->
  Modality.t list ->
  (Nx.host Nx.float32_t, Nx.host) t
(** [load ms] is the codecs of the modalities [ms], each read by {!Hub.codec}
    with [cache_dir] and [offline]: aion-base's files, which aion-large's
    repository holds too, byte for byte.

    Raises [Invalid_argument] as {!Hub.file}, before any file is fetched, and
    [Failure] as {!Hub.codec} and each codec's [of_archive]. *)

val scalar : ('a, 'd) t -> Modality.scalar -> 'd Scalar.t
(** [scalar t m] is [m]'s codec.

    Raises [Invalid_argument] if [t] was not loaded for [Scalar m]. *)

(** {1:tokens Tokens} *)

val encode : ('d Nx.float32_t, 'd) t -> 'd data -> 'd Tokens.t
(** [encode t d] is the tokens of [d], spectra padded to their modality's length
    ({!Spectrum.pad}). A scalar modality is absent for an object one of whose
    values is NaN or infinite, all its tokens with it, unless that value's
    codebook holds NaN edges: such a value carries no measurement, and a
    codebook would give it a token of the distribution's edge. A codebook with
    NaN edges, as seven of HSC's, was trained with NaN as a value, and every
    value it encodes keeps its token ({!Scalar.encode}). Presence is an array
    computed from the values, none read on the host, so that [encode]
    compiles; an image's or a spectrum's tokens are all present.

    Raises [Invalid_argument] if [t] was not loaded for [d]'s modality, as
    {!Spectrum.pad} for spectra, and as that codec's encode. *)

val decode : ('d Nx.float32_t, 'd) t -> Modality.t -> 'd Nx.int64_t -> 'd data
(** [decode t m k] is the data of [m]'s tokens [k], [[|b; n|]], or [[|b|]] for a
    modality of one token: images in every band of their survey, in channel
    order; spectra on the codec's latent grid ({!Spectrum.grid}); a scalar's
    values, each the upper end of its token's bin ({!Scalar.decode}).

    Raises [Invalid_argument] if [t] was not loaded for [m], and as that codec's
    decode. *)
