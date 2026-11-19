(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's modalities: the kinds of data its transformer reads and writes.

    A {e modality} is one kind of measurement of an object, such as a Legacy
    Survey image or a Gaia parallax. A codec turns a modality's data into
    {e tokens}, integers below the modality's {e vocabulary} size, and the
    transformer reads and predicts them, each modality with its own token
    embedding. AION-1 knows the 40 modalities of {!all}, each named in the
    transformer's weights by its {e key}, such as ["tok_flux_g"].

    A {!scalar} modality is one number per object, or the 55 coefficients of a
    Gaia XP spectrum, each tokenized by one codebook ({!Scalar}). A
    {!spectrum} modality is the spectra of one survey, all tokenized by one
    codec ({!Spectrum}). *)

(** {1:modalities Modalities} *)

(** The type for scalar modalities. *)
type scalar =
  | Flux_g  (** Legacy Survey flux in g, ["tok_flux_g"]. *)
  | Flux_r  (** Legacy Survey flux in r, ["tok_flux_r"]. *)
  | Flux_i  (** Legacy Survey flux in i, ["tok_flux_i"]. *)
  | Flux_z  (** Legacy Survey flux in z, ["tok_flux_z"]. *)
  | Flux_w1  (** Legacy Survey flux in WISE W1, ["tok_flux_w1"]. *)
  | Flux_w2  (** Legacy Survey flux in WISE W2, ["tok_flux_w2"]. *)
  | Flux_w3  (** Legacy Survey flux in WISE W3, ["tok_flux_w3"]. *)
  | Flux_w4  (** Legacy Survey flux in WISE W4, ["tok_flux_w4"]. *)
  | Shape_r  (** Legacy Survey half-light radius, ["tok_shape_r"]. *)
  | Shape_e1  (** Legacy Survey ellipticity e1, ["tok_shape_e1"]. *)
  | Shape_e2  (** Legacy Survey ellipticity e2, ["tok_shape_e2"]. *)
  | Ebv  (** Legacy Survey E(B-V), ["tok_ebv"]. *)
  | Z  (** Redshift, ["tok_z"]. *)
  | A_g  (** HSC extinction in g, ["tok_a_g"]. *)
  | A_r  (** HSC extinction in r, ["tok_a_r"]. *)
  | A_i  (** HSC extinction in i, ["tok_a_i"]. *)
  | A_z  (** HSC extinction in z, ["tok_a_z"]. *)
  | A_y  (** HSC extinction in y, ["tok_a_y"]. *)
  | Mag_g  (** HSC cmodel magnitude in g, ["tok_mag_g"]. *)
  | Mag_r  (** HSC cmodel magnitude in r, ["tok_mag_r"]. *)
  | Mag_i  (** HSC cmodel magnitude in i, ["tok_mag_i"]. *)
  | Mag_z  (** HSC cmodel magnitude in z, ["tok_mag_z"]. *)
  | Mag_y  (** HSC cmodel magnitude in y, ["tok_mag_y"]. *)
  | Shape11  (** HSC i-band SDSS shape moment 11, ["tok_shape11"]. *)
  | Shape22  (** HSC i-band SDSS shape moment 22, ["tok_shape22"]. *)
  | Shape12  (** HSC i-band SDSS shape moment 12, ["tok_shape12"]. *)
  | Flux_g_gaia  (** Gaia G mean flux, ["tok_flux_g_gaia"]. *)
  | Flux_bp_gaia  (** Gaia BP mean flux, ["tok_flux_bp_gaia"]. *)
  | Flux_rp_gaia  (** Gaia RP mean flux, ["tok_flux_rp_gaia"]. *)
  | Parallax  (** Gaia parallax, ["tok_parallax"]. *)
  | Ra  (** Right ascension, ["tok_ra"]. *)
  | Dec  (** Declination, ["tok_dec"]. *)
  | Xp_bp  (** Gaia XP BP coefficients, 55, ["tok_xp_bp"]. *)
  | Xp_rp  (** Gaia XP RP coefficients, 55, ["tok_xp_rp"]. *)

(** The type for spectrum modalities. *)
type spectrum =
  | Desi  (** DESI spectrum, ["tok_spectrum_desi"]. *)
  | Sdss  (** SDSS spectrum, ["tok_spectrum_sdss"]. *)

(** The type for modalities. *)
type t =
  | Image  (** Legacy Survey image, ["tok_image"]. *)
  | Image_hsc  (** HSC image, ["tok_image_hsc"]. *)
  | Segmap  (** Legacy Survey segmentation map, ["tok_segmap"]. *)
  | Spectrum of spectrum  (** A spectrum modality. *)
  | Catalog  (** Legacy Survey catalog of neighbours, ["catalog"]. *)
  | Scalar of scalar  (** A scalar modality. *)

(** The type for the kinds of codec. *)
type codec =
  | Image_codec  (** Images of both surveys. *)
  | Spectrum_codec  (** Spectra of both surveys. *)
  | Segmap_codec  (** Segmentation maps. *)
  | Catalog_codec  (** Catalogs. *)
  | Scalar_codec  (** Scalar modalities, one codec each. *)

val all : t list
(** [all] is AION-1's 40 modalities. *)

val scalars : scalar list
(** [scalars] is the 34 scalar modalities, [all]'s [Scalar _] in its order. *)

(** {1:facts Facts} *)

val key : t -> string
(** [key m] is [m]'s name in the transformer's weights and configuration. *)

val tokens : t -> int
(** [tokens m] is the number of tokens of one object's [m]: [576] for an image,
    [144] for a segmentation map, [273] for a spectrum, [55] for an XP spectrum,
    [1] for another scalar modality, and at most [50] for a catalog. *)

val vocab : t -> int
(** [vocab m] is the number of rows of [m]'s token embedding in the transformer:
    [4375] for an image, [1000] for a segmentation map, [3468] for a catalog,
    [1025] for redshift and [1024] for the others. *)

val codec : t -> codec
(** [codec m] is the kind of codec that tokenizes [m]. *)
