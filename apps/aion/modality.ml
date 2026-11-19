(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type scalar =
  | Flux_g
  | Flux_r
  | Flux_i
  | Flux_z
  | Flux_w1
  | Flux_w2
  | Flux_w3
  | Flux_w4
  | Shape_r
  | Shape_e1
  | Shape_e2
  | Ebv
  | Z
  | A_g
  | A_r
  | A_i
  | A_z
  | A_y
  | Mag_g
  | Mag_r
  | Mag_i
  | Mag_z
  | Mag_y
  | Shape11
  | Shape22
  | Shape12
  | Flux_g_gaia
  | Flux_bp_gaia
  | Flux_rp_gaia
  | Parallax
  | Ra
  | Dec
  | Xp_bp
  | Xp_rp

type spectrum = Desi | Sdss

type t =
  | Image
  | Image_hsc
  | Segmap
  | Spectrum of spectrum
  | Catalog
  | Scalar of scalar

type codec =
  | Image_codec
  | Spectrum_codec
  | Segmap_codec
  | Catalog_codec
  | Scalar_codec

let scalars =
  [
    Flux_g;
    Flux_r;
    Flux_i;
    Flux_z;
    Flux_w1;
    Flux_w2;
    Flux_w3;
    Flux_w4;
    Shape_r;
    Shape_e1;
    Shape_e2;
    Ebv;
    Z;
    A_g;
    A_r;
    A_i;
    A_z;
    A_y;
    Mag_g;
    Mag_r;
    Mag_i;
    Mag_z;
    Mag_y;
    Shape11;
    Shape22;
    Shape12;
    Flux_g_gaia;
    Flux_bp_gaia;
    Flux_rp_gaia;
    Parallax;
    Ra;
    Dec;
    Xp_bp;
    Xp_rp;
  ]

let all =
  [ Image; Image_hsc; Segmap; Spectrum Desi; Spectrum Sdss; Catalog ]
  @ List.map (fun s -> Scalar s) scalars

(* Upstream's token keys (aion/modalities.py, token_key), which name the
   embeddings in model.safetensors and the domains of config.json. *)
let scalar_key = function
  | Flux_g -> "tok_flux_g"
  | Flux_r -> "tok_flux_r"
  | Flux_i -> "tok_flux_i"
  | Flux_z -> "tok_flux_z"
  | Flux_w1 -> "tok_flux_w1"
  | Flux_w2 -> "tok_flux_w2"
  | Flux_w3 -> "tok_flux_w3"
  | Flux_w4 -> "tok_flux_w4"
  | Shape_r -> "tok_shape_r"
  | Shape_e1 -> "tok_shape_e1"
  | Shape_e2 -> "tok_shape_e2"
  | Ebv -> "tok_ebv"
  | Z -> "tok_z"
  | A_g -> "tok_a_g"
  | A_r -> "tok_a_r"
  | A_i -> "tok_a_i"
  | A_z -> "tok_a_z"
  | A_y -> "tok_a_y"
  | Mag_g -> "tok_mag_g"
  | Mag_r -> "tok_mag_r"
  | Mag_i -> "tok_mag_i"
  | Mag_z -> "tok_mag_z"
  | Mag_y -> "tok_mag_y"
  | Shape11 -> "tok_shape11"
  | Shape22 -> "tok_shape22"
  | Shape12 -> "tok_shape12"
  | Flux_g_gaia -> "tok_flux_g_gaia"
  | Flux_bp_gaia -> "tok_flux_bp_gaia"
  | Flux_rp_gaia -> "tok_flux_rp_gaia"
  | Parallax -> "tok_parallax"
  | Ra -> "tok_ra"
  | Dec -> "tok_dec"
  | Xp_bp -> "tok_xp_bp"
  | Xp_rp -> "tok_xp_rp"

let key = function
  | Image -> "tok_image"
  | Image_hsc -> "tok_image_hsc"
  | Segmap -> "tok_segmap"
  | Spectrum Desi -> "tok_spectrum_desi"
  | Spectrum Sdss -> "tok_spectrum_sdss"
  | Catalog -> "catalog"
  | Scalar s -> scalar_key s

(* aion/fourm/modality_info.py, and the shapes of model.safetensors' token_emb
   and pos_emb at Hub.revision. The catalog's 50 is its max_tokens; its position
   table has 512 rows. *)
let tokens = function
  | Image | Image_hsc -> 576
  | Segmap -> 144
  | Spectrum _ -> 273
  | Catalog -> 50
  | Scalar (Xp_bp | Xp_rp) -> 55
  | Scalar _ -> 1

let vocab = function
  | Image | Image_hsc -> 4375
  | Segmap -> 1000
  | Catalog -> 3468
  | Scalar Z -> 1025
  | Spectrum _ | Scalar _ -> 1024

let codec = function
  | Image | Image_hsc -> Image_codec
  | Spectrum _ -> Spectrum_codec
  | Segmap -> Segmap_codec
  | Catalog -> Catalog_codec
  | Scalar _ -> Scalar_codec
