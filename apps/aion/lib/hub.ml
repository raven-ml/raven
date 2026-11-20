(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let repo = "polymathic-ai/aion-base"

(* The commit the PyTorch reference and upstream's goldens were checked
   against, 2026-11-18. *)
let revision = "40541618104bab0fa85c8af68daeb867a720bb8c"

(* Each scalar codec's directory under codecs/, upstream's name of its
   column. *)
let directory : Modality.scalar -> string = function
  | Flux_g -> "FLUX_G"
  | Flux_r -> "FLUX_R"
  | Flux_i -> "FLUX_I"
  | Flux_z -> "FLUX_Z"
  | Flux_w1 -> "FLUX_W1"
  | Flux_w2 -> "FLUX_W2"
  | Flux_w3 -> "FLUX_W3"
  | Flux_w4 -> "FLUX_W4"
  | Shape_r -> "SHAPE_R"
  | Shape_e1 -> "SHAPE_E1"
  | Shape_e2 -> "SHAPE_E2"
  | Ebv -> "EBV"
  | Z -> "Z"
  | A_g -> "a_g"
  | A_r -> "a_r"
  | A_i -> "a_i"
  | A_z -> "a_z"
  | A_y -> "a_y"
  | Mag_g -> "g_cmodel_mag"
  | Mag_r -> "r_cmodel_mag"
  | Mag_i -> "i_cmodel_mag"
  | Mag_z -> "z_cmodel_mag"
  | Mag_y -> "y_cmodel_mag"
  | Shape11 -> "i_sdssshape_shape11"
  | Shape22 -> "i_sdssshape_shape22"
  | Shape12 -> "i_sdssshape_shape12"
  | Flux_g_gaia -> "phot_g_mean_flux"
  | Flux_bp_gaia -> "phot_bp_mean_flux"
  | Flux_rp_gaia -> "phot_rp_mean_flux"
  | Parallax -> "parallax"
  | Ra -> "ra"
  | Dec -> "dec"
  | Xp_bp -> "bp_coefficients"
  | Xp_rp -> "rp_coefficients"

let file (m : Modality.t) =
  let dir =
    match m with
    | Image | Image_hsc -> "image"
    | Spectrum _ -> "spectrum"
    | Scalar s -> directory s
    | Segmap | Catalog ->
        invalid_arg
          (Printf.sprintf "Aion.Hub.file: %s has no codec here"
             (Modality.key m))
  in
  "codecs/" ^ dir ^ "/model.safetensors"

let codec ?cache_dir ?offline m =
  Nx_io.load_safetensors
    (Kaun_hf.download_file ?cache_dir ?offline ~revision ~file:(file m) repo)

type model = Base | Large

(* aion-large's commit, read on 2026-11-19; its codecs/ files equal
   aion-base's at revision, byte for byte. *)
let large = ("polymathic-ai/aion-large", "cfdb89e8a22ddbdbbc4fe65c30fa2eb6eb7f9bcd")

let weights ?cache_dir ?offline m =
  let repo, revision = match m with Base -> (repo, revision) | Large -> large in
  Nx_io.load_safetensors
    (Kaun_hf.download_file ?cache_dir ?offline ~revision ~file:"model.safetensors"
       repo)
