(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

(* aion-base's token embeddings at Hub.revision, from model.safetensors' header:
   each key with its pos_emb's rows and its token_emb's rows. *)
let header =
  [
    ("catalog", 512, 3468);
    ("tok_a_g", 1, 1024);
    ("tok_a_i", 1, 1024);
    ("tok_a_r", 1, 1024);
    ("tok_a_y", 1, 1024);
    ("tok_a_z", 1, 1024);
    ("tok_dec", 1, 1024);
    ("tok_ebv", 1, 1024);
    ("tok_flux_bp_gaia", 1, 1024);
    ("tok_flux_g", 1, 1024);
    ("tok_flux_g_gaia", 1, 1024);
    ("tok_flux_i", 1, 1024);
    ("tok_flux_r", 1, 1024);
    ("tok_flux_rp_gaia", 1, 1024);
    ("tok_flux_w1", 1, 1024);
    ("tok_flux_w2", 1, 1024);
    ("tok_flux_w3", 1, 1024);
    ("tok_flux_w4", 1, 1024);
    ("tok_flux_z", 1, 1024);
    ("tok_image", 576, 4375);
    ("tok_image_hsc", 576, 4375);
    ("tok_mag_g", 1, 1024);
    ("tok_mag_i", 1, 1024);
    ("tok_mag_r", 1, 1024);
    ("tok_mag_y", 1, 1024);
    ("tok_mag_z", 1, 1024);
    ("tok_parallax", 1, 1024);
    ("tok_ra", 1, 1024);
    ("tok_segmap", 144, 1000);
    ("tok_shape11", 1, 1024);
    ("tok_shape12", 1, 1024);
    ("tok_shape22", 1, 1024);
    ("tok_shape_e1", 1, 1024);
    ("tok_shape_e2", 1, 1024);
    ("tok_shape_r", 1, 1024);
    ("tok_spectrum_desi", 273, 1024);
    ("tok_spectrum_sdss", 273, 1024);
    ("tok_xp_bp", 55, 1024);
    ("tok_xp_rp", 55, 1024);
    ("tok_z", 1, 1025);
  ]

let keys =
  test "the modalities are the 40 of the weights" (fun () ->
      let sorted l = List.sort compare l in
      equal (list string)
        (sorted (List.map (fun (k, _, _) -> k) header))
        (sorted (List.map Modality.key Modality.all)))

let facts =
  cases "a modality's tokens and vocabulary are its embeddings'"
    ~id:Modality.key Modality.all (fun m ->
      let _, positions, vocab =
        List.find (fun (k, _, _) -> k = Modality.key m) header
      in
      equal int vocab (Modality.vocab m);
      (* The catalog's position table has 512 rows, of which upstream's
         max_tokens uses 50 (aion/fourm/modality_info.py). *)
      let expected = if m = Catalog then 50 else positions in
      equal int expected (Modality.tokens m))

let scalars =
  test "the scalars are all's scalar modalities" (fun () ->
      equal (list string)
        (List.filter_map
           (function
             | Modality.Scalar _ as m -> Some (Modality.key m) | _ -> None)
           Modality.all)
        (List.map (fun s -> Modality.key (Scalar s)) Modality.scalars))

let () =
  exit
    (run "aion.modality" [ group "modality" [ keys; facts; scalars ] ])
