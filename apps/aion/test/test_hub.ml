(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    Sys.mkdir dir 0o755
  end

(* A cache holding [m]'s codec file where Kaun_hf caches the repository's file
   at Hub's revision. *)
let cache_with m archive =
  let cache_dir = Filename.temp_dir "aion" "hub" in
  let path =
    Kaun_hf.cache_path ~cache_dir ~revision:Hub.revision ~file:(Hub.file m)
      Hub.repo
  in
  mkdir_p (Filename.dirname path);
  Nx_io.save_safetensors path archive;
  cache_dir

let edges = Nx.create Nx.float32 [| 3 |] [| -1.; 0.5; 2. |]

(* The codecs' directories under codecs/ in the repository at Hub.revision. *)
let files =
  cases "file is the codec's path in the repository" ~id:fst
    Modality.
      [
      ("tok_image", (Image, "codecs/image/model.safetensors"));
      ("tok_image_hsc", (Image_hsc, "codecs/image/model.safetensors"));
      ("tok_spectrum_desi", (Spectrum Desi, "codecs/spectrum/model.safetensors"));
      ("tok_spectrum_sdss", (Spectrum Sdss, "codecs/spectrum/model.safetensors"));
      ("tok_flux_g", (Scalar Flux_g, "codecs/FLUX_G/model.safetensors"));
      ("tok_z", (Scalar Z, "codecs/Z/model.safetensors"));
      ("tok_a_g", (Scalar A_g, "codecs/a_g/model.safetensors"));
      ("tok_mag_y", (Scalar Mag_y, "codecs/y_cmodel_mag/model.safetensors"));
      ( "tok_shape12",
        (Scalar Shape12, "codecs/i_sdssshape_shape12/model.safetensors") );
      ( "tok_flux_bp_gaia",
        (Scalar Flux_bp_gaia, "codecs/phot_bp_mean_flux/model.safetensors") );
      ("tok_parallax", (Scalar Parallax, "codecs/parallax/model.safetensors"));
      ("tok_xp_rp", (Scalar Xp_rp, "codecs/rp_coefficients/model.safetensors"));
      ]
    (fun (_, (m, file)) -> equal string file (Hub.file m))

let no_codec =
  cases "file refuses a modality without a codec" ~id:Modality.key
    Modality.[ Segmap; Catalog ]
    (fun m ->
      raises_match (Exn.invalid_arg ~sub:(Modality.key m)) (fun () ->
          Hub.file m))

let cached =
  test "codec reads the codec's file at the revision" (fun () ->
      let entry = "_quantizer._index_to_val" in
      let m = Modality.Scalar Flux_g in
      let cache_dir =
        cache_with m (Nx_io.Archive.of_list [ (entry, Nx.P edges) ])
      in
      let a = Hub.codec ~cache_dir ~offline:true m in
      equal (list string) [ entry ] (Nx_io.Archive.names a);
      let read = Nx_io.Archive.tensor ~shape:[| 3 |] Nx.float32 entry a in
      equal (array float_exact) (Nx.to_array edges) (Nx.to_array read))

let missing =
  test "codec raises offline without the file" (fun () ->
      let cache_dir = Filename.temp_dir "aion" "empty" in
      raises_match Exn.failure (fun () ->
          Hub.codec ~cache_dir ~offline:true (Scalar Flux_g)))

(* Each model's weights at its repository's commit. *)
let models =
  cases "weights reads each model's file at its commit"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("base", Hub.Base, Hub.repo, Hub.revision);
      ( "large",
        Hub.Large,
        "polymathic-ai/aion-large",
        "cfdb89e8a22ddbdbbc4fe65c30fa2eb6eb7f9bcd" );
    ]
    (fun (_, m, repo, revision) ->
      let cache_dir = Filename.temp_dir "aion" "weights" in
      let path =
        Kaun_hf.cache_path ~cache_dir ~revision ~file:"model.safetensors" repo
      in
      mkdir_p (Filename.dirname path);
      Nx_io.save_safetensors path
        (Nx_io.Archive.of_list [ ("w", Nx.P edges) ]);
      let a = Hub.weights ~cache_dir ~offline:true m in
      equal (list string) [ "w" ] (Nx_io.Archive.names a);
      let empty = Filename.temp_dir "aion" "empty" in
      raises_match (Exn.failure ~sub:repo) (fun () ->
          Hub.weights ~cache_dir:empty ~offline:true m))

let () =
  exit
    (run "aion.hub"
       [
         group "codec" [ files; no_codec; cached; missing ];
         models;
       ])
