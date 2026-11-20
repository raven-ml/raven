(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The image and spectrum codecs against upstream (PolymathicAI/AION
   db7fba6df53b) with AION-1's weights, by the rounding ensemble's rule
   ([Spread]). The weights are 99 MB (spectrum) and 280 MB (image), and a run
   takes minutes.

   Each set has an ensemble file, AION_PARITY's ensemble/KIND/SET.safetensors,
   written by gen/KIND.py --weights aion-base: the set's rows of its source, the
   float64 reference's values (ref.NAME) and the ensemble's spreads at float32's
   scale (s.order.NAME) and float64's (s.order64.NAME). A set without its file
   fails. The image sets are golden (upstream's tests/test_data input, the
   Legacy Survey's channels), ls_desi, ls_catalog and hsc; the spectrum sets
   golden (unpadded), ls_desi, desi and sdss (rows trimmed of their -1 pads).

   Each set is checked twice: the codec as AION-1 ships it, against the spread
   at the order model's float32 scale, and the codec cast to float64, against
   float64's scale, where a wrong constant or structure shows by orders of
   magnitude. The checks are the rows below. *)

open Aion
open Part

(* A set's ensemble file and the size of its ensembles. *)
type set = { name : string; e : Nx.host Nx_io.Archive.t; n : int }

let ensemble kind ~name file =
  let p = path (Printf.sprintf "ensemble/%s/%s.safetensors" kind file) in
  if not (Sys.file_exists p) then begin
    fail ();
    printf "missing ensemble %s\n%!" p;
    None
  end
  else
    let e = Nx_io.load_safetensors p in
    Some { name; e; n = (ints (get Nx.int64 "n" e)).(0) }

let rows set = ints (get Nx.int64 "rows" set.e)
let ref_tokens set = get Nx.int64 "ref.tokens" set.e
let sub name = load ("subset/" ^ name ^ ".safetensors")

(* Rows *)

(* A row reads the reference's entry [name] and its spread at the scale [scale],
   each laid out as the output by [f]. *)
let entries (set, scale) f name =
  ( f (get Nx.float64 ("ref." ^ name) set.e),
    f (get Nx.float64 ("s." ^ scale ^ "." ^ name) set.e) )

let values c ?(f = floats) name x =
  let reference, spread = entries c f name in
  Spread.values ~n:(fst c).n ~reference ~spread x

let signs c ?(f = floats) name x =
  let reference, spread = entries c f name in
  Spread.signs ~n:(fst c).n ~reference ~spread x

let levels c ~edges name k =
  let reference, spread = entries c floats name in
  Spread.levels ~n:(fst c).n ~edges ~reference ~spread k

let mask_signs b = Array.map (fun b -> if b then 1. else -1.) (Nx.to_array b)

(* Each row's check of a set's outputs at a scale. *)
let check rows c o = List.iter (fun (name, row) -> report name (row c o)) rows

(* The count of tokens that differ from the reference's, and the first five's
   indices. *)
let differing got want =
  let d = ref [] in
  Array.iteri (fun i k -> if k <> want.(i) then d := i :: !d) got;
  let first = List.filteri (fun j _ -> j < 5) (List.rev !d) in
  printf "  tokens differing from the float64 reference: %d of %d%s\n"
    (List.length !d) (Array.length want)
    (String.concat "" (List.map (Printf.sprintf " (index %d)") first))

(* Each set whose ensemble file exists: [f set rung] checks it, [rung scale
   check] printing [scale]'s label and running [check (set, scale)]. *)
let run ~kind sets =
  let label = function
    | "order" -> "float32, order scale"
    | _ -> "float64, order scale at float64"
  in
  List.iter
    (fun (name, file, f) ->
      match ensemble kind ~name file with
      | None -> ()
      | Some set ->
          printf "\n== %s: %d objects, ensembles of %d\n%!" set.name
            (Array.length (rows set))
            set.n;
          let t0 = Unix.gettimeofday () in
          f set (fun scale check ->
              printf "%s:\n" (label scale);
              check (set, scale));
          printf "(%.0f s)\n%!" (Unix.gettimeofday () -. t0))
    sets;
  finish ()

(* Image *)

(* What the compiled codecs return too. *)
type image = { tokens : int array; (* [b; 576] *) decoded : float array }

(* The latent, [b; 576; 5], from upstream's [b; 5; 576]. *)
let cells t = floats (Nx.moveaxis 1 2 t)

let image_rows =
  [
    ( "token numbers",
      fun c o ->
        levels c ~edges:Decisions.digit_edges "bounded"
          (Decisions.digits o.tokens) );
    ("decoded", fun c o -> values c "decoded" o.decoded);
  ]

(* The rows of what eager computation alone shows: the latent. *)
let latent_rows = [ ("latent", fun c latent -> values c ~f:cells "pre" latent) ]

type image_set = {
  file : string;
  bands : Image.band list;
  images : int array -> Nx.host Nx.float32_t;
}

let ls = Image.[ Des_g; Des_r; Des_i; Des_z ]
let hsc = Image.[ Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y ]
let subset_images name key rows = take rows (get Nx.float32 key (sub name))

let hsc_images =
  { file = "hsc"; bands = hsc; images = subset_images "hsc" "image_array" }

let image_sets =
  (* Upstream's test feeds channels 5 to 8, the Legacy Survey's. *)
  let golden rows =
    let input = load "goldens/image_codec_input_batch.safetensors" in
    take rows (Nx.slice [ A; R (5, 9) ] (get Nx.float32 "image.array" input))
  in
  let ls_images file key =
    { file; bands = ls; images = subset_images file key }
  in
  [
    { file = "golden"; bands = ls; images = golden };
    ls_images "ls_desi" "legacysurvey.image_array";
    ls_images "ls_catalog" "image_array";
    hsc_images;
  ]

let image_outputs (type b) (t : ((float, b, Nx.host) Nx.t, Nx.host) Image.t)
    bands (x : (float, b, Nx.host) Nx.t) k =
  ( floats (Image.latent t bands x),
    {
      tokens = ints (Image.encode t bands x);
      decoded = floats (Image.decode t bands k);
    } )

let image () =
  let codec = Hub.codec ~cache_dir:(path "huggingface") ~offline:true Image in
  let t32 = Image.of_archive codec in
  let t64 = Nx.Ptree.cast (module Image) Nx.float64 t32 in
  run ~kind:"image"
    (List.map
       (fun s ->
         ( s.file,
           s.file,
           fun set rung ->
             let x = s.images (rows set) and k = ref_tokens set in
             let rung scale (latent, o) =
               rung scale (fun c ->
                   check latent_rows c latent;
                   check image_rows c o)
             in
             let o = image_outputs t32 s.bands x k in
             rung "order" o;
             differing (snd o).tokens (ints k);
             rung "order64" (image_outputs t64 s.bands (Nx.cast Nx.float64 x) k)
         ))
       image_sets)

(* Spectrum *)

(* What the compiled codecs return too. *)
type spectrum = {
  tokens : Nx.host Nx.int64_t; (* [b; 1 + n] *)
  flux : float array;
  mask : float array; (* 1 where bad, -1 elsewhere *)
}

(* The LFQ's input, [b; n; 10], from upstream's [b; 10; n]. *)
let lfq t = floats (Nx.transpose ~axes:[ 0; 2; 1 ] t)

(* [edges] is the codec's normalisation levels. *)
let spectrum_rows edges =
  [
    ( "normalisation token",
      fun c o ->
        levels c ~edges:(Array.sub edges 0 1023) "norm_pre"
          (ints (Nx.slice [ A; I 0 ] o.tokens)) );
    ( "LFQ bits",
      fun c o ->
        let k = Nx.slice [ A; R (1, Nx.dim 1 o.tokens) ] o.tokens in
        signs c ~f:lfq "pre" (floats (Lfq.decode Nx.float64 ~bits:10 k)) );
    ("flux", fun c o -> values c "flux" o.flux);
    ("mask", fun c o -> signs c "logit" o.mask);
  ]

(* The rows of what eager computation alone shows: the LFQ's input and the
   normalisation. *)
let prepared_rows =
  [
    ("LFQ input", fun c (pre, _) -> values c ~f:lfq "pre" pre);
    ("normalisation", fun c (_, norm) -> values c "norm_pre" norm);
  ]

let spectrum_of a pre rows =
  let f key = take rows (get Nx.float32 (pre ^ key) a) in
  Spectrum.
    {
      flux = f "flux";
      ivar = f "ivar";
      mask = Nx.not_equal_s (take rows (get Nx.uint8 (pre ^ "mask") a)) 0;
      wavelength = f "lambda";
    }

(* Each row cut to its width less its -1 wavelengths, as gen/spectrum.py cuts
   it, and padded as SDSS. *)
let trimmed (x : _ Spectrum.spectrum) =
  let pads = Nx.cast Nx.int64 (Nx.less_s x.wavelength 0.) in
  let width = Int64.of_int (Nx.dim 1 x.wavelength) in
  Spectrum.pad ~lengths:(Nx.rsub_s width (Nx.sum ~axes:[ 1 ] pads)) Sdss x

type spectrum_set = {
  label : string;
  source : string;
  modality : Modality.spectrum option; (* None: encoded unpadded *)
  spectra : int array -> (Nx.host Nx.float32_t, Nx.host) Spectrum.spectrum;
}

let subset_spectra name pre rows = spectrum_of (sub name) pre rows

let desi_spectra =
  {
    label = "desi";
    source = "desi";
    modality = Some Desi;
    spectra = subset_spectra "desi" "spectrum_";
  }

let spectrum_sets =
  let golden = "goldens/SPECTRUM_input_batch.safetensors" in
  [
    {
      label = "golden";
      source = "golden";
      modality = None;
      spectra = (fun rows -> spectrum_of (load golden) "spectrum." rows);
    };
    {
      label = "ls_desi";
      source = "ls_desi";
      modality = Some Desi;
      spectra = subset_spectra "ls_desi" "desi.spectrum_";
    };
    desi_spectra;
    (* The rows are padded here, after their -1 runs are cut. *)
    {
      label = "sdss (trimmed)";
      source = "sdss";
      modality = None;
      spectra = (fun rows -> trimmed (subset_spectra "sdss" "spectrum_" rows));
    };
  ]

let spectrum_outputs (type b)
    (t : ((float, b, Nx.host) Nx.t, Nx.host) Spectrum.t) modality
    (x : ((float, b, Nx.host) Nx.t, Nx.host) Spectrum.spectrum) k =
  let x = match modality with Some m -> Spectrum.pad m x | None -> x in
  let h, _ = Spectrum.latent t x in
  let d = Spectrum.decode t k in
  ( (floats h, floats (Spectrum.normalisation x)),
    {
      tokens = Spectrum.encode t x;
      flux = floats d.flux;
      mask = mask_signs d.mask;
    } )

let spectrum () =
  let t32 =
    Spectrum.of_archive
      (Hub.codec ~cache_dir:(path "huggingface") (Spectrum Desi))
  in
  let t64 = Nx.Ptree.cast (module Spectrum) Nx.float64 t32 in
  let checks = spectrum_rows (floats (Spectrum.levels t32)) in
  run ~kind:"spectrum"
    (List.map
       (fun s ->
         ( s.label,
           s.source,
           fun set rung ->
             let x = s.spectra (rows set) and k = ref_tokens set in
             let rung scale (prepared, o) =
               rung scale (fun c ->
                   check prepared_rows c prepared;
                   check checks c o)
             in
             let o = spectrum_outputs t32 s.modality x k in
             rung "order" o;
             differing (ints (snd o).tokens) (ints k);
             let x64 = Nx.Ptree.cast (module Spectrum.Spectra) Nx.float64 x in
             rung "order64" (spectrum_outputs t64 s.modality x64 k) ))
       spectrum_sets)
