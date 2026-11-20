(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AION-1's codecs and transformer compiled on a GPU, DEVICE as Rig_gpus.open_
   names it or else this machine's first, against upstream's float64 reference.
   test_gpu.exe runs the narrow fixtures on the GPU inside runtest.

   The codecs first: the set of the HSC image and DESI spectrum codecs, compiled
   on the GPU as main.exe compiles it, encodes the objects of the ensemble sets
   hsc and desi and decodes their reference tokens. Its checks are the image and
   spectrum parts' shared rows at float32's scale; the rows of what eager
   computation alone shows (the latent, the LFQ's input, the normalisation) are
   not checked. These checks gate.

   Then the transformer, against AION_PARITY's reference/f64, written by
   gen/transformer.py --weights aion-base --precision f64: on the first 2 of its
   8 objects, the decodes of redshift from photometry and of a DESI spectrum
   from an image, one encode of Legacy Survey and DESI inputs, and a one-step
   draw of that spectrum at temperature 0. For eager on the host and compiled on
   the GPU it prints the largest absolute difference from the reference and that
   over the reference's largest magnitude, and the argmax tokens that differ
   from the reference's and from eager's. These gate nothing. *)

open Aion
open Part

(* The codecs *)

let codecs at =
  let ms = Modality.[ Image_hsc; Spectrum Desi ] in
  let h = Codecs.load ~cache_dir:(path "huggingface") ~offline:true ms in
  let g = Nx.Ptree.place (module Codecs) at h in
  let encode, decode =
    let open Nx.Ptree in
    let set () = instantiate (module Codecs) and data () = v Codecs.walk_data in
    ( Rune.jit (set () @-> data () @-> returns (v Tokens.walk)) Codecs.encode g,
      Rune.jit
        (set () @-> v Modality.walk @-> tensor @-> returns (data ()))
        Codecs.decode g )
  in
  let encode d = encode (Nx.Ptree.place_fixed Codecs.walk_data at d)
  and decode m k =
    Nx.Ptree.place_fixed Codecs.walk_data Nx.Host.on (decode m (Nx.place at k))
  in
  let set kind file f =
    match Codec.ensemble kind ~name:file file with
    | None -> ()
    | Some set ->
        printf "codecs, %s: %d objects\n%!" file (Array.length (Codec.rows set));
        f set (Codec.rows set) (Codec.ref_tokens set)
  in
  set "image" "hsc" (fun set rows k ->
      let bands = Codec.hsc_images.bands
      and flux = Codec.hsc_images.images rows in
      let tokens = (encode (Image { bands; flux })).tokens in
      match decode Image_hsc k with
      | Image { flux; _ } ->
          Codec.check Codec.image_rows (set, "order")
            { tokens = ints tokens; decoded = floats flux }
      | _ -> failwith "not an image");
  set "spectrum" "desi" (fun set rows k ->
      let spectrum = Codec.desi_spectra.spectra rows in
      let tokens =
        (encode (Spectrum { modality = Desi; spectrum; lengths = None })).tokens
      in
      let levels = Spectrum.levels (Codecs.spectrum h Desi) in
      match decode (Spectrum Desi) k with
      | Spectrum { spectrum = d; _ } ->
          Codec.check
            (Codec.spectrum_rows (floats levels))
            (set, "order")
            {
              tokens = host tokens;
              flux = floats d.flux;
              mask = Codec.mask_signs (host d.mask);
            }
      | _ -> failwith "not a spectrum")

(* The transformer *)

let max_abs x = Nx.item [] (Nx.max (Nx.abs x))

let differing x r =
  let a = Nx.argmax ~axis:(-1) (Nx.cast Nx.float64 (host x))
  and b = Nx.argmax ~axis:(-1) (Nx.cast Nx.float64 (host r)) in
  Int64.to_int (Nx.item [] (Nx.sum (Nx.cast Nx.int64 (Nx.not_equal a b))))

let distance name x r =
  let d = max_abs (Nx.sub (Nx.cast Nx.float64 (host x)) r) in
  printf "  %-24s abs %.3g norm %.3g" name d (d /. max_abs r);
  if Nx.ndim r = 3 then
    printf " argmax %d/%d differ" (differing x r) (Nx.numel r / Nx.dim (-1) r);
  printf "\n%!"

let objects = 2
let fluxes = Modality.[ Flux_g; Flux_r; Flux_i; Flux_z ]
let scalars = fluxes @ Modality.[ Flux_w1; Flux_w2; Flux_w3; Flux_w4; Ebv ]

(* Each decode: its name, its inputs, its target and the reference's logits. *)
let decodes =
  Modality.
    [
      ( "redshift from photometry",
        Image :: List.map (fun s -> Scalar s) fluxes,
        Scalar Z,
        "logits.z_from_phot.upstream" );
      ( "a DESI spectrum from an image",
        [ Image ],
        Spectrum Desi,
        "logits.desi_from_image.patched" );
    ]

let transformer at =
  let r64 = load "reference/f64/model.safetensors" in
  let codecs = load "reference/f64/codecs.safetensors" in
  let rows x = Nx.slice [ Nx.R (0, objects) ] x in
  let input m : _ Tokens.t =
    let key = Modality.key m in
    match Nx_io.Archive.find ("ls_desi." ^ key ^ ".tokens") codecs with
    | Some (Nx.P t) ->
        let shape = [| objects; Modality.tokens m |] in
        let tokens = Nx.reshape shape (rows (Nx.cast Nx.int64 t)) in
        { modality = m; tokens; present = Nx.ones Nx.bool shape }
    | None -> failwith ("no tokens ls_desi." ^ key)
  in
  let on = Nx.Ptree.place_fixed Tokens.walk at in
  let hub = path "huggingface" in
  let t =
    Transformer.of_archive (Hub.weights ~cache_dir:hub ~offline:true Base)
  in
  let g = Nx.Ptree.place (module Transformer) at t in
  let model () = Nx.Ptree.instantiate (module Transformer) in
  let inputs () = Nx.Ptree.(list (v Tokens.walk)) in
  let encode =
    Rune.jit
      Nx.Ptree.(
        model () @-> inputs ()
        @-> returns (instantiate (module Transformer.Encoding)))
      Transformer.encode
  and forward =
    Rune.jit
      Nx.Ptree.(
        model () @-> inputs ()
        @-> list (v Transformer.walk_target)
        @-> returns (list tensor))
      Transformer.forward
  in
  List.iter
    (fun (name, ms, target, reference) ->
      printf "%s\n%!" name;
      let x = List.map input ms and r = rows (get Nx.float64 reference r64) in
      let whole : _ Transformer.target =
        { modality = target; positions = None }
      in
      let l = List.hd (Transformer.forward t x [ whole ]) in
      let lg = List.hd (forward g (List.map on x) [ whole ]) in
      distance "eager" l r;
      distance "compiled on the device" lg r;
      printf "  argmax: device %d differ from eager\n%!" (differing lg l))
    decodes;
  printf "Legacy Survey and DESI encoded\n%!";
  let x =
    List.map input
      (Modality.Image :: Spectrum Desi
      :: List.map (fun s -> Modality.Scalar s) scalars)
  in
  let r = rows (get Nx.float64 "embed.ls_desi.all" r64) in
  distance "eager" (Transformer.encode t x).context r;
  distance "compiled on the device" (encode g (List.map on x)).context r;
  printf "a DESI spectrum drawn from an image, one step, temperature 0\n%!";
  let target = Modality.Spectrum Desi in
  let keys = Nx.Rng.split_batch ~n:objects (Nx.Rng.key 7) in
  let reveal = [ Modality.tokens target ] and x = [ input Image ] in
  let k = Roar.sample ~keys ~temperature:0. ~reveal t x target in
  let draw () = Nx.Ptree.v Roar.walk_draw in
  let start =
    Rune.jit
      Nx.Ptree.(Nx.Rng.ptree @-> inputs () @-> returns (draw ()))
      (fun keys x -> Roar.start ~keys ~temperature:0. x target)
  and step =
    Rune.jit
      Nx.Ptree.(model () @-> v Walk.int @-> draw () @-> returns (draw ()))
      Roar.step
  in
  let d = start (Nx.Rng.place at keys) (List.map on x) in
  let kg = Roar.tokens (List.fold_left (fun d k -> step g k d) d reveal) in
  printf "  device %d of %d tokens differ from eager\n%!"
    (Int64.to_int
       (Nx.item [] (Nx.sum (Nx.cast Nx.int64 (Nx.not_equal k (host kg))))))
    (Nx.numel k)

let run device =
  let gpu =
    match device with None -> Rig_gpus.first () | Some d -> Rig_gpus.open_ d
  in
  match gpu with
  | Error why ->
      prerr_endline ("parity: no GPU: " ^ why);
      exit 1
  | Ok gpu ->
      let module D = (val Nx.devices [ gpu ]) in
      codecs D.on;
      transformer D.on;
      finish ()
