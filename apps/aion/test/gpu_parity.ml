(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AION-1's transformer compiled on a GPU against eager computation on the
   host and upstream's float64 reference. It lives outside runtest: it reads
   AION-1's weights (1.27 GB) and the PyTorch references, which no test run
   holds; test_gpu.exe runs the narrow fixtures on the GPU inside runtest. Run
   it as

   gpu_parity.exe DEVICE HUB REF64

   with DEVICE a GPU as Rig_gpus.open_ names it (cuda, nv, amd, metal), HUB a
   Hub cache holding aion-base's model.safetensors at Hub.revision, and REF64
   the float64 reference gen/reference.py writes (--precision f64): its
   model.safetensors and codecs.safetensors. On the first 2 of the
   reference's 8 objects it runs the decodes of redshift from photometry and
   of a DESI spectrum from an image, one encode of Legacy Survey and DESI
   inputs, and a one-step draw of that spectrum at temperature 0. For eager on
   the host and compiled on DEVICE it prints the largest absolute difference
   from the reference and that over the reference's largest magnitude, and
   the argmax tokens that differ from the reference's and from eager's. It
   gates nothing and exits 0 once every part ran. *)

open Aion

let printf = Printf.printf
let host x = Nx.place Nx.Host.on x

let get dt name a =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let max_abs x = Nx.item [] (Nx.max (Nx.abs x))

let error x r =
  let d = max_abs (Nx.sub (Nx.cast Nx.float64 (host x)) r) in
  (d, d /. max_abs r)

let differing x r =
  let a = Nx.argmax ~axis:(-1) (Nx.cast Nx.float64 (host x))
  and b = Nx.argmax ~axis:(-1) (Nx.cast Nx.float64 (host r)) in
  Int64.to_int (Nx.item [] (Nx.sum (Nx.cast Nx.int64 (Nx.not_equal a b))))

let report name x r =
  let d, n = error x r in
  printf "  %-24s abs %.3g norm %.3g" name d n;
  if Nx.ndim r = 3 then
    printf " argmax %d/%d differ" (differing x r) (Nx.numel r / Nx.dim (-1) r);
  printf "\n%!"

let objects = 2

let scalars =
  Modality.
    [ Flux_g; Flux_r; Flux_i; Flux_z; Flux_w1; Flux_w2; Flux_w3; Flux_w4; Ebv ]

let run at ~hub ~ref64 =
  let file name = Nx_io.load_safetensors (Filename.concat ref64 name) in
  let r64 = file "model.safetensors" and codecs = file "codecs.safetensors" in
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
  let whole m : _ Transformer.target = { modality = m; positions = None } in
  let logits name ms target reference =
    printf "%s\n%!" name;
    let x = List.map input ms in
    let r = rows (get Nx.float64 reference r64) in
    let l = List.hd (Transformer.forward t x [ whole target ]) in
    let lg = List.hd (forward g (List.map on x) [ whole target ]) in
    report "eager" l r;
    report "compiled on the device" lg r;
    printf "  argmax: device %d differ from eager\n%!" (differing lg l)
  in
  logits "redshift from photometry"
    (Modality.Image
    :: List.map
         (fun s -> Modality.Scalar s)
         Modality.[ Flux_g; Flux_r; Flux_i; Flux_z ])
    (Scalar Z) "logits.z_from_phot.upstream";
  logits "a DESI spectrum from an image" [ Modality.Image ] (Spectrum Desi)
    "logits.desi_from_image.patched";
  printf "Legacy Survey and DESI encoded\n%!";
  let x =
    List.map input
      (Modality.Image :: Modality.Spectrum Desi
      :: List.map (fun s -> Modality.Scalar s) scalars)
  in
  let r = rows (get Nx.float64 "embed.ls_desi.all" r64) in
  report "eager" (Transformer.encode t x).context r;
  report "compiled on the device"
    (encode g (List.map on x)).context
    r;
  printf "a DESI spectrum drawn from an image, one step, temperature 0\n%!";
  let target = Modality.Spectrum Desi and key = Nx.Rng.key 7 in
  let reveal = [ Modality.tokens target ] and x = [ input Image ] in
  let k = Roar.sample ~key ~temperature:0. ~reveal t x target in
  let draw () = Nx.Ptree.v Roar.walk_draw in
  let start =
    Rune.jit
      Nx.Ptree.(Nx.Rng.ptree @-> inputs () @-> returns (draw ()))
      (fun key x -> Roar.start ~key ~temperature:0. x target)
  and step =
    Rune.jit
      Nx.Ptree.(model () @-> v Walk.int @-> draw () @-> returns (draw ()))
      Roar.step
  in
  let d = start (Nx.Rng.place at key) (List.map on x) in
  let kg = Roar.tokens (List.fold_left (fun d k -> step g k d) d reveal) in
  printf "  device %d of %d tokens differ from eager\n%!"
    (Int64.to_int
       (Nx.item [] (Nx.sum (Nx.cast Nx.int64 (Nx.not_equal k (host kg))))))
    (Nx.numel k)

let () =
  match Sys.argv with
  | [| _; device; hub; ref64 |] ->
      let gpu = Result.error_to_failure (Rig_gpus.open_ device) in
      let module D = (val Nx.devices [ gpu ]) in
      run D.on ~hub ~ref64
  | _ ->
      prerr_endline "usage: gpu_parity.exe DEVICE HUB REF64";
      exit 2
