(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AION-1's codecs and transformer compiled on a GPU against eager computation
   on the host and upstream's float64 reference. It lives outside runtest: it
   reads AION-1's weights (1.27 GB) and the PyTorch references, which no test
   run holds; test_gpu.exe runs the narrow fixtures on the GPU inside runtest.
   Run it as

   AION_PARITY=DIR gpu_parity.exe DEVICE HUB REF64

   with DEVICE a GPU as Rig_gpus.open_ names it (cuda, nv, amd, metal), HUB a
   Hub cache holding aion-base at Hub.revision, its model.safetensors and its
   codecs, REF64 the float64 reference gen/reference.py writes (--precision
   f64): its model.safetensors and codecs.safetensors, and DIR the parity parts'
   directory, laid out as the README's Tests section says.

   The codecs first: the set of the HSC image and DESI spectrum codecs, compiled
   on DEVICE as main.exe compiles it, encodes the objects of DIR's ensemble sets
   hsc and desi and decodes their reference tokens. Each value lies within τ ŝ
   of upstream's float64 reference, ŝ the ensemble's spread at float32's scale,
   decisions near a boundary excepted, as image_parity and spectrum_parity hold
   the eager codecs. These checks gate: it exits 1 when one fails.

   Then the transformer: on the first 2 of the reference's 8 objects it runs the
   decodes of redshift from photometry and of a DESI spectrum from an image, one
   encode of Legacy Survey and DESI inputs, and a one-step draw of that spectrum
   at temperature 0. For eager on the host and compiled on DEVICE it prints the
   largest absolute difference from the reference and that over the reference's
   largest magnitude, and the argmax tokens that differ from the reference's and
   from eager's. These gate nothing. *)

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

(* The codecs *)

let failures = ref 0
let floats x = Nx.to_array (Nx.cast Nx.float64 (host x))
let ints x = Array.map Int64.to_int (Nx.to_array (Nx.cast Nx.int64 (host x)))
let signs b = Array.map (fun b -> if b then 1. else -1.) (Nx.to_array (host b))

let check name (c : Spread.check) =
  if c.failures <> [] then incr failures;
  printf "  %s%s\n%!"
    (if c.failures = [] then "" else "FAIL ")
    (Format.asprintf "%a" (Spread.pp name) c)

let codecs at ~hub dir =
  let path = Filename.concat dir and load = Nx_io.load_safetensors in
  let ms = Modality.[ Image_hsc; Spectrum Desi ] in
  let g =
    Nx.Ptree.place
      (module Codecs)
      at
      (Codecs.load ~cache_dir:hub ~offline:true ms)
  in
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
  (* A set's ensemble, its size, its rows of the subset, the subset, the
     reference's values and the spreads at float32's scale. *)
  let set kind name =
    let e = load (path ("ensemble/" ^ kind ^ "/" ^ name ^ ".safetensors")) in
    let rows = get Nx.int64 "rows" e in
    printf "codecs, %s: %d objects\n%!" name (Nx.dim 0 rows);
    let r key = floats (get Nx.float64 ("ref." ^ key) e)
    and s key = floats (get Nx.float64 ("s.order." ^ key) e) in
    let a = load (path ("subset/" ^ name ^ ".safetensors")) in
    (e, (ints (get Nx.int64 "n" e)).(0), rows, a, r, s)
  in
  let e, n, rows, a, r, s = set "image" "hsc" in
  let sub dt key = Nx.take ~axis:0 rows (get dt key a) in
  let flux = sub Nx.float32 "image_array" in
  let k = (encode (Image { bands = Codecs.bands Image_hsc; flux })).tokens in
  check "token digits"
    (Spread.levels ~n ~edges:Decisions.digit_edges ~reference:(r "bounded")
       ~spread:(s "bounded")
       (Decisions.digits (ints k)));
  (match decode Image_hsc (get Nx.int64 "ref.tokens" e) with
  | Image { flux; _ } ->
      check "decoded"
        (Spread.values ~n ~reference:(r "decoded") ~spread:(s "decoded")
           (floats flux))
  | _ -> failwith "not an image");
  let e, n, rows, a, r, s = set "spectrum" "desi" in
  let sub dt key = Nx.take ~axis:0 rows (get dt key a) in
  let f key = sub Nx.float32 ("spectrum_" ^ key) in
  let x : _ Spectrum.spectrum =
    {
      flux = f "flux";
      ivar = f "ivar";
      mask = Nx.not_equal_s (sub Nx.uint8 "spectrum_mask") 0;
      wavelength = f "lambda";
    }
  in
  let k =
    host
      (encode (Spectrum { modality = Desi; spectrum = x; lengths = None }))
        .tokens
  in
  let levels =
    Spectrum.levels
      (Spectrum.of_archive
         (Hub.codec ~cache_dir:hub ~offline:true (Spectrum Desi)))
  in
  (* The LFQ's input, [b; n; 10], from upstream's [b; 10; n]. *)
  let lfq key =
    floats (Nx.transpose ~axes:[ 0; 2; 1 ] (get Nx.float64 key e))
  in
  check "normalisation token"
    (Spread.levels ~n
       ~edges:(Array.sub (floats levels) 0 1023)
       ~reference:(r "norm_pre") ~spread:(s "norm_pre")
       (ints (Nx.slice [ A; I 0 ] k)));
  check "LFQ bits"
    (Spread.signs ~n ~reference:(lfq "ref.pre") ~spread:(lfq "s.order.pre")
       (floats
          (Lfq.decode Nx.float64 ~bits:10 (Nx.slice [ A; R (1, Nx.dim 1 k) ] k))));
  match decode (Spectrum Desi) (get Nx.int64 "ref.tokens" e) with
  | Spectrum { spectrum = d; _ } ->
      check "flux"
        (Spread.values ~n ~reference:(r "flux") ~spread:(s "flux")
           (floats d.flux));
      check "mask"
        (Spread.signs ~n ~reference:(r "logit") ~spread:(s "logit")
           (signs d.mask))
  | _ -> failwith "not a spectrum"

(* The transformer *)

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

let () =
  match (Sys.argv, Sys.getenv_opt "AION_PARITY") with
  | [| _; device; hub; ref64 |], Some dir when dir <> "" ->
      let gpu = Result.error_to_failure (Rig_gpus.open_ device) in
      let module D = (val Nx.devices [ gpu ]) in
      codecs D.on ~hub dir;
      run D.on ~hub ~ref64;
      printf "failing codec checks: %d\n" !failures;
      exit (if !failures = 0 then 0 else 1)
  | _ ->
      prerr_endline "usage: AION_PARITY=DIR gpu_parity.exe DEVICE HUB REF64";
      exit 2
