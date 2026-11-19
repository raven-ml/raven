(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectrum codec against upstream (PolymathicAI/AION db7fba6df53b) with
   AION-1's weights, by the rounding ensemble's rule ([Spread]). It lives
   outside runtest: the weights are 99 MB, the ensemble's files tens of MB, and
   a run takes minutes. It runs under the alias parity, on AION_PARITY's
   huggingface, goldens and subset, and ensemble/spectrum:
   gen/spectrum_ensemble.py's files, one per set: golden, ls_desi, desi and sdss
   (rows trimmed of their -1 pads). A set without its file fails. Each holds the
   set's objects, the float64 reference's values and the ensemble's spreads at
   float32's and float64's scales.

   Two checks per set:

   - float32: the codec as AION-1 ships it, its spread the ensemble's at the
   order model's float32 scale;

   - float64: the codec cast to float64, its spread the ensemble's at float64's
   scale, where a wrong constant or structure shows by orders of magnitude.

   Each checks the LFQ's input, the normalisation, the tokens (the
   normalisation's and each LFQ bit), and the decoding of the reference's tokens
   (flux and mask). It exits 1 when a check fails. *)

open Aion

(* The inputs, under AION_PARITY as the README's Tests section lays them out. *)
let dir =
  match Sys.getenv_opt "AION_PARITY" with
  | Some d when d <> "" -> d
  | _ ->
      prerr_endline
        "spectrum_parity: AION_PARITY is unset: set it to the directory of the \
         README's Tests section";
      exit 1

let path = Filename.concat dir
let printf = Printf.printf
let failures = ref 0
let load path = Nx_io.load_safetensors path

let get dt name a =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let floats t = Nx.to_array (Nx.cast Nx.float64 t)
let ints t = Array.map Int64.to_int (Nx.to_array (Nx.cast Nx.int64 t))

let take rows t =
  let p =
    Nx.create Nx.int64 [| Array.length rows |] (Array.map Int64.of_int rows)
  in
  Nx.take ~axis:0 p t

(* The LFQ's input, [b; n; 10], from upstream's [b; 10; n]. *)
let lfq t = floats (Nx.transpose ~axes:[ 0; 2; 1 ] t)

type set = {
  name : string;
  modality : Modality.spectrum option; (* None: encoded unpadded *)
  spectra : (Nx.float32_elt, Nx.host) Spectrum.spectrum;
  e : Nx.host Nx_io.Archive.t;
}

let spectrum a pre rows =
  let f key = take rows (get Nx.float32 (pre ^ key) a) in
  Spectrum.
    {
      flux = f "flux";
      ivar = f "ivar";
      mask = Nx.not_equal_s (take rows (get Nx.uint8 (pre ^ "mask") a)) 0;
      wavelength = f "lambda";
    }

(* Each row cut before its run of -1 wavelengths and padded as SDSS. *)
let trimmed (x : _ Spectrum.spectrum) =
  let w = floats x.wavelength and m = Nx.dim 1 x.wavelength in
  let rows =
    List.init (Nx.dim 0 x.wavelength) (fun i ->
        let n = ref m in
        while !n > 0 && w.((i * m) + !n - 1) < 0. do
          decr n
        done;
        let c t = Nx.slice [ Nx.R (i, i + 1); Nx.R (0, !n) ] t in
        Spectrum.pad Modality.Sdss
          Spectrum.
            {
              flux = c x.flux;
              ivar = c x.ivar;
              mask = c x.mask;
              wavelength = c x.wavelength;
            })
  in
  let cat f = Nx.concatenate ~axis:0 (List.map f rows) in
  Spectrum.
    {
      flux = cat (fun x -> x.flux);
      ivar = cat (fun x -> x.ivar);
      mask = cat (fun x -> x.mask);
      wavelength = cat (fun x -> x.wavelength);
    }

(* The sets whose ensemble file exists. A missing one is named, skipped and
   counted as a failure. *)
let sets ~goldens ~subset ~ensemble =
  let sub name = load (Filename.concat subset (name ^ ".safetensors")) in
  let input = Filename.concat goldens "SPECTRUM_input_batch.safetensors" in
  let set file name modality spectra =
    let path = Filename.concat ensemble (file ^ ".safetensors") in
    if not (Sys.file_exists path) then begin
      incr failures;
      printf "missing ensemble %s\n%!" path;
      []
    end
    else
      let e = load path in
      let rows = ints (get Nx.int64 "rows" e) in
      [ { name; modality; spectra = spectra rows; e } ]
  in
  List.concat
    [
      set "golden" "golden" None (spectrum (load input) "spectrum.");
      set "ls_desi" "ls_desi" (Some Desi)
        (spectrum (sub "ls_desi") "desi.spectrum_");
      set "desi" "desi" (Some Desi) (spectrum (sub "desi") "spectrum_");
      (* The rows are padded here, after their -1 runs are cut. *)
      set "sdss" "sdss (trimmed)" None (fun rows ->
          trimmed (spectrum (sub "sdss") "spectrum_" rows));
    ]

(* What a check compares, for every object of a set. *)
type outputs = {
  pre : float array;
  norm : float array;
  tokens : int array; (* [b; 1 + n] *)
  flux : float array;
  mask : float array; (* 1 where bad, -1 elsewhere *)
}

let compute (type b) (t : (b, Nx.host) Spectrum.t) (set : set)
    (x : (b, Nx.host) Spectrum.spectrum) =
  let padded =
    match set.modality with Some m -> Spectrum.pad m x | None -> x
  in
  let k =
    match set.modality with
    | Some m -> Spectrum.encode t m x
    | None -> Spectrum.encode_unpadded t x
  in
  let h, _ = Spectrum.latent t padded in
  let d = Spectrum.decode t (get Nx.int64 "ref.tokens" set.e) in
  {
    pre = floats h;
    norm = floats (Spectrum.normalisation padded);
    tokens = ints k;
    flux = floats d.flux;
    mask = Array.map (fun b -> if b then 1. else -1.) (Nx.to_array d.mask);
  }

let check ~n ~levels ~scale (set : set) o =
  let e = set.e in
  let r name = floats (get Nx.float64 ("ref." ^ name) e)
  and s name = floats (get Nx.float64 ("s." ^ scale ^ "." ^ name) e) in
  let pre_ref = lfq (get Nx.float64 "ref.pre" e)
  and pre_s = lfq (get Nx.float64 ("s." ^ scale ^ ".pre") e) in
  let n1 = Nx.dim 1 (get Nx.int64 "ref.tokens" e) in
  let b = Array.length o.tokens / n1 in
  let bits =
    Array.init
      (b * (n1 - 1) * 10)
      (fun j ->
        let k =
          o.tokens.((j / 10 / (n1 - 1) * n1) + 1 + (j / 10 mod (n1 - 1)))
        in
        if k land (1 lsl (9 - (j mod 10))) <> 0 then 1. else -1.)
  in
  let checks =
    [
      ("LFQ input", Spread.values ~n ~reference:pre_ref ~spread:pre_s o.pre);
      ( "normalisation",
        Spread.values ~n ~reference:(r "norm_pre") ~spread:(s "norm_pre") o.norm
      );
      ( "normalisation token",
        Spread.levels ~n ~edges:(Array.sub levels 0 1023)
          ~reference:(r "norm_pre") ~spread:(s "norm_pre")
          (Array.init b (fun i -> o.tokens.(i * n1))) );
      ("LFQ bits", Spread.signs ~n ~reference:pre_ref ~spread:pre_s bits);
      ("flux", Spread.values ~n ~reference:(r "flux") ~spread:(s "flux") o.flux);
      ("mask", Spread.signs ~n ~reference:(r "logit") ~spread:(s "logit") o.mask);
    ]
  in
  List.iter
    (fun (name, (c : Spread.check)) ->
      if c.failures <> [] then incr failures;
      printf "  %s%s\n"
        (if c.failures = [] then "" else "FAIL ")
        (Format.asprintf "%a" (Spread.pp name) c))
    checks;
  printf "%!"

let () =
  let hub = path "huggingface" and goldens = path "goldens" in
  let subset = path "subset" and ensemble = path "ensemble/spectrum" in
  let t32 = Spectrum.of_archive (Hub.codec ~cache_dir:hub (Spectrum Desi)) in
  let t64 = Spectrum.cast Nx.float64 t32 in
  let levels = floats (Spectrum.levels t32) in
  List.iter
    (fun (set : set) ->
      let n = (ints (get Nx.int64 "n" set.e)).(0) in
      printf "\n== %s: %d objects, ensembles of %d\n%!" set.name
        (Nx.dim 0 set.spectra.flux)
        n;
      let t0 = Unix.gettimeofday () in
      let o32 = compute t32 set set.spectra in
      printf "float32, order scale:\n";
      check ~n ~levels ~scale:"order" set o32;
      let reference = ints (get Nx.int64 "ref.tokens" set.e) in
      let differ = ref [] in
      Array.iteri
        (fun i k -> if k <> reference.(i) then differ := i :: !differ)
        o32.tokens;
      printf "  tokens differing from the float64 reference: %d of %d%s\n"
        (List.length !differ) (Array.length reference)
        (String.concat "" (List.rev_map (Printf.sprintf " (index %d)") !differ));
      let x64 =
        Spectrum.
          {
            flux = Nx.cast Nx.float64 set.spectra.flux;
            ivar = Nx.cast Nx.float64 set.spectra.ivar;
            mask = set.spectra.mask;
            wavelength = Nx.cast Nx.float64 set.spectra.wavelength;
          }
      in
      printf "float64, order scale at float64:\n";
      check ~n ~levels ~scale:"order64" set (compute t64 set x64);
      printf "(%.0f s)\n%!" (Unix.gettimeofday () -. t0))
    (sets ~goldens ~subset ~ensemble);
  printf "\nfailing checks: %d\n" !failures;
  exit (if !failures = 0 then 0 else 1)
