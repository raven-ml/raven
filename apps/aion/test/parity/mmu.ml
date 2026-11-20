(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Survey and Codecs against upstream on real Multimodal Universe rows, with
   AION-1's codecs (400 MB).

   AION_PARITY's mmu holds rows cut from MMU's files by gen/mmu_cut.py (the
   tutorial's cross-matched file, a file of each survey, Gaia's whole healpix
   1631) and tokens.safetensors, upstream's tokens of the same rows
   (gen/mmu_tokens.py: the columns cast to float32 as the tutorial casts them,
   SDSS rows trimmed, upstream's CodecManager at db7fba6d).

   For every modality of every file, the tokens Codecs.encode makes of
   Survey.read's data must equal upstream's, token for token, where a token is
   present; a value that is not finite is absent here (Codecs.encode) and
   counted apart. *)

open Aion
open Part

(* Each file with its name in tokens.safetensors, its survey and its group. *)
let files =
  Survey.
    [
      ( "tutorial/legacysurvey",
        Legacy_survey,
        "tutorial.h5",
        Some "/legacysurvey" );
      ("tutorial/desi", Desi, "tutorial.h5", Some "/desi");
      ("ls", Legacy_survey, "ls.h5", None);
      ("hsc", Hsc, "hsc.h5", None);
      ("desi", Desi, "desi.h5", None);
      ("sdss", Sdss, "sdss.h5", None);
      ("gaia", Gaia, "gaia.h5", None);
    ]

(* Rows a call: an image batch of 4 holds about 300 MB of activations. *)
let batch = 4

let run () =
  let dir = path "mmu" and cache_dir = path "huggingface" in
  let upstream = load "mmu/tokens.safetensors" in
  List.iter
    (fun (name, survey, file, group) ->
      let t = Survey.load ?group survey (Filename.concat dir file) in
      let ms = Survey.modalities t in
      let codecs = Codecs.load ~cache_dir ms in
      List.iter
        (fun m ->
          let key = name ^ "." ^ Modality.key m in
          match Nx_io.Archive.find key upstream with
          | None -> printf "%-40s no upstream tokens\n%!" key
          | Some p ->
              let want = Nx.unpack Nx.int64 p in
              let n = Nx.dim 0 want and w = Nx.to_array want in
              let per = Array.length w / n in
              let same = ref 0 and diff = ref 0 and absent = ref 0 in
              let lo = ref 0 in
              while !lo < n do
                let rows =
                  Array.init (min batch (n - !lo)) (fun k -> !lo + k)
                in
                let i = Codecs.encode codecs (Survey.read t rows m) in
                let got = Nx.to_array i.tokens in
                let present = Nx.to_array i.present in
                Array.iteri
                  (fun k g ->
                    if not present.(k) then incr absent
                    else if g = w.((!lo * per) + k) then incr same
                    else incr diff)
                  got;
                lo := !lo + batch
              done;
              if !diff > 0 then fail ();
              printf "%-40s %6d equal %4d differ %4d absent\n%!" key !same !diff
                !absent)
        ms)
    files;
  finish ()
