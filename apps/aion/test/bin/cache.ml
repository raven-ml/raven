(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* cache.exe MODEL DIR: a Hub cache in DIR, as Kaun_hf keeps one, holding the
   file MODEL as aion-base's weights and codebooks of 1024 edges for the
   scalar modalities of the narrow transformer: Flux_g's even in log space
   from -3 to 8, Z's from 0 to 6, and XP coefficient i's from -3 + i / 100 to
   3 + i / 100, so each coefficient's codebook is its own. *)

open Aion

let save dir file a = Hub_cache.plant dir file (Nx_io.Archive.of_list a)

let even lo hi = Nx.linspace Nx.float32 lo hi 1024

let () =
  let model = Sys.argv.(1) and dir = Sys.argv.(2) in
  save dir "model.safetensors"
    (List.map
       (fun n -> (n, Option.get (Nx_io.Archive.find n (Nx_io.load_safetensors model))))
       (Nx_io.Archive.names (Nx_io.load_safetensors model)));
  let codec m entries =
    save dir (Hub.file (Scalar m))
      (List.map (fun (k, v) -> (k, Nx.P v)) entries)
  in
  codec Flux_g [ ("_quantizer._index_to_val", even (-3.) 8.) ];
  codec Z [ ("_quantizer.buckets", even 0. 6.) ];
  List.iter
    (fun m ->
      codec m
        (List.init 55 (fun i ->
             let o = float_of_int i /. 100. in
             ( Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i,
               even (-3. +. o) (3. +. o) ))))
    Modality.[ Xp_bp; Xp_rp ]
