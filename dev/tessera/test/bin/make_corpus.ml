(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* make_corpus.exe DIR TARGETS: a corpus of 96 synthetic pixels in DIR and their
   targets, float32 [[96; 32]], in TARGETS. *)

let () =
  let n = 96 in
  let counts k = List.init n (fun i -> 1 + (i * k mod 37)) in
  let px = Synth.pixels { Synth.s2 = counts 7; s1 = counts 11; seed = 1 } in
  Tessera.Pixels.save Sys.argv.(1) px;
  Nx_io.save_npy Sys.argv.(2)
    (Nx.Rng.normal ~key:(Nx.Rng.key 2) Nx.float32 [| n; 32 |])
