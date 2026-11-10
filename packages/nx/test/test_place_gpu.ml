(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Placing onto this machine's first GPU: a placement from the host, or from
   a split of columns over two software devices, which gathers windows into the
   GPU's memory, keeps every byte. test_place states on held devices that a
   placement returns with its copies queued; a GPU may finish a copy before
   the test looks, so here only the bytes are stated. It skips without a
   GPU. *)

open Windtrap
module A = Nx_array

let gpu = lazy (Result.to_option (Rig_gpus.first ()))

module Mem = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

let law_gpu (Nx_support.Bits a, from_split) =
  let d =
    match Lazy.force gpu with
    | Some d -> d
    | None -> skip ~reason:"no GPU" ()
  in
  let module G = (val Nx.devices [ d ]) in
  cover "from a split of columns" from_split;
  cover "a source of 64 KiB or more"
    (A.Dtype.bytes (A.dtype a) (A.Layout.numel (A.layout a)) >= 65536);
  let x = Nx.Repr.of_array Nx.Host.v a in
  let y =
    if from_split then Nx.place G.on (Nx.place (Mem.split ~axis:1) x)
    else Nx.place G.on x
  in
  let back = Nx.place Nx.Host.on y in
  equal (array int) ~msg:"its bytes" (Nx_support.bytes a)
    (Nx_support.bytes (require_some (Nx.Repr.array back)))

let () =
  exit
    (run "nx place on a GPU"
       [
         prop ~count:50 "a value placed on a GPU and back keeps its bytes"
           (Gen.pair Nx_support.bits Gen.bool)
           law_gpu;
       ])
