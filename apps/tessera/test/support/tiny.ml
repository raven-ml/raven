(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tessera

(* The tile's pixels whose observations [keep] takes, and their indices. *)
let pixels dir keep =
  let px, _ = Pixels.of_tile (Tile.load dir) in
  let l2 = Nx.to_array (Nx_ragged.lengths px.s2)
  and l1 = Nx.to_array (Nx_ragged.lengths px.s1) in
  let kept =
    List.filter
      (fun p -> keep (l2.(p) > 0L) (l1.(p) > 0L))
      (List.init (Pixels.length px) Fun.id)
  in
  let indices =
    Nx.create Nx.int64
      [| List.length kept |]
      (Array.of_list (List.map Int64.of_int kept))
  in
  (indices, Pixels.take ~indices px)

let step dir =
  let indices, px = pixels dir ( || ) in
  let n = Pixels.length px in
  let px, counts = Batch.expand px (Batch.v2_counts px) in
  let shape =
    { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 1; width = 32 }
  in
  let b, _ = Batch.pack shape px counts ~from:0 in
  (indices, b, Distil.keys ~key:(Nx.Rng.key 0) ~first:0 b)

let views dir =
  let _, px = pixels dir ( && ) in
  Barlow.group ~key:(Nx.Rng.key 0) ~length:4 ~first:0 px
