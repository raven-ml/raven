(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let cached ~placement cfg (p : ((float, 'b, 'd) Nx.t, 'd) Gpt_oss.params) =
  let block = Nx.Ptree.instantiate (module Gpt_oss.Block)
  and cache = Nx.Ptree.instantiate (module Attention.Cache)
  and index = Nx.Ptree.v Cache_index.walk in
  let f =
    Rune.jit
      Nx.Ptree.(
        block @-> cache @-> index @-> tensor @-> returns (pair tensor cache))
      (Gpt_oss.block cfg)
  in
  let block b c index x = f b (Nx.Ptree.donate cache c) index (Nx.donate x) in
  (* The ids embedded and the index resolved for each layer kind, once a call:
     the blocks read the index's addressing and compute none of it. *)
  let embed =
    Rune.jit
      Nx.Ptree.(index @-> tensor @-> returns (triple tensor index index))
      (fun index ids ->
        let at = Gpt_oss.windows cfg index in
        (Embedding.apply p.tok ids, at Sliding, at Full))
  in
  fun caches index ids ->
    let x, sliding, full =
      embed (Cache_index.place placement index) (Nx.place placement ids)
    in
    let index = function Gpt_oss.Sliding -> sliding | Full -> full in
    let rec go x rev layers blocks caches =
      match (layers, blocks, caches) with
      | [], [], [] -> (x, List.rev rev)
      | layer :: layers, b :: blocks, cache :: caches ->
          let x, cache = block b cache (index layer) x in
          go x (cache :: rev) layers blocks caches
      | _ ->
          invalid_arg
            "Layer_loop.cached: the configuration, the parameters and the \
             caches differ in their number of blocks"
    in
    go x [] cfg.Gpt_oss.layers p.blocks caches

let greedy ~placement cfg p =
  let head h =
    let last = Nx.slice [ A; I (Nx.dim 1 h - 1) ] h in
    Nx.argmax ~axis:1 (Gpt_oss.logits cfg p last)
  in
  let cached = cached ~placement cfg p and head = Rune.jit' head in
  fun caches index ids ->
    let h, caches = cached caches index ids in
    (head h, caches)
