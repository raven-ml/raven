(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let cached ~placement cfg (p : ((float, 'b, 'd) Nx.t, 'd) Gpt_oss.params) =
  let block = Nx.Ptree.instantiate (module Gpt_oss.Block)
  and cache = Nx.Ptree.instantiate (module Attention.Cache)
  and index = Nx.Ptree.v Cache_index.walk in
  let compile kind =
    let f =
      Rune.jit
        Nx.Ptree.(
          block @-> cache @-> index @-> tensor @-> returns (pair tensor cache))
        (Gpt_oss.block cfg kind)
    in
    fun b c index x -> f b (Nx.Ptree.donate cache c) index (Nx.donate x)
  in
  let sliding = compile Gpt_oss.Sliding and full = compile Gpt_oss.Full in
  let embed = Rune.jit' (Embedding.apply p.tok) in
  fun caches index ids ->
    let index = Cache_index.place placement index in
    let rec go x rev layers blocks caches =
      match (layers, blocks, caches) with
      | [], [], [] -> (x, List.rev rev)
      | kind :: layers, b :: blocks, cache :: caches ->
          let block =
            match kind with Gpt_oss.Sliding -> sliding | Full -> full
          in
          let x, cache = block b cache index x in
          go x (cache :: rev) layers blocks caches
      | _ ->
          invalid_arg
            "Layer_loop.cached: the configuration, the parameters and the \
             caches differ in their number of blocks"
    in
    go (embed (Nx.place placement ids)) [] cfg.Gpt_oss.layers p.blocks caches

let greedy ~jit ~placement cfg p =
  let head h =
    let last = Nx.slice [ A; I (Nx.dim 1 h - 1) ] h in
    Nx.argmax ~axis:1 (Gpt_oss.logits cfg p last)
  in
  if not jit then fun caches index ids ->
    let index = Cache_index.place placement index in
    let h, caches =
      Gpt_oss.cached cfg p caches index (Nx.place placement ids)
    in
    (head h, caches)
  else
    let cached = cached ~placement cfg p and head = Rune.jit' head in
    fun caches index ids ->
      let h, caches = cached caches index ids in
      (head h, caches)
