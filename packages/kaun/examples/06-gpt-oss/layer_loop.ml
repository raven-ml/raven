(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let cached ~devices cfg (p : (float, 'b) Nx.t Gpt_oss.params) =
  let block = Nx.Ptree.instantiate (module Gpt_oss.Block)
  and cache = Nx.Ptree.instantiate (module Attention.Cache) in
  let compile kind =
    Rune.jit ~devices
      Nx.Ptree.(
        block @-> consumes cache @@ Cache_index.ptree @-> consumes tensor
        @@ returns (pair tensor cache))
      (Gpt_oss.block cfg kind)
  in
  let sliding = compile Gpt_oss.Sliding and full = compile Gpt_oss.Full in
  let embed = Rune.jit' ~devices (Embedding.apply p.tok) in
  let placement = Nx.Placement.replicated devices in
  fun caches index ids ->
    let index =
      Nx.Ptree.map Cache_index.ptree (fun _ x -> Nx.place placement x) index
    in
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
    go (embed ids) [] cfg.Gpt_oss.layers p.blocks caches

let greedy ?devices cfg p =
  let head h =
    let last = Nx.slice [ A; I (Nx.dim 1 h - 1) ] h in
    Nx.argmax ~axis:1 (Gpt_oss.logits cfg p last)
  in
  match devices with
  | None ->
      fun caches index ids ->
        let h, caches = Gpt_oss.cached cfg p caches index ids in
        (head h, caches)
  | Some devices ->
      let cached = cached ~devices cfg p and head = Rune.jit' ~devices head in
      fun caches index ids ->
        let h, caches = cached caches index ids in
        (head h, caches)
