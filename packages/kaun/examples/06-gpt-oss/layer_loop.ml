(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

let cached (type b) ~device cfg (p : (float, b) Nx.t Gpt_oss.params) =
  let module Block =
    (val Gpt_oss.block_ptree ()
        : Nx.Ptree.S with type t = (float, b) Nx.t Gpt_oss.block)
  in
  (* What a block program reads. *)
  let module Layer = struct
    type t = { b : Block.t; index : Cache_index.t }

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) l =
      let b = Block.map f l.b in
      { b; index = Cache_index.map f l.index }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) l l' =
      let b = Block.map2 f l.b l'.b in
      { b; index = Cache_index.map2 f l.index l'.index }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) l =
      Block.iter f l.b;
      Cache_index.iter f l.index
  end in
  (* What a block program consumes and returns. *)
  let module Stream = struct
    type t = { cache : (float, b) Nx.t Attention.Cache.t; x : (float, b) Nx.t }

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) s =
      let cache = Attention.Cache.map f s.cache in
      { cache; x = f s.x }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) s s' =
      let cache = Attention.Cache.map2 f s.cache s'.cache in
      { cache; x = f s.x s'.x }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) s =
      Attention.Cache.iter f s.cache;
      f s.x
  end in
  let compile kind =
    Rune.jit_step ~device
      (module Layer)
      (module Stream)
      (fun { Layer.b; index } { Stream.cache; x } ->
        let x, cache = Gpt_oss.block cfg kind b cache index x in
        { Stream.cache; x })
  in
  let sliding = compile Gpt_oss.Sliding and full = compile Gpt_oss.Full in
  let embed = Rune.jit' ~device (Embedding.apply p.tok) in
  fun caches index ids ->
    let index = Cache_index.map (Rune.to_device ~device) index in
    let rec go x rev layers blocks caches =
      match (layers, blocks, caches) with
      | [], [], [] -> (x, List.rev rev)
      | kind :: layers, b :: blocks, cache :: caches ->
          let block =
            match kind with Gpt_oss.Sliding -> sliding | Full -> full
          in
          let { Stream.cache; x } =
            block { Layer.b; index } { Stream.cache; x }
          in
          go x (cache :: rev) layers blocks caches
      | _ ->
          invalid_arg
            "Layer_loop.cached: the configuration, the parameters and the \
             caches differ in their number of blocks"
    in
    go (embed ids) [] cfg.Gpt_oss.layers p.blocks caches

let greedy ?device cfg p =
  let head h =
    let last = Nx.slice [ A; I (Nx.dim 1 h - 1) ] h in
    Nx.argmax ~axis:1 (Gpt_oss.logits cfg p last)
  in
  match device with
  | None ->
      fun caches index ids ->
        let h, caches = Gpt_oss.cached cfg p caches index ids in
        (head h, caches)
  | Some device ->
      let cached = cached ~device cfg p and head = Rune.jit' ~device head in
      fun caches index ids ->
        let h, caches = cached caches index ids in
        (head h, caches)
