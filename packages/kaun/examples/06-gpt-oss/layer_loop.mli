(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** gpt-oss compiled one layer kind at a time.

    The model is {!Gpt_oss.block} folded over the layers. Here each layer kind
    is compiled once, as a program that reads a block's weights and the cache
    index and is given that layer's cache and the residual stream, and a host
    loop calls it once per layer. Compilation cost does not grow with depth, a
    call holds one layer's intermediates at a time, so a whole prompt goes
    through in one call, and each layer's weights stay the leaves the import
    placed. *)

val cached :
  placement:'d Nx.Placement.t ->
  'd Gpt_oss.config ->
  ((float, 'b, 'd) Nx.t, 'd) Gpt_oss.params ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list ->
  'e Kaun.Cache_index.t ->
  'e Nx.int64_t ->
  (float, 'b, 'd) Nx.t * ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [cached ~placement cfg p] is {!Gpt_oss.cached}[ cfg p] run where [p] is
    placed by an embedding program and one block program per layer kind, each
    compiled once per call shape. Apply it once and reuse the result: the
    partial application holds the programs. A block program is {!Gpt_oss.block}
    compiled on the signature

    {[
    Nx.Ptree.(
      block @-> cache @-> v Cache_index.walk @-> tensor
      @-> returns (pair tensor cache))
    ]}

    where [block] and [cache] instantiate {!Gpt_oss.Block} and
    {!Kaun.Attention.Cache}: it reads its layer's weights and the index, and is
    given its layer's cache and the residual stream, both donated
    ({!Nx.Ptree.donate}). A call places the index and the ids at [placement]
    once for all the layers and donates the caches it is given. The programs run
    on the devices [p] is placed on ({!Gpt_oss.of_hf}[ ~placement]), whose
    leaves they read where they are, split ones included; [placement] is a copy
    on each of them. A program over several devices is evaluated by nx, so a
    set of GPUs, which computes only compiled programs, raises. *)

val greedy :
  jit:bool ->
  placement:'d Nx.Placement.t ->
  'd Gpt_oss.config ->
  ((float, 'b, 'd) Nx.t, 'd) Gpt_oss.params ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list ->
  'e Kaun.Cache_index.t ->
  'e Nx.int64_t ->
  'd Nx.int64_t * ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [greedy ~jit ~placement cfg p caches index ids] is the most likely next
    token of each sequence, of shape [[| batch |]], after the tokens [ids], and
    the caches with their keys and values written. The index and the ids are
    placed at [placement]. With [jit] it is {!cached} and a compiled head over
    the last position; without, {!Gpt_oss.cached} run eagerly, which a set that
    computes under [Rune.jit] alone cannot. Apply it to [cfg p] once and reuse
    the result. *)
