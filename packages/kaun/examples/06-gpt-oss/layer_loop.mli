(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** gpt-oss compiled one layer kind at a time.

    The model is {!Gpt_oss.block} folded over the layers. Here each layer kind
    is compiled once, as a program that reads a block's weights and the cache
    index and consumes that layer's cache and the residual stream, and a host
    loop calls it once per layer. Compilation cost does not grow with depth, a
    call holds one layer's intermediates at a time, so a whole prompt goes
    through in one call, and each layer's weights stay the leaves the import
    placed. *)

val cached :
  device:Nx.Device.t ->
  Gpt_oss.config ->
  (float, 'b) Nx.t Gpt_oss.params ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t list ->
  Kaun.Cache_index.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Kaun.Attention.Cache.t list
(** [cached ~device cfg p] is {!Gpt_oss.cached}[ cfg p] run on [device] by an
    embedding program and one block program per layer kind, each compiled once
    per call shape. Apply it once and reuse the result: the partial application
    holds the programs. A block program is {!Gpt_oss.block} compiled on the
    signature

    {[
    Nx.Ptree.(
      block @-> consumes cache @@ Cache_index.ptree @-> consumes tensor
      @@ returns (pair tensor cache))
    ]}

    where [block] and [cache] instantiate {!Gpt_oss.Block} and
    {!Kaun.Attention.Cache}: it reads its layer's weights and the index, and
    consumes its layer's cache and the residual stream, whose storage its
    results take. A call places the index on [device] once for all the layers
    and consumes the caches it is given. [p] is best placed on [device]
    ({!Gpt_oss.of_hf}[ ~placement]): its leaves are read where they are. *)

val greedy :
  ?device:Nx.Device.t ->
  Gpt_oss.config ->
  (float, 'b) Nx.t Gpt_oss.params ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t list ->
  Kaun.Cache_index.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (int32, Nx.int32_elt) Nx.t * (float, 'b) Nx.t Kaun.Attention.Cache.t list
(** [greedy ?device cfg p caches index ids] is the most likely next token of
    each sequence, of shape [[| batch |]], after the tokens [ids], and the
    caches with their keys and values written. With [device] it is {!cached} and
    a compiled head over the last position; without, {!Gpt_oss.cached} run
    eagerly. Apply it to [cfg p] once and reuse the result. *)
