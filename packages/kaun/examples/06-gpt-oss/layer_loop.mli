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
  device:string ->
  Gpt_oss.config ->
  (float, 'b) Nx.t Gpt_oss.params ->
  (float, 'b) Nx.t Gpt_oss.Cache.t ->
  Kaun.Cache_index.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Gpt_oss.Cache.t
(** [cached ~device cfg p] is {!Gpt_oss.cached}[ cfg p] run on [device] by an
    embedding program and one block program per layer kind, each compiled once
    per call shape. Apply it once and reuse the result: the partial application
    holds the programs. A call places the index on [device] once for all the
    layers and consumes the caches it is given, as {!Rune.jit_step} does. [p] is
    best placed on [device] ({!Gpt_oss.of_hf}[ ~device]): its leaves are read
    where they are. *)

val greedy :
  ?device:string ->
  Gpt_oss.config ->
  (float, 'b) Nx.t Gpt_oss.params ->
  (float, 'b) Nx.t Gpt_oss.Cache.t ->
  Kaun.Cache_index.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (int32, Nx.int32_elt) Nx.t * (float, 'b) Nx.t Gpt_oss.Cache.t
(** [greedy ?device cfg p caches index ids] is the most likely next token of
    each sequence, of shape [[| batch |]], after the tokens [ids], and the
    caches with their keys and values written. With [device] it is {!cached} and
    a compiled head over the last position; without, {!Gpt_oss.cached} run
    eagerly. Apply it to [cfg p] once and reuse the result. *)
