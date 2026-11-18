(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The distillation step and the pretraining group of [upstream/tiny], which
    [gen/tiny.py] records with TESSERA's own code and [test_parity] checks. *)

val step :
  string -> Nx.host Nx.int64_t * Nx.host Tessera.Batch.t * Nx.host Nx.Rng.key
(** [step dir] is the step over the tile [dir]: the index in the tile of each
    pixel with an observation, the batch of those pixels, duplicate-filled
    ({!Tessera.Batch.expand}), one a row of 32 tokens, and its places' keys,
    {!Tessera.Distil.keys} of [Nx.Rng.key 0] from position [0]. *)

val views : string -> Nx.host Tessera.Barlow.views
(** [views dir] is the pretraining group of the tile [dir]'s pixels with an
    observation of each sensor: {!Tessera.Barlow.group} of [Nx.Rng.key 0], views
    of 4 observations, from position [0]. *)
