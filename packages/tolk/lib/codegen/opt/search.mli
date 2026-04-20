(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Beam search kernel optimiser.

    Explores the space of kernel optimisations by compiling, timing, and
    selecting the best candidates over multiple rounds. *)

val beam_search :
  ?allow_test_size:bool ->
  ?disable_cache:bool ->
  Postrange.t ->
  Device.Buffer.t list ->
  int ->
  Device.t ->
  Postrange.t
(** [beam_search s rawbufs amt device] optimises scheduler [s] using
    beam search with beam width [amt].

    - [allow_test_size] (default [true]) scales down global dimensions
      during timing to stay within hardware limits.
    - [disable_cache] (default from [IGNORE_BEAM_CACHE] env) skips the
      on-disk result cache.

    Returns the best scheduler found. *)
