(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Beam search kernel optimiser.

    Explores the space of kernel optimisations by compiling, timing, and
    selecting the best candidates over multiple rounds. *)

val actions : Tolk_uop.Uop.Opt.t list
(** The fixed candidate optimisation set explored by {!beam_search}. Each
    round applies one of these to every scheduler kept from the previous
    round. *)

val beam_parallel : int Helpers.Context_var.t
(** [beam_parallel] is the number of domains {!beam_search} uses to compile a
    round's candidates concurrently ([BEAM_PARALLEL] environment variable;
    [0], the default, compiles sequentially). Only the CPU-side compile runs
    in parallel; candidates are always timed one at a time. Override it for a
    scope with {!Helpers.Context_var.with_context}. *)

val beam_search :
  to_program:(Device.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  ?allow_test_size:bool ->
  ?disable_cache:bool ->
  Postrange.t ->
  Device.Buffer.t list ->
  var_vals:(string * int) list ->
  int ->
  Device.t ->
  Postrange.t
(** [beam_search ~to_program s rawbufs ~var_vals amt device] optimises scheduler [s] using
    beam search with beam width [amt]. [var_vals] supplies the named scalar
    values used for candidate filtering, estimates, launch dimensions and timing.
    [to_program] compiles kernels needed by the shared queue path without
    recursively invoking beam search. Every symbolic variable in [s] must have a value within its declared bounds.
    Missing or out-of-bounds values raise [Invalid_argument].

    - [allow_test_size] (default [true]) scales down global dimensions
      during timing to stay within hardware limits.
    - [disable_cache] (default from [IGNORE_BEAM_CACHE] env) bypasses
      on-disk cache reads. Successful searches still update the cache when
      caching is enabled.

    Returns the best scheduler found. *)
