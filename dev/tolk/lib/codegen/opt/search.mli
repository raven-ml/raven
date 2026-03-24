(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Beam search kernel optimizer.

    Enumerates optimization actions and iteratively searches for the fastest
    kernel configuration by compiling and timing candidates.

    *)

val actions : Tolk_ir.Kernel.Opt.t list
(** Static list of ~200 candidate optimizations. *)

val get_kernel_actions :
  ?include_0:bool ->
  ?max_up:int ->
  Postrange.t ->
  (int * Postrange.t) list
(** [get_kernel_actions ?include_0 ?max_up s] enumerates all valid
    optimization actions for the current scheduler state.

    Returns [(action_id, scheduler)] pairs. When [include_0] is [true]
    (default), the identity action [(0, s)] is included. [max_up] defaults
    to the [BEAM_UPCAST_MAX] environment variable (default [256]). *)

val get_test_global_size : int array -> int -> int array * float
(** [get_test_global_size global_size max_global_size] scales down
    [global_size] dimensions so their product fits within [max_global_size].
    Returns [(scaled_dims, factor)] where [factor] is the ratio of the
    original product to the scaled product. Halves dimensions from last to
    first, stopping at 16 per dimension.

    Used by beam search to quickly evaluate scaled-down kernels. *)

val beam_search :
  ?allow_test_size:bool ->
  ?disable_cache:bool ->
  Postrange.t ->
  Device.Buffer.t list ->
  int ->
  Device.t ->
  Postrange.t
(** [beam_search ?allow_test_size ?disable_cache s rawbufs amt device] finds
    the best optimization by iterative beam search. Expands, compiles, and
    times candidates at each step, keeping the top [amt] performers. The
    renderer is read from [Postrange.ren s].

    When [allow_test_size] is [true] (default), global dimensions are scaled
    down during timing for faster beam iterations.

    When [disable_cache] is [true], disk cache is bypassed regardless of the
    [CACHELEVEL] and [IGNORE_BEAM_CACHE] environment variables. Default is
    [false].

    Uses {!Lowering.lower_and_linearize} and {!Device.compile_program}
    internally to compile candidates. *)
