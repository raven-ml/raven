(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Searching kernel optimisations by measuring them.

    A beam search picks a kernel's optimisations ({!Opt.t}) by timing the
    programs they make. Starting from the kernel as it is scheduled, each round
    applies every optimisation of a fixed table ({!actions}) to each of the
    fastest kernels found so far, compiles the results, times them, and keeps
    the fastest, until a round no longer gains.

    The search is a diagnostic: it compiles and times hundreds of programs per
    kernel, so it is run offline, on a few kernels extracted from a program, and
    the optimisations it finds are then applied by name ({!Ops.kernel_info}'s
    [opts_to_apply]). No kernel is searched unless its argument asks for a width
    ({!Ops.kernel_info}'s [beam]) and its compiler is given the search:
    [Codegen.to_program ~beam:(beam_search ~measure)].

    Nothing here runs a program. The search times candidates with the
    measurement it is given, which runs them on a device. *)

(** {1:actions Actions} *)

val actions : Opt.t list
(** [actions] is the optimisations the search tries, in order:
    - splits of the axes [0] to [9] into {!Opt.Upcast}, then {!Opt.Unroll},
      lanes by [0], [2], [3], [4], [5] and [7];
    - splits of the axes [0] to [7] into {!Opt.Local} threads by [0], [2], [3],
      [4], [8], [13], [16] and [29], then from the top by [13], [16], [28],
      [29], [32], [49], [64] and [256];
    - if the environment variable [BEAM_PADTO] holds a nonzero integer, pads of
      the axes [0] to [6] to a multiple of [32];
    - a split of the axis [0] into [32] local threads;
    - tensor cores on the axis [0] with the level [tc_opt] [0], then on the axes
      [0] to [8] with the level of [TC_OPT] (default [2]), each with the first
      core that fits and the level [use_tc] of [TC] (default [1]);
    - swaps of each two of the axes [0] to [4].

    The environment is read when the program starts ({!Helpers.getenv}). *)

val get_kernel_actions :
  ?include_0:bool ->
  ?max_up:int ->
  Postrange.Scheduler.t ->
  (int * Postrange.Scheduler.t) list
(** [get_kernel_actions ~include_0 ~max_up k] is the kernels that one action
    makes of [k]: [(i + 1, k')] for the action [i] of {!actions} and [k'] a copy
    of [k] it was applied to, in the order of {!actions}, after [(0, k)] if
    [include_0] (default [true]). [k] is left as it is. An action is left out
    when:
    - it does not apply ({!Postrange.Scheduler.apply_opt});
    - it is no tensor core and its axis is not one of [k]'s, or it splits a
      whole axis by its size while {!actions} splits it by [0] alike;
    - [k'] has more upcast and unrolled lanes than [max_up] (default the
      environment variable [BEAM_UPCAST_MAX], or [256]), not counting a tensor
      core's product for each of its threads, or more warp and local threads
      than [BEAM_LOCAL_MAX] (default [1024]). When [BEAM_LOG_SURPASS_MAX] holds
      a nonzero integer, these are reported on standard output.

    Raises [Invalid_argument] as {!Postrange.Scheduler.apply_opt} does, and if
    whether a kernel has too many lanes or threads depends on the value of a
    variable. *)

(** {1:search Searching} *)

val beam_search :
  measure:(cold:bool -> vars:(string * int) list -> Ops.t -> float) ->
  ?allow_test_size:bool ->
  int ->
  Postrange.Scheduler.t ->
  Postrange.Scheduler.t
(** [beam_search ~measure ~allow_test_size amt k] is [k], or a copy of it, with
    the optimisations that a beam search of width [amt] finds fastest. [k]'s
    kernel is the sink that {!Postrange.apply_opts} optimises.

    [measure ~cold ~vars prg] is the time in seconds of one run of the compiled
    program [prg] ({!Op.Program}) on a device of the target of [k]'s renderer,
    with each variable bound to its value in [vars], and with the device's
    caches invalidated first if [cold] and the device can. The search asks for
    cold runs, with each variable of [k]'s kernel ({!Ops.variables}) bound to
    the middle of its bounds, [(vmin + vmax) / 2] rounded down. A measurement
    that raises [Failure] drops its candidate; any other exception is raised by
    the search. The search limits neither a compilation nor a run: a candidate
    whose compilation or run hangs hangs the search, one more reason to run it
    offline.

    The beam holds up to [amt] kernels, and starts as [k], of an infinite time.
    Each round:
    + The candidates are the kernels {!get_kernel_actions} makes of the beam's
      kernels ([~include_0:false]), in order.
    + Each candidate's kernel is compiled for [k]'s renderer
      ({!Codegen.to_program}), named ["test"] and with its storage placed on the
      renderer's device, on the domains {!Worker.map} spreads them over. A
      candidate whose compilation raises [Failure] is dropped, and so is one
      whose compilation raises another exception, unless the environment
      variable [BEAM_STRICT_MODE] holds a nonzero integer and the exception is
      raised. A candidate is also dropped if its program has [BEAM_UOPS_MAX]
      (default [3000]) instructions or more, unless that is [0] or less.
    + In order, a candidate is dropped if its binary was timed before in this
      search, or if its program's estimated operations ({!Ops.estimates}, [0] if
      unknown) are more than [1000] times the fewest of this round's programs so
      far. Each other one is measured up to three times, stopping once its least
      time exceeds three times the time of the beam's first kernel; its time is
      the least measured.
    + The search ends if no candidate was timed, if the fastest took less than
      [BEAM_MIN_PROGRESS] microseconds (default [0.01]), or if it is faster than
      the beam's first kernel by less than that. It then keeps the fastest
      candidate alone if it is faster than the beam's first kernel. Otherwise
      the beam becomes the [amt] fastest candidates, ties in order, and a new
      round starts.

    The result is the beam's first kernel.

    With [allow_test_size] (default: whether the environment variable
    [BEAM_ESTIMATE] holds a nonzero integer, default [1]), a program launching
    more than [65536] workgroups is measured launching fewer: of its global
    sizes, the last one greater than [16] is halved until their product is at
    most [65536], and its time is scaled up by the ratio of the products.

    While the setting {!Helpers.cachelevel} is positive, each search keeps the
    optimisations it finds in the {!Helpers.Diskcache} table ["beam_search"],
    keyed by what the search is a function of but the times it measures: [k]'s
    kernel ({!Ops.key}), [amt], [allow_test_size], the name and target of [k]'s
    renderer and the table of its compiler's binaries
    ({!Renderer.Compiler.cachekey}), the candidate {!actions}, the settings and
    variables that shape compilation ({!Helpers.shaping}), among them the
    environment variables [BEAM_PADTO], [BEAM_UOPS_MAX], [BEAM_UPCAST_MAX],
    [BEAM_LOCAL_MAX] and [BEAM_MIN_PROGRESS], and the sources of this library.
    Unless {!Helpers.ignore_beam_cache} holds, a search whose key is kept
    measures nothing, and applies the optimisations kept beyond as many as [k]
    has to a copy of [k]. A kept result is what an earlier search measured
    fastest: another search may measure otherwise.

    When the setting {!Helpers.debug} is [2] or more, the progress of the search
    is printed on standard output. When the environment variable [BEAM_DEBUG]
    holds a positive integer, so are the kernel searched, the candidates whose
    measurement failed and the result; from [2], every candidate timed.

    Raises [Invalid_argument] if [amt] is not positive or as
    {!get_kernel_actions} does, and [Failure] if the kept optimisations are
    malformed. *)
