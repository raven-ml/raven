(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Searching kernel optimisations by measuring them.

    A beam search picks a kernel's optimisations ({!Opt.t}) by timing the
    programs they make. Starting from the kernel as it is scheduled, each round
    applies every optimisation of a table ({!actions}) to each of the fastest
    kernels found so far, compiles the results, times them, and keeps the
    fastest, until a round no longer gains.

    The search is a diagnostic: it compiles and times hundreds of programs per
    kernel, so it is run offline, on a few kernels extracted from a program, and
    the optimisations it finds are then applied by name ({!Ops.kernel_info}'s
    [opts_to_apply]). No kernel is searched unless its argument asks for a width
    ({!Ops.kernel_info}'s [beam]) and its compiler is given the search:
    [Codegen.to_program ~beam:(beam_search ~link ~time)].

    Nothing here runs a program. The search links and times candidates with the
    functions it is given, which run them on a device. *)

(** {1:actions Actions} *)

val actions : unit -> Opt.t list
(** [actions ()] is the optimisations the search tries under the current
    settings, in order:
    - splits of the axes [0] to [9] into {!Opt.Upcast}, then {!Opt.Unroll},
      lanes by [0], [2], [3], [4], [5] and [7];
    - splits of the axes [0] to [7] into {!Opt.Local} threads by [0], [2], [3],
      [4], [8], [13], [16] and [29], then from the top by [13], [16], [28],
      [29], [32], [49], [64] and [256];
    - under {!Setting.beam_padto}, pads of the axes [0] to [6] to a multiple of
      [32];
    - a split of the axis [0] into [32] local threads;
    - tensor cores on the axis [0] with the level [tc_opt] [0], then on the axes
      [0] to [8] with the level [2], which admits every kernel a tensor core
      fits, each with the first core that fits and the level [use_tc] of
      {!Setting.use_tc};
    - swaps of each two of the axes [0] to [4]. *)

val get_kernel_actions :
  ?max_up:int -> Postrange.Scheduler.t -> (Opt.t * Postrange.Scheduler.t) list
(** [get_kernel_actions ~max_up k] is the kernels that one action makes of [k]:
    [(a, k')] for an action [a] of [actions ()] and [k'] a copy of [k] it was
    applied to, in the order of {!actions}. [k] is left as it is. An action is
    left out when:
    - it does not apply ({!Postrange.Scheduler.apply_opt});
    - it is no tensor core and its axis is not one of [k]'s, or it splits a
      whole axis by its size while [actions ()] splits it by [0] alike;
    - [k'] has more upcast and unrolled lanes than [max_up] (default
      {!Setting.beam_upcast_max}), not counting a tensor core's product for each
      of its threads, or more warp and local threads than
      {!Setting.beam_local_max}. Under {!Setting.beam_log_surpass_max}, these
      are reported on standard output.

    Raises [Invalid_argument] as {!Postrange.Scheduler.apply_opt} does, and if
    whether a kernel has too many lanes or threads depends on the value of a
    variable. *)

(** {1:search Searching} *)

val beam_search :
  link:(Ops.t -> 'linked) ->
  time:(vars:(string * int) list -> 'linked -> float) ->
  ?allow_test_size:bool ->
  int ->
  Postrange.Scheduler.t ->
  Postrange.Scheduler.t
(** [beam_search ~link ~time ~allow_test_size amt k] is [k], or a copy of it,
    with the optimisations that a beam search of width [amt] finds fastest.
    [k]'s kernel is the sink that {!Postrange.apply_opts} optimises.

    [link prg] is the compiled program [prg] ({!Op.Program}) of [k]'s kernel
    made ready to run on a device of the target of [k]'s renderer, and
    [time ~vars p] is the time in seconds of one run of the program [p] links,
    from cold caches where the device can, with each variable bound to its value
    in [vars]: each variable of [k]'s kernel ({!Ops.variables}) bound to the
    middle of its bounds, [(vmin + vmax) / 2] rounded down. The search links
    each program it times once, and times it once per sample. A program whose
    [link] or [time] raises [Failure] is dropped; the search raises any other
    exception. The search limits neither a compilation nor a run: a candidate
    whose compilation or run hangs hangs the search, one more reason to run it
    offline.

    A kernel is compiled for [k]'s renderer named ["test"], with its storage
    placed on the renderer's device. It is first linearized
    ({!Codegen.linearize}) and dropped if it has {!Setting.beam_uops_max}
    instructions or more, unless that is [0] or less. It is then completed
    ({!Codegen.compile}), and neither its program nor its binary is kept past
    the search. A kernel whose compilation raises [Failure] is dropped, and so
    is one whose compilation raises another exception, unless
    {!Setting.beam_strict_mode} holds and the exception is raised. A search
    compiles each kernel ({!Postrange.Scheduler.ast}) once: a candidate whose
    kernel it met before takes that kernel's result. It also compiles each
    source once: a kernel that renders a source compiled or rejected before
    takes that binary or rejection. Candidates are compiled without the disk
    cache of their compiler's binaries.

    A program is timed up to three times, stopping once its least time exceeds
    an early stop. Its samples are the times measured.

    The search compiles and times [k] itself, with no early stop. The beam holds
    up to [amt] kernels and starts as [k] with its samples, or with none if [k]
    was dropped. Each round:
    + The candidates are the kernels {!get_kernel_actions} makes of the beam's
      kernels, in order, compiled on the domains {!Worker.map} spreads them
      over.
    + In order, a candidate is dropped if its binary was timed before in this
      search, or if its program's estimated operations ({!Ops.estimates}, [0] if
      unknown) are more than [1000] times the fewest of this round's programs so
      far. Each other one is timed with an early stop of three times the least
      sample of the beam's first kernel.
    + The fastest candidate is the one of least sample, the first of ties. The
      round progresses if each of its samples is less than each sample of the
      beam's first kernel by more than {!Setting.beam_min_progress}
      microseconds. A kernel with no samples is slower than any. If the round
      progresses, the beam becomes the [amt] candidates of least sample, ties in
      order, and a new round starts. Otherwise the search ends.

    The result is the beam's first kernel. It is [k] unless a candidate
    progressed on it.

    With [allow_test_size] (default {!Setting.beam_estimate}), a program
    launching more than [65536] workgroups is timed launching fewer: of its
    global sizes, the last one greater than [16] is halved until their product
    is at most [65536], and its time is scaled up by the ratio of the products.

    While the setting {!Setting.cachelevel} is positive, each search keeps the
    optimisations it finds in the {!Helpers.Diskcache} table ["beam_search"],
    keyed by what the search is a function of but the times it measures: [k]'s
    kernel ({!Ops.key}), [amt], [allow_test_size], the name and target of [k]'s
    renderer and the table of its compiler's binaries
    ({!Renderer.Compiler.cachekey}), the settings that shape compilation
    ({!Setting.shaping}), among them those that pick the candidates
    ({!Setting.use_tc} and {!Setting.beam_padto}) and how it measures and stops
    ({!Setting.beam_uops_max}, {!Setting.beam_upcast_max},
    {!Setting.beam_local_max}, {!Setting.beam_min_progress} and
    {!Setting.beam_estimate}), and the sources of this library. Unless
    {!Setting.ignore_beam_cache} holds, a search whose key is kept compiles,
    links and times nothing, and applies the optimisations kept beyond as many
    as [k] has to a copy of [k]. A kept result is what an earlier search
    measured fastest: another search may measure otherwise.

    When the setting {!Setting.debug} is [2] or more, the progress of the search
    is printed on standard output, and {!Setting.beam_debug} prints more.

    Raises [Invalid_argument] if [amt] is not positive or as
    {!get_kernel_actions} does, and [Failure] if the kept optimisations are
    malformed. *)
