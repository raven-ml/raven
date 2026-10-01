(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Code generation: from a kernel to a program.

    A kernel is the sink of a graph of loads, stores and arithmetic over ranges,
    with its {!Ops.kernel_info} as argument. Code generation turns it into a
    program for a renderer ({!Renderer.t}) in three steps:
    - {!full_rewrite_to_sink} optimises the kernel's loops, then lowers it to
      the nodes the renderer writes: vector axes are expanded into lanes,
      reductions into accumulators, loops into launch dimensions, vectors into
      scalars and merged accesses, and the operations and types the target lacks
      into ones it has;
    - {!Linearizer.linearize} orders those nodes into a list of instructions;
    - the renderer writes the list as source, and its compiler turns the source
      into a binary ({!Renderer.Compiler}).

    {!to_program} runs the three and returns an {!Op.Program} that holds each
    step's result. Nothing here runs a program or opens a device. *)

(** {1:lowering Lowering} *)

val full_rewrite_to_sink :
  ?optimize:bool ->
  ?beam:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  Ops.t ->
  Renderer.t ->
  Ops.t
(** [full_rewrite_to_sink ~optimize ~beam ast ren] is the kernel [ast] lowered
    for [ren], ready to be linearized.

    If [optimize] (default [true]), its loops are first simplified and optimised
    ({!Postrange.apply_opts}): with the optimisations its argument lists, or
    else, if its argument asks for a beam search of width [w] greater than [0],
    with [beam w], or else with {!Heuristic.hand_coded_optimizations} unless the
    setting {!Helpers.noopt} holds. The settings {!Helpers.disable_fast_idiv}
    and {!Helpers.transcendental} choose how divisions by constants and
    transcendental functions are decomposed ({!Decomp_op.late_patterns},
    {!Transcendental.patterns}). When the setting {!Helpers.spec} is [1] or
    more, [ast] is checked against {!Spec.tensor} and the result against
    {!Spec.program}; when the environment variable [DBGTV] is set, a result that
    fails is first printed on standard output ({!Render.pp_uops}).

    Raises [Invalid_argument] if [optimize] holds and [ast]'s argument is not a
    {!Ops.kernel_info}, if its beam search is asked for without [beam], if an
    optimisation its argument lists does not apply, with the reason, if a check
    fails, or if a pass does. *)

(** {1:linearizing Linearizing} *)

val line_rewrite :
  Ops.t list ->
  ('ctx, Ops.t * Ops.t list) Ops.Pattern_matcher.t ->
  'ctx ->
  Ops.t list
(** [line_rewrite l m ctx] rewrites the instruction list [l], each node after
    its sources: each node is rebuilt on the results of its sources, and a rule
    of [m] that matches the rebuilt node gives its result, the node its users
    see, and the instructions that replace it; a node no rule matches stays. [l]
    must list each node after its sources. *)

val pm_linearize_cleanups : (unit, Ops.t * Ops.t list) Ops.Pattern_matcher.t
(** [pm_linearize_cleanups] makes a gated {!Op.Store} a store inside an {!Op.If}
    on its gate, closed by an {!Op.Endif}. It raises [Invalid_argument] on an
    {!Op.If} or {!Op.Endif} already in the list. *)

(** {1:programs Programs} *)

val to_program :
  ?beam:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  Ops.t ->
  Renderer.t ->
  Ops.t
(** [to_program ~beam ast ren] is the program of [ast] for [ren]: an
    {!Op.Program} whose argument is its {!Ops.program_info} for [ren]'s target,
    and whose sources are:
    + the lowered sink ({!full_rewrite_to_sink}, optimised unless [ast] is
      tagged), whose kernel information holds the program's cost estimates
      ({!Renderer.Estimates.of_uops}, not counting index arithmetic);
    + an {!Op.Linear} of the instructions: the sink linearized
      ({!Linearizer.linearize}), gated stores made conditional
      ({!pm_linearize_cleanups}), and each {!Op.Alloc} made a {!Op.Buffer};
    + an {!Op.Source} of the source [ren] writes for them;
    + an {!Op.Binary} of that source compiled by [ren]'s compiler
      ({!Renderer.Compiler.compile_cached}).

    [ast] is a kernel's sink, or a program whose sources are the first of these,
    from a sink already lowered; the missing ones are then added, and the
    argument if it is not a {!Ops.program_info}.

    Programs are kept: a second call with an equal [ast], a renderer of the same
    name and target, and the same values of the settings that shape the program
    ({!Helpers.noopt}, {!Helpers.emulated_dtypes}, {!Helpers.use_tc},
    {!Helpers.disable_fast_idiv}, {!Helpers.transcendental},
    {!Helpers.allow_tf32}, {!Helpers.default_float}, {!Helpers.default_int},
    {!Helpers.tc_select}, {!Helpers.tc_opt} and {!Helpers.tc_min_globals})
    returns the program the first made. [to_program] may be called from several
    domains at once, and makes each program once: a call that asks for a program
    another domain is making waits for it. A call that raises keeps nothing, and
    the next call makes the program anew.

    Programs are also kept on disk ({!Helpers.Diskcache}, table ["to_program"]),
    for later processes: a program is read back for the same [ast], renderer and
    settings, the values of the setting {!Helpers.tuple_order} and of the
    environment variables [MV], [MV_BLOCKSIZE], [MV_THREADS_PER_ROW],
    [MV_ROWS_PER_THREAD], [ALIGNED] and [EXPAND_SSA], and the same sources of
    this library. A program read back is not checked against {!Spec.program}
    again. An entry that does not read as a program is made anew and replaced.
    The program of a kernel that asks for a beam search is not kept on disk.

    When the setting {!Helpers.debug} is [3] or more, the optimisations applied
    are printed on standard output, from [4] the source too, and from [7] the
    binary is disassembled ({!Renderer.Compiler.disassemble}).

    Raises [Invalid_argument] if [ast] is neither an {!Op.Sink} with kernel
    information nor an {!Op.Program}, or as {!full_rewrite_to_sink} does. Raises
    {!Renderer.Compiler.Compile_error} if the compiler rejects the source. *)
