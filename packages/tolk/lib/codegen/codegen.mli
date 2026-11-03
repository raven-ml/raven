(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Code generation: from a kernel to a program.

    A kernel is the sink of a graph of loads, stores and arithmetic over ranges,
    with its {!Ops.type-kernel_info} as argument. Code generation turns it into
    a program for a renderer ({!Renderer.t}) in three steps:
    - {!full_rewrite_to_sink} optimises the kernel's loops, then lowers it to
      the nodes the renderer writes: vector axes are expanded into lanes,
      reductions into accumulators, loops into launch dimensions, vectors into
      scalars and merged accesses, and the operations and types the target lacks
      into ones it has;
    - {!Linearizer.linearize} orders those nodes into a list of instructions;
    - the renderer writes the list as source, and its compiler turns the source
      into a binary ({!Renderer.Compiler}).

    {!to_program} runs the three and returns an {!Op.Program} that holds each
    step's result, which it keeps for later calls and processes. {!linearize}
    then {!compile} run the same steps and keep nothing, for programs that live
    only as long as their caller, as a search's candidates do. Nothing here runs
    a program or opens a device. *)

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
    setting {!Setting.noopt} holds. The settings {!Setting.disable_fast_idiv}
    and {!Setting.transcendental} choose how divisions by constants and
    transcendental functions are decomposed ({!Decomp_op.late_patterns},
    {!Transcendental.val-patterns}). When the setting {!Setting.spec} is [1] or
    more, [ast] is checked against {!Spec.tensor} and the result against
    {!Spec.program}; when {!Setting.dbgtv} is not empty, a result that fails is
    first printed on standard output ({!Render.render_uir}). When
    {!Setting.debug} is [5] or more, [ast] is printed so before it is
    lowered.

    Raises [Invalid_argument] if [optimize] holds and [ast]'s argument is not a
    {!Ops.type-kernel_info}, if its beam search is asked for without [beam], if
    an optimisation its argument lists does not apply, with the reason, if a
    check fails, or if a pass does. *)

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
    + the lowered sink ({!full_rewrite_to_sink}, optimised unless [ast]'s
      kernel information is [optimized]), whose kernel information holds the
      program's cost estimates
      ({!Renderer.Estimates.of_uops}, not counting index arithmetic);
    + an {!Op.Linear} of the instructions: the sink linearized
      ({!Linearizer.linearize}), gated stores made conditional
      ({!pm_linearize_cleanups}), and each {!Op.Alloc} made a {!Op.Buffer};
    + an {!Op.Source} of the source [ren] writes for them;
    + an {!Op.Binary} of that source compiled by [ren]'s compiler
      ({!Renderer.Compiler.compile_cached}).

    A call of a sink inside [ast] runs its body as part of the program: before
    [ast] is lowered, each such body, its own calls first, is lowered alone
    without optimisation ({!full_rewrite_to_sink}[ ~optimize:false]) and becomes
    the {!Op.Linear} of its instructions, made as the program's are. A call of a
    program not yet compiled becomes the {!Op.Linear} of the program's
    instructions; such a program must have been made as a body another kernel
    calls ({!Ops.kernel_info.kernel} [false]) where [ren] packs arguments
    ({!Renderer.t.packed_args}), and is refused otherwise, its arguments being
    packed for a launch. A call of a compiled program is refused. Each body is named
    after its call ({!Helpers.to_function_name}), a second body of one name in
    [ast] followed by [_1], and so on. The renderer writes such a body as a
    function ({!Cstyle}). A body that stores to LOCAL memory writes each of its
    call's LOCAL arguments, and one that loads from LOCAL memory reads them:
    workgroup barriers order these accesses against the program's, as they
    order the program's own, a read of a written buffer waiting at one and a
    loop that writes and reads a buffer ending each trip with one. A body
    is a scope of its own: the program's loops, the slots of its variables
    (after those of its parameters) and of the storage it makes (after its own)
    do not depend on a body's, even where a node of the body is one of the
    program's.

    [ast] is a kernel's sink. [to_program ast ren] and
    [compile (linearize ast ren) ren] are equal programs: [to_program] is the
    pipeline that keeps what it makes, {!linearize} and {!compile} the one that
    keeps nothing.

    Programs are kept: a second call with an [ast] equal to the first's but
    for which device each parameter that names one names, a renderer of the
    same name and target whose compiler has the same name of its toolchain and
    options ({!Renderer.Compiler.cachekey}), if any, and the same values of the
    settings and variables that shape what compilation makes
    ({!Setting.shaping}) returns the program the first made, each parameter
    naming the device [ast]'s parameter of its slot names. Its source and
    binary are the first's, and name no device. [to_program] may be
    called from several domains at once, and makes each program once: a call
    that asks for a program another domain is making waits for it. A call that
    raises keeps nothing, and the next call makes the program anew.

    Programs are also kept on disk ({!Helpers.Diskcache}, table ["to_program"]),
    for later processes, up to their source: a program is read back under the
    same key, the same table of its compiler's binaries
    ({!Renderer.Compiler.cachekey}), which names the compiler and its options,
    and the same sources of this library, and its binary is then the one that
    table keeps for its source ({!Renderer.Compiler.compile_cached}). A program
    read back is not checked against {!Spec.program} again. An entry that does
    not read as a program up to its source is made anew and replaced. A program
    is kept on disk only while its compiler keeps its binaries there, and the
    program of a kernel that asks for a beam search is not.

    When the setting {!Setting.debug} is [3] or more, the optimisations applied
    are printed on standard output, from [4] the source too, and from [7] the
    binary is disassembled ({!Renderer.Compiler.disassemble}), for a program
    read back from disk too.

    Raises [Invalid_argument] if [ast] is not an {!Op.Sink} with kernel
    information, if a call of a sink inside it has no name, or as
    {!full_rewrite_to_sink} does. Raises
    {!Renderer.Compiler.Compile_error} if the compiler rejects the source. *)

val to_programs :
  ?beam:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  (Ops.t * Renderer.t) list ->
  unit
(** [to_programs ~beam kernels] makes the program of each kernel [(ast, ren)]
    of [kernels] that {!read}, run on the calling domain, does not find, as
    {!to_program}[ ~beam ast ren] does and keeps it, so that a later
    {!to_program} returns it; kernels whose programs are all found spawn no
    domain. Up to as many workers as the setting {!Setting.parallel} allows
    run on {!Worker.map}; each takes kernels to lower and render until none is
    left, then compiles its sources, each compiler's by one call of
    {!Renderer.Compiler.compile_all}, so that a compiler that compiles several
    sources in one run of its toolchain ({!Renderer.Compiler.batch}) starts it
    once per worker. When a kernel asks for a beam search, whose timings
    concurrent compiles would disturb, each kernel is made in turn, in order,
    by {!to_program}. A kernel another domain is making at once may be made
    twice, and the first made is kept.

    Raises what {!to_program} raises, that of the first worker that raises,
    and then keeps no program. *)

val program_key : Ops.t -> Renderer.t -> string
(** [program_key ast ren] is the key under which {!to_program} keeps the
    program of the kernel's sink [ast] for [ren], in memory and on disk: two
    kernels of one key take one program. Kernels that differ only in the order
    of a commutative index operation's operands have one key, and one program
    made from either, with one source. The key names [ren] by its name, target
    and compiler, and no device: renderers that agree in those make one
    program ({!Renderer}). *)

val programs : unit -> int
(** [programs ()] is the number of programs {!to_program} and {!read} have
    made in this process, from their kernels or read back from disk. A call
    that finds its program kept in memory makes none, and one that raises or
    reads none makes none. *)

val read : Ops.t -> Renderer.t -> Ops.t option
(** [read ast ren] is the program {!to_program} returns for [ast] and [ren] when
    it can do so without rendering or compiling: the program this process has
    made under the same key, or else one kept on disk whose binary its
    compiler's table keeps too, which [read] then keeps as {!to_program}
    would. It is [None] otherwise, and then renders, compiles, shows and keeps
    nothing. It waits for no program being made: one another domain is making
    is [None].

    Raises [Invalid_argument] as {!to_program} does before rendering. *)

val pack_args : Ops.t -> Ops.t list -> Ops.t
(** [pack_args sink params] is the lowered kernel [sink] reading its
    parameters [params] as bytes: one parameter, [args], of eight bytes per
    parameter, holds each at its offset ({!Device.Tiny_elf.iter_sig}). A scalar
    is the value loaded there, and a buffer the address loaded there, read as
    the buffer ({!Op.Refparam}). [params] must be in slot order, the order a
    launch lays the arguments out in; another order reads the wrong bytes.
    {!linearize} packs so, after taking its estimates, the parameters of a
    kernel the runtime launches ({!Ops.kernel_info.kernel}) for a renderer with
    {!Renderer.t.packed_args}. *)

val linearize : Ops.t -> Renderer.t -> Ops.t
(** [linearize ast ren] is the program of the kernel's sink [ast] for [ren] up
    to its instructions: an {!Op.Program} whose sources are the first two of
    {!to_program}'s, the lowered sink and the {!Op.Linear} of its instructions,
    and whose argument is its {!Ops.program_info}. It renders, compiles and
    keeps nothing.

    Raises as {!to_program} does before rendering. *)

val compile : Ops.t -> Renderer.t -> Ops.t
(** [compile prg ren] is the program {!linearize} made, [prg], completed for
    [ren] with the source [ren] writes and its binary, made anew and kept
    nowhere: neither the program, in memory or on disk, nor its binary, which
    [ren]'s compiler makes ({!Renderer.Compiler.compile}) without reading or
    filling its disk cache table, so that {!Setting.assert_compile} does not
    apply. A beam search hands it a renderer whose compiler reads and fills
    that table ({!Search.beam_search}).

    When the setting {!Setting.debug} is [4] or more, the source is printed on
    standard output, and from [7] the binary is disassembled.

    Raises [Invalid_argument] if [prg] is not a program {!linearize} made, and
    {!Renderer.Compiler.Compile_error} if the compiler rejects the source. *)
