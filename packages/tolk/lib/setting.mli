(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Settings.

    A setting is a value of the library's global state, read from an environment
    variable when the program starts and overridable for the extent of a
    function call with {!context}. It is declared once, with its variable and
    what its value reaches. Besides settings, tolk reads only where it finds
    files: [CACHEDB] ({!Helpers.cachedb}), the system's [PATH],
    [LD_LIBRARY_PATH], [HOME] and [XDG_CACHE_HOME], and the variable that names
    a library's file ([C.findlib]). Tolk's settings are all declared here
    ({!tolk}), so that each can be bound with {!context}. A library that
    compiles with tolk declares its own settings with this module too, so that
    the caches key on those that reach its output.

    A setting has an initial value, read from its environment variable when it
    is declared, and a current value on each domain: the innermost {!context}
    override made on the domain, or else the value it had on the domain that
    spawned it, when it spawned it, or else the initial value.

    Integers are an optional [+] or [-] followed by decimal digits, where a
    single underscore may separate two digits, surrounded by any number of
    spaces, tabs, newlines, vertical tabs, form feeds and carriage returns.
    Numbers are an optional [+] or [-] followed by [inf], [infinity] or [nan] in
    any case, or by a decimal number: digits with an optional fraction ([1.5],
    [1.] or [.5]) and an optional exponent ([e] or [E], an optional sign, and
    digits), with digits and white space as for integers. *)

(** {1:reach Reach} *)

(** What a setting's value reaches.

    A compilation makes a schedule, a kernel's program, the optimizations a
    search picks or a compiled schedule from its arguments: the graph, and the
    renderer with its compiler. A setting reaches {!Output} if a change of its
    value alone can change what a compilation that returns makes, for the same
    measurements, since what a search picks depends on the times it measures. It
    reaches {!Process} otherwise: then a change makes compilation only print,
    check, keep, look up or work in parallel otherwise, or raise where it
    returned.

    A compiler names the tools it runs and their options in its cache key
    ({!Renderer.Compiler.v}), which the caches take with what it makes, so a
    setting read when a compiler is made, such as where its tools are, reaches
    the process. *)
type reach =
  | Output
      (** The caches of what compilation makes key on its value ({!shaping}). *)
  | Process  (** No cache keys on its value. *)

(** {1:settings Settings}

    Declare a setting at a module's top level, so that it is declared before
    anything is compiled. Every constructor raises [Invalid_argument] if a
    setting named [key] is already declared, or if the variable [key] holds
    something its type does not read, non-ASCII digits and white space included,
    or an integer out of [int]'s range. *)

type 'a t
(** The type for settings of type ['a]. *)

val int : reach:reach -> string -> int -> int t
(** [int ~reach key default] is a setting whose initial value is the integer the
    variable [key] holds, or [default] if [key] is unset. *)

val bool : reach:reach -> string -> bool -> bool t
(** [bool ~reach key default] is a setting whose initial value is [true] iff the
    variable [key] holds a nonzero integer, and [default] if [key] is unset. *)

val float : reach:reach -> string -> float -> float t
(** [float ~reach key default] is a setting whose initial value is the number
    the variable [key] holds, or [default] if [key] is unset. *)

val string : reach:reach -> string -> string -> string t
(** [string ~reach key default] is a setting whose initial value is the variable
    [key] as written, or [default] if [key] is unset. A variable set to the
    empty string is [""]. *)

val int_option : reach:reach -> string -> int option t
(** [int_option ~reach key] is a setting whose initial value is [Some n] if the
    variable [key] holds the integer [n], and [None] if [key] is unset: the
    setting's documentation states what [None] stands for. *)

val key : 'a t -> string
(** [key v] is the name of [v]'s environment variable. *)

val value : 'a t -> 'a
(** [value v] is [v]'s current value on the calling domain. *)

(** The type for settings bound to values. *)
type binding = B : 'a t * 'a -> binding  (** [B (v, x)] binds [v] to [x]. *)

val context : binding list -> (unit -> 'a) -> 'a
(** [context bindings f] is [f ()], run with each setting of [bindings] holding
    its bound value; when a setting is bound twice, the later binding wins. Each
    setting gets its previous value back when [f] returns or raises.

    The overrides are seen by what runs meanwhile on the calling domain, and by
    the domains it spawns meanwhile, which start with its values. Other domains
    do not see them, so domains override settings independently. *)

val shaping : unit -> (string * string) list
(** [shaping ()] is the name and the current value on the calling domain, as
    text, of each setting whose reach is {!Output}, sorted by name: what the
    caches of programs, schedules and searches key on. Two values have the same
    text only if they are equal: a number is written in hexadecimal and [None]
    as [""]. While no such setting changes on the calling domain, the result is
    the same list, and reading it allocates nothing. *)

(** {1:tolk Tolk's settings}

    Levels and counts are integers; switches are [true] when their variable
    holds a nonzero integer. *)

val debug : int t
(** [debug] is the verbosity of diagnostics printed on standard output, from
    [DEBUG]. [0] prints nothing and each level adds detail. Defaults to [0]. *)

val beam : int t
(** [beam] is the width of the beam search that picks kernel optimizations, from
    [BEAM]. [0] applies hand-coded optimizations instead. Defaults to [0]. *)

val jitbeam : int option t
(** [jitbeam] is the width of the beam search for the kernels of a captured
    schedule ([Jit]), from [JITBEAM]. [None], the default, stands for {!beam}'s
    current value. *)

val noopt : bool t
(** [noopt] disables kernel optimizations, from [NOOPT]. Defaults to [false]. *)

val no_color : bool t
(** [no_color] makes {!Helpers.colored} leave text unchanged, from [NO_COLOR].
    Defaults to [false]. *)

val use_tc : int t
(** [use_tc] is how kernel optimization uses tensor cores, from [TC]. [0] never
    uses them, [1] uses them, [2] shapes the kernel for them without emitting
    tensor core instructions. Defaults to [1]. *)

val tc_select : int t
(** [tc_select] is the tensor core kernel optimization uses, from [TC_SELECT]:
    [-1] tries the target's tensor cores in order and uses the first that fits,
    [n] uses only the [n]-th. Defaults to [-1]. *)

val tc_opt : int t
(** [tc_opt] is which kernels hand-coded optimizations ([Heuristic]) let use
    tensor cores, from [TC_OPT]. [0] admits kernels with a single reduce axis
    multiplying loaded values, [1] also kernels with several reduce axes and
    casted operands, [2] also kernels whose axes must be padded to the tensor
    core's dimensions. Defaults to [0]. *)

val tc_min_globals : int t
(** [tc_min_globals] is the number of global axes below which tensor core
    optimization does not upcast its N axis, from [TC_MIN_GLOBALS]. Defaults to
    [0]. *)

val transcendental : int t
(** [transcendental] is how code generation decomposes transcendental functions
    into polynomial approximations, from [TRANSCENDENTAL]: from [2] on, all of
    them; below, those the target does not support. Defaults to [1]. *)

val split_reduceop : bool t
(** [split_reduceop] lets scheduling split a large reduction into two kernels to
    expose more parallelism, from [SPLIT_REDUCEOP]. Defaults to [true]. *)

val no_memory_planner : bool t
(** [no_memory_planner] keeps scheduling from reusing the memory of buffers that
    are no longer needed, from [NO_MEMORY_PLANNER]. Defaults to [false]. *)

val ring : int t
(** [ring] is when allreduce uses the ring algorithm, from [RING]: [0] never,
    [1] across more than two devices on large enough inputs, [2] always.
    Defaults to [1]. *)

val all2all : int t
(** [all2all] is when allreduce uses the all-to-all algorithm, which takes
    precedence over the ring, from [ALL2ALL], with the levels of {!ring}.
    Defaults to [0]. *)

val allreduce_cast : bool t
(** [allreduce_cast] makes the allreduce of a value cast up from a 16-bit float
    exchange the 16-bit values, from [ALLREDUCE_CAST]. Defaults to [true]. *)

val allreduce_node_ndevs : int t
(** [allreduce_node_ndevs] is the number of devices per node, from
    [ALLREDUCE_NODE_NDEVS]. When positive and dividing the number of devices,
    allreduce reduces within each node before crossing nodes, device [k] of a
    node exchanging with device [k] of the others. [0] treats all devices as one
    node. Defaults to [0]. *)

val cachelevel : int t
(** [cachelevel] enables the {!Helpers.Diskcache} when positive, from
    [CACHELEVEL]. Defaults to [2]. *)

val ignore_beam_cache : bool t
(** [ignore_beam_cache] makes the beam search ignore the results it cached, from
    [IGNORE_BEAM_CACHE]. Defaults to [false]. *)

val disable_fast_idiv : bool t
(** [disable_fast_idiv] keeps code generation from replacing integer division by
    a constant with a multiplication and a shift, from [DISABLE_FAST_IDIV].
    Defaults to [true]. *)

val max_kernel_buffers : int t
(** [max_kernel_buffers] is the number of buffers one kernel may access, from
    [MAX_KERNEL_BUFFERS]. [0] uses the device's limit. Defaults to [0]. *)

val emulated_dtypes : string list t
(** [emulated_dtypes] names the data types code generation emulates with other
    types, as if the target did not support them, from [EMULATED_DTYPES]: a
    [,]-separated list, empty items dropped. Defaults to [[]]. *)

val default_float : string t
(** [default_float] names the data type of floating-point values that do not
    state one, from [DEFAULT_FLOAT]. Defaults to ["float32"]. *)

val default_int : string t
(** [default_int] names the data type of integer values that do not state one,
    from [DEFAULT_INT]. Defaults to ["int32"]. *)

val parallel : int t
(** [parallel] is the number of domains compiling kernels and running the beam
    search, from [PARALLEL]. [0] works on the calling domain. Defaults to the
    number of CPUs available to the process, bounded by its cgroup's CPU quota
    and by the runtime's maximum number of domains. *)

val spec : int t
(** [spec] is how much of the graph is checked against its specification, from
    [SPEC]. [0] checks nothing, [1] checks the graphs passed between stages, [2]
    also checks every node when it is created, [3] also computes each created
    node's shape. Defaults to [1]. *)

val check_oob : bool t
(** [check_oob] makes specification checks prove that memory accesses stay
    within their buffers, from [CHECK_OOB]. Defaults to [false]. *)

val debug_rangeify : bool t
(** [debug_rangeify] prints the steps of range assignment, from
    [DEBUG_RANGEIFY]. Defaults to [false]. *)

val tuple_order : bool t
(** [tuple_order] makes linearization order nodes of equal priority by their
    structure, from [TUPLE_ORDER]. Otherwise they keep their topological order.
    Defaults to [true]. *)

val ccache : bool t
(** [ccache] keeps compiled programs and binaries in the {!Helpers.Diskcache},
    from [CCACHE]. Defaults to [true]. *)

val allow_tf32 : bool t
(** [allow_tf32] lets float32 matrix multiplications use TF32 tensor cores on
    NVIDIA devices, from [ALLOW_TF32]. Defaults to [false]. *)

val scache : int t
(** [scache] is where schedules are cached, from [SCACHE]: [0] nowhere, [1] in
    memory, [2] or more also in the {!Helpers.Diskcache}. Defaults to [2]. *)

val disallow_broadcast : bool t
(** [disallow_broadcast] makes an elementwise operation on operands of different
    shapes fail instead of broadcasting them, from [DISALLOW_BROADCAST].
    Defaults to [false]. *)

val sum_dtype : string t
(** [sum_dtype] names the least data type that sums of floats accumulate in
    ({!Dtype.sum_acc}), from [SUM_DTYPE]. Defaults to ["float32"]. *)

val late_allreduce : bool t
(** [late_allreduce] leaves each allreduce to a call of its own
    ({!Allreduce.create_allreduce_function}), from [LATE_ALLREDUCE]. Otherwise
    sharding ({!Multi}) replaces an allreduce of a value that is not sharded
    ({!Allreduce.handle_allreduce}). Defaults to [true]. *)

val ring_allreduce_threshold : int t
(** [ring_allreduce_threshold] is the number of elements a value must exceed for
    allreduce to use the all-to-all or ring algorithm at level [1] of {!all2all}
    or {!ring}, from [RING_ALLREDUCE_THRESHOLD]. Defaults to [256000]. *)

val reduceop_split_threshold : int t
(** [reduceop_split_threshold] is how many times as many elements as its output
    a reduction's input must have for scheduling to split it, under
    {!split_reduceop}, from [REDUCEOP_SPLIT_THRESHOLD]. Defaults to [32768]. *)

val reduceop_split_size : int t
(** [reduceop_split_size] is [n] such that the first kernel of a split reduction
    outputs at most [2]{^ [n]} elements, from [REDUCEOP_SPLIT_SIZE]. Defaults to
    [22]. *)

val hcq_num_sdma : int option t
(** [hcq_num_sdma] is the number of copy queues copies take
    ({!Hcq2.sched_batches}), at least one, from [HCQ_NUM_SDMA]. [None], the
    default, stands for as many as the AMD devices the copies reach, at most
    [8], under {!all2all}, and for [1] otherwise. *)

val mv : bool t
(** [mv] lets hand-coded optimizations lay out matrix-vector products
    ({!Heuristic.hand_coded_optimizations}), from [MV]. Defaults to [true]. *)

val dmc : bool t
(** [dmc] keeps code generation from merging loads, and stores, of consecutive
    elements into vector accesses ({!Coalesce.memory_coalescing}), from [DMC].
    Defaults to [false]. *)

val allow_half8 : bool t
(** [allow_half8] lets that merging make accesses of eight 16-bit floats, from
    [ALLOW_HALF8]. Defaults to [false]. *)

val expand_ssa : bool t
(** [expand_ssa] makes C-style code generation ({!Cstyle}) give every value a
    local variable, from [EXPAND_SSA]. Otherwise a value read once is written
    into the expression that reads it. Defaults to [false]. *)

val aligned : bool t
(** [aligned] aligns the vector types of {!Cstyle.clang} to their size, from
    [ALIGNED]. Otherwise they are aligned to one byte, so that buffers at any
    address can be passed. Defaults to [true]. *)

val waves_per_sh : int t
(** [waves_per_sh] is the most waves an AMD dispatch runs on each shader array
    ({!Ops_amd}), from [WAVES_PER_SH]. [0] sets no limit. Defaults to [0]. *)

val beam_padto : bool t
(** [beam_padto] adds pads of axes to the candidates of the beam search
    ({!Search.actions}), from [BEAM_PADTO]. Defaults to [false]. *)

val beam_uops_max : int t
(** [beam_uops_max] is the number of instructions from which the beam search
    drops a candidate, none if it is [0] or less, from [BEAM_UOPS_MAX]. Defaults
    to [3000]. *)

val beam_upcast_max : int t
(** [beam_upcast_max] is the most upcast and unrolled lanes a candidate of the
    beam search may have ({!Search.get_kernel_actions}), from [BEAM_UPCAST_MAX].
    Defaults to [256]. *)

val beam_local_max : int t
(** [beam_local_max] is the most warp and local threads a candidate of the beam
    search may have, from [BEAM_LOCAL_MAX]. Defaults to [1024]. *)

val beam_min_progress : float t
(** [beam_min_progress] is the microseconds a round of the beam search must gain
    for the search to go on, from [BEAM_MIN_PROGRESS]. Defaults to [0.01]. *)

val beam_estimate : bool t
(** [beam_estimate] lets the beam search time a program that launches more than
    [65536] workgroups launching fewer, and scale its time up, from
    [BEAM_ESTIMATE]. Defaults to [true]. *)

val beam_strict_mode : bool t
(** [beam_strict_mode] makes the beam search raise what the compilation of a
    candidate raises, but [Failure], instead of dropping the candidate, from
    [BEAM_STRICT_MODE]. Defaults to [false]. *)

val beam_log_surpass_max : bool t
(** [beam_log_surpass_max] prints the candidates the beam search drops for
    exceeding {!beam_upcast_max}, {!beam_local_max} or {!beam_uops_max}, from
    [BEAM_LOG_SURPASS_MAX]. Defaults to [false]. *)

val beam_debug : int t
(** [beam_debug] is the verbosity of the beam search's diagnostics, from
    [BEAM_DEBUG]: from [1], the kernel searched, the candidates whose timing
    failed and the result; from [2], every candidate timed. Defaults to [0]. *)

val cc : string t
(** [cc] is the program {!Compiler_cpu.clang} runs, from [CC], read when the
    compiler is made. Defaults to ["clang"]. *)

val cuda_path : string t
(** [cuda_path] is the directory of the CUDA toolkit whose headers
    {!Compiler_cuda.nvrtc} includes, from [CUDA_PATH], read when the compiler is
    made. [""], the default, searches the usual directories. *)

val rocm_path : string t
(** [rocm_path] is the directory of the ROCm installation whose comgr library
    {!Compiler_amd.hip} loads, from [ROCM_PATH]. Defaults to ["/opt/rocm"]. *)

val assert_compile : bool t
(** [assert_compile] makes compiling a source whose binary no disk cache table
    holds raise ({!Renderer.Compiler.compile_cached}), from [ASSERT_COMPILE]. A
    beam search's candidates compile regardless ({!Codegen.compile}): a search
    that is not kept runs under it, and the kernel compiled with its result then
    raises if its own binary is not kept. Defaults to [false]. *)

val rewrite_stack_limit : int t
(** [rewrite_stack_limit] bounds the work list of {!Ops.graph_rewrite}, from
    [REWRITE_STACK_LIMIT]. Defaults to [250000]. *)

val debug_linearize : bool t
(** [debug_linearize] prints each node linearization places, with its position,
    operation, ranges and priority ({!Linearizer.linearize}), from
    [DEBUG_LINEARIZE]. Defaults to [false]. *)

val dbgtv : string t
(** [dbgtv], when not empty, makes code generation print the instructions of a
    program that fails its specification check ({!Codegen}), from [DBGTV].
    Defaults to [""]. *)
