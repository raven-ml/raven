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
    a library's file ([C.findlib]). A library that compiles with tolk declares
    its settings here too, so that the caches key on those that reach its
    output.

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

(** What a setting changes. *)
type reach =
  | Output
      (** What compilation makes: programs, schedules and the optimisations a
          search finds, whose caches key on its value ({!shaping}). *)
  | Process
      (** Only how the process runs: what it prints, keeps or checks, where it
          finds its tools, or how many domains compile. *)

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
    program ([Jit]), from [JITBEAM]. [None], the default, stands for {!beam}'s
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

val tc_opt : int option t
(** [tc_opt] is which kernels may use tensor cores, from [TC_OPT]. [0] admits
    kernels with a single reduce axis multiplying loaded values, [1] also
    kernels with several reduce axes and casted operands, [2] also kernels whose
    axes must be padded to the tensor core's dimensions. [None], the default,
    stands for [0] in hand-coded optimizations ([Heuristic]) and for [2] in a
    beam search ([Search]), which measures what it admits. *)

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
(** [ccache] caches compiled programs in the {!Helpers.Diskcache}, from
    [CCACHE]. Defaults to [true]. *)

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
