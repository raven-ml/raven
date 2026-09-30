(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Settings, integer and list utilities, terminal text and the disk cache.

    Settings are the library's only global state besides the cache. Each is read
    from an environment variable when the program starts and can be overridden
    for the extent of a function call with {!context}. *)

(** {1:env Environment}

    These functions read each variable once: the first call with a given key and
    default reads the environment, and later calls with the same key and default
    return its result, whatever the environment holds by then. A call that
    raises is not remembered. *)

val getenv : string -> int -> int
(** [getenv key default] is the integer held by the environment variable [key],
    or [default] if [key] is unset. The value is an optional [+] or [-] followed
    by decimal digits, where a single underscore may separate two digits,
    surrounded by any number of spaces, tabs, newlines, vertical tabs, form
    feeds and carriage returns.

    Raises [Invalid_argument] if [key] holds anything else, non-ASCII digits and
    white space included, or an integer out of [int]'s range. *)

val getenv_float : string -> float -> float
(** [getenv_float key default] is the number held by the environment variable
    [key], or [default] if [key] is unset. The value is an optional [+] or [-]
    followed by [inf], [infinity] or [nan] in any case, or by a decimal number:
    digits with an optional fraction ([1.5], [1.] or [.5]) and an optional
    exponent ([e] or [E], an optional sign, and digits). Digits and white space
    are as for {!getenv}.

    Raises [Invalid_argument] if [key] holds anything else. *)

val getenv_string : string -> string -> string
(** [getenv_string key default] is the value of the environment variable [key],
    or [default] if [key] is unset. A variable set to the empty string is [""].
*)

(** {1:settings Settings} *)

(** Settings.

    A setting has an initial value, read from its environment variable when it
    is declared, and a current value on each domain: the innermost {!context}
    override made on the domain, or else the value it had on the domain that
    spawned it, when it spawned it, or else the initial value. *)
module Context_var : sig
  type 'a t
  (** The type for settings of type ['a]. *)

  val int : string -> int -> int t
  (** [int key default] is a setting whose initial value is
      [getenv key default].

      Raises [Invalid_argument] if a setting named [key] exists already, or if
      the variable [key] does not hold an integer. *)

  val bool : string -> bool -> bool t
  (** [bool key default] is a setting whose initial value is [true] iff the
      variable [key] holds a nonzero integer, and [default] if [key] is unset.

      Raises [Invalid_argument] if a setting named [key] exists already, or if
      the variable [key] does not hold an integer. *)

  val string : string -> string -> string t
  (** [string key default] is a setting whose initial value is
      [getenv_string key default].

      Raises [Invalid_argument] if a setting named [key] exists already. *)

  val key : 'a t -> string
  (** [key v] is the name of [v]'s environment variable. *)

  val value : 'a t -> 'a
  (** [value v] is [v]'s current value on the calling domain. *)
end

(** The type for settings bound to values. *)
type binding =
  | B : 'a Context_var.t * 'a -> binding
      (** [B (v, x)] binds setting [v] to [x]. *)

val context : binding list -> (unit -> 'a) -> 'a
(** [context bindings f] is [f ()], run with each setting of [bindings] holding
    its bound value; when a setting is bound twice, the later binding wins. Each
    setting gets its previous value back when [f] returns or raises.

    The overrides are seen by what runs meanwhile on the calling domain, and by
    the domains it spawns meanwhile, which start with its values. Other domains
    do not see them, so domains override settings independently. *)

(** {2:targets Compilation targets} *)

(** Compilation targets.

    A target selects, for one device, the renderer that generates its code, the
    architecture to generate for, and the interface and device indices to open
    it with. Empty fields are unspecified. *)
module Target : sig
  type t = {
    device : string;  (** The device, e.g. ["AMD"]. *)
    renderer : string;  (** The renderer, e.g. ["HIP"]. *)
    arch : string;  (** The architecture, e.g. ["gfx1100"]. *)
    interface : string;  (** The interface opening the device. *)
    indices : string;  (** The indices of the devices to open. *)
  }
  (** The type for targets. *)

  val parse : string -> (t, string) result
  (** [parse s] parses [s], written
      [[INTERFACE[:INDICES]+]DEVICE[:RENDERER[:ARCH]]]. [DEVICE] and [RENDERER]
      are uppercased; the other fields are kept as written. It is an error if
      [s] has more than one [+], naming [s], or if the part after the [+] has
      more than two [:], naming that part. *)

  val to_string : t -> string
  (** [to_string t] is [t] written as {!parse} reads it, without trailing
      separators. *)
end

val dev : Target.t list Context_var.t
(** [dev] is the targets requested for devices, from the variable [DEV]: a
    [;]-separated list of targets in {!Target.parse}'s syntax. It defaults to a
    single empty target.

    Raises [Invalid_argument] at initialization if [DEV] holds a malformed
    target. *)

val target : ?arch:string -> string -> Target.t
(** [target ~arch device] is the target for [device]: the first target of {!dev}
    that names [device] or no device, or an empty one, with its device set to
    [device] and its architecture to [arch] if it has none. [arch] defaults to
    [""]. *)

(** {2:list List of settings}

    Levels and counts are integers; switches are [true] when their variable
    holds a nonzero integer. *)

val debug : int Context_var.t
(** [debug] is the verbosity of diagnostics printed on standard output, from
    [DEBUG]. [0] prints nothing and each level adds detail. Defaults to [0]. *)

val beam : int Context_var.t
(** [beam] is the width of the beam search that picks kernel optimizations, from
    [BEAM]. [0] applies hand-coded optimizations instead. Defaults to [0]. *)

val noopt : bool Context_var.t
(** [noopt] disables kernel optimizations, from [NOOPT]. Defaults to [false]. *)

val no_color : bool Context_var.t
(** [no_color] makes {!colored} leave text unchanged, from [NO_COLOR]. Defaults
    to [false]. *)

val use_tc : int Context_var.t
(** [use_tc] is how kernel optimization uses tensor cores, from [TC]. [0] never
    uses them, [1] uses them, [2] shapes the kernel for them without emitting
    tensor core instructions. Defaults to [1]. *)

val tc_select : int Context_var.t
(** [tc_select] is the tensor core kernel optimization uses, from [TC_SELECT]:
    [-1] tries the target's tensor cores in order and uses the first that fits,
    [n] uses only the [n]-th. Defaults to [-1]. *)

val tc_opt : int Context_var.t
(** [tc_opt] is which kernels may use tensor cores, from [TC_OPT]. [0] admits
    kernels with a single reduce axis multiplying loaded values, [1] also
    kernels with several reduce axes and casted operands, [2] also kernels whose
    axes must be padded to the tensor core's dimensions. Defaults to [0]. *)

val tc_min_globals : int Context_var.t
(** [tc_min_globals] is the number of global axes below which tensor core
    optimization does not upcast its N axis, from [TC_MIN_GLOBALS]. Defaults to
    [0]. *)

val transcendental : int Context_var.t
(** [transcendental] is how code generation decomposes transcendental functions
    into polynomial approximations, from [TRANSCENDENTAL]: from [2] on, all of
    them; below, those the target does not support. Defaults to [1]. *)

val split_reduceop : bool Context_var.t
(** [split_reduceop] lets scheduling split a large reduction into two kernels to
    expose more parallelism, from [SPLIT_REDUCEOP]. Defaults to [true]. *)

val no_memory_planner : bool Context_var.t
(** [no_memory_planner] keeps scheduling from reusing the memory of buffers that
    are no longer needed, from [NO_MEMORY_PLANNER]. Defaults to [false]. *)

val ring : int Context_var.t
(** [ring] is when allreduce uses the ring algorithm, from [RING]: [0] never,
    [1] across more than two devices on large enough inputs, [2] always.
    Defaults to [1]. *)

val all2all : int Context_var.t
(** [all2all] is when allreduce uses the all-to-all algorithm, which takes
    precedence over the ring, from [ALL2ALL], with the levels of {!ring}.
    Defaults to [0]. *)

val allreduce_cast : bool Context_var.t
(** [allreduce_cast] makes the allreduce of a value cast up from a 16-bit float
    exchange the 16-bit values, from [ALLREDUCE_CAST]. Defaults to [true]. *)

val allreduce_node_ndevs : int Context_var.t
(** [allreduce_node_ndevs] is the number of devices per node, from
    [ALLREDUCE_NODE_NDEVS]. When positive and dividing the number of devices,
    allreduce reduces within each node before crossing nodes, device [k] of a
    node exchanging with device [k] of the others. [0] treats all devices as one
    node. Defaults to [0]. *)

val cachelevel : int Context_var.t
(** [cachelevel] enables the {!Diskcache} when positive, from [CACHELEVEL].
    Defaults to [2]. *)

val ignore_beam_cache : bool Context_var.t
(** [ignore_beam_cache] makes the beam search ignore the results it cached, from
    [IGNORE_BEAM_CACHE]. Defaults to [false]. *)

val disable_fast_idiv : bool Context_var.t
(** [disable_fast_idiv] keeps code generation from replacing integer division by
    a constant with a multiplication and a shift, from [DISABLE_FAST_IDIV].
    Defaults to [true]. *)

val max_kernel_buffers : int Context_var.t
(** [max_kernel_buffers] is the number of buffers one kernel may access, from
    [MAX_KERNEL_BUFFERS]. [0] uses the device's limit. Defaults to [0]. *)

val emulated_dtypes : string list Context_var.t
(** [emulated_dtypes] names the data types code generation emulates with other
    types, as if the target did not support them, from [EMULATED_DTYPES]: a
    [,]-separated list, empty items dropped. Defaults to [[]]. *)

val default_float : string Context_var.t
(** [default_float] names the data type of floating-point values that do not
    state one, from [DEFAULT_FLOAT]. Defaults to ["float32"]. *)

val default_int : string Context_var.t
(** [default_int] names the data type of integer values that do not state one,
    from [DEFAULT_INT]. Defaults to ["int32"]. *)

val parallel : int Context_var.t
(** [parallel] is the number of domains compiling kernels and running the beam
    search, from [PARALLEL]. [0] works on the calling domain. Defaults to the
    number of CPUs available to the process, bounded by its cgroup's CPU quota
    and by the runtime's maximum number of domains. *)

val spec : int Context_var.t
(** [spec] is how much of the graph is checked against its specification, from
    [SPEC]. [0] checks nothing, [1] checks the graphs passed between stages, [2]
    also checks every node when it is created, [3] also computes each created
    node's shape. Defaults to [1]. *)

val check_oob : bool Context_var.t
(** [check_oob] makes specification checks prove that memory accesses stay
    within their buffers, from [CHECK_OOB]. Defaults to [false]. *)

val debug_rangeify : bool Context_var.t
(** [debug_rangeify] prints the steps of range assignment, from
    [DEBUG_RANGEIFY]. Defaults to [false]. *)

val tuple_order : bool Context_var.t
(** [tuple_order] makes linearization order nodes of equal priority by their
    structure, from [TUPLE_ORDER]. Otherwise they keep their topological order.
    Defaults to [true]. *)

val ccache : bool Context_var.t
(** [ccache] caches compiled programs in the {!Diskcache}, from [CCACHE].
    Defaults to [true]. *)

val allow_tf32 : bool Context_var.t
(** [allow_tf32] lets float32 matrix multiplications use TF32 tensor cores on
    NVIDIA devices, from [ALLOW_TF32]. Defaults to [false]. *)

val scache : int Context_var.t
(** [scache] is whether schedules are cached in memory, from [SCACHE]: [0]
    not at all, [1] or more in memory. Defaults to [1]. *)

val disallow_broadcast : bool Context_var.t
(** [disallow_broadcast] makes an elementwise operation on operands of different
    shapes fail instead of broadcasting them, from [DISALLOW_BROADCAST].
    Defaults to [false]. *)

(** {1:lists Integers and lists} *)

val prod : int list -> int
(** [prod l] is the product of the elements of [l], [1] if [l] is empty. *)

val dedup : (module Hashtbl.HashedType with type t = 'a) -> 'a list -> 'a list
(** [dedup (module H) l] is [l] with each element kept only at its first
    occurrence, elements being compared with [H.equal]. *)

val argsort : int list -> int list
(** [argsort l] is the positions of the elements of [l], ordered by increasing
    element; equal elements keep their order. On a permutation it is the inverse
    permutation. *)

val all_same : ('a -> 'a -> bool) -> 'a list -> bool
(** [all_same equal l] is [true] iff every element of [l] is [equal] to the
    first. It is [true] on [[]]. *)

val get_single_element : 'a list -> 'a
(** [get_single_element l] is the element of [l].

    Raises [Invalid_argument] if [l] does not have exactly one element. *)

val ceildiv : int -> int -> int
(** [ceildiv num amt] is [num / amt] rounded up.

    Raises [Division_by_zero] if [amt] is [0]. *)

val round_up : int -> int -> int
(** [round_up num amt] is the smallest multiple of [amt] greater than or equal
    to [num], for positive [amt].

    Raises [Division_by_zero] if [amt] is [0]. *)

val floordiv : int -> int -> int
(** [floordiv x y] is [x / y] rounded down, and [0] if [y] is [0]. *)

val floormod : int -> int -> int
(** [floormod x y] is [x - floordiv x y * y]: the remainder of the division
    rounded down, with the sign of [y]. It is [x] if [y] is [0]. *)

val lo32 : int -> int
(** [lo32 x] is the low 32 bits of [x]. *)

val hi32 : int -> int
(** [hi32 x] is [x] shifted right by 32 bits, keeping its sign. *)

val data64 : int -> int * int
(** [data64 x] is [(hi32 x, lo32 x)]. *)

val data64_le : int -> int * int
(** [data64_le x] is [(lo32 x, hi32 x)]. *)

(** {1:select Selection} *)

val select_by_name :
  error:string ->
  ('a -> string) ->
  string ->
  'a list ->
  ('a list, string) result
(** [select_by_name ~error name query candidates] is the candidates [c] with
    [name c = query] in their order, or [candidates] if [query] is [""]. If
    there are none, the error is [error], followed by the name of a candidate
    that [query] may have misspelled, if one is similar enough. *)

val select_first_inited :
  error:string -> (unit -> ('a, string) result) list -> ('a, string) result
(** [select_first_inited ~error candidates] is the value of the first of
    [candidates] that initializes, trying them in order. If none does, the error
    is the only candidate's error, or [error] followed by each candidate's
    error, one per line. An exception raised by a candidate is not caught: it
    escapes, and the later candidates are not tried. *)

(** {1:text Terminal text} *)

(** The type for the colors of terminal text. *)
type color =
  | Black
  | Red
  | Green
  | Yellow
  | Blue
  | Magenta
  | Cyan
  | White
  | Bright_black
  | Bright_red
  | Bright_green
  | Bright_yellow
  | Bright_blue
  | Bright_magenta
  | Bright_cyan
  | Bright_white

val colored : ?background:bool -> color -> string -> string
(** [colored c s] is [s] wrapped in the ANSI escape sequences that paint it, or
    its background if [background] is [true], in [c]. It is [s] itself when
    {!no_color} is set. [background] defaults to [false]. *)

val time_to_str : ?w:int -> float -> string
(** [time_to_str ~w t] is the duration of [t] seconds right-aligned in [w]
    columns with two decimals, followed by a two-column unit: [s ] above 10
    seconds, [ms] above 10 milliseconds, and [us] otherwise. [w] defaults to
    [8]. *)

val size_to_str : int -> string
(** [size_to_str s] is the size of [s] bytes, with two decimals in the largest
    of [GB], [MB] and [KB] it reaches (powers of 1024), or as [s B] below a
    kilobyte. *)

val ansistrip : string -> string
(** [ansistrip s] is [s] without its ANSI escape sequences that erase a line or
    set graphic attributes. The latter run from the escape character and an
    opening bracket to the first ['m'] on the same line. *)

val ansilen : string -> int
(** [ansilen s] is the number of characters of [ansistrip s], counting UTF-8
    encoded characters once. *)

val ansipad : string -> int -> string
(** [ansipad s w] is [s] followed by the spaces that bring its {!ansilen} to
    [w], none if it has that length already. *)

val strip_parens : string -> string
(** [strip_parens s] is [s] without its first and last characters if they are
    parentheses enclosing the whole of [s], and [s] otherwise. *)

val pluralize : string -> int -> string
(** [pluralize st cnt] is [cnt] followed by [st], with an [s] unless [cnt] is
    [1]. *)

val to_function_name : string -> string
(** [to_function_name s] is [ansistrip s] with each character other than an
    ASCII letter, digit or underscore replaced by its Unicode code point in
    uppercase hexadecimal, at least two digits. The result is a valid C
    identifier unless empty or starting with a digit. *)

(** {1:cache Disk cache} *)

val cache_dir : string
(** [cache_dir] is the directory of the files tolk caches: [tolk] in
    [XDG_CACHE_HOME] if that variable is set, and otherwise in
    [~/Library/Caches] on macOS and [~/.cache] elsewhere. *)

val cachedb : string
(** [cachedb] is the directory of the {!Diskcache}, from the variable [CACHEDB].
    Defaults to [cache] in {!cache_dir}, made absolute against the current
    directory when {!cache_dir} is relative. *)

(** Persistent key-value tables.

    A table maps strings to strings across runs and processes. Entries are
    written atomically, so a reader sees a complete entry or none, and the last
    writer of a key wins. Tables are versioned with the library: entries written
    by a library whose cached data meant something else are not seen. Tables and
    keys are any strings; entries stay within {!cachedb} whatever they hold. The
    cache is disabled while {!cachelevel} is not positive. *)
module Diskcache : sig
  val get : table:string -> string -> string option
  (** [get ~table key] is the value of [key] in [table], if any. It is [None]
      while the cache is disabled.

      Raises [Failure] if the entry exists but is malformed. *)

  val put : table:string -> string -> string -> unit
  (** [put ~table key value] binds [key] to [value] in [table]. It does nothing
      while the cache is disabled.

      Raises [Sys_error] if the entry cannot be written. *)

  val clear : unit -> unit
  (** [clear ()] removes every entry of every table, of every version. Files
      under {!cachedb} that the cache did not write are kept. *)
end

(** {1:exec Programs} *)

val system : ?input:string -> string -> string
(** [system ~input cmd] is the output of the command [cmd], a program and its
    arguments separated by spaces, run with [input] on its standard input or,
    without [input], with the process's: what it writes on its standard output
    and standard error, together, without the white space that starts and ends
    it. With {!debug} at least 1, it prints how many bytes [cmd] returned and
    how long it took.

    Raises [Failure] naming [cmd] with the reason and output if [cmd] cannot run
    or exits otherwise than with status [0]. *)

val cpu_objdump : string -> unit
(** [cpu_objdump lib] prints the instructions of the object file [lib], as
    [objdump -d] prints them.

    Raises [Failure] as {!system} if [objdump] fails. *)

val amdgpu_disassemble : string -> unit
(** [amdgpu_disassemble lib] prints the instructions of the AMD GPU code object
    [lib], as [llvm-objdump -d] prints them, without the padding instructions
    that end it ([s_nop 0] and [s_code_end]). [llvm-objdump] is Homebrew's on
    macOS, and elsewhere the first of ROCm's and those named [llvm-objdump-21],
    [llvm-objdump-20] and [llvm-objdump] on [PATH].

    Raises [Failure] if there is no [llvm-objdump], or as {!system} if it fails.
*)
