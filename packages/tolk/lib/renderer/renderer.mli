(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Renderers: what code generation may emit for a target, and how it becomes a
    program.

    A renderer describes one target to code generation: the launch sizes and
    memory it offers, the operations and data types it has natively, its tensor
    cores, the rewrites it needs last, how a linearized kernel is written as
    source, and the compiler that turns that source into a binary. Each target
    builds its renderer with {!v}.

    A renderer holds only facts its target fixes, never those of a device's
    physical part, such as its compute units or cache bytes: every device of
    a target runs one program of a kernel, which computes the same bits on
    each. A program is kept under its renderer's name and target
    ({!Codegen.program_key}), so renderers of one name and target must make
    the same programs, and a renderer made with other facts takes another
    name. *)

(** {1:storage Storage} *)

val with_storage : Ops.t -> Dtype.t -> Ops.t
(** [with_storage u dt] is the access [u] restated at [dt] on the storage it
    reaches through first sources: the {!Op.Param}, {!Op.Buffer} or {!Op.Alloc}
    found from [u] by following first sources gets element type [dt], and the
    nodes on the way are rebuilt on it. An access takes its data type from its
    storage, so accessing memory at another type retypes the storage.

    Raises [Invalid_argument] if the first sources end before reaching a
    storage. *)

(** {1:estimates Cost estimates} *)

(** Cost estimates of kernels. *)
module Estimates : sig
  type t = Ops.estimates
  (** The type for estimates. *)

  val zero : t
  (** [zero] is the estimate of no work: every count is [0]. *)

  val add : t -> t -> t
  (** [add e0 e1] is the estimate of both [e0]'s and [e1]'s work: each count is
      the sum of theirs, exactly: an [Int] where it fits one, else a constant
      node. *)

  val simplify : t -> t
  (** [simplify e] is [e] with each count simplified ({!Shape.ssimplify}). *)

  val of_uops : ?ignore_indexing:bool -> Ops.t list -> t
  (** [of_uops ~ignore_indexing uops] estimates the linearized kernel [uops]:
      - [ops] counts arithmetic and logic operations, a multiply-add as two, and
        a tensor core product as [2 N M K] shared among the threads of its warp;
      - [lds] counts the bytes of each load and store, except those of
        registers;
      - [mem] counts, for each parameter, the bytes loaded from it and the bytes
        stored into it, each at most the parameter's size: a buffer read again
        is counted once.

      Each operation counts once per iteration of the ranges and hardware
      indices around it. A range with no trip count, a loop, counts as one
      iteration, a trip count that depends on a hardware index is taken at
      index [0], and one that depends on another range, as a triangular loop's
      does, or on a value loaded from memory, at its greatest value: a count
      reads no value but the run's variables. [ops] and each parameter's part
      of [mem] are simplified; [lds] is not. With [ignore_indexing] (default
      [false]), the operations that compute the indices of {!Op.Index} and
      {!Op.Shrink} nodes, up to the {!Op.End} or {!Op.Backedge} of any loop
      they read, are not counted.

      The counts are exact: an integer count that does not fit an [int], as of
      a loop bounded by a value loaded from memory, is a node of constants,
      which [ops] and [mem] simplify to one.

      Raises [Invalid_argument] if an {!Op.End} or {!Op.Backedge} closes no
      {!Op.Range}, or if an {!Op.Wmma} node has no tensor core argument. *)
end

(** {1:compilers Compilers} *)

(** Compilers: source to binary.

    A compiler turns the source a renderer writes into the bytes of a program.
    Compiling runs nothing: a compiler may call a toolchain, and returns its
    output. *)
module Compiler : sig
  exception Compile_error of string
  (** Raised by a compiler that rejects its source, with the toolchain's
      message. *)

  type t
  (** The type for compilers. *)

  val v :
    ?cachekey:(unit -> string) ->
    ?disassemble:(string -> unit) ->
    (string -> string) ->
    t
  (** [v ~cachekey ~disassemble compile] is the compiler that compiles a source
      with [compile]. [disassemble] prints a binary as instructions on standard
      output; it defaults to printing nothing. [cachekey ()] names everything
      besides a source that determines its binary: the toolchain, its version
      and its options. The caches of programs and searches key on it, and
      binaries are kept in the {!Helpers.Diskcache} table it names while the
      setting {!Setting.ccache} holds. [cachekey] is called when the name is
      first needed, and again only after it raised. *)

  val batch :
    ?cachekey:(unit -> string) ->
    ?disassemble:(string -> unit) ->
    (string list -> string list) ->
    t
  (** [batch ~cachekey ~disassemble compile] is {!v}'s compiler, whose
      [compile] is given several sources at once and returns their binaries in
      their order. Each binary is the one [compile] makes of its source given
      alone. *)

  val cachekey : t -> string option
  (** [cachekey c] is the name of [c]'s toolchain and options, if [c] has one:
      the table of its binaries, whether or not they are kept.

      Raises what [c]'s [cachekey] function raises. *)

  val compile : t -> string -> string
  (** [compile c src] is [src] compiled by [c].

      Raises {!Compile_error} if [c] rejects [src]. *)

  val kept : t -> string -> string option
  (** [kept c src] is the binary of [src] held in [c]'s disk cache table while
      {!Setting.ccache} holds, if any: what {!compile_cached} returns without
      compiling. An entry that does not read, or cannot be read, is none.

      Raises what {!cachekey} raises. *)

  val compile_cached : t -> string -> string
  (** [compile_cached c src] is the binary of [src] held in [c]'s disk cache
      table, or else [compile c src] ({!compile}), which is then kept there,
      while {!Setting.ccache} holds; otherwise it is [compile c src]. An entry
      that does not read, or cannot be read, is compiled anew and replaced.

      Raises {!Compile_error} as {!compile}, what {!cachekey} raises, and
      [Invalid_argument] naming [src] if it must be compiled under
      {!Setting.assert_compile}. *)

  val compile_all : t -> string list -> string list
  (** [compile_all c srcs] is [List.map (compile_cached c) srcs], with the
      sources that [c]'s disk cache does not hold, each once, compiled by one
      call of [c]'s function: one of {!batch}'s compiles them in one run of its
      toolchain.

      Raises as {!compile_cached} does, for the first of [srcs] that raises. *)

  val disassemble : t -> string -> unit
  (** [disassemble c lib] prints the binary [lib] as instructions on standard
      output, if [c] can. *)
end

(** {1:renderers Renderers} *)

type registers = {
  count : int;  (** The number of vector registers. *)
  bytes : int;  (** The bytes of a vector register. *)
}
(** The type for a target's vector register file. *)

type t = private {
  name : string;
      (** The renderer's name, which tells apart renderers of one target. *)
  target : Helpers.Target.t;  (** The target rendered for. *)
  suffix : string;
      (** A mark of the renderer's output, which tells apart programs of one
          device that different renderers made. *)
  supports_float4 : bool;
      (** Loads and stores of consecutive elements may be merged into one vector
          access. *)
  half8 : bool;
      (** Vectors of eight 16-bit floats are written, so that a copy of
          16 bytes of them from global into local memory merges into one
          ({!Coalesce.memory_coalescing}). *)
  registers : unit -> registers option;
      (** [registers ()] is the target's vector register file, if known, asked
          when first needed. It sets the widest merged access
          ({!Coalesce.memory_coalescing}) and, on the [CPU] device, how many
          lanes a kernel's accumulators hold ({!Heuristic}). *)
  vector_lanes : int;
      (** The most lanes of a vector that arithmetic, logic, comparisons,
          selections and conversions compute at once: an operation of more
          lanes computes them in vectors of [vector_lanes] and a last of the
          rest, its last [vector_lanes + 1] lanes as two vectors of
          [vector_lanes / 2 + 1] and [vector_lanes - vector_lanes / 2] where
          one lane would be left. At [1] or less, each lane is computed
          apart. *)
  has_local : bool;
      (** The target launches workgroups of several threads, numbered by local
          indices. *)
  has_shared : bool;  (** The threads of a workgroup share memory. *)
  global_max : int list;
      (** The greatest number of workgroups on each axis, [x], [y] and [z]. *)
  local_max : int list;
      (** The greatest number of threads of a workgroup on each axis. *)
  global_prod_max : int list option;
      (** If any, the greatest number of threads on each axis, workgroups times
          their threads. *)
  shared_max : int;  (** The bytes of memory a workgroup shares. *)
  async_copy : bool;
      (** A thread copies from global to shared memory without waiting for
          the copy, which a barrier then waits for: PTX's [cp.async], from
          sm_80. Staged operands then take two stages ({!Stage.depth}). *)
  tensor_cores : Tc.t list;  (** The tensor cores, in order of preference. *)
  per_byte_first : int;
      (** The multiply-adds per byte of a contraction's streamed operand up to
          which it streams before a tensor core is tried ({!Heuristic}): where
          a core's tiles start to run faster than threads that stream the
          operand, a crossover of the target's achieved rates. It moves the
          order of a sum, so it is fixed per target, never read from a
          device. *)
  core_threads : int;
      (** The threads a kernel keeps once a tensor core's splits apply: a
          split applies only while [core_threads / t] workgroups of the core's
          [t] threads remain ({!Heuristic}). Fewer leave memory short of loads
          in flight; more refuse reuse. It is a bound fitted to the products
          timed on one part of a target: 16384 on an RTX 5000 Ada (sm_89),
          and Metal's 32768 on an M1 Max, other Apple parts unmeasured. It
          moves the order of a sum, so it is fixed per target, never read
          from a device. *)
  stream_threads : int;
      (** The threads that keep a part's memory busy, about as many loads in
          flight as its latency times its bandwidth. A kernel that streams an
          operand runs about as many ({!Heuristic}). It is a bound fitted to
          the products timed on one part, 32768 on an RTX 5000 Ada (sm_89),
          which every target takes until it measures its own. Fixed per
          target, as [core_threads] is. *)
  local_workgroups : int;
      (** The fewest workgroups a split of threads into larger workgroups
          leaves, so that each compute unit of a part takes about as many
          ({!Heuristic}). A local split of a tensor core's [N] keeps the
          threads and packs a workgroup's warps onto one compute unit, which
          balances the units coarser: on an RTX 5000 Ada's 100 SMs a split
          that left 512 workgroups ran as fast as one that left 256, and 128
          or 144 lost up to 1.34 times. It governs a core whose operands are
          not staged; a staged core's split shares each copied tile among its
          warps. Fixed per target, as [core_threads] is. *)
  extra_matcher : (unit, Ops.t) Ops.Pattern_matcher.t;
      (** The rewrites the target needs after every other. *)
  launch_members : int;
      (** The most members a launch's program holds ({!Shape.launch}): a
          launch calls each member on its own, and the target's compiler
          takes longer for each past this count. A larger launch runs as
          several. *)
  launch_args : int;
      (** The most arguments a launch passes: on CUDA, the 4 KiB of parameters
          a kernel takes, 512 of 8 bytes. A launch of more runs as several. *)
  launch_locals : bool;
      (** Whether a member of a launch can declare workgroup memory. A member
          that fixes its threads or reduces may hold values there; where this
          is [false], as on Metal, such a member runs alone. *)
  packed_args : bool;
      (** A kernel takes its arguments as bytes: one parameter, which holds
          each argument at its offset ({!Device.Tiny_elf.iter_sig}), a buffer
          by its address. The kernel loads each argument as its type, so the
          bytes must start on 8 bytes, the largest an argument takes. *)
  code_for_op : (Op.t * (string list -> Dtype.t -> string)) list;
      (** The operations the target has natively, each with how it is written:
          [f operands dtype] is the expression of the operation on the
          expressions [operands], producing [dtype]. Code generation decomposes
          the others. *)
  native : Dtype.t -> bool;
      (** [native dt] is [true] iff the target has values of [dt]; see
          {!supported_dtypes}. *)
  fma : Dtype.t -> bool;
      (** [fma dt] holds only where an {!Op.Mulacc} of the float [dt], as the
          target writes it, rounds once to [dt], subnormals aside on a
          target that flushes them. Code generation rewrites a float
          narrower than a float32 otherwise, unless its multiply-add in a
          float32 already rounds once ({!Codegen.full_rewrite_to_sink}). *)
  render : Ops.t list -> string;
      (** [render uops] is the source of the linearized kernel [uops]. *)
  compiler : Compiler.t;  (** The compiler of the rendered source. *)
}
(** The type for renderers. *)

val v :
  ?name:string ->
  ?suffix:string ->
  ?supports_float4:bool ->
  ?half8:bool ->
  ?registers:(unit -> registers option) ->
  ?vector_lanes:int ->
  ?has_local:bool ->
  ?has_shared:bool ->
  ?global_max:int list ->
  ?local_max:int list ->
  ?global_prod_max:int list ->
  ?shared_max:int ->
  ?async_copy:bool ->
  ?tensor_cores:Tc.t list ->
  ?per_byte_first:int ->
  ?core_threads:int ->
  ?stream_threads:int ->
  ?local_workgroups:int ->
  ?extra_matcher:(unit, Ops.t) Ops.Pattern_matcher.t ->
  ?launch_members:int ->
  ?launch_args:int ->
  ?launch_locals:bool ->
  ?packed_args:bool ->
  ?code_for_op:(Op.t * (string list -> Dtype.t -> string)) list ->
  ?native:(Dtype.t -> bool) ->
  ?fma:(Dtype.t -> bool) ->
  ?render:(Ops.t list -> string) ->
  ?compiler:Compiler.t ->
  Helpers.Target.t ->
  t
(** [v target] is the renderer for [target] with these fields. The defaults
    describe a target that renders nothing:
    - [name] is ["Renderer"] and [suffix] is [""];
    - [supports_float4], [has_local] and [has_shared] are [true], [registers]
      gives [None], [half8] is [false] and [vector_lanes] is [1];
    - [global_max] and [local_max] are [0x8FFFFFFF] on each axis, the greatest
      size a 32-bit signed index holds, and [global_prod_max] is [None];
    - [shared_max] is [32768], and [async_copy] is [false];
    - [tensor_cores] and [code_for_op] are empty, and so are the rules of
      [extra_matcher];
    - [per_byte_first] is [64], [sm_89]'s crossover, [core_threads] is
      [16384], [stream_threads] [32768] and [local_workgroups] [256],
      [sm_89]'s, which every target takes until its own are measured;
    - [launch_members] is [max_int], until a target's compiler is measured;
    - [launch_args] is [max_int];
    - [launch_locals] is [true];
    - [packed_args] is [false];
    - [native] holds for every data type, and [fma] for {!Dtype.Float32} and
      {!Dtype.Float64};
    - [render] raises [Invalid_argument];
    - [compiler] returns its source unchanged, and caches nothing. *)

val with_compiler : Compiler.t -> t -> t
(** [with_compiler c r] is [r] with its rendered source compiled by [c]. *)

val supported_dtypes : t -> Dtype.t list
(** [supported_dtypes r] is the data types of {!Dtype.all}, in order, that [r]
    has natively, without {!Dtype.Float64} if {!Dtype.Int64} is emulated
    (setting {!Setting.emulated_dtypes}): a double cannot be bitcast without a
    64-bit integer.

    Raises [Invalid_argument] if {!Setting.emulated_dtypes} names no data type
    ({!Dtype.of_string}). *)
