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
    builds its renderer with {!v}. *)

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
      the sum of theirs.

      Raises [Invalid_argument] if a count does not fit an [int]. *)

  val simplify : t -> t
  (** [simplify e] is [e] with each count simplified ({!Ops.ssimplify}). *)

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
      iteration, and a trip count that depends on a hardware index is taken at
      index [0]. [ops] and each parameter's part of [mem] are simplified; [lds]
      is not. With [ignore_indexing] (default [false]), the operations that
      compute the indices of {!Op.Index} and {!Op.Shrink} nodes, up to the
      {!Op.End} or {!Op.Backedge} of any loop they read, are not counted.

      Raises [Invalid_argument] if an {!Op.End} or {!Op.Backedge} closes no
      {!Op.Range}, if an {!Op.Wmma} node has no tensor core argument, or if a
      count does not fit an [int]. *)
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

  val cachekey : t -> string option
  (** [cachekey c] is the name of [c]'s toolchain and options, if [c] has one:
      the table of its binaries, whether or not they are kept.

      Raises what [c]'s [cachekey] function raises. *)

  val compile : t -> string -> string
  (** [compile c src] is [src] compiled by [c].

      Raises {!Compile_error} if [c] rejects [src]. *)

  val compile_cached : t -> string -> string
  (** [compile_cached c src] is the binary of [src] held in [c]'s disk cache
      table, or else [compile c src] ({!compile}), which is then kept there,
      while {!Setting.ccache} holds; otherwise it is [compile c src]. An entry
      that does not read is compiled anew and replaced.

      Raises {!Compile_error} as {!compile}, what {!cachekey} raises, and
      [Invalid_argument] naming [src] if it must be compiled under
      {!Setting.assert_compile}. *)

  val disassemble : t -> string -> unit
  (** [disassemble c lib] prints the binary [lib] as instructions on standard
      output, if [c] can. *)
end

(** {1:renderers Renderers} *)

type t = private {
  name : string;
      (** The renderer's name, which tells apart renderers of one target. *)
  target : Helpers.Target.t;  (** The target rendered for. *)
  suffix : string;
      (** A mark of the renderer's output, which tells apart programs of one
          device that different renderers made. *)
  supports_float4 : bool;
      (** Loads and stores of four or two consecutive elements may be merged
          into one vector access. *)
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
  tensor_cores : Tc.t list;  (** The tensor cores, in order of preference. *)
  extra_matcher : (unit, Ops.t) Ops.Pattern_matcher.t;
      (** The rewrites the target needs after every other. *)
  code_for_op : (Op.t * (string list -> Dtype.t -> string)) list;
      (** The operations the target has natively, each with how it is written:
          [f operands dtype] is the expression of the operation on the
          expressions [operands], producing [dtype]. Code generation decomposes
          the others. *)
  native : Dtype.t -> bool;
      (** [native dt] is [true] iff the target has values of [dt]; see
          {!supported_dtypes}. *)
  render : Ops.t list -> string;
      (** [render uops] is the source of the linearized kernel [uops]. *)
  compiler : Compiler.t;  (** The compiler of the rendered source. *)
}
(** The type for renderers. *)

val v :
  ?name:string ->
  ?suffix:string ->
  ?supports_float4:bool ->
  ?has_local:bool ->
  ?has_shared:bool ->
  ?global_max:int list ->
  ?local_max:int list ->
  ?global_prod_max:int list ->
  ?shared_max:int ->
  ?tensor_cores:Tc.t list ->
  ?extra_matcher:(unit, Ops.t) Ops.Pattern_matcher.t ->
  ?code_for_op:(Op.t * (string list -> Dtype.t -> string)) list ->
  ?native:(Dtype.t -> bool) ->
  ?render:(Ops.t list -> string) ->
  ?compiler:Compiler.t ->
  Helpers.Target.t ->
  t
(** [v target] is the renderer for [target] with these fields. The defaults
    describe a target that renders nothing:
    - [name] is ["Renderer"] and [suffix] is [""];
    - [supports_float4], [has_local] and [has_shared] are [true];
    - [global_max] and [local_max] are [0x8FFFFFFF] on each axis, the greatest
      size a 32-bit signed index holds, and [global_prod_max] is [None];
    - [shared_max] is [32768];
    - [tensor_cores] and [code_for_op] are empty, and so are the rules of
      [extra_matcher];
    - [native] holds for every data type;
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
