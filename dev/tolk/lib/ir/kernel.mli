(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Codegen-oriented DAG IR.

    [Kernel] is the memory-level graph stage of [ir_next]. Nodes describe
    indexed buffer accesses, loop structure, and late kernel operations that
    still precede linear backend emission.

    The public surface is intentionally narrow:
    - build nodes with the top-level constructors;
    - inspect nodes with {!view};
    - analyze them with {!dtype}, {!sort}, {!children}, and {!toposort};
    - validate with {!validate};
    - rewrite DAGs with {!map_children}, {!rebuild}, and {!rewrite_fixpoint}.

    Validation is intentionally relaxed for late-kernel IR: transient
    vectorized index-like values are allowed before devectorization, and
    {!Reduce} sources may be wider vectors of the same scalar type for later
    horizontal reduction lowering. *)

(** {1:types Types} *)

type t
(** A kernel DAG node. Values are hash-consed on demand with {!intern}. *)

type sort =
  | Value  (** Scalar or vector computation. *)
  | Pointer  (** Pointer into a buffer. *)
  | Index  (** Index-like value or loop variable. *)
  | Effect  (** Side-effecting node (store, barrier). *)
(** Coarse node role.

    Pointer and effect nodes are visible directly through {!sort} rather
    than recovered indirectly from validators. [Index] covers index-like
    values and loop variables. *)

type bufferize_device =
  | Device_single of string  (** A single named device. *)
  | Device_multi of string list  (** Multiple devices for sharded buffers. *)
  | Device_index of int  (** Device selected by index. *)
(** Bufferization device selector. *)

type estimate =
  | Int of int  (** Concrete count. *)
  | Symbolic of string  (** Symbolic expression (e.g. depends on a variable). *)
(** Static or symbolic cost estimate. *)

type estimates = {
  ops : estimate;  (** Arithmetic operation count. *)
  lds : estimate;  (** Local data share (LDS) access count. *)
  mem : estimate;  (** Global memory access count. *)
}
(** Kernel cost estimates. *)

module Opt : sig
  type t =
    | Local of { axis : int; amount : int }
        (** Split [axis] into local (workgroup-shared) tiles of [amount]. *)
    | Upcast of { axis : int; amount : int }
        (** Vectorize [axis] by [amount] lanes. *)
    | Unroll of { axis : int; amount : int }
        (** Unroll [axis] by [amount] iterations. *)
    | Group of { axis : int; amount : int }
        (** Split [axis] into workgroups of [amount]. *)
    | Grouptop of { axis : int; amount : int }
        (** Like {!Group} but takes the top portion of [axis]. *)
    | Thread of { axis : int; amount : int }
        (** Split [axis] into per-thread tiles of [amount]. *)
    | Nolocals  (** Disable local memory usage for this kernel. *)
    | Tc of { axis : int; tc_select : int; tc_opt : int; use_tc : int }
        (** Tensor-core configuration. *)
    | Padto of { axis : int; amount : int }
        (** Pad [axis] to a multiple of [amount]. *)
    | Swap of { axis : int; with_axis : int }
        (** Swap two axes in the schedule. *)
  (** Search and schedule options attached to kernel metadata. *)

  val to_string : t -> string
  (** [to_string opt] is a compact textual form of [opt]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats options with {!to_string}. *)
end

type bufferize_opts = {
  device : bufferize_device option;
      (** Target device, or [None] for default placement. *)
  addrspace : Dtype.addr_space;  (** Memory address space for the buffer. *)
  removable : bool;
      (** [true] if the buffer can be elided by later optimizations. *)
}
(** Bufferization options. *)

type kernel_info = {
  name : string;  (** Kernel name for debugging and codegen. *)
  axis_kinds : Axis_kind.t list;  (** Kind assignment per schedule axis. *)
  dont_use_locals : bool;
      (** [true] if local memory was disabled (e.g. via {!Opt.Nolocals}). *)
  applied_opts : Opt.t list;  (** Schedule options already applied. *)
  opts_to_apply : Opt.t list option;
      (** Remaining options to apply, or [None] for auto-tuning. *)
  estimates : estimates option;  (** Cost estimates, if computed. *)
  metadata_tags : string list;  (** User-facing tags from call-site metadata. *)
}
(** Non-semantic kernel annotations currently carried by {!Sink}. *)

type view =
  | Sink of { srcs : t list; kernel_info : kernel_info option }
      (** Kernel root gathering semantic sources. *)
  | Group of { srcs : t list }
      (** Groups effect children without producing a value. *)
  | After of { src : t; deps : t list }
      (** Sequences [src] after [deps]. *)
  | Param of { idx : int; dtype : Dtype.ptr }
      (** Global buffer parameter at index [idx]. *)
  | Param_image of { idx : int; dtype : Dtype.ptr; width : int; height : int }
      (** Image buffer parameter with pixel dimensions. *)
  | Define_local of { size : int; dtype : Dtype.ptr }
      (** Local (workgroup-shared) memory buffer of [size] elements. *)
  | Define_reg of { size : int; dtype : Dtype.ptr }
      (** Register-backed buffer of [size] elements. *)
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
      (** Scalar loop or index variable bounded by \[[lo];[hi]\]. *)
  | Bufferize of {
      src : t;
      ranges : t list;
      dtype : Dtype.ptr;
      opts : bufferize_opts;
    }  (** Materializes [src] into a buffer. *)
  | Const of { value : Const.t; dtype : Dtype.t }
      (** Compile-time constant. *)
  | Invalid_index of { dtype : Dtype.t }
      (** Invalid index sentinel. *)
  | Index of { ptr : t; idxs : t list; gate : t option; dtype : Dtype.ptr }
      (** Indexes into [ptr] with per-dimension [idxs] and optional [gate]. *)
  | Ptrcat of { srcs : t list; dtype : Dtype.ptr }
      (** Concatenates pointer bundles. *)
  | Load of { src : t; alt : t option; dtype : Dtype.t }
      (** Loads from pointer [src]. [alt] is used when gated. *)
  | Store of { dst : t; value : t; ranges : t list }
      (** Stores [value] through pointer [dst]. *)
  | Unary of { op : Op.unary; src : t; dtype : Dtype.t }
      (** Unary arithmetic or transcendental. *)
  | Binary of { op : Op.binary; lhs : t; rhs : t; dtype : Dtype.t }
      (** Binary arithmetic, logic, or comparison. *)
  | Ternary of { op : Op.ternary; a : t; b : t; c : t; dtype : Dtype.t }
      (** Ternary operation ([Where] or [Mulacc]). *)
  | Cast of { src : t; dtype : Dtype.t }
      (** Type cast. *)
  | Bitcast of { src : t; dtype : Dtype.t }
      (** Bit-preserving reinterpretation. *)
  | Vectorize of { srcs : t list; dtype : Dtype.t }
      (** Packs scalar [srcs] into a vector. *)
  | Cat of { srcs : t list; dtype : Dtype.t }
      (** Concatenates vectors with a common scalar type. *)
  | Gep of { src : t; idx : int; dtype : Dtype.t }
      (** Extracts element [idx] from a vector. *)
  | Range of { size : t; dtype : Dtype.t; axis : int; kind : Axis_kind.t }
      (** Loop or index variable over \[[0];[size-1]\] on [axis]. *)
  | End of { value : t; ranges : t list }
      (** Closes loop [ranges] around [value]. *)
  | Barrier  (** Workgroup barrier. *)
  | Special of { dim : Special_dim.t; size : t; dtype : Dtype.t }
      (** Backend-provided hardware index. *)
  | Reduce of { op : Op.reduce; src : t; ranges : t list; dtype : Dtype.t }
      (** Reduces [src] over [ranges] with [op]. *)
  | Unroll of { src : t; axes : (int * int) list; dtype : Dtype.t }
      (** Encodes unrolled lanes of [src]. *)
  | Contract of { src : t; axes : (int * int) list; dtype : Dtype.t }
      (** Contracts unrolled structure back into a vector dtype. *)
  | Wmma of {
      name : string;
      a : t;
      b : t;
      c : t;
      dtype : Dtype.t;
      dims : int * int * int;
      dtype_in : Dtype.scalar;
      dtype_out : Dtype.scalar;
      device : string;
      threads : int;
      upcast_axes : (int * int) list * (int * int) list * (int * int) list;
      reduce_axes : int list;
    }  (** Tensor-core matrix multiply-accumulate primitive. *)
  | Custom of { fmt : string; args : t list }
      (** Backend-specific effect or statement. *)
  | Custom_inline of { fmt : string; args : t list; dtype : Dtype.t }
      (** Backend-specific inline value expression. *)
(** Read-only node view. Pattern-match via {!view}. *)

(** {1:building Building} *)

val sink : ?kernel_info:kernel_info -> t list -> t
(** [sink ?kernel_info srcs] is a kernel root with semantic sources [srcs]. *)

val group : t list -> t
(** [group srcs] groups effect-like children without introducing a value. *)

val after : src:t -> deps:t list -> t
(** [after ~src ~deps] sequences [src] after [deps]. *)

val param : idx:int -> dtype:Dtype.ptr -> t
(** [param ~idx ~dtype] is a global buffer parameter. *)

val param_image : idx:int -> dtype:Dtype.ptr -> width:int -> height:int -> t
(** [param_image ~idx ~dtype ~width ~height] is an image parameter. *)

val define_local : size:int -> dtype:Dtype.ptr -> t
(** [define_local ~size ~dtype] defines a local-memory buffer. *)

val define_reg : size:int -> dtype:Dtype.ptr -> t
(** [define_reg ~size ~dtype] defines a register-backed buffer. *)

val define_var : name:string -> lo:int -> hi:int -> ?dtype:Dtype.t -> unit -> t
(** [define_var ~name ~lo ~hi ()] is a scalar loop or index variable.

    [dtype] defaults to {!Dtype.index}. *)

val bufferize :
  src:t -> ranges:t list -> dtype:Dtype.ptr -> opts:bufferize_opts -> t
(** [bufferize ~src ~ranges ~dtype ~opts] materializes [src] into a buffer. *)

val const : Const.t -> t
(** [const c] is a constant node with dtype derived from [c]. *)

val invalid_index : ?lanes:int -> unit -> t
(** [invalid_index ?lanes ()] is the invalid index sentinel.

    [lanes] defaults to [1]. *)

val index : ptr:t -> idxs:t list -> ?gate:t -> unit -> t
(** [index ~ptr ~idxs ?gate ()] indexes pointer [ptr].

    The result pointer dtype is derived from [ptr].

    Raises [Invalid_argument] if [ptr] does not produce a pointer. *)

val ptrcat : srcs:t list -> dtype:Dtype.ptr -> t
(** [ptrcat ~srcs ~dtype] concatenates pointer bundles. *)

val load : src:t -> ?alt:t -> unit -> t
(** [load ~src ?alt ()] loads from pointer [src].

    The result dtype is derived from [src].

    Raises [Invalid_argument] if [src] does not produce a pointer. *)

val store : dst:t -> value:t -> ranges:t list -> t
(** [store ~dst ~value ~ranges] stores [value] through pointer [dst]. *)

val unary : op:Op.unary -> src:t -> t
(** [unary ~op ~src] applies [op] to [src]. The result dtype is derived from
    [src]. *)

val binary : op:Op.binary -> lhs:t -> rhs:t -> t
(** [binary ~op ~lhs ~rhs] applies a binary operation.

    Comparisons return a boolean dtype with the lane count of [lhs]. Other
    operators inherit the dtype of [lhs]. *)

val ternary : op:Op.ternary -> a:t -> b:t -> c:t -> t
(** [ternary ~op ~a ~b ~c] applies a ternary operation.

    [Where] inherits the dtype of [b]. [Mulacc] inherits the dtype of [a]. *)

val cast : src:t -> dtype:Dtype.t -> t
(** [cast ~src ~dtype] casts [src] to [dtype]. *)

val bitcast : src:t -> dtype:Dtype.t -> t
(** [bitcast ~src ~dtype] bitcasts [src] to [dtype]. *)

val vectorize : srcs:t list -> t
(** [vectorize ~srcs] vectorizes scalar sources.

    Raises [Invalid_argument] if [srcs] is empty or a source dtype is not
    available. *)

val cat : srcs:t list -> t
(** [cat ~srcs] concatenates vectors with a common scalar type.

    Raises [Invalid_argument] if [srcs] is empty or a source dtype is not
    available. *)

val gep : src:t -> idx:int -> t
(** [gep ~src ~idx] extracts element [idx] from vector [src].

    Raises [Invalid_argument] if [src] does not produce a dtype. *)

val range :
  size:t -> axis:int -> kind:Axis_kind.t -> ?dtype:Dtype.t -> unit -> t
(** [range ~size ~axis ~kind ()] is a loop/index variable over [size].

    [dtype] defaults to {!Dtype.index}. *)

val end_ : value:t -> ranges:t list -> t
(** [end_ ~value ~ranges] closes loop ranges around [value]. *)

val barrier : t
(** [barrier] is a barrier effect. *)

val special : dim:Special_dim.t -> size:t -> ?dtype:Dtype.t -> unit -> t
(** [special ~dim ~size ()] is a backend special index.

    [dtype] defaults to {!Dtype.int32}. *)

val reduce : op:Op.reduce -> src:t -> ranges:t list -> dtype:Dtype.t -> t
(** [reduce ~op ~src ~ranges ~dtype] reduces [src] over [ranges]. *)

val unroll : src:t -> axes:(int * int) list -> dtype:Dtype.t -> t
(** [unroll ~src ~axes ~dtype] encodes unrolled lanes of [src]. *)

val contract : src:t -> axes:(int * int) list -> dtype:Dtype.t -> t
(** [contract ~src ~axes ~dtype] contracts unrolled structure back into a vector
    dtype. *)

val wmma :
  name:string ->
  a:t ->
  b:t ->
  c:t ->
  dtype:Dtype.t ->
  dims:int * int * int ->
  dtype_in:Dtype.scalar ->
  dtype_out:Dtype.scalar ->
  device:string ->
  threads:int ->
  upcast_axes:(int * int) list * (int * int) list * (int * int) list ->
  reduce_axes:int list ->
  t
(** [wmma ~name ~a ~b ~c ~dtype ~dims ~dtype_in ~dtype_out ~device ~threads
    ~upcast_axes ~reduce_axes] is a tensor-core matrix multiply-accumulate
    primitive. [dims] is [(M, N, K)], [dtype_in] and [dtype_out] are the
    input and output scalar types, and [threads] is the warp thread count. *)

val custom : fmt:string -> args:t list -> t
(** [custom ~fmt ~args] is a backend-specific effect or statement node. *)

val custom_inline : fmt:string -> args:t list -> dtype:Dtype.t -> t
(** [custom_inline ~fmt ~args ~dtype] is a backend-specific inline value node.
*)

val gep_multi : src:t -> idxs:int list -> t
(** [gep_multi ~src ~idxs] extracts elements at [idxs] from vector [src].

    Returns [src] unchanged if [idxs] is [[0]] and [src] is scalar.
    Returns a single {!Gep} for one index. Returns {!Vectorize} of {!Gep}s
    for multiple indices. *)

val broadcast : t -> int -> t
(** [broadcast node n] repeats [node] into an [n]-wide vector.

    Scalars become {!Vectorize} with [n] copies. Vectors become {!Cat} of
    [n] copies. Pointer nodes return unchanged. [n <= 1] returns [node]. *)

val const_int : int -> t
(** [const_int n] is an {!Dtype.index} constant [n]. *)

val const_float : float -> t
(** [const_float x] is a {!Dtype.float32} constant [x]. *)

val const_bool : bool -> t
(** [const_bool b] is a {!Dtype.bool} constant [b]. *)

val zero_like : t -> t
(** [zero_like node] is a zero constant matching [node]'s dtype (including
    vector width). Float dtypes get [0.0], bool gets [false], integers get [0].

    Raises [Invalid_argument] if [node] has no dtype. *)

(** {1:inspection Inspecting} *)

val view : t -> view
(** [view n] is the read-only view of [n]. *)

val dtype : t -> Dtype.t option
(** [dtype n] is the value dtype of [n], if any. Effect nodes return [None]. *)

val sort : t -> sort
(** [sort n] is the coarse role of [n]. *)

val children : t -> t list
(** [children n] are the direct input nodes of [n]. *)

val toposort : t -> t list
(** [toposort root] is [root]'s dependency order, from leaves to [root]. *)

val intern : t -> t
(** [intern root] hash-conses equal nodes within the DAG reachable from [root].
*)

val is_alu : t -> bool
(** [is_alu node] is [true] for {!Unary}, {!Binary}, and {!Ternary} nodes. *)

val is_ptr : t -> bool
(** [is_ptr node] is [true] for pointer-producing nodes ({!Param},
    {!Param_image}, {!Define_local}, {!Define_reg}, {!Bufferize}, {!Index},
    {!Ptrcat}), including through {!After}/{!Cast}/{!Bitcast} wrappers. *)

val dtype_or : Dtype.t -> t -> Dtype.t
(** [dtype_or default node] is the value dtype of [node], or [default] if
    the node has no dtype. *)

module Ref_tbl : Hashtbl.S with type key = t
(** Hash table keyed by physical identity ([==]). *)

(** {1:validation Validation} *)

val validate : t -> unit
(** [validate root] checks kernel invariants.

    Raises [Failure] on the first violation. *)

(** {1:rewriting Rewriting} *)

val first_match : (t -> t option) list -> t -> t option
(** [first_match rules node] tries each rule in order, returning the first
    [Some]. Returns [None] if no rule matches. *)

val replace : t -> ?children:t list -> ?dtype:Dtype.t -> unit -> t
(** [replace node ?children ?dtype ()] rebuilds [node], substituting
    [children] and/or [dtype] where provided. Unchanged fields are preserved.

    [children] must have the same length as [children node]. [dtype] applies
    only to value-dtype nodes; pointer-dtype nodes and effect nodes ignore it.

    The result is NOT interned; call {!intern} if hash-consing is needed. *)

val map_children : (t -> t) -> view -> view
(** [map_children f v] rebuilds the direct children of [v] with [f]. *)

val rebuild : (t -> t option) -> t -> t
(** [rebuild f root] rebuilds [root] bottom-up.

    Children are rebuilt first. Then [f] is applied to the rebuilt node. When
    [f n] is [Some n'], [n'] replaces [n]. When it is [None], [n] is kept. The
    result is interned before being returned. *)

val rewrite_fixpoint : ?max_iters:int -> (t -> t option) -> t -> t
(** [rewrite_fixpoint ?max_iters f root] repeatedly applies {!rebuild} with [f]
    until the root stops changing.

    Raises [Failure] if [max_iters] rewrites are applied without reaching a
    fixpoint. [max_iters] defaults to [16]. *)

(** {1:formatting Formatting} *)

val pp_view : Format.formatter -> t -> unit
(** [pp_view] formats one node with local ids relative to that node's DAG. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats the whole DAG rooted at its argument. *)
