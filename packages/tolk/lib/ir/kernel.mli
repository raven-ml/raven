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
    - rewrite DAGs with {!map_children} and {!graph_rewrite}.

    Validation is intentionally relaxed for late-kernel IR: transient
    vectorized index-like values are allowed before devectorization, and
    {!Reduce} sources may be wider vectors of the same scalar type for later
    horizontal reduction lowering. *)

(** {1:types Types} *)

type t
(** A kernel DAG node. Values are hash-consed: structurally identical nodes
    are physically identical, enabling correct deduplication in
    {!graph_rewrite}. *)

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
  | Symbolic of t  (** Symbolic expression depending on runtime variables. *)
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

  val axis : t -> int option
  (** [axis opt] is the axis of [opt], or [None] for [Nolocals]. *)

  val amount : t -> int option
  (** [amount opt] is the amount/arg of [opt], or [None] for [Tc], [Swap],
      and [Nolocals]. *)

  val with_amount : t -> int -> t
  (** [with_amount opt n] returns [opt] with its amount replaced by [n].
      No-op for [Tc], [Swap], and [Nolocals]. *)
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
}
(** Non-semantic kernel annotations currently carried by {!Sink}. *)

type view =
  | Sink of { srcs : t list; kernel_info : kernel_info option }
      (** Kernel root gathering semantic sources. *)
  | Group of { srcs : t list }
      (** Groups effect children without producing a value. *)
  | After of { src : t; deps : t list }
      (** Sequences [src] after [deps]. *)
  | Param of { idx : int; dtype : Dtype.Ptr.t }
      (** Global buffer parameter at index [idx]. *)
  | Param_image of { idx : int; dtype : Dtype.Ptr.t; width : int; height : int }
      (** Image buffer parameter with pixel dimensions. *)
  | Define_local of { size : int; dtype : Dtype.Ptr.t }
      (** Local (workgroup-shared) memory buffer of [size] elements. *)
  | Define_reg of { size : int; dtype : Dtype.Ptr.t; slot : int }
      (** Register-backed buffer of [size] elements at accumulator [slot]. *)
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.Val.t }
      (** Scalar loop or index variable bounded by \[[lo];[hi]\]. *)
  | Bufferize of {
      src : t;
      ranges : t list;
      dtype : Dtype.Ptr.t;
      opts : bufferize_opts;
    }  (** Materializes [src] into a buffer. *)
  | Const of { value : Const.t; dtype : Dtype.Val.t }
      (** Compile-time constant. *)
  | Vconst of { values : Const.t list; dtype : Dtype.Val.t }
      (** Vector of compile-time constants (one per lane). *)
  | Invalid_index of { dtype : Dtype.Val.t }
      (** Invalid index sentinel. *)
  | Index of { ptr : t; idxs : t list; gate : t option; dtype : Dtype.t }
      (** Indexes into [ptr] with per-dimension [idxs] and optional [gate].
          When [dtype] is [Ptr _], the node is a pointer-typed index (buffer
          address). When [dtype] is [Val _], it is a value-typed index that
          [pm_add_loads] will later wrap with {!Load}. *)
  | Ptrcat of { srcs : t list; dtype : Dtype.Ptr.t }
      (** Concatenates pointer bundles. *)
  | Load of { src : t; alt : t option; dtype : Dtype.Val.t }
      (** Loads from pointer [src]. [alt] is used when gated. *)
  | Store of { dst : t; value : t; ranges : t list }
      (** Stores [value] through pointer [dst]. *)
  | Unary of { op : Op.unary; src : t; dtype : Dtype.Val.t }
      (** Unary arithmetic or transcendental. *)
  | Binary of { op : Op.binary; lhs : t; rhs : t; dtype : Dtype.Val.t }
      (** Binary arithmetic, logic, or comparison. *)
  | Ternary of { op : Op.ternary; a : t; b : t; c : t; dtype : Dtype.Val.t }
      (** Ternary operation ([Where] or [Mulacc]). *)
  | Cast of { src : t; dtype : Dtype.t }
      (** Type cast. When [dtype] is [Ptr _], this is a pointer reinterpretation
          (e.g. widening an Index pointer for grouped loads). *)
  | Bitcast of { src : t; dtype : Dtype.Val.t }
      (** Bit-preserving reinterpretation. *)
  | Vectorize of { srcs : t list; dtype : Dtype.t }
      (** Packs scalar [srcs] into a vector.  When the sources are pointers,
          [dtype] is [Ptr _] with [v = List.length srcs]. *)
  | Vcat of { srcs : t list; dtype : Dtype.Val.t }
      (** Concatenates vectors with a common scalar type. *)
  | Gep of { src : t; idxs : int list; dtype : Dtype.Val.t }
      (** Extracts elements at [idxs] from a vector. When [idxs] has one
          element, the result is scalar. When [idxs] has multiple elements,
          the result is a vector of the extracted elements. *)
  | Range of { size : t; dtype : Dtype.Val.t; axis : int; sub : int list; kind : Axis_kind.t }
      (** Loop or index variable over \[[0];[size-1]\] on [axis]. *)
  | End of { value : t; ranges : t list }
      (** Closes loop [ranges] around [value]. *)
  | Barrier  (** Workgroup barrier. *)
  | Special of { dim : Special_dim.t; size : t; dtype : Dtype.Val.t }
      (** Backend-provided hardware index. *)
  | Reduce of { op : Op.reduce; src : t; ranges : t list; dtype : Dtype.Val.t }
      (** Reduces [src] over [ranges] with [op]. *)
  | Unroll of { src : t; axes : (int * int) list; dtype : Dtype.Val.t }
      (** Encodes unrolled lanes of [src]. *)
  | Contract of { src : t; axes : (int * int) list; dtype : Dtype.Val.t }
      (** Contracts unrolled structure back into a vector dtype. *)
  | Wmma of {
      name : string;
      a : t;
      b : t;
      c : t;
      dtype : Dtype.Val.t;
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
  | Custom_inline of { fmt : string; args : t list; dtype : Dtype.Val.t }
      (** Backend-specific inline value expression. *)
(** Read-only node view. Pattern-match via {!view}. *)

(** {1:building Building} *)

val sink : ?kernel_info:kernel_info -> t list -> t
(** [sink ?kernel_info srcs] is a kernel root with semantic sources [srcs]. *)

val group : t list -> t
(** [group srcs] groups effect-like children without introducing a value.

    Returns [src] unchanged when [srcs] is a singleton list. *)

val after : src:t -> deps:t list -> t
(** [after ~src ~deps] sequences [src] after [deps].

    Returns [src] unchanged when [deps] is empty. *)

val param : idx:int -> dtype:Dtype.Ptr.t -> t
(** [param ~idx ~dtype] is a global buffer parameter. *)

val param_image : idx:int -> dtype:Dtype.Ptr.t -> width:int -> height:int -> t
(** [param_image ~idx ~dtype ~width ~height] is an image parameter. *)

val define_local : size:int -> dtype:Dtype.Ptr.t -> t
(** [define_local ~size ~dtype] defines a local-memory buffer. *)

val define_reg : size:int -> dtype:Dtype.Ptr.t -> slot:int -> t
(** [define_reg ~size ~dtype ~slot] defines a register-backed buffer.

    [slot] is a unique accumulator index that prevents parallel reduce
    accumulators from being merged by {!intern}. *)

val define_var : name:string -> lo:int -> hi:int -> ?dtype:Dtype.Val.t -> unit -> t
(** [define_var ~name ~lo ~hi ()] is a scalar loop or index variable.

    [dtype] defaults to {!Dtype.Val.index}. *)

val bufferize :
  src:t -> ranges:t list -> dtype:Dtype.Ptr.t -> opts:bufferize_opts -> t
(** [bufferize ~src ~ranges ~dtype ~opts] materializes [src] into a buffer. *)

val const : Const.t -> t
(** [const c] is a constant node with dtype derived from [c]. *)

val vconst : values:Const.t list -> dtype:Dtype.Val.t -> t
(** [vconst ~values ~dtype] is a vector constant with one value per lane. *)

val invalid_index : ?lanes:int -> unit -> t
(** [invalid_index ?lanes ()] is the invalid index sentinel.

    [lanes] defaults to [1]. *)

val index : ptr:t -> idxs:t list -> ?gate:t -> ?as_ptr:bool -> unit -> t
(** [index ~ptr ~idxs ?gate ?as_ptr ()] indexes pointer [ptr].

    When [as_ptr] is [true] (the default), the result is a pointer-typed
    index ([dtype = Ptr _]). When [as_ptr] is [false], the result is a
    value-typed index ([dtype = Val _]) that [pm_add_loads] will later
    wrap with {!Load}.

    Raises [Invalid_argument] if [ptr] does not produce a pointer. *)

val index_raw : ptr:t -> idxs:t list -> ?gate:t -> dtype:Dtype.t -> unit -> t
(** [index_raw ~ptr ~idxs ?gate ~dtype ()] creates an Index node with an
    explicit dtype. Unlike {!index}, this does not validate [ptr] and does
    not derive the dtype from [ptr]. Used by rewrite rules that need to
    change an Index's dtype directly (e.g., [pm_add_loads]). *)

val ptrcat : srcs:t list -> dtype:Dtype.Ptr.t -> t
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
(** [cast ~src ~dtype] casts [src] to [dtype]. When [dtype] is [Ptr _], the
    result is a pointer-typed node (e.g. widening an Index for grouped loads). *)

val bitcast : src:t -> dtype:Dtype.Val.t -> t
(** [bitcast ~src ~dtype] bitcasts [src] to [dtype]. *)

val vectorize : srcs:t list -> t
(** [vectorize ~srcs] vectorizes scalar sources.

    Raises [Invalid_argument] if [srcs] is empty or a source dtype is not
    available. *)

val vcat : srcs:t list -> t
(** [vcat ~srcs] concatenates vectors with a common scalar type.

    Raises [Invalid_argument] if [srcs] is empty or a source dtype is not
    available. *)

val gep : src:t -> idx:int -> t
(** [gep ~src ~idx] extracts element [idx] from vector [src].

    Raises [Invalid_argument] if [src] does not produce a dtype. *)

val range :
  size:t -> axis:int -> ?sub:int list -> kind:Axis_kind.t -> ?dtype:Dtype.Val.t -> unit -> t
(** [range ~size ~axis ~kind ()] is a loop/index variable over [size].

    [dtype] defaults to {!Dtype.Val.index}. *)

val end_ : value:t -> ranges:t list -> ?tag:string -> unit -> t
(** [end_ ~value ~ranges ()] closes loop ranges around [value].

    [tag] sets the node's tag. Pass [~tag:"mergeable"] to mark Ends
    created by reduce-to-accumulator lowering. *)

val tag : t -> string option
(** [tag node] is the node's tag, or [None]. *)

val with_tag : string -> t -> t
(** [with_tag s node] returns a node with the same view as [node] and tag
    [Some s]. Because tags are part of the hash-consing key, the result may
    be a different physical node than [node]. *)

val barrier : t
(** [barrier] is a barrier effect. *)

val special : dim:Special_dim.t -> size:t -> ?dtype:Dtype.Val.t -> unit -> t
(** [special ~dim ~size ()] is a backend special index.

    [dtype] defaults to {!Dtype.Val.int32}. *)

val reduce : op:Op.reduce -> src:t -> ranges:t list -> dtype:Dtype.Val.t -> t
(** [reduce ~op ~src ~ranges ~dtype] reduces [src] over [ranges]. *)

val unroll : src:t -> axes:(int * int) list -> dtype:Dtype.Val.t -> t
(** [unroll ~src ~axes ~dtype] encodes unrolled lanes of [src]. *)

val contract : src:t -> axes:(int * int) list -> dtype:Dtype.Val.t -> t
(** [contract ~src ~axes ~dtype] contracts unrolled structure back into a vector
    dtype. *)

val wmma :
  name:string ->
  a:t ->
  b:t ->
  c:t ->
  dtype:Dtype.Val.t ->
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

val custom_inline : fmt:string -> args:t list -> dtype:Dtype.Val.t -> t
(** [custom_inline ~fmt ~args ~dtype] is a backend-specific inline value node.
*)

val gep_multi : src:t -> idxs:int list -> t
(** [gep_multi ~src ~idxs] extracts elements at [idxs] from vector [src].

    Returns [src] unchanged if [idxs] is [[0]] and [src] is scalar.
    Returns a single scalar {!Gep} for one index. Returns a multi-element
    {!Gep} for multiple indices. *)

val broadcast : t -> int -> t
(** [broadcast node n] repeats [node] into an [n]-wide vector.

    Scalars become {!Vectorize} with [n] copies. Vectors become {!Vcat} of
    [n] copies. Pointer nodes become {!Vectorize} with pointer vector
    width [n]. [n <= 1] returns [node]. *)

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

val dtype : t -> Dtype.t
(** [dtype n] is the dtype of [n].

    Raises [Invalid_argument] if [n] has no dtype (e.g. effect nodes). *)

val dtype_opt : t -> Dtype.t option
(** [dtype_opt n] is the dtype of [n], or [None] for effect nodes. *)

val sort : t -> sort
(** [sort n] is the coarse role of [n]. *)

val children : t -> t list
(** [children n] are the direct input nodes of [n]. *)

val toposort : t -> t list
(** [toposort root] is [root]'s dependency order, from leaves to [root]. *)

val intern : t -> t
(** [intern root] hash-conses equal nodes within the DAG reachable from [root].
*)

(* CR: replace the ad-hoc is_* and range_kind/range_axis accessors with a
   small set of *_arg projections (const_arg, range_arg, …) that return
   option types suitable for pattern matching.  See const_arg below. *)

val const_arg : t -> Const.view option
(** [const_arg node] is [Some v] when [node] is a {!Const}, where [v] is
    the constant's value as a {!Const.view}. *)

val is_alu : t -> bool
(** [is_alu node] is [true] for {!Unary}, {!Binary}, and {!Ternary} nodes. *)

val is_ptr : t -> bool
(** [is_ptr node] is [true] for pointer-producing nodes ({!Param},
    {!Param_image}, {!Define_local}, {!Define_reg}, {!Bufferize}, {!Index},
    {!Ptrcat}, {!Vectorize} with [Ptr _] dtype), including through
    {!After}/{!Cast}/{!Bitcast} wrappers. *)

val ptr_dtype : t -> Dtype.Ptr.t
(** [ptr_dtype n] is the pointer dtype of [n]. Follows through
    {!After}/{!Cast}/{!Bitcast} wrappers.

    Raises [Invalid_argument] if [n] is not a pointer-producing node. *)

val is_range : t -> bool
(** [is_range node] is [true] for {!Range} nodes. *)

val is_const : t -> bool
(** [is_const node] is [true] for {!Const} nodes. *)

val range_size : t -> t
(** [range_size node] is the [size] child of a {!Range} node.

    Raises [Invalid_argument] if [node] is not a {!Range}. *)

val range_axis : t -> int
(** [range_axis node] is the [axis] of a {!Range} node.

    Raises [Invalid_argument] if [node] is not a {!Range}. *)

val range_kind : t -> Axis_kind.t
(** [range_kind node] is the [kind] of a {!Range} node.

    Raises [Invalid_argument] if [node] is not a {!Range}. *)

val range_sub : t -> int list
(** [range_sub node] is the [sub] indices of a {!Range} node.

    Raises [Invalid_argument] if [node] is not a {!Range}. *)

val const_to_int : t -> int
(** [const_to_int node] extracts the integer value of a {!Const} node.

    Raises [Invalid_argument] if [node] is not an integer constant. *)


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
    to nodes that carry a dtype field; effect nodes ignore it.

    The result is interned (hash-consed via {!mk}). *)

val map_children : (t -> t) -> view -> view
(** [map_children f v] rebuilds the direct children of [v] with [f]. *)


val graph_rewrite : ?name:string -> (t -> t option) -> t -> t
(** [graph_rewrite ?name f root] applies [f] to every node in the DAG
    rooted at [root] in a single pass. Each node is processed at most
    once. When a rewrite produces a new node, that node is fully
    processed (its children are visited), but already-processed nodes
    are never re-visited. [name] is used in error messages. *)

val substitute : ?tags:int Ref_tbl.t -> (t * t) list -> t -> t
(** [substitute ?tags mappings root] replaces nodes in [root] by physical
    identity ([==]). Each [(old, new_)] pair causes [old] to be replaced
    with [new_].

    When [tags] is provided, tag propagation is enabled: if a replaced or
    rebuilt node has an entry in [tags], the entry is copied to the new node. *)

(** {1:analysis Analysis} *)

val backward_slice : t -> t list
(** [backward_slice root] is all nodes transitively reachable from [root]
    (walking children), in topological order (leaves first).

    {b Note.} The result includes [root] itself (as the last element). *)

val in_backward_slice : t -> t -> bool
(** [in_backward_slice needle haystack] is [true] if [needle] appears in the
    transitive dependencies of [haystack]. Uses physical identity ([==]). *)

val find_nodes : (t -> bool) -> t -> t list
(** [find_nodes pred root] returns all nodes in [root]'s DAG satisfying [pred],
    in topological order. *)

val divides : t -> int -> t option
(** [divides node v] is [Some q] if [node] can be symbolically shown to be
    divisible by [v], where [q] is the quotient node ([node / v]).  Returns
    [None] when divisibility cannot be proved.

    Handles {!Const}, {!Binary} [Add] (both operands must divide), and
    {!Binary} [Mul] (either operand may divide). *)

val vmin : t -> int
(** [vmin node] is a lower bound on the value [node] can take. *)

val vmax : t -> int
(** [vmax node] is an upper bound on the value [node] can take. *)

val sym_infer : t -> (string * int) list -> int
(** [sym_infer node var_vals] evaluates [node] to a concrete integer by
    substituting each {!Define_var} with its value from [var_vals]
    (matched by name).

    Raises [Failure] if the expression contains nodes that cannot be
    evaluated (e.g. loads, stores, non-arithmetic ops). *)

val range_start : t -> int option
(** [range_start v] is the child index at which range arguments begin for
    nodes that carry them.

    Returns [Some 1] for {!view.Bufferize}, {!view.Reduce}, {!view.End};
    [Some 2] for {!view.Store}; [Some 3] for {!view.Wmma};
    [None] for all other nodes. *)

val ended_ranges : ?live:(t -> t list) -> t -> t list
(** [ended_ranges ?live node] is the list of ranges closed by [node].

    For {!view.Bufferize}, {!view.Reduce}, {!view.Store}, {!view.Wmma}, and
    {!view.End}: range children from the range-start offset onward. For
    {!view.After}: the union of [ended_ranges] of deps. For {!view.Contract}:
    ranges from the source whose axis matches one of the contract's axis IDs,
    looked up via [live]. Otherwise: empty.

    [live] defaults to [fun _ -> []] and is required for correct {!view.Contract}
    handling. {!live_ranges_tbl} provides the appropriate lookup automatically. *)

val live_ranges : t -> t list
(** [live_ranges node] is the set of {!view.Range} nodes that are transitively
    reachable from [node]'s children and have not been ended by any inner
    {!view.Reduce}, {!view.Store}, or {!view.End} node. If [node] is itself a
    {!view.Range}, it is included.

    {b Note.} Computed by a full bottom-up traversal of [node]'s DAG.
    Not cached — callers that need live ranges for many nodes in the same
    DAG should use {!live_ranges_tbl} instead. *)

val live_ranges_tbl : t -> t list Ref_tbl.t
(** [live_ranges_tbl root] precomputes {!live_ranges} for every node in the
    DAG rooted at [root]. The returned table maps each node to its live
    ranges.

    Use this when the gate function of a traversal needs live-range
    information for many nodes. *)

(** {1:operators Operators}

    {!module-O} provides infix operators for building arithmetic Kernel DAG
    nodes. Open locally in codegen modules:

    {[
      let open Kernel.O in
      let idx = base * int_ stride + offset in
      ...
    ]} *)

module O : sig
  val ( + ) : t -> t -> t
  (** Binary {!Op.Add}. *)

  val ( * ) : t -> t -> t
  (** Binary {!Op.Mul}. *)

  val ( / ) : t -> t -> t
  (** Binary {!Op.Idiv}. *)

  val ( mod ) : t -> t -> t
  (** Binary {!Op.Mod}. *)

  val ( < ) : t -> t -> t
  (** Binary {!Op.Cmplt}. Result has boolean scalar dtype. *)

  val eq : t -> t -> t
  (** Binary {!Op.Cmpeq}. *)

  val ne : t -> t -> t
  (** Binary {!Op.Cmpne}. *)

  val where : t -> t -> t -> t
  (** [where cond then_ else_] is {!Op.Where}. *)

  val neg : t -> t
  (** Unary {!Op.Neg}. *)

  val not_ : t -> t
  (** Logical NOT: [eq node (bool_ false)]. *)

  val cast : Dtype.t -> t -> t
  (** [cast dtype node] casts [node] to [dtype]. *)

  val int_ : int -> t
  (** [int_ n] is [const_int n] ({!Dtype.index}-typed). *)

  val float_ : float -> t
  (** [float_ x] is [const_float x] ({!Dtype.float32}-typed). *)

  val bool_ : bool -> t
  (** [bool_ b] is [const_bool b]. *)
end

(** {1:comparison Comparison} *)

val compare_structure : t -> t -> int
(** [compare_structure a b] compares two nodes by recursive structural key
    (op ordinal, arg, dtype, children). Used for canonicalizing commutative
    operations. *)

(** {1:formatting Formatting} *)

val pp_view : Format.formatter -> t -> unit
(** [pp_view] formats one node with local ids relative to that node's DAG. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats the whole DAG rooted at its argument. *)

val view_op_name : view -> string
(** [view_op_name v] is the operation name of [v] as an ["Ops.XXX"] string
    (e.g., ["Ops.SINK"], ["Ops.LOAD"], ["Ops.ADD"]). *)

val print_uops : ?label:string -> t -> unit
(** [print_uops ?label root] prints the DAG rooted at [root] in columnar
    format to stderr (one node per line: id, op, dtype, sources, value).
    When [label] is provided, ["=== label ==="] is printed before the
    listing. *)
