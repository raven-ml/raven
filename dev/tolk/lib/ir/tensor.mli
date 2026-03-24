(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** High-level tensor graph IR.

    [Tensor] is the value-graph stage of [ir_next]. Programs are array-backed
    graphs with stable ids and backward-only references.

    The public surface is intentionally uniform:
    - build graphs with {!create}, {!emit}, and the smart constructors;
    - inspect with {!view}, {!dtype}, and {!children};
    - validate with {!validate};
    - rewrite with {!map_children}, {!rebuild}, and {!rewrite_fixpoint}. *)

(** {1:types Types} *)

type t
(** A tensor graph. *)

type id = int
(** Node id. An index into the tensor graph array. *)

type builder
(** Mutable tensor graph builder. *)

type device =
  | Single of string  (** A single named device. *)
  | Multi of string list  (** Multiple devices for sharded tensors. *)
(** Device placement selector. *)

type metadata = {
  name : string;  (** Operation name. *)
  caller : string;  (** Call-site identifier. *)
  backward : bool;  (** [true] if emitted during backward pass. *)
}
(** Call-site metadata. *)

type view =
  | Sink of { srcs : id list; kernel_info : Kernel.kernel_info option }
      (** Graph root gathering semantic sources. *)
  | Group of { srcs : id list }
      (** Groups effect children without producing a value. *)
  | After of { src : id; deps : id list; dtype : Dtype.t }
      (** Sequences [src] after [deps]. *)
  | Unique of { id : int }
      (** Unique buffer identity tag. *)
  | Lunique of { id : int }
      (** Lazy unique buffer identity tag. *)
  | Device of { device : device }
      (** Device placement node. *)
  | Buffer of { unique : id; device : id; size : int; dtype : Dtype.t }
      (** Allocated buffer of [size] elements. *)
  | Buffer_view of { src : id; size : int; offset : int; dtype : Dtype.t }
      (** View into an existing buffer at [offset]. *)
  | Const of { value : Const.t; dtype : Dtype.t; srcs : id list }
      (** Compile-time constant. [srcs] are scheduling dependencies. *)
  | Vconst of { values : Const.t list; dtype : Dtype.t; srcs : id list }
      (** Vector of compile-time constants. *)
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
      (** Symbolic variable bounded by \[[lo];[hi]\]. *)
  | Bind of { var : id; value : id option; dtype : Dtype.t }
      (** Binds a symbolic variable to a concrete value. *)
  | Param of {
      slot : int;
      dtype : Dtype.t;
      shape : id option;
      device : id option;
    }  (** Function parameter at [slot]. *)
  | Call of {
      callee : callee;
      args : id list;
      info : call_info;
      dtype : Dtype.t;
    }  (** Calls [callee] with [args]. *)
  | Detach of { src : id; dtype : Dtype.t }
      (** Detaches [src] from the gradient tape. *)
  | Contiguous of { src : id; ranges : id list; dtype : Dtype.t }
      (** Forces [src] into contiguous memory layout. *)
  | Contiguous_backward of { src : id; dtype : Dtype.t }
      (** Backward-pass contiguous marker. *)
  | Copy of { src : id; device : id; dtype : Dtype.t }
      (** Copies [src] to [device]. *)
  | Allreduce of { src : id; device : id; op : Op.reduce; dtype : Dtype.t }
      (** All-reduce [src] across devices with [op]. *)
  | Multi of { src : id; axis : int; dtype : Dtype.t }
      (** Distributes [src] across devices along [axis]. *)
  | Mstack of { srcs : id list; dtype : Dtype.t }
      (** Stacks per-device shards into a multi-device tensor. *)
  | Mselect of { src : id; index : int; dtype : Dtype.t }
      (** Selects shard [index] from a multi-device tensor. *)
  | Reduce_axis of {
      src : id;
      op : Op.reduce;
      axes : int list;
      dtype : Dtype.t;
    }  (** Reduces [src] with [op] along [axes]. *)
  | Reduce of { src : id; ranges : id list; op : Op.reduce; dtype : Dtype.t }
      (** Reduces [src] over [ranges] with [op]. *)
  | Reshape of { src : id; shape : id; dtype : Dtype.t }
      (** Reshapes [src] to [shape]. *)
  | Expand of { src : id; shape : id; dtype : Dtype.t }
      (** Broadcasts [src] to [shape]. *)
  | Pad of { src : id; before : id; after : id; dtype : Dtype.t }
      (** Pads [src] with zeros. *)
  | Shrink of { src : id; before : id; after : id; dtype : Dtype.t }
      (** Shrinks [src] by trimming edges. *)
  | Permute of { src : id; order : int list; dtype : Dtype.t }
      (** Permutes axes of [src] according to [order]. *)
  | Flip of { src : id; dims : bool list; dtype : Dtype.t }
      (** Reverses [src] along dimensions where [dims] is [true]. *)
  | Range of { size : id; dtype : Dtype.t; axis : int; sub : int list; kind : Axis_kind.t }
      (** Loop variable over \[[0];[size-1]\] on [axis]. *)
  | End of { value : id; ranges : id list }
      (** Closes loop [ranges] around [value]. *)
  | Index of { ptr : id; idxs : id list; gate : id option; dtype : Dtype.t }
      (** Indexes into [ptr] with per-dimension [idxs] and optional [gate]. *)
  | Store of { dst : id; value : id }
      (** Stores [value] through pointer [dst]. *)
  | Vectorize of { srcs : id list; dtype : Dtype.t }
      (** Packs scalar [srcs] into a vector. *)
  | Cast of { src : id; dtype : Dtype.t }
      (** Type cast. *)
  | Bitcast of { src : id; dtype : Dtype.t }
      (** Bit-preserving reinterpretation. *)
  | Unary of { op : Op.unary; src : id; dtype : Dtype.t }
      (** Unary arithmetic or transcendental. *)
  | Binary of { op : Op.binary; lhs : id; rhs : id; dtype : Dtype.t }
      (** Binary arithmetic, logic, or comparison. *)
  | Ternary of { op : Op.ternary; a : id; b : id; c : id; dtype : Dtype.t }
      (** Ternary operation ([Where] or [Mulacc]). *)
  | Noop of { src : id option; dtype : Dtype.t }
      (** Pass-through scheduling marker. *)
  | Bufferize of {
      src : id;
      ranges : id list;
      dtype : Dtype.t;
      opts : Kernel.bufferize_opts;
    }  (** Materializes [src] into a buffer during schedule. *)
  | Invalid_index of { dtype : Dtype.t }
      (** Invalid index sentinel for PAD range transformations. *)
  | Define_local of { size : int; dtype : Dtype.ptr }
      (** Local (workgroup-shared) memory buffer of [size] elements. *)
  | Barrier
      (** Workgroup barrier. *)

and emit = view -> id
(** Node emitter used by gradient hooks. *)

and grad_fxn = emit:emit -> grad_output:id -> call:id -> id option list
(** Custom gradient callback. *)

and callee =
  | Ref of id  (** Reference to an in-graph callable. *)
  | Ast of Kernel.t  (** Inline kernel AST. *)
(** Call target. *)

and call_info = {
  grad_fxn : grad_fxn option;  (** Custom gradient, if any. *)
  metadata : metadata list;  (** Call-site metadata stack. *)
  name : string option;  (** Optional kernel name override. *)
  precompile : bool;  (** [true] to precompile the kernel. *)
}
(** Call annotations. *)

(** {1:building Building} *)

(* CR: Not sure we want all of these constructors here. Are they used? Do we have constructors like this in the other IR (kernel and program)?
   If not, consider removing them and adding a top-level tensor.ml with a numpy-like API to build a Tolk_ir.Tensor.t *)

val create : unit -> builder
(** [create ()] is an empty tensor graph builder. *)

val emit : builder -> view -> id
(** [emit b v] appends [v] to [b] and returns its id. *)

val finish : builder -> t
(** [finish b] is the graph built so far. *)

val shape : builder -> Shape.t -> id
(** [shape b s] emits the canonical node encoding of shape [s]. *)

val sink : builder -> ?kernel_info:Kernel.kernel_info -> id list -> id
(** [sink b ?kernel_info srcs] emits a graph root with sources [srcs]. *)

val group : builder -> id list -> id
(** [group b srcs] emits a group of effect children. *)

val after : builder -> src:id -> deps:id list -> id
(** [after b ~src ~deps] emits a sequencing node. *)

val unique : builder -> id:int -> id
(** [unique b ~id] emits a unique buffer identity tag. *)

val lunique : builder -> id:int -> id
(** [lunique b ~id] emits a lazy unique buffer identity tag. *)

val device : builder -> device -> id
(** [device b d] emits a device placement node. *)

val buffer :
  builder -> unique:id -> device:id -> size:int -> dtype:Dtype.t -> id
(** [buffer b ~unique ~device ~size ~dtype] emits a buffer allocation. *)

val buffer_view :
  builder -> src:id -> size:int -> offset:int -> dtype:Dtype.t -> id
(** [buffer_view b ~src ~size ~offset ~dtype] emits a view into buffer
    [src]. *)

val const : builder -> ?srcs:id list -> Const.t -> id
(** [const b ?srcs c] emits a constant node. [srcs] are scheduling
    dependencies. *)

val vconst :
  builder -> values:Const.t list -> dtype:Dtype.t -> ?srcs:id list -> unit -> id
(** [vconst b ~values ~dtype ()] emits a vector constant. *)

val define_var :
  builder -> name:string -> lo:int -> hi:int -> ?dtype:Dtype.t -> unit -> id
(** [define_var b ~name ~lo ~hi ()] emits a symbolic variable.

    [dtype] defaults to {!Dtype.index}. *)

val bind : builder -> var:id -> ?value:id -> unit -> id
(** [bind b ~var ?value ()] binds symbolic variable [var] to [value]. *)

val param :
  builder -> slot:int -> dtype:Dtype.t -> ?shape:id -> ?device:id -> unit -> id
(** [param b ~slot ~dtype ()] emits a function parameter at [slot]. *)

val call :
  builder ->
  callee:callee ->
  args:id list ->
  info:call_info ->
  dtype:Dtype.t ->
  id
(** [call b ~callee ~args ~info ~dtype] emits a call to [callee]. *)

val assign : builder -> target:id -> value:id -> ?extras:id list -> unit -> id
(** [assign b ~target ~value ()] assigns [value] to buffer [target].

    Emits a [Store] of [value] into [target], then an [After] sequencing
    [target] after the store (and any [extras]). Returns the [After] id. *)

val detach : builder -> src:id -> id
(** [detach b ~src] detaches [src] from the gradient tape. *)

val contiguous : builder -> src:id -> ?ranges:id list -> unit -> id
(** [contiguous b ~src ()] forces [src] into contiguous layout. *)

val contiguous_backward : builder -> src:id -> id
(** [contiguous_backward b ~src] emits a backward-pass contiguous marker. *)

val copy : builder -> src:id -> device:id -> unit -> id
(** [copy b ~src ~device ()] copies [src] to [device]. For multi-device
    sources, use {!mselect} before copying to select the desired shard. *)

val allreduce : builder -> src:id -> device:id -> op:Op.reduce -> id
(** [allreduce b ~src ~device ~op] all-reduces [src] across devices. *)

val multi : builder -> src:id -> axis:int -> id
(** [multi b ~src ~axis] distributes [src] along [axis]. *)

val mstack : builder -> srcs:id list -> id
(** [mstack b ~srcs] stacks per-device shards. *)

val mselect : builder -> src:id -> index:int -> id
(** [mselect b ~src ~index] selects shard [index] from a multi tensor. *)

val reduce_axis : builder -> src:id -> op:Op.reduce -> axes:int list -> id
(** [reduce_axis b ~src ~op ~axes] reduces [src] along [axes]. Dtype is
    inherited from [src]. *)

val reduce :
  builder -> src:id -> ranges:id list -> op:Op.reduce -> dtype:Dtype.t -> id
(** [reduce b ~src ~ranges ~op ~dtype] reduces [src] over [ranges]. *)

val reshape : builder -> src:id -> shape:id -> id
(** [reshape b ~src ~shape] reshapes [src]. *)

val expand : builder -> src:id -> shape:id -> id
(** [expand b ~src ~shape] broadcasts [src] to [shape]. *)

val pad : builder -> src:id -> before:id -> after:id -> id
(** [pad b ~src ~before ~after] pads [src] with zeros. *)

val shrink : builder -> src:id -> before:id -> after:id -> id
(** [shrink b ~src ~before ~after] trims edges of [src]. *)

val permute : builder -> src:id -> order:int list -> id
(** [permute b ~src ~order] permutes axes of [src]. *)

val flip : builder -> src:id -> dims:bool list -> id
(** [flip b ~src ~dims] reverses [src] along flagged dimensions. *)

val range :
  builder ->
  size:id ->
  axis:int ->
  ?sub:int list ->
  kind:Axis_kind.t ->
  ?dtype:Dtype.t ->
  unit ->
  id
(** [range b ~size ~axis ~kind ()] emits a loop variable.

    [dtype] defaults to {!Dtype.index}. *)

val end_ : builder -> value:id -> ranges:id list -> id
(** [end_ b ~value ~ranges] closes loop ranges around [value]. *)

val index :
  builder -> ptr:id -> idxs:id list -> ?gate:id -> dtype:Dtype.t -> unit -> id
(** [index b ~ptr ~idxs ?gate ~dtype ()] indexes into [ptr]. *)

val store : builder -> dst:id -> value:id -> id
(** [store b ~dst ~value] stores [value] through pointer [dst]. *)

val vectorize : builder -> srcs:id list -> id
(** [vectorize b ~srcs] packs scalar sources into a vector. *)

val cast : builder -> src:id -> dtype:Dtype.t -> id
(** [cast b ~src ~dtype] casts [src] to [dtype]. *)

val bitcast : builder -> src:id -> dtype:Dtype.t -> id
(** [bitcast b ~src ~dtype] bitcasts [src] to [dtype]. *)

val unary : builder -> op:Op.unary -> src:id -> id
(** [unary b ~op ~src] applies unary [op] to [src]. *)

val binary : builder -> op:Op.binary -> lhs:id -> rhs:id -> id
(** [binary b ~op ~lhs ~rhs] applies binary [op]. *)

val ternary : builder -> op:Op.ternary -> a:id -> b:id -> c:id -> id
(** [ternary b ~op ~a ~b ~c] applies ternary [op]. *)

val noop : builder -> ?src:id -> dtype:Dtype.t -> unit -> id
(** [noop b ?src ~dtype ()] emits a pass-through scheduling marker. *)

val bufferize :
  builder -> src:id -> ranges:id list -> dtype:Dtype.t -> opts:Kernel.bufferize_opts -> id
(** [bufferize b ~src ~ranges ~dtype ~opts] materializes [src] into a buffer. *)

val invalid_index : builder -> dtype:Dtype.t -> id
(** [invalid_index b ~dtype] emits an invalid index sentinel. *)

val define_local : builder -> size:int -> dtype:Dtype.ptr -> id
(** [define_local b ~size ~dtype] defines a local-memory buffer. *)

val barrier : builder -> id
(** [barrier b] emits a workgroup barrier. *)

(** {1:inspection Inspecting} *)

val view : t -> id -> view
(** [view t id] is the instruction at [id]. *)

val length : t -> int
(** [length t] is the number of nodes in [t]. *)

val dtype : t -> id -> Dtype.t option
(** [dtype t id] is the node dtype of [id], if any. *)

val children : t -> id -> id list
(** [children t id] are the direct input ids of instruction [id]. *)

(** {1:validation Validation} *)

val validate : t -> unit
(** [validate t] checks tensor invariants.

    Raises [Failure] on the first violation. *)

(** {1:rewriting Rewriting} *)

val node_dtype : view -> Dtype.t option
(** [node_dtype v] is the dtype of the view [v], if any. *)

val children_of : view -> id list
(** [children_of v] are the direct child ids of view [v]. *)

val map_children : (id -> id) -> view -> view
(** [map_children f v] rebuilds the direct children of [v] with [f]. *)

val rebuild : (id -> view -> view option) -> t -> t
(** [rebuild f t] rebuilds [t] in id order.

    Children are rewritten first. [f id v] sees the rebuilt view of the node
    previously at [id]. When it returns [Some v'], [v'] replaces [v]. *)

(* CR: do we need rewrite_fixpoint, rebuild_grow and rewrite_fixpoint_grow? If they are unused, plan how to remove them. *)
    
val rewrite_fixpoint : ?max_iters:int -> (id -> view -> view option) -> t -> t
(** [rewrite_fixpoint ?max_iters f t] repeatedly applies {!rebuild} with [f]
    until the graph stops changing.

    Raises [Failure] if [max_iters] passes are reached without a fixpoint.
    [max_iters] defaults to [16]. *)

val rebuild_grow :
  (lookup:(id -> view) -> (view -> id) -> id -> view -> view option) -> t -> t
(** [rebuild_grow f t] is like {!rebuild}, but the rewrite function [f]
    receives:
    - [~lookup]: look up the view of an already-emitted node in the
      in-construction graph (use this instead of indexing the old program)
    - [emit]: callback [(view -> id)] to create auxiliary nodes
    - [id]: the old node id
    - [view]: the view with children already remapped

    The output may be larger than the input. *)

val rewrite_fixpoint_grow :
  ?max_iters:int ->
  (lookup:(id -> view) -> (view -> id) -> id -> view -> view option) ->
  t ->
  t
(** [rewrite_fixpoint_grow ?max_iters f t] repeatedly applies {!rebuild_grow}
    with [f] until the graph stops changing. *)

val merge_builder : t -> builder -> t * (id -> id)
(** [merge_builder program extra] appends all nodes from [extra] to [program].
    Returns the extended program and a shift function that maps builder ids to
    their positions in the extended program. *)

(** {1:analysis Analysis} *)

val extract_int_shape : t -> id -> int list option
(** [extract_int_shape t id] decodes a concrete int list from a
    shape-encoding node (Vectorize of Consts, a single Const, or an
    empty Vconst). Returns [None] if any dimension is symbolic. *)

val extract_marg : t -> view -> int list option
(** [extract_marg t v] extracts the shape argument from a Reshape or Expand
    view. Returns [None] for other ops or symbolic shapes. *)

val extract_marg_pairs : t -> view -> (int * int) list option
(** [extract_marg_pairs t v] extracts the (before, after) pairs from a Pad or
    Shrink view. Returns [None] for other ops or symbolic values. *)

val compute_shapes : t -> int list option array
(** [compute_shapes t] computes the shape of every node in [t]. The returned
    array maps each id to its shape, or [None] for nodes without shapes
    (effects, ranges, etc.). *)

val compute_devices : t -> device option array
(** [compute_devices t] computes the device of every node in [t]. *)

val base : t -> id -> id
(** [base t id] follows through movement ops (Reshape, Expand, Pad, Shrink,
    Permute, Flip, Multi, Detach) to the underlying buffer node. *)

val consumer_map : t -> id list array
(** [consumer_map t] builds a consumer map: for each node, the list of nodes
    that reference it as a child. *)

val backward_slice : t -> id -> id list
(** [backward_slice t root] is all transitive dependencies of [root]
    (inclusive), in topological order. *)

(** {1:formatting Formatting} *)

val pp_view : Format.formatter -> view -> unit
(** [pp_view] formats one tensor node view. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats the whole graph. *)
