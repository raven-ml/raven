(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** High-level tensor graph IR.

    Nodes are hash-consed: structurally identical nodes are physically
    identical ([==]), enabling efficient deduplication during graph
    rewriting.

    Build nodes with the smart constructors ({!sink}, {!after},
    {!const}, …); inspect with {!view}, {!dtype}, and {!children};
    rewrite with {!graph_rewrite} and {!substitute}. *)

(** {1:types Types} *)

type t
(** A tensor graph node. Hash-consed: structurally identical nodes
    are physically identical. *)

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

(** {2:view Node views} *)

type view =
  | Sink of { srcs : t list; kernel_info : Kernel.kernel_info option }
  | Group of { srcs : t list }
  | After of { src : t; deps : t list; dtype : Dtype.t }
  | Unique of { id : int }
  | Lunique of { id : int }
  | Device of { device : device }
  | Buffer of { unique : t; device : t; size : int; dtype : Dtype.t }
  | Buffer_view of { src : t; size : int; offset : int; dtype : Dtype.t }
  | Const of { value : Const.t; dtype : Dtype.t; srcs : t list }
  | Vconst of { values : Const.t list; dtype : Dtype.t; srcs : t list }
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
  | Bind of { var : t; value : t option; dtype : Dtype.t }
  | Param of {
      slot : int;
      dtype : Dtype.t;
      shape : t option;
      device : t option;
    }
  | Call of {
      callee : callee;
      args : t list;
      info : call_info;
      dtype : Dtype.t;
    }
  | Detach of { src : t; dtype : Dtype.t }
  | Contiguous of { src : t; ranges : t list; opts : Kernel.Opt.t list; dtype : Dtype.t }
  | Contiguous_backward of { src : t; dtype : Dtype.t }
  | Copy of { src : t; device : t; dtype : Dtype.t }
  | Allreduce of { src : t; device : t; op : Op.reduce; dtype : Dtype.t }
  | Multi of { src : t; axis : int; dtype : Dtype.t }
  | Mstack of { srcs : t list; dtype : Dtype.t }
  | Mselect of { src : t; index : int; dtype : Dtype.t }
  | Reduce_axis of {
      src : t;
      op : Op.reduce;
      axes : int list;
      dtype : Dtype.t;
    }
  | Reduce of { src : t; ranges : t list; op : Op.reduce; dtype : Dtype.t }
  | Reshape of { src : t; shape : t; dtype : Dtype.t }
  | Expand of { src : t; shape : t; dtype : Dtype.t }
  | Pad of { src : t; before : t; after : t; dtype : Dtype.t }
  | Shrink of { src : t; before : t; after : t; dtype : Dtype.t }
  | Permute of { src : t; order : int list; dtype : Dtype.t }
  | Flip of { src : t; dims : bool list; dtype : Dtype.t }
  | Range of {
      size : t;
      dtype : Dtype.t;
      axis : int;
      sub : int list;
      kind : Axis_kind.t;
    }
  | End of { value : t; ranges : t list }
  | Index of { ptr : t; idxs : t list; gate : t option; dtype : Dtype.t }
  | Store of { dst : t; value : t }
  | Vectorize of { srcs : t list; dtype : Dtype.t }
  | Cast of { src : t; dtype : Dtype.t }
  | Bitcast of { src : t; dtype : Dtype.t }
  | Unary of { op : Op.unary; src : t; dtype : Dtype.t }
  | Binary of { op : Op.binary; lhs : t; rhs : t; dtype : Dtype.t }
  | Ternary of { op : Op.ternary; a : t; b : t; c : t; dtype : Dtype.t }
  | Noop of { src : t option; dtype : Dtype.t }
  | Bufferize of {
      src : t;
      ranges : t list;
      dtype : Dtype.t;
      opts : Kernel.bufferize_opts;
    }
  | Invalid_index of { dtype : Dtype.t }
  | Define_local of { size : int; dtype : Dtype.Ptr.t }
  | Barrier
  | Linear of { srcs : t list }
  | Shaped_wmma of {
      a : t; b : t; acc : t;
      dims : int * int * int;
      device : string;
      threads : int;
      dtype : Dtype.t;
    }
(** Node views. Each variant describes one tensor operation with
    direct references to child nodes. *)

and callee =
  | Ref of t  (** Reference to an in-graph callable. *)
  | Ast of Kernel.t  (** Inline kernel AST. *)
(** Call target. *)

and call_info = {
  grad_fxn : grad_fxn option;
  metadata : metadata list;
  name : string option;
  precompile : bool;
}
(** Call annotations. *)

and grad_fxn = grad_output:t -> call:t -> t option list
(** Custom gradient callback. *)

(** {1:constructors Constructors} *)

val sink : ?kernel_info:Kernel.kernel_info -> t list -> t
(** [sink ?kernel_info srcs] is a graph root gathering [srcs]. *)

val group : t list -> t
(** [group srcs] groups effect children. Returns [src] directly
    when [srcs] is a singleton. *)

val after : src:t -> deps:t list -> t
(** [after ~src ~deps] sequences [src] after [deps]. *)

val unique : id:int -> t
(** [unique ~id] is a unique buffer identity tag. *)

val lunique : id:int -> t
(** [lunique ~id] is a lazy unique buffer identity tag. *)

val device : device -> t
(** [device d] is a device placement node. *)

val buffer : unique:t -> device:t -> size:int -> dtype:Dtype.t -> t
(** [buffer ~unique ~device ~size ~dtype] is a buffer allocation. *)

val buffer_view : src:t -> size:int -> offset:int -> dtype:Dtype.t -> t
(** [buffer_view ~src ~size ~offset ~dtype] is a view into [src]. *)

val const : ?srcs:t list -> Const.t -> Dtype.t -> t
(** [const ?srcs c dt] is a constant [c] of type [dt]. [srcs] are
    scheduling dependencies (default [[]]). *)

val vconst :
  values:Const.t list -> dtype:Dtype.t -> ?srcs:t list -> unit -> t
(** [vconst ~values ~dtype ()] is a vector of constants. *)

val define_var :
  name:string -> lo:int -> hi:int -> ?dtype:Dtype.t -> unit -> t
(** [define_var ~name ~lo ~hi ()] is a symbolic variable bounded by
    \[[lo];[hi]\]. [dtype] defaults to {!Dtype.index}. *)

val bind : var:t -> ?value:t -> dtype:Dtype.t -> unit -> t
(** [bind ~var ?value ~dtype ()] binds [var] to [value]. *)

val param :
  slot:int -> dtype:Dtype.t -> ?shape:t -> ?device:t -> unit -> t
(** [param ~slot ~dtype ()] is a function parameter at [slot]. *)

val call :
  callee:callee -> args:t list -> info:call_info -> dtype:Dtype.t -> t
(** [call ~callee ~args ~info ~dtype] calls [callee] with [args]. *)

val assign : target:t -> value:t -> ?extras:t list -> unit -> t
(** [assign ~target ~value ()] stores [value] into [target] and
    returns an {!After} sequencing [target] after the store. *)

val detach : src:t -> t
(** [detach ~src] detaches [src] from the gradient tape. *)

val contiguous : src:t -> ?ranges:t list -> ?opts:Kernel.Opt.t list -> unit -> t
(** [contiguous ~src ()] forces [src] into contiguous layout. *)

val contiguous_backward : src:t -> t
(** [contiguous_backward ~src] is a backward-pass contiguous marker. *)

val copy : src:t -> device:t -> unit -> t
(** [copy ~src ~device ()] copies [src] to [device]. *)

val allreduce : src:t -> device:t -> op:Op.reduce -> dtype:Dtype.t -> t
(** [allreduce ~src ~device ~op ~dtype] all-reduces [src]. *)

val multi : src:t -> axis:int -> t
(** [multi ~src ~axis] distributes [src] along [axis]. *)

val mstack : srcs:t list -> t
(** [mstack ~srcs] stacks per-device shards. *)

val mselect : src:t -> index:int -> t
(** [mselect ~src ~index] selects shard [index]. *)



val reduce_axis : src:t -> op:Op.reduce -> axes:int list -> t
(** [reduce_axis ~src ~op ~axes] reduces [src] along [axes]. *)

val reduce : src:t -> ranges:t list -> op:Op.reduce -> dtype:Dtype.t -> t
(** [reduce ~src ~ranges ~op ~dtype] reduces [src] over [ranges]. *)

val reshape : src:t -> shape:t -> t
(** [reshape ~src ~shape] reshapes [src] to [shape]. *)

val expand : src:t -> shape:t -> t
(** [expand ~src ~shape] broadcasts [src] to [shape]. *)

val pad : src:t -> before:t -> after:t -> t
(** [pad ~src ~before ~after] pads [src] with zeros. *)

val shrink : src:t -> before:t -> after:t -> t
(** [shrink ~src ~before ~after] trims edges of [src]. *)

val permute : src:t -> order:int list -> t
(** [permute ~src ~order] permutes axes of [src]. *)

val flip : src:t -> dims:bool list -> t
(** [flip ~src ~dims] reverses [src] along flagged dimensions. *)

val range :
  size:t -> axis:int -> ?sub:int list -> kind:Axis_kind.t ->
  ?dtype:Dtype.t -> unit -> t
(** [range ~size ~axis ~kind ()] is a loop variable over
    \[[0];[size-1]\]. [dtype] defaults to {!Dtype.index}. *)

val end_ : value:t -> ranges:t list -> t
(** [end_ ~value ~ranges] closes loop [ranges] around [value]. *)

val index :
  ptr:t -> idxs:t list -> ?gate:t -> dtype:Dtype.t -> unit -> t
(** [index ~ptr ~idxs ?gate ~dtype ()] indexes into [ptr]. *)

val store : dst:t -> value:t -> t
(** [store ~dst ~value] stores [value] through [dst]. *)

val vectorize : srcs:t list -> t
(** [vectorize ~srcs] packs scalar [srcs] into a vector. *)

val cast : src:t -> dtype:Dtype.t -> t
(** [cast ~src ~dtype] casts [src] to [dtype]. *)

val bitcast : src:t -> dtype:Dtype.t -> t
(** [bitcast ~src ~dtype] bitcasts [src] to [dtype]. *)

val unary : op:Op.unary -> src:t -> t
(** [unary ~op ~src] applies unary [op]. *)

val binary : op:Op.binary -> lhs:t -> rhs:t -> t
(** [binary ~op ~lhs ~rhs] applies binary [op]. *)

val ternary : op:Op.ternary -> a:t -> b:t -> c:t -> t
(** [ternary ~op ~a ~b ~c] applies ternary [op]. *)

val noop : ?src:t -> dtype:Dtype.t -> unit -> t
(** [noop ?src ~dtype ()] is a pass-through scheduling marker. *)

val bufferize :
  src:t -> ranges:t list -> dtype:Dtype.t ->
  opts:Kernel.bufferize_opts -> t
(** [bufferize ~src ~ranges ~dtype ~opts] materializes [src]. *)

val invalid_index : dtype:Dtype.t -> t
(** [invalid_index ~dtype] is an invalid index sentinel. *)

val define_local : size:int -> dtype:Dtype.Ptr.t -> t
(** [define_local ~size ~dtype] defines a local-memory buffer. *)

val barrier : t
(** [barrier] is a workgroup barrier. *)

val linear : t list -> t
(** [linear srcs] is a linearized schedule of [srcs]. *)

val shaped_wmma :
  a:t -> b:t -> acc:t ->
  dims:(int * int * int) -> device:string -> threads:int ->
  dtype:Dtype.t -> t
(** [shaped_wmma ~a ~b ~acc ~dims ~device ~threads ~dtype] is a shaped
    tensor-core WMMA operation. Lowered to kernel-level {!Kernel.view.Wmma}
    during scheduling. *)

val replace : t -> ?children:t list -> ?dtype:Dtype.t -> unit -> t
(** [replace n ?children ?dtype ()] rebuilds [n] with substituted
    children and/or dtype. Unchanged fields are preserved.

    [children] must have the same length as [children n]. *)

(** {1:inspection Inspection} *)

val view : t -> view
(** [view n] is the operation [n] represents. *)

val children : t -> t list
(** [children n] are the direct input nodes of [n]. *)

val dtype : t -> Dtype.t option
(** [dtype n] is [n]'s dtype, if any. *)

val tag : t -> int
(** [tag n] is [n]'s unique identity. Two nodes are physically
    identical iff their tags are equal. *)

(** {1:traversal Traversal} *)

val toposort :
  ?gate:(t -> bool) -> ?enter_calls:bool -> t -> t list
(** [toposort ?gate ?enter_calls root] is all transitive
    dependencies of [root] in topological order (leaves first).

    [gate] controls descent: when it returns [false] for a node,
    that node's children are not visited. Defaults to [fun _ -> true].

    [enter_calls] controls whether CALL bodies (the callee) are
    entered. Defaults to [true]. When [false], [callee] is treated
    as opaque. *)

val backward_slice : t -> t list
(** [backward_slice root] is {!toposort} [root] without [root]
    itself. *)

val variables : t -> t list
(** [variables root] is all {!Define_var} nodes reachable from
    [root], in topological order. *)

val ranges : t -> t list
(** [ranges root] is all {!Range} nodes reachable from [root],
    in topological order. *)

(** {1:rewriting Rewriting} *)

val children_of : view -> t list
(** [children_of v] are the direct child nodes of [v]. *)

val map_children : (t -> t) -> view -> view
(** [map_children f v] rebuilds the children of [v] with [f]. *)

val node_dtype : view -> Dtype.t option
(** [node_dtype v] is the dtype of [v], if any. *)

val graph_rewrite :
  ?name:string -> ?enter_calls:bool ->
  ?on_rebuild:(old_n:t -> new_n:t -> unit) ->
  (t -> t option) -> t -> t
(** [graph_rewrite ?name ?enter_calls f root] rewrites [root]'s DAG.

    Processes nodes bottom-up using a 3-stage stack:
    {ul
    {- Stage 0: push children for processing.}
    {- Stage 1: rebuild with rewritten children, apply [f].
       When [f] returns [Some n'], [n'] replaces the node and
       is re-processed.}
    {- Stage 2: link the original node to its final replacement.}}

    Nodes that depend on not-yet-ready replacements are added to
    a waitlist and resumed when the dependency resolves.

    [enter_calls] controls whether CALL bodies are entered.
    Defaults to [true]. *)

val substitute : (t * t) list -> t -> t
(** [substitute mappings root] replaces nodes by physical identity
    ([==]). Each [(old, new_)] pair causes [old] to be replaced
    with [new_] throughout the DAG. *)

val first_match : (t -> t option) list -> t -> t option
(** [first_match rules n] tries each rule in order, returning
    the first [Some]. Returns [None] if no rule matches. *)

(** {1:analysis Analysis} *)

val base : t -> t
(** [base n] follows through movement ops (Reshape, Expand, Pad,
    Shrink, Permute, Flip, Multi, Detach) to the underlying buffer
    node. *)

val extract_int_shape : t -> int list option
(** [extract_int_shape n] decodes a concrete int list from a
    shape-encoding node (Vectorize of Consts, single Const, or
    empty Vconst). Returns [None] if any dimension is symbolic. *)

val extract_marg : view -> int list option
(** [extract_marg v] extracts the shape argument from a Reshape
    or Expand view. Returns [None] for other ops or symbolic
    shapes. *)

val extract_marg_pairs : view -> (int * int) list option
(** [extract_marg_pairs v] extracts (before, after) pairs from
    a Pad or Shrink view. Returns [None] for other ops or
    symbolic values. *)

val compute_shapes : t -> (t -> int list option)
(** [compute_shapes root] computes the shape of every node
    reachable from [root]. Returns a lookup function. *)

val compute_devices : t -> (t -> device option)
(** [compute_devices root] computes the device of every node
    reachable from [root]. Returns a lookup function. *)

val consumer_map : t -> (t -> t list)
(** [consumer_map root] builds a consumer map: for each node
    reachable from [root], the list of nodes that reference it
    as a child. Returns a lookup function. *)

(** {1:formatting Formatting} *)

val pp_view : Format.formatter -> view -> unit
(** [pp_view] formats one tensor node view. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats the DAG rooted at a node. *)

