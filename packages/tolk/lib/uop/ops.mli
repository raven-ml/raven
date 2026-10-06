(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The intermediate representation: nodes, patterns and graph rewriting.

    A node ({!t}) is an operation ({!Op.t}) applied to source nodes, with a
    payload ({!arg}), an annotation ({!Tag.t}) and a data type derived from the
    three. Nodes form a directed acyclic graph, and one vocabulary serves every
    stage of compilation: tensor graphs, kernels, programs and command
    sequences.

    {b Identity.} Nodes are hash-consed: building a node equal to a live one
    returns that node, so two nodes are structurally equal iff they are
    physically equal ([==]). A node lives as long as something references it.
    Nodes can be built and read from any domain.

    {b Construction.} {!v} builds any node; the constructors below build the
    common ones with the defaults and checks of each operation. Arithmetic
    promotes its operands to a common data type, and has their broadcast shape
    ({!module-type-Elementwise}).

    {b Shapes.} A node's shape is a list of {!sint}: integers, or integer nodes
    for sizes known only when the program runs. Comparing such sizes needs
    simplification, so shapes, movements and the functions that read them are
    {!Shape}'s.

    {b Patterns.} {!Upat} describes nodes, {!Pattern_matcher} pairs patterns
    with rules, and {!graph_rewrite} rewrites a graph with them to a fixed
    point.

    {b Errors.} A broken precondition raises [Invalid_argument]. *)

type t
(** The type for nodes. *)

(** {1:axes Axis types} *)

(** Axis types.

    The role of a loop variable ({!Op.Range}) in a kernel. *)
module Axis_type : sig
  (** The type for axis types. *)
  type t =
    | Device  (** Across devices. *)
    | Global  (** Across the workgroups of a launch. *)
    | Warp  (** Across the threads of a warp. *)
    | Local  (** Across the threads of a workgroup. *)
    | Weak  (** Not yet assigned. *)
    | Reduce  (** Reduced over. *)
    | Upcast  (** Unrolled into vector lanes of a thread. *)
    | Unroll  (** A reduction unrolled within a thread. *)
    | Placeholder  (** Stands for an axis while it is built. *)
    | Loop  (** A serial loop. *)

  val equal : t -> t -> bool
  (** [equal a0 a1] is [true] iff [a0] and [a1] are the same axis type. *)

  val compare : t -> t -> int
  (** [compare] orders axis types by their declaration order in {!t}. *)

  val letter : t -> string
  (** [letter a] is the letter that names [a] in kernel names: ["d"], ["g"],
      ["w"], ["l"], ["L"] for {!Weak} and {!Loop}, ["R"], ["u"] and ["r"].

      Raises [Invalid_argument] on {!Placeholder}. *)

  val color : t -> Helpers.color
  (** [color a] is the colour [a]'s letter is printed in.

      Raises [Invalid_argument] on {!Placeholder}. *)

  val position : t -> int
  (** [position a] is [a]'s rank in the order in which a kernel lists its axes:
      [-2] for {!Device}, [-1] for {!Weak} and {!Loop}, then {!Global}, {!Warp},
      {!Local}, {!Upcast}, {!Reduce} and {!Unroll} from [0] to [5].

      Raises [Invalid_argument] on {!Placeholder}. *)

  val of_string : string -> (t, string) result
  (** [of_string s] is the axis type named [s] in upper case, without its
      [AxisType.] prefix: [REDUCE]. The error names [s]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats an axis type as [AxisType.] followed by its name in upper
      case: [AxisType.REDUCE]. *)
end

(** {1:devices Devices} *)

(** The type for device placements: one device, or one value spread over
    several. Devices are named by the caller, such as ["CPU"] or ["AMD:1"]; only
    a name starting with ["DISK"] has a meaning here, a disk. *)
type device = Single of string | Multi of string list

val equal_device : device -> device -> bool
(** [equal_device d0 d1] is [true] iff [d0] and [d1] are the same placement. *)

val pp_device : Format.formatter -> device -> unit
(** [pp_device] formats a placement as a quoted name or a tuple of quoted names:
    ['CPU'], [('CPU:0', 'CPU:1')]. *)

val is_disk_device : device -> bool
(** [is_disk_device d] is [true] iff a device of [d] is a disk: its name, up to
    its first [:], is [DISK] in any case. *)

(** {1:sint Symbolic integers} *)

(** The type for symbolic integers: an integer, or an integer node whose value
    is known when the program runs. Functions that return one return [Int]
    whenever the value is known. *)
type sint = Int of int | Sym of t

(** {1:args Arguments} *)

type param_arg = {
  slot : int;
      (** The parameter's position in its function, or [-1] for a named
          variable. *)
  dtype : Dtype.t;  (** The element type. *)
  size : int option;
      (** The number of elements, or [None] for a scalar. Never symbolic: a
          symbolic size is a larger parameter shrunk. *)
  vmin_vmax : (Dtype.value * Dtype.value) option;
      (** The least and greatest value of a variable. *)
  multiple_of : int option;  (** A number every value is a multiple of. *)
  name : string option;  (** The name of a variable or a named buffer. *)
  addrspace : Dtype.addr_space option;
      (** Where the storage lives; {!Dtype.Alu} for a scalar variable. *)
  device : device option;  (** The placement of the storage. *)
  volatile : bool;  (** Accesses must not be merged or reordered. *)
  bind_on_realize : bool;
      (** The storage is bound by whoever realizes the graph, not by a call. *)
  bound : Dtype.value option;  (** The value a variable is bound to. *)
  phase : int;
      (** The bytes by which the storage's first element lies past a multiple
          of [align]: from [0] to [align - 1], a multiple of the element's size
          or of [align], whichever is less, which a copy that changes the
          element type must keep. Vector accesses start only where they are
          aligned to their width. *)
  align : int;
      (** The power of two, from [1] to [16], the width of the widest vector
          access, modulo which the start of the storage is known ([phase]). No
          vector access is wider. *)
}
(** The type for the arguments of {!Op.Param}, {!Op.Buffer} and {!Op.Alloc}:
    storage, or a scalar variable. *)

val param_arg :
  ?size:int ->
  ?vmin_vmax:Dtype.value * Dtype.value ->
  ?multiple_of:int ->
  ?name:string ->
  ?addrspace:Dtype.addr_space option ->
  ?device:device ->
  ?volatile:bool ->
  ?bind_on_realize:bool ->
  ?bound:Dtype.value ->
  ?phase:int ->
  ?align:int ->
  slot:int ->
  Dtype.t ->
  param_arg
(** [param_arg ~slot dtype] is the argument with these fields. [addrspace]
    defaults to [Some Global]; the flags default to [false], [phase] to [0],
    [align] to [16] and the other fields to [None].

    Raises [Invalid_argument] if [align] is not a power of two from [1] to
    [16], or [phase] is not below [align] and a multiple of [dtype]'s size or
    of [align], whichever is less. *)

val pp_param_arg : Format.formatter -> param_arg -> unit
(** [pp_param_arg] formats the slot, the type and the size, then the fields that
    differ from their defaults, by name:
    [ParamArg(-1, dtypes.weakint, vmin_vmax=(1, 10), name='i', ...)]. *)

type estimates = {
  ops : sint;  (** Arithmetic operations. *)
  lds : sint;  (** Bytes loaded and stored. *)
  mem : sint;  (** Bytes of memory touched, each counted once. *)
}
(** The type for the cost estimates of a kernel. *)

val pp_estimates : Format.formatter -> estimates -> unit
(** [pp_estimates] formats [Estimates(ops=0, lds=0, mem=0)]. *)

type split = {
  iterations : sint;  (** The loop's iterations, [0] to [iterations - 1]. *)
  lo : int;  (** The slot of the variable that holds a block's first. *)
  hi : int;  (** The slot of the variable that holds the one after its last. *)
}
(** The type for the loop a host program's launch splits into blocks that the
    host's cores run at once. Its program runs the iterations from the value of
    its variable [lo] up to that of [hi]. *)

type kernel_info = {
  name : string;  (** The kernel's name. *)
  applied_opts : Opt.t list;  (** The optimisations applied, in order. *)
  opts_to_apply : Opt.t list option;
      (** The optimisations to apply, or [None] to choose them. *)
  estimates : estimates option;  (** The kernel's cost. *)
  beam : int;  (** The beam width to search optimisations with, or [0]. *)
  split : split option;  (** The loop its launch splits, if any. *)
}
(** The type for the arguments of a kernel's {!Op.Sink}. *)

val kernel_info :
  ?name:string ->
  ?applied_opts:Opt.t list ->
  ?opts_to_apply:Opt.t list ->
  ?estimates:estimates ->
  ?beam:int ->
  ?split:split ->
  unit ->
  kernel_info
(** [kernel_info ()] is the argument with these fields. [name] defaults to
    ["test"], the lists to [[]] and [None], [beam] to [0] and [split] to [None].
*)

val function_name : kernel_info -> string
(** [function_name k] is [k.name] as an identifier
    ({!Helpers.to_function_name}). *)

val pp_kernel_info : Format.formatter -> kernel_info -> unit
(** [pp_kernel_info] formats
    [KernelInfo(name='test', applied_opts=(), opts_to_apply=None,
     estimates=None, beam=0)]. *)

type program_info = {
  global_size : sint list;  (** The number of workgroups on each axis. *)
  local_size : sint list;  (** The threads of a workgroup on each axis. *)
  vars : t list;  (** The scalar variables, by slot. *)
  globals : int list;  (** The slots of the buffers. *)
  outs : int list;  (** The slots of the buffers written. *)
  ins : int list;  (** The slots of the buffers read. *)
  target : Helpers.Target.t;  (** The target compiled for. *)
}
(** The type for the arguments of {!Op.Program}. *)

val vals : program_info -> (string * int) list -> int list
(** [vals p vars] is the value in [vars] of each of [p.vars], in order.

    Raises [Invalid_argument] naming a variable that [vars] lacks. *)

val pp_program_info : Format.formatter -> program_info -> unit
(** [pp_program_info] formats
    [ProgramInfo(global_size=(1, 1, 1), local_size=(1, 1, 1), vars=(), ...)]. *)

(** What later passes may do to a buffer an {!Op.Stage} makes. *)
type keep =
  | Removable
      (** Drop the axes its value does not vary along, and inline it back
          where that costs little. *)
  | Broadcast
      (** As {!Removable}, for a value read where it is broadcast: it is not
          inlined back where computing it runs a transcendental function, which
          its readers would compute again for each element of the ranges it
          does not vary along. *)
  | Whole
      (** Nothing: a value the user materialises, or a custom kernel reads,
          is stored as it is. *)

type bufferize_opts = {
  device : device option;  (** Where the new buffer lives. *)
  addrspace : Dtype.addr_space;  (** Its address space. *)
  keep : keep;  (** What later passes may do to it. *)
}
(** The type for the arguments of an {!Op.Stage} that makes a buffer. *)

val pp_bufferize_opts : Format.formatter -> bufferize_opts -> unit
(** [pp_bufferize_opts] formats
    [BufferizeOpts(device='CPU', addrspace=AddrSpace.GLOBAL, removable=True,
     broadcast=False)]: [removable] is [false] for {!Whole} only, and
    [broadcast] [true] for {!Broadcast} only. *)

type hcq_kernel = {
  devices : string list;  (** The devices the kernel runs on. *)
  name : string;  (** The kernel's name. *)
  estimates : estimates;  (** Its cost. *)
  stamps : int list;  (** The timestamp slots of its launches. *)
  profile_key : string option;  (** The key its profile records carry. *)
  input_slots : int list;  (** The argument slots of its buffers. *)
  outs : int list;  (** The buffers it writes, among [input_slots]. *)
  ins : int list;  (** The buffers it reads, among [input_slots]. *)
}
(** The type for the kernels a command-queue call enqueues. *)

type hcq_info = {
  device : string list;  (** The devices whose queues the call submits. *)
  kernels : hcq_kernel list;  (** The kernels it enqueues. *)
  estimates : estimates;  (** Their total cost. *)
  nargs : int;  (** The number of arguments, once lowered; [0] before. *)
  table : int;  (** The argument holding the address table, or [-1]. *)
  inputs : (t * int * string) list;
      (** Each address the table holds from the run's storage, in table order:
          the storage, the byte offset and the device the address is taken on.
      *)
  slots : (string * int) list;
      (** Each device's position of its batch slots among the arguments. *)
  written_bufs : t list;  (** The arguments the call writes. *)
  writes : t list;
      (** The storage its kernels and copies write, those that also read it
          included: the storage under each output of each of its calls, and
          under every argument of a call whose outputs are not known. *)
  copies : (string * string * int) list;
      (** Each copy between two devices a run makes, once for each trip of the
          ranges around it: the device it copies from, the device it copies
          into and its bytes. *)
}
(** The type for the data of a call that submits command queues. [writes] and
    [copies] have no counterpart in tinygrad and are not formatted. *)

val pp_hcq_info : Format.formatter -> hcq_info -> unit
(** [pp_hcq_info] formats
    [HCQInfo(device=('AMD',), kernels=(), estimates=Estimates(ops=0, lds=0,
     mem=0), nargs=0, table=-1, inputs=(), slots=(), written_bufs=())]. *)

type call_info = {
  name : string option;  (** The name of the function called. *)
  precompile : bool;  (** Compile the body on its own. *)
  aux : hcq_info option;  (** The queues it submits, for such a call. *)
  dtype : Dtype.t;  (** The type returned, {!Dtype.Void} for none. *)
}
(** The type for the arguments of {!Op.Call}. *)

val pp_call_info : Format.formatter -> call_info -> unit
(** [pp_call_info] formats
    [CallInfo(None, 'f', False, False, dtype=dtypes.int)], the [dtype] field
    only when it is not [void]. *)

type wmma = {
  dims : int * int * int;  (** The matrix dimensions N, M and K. *)
  dtype_in : Dtype.t;  (** The type of the multiplied operands. *)
  threads : int;  (** The threads that cooperate on one product. *)
  upcast_axes :
    ((int list * int) list * (int list * int) list * (int list * int) list)
    option;
      (** For each operand, the axes still to fold into it and their sizes, or
          [None] once folded. *)
}
(** The type for the arguments of {!Op.Wmma}. *)

(** The type for node arguments. Each operation carries one shape of argument,
    given by {!v}'s table; the others carry [No_arg]. *)
type arg =
  | No_arg
  | Const of Dtype.const  (** {!Op.Const}: its value. *)
  | Dtype of Dtype.t  (** {!Op.Cast}, {!Op.Bitcast}: the target type. *)
  | Param of param_arg  (** {!Op.Param}, {!Op.Buffer}, {!Op.Alloc}. *)
  | Range of { axis_id : int list; axis_type : Axis_type.t }
      (** {!Op.Range}: its identity and role. *)
  | Reduce of { op : Op.t; num_axes : int }
      (** {!Op.Reduce}: the operation, and how many leading axes it reduces ([0]
          for a kernel's reduction over range sources). *)
  | Allreduce of { op : Op.t; device : device }  (** {!Op.Allreduce}. *)
  | Device of device  (** {!Op.Copy}: the target; {!Op.Getaddr}: the device. *)
  | Shard of int  (** {!Op.Mselect}: the shard selected. *)
  | Axes of int list
      (** {!Op.Permute}: the permutation; {!Op.Unshard}: the sharded axes,
          ascending. *)
  | Flips of bool list  (** {!Op.Flip}: which axes are reversed. *)
  | String of string
      (** {!Op.Special}, {!Op.Custom_function}: a name; {!Op.Source}: the source
          text. *)
  | Bytes of string  (** {!Op.Binary}: the bytes. *)
  | Queue of { devices : string list; queue : string }
      (** {!Op.Linear}: the command queue of [devices] it is encoded for. *)
  | Region of { name : string; align : int }
      (** {!Op.Linear}: what its sources lay out, such as ["kernargs"], and the
          alignment in bytes of its start in memory. It prints as its name alone
          when [align] is [128]. *)
  | Code of { code : string; dtype : Dtype.t }
      (** {!Op.Custom}, {!Op.Customi}: source text; {!Op.Ins}: an instruction;
          with the type produced. *)
  | Bufferize of bufferize_opts  (** {!Op.Stage}. *)
  | Kernel of kernel_info  (** {!Op.Sink}: the root of a kernel. *)
  | Program of program_info  (** {!Op.Program}. *)
  | Call of call_info  (** {!Op.Call}. *)
  | Wmma of wmma  (** {!Op.Wmma}. *)

val equal_arg : arg -> arg -> bool
(** [equal_arg a0 a1] is structural equality, with constants compared by
    {!Dtype.equal_const} and nodes by [==]. *)

val pp_arg : Format.formatter -> arg -> unit
(** [pp_arg] formats an argument as the literal that denotes it: [None], [42],
    [ConstFloat(1.5)], [True], [Invalid], [dtypes.float], [(0, AxisType.WEAK)],
    [(Ops.ADD, 0)], ['CPU'], [b'ab\x00'], [ParamArg(...)]. *)

(** {1:tags Tags} *)

(** Tags.

    A tag annotates a node for the passes that handle it; it is part of the
    node's identity, and ignored by everything else. Tags are written as
    literals. *)
module Tag : sig
  (** The type for tags. *)
  type t =
    | Bool of bool
    | Int of int
    | String of string
    | Bytes of string
        (** Bytes, such as the program libraries a Metal command buffer's tag
            lists. *)
    | Dtype of Dtype.t
    | Tuple of t list

  val equal : t -> t -> bool
  (** [equal g0 g1] is [true] iff [g0] and [g1] are the same literal. *)

  val hash : t -> int
  (** [hash g] is a hash of [g], compatible with {!equal}. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a tag as the literal it is: [True], [1], ['mergeable'],
      [b'\x00'], [dtypes.int], [(0, dtypes.int)], [()]. *)
end

(** {1:nodes Nodes} *)

val v : ?src:t list -> ?arg:arg -> ?tag:Tag.t -> Op.t -> t
(** [v ~src ~arg ~tag op] is the node [op] of [src] (default [[]]) with argument
    [arg] (default [No_arg]) and tag [tag] (default none).

    Its data type is derived from [op], [src] and [arg] ({!dtype_of}). The
    operations whose type comes from their argument need its shape: {!Op.Const}
    a [Const], {!Op.Cast} and {!Op.Bitcast} a [Dtype], {!Op.Param}, {!Op.Buffer}
    and {!Op.Alloc} a [Param], {!Op.Custom}, {!Op.Customi} and {!Op.Ins} a
    [Code]. When the setting {!Setting.spec} is 2 or more, the node is also
    checked against the whole specification, the first time it is built.

    Raises [Invalid_argument] if no type can be derived, as for a {!Op.Where}
    whose condition is not boolean, or if the check fails. *)

val op : t -> Op.t
(** [op u] is [u]'s operation. *)

val dtype : t -> Dtype.t
(** [dtype u] is [u]'s data type. *)

val src : t -> t list
(** [src u] is [u]'s sources, in order. *)

val nth : t -> int -> t
(** [nth u i] is the [i]th source of [u], from [0].

    Raises [Invalid_argument] if [u] has no [i]th source. *)

val arg : t -> arg
(** [arg u] is [u]'s argument. *)

val tag : t -> Tag.t option
(** [tag u] is [u]'s tag. *)

val replace : ?op:Op.t -> ?src:t list -> ?arg:arg -> ?tag:Tag.t option -> t -> t
(** [replace u] is [u] with the given fields replaced; [u] itself if none
    changes. *)

val rtag : ?tag:Tag.t -> t -> t
(** [rtag ~tag u] is [u] tagged [tag], by default [Bool true]. *)

val equal : t -> t -> bool
(** [equal u0 u1] is [u0 == u1]: structural equality, since nodes are
    hash-consed. *)

val compare : t -> t -> int
(** [compare] is a total order compatible with {!equal}. It depends on the order
    in which nodes were built; see {!compare_structure} for one that does not.
*)

val hash : t -> int
(** [hash u] is a hash of [u], compatible with {!equal}. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a node and its sources, one per line and indented, as the calls
    that build them:
    {v
    UOp(Ops.ADD, arg=None, src=(
      x0:=UOp(Ops.CONST, arg=42, src=()),
      x0,))
    v}
    A node reached more than once is named [xN:=] where first printed and [xN]
    afterwards. The argument prints as {!pp_arg}, and a tag follows it as
    [, tag=] and the tag, a string without its quotes. *)

val compare_structure : t -> t -> int
(** [compare_structure u0 u1] orders by operation, then argument as {!pp_arg}
    prints it, then data type, then sources, recursively. It ignores tags, so it
    is [0] for nodes that differ only by their tags. The order does not depend
    on how or when the nodes were built. *)

val key : t -> string
(** [key u] is a digest of [u]'s operation, type, argument and sources,
    recursively: equal for equal graphs, whenever and wherever they are built.
*)

module Tbl : Hashtbl.S with type key = t
(** Hash tables keyed by nodes. *)

(** Sets of nodes that remember the order in which nodes joined them. *)
module Nodes : sig
  type node := t

  type t
  (** The type for sets of nodes. *)

  val mem : node -> t -> bool
  (** [mem u s] is [true] iff [u] is in [s]. *)

  val to_list : t -> node list
  (** [to_list s] is the nodes of [s] in the order they joined it. *)

  val fold : (node -> 'a -> 'a) -> t -> 'a -> 'a
  (** [fold f s acc] folds [f] over the nodes of [s] in the order they joined
      it. *)

  val cardinal : t -> int
  (** [cardinal s] is the number of nodes of [s]. *)
end

(** {1:types Data types} *)

val dtype_of : Op.t -> t list -> arg -> Dtype.t
(** [dtype_of op src arg] is the data type of the node [op] of [src] with
    argument [arg]. Unary operations and movements keep their source's type,
    comparisons are {!Dtype.Bool}, transcendentals widen to a float,
    broadcastable operations promote ({!promo_dtype}), and effects are
    {!Dtype.Void}.

    Raises [Invalid_argument] if there is none. *)

val promo_dtype : t list -> Dtype.t
(** [promo_dtype us] is the common type of [us]: theirs if they share it, their
    least upper bound ({!Dtype.least_upper}) otherwise. *)

val identity_element : Op.t -> Dtype.t -> Dtype.const
(** [identity_element op dt] is the value [e] of type [dt] with [op e x = x]:
    [0] for {!Op.Add}, [1] for {!Op.Mul}, and [dt]'s least value for {!Op.Max}.

    Raises [Invalid_argument] for any other [op]. *)

(** {1:graphs Graphs} *)

(** The type for how a walk of a graph treats a call ({!Op.Call}), whose first
    source is its body. *)
type calls =
  | Enter  (** A call's body is walked with its arguments. *)
  | Skip  (** A call's body is left out: the call is its arguments alone. *)

val toposort : ?gate:(t -> bool) -> calls:calls -> t -> t list
(** [toposort ~gate ~calls u] is [u] and the nodes it reaches, each after its
    sources, in the order a depth-first walk of the sources, in order, finishes
    them. The walk enters only the nodes [gate] accepts (default all), and
    treats calls as [calls] says. *)

val topovisit : t -> (t -> 'a) -> 'a Tbl.t -> 'a
(** [topovisit u f cache] is [f u], after [f] has been applied to each node [u]
    reaches, each after its sources, and stored in [cache]. Nodes already in
    [cache] are not visited again. *)

val backward_slice : calls:calls -> t -> Nodes.t
(** [backward_slice ~calls u] is the nodes [u] reaches, without [u], in
    {!toposort}[ ~calls]'s order. *)

val backward_slice_with_self : calls:calls -> t -> Nodes.t
(** [backward_slice_with_self ~calls u] is [u] followed by
    [backward_slice ~calls u]. *)

val op_in_backward_slice_with_self : calls:calls -> t -> Op.t list -> bool
(** [op_in_backward_slice_with_self ~calls u ops] is [true] iff [u] or a node of
    [backward_slice ~calls u] has an operation in [ops]. *)

val reaches : calls:calls -> t -> t -> bool
(** [reaches ~calls u x] is [true] iff [x] is [u] or a node [u] reaches,
    treating calls as [calls] says. *)

val split_uop : t -> Op.t -> t list
(** [split_uop u op] is the operands of the tree of [op] nodes rooted at [u],
    left to right: [[u]] if [u]'s operation is not [op]. *)

(** {1:shapes Shapes} *)

val broadcast_shape : sint list list -> sint list
(** [broadcast_shape shapes] is the shape [shapes] broadcast to: aligned to the
    right, each axis is the size of its sources that is not [1], or [1] if all
    are.

    Raises [Invalid_argument] if an axis has two different sizes other than [1].
*)

val to_max_shape : sint list -> int list
(** [to_max_shape s] is [s] with each symbolic size replaced by its greatest
    value. *)

val sint_to_uop : ?dtype:Dtype.t -> sint -> t
(** [sint_to_uop ~dtype s] is [s] as a node of type [dtype], by default
    {!Dtype.Weak_int}. *)

(** {1:ranges Ranges} *)

val range_start : Op.t -> int option
(** [range_start op] is the index of the first source of an [op] node that is a
    range it ends: [1] for {!Op.Stage}, {!Op.Reduce}, {!Op.End} and {!Op.Call},
    [0] for {!Op.Linear}, and [None] otherwise. *)

val ended_ranges : t -> t list
(** [ended_ranges u] is the ranges that [u] closes: those among its range
    sources, those a {!Op.Barrier} or an {!Op.After}'s effects close, the loop a
    {!Op.Backedge} repeats, an {!Op.Unshard}'s sharding ranges. *)

val ranges : t -> Nodes.t
(** [ranges u] is the ranges [u] runs inside: [u] itself if it is a range, then
    those its sources run inside, minus those it closes. *)

val axis_id : t -> int list
(** [axis_id r] is the identity of the range [r].

    Raises [Invalid_argument] if [r] is not a range. *)

val axis_type : t -> Axis_type.t
(** [axis_type r] is the role of the range [r].

    Raises [Invalid_argument] if [r] is not a range. *)

val range_str : ?color:bool -> t -> string
(** [range_str ~color r] is [r]'s identity written with underscores, a negative
    part written [m] and its magnitude: [0_1], [m1]. It is in [r]'s axis colour
    if [color] (default [false]). *)

val multirange_str : ?color:bool -> ?pad:int -> t list -> string
(** [multirange_str ~color ~pad rs] is the {!range_str} of each of [rs], sorted
    by argument, separated by commas and padded with spaces to [pad] printed
    columns. *)

(** {1:eval Evaluating} *)

val vmin : t -> Dtype.value
(** [vmin u] is a lower bound of every value [u] can take. *)

val vmax : t -> Dtype.value
(** [vmax u] is an upper bound of every value [u] can take. Bounds follow
    interval arithmetic through the integer operations, comparisons, selections,
    casts and loads from constant tables; elsewhere they are the bounds of [u]'s
    type. A committed integer wraps at its width, and so does a weak integer
    operand of an operation on one ({!operand_bounds}): an interval that leaves
    its type is one value wrapped, or the type's bounds. A table holding a NaN
    has its type's bounds. *)

val operand_bounds : t -> t -> Dtype.value * Dtype.value
(** [operand_bounds u s] is the bounds of [u]'s source [s] as [u] reads it: an
    operation on a committed integer commits a weak integer operand to its type,
    which wraps it, so bounds that leave the type are one value wrapped, or the
    type's; an operation on a committed float commits a weak float operand to
    its type, which rounds it, and its bounds round with it. *)

val overflows : t -> Dtype.t -> bool
(** [overflows u dt] is [true] iff [u]'s bounds reach outside [dt]'s. *)

val stored_bounds : t -> Dtype.t -> (Dtype.value * Dtype.value) option
(** [stored_bounds u dt] is [Some (vmin u, vmax u)] iff [u]'s bounds hold more
    than one value and are narrower than [dt]'s: the bounds of storage of [dt]
    that holds only [u]'s values, which every read of it keeps. *)

val exact : Dtype.t -> Dtype.value list -> bool
(** [exact dt vs] is [true] iff [dt] is not a committed integer type, or each of
    [vs] is one of its values. Arithmetic on a committed integer type wraps at
    its width, so it computes what unbounded integers do only where every value
    it takes, [vs], is [exact]. *)

val exec_alu :
  ?truncate_output:bool -> Op.t -> Dtype.t -> Dtype.const list -> Dtype.const
(** [exec_alu ~truncate_output op dt args] computes the arithmetic operation
    [op] on the scalars [args], one lane of a vector operation, exactly, then
    truncates the result to [dt] ({!Dtype.truncate}) if [truncate_output]
    (default [true]). Integers are unbounded before truncation. Out-of-domain
    values follow IEEE: [log2 0.] is [-inf], [sqrt] of a negative is NaN,
    [1. /. 0.] is an infinity of the operand's sign, [sin] of an infinity is
    NaN, and an overflowing [exp2] or [pow] is the infinity IEEE gives. The NaN
    of an operation none of whose operands is NaN is {!Dtype.nan}, whatever the
    host gives. {!Op.Fdiv} is IEEE division: a zero divisor gives an infinity of
    the quotient's sign, and [0. /. 0.] is NaN. {!Op.Trunc} of a float is
    IEEE's, a float of its operand's sign: [trunc (-0.5)] is [-0.]. {!Op.Max} is
    [y] if [x < y] and [x] otherwise, as compiled code computes it, so a NaN
    wins only as the first operand. A binary operation on [`Invalid] is
    [`Invalid]. {!Op.Where} is the branch it picks, [`Invalid] included; its
    condition must not be [`Invalid].

    Raises [Invalid_argument] if [op] is not an arithmetic operation or the
    arguments do not fit it, as for a shift by a negative count, which has no
    value. *)

(** {1:syntax Construction} *)

val const : ?dtype:Dtype.t -> Dtype.const -> t
(** [const ~dtype c] is the constant [c] of type [dtype]. Without [dtype], or
    for [`Invalid], it is the weak literal of [c]'s kind ({!Dtype.of_const}). A
    constant of any other type is a {!Op.Cast} of the literal, whose value is
    [c] converted to [dtype] ({!Dtype.const}). *)

val int : ?dtype:Dtype.t -> int -> t
(** [int ~dtype n] is [const ~dtype (`Int n)]. *)

val float : ?dtype:Dtype.t -> float -> t
(** [float ~dtype x] is [const ~dtype (`Float x)]. *)

val bool : ?dtype:Dtype.t -> bool -> t
(** [bool ~dtype b] is [const ~dtype (`Bool b)]. *)

val invalid : t
(** [invalid] is [const `Invalid]. *)

val value : t -> Dtype.const
(** [value c] is the value of the constant [c], read through a cast.

    Raises [Invalid_argument] if [c] is neither a constant nor a cast of one. *)

val is_invalid : t -> bool
(** [is_invalid u] is [true] iff [u] is the constant [`Invalid]. *)

val ccast : t -> Dtype.t -> t
(** [ccast u dt] is [u] as a [dt] constant if [u] is a constant, and [cast u dt]
    otherwise. *)

val cconst : Dtype.t -> Dtype.const -> t
(** [cconst dt c] is the cast to [dt] of the literal [c], even where {!const}
    would fold the cast away. *)

val sink : ?kernel:kernel_info -> ?tag:Tag.t -> t list -> t
(** [sink ~kernel us] is the {!Op.Sink} of [us], the root of the kernel [kernel]
    if given. *)

val group : t list -> t
(** [group us] is the {!Op.Group} of [us], or the one node of [us]. *)

val broadcast : t -> int -> t
(** [broadcast u n] is the {!Op.Stack} of [n] copies of [u], or [u] if [n] is
    [1]. *)

val index : ?tag:Tag.t -> t -> t list -> t
(** [index u idxs] is the {!Op.Index} of [u] by [idxs]; the [i]th source of a
    stack if [u] is a stack and [idxs] the constant [i]. *)

val load : ?tag:Tag.t -> t -> t list -> t
(** [load p rest] is the {!Op.Load} through [p]. *)

val store : ?gate:t -> ?tag:Tag.t -> t -> t -> t
(** [store p x] is the {!Op.Store} of [x] through [p], guarded by [gate]. *)

val end_ : t -> t list -> t
(** [end_ u rs] is the {!Op.End} of the ranges [rs] around [u], or [u] if [rs]
    is empty. *)

val backedge : t -> loop:t -> cond:t -> t
(** In a kernel, [backedge u ~loop ~cond] runs [u], then repeats [loop] while
    [cond] holds.

    Around a call of a schedule, or an {!Op.Linear} of calls, it tests [cond]
    before each trip: [loop] is a {!Axis_type.Loop} range, or the [0] that a
    range of one trip simplifies to, [cond] is storage of one boolean, and the
    calls run in order once per trip of [loop] while [cond] holds, so they may
    run no trip. *)

val after : ?tag:Tag.t -> t -> t list -> t
(** [after u deps] is [u] ordered after [deps], or [u] if [deps] is empty. *)

val without_after : t -> t
(** [without_after u] is [u] without its {!Op.After}s. *)

val barrier : t -> t list -> t
(** [barrier u rest] is the {!Op.Barrier} of [u] and [rest]. *)

val ins : ?src:t list -> ?dtype:Dtype.t -> ?tag:Tag.t option -> t -> string -> t
(** [ins u i] is the instruction [i] with [u]'s sources, type and tag, unless
    given. *)

val range :
  ?axis_type:Axis_type.t ->
  ?dtype:Dtype.t ->
  ?src:t list ->
  sint ->
  int list ->
  t
(** [range ~axis_type ~dtype ~src end_ axis_id] is the loop variable counting
    from [0] to [end_], excluded, of type [dtype] (default {!Dtype.Weak_int})
    and role [axis_type] (default {!Axis_type.Weak}), ordered after [src]. *)

val loop : int -> t
(** [loop axis] is the unbounded loop of identity [[axis]]. *)

val special : sint -> string -> t
(** [special end_ name] is the hardware index [name], bounded by [end_]. *)

val wmma :
  ?upcast_axes:
    (int list * int) list * (int list * int) list * (int list * int) list ->
  t ->
  t ->
  acc:t ->
  dims:int * int * int ->
  threads:int ->
  t
(** [wmma a b ~acc ~dims ~threads] is the matrix multiply-accumulate of [a] and
    [b] into [acc]. *)

val reduce : t -> Op.t -> t list -> t
(** [reduce u op ranges] is the kernel reduction of [u] with [op] over [ranges].
*)

val get_idx : t -> t
(** [get_idx u] is [u] without its validity condition. *)

val get_valid : t -> t
(** [get_valid u] is [u]'s validity condition: [true] unless [u] is [`Invalid].
*)

val bufferize : ?opts:bufferize_opts -> t -> t list -> t
(** [bufferize ~opts u ranges] is the {!Op.Stage} of [u] over [ranges], into a
    buffer placed by [opts] if given. *)

(** {1:multi Several devices} *)

val device : t -> device option
(** [device u] is where [u] lives: its argument's device for storage, copies and
    reductions across devices, its sources' first otherwise. *)

val on_disk : t -> bool
(** [on_disk u] is [true] iff [u] lives on a single disk device. *)

val sharding : t -> (int * t) list
(** [sharding u] is each axis an {!Op.Unshard} [u] is sharded on, with the range
    over its shards; [[]] for any other node. *)

val copy_to_device : ?shard:int -> t -> device -> t
(** [copy_to_device ~shard u d] is the copy of [u], or of its shard [shard], to
    [d].

    Raises [Invalid_argument] if [d] is a disk or [u]'s type is weak. *)

val device_range_src : device option -> t list
(** [device_range_src d] is the {!Axis_type.Device} range a node on several
    devices carries, or [[]]. *)

val mselect : t -> int -> t
(** [mselect u i] is [u]'s shard on its [i]th device. *)

val mstack : t -> t list -> t
(** [mstack u rest] is the multi-device value of [u :: rest], or [u]. *)

val allreduce : t -> Op.t -> device -> t
(** [allreduce u op d] reduces [u] across [d] with [op].

    Raises [Invalid_argument] if [u] is not on several devices. *)

(** {1:movement Movement} *)

val base : t -> t
(** [base u] is [u] without its movements. *)

val unsharded_base : t -> t
(** [unsharded_base u] is [base u], also without an {!Op.Unshard}. *)

val storage_base : t -> t
(** [storage_base u] is the storage [u] targets: [unsharded_base u] without
    bitcasts and {!Op.After}s. *)

(** The type for the arguments of movements. *)
type movement =
  | Reshape of sint list  (** The new shape. *)
  | Expand of sint list  (** The axes added in front. *)
  | Pad of (sint * sint) list
      (** Where each axis starts in the result, and the result's size. *)
  | Shrink of (sint * sint) list
      (** Where each axis starts in the source, and the result's size. *)
  | Permute of int list  (** The source axis of each result axis. *)
  | Flip of bool list  (** Which axes are reversed. *)

(** {1:storage Storage}

    Sizes of storage are [int]s: a shape of more than [max_int] elements raises
    [Invalid_argument]. *)

val addrspace : t -> Dtype.addr_space option
(** [addrspace u] is the address space [u] lives in. *)

val buf_uop : t -> t
(** [buf_uop u] is the storage node [u] accesses. *)

val is_virtual : t -> bool
(** [is_virtual u] is [true] iff [u] cannot back a buffer as it is: it has no
    device, or its type is weak. *)

val has_buffer_identity : ?after_ok:bool -> t -> bool
(** [has_buffer_identity ~after_ok u] is [true] iff [u] names storage, through
    reshapes, unshards, shard selections and, if [after_ok], an {!Op.After}. *)

val needs_storage : t -> bool
(** [needs_storage u] is [true] iff realizing [u] allocates. *)

val getaddr : ?device:string -> t -> t
(** [getaddr ~device u] is the address of the storage or command sequence [u] on
    [device] (default [u]'s first), or [u] if it has none. *)

val unique_num : unit -> int
(** [unique_num ()] is a slot number no other call returns. *)

val new_buffer : ?slot:int -> ?phase:int -> device -> int -> Dtype.t -> t
(** [new_buffer ~slot ~phase d size dt] is a {!Op.Buffer} of [size] elements of
    [dt] on [d], in [slot] (default {!unique_num}), whose first element lies
    [phase] bytes past a 16-byte boundary (default [0]; see {!param_arg}).

    Raises [Invalid_argument] if [dt] is weak. *)

val set : ?ends:t list -> t -> t -> t
(** [set p x] stores [x] through [p], ends [ends], and is [p]'s storage after
    the store. *)

(** {1:variables Variables} *)

val variable :
  ?dtype:Dtype.t ->
  ?multiple_of:int ->
  string ->
  Dtype.value ->
  Dtype.value ->
  t
(** [variable ~dtype ~multiple_of name lo hi] is the scalar variable [name] of
    type [dtype] (default {!Dtype.Weak_int}) ranging over [lo] to [hi]. *)

val expr : t -> string
(** [expr u] is the name of the parameter or buffer [u].

    Raises [Invalid_argument] if it has none. *)

(** {1:symbolic Divisibility} *)

val const_factor : t -> Bigint.t
(** [const_factor u] is a known integer that divides every value of [u]. *)

val pop_const : ?op:Op.t -> t -> t * Dtype.const
(** [pop_const ~op u] is [(x, c)] for [u = op x c] with [c] a constant, and
    [(u, identity_element op)] otherwise. [op] defaults to {!Op.Add}. *)

(** {1:calls Calls} *)

val body : t -> t
(** [body c] is the body the call [c] calls.

    Raises [Invalid_argument] if [c] is not a call. *)

val src_without_body : t -> t list
(** [src_without_body u] is a call's arguments, and any other node's sources. *)

val opaque_call_bodies : Op.Set.t
(** [opaque_call_bodies] is the operations a call body can have: {!Op.Sink},
    {!Op.Program}, {!Op.Linear}, {!Op.Store} and {!Op.Custom_function}. *)

val is_inline_call : t -> bool
(** [is_inline_call u] is [true] iff [u] calls a plain sink that is not compiled
    on its own. *)

val has_unbound_outputs : t -> bool
(** [has_unbound_outputs c] is [true] iff the call [c] still has outputs that no
    buffer is bound to. *)

val unbound_outputs : t -> t list
(** [unbound_outputs c] is those outputs, each ordered after [c]. *)

val custom_function : string -> t list -> t
(** [custom_function name args] is the external function [name]. *)

val call :
  ?ret_dtype:Dtype.t ->
  ?name:string ->
  ?precompile:bool ->
  ?aux:hcq_info ->
  t ->
  t list ->
  t
(** [call body args] calls [body] on [args], returning [ret_dtype] (default
    {!Dtype.Void}).

    Raises [Invalid_argument] if [body] cannot be called ({!opaque_call_bodies})
    or a range other than a device range leaks out of it. *)

(** {1:elementwise Elementwise} *)

(** Elementwise operations.

    Nodes ({!Ops}) and patterns ({!Upat}) share them. On nodes, a binary
    operation first promotes its operands to their least upper type
    ({!Dtype.least_upper}), a weak constant staying weak; the node's shape is
    their broadcast shape ({!broadcast_shape}). On patterns, each builds the
    pattern of the node the operation builds. *)
module type Elementwise = sig
  type t

  val alu : t -> Op.t -> t list -> t
  (** [alu x op rest] is the operation [op] on [x :: rest] as given. *)

  val cast : t -> Dtype.t -> t
  (** [cast x dt] is [x] converted to [dt], or [x] if it is of type [dt]. *)

  (** {2:arith Arithmetic}

      [add], [mul], [lt], [ne], [maximum], [bitwise_and], [bitwise_or],
      [bitwise_xor], [shl] and [shr] are {!Op.Add}, {!Op.Mul}, {!Op.Cmplt},
      {!Op.Cmpne}, {!Op.Max}, {!Op.And}, {!Op.Or}, {!Op.Xor}, {!Op.Shl} and
      {!Op.Shr} of their promoted operands. [reciprocal], [trunc], [sqrt],
      [exp2] and [log2] are {!Op.Reciprocal}, {!Op.Trunc}, {!Op.Sqrt},
      {!Op.Exp2} and {!Op.Log2} of their operand. The others are built from
      these. *)

  val add : t -> t -> t
  val mul : t -> t -> t
  val lt : t -> t -> t
  val ne : t -> t -> t
  val maximum : t -> t -> t
  val bitwise_and : t -> t -> t
  val bitwise_or : t -> t -> t
  val bitwise_xor : t -> t -> t
  val shl : t -> t -> t
  val shr : t -> t -> t
  val reciprocal : t -> t
  val trunc : t -> t
  val sqrt : t -> t
  val exp2 : t -> t
  val log2 : t -> t

  val sub : t -> t -> t
  (** [sub x y] is [x + neg y]. *)

  val neg : t -> t
  (** [neg x] is [logical_not x] for booleans and [x * -1] otherwise. *)

  val div : ?rounding:[ `Trunc | `Floor ] -> t -> t -> t
  (** [div ~rounding x y] divides integers by {!Op.Cdiv} or {!Op.Floordiv} when
      rounding; otherwise it multiplies [x], as a float, by [y]'s reciprocal,
      then rounds as asked. *)

  val mod_ : t -> t -> t
  (** [mod_ x y] is the remainder of [div ~rounding:`Floor x y]. *)

  val fmod : t -> t -> t
  (** [fmod x y] is the remainder of [div ~rounding:`Trunc x y]. *)

  val floor : t -> t
  (** [floor x] is [trunc x], less one where that is above [x]. *)

  val pow : t -> t -> t
  (** [pow x y] is {!Op.Pow} of [x] and [y].

      Raises [Invalid_argument] if the promoted type is not a float and [y] is a
      constant other than a non-negative integer. *)

  val gt : t -> t -> t
  (** [gt x y] is [lt y x]. *)

  val le : t -> t -> t
  (** [le x y] is [logical_not (gt x y)]. *)

  val ge : t -> t -> t
  (** [ge x y] is [logical_not (lt x y)]. *)

  val eq : t -> t -> t
  (** [eq x y] is [logical_not (ne x y)]. *)

  val logical_not : t -> t
  (** [logical_not x] is [ne (cast x Bool) true]. *)

  val bitwise_not : t -> t
  (** [bitwise_not x] is [logical_not x] for booleans, and [x] xor all ones
      otherwise. *)

  val where : t -> t -> t -> t
  (** [where c x y] is [x] where [c] holds and [y] elsewhere. *)

  val minimum : t -> t -> t
  (** [minimum x y] is [neg (maximum (neg x) (neg y))] for floats; for integers,
      the same through an order-reversing xor. *)

  (** Operators, for local opening: [O.(x + int 4 * y)]. Each is the function of
      the same meaning: [+] is {!add}, [//] is [div ~rounding:`Floor], [%] is
      {!mod_}, [<>] is {!ne}, [land], [lor], [lxor] and [lnot] are the bitwise
      operations, [lsl] and [lsr] the shifts. *)
  module O : sig
    val int : int -> t
    (** [int n] is the weak literal [n]. *)

    val float : float -> t
    (** [float x] is the weak literal [x]. *)

    val bool : bool -> t
    (** [bool b] is the literal [b]. *)

    val ( + ) : t -> t -> t
    val ( - ) : t -> t -> t
    val ( * ) : t -> t -> t
    val ( / ) : t -> t -> t
    val ( // ) : t -> t -> t
    val ( % ) : t -> t -> t
    val ( ~- ) : t -> t
    val ( < ) : t -> t -> t
    val ( > ) : t -> t -> t
    val ( <= ) : t -> t -> t
    val ( >= ) : t -> t -> t
    val ( <> ) : t -> t -> t
    val ( land ) : t -> t -> t
    val ( lor ) : t -> t -> t
    val ( lxor ) : t -> t -> t
    val lnot : t -> t
    val ( lsl ) : t -> t -> t
    val ( lsr ) : t -> t -> t
  end
end

include Elementwise with type t := t

val bitcast : t -> Dtype.t -> t
(** [bitcast x dt] reinterprets [x]'s bits as [dt], or is [x] if it is of type
    [dt].

    Raises [Invalid_argument] if either type is weak. *)

val commit_dtype : ?default_int:Dtype.t -> t -> Dtype.t
(** [commit_dtype x] is the type [x] is stored as: a weak integer commits to the
    narrowest type holding its bounds ({!Dtype.commit_int}), a weak float to its
    default ({!Dtype.strong}). *)

val element_size : t -> int
(** [element_size x] is the size in bytes of one of [x]'s elements.

    Raises [Invalid_argument] if [x]'s type is weak. *)

val contiguous : t -> t
(** [contiguous x] is [x] materialised into its own buffer: [x] itself if it is
    weak, already staged, placed nowhere, or storage. *)

val usum : t -> t list -> t
(** [usum x ys] is the sum of [x :: ys], or their disjunction if [x] is boolean.
*)

val uprod : t -> t list -> t
(** [uprod x ys] is the product of [x :: ys], or their conjunction if [x] is
    boolean. *)

(** {1:patterns Patterns} *)

(** Patterns.

    A pattern describes nodes by their operation, type, argument, tag and
    sources, and names the nodes it matches. A name used twice must match the
    same node. *)
module Upat : sig
  type node := t

  type t
  (** The type for patterns. *)

  val v :
    ?op:Op.Set.t ->
    ?dtype:Dtype.t list ->
    ?src:t list ->
    ?perm:t list ->
    ?each:t ->
    ?allow_any_len:bool ->
    ?arg:arg ->
    ?name:string ->
    ?tag:Tag.t list ->
    ?early_reject:Op.t list ->
    unit ->
    t
  (** [v ()] matches the nodes of an operation in [op], a type in [dtype],
      argument [arg] and a tag in [tag], each when given. Sources are matched in
      order by [src], in any order by [perm], or each by [each]; [src] and
      [perm] require exactly as many sources unless [allow_any_len]. [name]
      names the node matched. [early_reject] is operations the node's sources
      must include for the pattern to be tried, by default the operations the
      source patterns require.

      Arguments compare as numbers do: [`Int 0], [`Float 0.] and [`Bool false]
      are equal, and so are two NaNs.

      Raises [Invalid_argument] if more than one of [src], [perm] and [each] is
      given. *)

  val op :
    ?dtype:Dtype.t list ->
    ?src:t list ->
    ?perm:t list ->
    ?each:t ->
    ?allow_any_len:bool ->
    ?arg:arg ->
    ?name:string ->
    ?tag:Tag.t list ->
    ?early_reject:Op.t list ->
    Op.t ->
    t
  (** [op o] is [v ~op:(Op.Set.of_list [o]) ()]. *)

  val wild : t
  (** [wild] is [v ()], which matches any node. *)

  val var : ?dtype:Dtype.t list -> string -> t
  (** [var ~dtype name] matches any node of a type in [dtype], named [name]. *)

  val cvar : ?dtype:Dtype.t list -> ?arg:Dtype.const -> string -> t
  (** [cvar ~dtype ~arg name] matches a constant, named [name]. *)

  val const : ?dtype:Dtype.t list -> Dtype.const -> t
  (** [const ~dtype c] matches the constant [c]. *)

  val any : t list -> t
  (** [any ps] matches what any of [ps] matches. *)

  val named : string -> t -> t
  (** [named name p] is [p] naming the node it matches [name]. *)

  val or_casted : ?name:string -> t -> t
  (** [or_casted p] matches [p] or a cast of it. *)

  val or_bitcasted : ?name:string -> t -> t
  (** [or_bitcasted p] matches [p] or a bitcast of it. *)

  val or_after : ?name:string -> t -> t
  (** [or_after p] matches [p] or [p] ordered after anything. *)

  val f :
    ?dtype:Dtype.t list ->
    ?name:string ->
    ?arg:arg ->
    ?allow_any_len:bool ->
    t ->
    Op.t ->
    t
  (** [f p o] matches the node [o] of the one source [p] matches: [f p Cast]
      matches a cast of [p] to any type. *)

  val sink : ?name:string -> t list -> t
  val index : ?name:string -> ?allow_any_len:bool -> t -> t list -> t
  val load : ?name:string -> ?allow_any_len:bool -> t -> t list -> t
  val store : ?name:string -> ?allow_any_len:bool -> t -> t list -> t

  val reduce :
    ?name:string -> ?op:Op.t -> ?allow_any_len:bool -> t -> t list -> t

  val broadcast : ?name:string -> t -> t
  (** [broadcast p] matches a stack whose sources each match [p]. *)

  val after : ?name:string -> ?allow_any_len:bool -> t -> t list -> t
  val end_ : ?name:string -> ?allow_any_len:bool -> t -> t list -> t
  val backedge : ?name:string -> t -> loop:t -> cond:t -> t

  val bitcast : ?dtype:Dtype.t -> t -> t
  (** [bitcast ~dtype p] matches a bitcast of [p], to [dtype] if given. *)

  val dtype : t -> Dtype.t
  (** [dtype p] is the first type [p] requires, or {!Dtype.Void}. *)

  include Elementwise with type t := t
  (** Commutative operations match their operands in either order, and the
      literals of {!O} match constants of any type ({!cvar}). [cast p dt] is [p]
      if [p] requires exactly [dt]. [//] and [%] match {!Op.Floordiv} and
      {!Op.Floormod}. *)

  include module type of O
  (** The operators are also at the top, so that [Upat.(var "x" + cvar "c")]
      reads as the pattern it builds. *)

  val match_ : t -> node -> (string * node) list list
  (** [match_ p u] is the naming of each way [p] matches [u]; [[]] if it does
      not. *)
end

(** Pattern matchers.

    An ordered list of rules, each a pattern and a function of the nodes it
    names. Matching a node tries the rules in order and is the first result. A
    matcher is typed by the context its rules read, ['ctx], and by its results,
    ['r]. A rewriter's results are nodes; a fold's are anything else, such as
    verdicts or text. *)
module Pattern_matcher : sig
  type node := t

  type ('ctx, 'r) rule
  (** The type for rules. *)

  val rule : Upat.t -> ((string -> node) -> 'r option) -> ('ctx, 'r) rule
  (** [rule p f] is [f m] for a node [p] matches, where [m name] is the node [p]
      names [name]. [m] raises [Invalid_argument] on a name the match did not
      bind. *)

  val rule_ctx :
    Upat.t -> ('ctx -> (string -> node) -> 'r option) -> ('ctx, 'r) rule
  (** [rule_ctx p f] is {!rule}, with the context passed to [f]. *)

  type ('ctx, 'r) t
  (** The type for pattern matchers. *)

  val v : (unit -> ('ctx, node) rule list) -> ('ctx, node) t
  (** [v rules] is the rewriter of the rules [rules ()] returns, which it
      calls on its first rewrite. Domains that rewrite first at once may each
      call [rules ()]; all of them use the one result the matcher keeps. A
      rule declines by returning [None] or the node it matched; the next rule
      is then tried.

      The first rewrite raises [Invalid_argument] if a rule's pattern has no
      operation. *)

  val fold : (unit -> ('ctx, 'r) rule list) -> ('ctx, 'r) t
  (** [fold rules] is the matcher of the rules [rules ()] returns, which it
      calls on its first rewrite as {!v} does, and whose results are not the
      nodes they match. A rule declines by returning [None].

      The first rewrite raises [Invalid_argument] if a rule's pattern has no
      operation. *)

  val append : ('ctx, 'r) t -> ('ctx, 'r) t -> ('ctx, 'r) t
  (** [append m0 m1] tries [m0]'s rules, then [m1]'s, each declining as its
      matcher says. *)

  val concat : ('ctx, 'r) t list -> ('ctx, 'r) t
  (** [concat ms] tries the rules of [ms] in order. *)

  val with_ctx : (unit, 'r) t -> ('ctx, 'r) t
  (** [with_ctx m] is [m], ignoring the context, for appending to matchers that
      read one. *)

  val rewrite : ('ctx, 'r) t -> 'ctx -> node -> 'r option
  (** [rewrite m ctx u] is the result of the first rule of [m] that matches [u]
      and does not decline, trying each naming of each pattern in turn. *)
end

(** {1:rewrite Rewriting} *)

exception Bottom_up_gate
(** Raised by a rule before the sources ({!Before_sources}, {!Around_sources}'s
    [before]) under {!Fixed_point} to keep the node it last produced and leave
    its sources unvisited. Raised by a rule after the sources, or under {!Once},
    it escapes {!graph_rewrite}. *)

(** The type for how often a rewrite applies its rules to a node and its
    results. *)
type pass =
  | Fixed_point
      (** A rule before the sources rewrites a node until none applies, and the
          sources of the result are then rewritten. A node that a rule after the
          sources rewrites, or that is rebuilt on rewritten sources, is
          rewritten again in turn, to a fixed point. *)
  | Once
      (** A rule rewrites a node at most once, and its result is never rewritten
          again: a node that a rule before the sources rewrites is replaced and
          its sources are left as they are. *)

(** The type for the rules of a rewrite and when they meet a node. *)
type 'ctx rules =
  | After_sources of ('ctx, t) Pattern_matcher.t
      (** Rewrites each node once its sources are rewritten. *)
  | Before_sources of ('ctx, t) Pattern_matcher.t
      (** Rewrites each node before its sources. *)
  | Around_sources of {
      before : ('ctx, t) Pattern_matcher.t;
      after : ('ctx, t) Pattern_matcher.t;
    }
      (** Rewrites each node with [before] before its sources and with [after]
          once they are rewritten. *)

val graph_rewrite : calls:calls -> pass:pass -> ctx:'ctx -> t -> 'ctx rules -> t
(** [graph_rewrite ~calls ~pass ~ctx u rules] is [u] rewritten by [rules] as
    [pass] says: each node is rebuilt on its rewritten sources and rewritten by
    the first rule that applies, and shared nodes stay shared. A call's body is
    rewritten with its arguments under [Enter], and left alone under [Skip].

    Raises [Invalid_argument] if a {!Fixed_point} rewrite does not terminate: a
    rule before the sources cycles, the work list exceeds
    {!Setting.rewrite_stack_limit}, or a node's rewrite depends on itself, and
    {!Bottom_up_gate} as that exception's description says. *)

val substitute :
  ?extra_pm:(t Tbl.t, t) Pattern_matcher.t ->
  calls:calls ->
  pass:pass ->
  t ->
  (t * t) list ->
  t
(** [substitute ~calls ~pass u subs] is [u] with each node of [subs] replaced by
    its pair, top-first, rewriting with [extra_pm] as well, which reads the
    substitution, and treating calls and results as {!graph_rewrite} does. Nodes
    paired with themselves are ignored. *)

val pm_substitute : (t Tbl.t, t) Pattern_matcher.t
(** [pm_substitute] replaces each node its context maps by its image. *)

val remove_all_tags : (unit, t) Pattern_matcher.t
(** [remove_all_tags] removes every node's tag. *)

val pm_drop_after : (unit, t) Pattern_matcher.t
(** [pm_drop_after] replaces each {!Op.After} with its first source. *)

val resolve_returned_after : t -> t -> t option
(** [resolve_returned_after r effects] is the value stored into the output [r]
    by the one store of [effects] into [r]'s storage: [r] after that store if
    [r] is a parameter of the enclosing function. *)

val gate_kernel_sink : t -> bool
(** [gate_kernel_sink u] is [false] for a linear program and a kernel's sink, so
    walks gated by it do not enter kernels. *)

(**/**)

(* The memos of the properties [Shape] and [Call] compute: only they write them. *)

val shape_memo : t -> sint list option option
val set_shape_memo : t -> sint list option -> unit
val movement_memo : t -> movement option
val set_movement_memo : t -> movement -> unit
val axis_memo : t -> int option option
val set_axis_memo : t -> int option -> unit
