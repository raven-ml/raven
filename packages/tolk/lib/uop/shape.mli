(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Shapes and simplification.

    The functions on nodes ({!Ops.t}) that need {!simplify}: a symbolic size
    is compared, bounded and computed by simplifying the node that holds it.
    Shapes ({!shape}), the movements that change them, storage of a shape,
    sharding, variables, divisibility and calls all read sizes, and so does
    the simplification itself, whose rules fold constants into the shapes of
    the nodes they replace.

    {b Errors.} A broken precondition raises [Invalid_argument]. *)

(** {1:shapes Shapes} *)

val shape_opt : Ops.t -> Ops.sint list option
(** [shape_opt u] is [u]'s shape, or [None] for nodes that have none, such as
    effects and program structure. Movements check their argument against their
    source's shape; broadcastable operations broadcast their sources' shapes
    ({!Ops.broadcast_shape}).

    Raises [Invalid_argument] if a movement does not fit its source's shape, or
    if sources cannot be broadcast. *)

val shape : Ops.t -> Ops.sint list
(** [shape u] is [u]'s shape.

    Raises [Invalid_argument] if [u] has none. *)

val ndim : Ops.t -> int
(** [ndim u] is the length of [u]'s shape. *)

val numel : Ops.t -> Ops.sint
(** [numel u] is the product of [u]'s shape. *)

val max_shape : Ops.t -> int list
(** [max_shape u] is [u]'s shape with each symbolic size replaced by its
    greatest value ({!Ops.to_max_shape}). *)

val max_numel : Ops.t -> int
(** [max_numel u] is the product of [max_shape u]. *)

val broadcast_axes : Ops.sint list -> Ops.sint list -> int list
(** [broadcast_axes src out] is the axes of [out] that broadcasting [src] to
    [out] adds or expands.

    Raises [Invalid_argument] if [src] has more axes than [out]. *)

(** {1:resolve Resolving} *)

val resolve : ?default:bool -> Ops.t -> bool
(** [resolve ~default u] is the value of the boolean node [u] if its
    simplification ({!simplify}) has one possible value, and [default] (default
    [true]) otherwise.

    Raises [Invalid_argument] if [u] is not boolean. *)

val simplify : Ops.t -> Ops.t
(** [simplify u] is [u] rewritten with the symbolic rules of the library to a
    fixed point. A constant is itself, and so is a sink of constants and stacks
    of constants, which the rules leave as they are.

    Raises [Invalid_argument] if the rules are not installed: the library
    installs them when it is initialised (see {!Private}). *)

val ssimplify : Ops.t -> Ops.sint
(** [ssimplify u] is [simplify u] as an [Int] if it is an integer constant, a
    [Sym] otherwise. *)

val smax : Ops.sint list -> Ops.sint
(** [smax ss] is the greatest of [ss], a {!Op.Max} node if some are symbolic,
    simplified.

    Raises [Invalid_argument] if [ss] is empty. *)

val smin : Ops.sint list -> Ops.sint
(** [smin ss] is the least of [ss], as {!smax}. *)

val to_bool : Ops.t -> bool
(** [to_bool u] is the value of the boolean node [u].

    Raises [Invalid_argument] if [u] is not boolean or its simplification has
    more than one possible value. *)

val to_z : Ops.t -> Bigint.t
(** [to_z u] is the value of the integer node [u], as {!to_bool}. *)

val to_float : Ops.t -> float
(** [to_float u] is the value of the float node [u], as {!to_bool}. *)

(** Symbolic integers.

    Arithmetic computes on integers, and builds nodes as soon as an operand is
    symbolic. It is exact: a result that does not fit an [int] raises
    [Invalid_argument]. *)
module Sint : sig
  type node := Ops.t
  type t = Ops.sint

  val ( + ) : t -> t -> t
  val ( - ) : t -> t -> t
  val ( * ) : t -> t -> t

  val ( // ) : t -> t -> t
  (** Division rounding towards negative infinity. *)

  val ( % ) : t -> t -> t
  (** The remainder of [//], with the sign of the divisor. *)

  val prod : t list -> t
  (** [prod ss] is the product of [ss], [Int 1] if empty. *)

  (** The type for conditions on symbolic integers: known, or a boolean node. *)
  type cond = Known of bool | Cond of node

  val ( < ) : t -> t -> cond
  val ( <= ) : t -> t -> cond
  val ( > ) : t -> t -> cond
  val ( >= ) : t -> t -> cond
  val ( <> ) : t -> t -> cond

  val resolve : ?default:bool -> cond -> bool
  (** [resolve ~default c] is [c]'s value if known, and {!resolve}'s
      otherwise. *)

  val equal : t -> t -> bool
  (** [equal s0 s1] is [true] iff both are the same integer or the same node. It
      does not decide whether two nodes have the same value. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats an integer in decimal, and a node as {!Ops.pp} does. *)
end

(** {1:eval Evaluating} *)

val sym_infer : Ops.sint -> (string * int) list -> int
(** [sym_infer s vars] is the value of [s] with each variable named in [vars]
    bound to its value. Integer arithmetic is exact, divisions round as their
    operations say, and casts convert without truncating.

    Raises [Invalid_argument] if [s] reads a variable [vars] lacks. *)

val sym_compile : Ops.sint -> (Ops.t -> 'env -> int) -> 'env -> int
(** [sym_compile s var] is the function computing [s] in an environment, each
    variable [v] of [s] read by [var v]: [sym_compile s var env] is
    [sym_infer s vars] for [vars] binding each variable [v] to [var v env]. [s]
    is simplified once, and integer arithmetic that fits an [int] is computed on
    [int]s, so that a call costs a few operations: a value that a schedule
    computes on each run, such as the offset of a view that moves with a
    range.

    The function raises as {!sym_infer} does, and as [var] does. *)

(** {1:syntax Construction} *)

val consts : ?dtype:Dtype.t -> Dtype.const list -> Ops.t
(** [consts ~dtype cs] is the {!stack} of [const ~dtype c] for each [c] of [cs].
    [dtype] defaults to the committed type of the literals [cs]
    ({!Dtype.of_consts}). *)

val const_like : ?dtype:Dtype.t -> Ops.t -> Dtype.const -> Ops.t
(** [const_like ~dtype u c] is the constant [c] of type [dtype] (default [u]'s),
    expanded to [u]'s shape. *)

val vconst_like : Ops.t -> Dtype.const -> Ops.t
(** [vconst_like u c] is the constant [c] of [u]'s type repeated [max_numel u]
    times. *)

val stack : ?axis:int -> Ops.t list -> Ops.t
(** [stack ~axis us] is the {!Op.Stack} of [us], each converted to their common
    type ({!Ops.ccast}) except [`Invalid] constants, with the new axis moved to
    [axis] (default [0]).

    Raises [Invalid_argument] if [us] is empty or its nodes differ in shape. *)

val rop : Ops.t -> Op.t -> int list -> Ops.t
(** [rop u op axes] reduces the axes [axes] of [u] with [op]: axes of size [1]
    are reshaped away, and the others are permuted to the front and reduced by
    one {!Op.Reduce}. *)

val valid : Ops.t -> Ops.t -> Ops.t
(** [valid u cond] is [u] where [cond] holds and [`Invalid] elsewhere. *)

val contract : Ops.t -> Ops.t list -> Ops.t
(** [contract u rs] is the stack of [u] with the upcast ranges [rs] substituted
    by each of their values, the last range varying fastest.

    Raises [Invalid_argument] if a range of [rs] is not {!Ops.Axis_type.Upcast}. *)

(** {1:multi Several devices} *)

val axis : Ops.t -> int option
(** [axis u] is the axis [u] is sharded on, if it is sharded on one.

    Raises [Invalid_argument] if [u] is sharded on several axes, or if a reshape
    moves elements between shards. *)

val bounds : Ops.t -> (Ops.sint * Ops.sint) list
(** [bounds u] is the start and end of each shard along [u]'s axis.

    Raises [Invalid_argument] if [u] is not sharded. *)

val shard_shape : Ops.t -> Ops.sint list
(** [shard_shape u] is the shape of one of [u]'s shards. *)

val max_shard_shape : Ops.t -> int list
(** [max_shard_shape u] is [to_max_shape (shard_shape u)]. *)

val unshard : ?ranges:Ops.t list -> Ops.t -> int list -> Ops.t
(** [unshard ~ranges u axes] reassembles [u], sharded on [axes] over [ranges]
    (default one {!Ops.Axis_type.Device} range over [u]'s devices).

    Raises [Invalid_argument] if the lengths differ or an axis repeats. *)

val shard_slice : Ops.t -> int -> Ops.t -> Ops.t
(** [shard_slice u axis r] is the part of [u] along [axis] that the range [r]
    selects: the [n] elements of [axis] from [r * n], where [n] is the size of
    [axis] divided by [r]'s count. It is [u] if [u] is a scalar.

    Raises [Invalid_argument] if [r]'s count does not divide the axis. *)

val shard : ?axis:int -> Ops.t -> string list -> Ops.t
(** [shard ~axis u devices] is [u] copied to [devices], split along [axis] if
    given. *)

val store_call : Ops.t -> Ops.t -> Ops.t
(** [store_call dst src] is the call that copies [src] into [dst]. *)

(** {1:movement Movement} *)

val marg : Ops.t -> Ops.movement
(** [marg u] is the argument of the movement [u].

    Raises [Invalid_argument] if [u] is not a movement. *)

val as_shape : Ops.t -> Ops.sint list
(** [as_shape u] is the shape the node [u] denotes: its constant, its stack's
    sources, or [u] itself. *)

val mop : Ops.t -> Ops.movement -> Ops.t
(** [mop u m] applies the movement [m] to [u] as one node, its shape arguments
    stored as simplified sources. An empty expand is [u], and so is an empty pad
    or shrink of a scalar.

    Raises [Invalid_argument] on an empty pad or shrink of a node that is not a
    scalar, or where {!shape_opt} does. *)

val reshape : Ops.t -> Ops.sint list -> Ops.t
(** [reshape u shape] is [u] with shape [shape], or [u] if unchanged; a size of
    [-1] is inferred.

    Raises [Invalid_argument] if the element counts differ or [-1] appears
    twice. *)

val expand : Ops.t -> Ops.sint list -> Ops.t
(** [expand u shape] broadcasts [u] to [shape]; a size of [-1] keeps [u]'s. *)

val permute : Ops.t -> int list -> Ops.t
(** [permute u order] is [u] with its axes in [order], or [u]; negative axes
    count from the end.

    Raises [Invalid_argument] if [order] is not a permutation. *)

val flip : Ops.t -> int list -> Ops.t
(** [flip u axes] reverses [axes] of [u]. *)

val shrink : Ops.t -> (Ops.sint * Ops.sint) option list -> Ops.t
(** [shrink u bounds] keeps, on each axis, the elements from its start to its
    end, excluded; [None] keeps the axis whole. *)

val shrink_to : Ops.t -> Ops.sint option list -> Ops.t
(** [shrink_to u shape] keeps the first elements of each axis. *)

val pad : ?value:Dtype.const -> Ops.t -> (Ops.sint * Ops.sint) option list -> Ops.t
(** [pad ~value u padding] adds, on each axis, the given numbers of elements of
    [value] (default [0]) before and after it; a negative number removes
    elements. *)

val pad_to : ?value:Dtype.const -> Ops.t -> Ops.sint option list -> Ops.t
(** [pad_to ~value u shape] pads the end of each axis to [shape]. *)

val flatten : ?start:int -> ?stop:int -> Ops.t -> Ops.t
(** [flatten ~start ~stop u] merges the axes from [start] (default [0]) to
    [stop] (default [-1]), included. *)

val unflatten : Ops.t -> int -> Ops.sint list -> Ops.t
(** [unflatten u axis sizes] splits [axis] into [sizes]. *)

val squeeze : ?axis:int -> Ops.t -> Ops.t
(** [squeeze ~axis u] removes [axis] if its size is [1], or every axis of size
    [1]. *)

val unsqueeze : Ops.t -> int -> Ops.t
(** [unsqueeze u axis] is [u] with an axis of size [1] inserted at [axis], which
    counts from the end of the new shape when negative.

    Raises [Invalid_argument] if [axis] is out of range. *)

val transpose : Ops.t -> int -> int -> Ops.t
(** [transpose u a b] exchanges the axes [a] and [b] of [u].

    Raises [Invalid_argument] if an axis is out of range. *)

val split : ?axis:int -> Ops.t -> int list -> Ops.t list
(** [split ~axis u sizes] is the consecutive slices of [u] along [axis] (default
    [0]) of [sizes] elements.

    Raises [Invalid_argument] if [axis] is out of range or of symbolic size, or
    if [sizes] does not sum to its size. *)

val repeat : Ops.t -> int list -> Ops.t
(** [repeat u repeats] tiles [u] [repeats] times along each axis, the axes
    aligned to the right: a [repeats] longer than [u]'s shape adds leading axes.

    Raises [Invalid_argument] if a movement does. *)

val pool : ?stride:int list -> ?dilation:int list -> Ops.t -> int list -> Ops.t
(** [pool ~stride ~dilation u kernel] is the windows of [kernel] over the last
    axes of [u], each [stride] apart (default [1]) and [dilation] between its
    elements (default [1]): [u]'s leading axes, then the number of windows along
    each pooled axis, then the kernel's axes. Only movements build it: [u] is
    repeated and read back in rows one element longer, so that windows overlap
    without padding.

    Raises [Invalid_argument] if [u] has fewer axes than [kernel], if [stride]
    or [dilation] do not have one entry per kernel axis, or if a dilated kernel
    is longer than its axis. *)

val cat : ?axis:int -> Ops.t -> Ops.t list -> Ops.t
(** [cat ~axis u rest] concatenates [u :: rest] along [axis] (default [0]). *)

val cumalu : Ops.t -> int -> Op.t -> Ops.t
(** [cumalu u axis op] is the inclusive running [op] ({!Op.Add}, {!Op.Mul} or
    {!Op.Max}) of [u] along [axis], at [u]'s type: each element reduces the
    window of the elements up to it. Past 512 elements it runs in two stages,
    within chunks of 256 and across the chunks' last elements. It is [u] if an
    axis of [u] is empty.

    Raises [Invalid_argument] if [axis] has a symbolic size. *)

val arange : ?start:int -> ?step:int -> ?dtype:Dtype.t -> int -> Ops.t
(** [arange ~start ~step ~dtype stop] is the vector of the integers from [start]
    (default [0]) up to [stop], excluded, by [step] (default [1]), down to it
    for a negative [step], of type [dtype] (default the first of
    {!Dtype.default_int}, {!Dtype.Int32}, {!Dtype.Int64} and {!Dtype.Uint64}
    that holds them). It is empty if [stop] is not beyond [start]. Its elements
    are running sums of [step].

    Raises [Invalid_argument] if [step] is [0] or [dtype] does not hold the
    integers. *)

val nbytes : Ops.t -> int
(** [nbytes u] is the number of bytes of [u]'s elements. *)

(** {1:storage Storage}

    Sizes of storage are [int]s: a shape of more than [max_int] elements raises
    [Invalid_argument]. *)

val empty : ?device:Ops.device -> Ops.sint list -> Dtype.t -> Ops.t
(** [empty ~device shape dt] is uninitialised storage of [shape] for whoever
    realizes the graph to bind.

    Raises [Invalid_argument] if [dt] is weak. *)

val empty_like : ?dtype:Dtype.t -> ?device:Ops.device -> Ops.t -> Ops.t
(** [empty_like u] is uninitialised storage of [u]'s shape, type and placement,
    sharded as [u]. *)

val clone : ?device:Ops.device -> Ops.t -> Ops.t
(** [clone ~device u] is a copy of [u] into new storage on [device] (default
    [u]'s).

    Raises [Invalid_argument] if [device] is a disk. *)

val alloc :
  ?slot:int ->
  ?addrspace:Dtype.addr_space ->
  ?device:Ops.device ->
  ?axis:int ->
  Ops.sint list ->
  Dtype.t ->
  Ops.t
(** [alloc shape dt] is call-local storage of [shape], sharded on [axis]. *)

val alloc_like : ?slot:int -> ?addrspace:Dtype.addr_space -> Ops.t -> Ops.t
(** [alloc_like u] is [alloc] of [u]'s shard shape and type. *)

val placeholder :
  ?slot:int ->
  ?addrspace:Dtype.addr_space ->
  ?device:Ops.device ->
  ?volatile:bool ->
  ?tag:Ops.Tag.t ->
  int list ->
  Dtype.t ->
  Ops.t
(** [placeholder shape dt] is a parameter of [shape] and [dt] ([dt] committed,
    {!Dtype.strong}), or workgroup or register storage for those address spaces.
    A [String] tag also names it.

    Raises [Invalid_argument] if local storage gets a device. *)

val placeholder_like : ?addrspace:Dtype.addr_space -> Ops.t -> int -> Ops.t
(** [placeholder_like u slot] is a placeholder of [u]'s shard shape and type. *)

val param :
  ?shape:Ops.sint list ->
  ?device:Ops.device ->
  ?vmin_vmax:Dtype.value * Dtype.value ->
  ?multiple_of:int ->
  ?name:string ->
  ?addrspace:Dtype.addr_space option ->
  ?volatile:bool ->
  ?phase:int ->
  ?align:int ->
  int ->
  Dtype.t ->
  Ops.t
(** [param ~shape slot dt] is the parameter [slot] of type [dt]: a scalar
    without [shape], flat storage of its greatest size viewed as [shape]
    otherwise, whose first element lies [phase] bytes past a multiple of
    [align] (defaults [0] and [16]; see {!Ops.param_arg}).

    Raises [Invalid_argument] if [dt] is weak. *)

val param_like : Ops.t -> int -> Ops.t
(** [param_like u slot] is a parameter in [slot] that [u] can be passed to: a
    scalar variable without its name and value, one shard of a sharded value, or
    storage of [u]'s shape. Storage has the phase and alignment of the storage
    [u] views ({!storage_phase}).

    Raises [Invalid_argument] if the phase does not fit [u]'s type. *)

val storage_phase : Ops.t -> int * int
(** [storage_phase u] is the alignment and phase [(align, phase)]
    ({!Ops.param_arg}) of the storage [u] views: its storage's, moved by the bytes a
    shrink skips when it shrinks the storage seen whole, a buffer or a stage
    the schedule allocates, in row-major order. A shrink by a symbolic start
    known only to multiples of fewer bytes than the storage's alignment, such
    as a window that moves with a range, lowers the alignment to the largest
    power of two those bytes are a multiple of, and a symbolic start into a
    view that reorders or pads the storage keeps only the element's size. Any
    other view keeps its storage's: a reordering, a bitcast, an ordering or a
    shard selection. A stage of a view of a buffer through movements and
    bitcasts is the view when scheduling finds it contiguous, and storage of
    its own on a 16-byte boundary otherwise, so it has what both hold: phase
    [0] and the largest power of two up to the buffer's alignment that the
    view's first byte is a multiple of; [(1, 0)] when a size is symbolic or the
    buffer sharded. A view whose first or last element is padding, or whose
    last element does not lie as many elements past its first as a run of its
    size, is no run, and its stage has [(16, 0)]. Storage on a disk, which no vector access reads, and
    anything else that is not storage or a view of it have [(16, 0)]. *)

val view_as : ?axis:int -> Ops.t -> Ops.sint list -> Ops.t
(** [view_as ~axis u shape] views the flat storage [u] as [shape], sharded on
    [axis]. *)

(** {1:variables Variables} *)

val is_variable : Ops.t -> bool
(** [is_variable u] is [true] iff [u] is a scalar variable with a range. *)

val is_bound_var : Ops.t -> bool
(** [is_bound_var u] is [true] iff [u] is a variable bound to a value. *)

val bind : Ops.t -> Dtype.value -> Ops.t
(** [bind v x] is [v] bound to [x].

    Raises [Invalid_argument] if [v] is not an unbound variable, or [x] is out
    of its range or not a multiple of its [multiple_of]. *)

val unbound : Ops.t -> Ops.t
(** [unbound v] is [v] without its value and tag. *)

val unbind : Ops.t -> Ops.t * Dtype.value
(** [unbind v] is [(unbound v, x)] for [v] bound to [x].

    Raises [Invalid_argument] if [v] is not bound. *)

val unbind_all : Ops.t -> Ops.t * (Ops.t * Dtype.value) list
(** [unbind_all u] is [u] with each bound variable unbound, and each variable
    with its value. *)

val variables : Ops.t -> Ops.t list
(** [variables u] is the unbound scalar variables [u] reads, with a
    ["_device_num"] variable for each device range, sorted by name and slot. *)

(** {1:symbolic Divisibility} *)

val divides : Ops.t -> Bigint.t -> Ops.t option
(** [divides u n] is [u / n] if [u] is known to be a multiple of [n]. *)

val gcd : Ops.t list -> Ops.t
(** [gcd us] is a common divisor of [us]: their common factors times the
    greatest common divisor of their constant coefficients, a scalar constant
    whatever the shape of [us].

    Raises [Invalid_argument] if [us] is empty. *)

val divide_exact : Ops.t -> Ops.t -> Ops.t option
(** [divide_exact u d] is [u / d] if it divides exactly. *)

(** {1:calls Calls} *)

val call_with_outputs :
  ?name:string ->
  ?precompile:bool ->
  ?aux:Ops.hcq_info ->
  ?output_pos:int list ->
  Ops.t list ->
  Ops.t list ->
  Ops.t list
(** [call_with_outputs values args] calls a body that computes [values] from
    [args], and is the outputs, each ordered after the call. Each output is new
    storage passed to the call at its position in [output_pos] (default after
    [args]); [args] fill the other positions in order.

    Raises [Invalid_argument] if [output_pos] is not strictly ascending within
    the argument list. *)

val call_with_output : ?name:string -> ?precompile:bool -> Ops.t -> Ops.t list -> Ops.t
(** [call_with_output value args] is the one output of
    [call_with_outputs [value] args]. *)

val custom_kernel : Ops.t list -> (Ops.t list -> Ops.t) -> Ops.t list
(** [custom_kernel args f] calls the kernel [f] builds on placeholders of
    [args], and is [args], each ordered after it. *)

(** {1:programs Programs} *)

val program_info_of_sink : ?target:Helpers.Target.t -> Ops.t -> Ops.program_info
(** [program_info_of_sink ~target sink] reads the program's launch dimensions
    from its {!Op.Special} nodes, its variables and buffers from its {!Op.Param}
    nodes, and which buffers it reads and writes from its loads and stores. When
    none is found reading or writing, every buffer is taken to do both. [target]
    defaults to the empty target. *)

val launch_dims : Ops.program_info -> (string * int) list -> int list * int list
(** [launch_dims p vars] is [p]'s global and local sizes with each variable
    named in [vars] replaced by its value ({!sym_infer}). *)

(**/**)

(** Late bindings.

    The rules {!simplify} applies are defined by a later module of the library,
    which installs them here when the library is initialised, before any
    program runs. *)
module Private : sig
  val set_symbolic : (unit, Ops.t) Ops.Pattern_matcher.t -> unit
  (** [set_symbolic m] makes [m] the rules of {!simplify}.

      Raises [Invalid_argument] if they are set already. *)
end
