(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Shapes and simplification.

    The functions on nodes ({!Ops.t}) that need {!simplify}: a symbolic size is
    compared, bounded and computed by simplifying the node that holds it. Shapes
    ({!shape}), the movements that change them, storage of a shape, sharding,
    variables, divisibility and calls all read sizes, and so does the
    simplification itself, whose rules fold constants into the shapes of the
    nodes they replace.

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
(** [simplify u] is [u] rewritten with {!symbolic} to a fixed point. A constant
    is itself, and so is a sink of constants and stacks of constants, which the
    rules leave as they are. *)

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
  (** [resolve ~default c] is [c]'s value if known, and {!resolve}'s otherwise.
  *)

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
    computes on each run, such as the offset of a view that moves with a range.

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

    Raises [Invalid_argument] if a range of [rs] is not {!Ops.Axis_type.Upcast}.
*)

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

val pad :
  ?value:Dtype.const -> Ops.t -> (Ops.sint * Ops.sint) option list -> Ops.t
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
    otherwise, whose first element lies [phase] bytes past a multiple of [align]
    (defaults [0] and [16]; see {!Ops.param_arg}).

    Raises [Invalid_argument] if [dt] is weak. *)

val param_like : Ops.t -> int -> Ops.t
(** [param_like u slot] is a parameter in [slot] that [u] can be passed to: a
    scalar variable without its name and value, one shard of a sharded value, or
    storage of [u]'s shape. Storage has the phase and alignment of the storage
    [u] views ({!storage_phase}).

    Raises [Invalid_argument] if the phase does not fit [u]'s type. *)

val storage_phase : Ops.t -> int * int
(** [storage_phase u] is the alignment and phase [(align, phase)]
    ({!Ops.param_arg}) of the storage [u] views: its storage's, moved by the
    bytes a shrink skips when it shrinks the storage seen whole, a buffer or a
    stage the schedule allocates, in row-major order. A shrink by a symbolic
    start known only to multiples of fewer bytes than the storage's alignment,
    such as a window that moves with a range, lowers the alignment to the
    largest power of two those bytes are a multiple of, and a symbolic start
    into a view that reorders or pads the storage keeps only the element's size.
    Any other view keeps its storage's: a reordering, a bitcast, an ordering or
    a shard selection. A stage of a view of a buffer through movements and
    bitcasts is the view when scheduling finds it contiguous, and storage of its
    own on a 16-byte boundary otherwise, so it has what both hold: phase [0] and
    the largest power of two up to the buffer's alignment that the view's first
    byte is a multiple of; [(1, 0)] when a size is symbolic or the buffer
    sharded. A view whose first or last element is padding, or whose last
    element does not lie as many elements past its first as a run of its size,
    is no run, and its stage has [(16, 0)]. Storage on a disk, which no vector
    access reads, and anything else that is not storage or a view of it have
    [(16, 0)]. *)

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

val call_with_output :
  ?name:string -> ?precompile:bool -> Ops.t -> Ops.t list -> Ops.t
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

(** {1:rules Simplification rules}

    The rewrites of {!simplify}: they replace a graph by a simpler one with the
    same values, by constant folding, algebraic identities, reasoning on the
    bounds of integers ({!Ops.vmin}, {!Ops.vmax}) and on the conditions that
    guard them. {!symbolic_simple} folds one node at a time and {!symbolic}
    matches deeper and canonicalises index arithmetic; {!Symbolic.sym} adds the
    rewrites code generation relies on.

    {b Values.} Every rewrite keeps each value bit for bit, IEEE's signed zeros,
    infinities, NaN and subnormals included, and wraps integers at their type,
    as compiled code computes them. The exception is the rounding of powers: a
    constant power is computed by products and square roots, and [c ** x] for a
    positive finite constant [c] as [exp2 (x * log2 c)]. So the rewrites that
    reassociate or distribute arithmetic, or rely on it never overflowing, apply
    to integers only, and to a committed integer only where no value it computes
    wraps ({!Ops.exact}); a weak integer, such as an index, never wraps.

    {b Invalid values.} An index that is {!Ops.invalid} where a condition fails,
    [where cond x invalid], is a {e gated} value ({!invalid_gate}). The rewrites
    keep the gate outermost, so that it reaches the load or store that reads the
    index, which then does nothing where the gate fails. *)

val invalid_gate : Ops.Upat.t
(** [invalid_gate] matches [where cond x i] with [i] the constant
    {!Ops.invalid}, naming its sources ["cond"], ["x"] and ["i"]. *)

val pm_remove_invalid : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_remove_invalid] replaces each {!Ops.invalid} that a gate or a stack
    holds by [0] of the holder's type, once the gates have been read. *)

val symbolic_simple : (unit, Ops.t) Ops.Pattern_matcher.t
(** [symbolic_simple] folds nodes whose value follows from their operation and
    their sources alone:

    - {b invalid values}: an operation on a gated value moves inside the gate;
      an operation on {!Ops.invalid} is invalid; a gate on a reduction's operand
      that does not depend on its ranges moves out of the reduction; a store to
      an invalid index does nothing, and a load from one is its alternative
      value, or [0];
    - {b identities}: [x + 0], [x lxor 0], [x lor 0], [x lsl 0], [x lsr 0],
      [x * 1] and [x // 1] are [x], a float [x + 0] only for [-0.]; [x // -1] is
      [-x]; [x // x] is [1]; [(x lxor y) lxor y] is [x]; [(x % y) % y] is
      [x % y]; a boolean [x land c] and [x lor c] with [c] constant are [x] or
      [c]; [x <> false], a double negation, [where x true false] and an
      idempotent operation of [x] with itself are [x]; [where x false true] is
      the negation of [x]; a boolean cast to an integer and compared to [0] or
      [1] is the boolean or its negation, and to any other integer [true]; the
      truncation of an integer is itself;
    - {b recombination}: in a weak integer sum, [(b % d) * m] and a term
      [q * (d * m)], where [q] is [b' // d] for some [b'] congruent to [b]
      modulo [d] up to constants, recombine into [b' * m]; where [q] is
      [(b' // d) % k] with [k] positive, into [(b' % (d * k)) * m];
    - {b zeros}: [x < x] is [false], [x <> x] is [false] for integers and
      booleans, [x % x], [x lxor x] and [x land 0] are [0], and so is [x * 0]
      for integers and booleans; a mask that clears only bits that a right shift
      or a division by a power of two drops is removed;
    - {b constants}: an arithmetic operation on constants is its value
      ({!Ops.exec_alu}), except {!Op.Threefry}; weak constants keep their
      mathematical value, a committed integer holds its type's value (a weak
      operand of an operation on one is read, and the result written, at its
      width), and an operation mixing weak and committed constants commits the
      weak ones to the promoted type; a cast of a constant is the constant of
      the cast's type; [0 / 0] is NaN;
    - {b booleans}: a boolean [*] is [land], and a boolean [+] or maximum is
      [lor];
    - {b casts}: a cast or bitcast to its operand's type is its operand; a
      bitcast of a constant is the constant with the same bits; a cast through a
      type that holds every value of the result's type, back to that type, is
      its operand; two bitcasts are one; a cast to a boolean is [x <> 0];
    - {b powers}: [x ** c], for a constant [c] that is an integer or a half
      integer, [0] or at least [1] in magnitude, is a product of powers of [x],
      its reciprocal and its square root, a float half-integer power being [+0.]
      at [-0.] and [+inf] at [-inf]; [c ** x] is [c] if [c = 1], and
      [exp2 (x * log2 c)] for positive finite [c]; a 64-bit integer packed from
      two 32-bit halves and unpacked again is the half read;
    - {b selections}: a selection between equal values is that value, and a
      selection by a constant is the branch it picks, keeping the selection's
      type;

    then cleans up movements ({!mop_cleanup}). *)

val commutative : (unit, Ops.t) Ops.Pattern_matcher.t
(** [commutative] orders the two operands of each commutative weak integer
    operation by {!Ops.compare_structure}, least first, so that two sums of the
    same terms become the same node. *)

val symbolic : (unit, Ops.t) Ops.Pattern_matcher.t
(** [symbolic] is {!symbolic_simple} and {!commutative}, followed by rewrites
    that match deeper:

    - {b terms}: [x lor not x] is [true]; [x + x] is [x * 2]; for integers, like
      terms combine, [x * c0 + x * c1] into [x * (c0 + c1)] and [y + x + x] into
      [y + x * 2], also as the last two terms of a longer sum, and [-(x + c)] is
      [-x + -c]; [c * (x + c')] is [c * x + c * c'] for a weak integer [x];
    - {b selections}: a selection by a negation swaps its branches; within
      [where c t f], [c] is [true] in [t] and [false] in [f], unless an
      {!Op.Index} is involved; [where g x 0 <> 0] is [g land (x <> 0)]; nested
      selections sharing a branch merge their conditions with [land] or [lor];
      an operation on two selections by the same condition, one of whose branch
      pairs is constant, selects between the operations on the branches, also as
      the last two terms of an integer sum; for integers,
      [where c t 0 + where c 0 f] is [where c t f];
    - {b bounds}: a comparison, division, remainder, variable, {!Op.After},
      {!Op.Special} or range with a constant end whose bounds are equal is that
      constant; an integer maximum of two operands whose bounds, as it reads
      them ({!Ops.operand_bounds}), do not overlap is the greater, committed to
      its type; an integer selection that computes a maximum,
      [where (a < b) b a] with [a] or [b] a constant, is {!Ops.maximum};
    - {b constants}: two applications of an associative operation to constants
      fold the constants together, sums, products and maxima for integers only;
      [(x // c1) // c2] is [x // (c1 * c2)] for positive [c2] where [c1 * c2]
      does not wrap ({!Ops.exact}); constants move to the end of integer sums
      and products;
    - {b comparisons}, on integers: [c0 + x < c1] is [x < c1 - c0] where neither
      side wraps; [c0 * x < c1] divides both sides by [c0], rounding up, and
      flips [x]'s sign if [c0] is negative; [x // d < c] is [x < c * d] for
      positive [d] and [c * d < x] for negative [d]; in [x < c] with [c]
      positive, a divisor [d] common to [c] and the coefficients of some terms
      of [x], whose other terms stay within [0] and [d - 1], divides out;
      [-x < -y] is [y < x]; [not (x < 1)] with [x] a sum of non-negative terms
      with positive coefficients drops the coefficients;
    - {b ranges}: a range modulo its end is the range, and divided by its end is
      [0];
    - {b casts}: a cast through a type that holds every value of the operand is
      one cast; a cast of an integer that fits the intermediate integer type is
      one cast; a cast of an unsigned [x land y], or [x lsr k] with [k] below
      [x]'s width, to a wider unsigned type is the operation on the widened
      operands; a binary operation on 64-bit or weak integers whose values fit
      32 bits computes in {!Dtype.Int32} and casts back; a cast of [x + c] to a
      signed integer is the cast of [x] plus [c];
    - {b ordering}: an {!Op.After} waits only on ranges, stores, calls,
      barriers, ends, backedges, linear programs and stages, and on the sources
      of any other node it names; an {!Op.After} or {!Op.End} of nothing is its
      value, and an {!Op.End} drops the ranges that became constants;

    then simplifies division and remainder ({!div_and_mod_symbolic}), and
    restores bare constant operands ({!pm_uncast_const}). *)

(** {2:divmod Division and remainder} *)

val div_and_mod_symbolic : (unit, Ops.t) Ops.Pattern_matcher.t
(** [div_and_mod_symbolic] rewrites divisions and remainders. With [x] and [y]
    nodes and [a], [c] and [d] constants:

    - [(x // c + a) // d] is [(x + a * c) // (c * d)] when [d] is positive and,
      for a committed integer [x], no value wraps ({!Ops.exact});
    - for a weak integer [x] and [c % d] other than [c], [(x + c) // d] is
      [(x + c % d) // d + c // d], and [(x + c) % d] is [(x + c % d) % d].

    Any other weak integer [x // y] or [x % y] takes the first of the following
    forms that applies. When the quotient has one possible value [q], [x // y]
    is [q] and [x % y] is [x - q * y]. When [x] is a parameter declared a
    multiple of [m], and [m] is a multiple of the constant [y], [x % y] is [0]
    and [x // y] stays as it is.

    When [y] is a positive constant [c], with [x] a sum of terms and a constant:

    - [(z % (k * c)) // c] is [(z // c) % k] for a positive [k];
    - in [x % c], a term [z % m], with [m] a multiple of [c], is [z];
    - when replacing each term's factor by one of its residues modulo [c] leaves
      a sum within one period of [c], that sum gives the remainder, and the
      factors less their residues give the quotient;
    - a divisor common to [c] and every term's factor is divided out of both,
      when the quotient stays non-negative;
    - for each factor [f] of a term that divides [c], [x // c] is
      [(x // f) // (c / f)], and [x % c] its reconstruction, keeping the
      smallest result.

    Otherwise:

    - a common divisor of [y] and of the terms of [x] is divided out of both;
    - for [x] and [y] non-negative, the terms of [x] that are multiples of [y]
      leave the division.

    Raises [Division_by_zero] if a weak integer division or remainder has a
    divisor that is always [0], or a positive constant divisor and a dividend
    with a term that is the constant [0] or a product by it. The symbolic rules
    fold such a term away before these rules see it. *)

(** {2:values Values of constants}

    A rule that computes with the value of {!Ops.invalid}, which is no number,
    does not apply: it reads each constant with {!number}, and declines through
    {!rule}. The rules of {!Symbolic} share them. *)

exception Not_a_number
(** The exception {!number} raises on [`Invalid]. *)

val number : Dtype.const -> Dtype.value
(** [number c] is the value of [c].

    Raises {!Not_a_number} if [c] is [`Invalid]. *)

val rule :
  Ops.Upat.t ->
  ((string -> Ops.t) -> Ops.t option) ->
  ('ctx, Ops.t) Ops.Pattern_matcher.rule
(** [rule p f] is [Ops.Pattern_matcher.rule p f], which declines where [f]
    raises {!Not_a_number}. *)

(** {2:movements Movements} *)

val mop_cleanup : (unit, Ops.t) Ops.Pattern_matcher.t
(** [mop_cleanup] merges and removes movements and indexing:

    - a shrink of a shrink is one shrink of the inner one's source, starting at
      the sum of their starts, of the outer one's sizes;
    - a reshape of a reshape is one reshape of the inner one's source, and a
      reshape to its source's shape is its source;
    - a permutation of a permutation is their composition, and the identity
      permutation is its source;
    - a stack of the elements [0], [1], ... of a node [x], in order and of [x]'s
      shape, is [x];
    - indexing a stack by a constant [i] and further indices is its [i]th source
      indexed by those indices;
    - indexing an index, when every index of both is a scalar, is one index by
      the inner indices followed by the outer ones;
    - indexing the storage [b] indexed by [i] with as many indices as [i] has
      axes is indexing [b] by the element of [i] they select. *)

(** {2:weak Weak constants} *)

val pm_uncast_const : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_uncast_const] removes the cast from each committed constant source of a
    broadcastable node, leaving the bare weak literal, when the node's sources
    still derive the same least upper type and the node the same type. Rules
    keyed on constant values then match the literal. *)
