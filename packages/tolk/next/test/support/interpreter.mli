(** Values of scalar graphs, and what kernels write.

    A scalar graph is a node over constants, scalar parameters and leaves of
    index arithmetic (variables, ranges and hardware indices), built with the
    arithmetic operations, casts and bit reinterpretations, reductions over
    ranges, and reads of storage. Its value is computed node by node, each
    operation rounding its result to its type ({!Tolk_next.Ops.exec_alu}), so
    that it is the value compiled code computes, rounding included.

    A reduction ({!Tolk_next.Op.Reduce}) of a kernel is its operation folded,
    from its identity ({!Tolk_next.Ops.identity_element}), over the value of its
    first source at each value of the ranges it reduces over: each range counts
    from [0] below its end, evaluated where the ranges before it are bound, and
    the last range varies fastest. An index ({!Tolk_next.Op.Index}) of storage
    by one integer reads that element of the storage, and a load
    ({!Tolk_next.Op.Load}) is the value of what it loads.

    The value of an index where its gate fails ({!Tolk_next.Ops.valid}) is
    [`Invalid], which poisons what reads it: an operation of an [`Invalid]
    operand, a read at an [`Invalid] index, a reduction of an [`Invalid] value,
    and a selection by an [`Invalid] condition, is [`Invalid]. A selection picks
    its branch, [`Invalid] included.

    A kernel is denoted by the set of its writes ({!writes}): which element of
    which storage takes which value. It has no memory state and no order: a
    rewrite of its loops keeps the kernel iff it keeps that set. *)

open Tolk_next

val name : Ops.t -> string option
(** [name u] is the name that binds [u] in {!eval}: a variable's name, a
    hardware index's name, and [r] and the identity of a range
    ({!Tolk_next.Ops.range_str}), as [r0] or [r1_2]. It is [None] for any other
    node. *)

val eval :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  ?buffers:(int * Dtype.value array) list ->
  Ops.t ->
  Dtype.const
(** [eval ~vars ~params ~buffers u] is the value of [u], each variable, range
    and hardware index replaced by the value that [vars] binds to its {!name}, a
    variable [vars] does not bind by the value it is bound to, each other scalar
    parameter of slot [s] by the value that [params] gives [s], and the element
    [i] of storage of slot [s] by the element [i] of the array that [buffers]
    gives [s] (all default [[]]). A reduction binds the ranges it reduces over,
    whatever [vars] binds them to. A weak integer operand of an operation on a
    committed integer type takes that type and wraps, as compiled code commits
    it. A cast converts as {!Dtype.const} then wraps to the target type
    ({!Dtype.truncate}); a bit reinterpretation is {!Dtype.bitcast}. Both
    branches of a selection are evaluated.

    Raises [Invalid_argument] if [u] reads an unbound variable, a range or a
    hardware index that [vars] does not bind, a parameter that [params] does not
    give, storage that [buffers] does not give or outside its array, or holds an
    operation other than a constant, a scalar leaf, an arithmetic operation, a
    cast, a bit reinterpretation, a reduction over ranges, an index of storage
    by one integer or a load. *)

val overflows :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  ?buffers:(int * Dtype.value array) list ->
  Ops.t ->
  bool
(** [overflows ~vars ~params ~buffers u] is [true] iff an arithmetic operation,
    a cast, a step of a reduction or the commitment of a weak integer operand in
    [u], its leaves, parameters and storage bound as {!eval} binds them, gives a
    committed integer type a value outside that type, which {!eval} then wraps.
    Where it is [false], {!eval} computes what exact integer arithmetic does.

    Raises [Invalid_argument] as {!eval} does. *)

(** {1:kernels Kernels} *)

val writes :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  ?buffers:(int * Dtype.value array) list ->
  Ops.t ->
  (int * int * Dtype.value) list
(** [writes ~vars ~params ~buffers u] is the set of writes of the stores
    ({!Tolk_next.Op.Store}) under [u], each [(slot, i, v)]: element [i] of the
    storage of slot [slot] takes the value [v]. The list is sorted by slot, then
    index, then value ({!Dtype.Value.compare}), and holds each write once.

    A store [Store (Index (storage, index), value)], or
    [Store (Index (storage, index), value, gate)], writes once for each binding
    of the ranges it runs inside ({!Tolk_next.Ops.ranges}): the ranges are taken
    in the order of the store's {!Tolk_next.Ops.toposort}, each counting from
    [0] below its end, evaluated where the ranges before it are bound, and are
    bound by {!name} on top of [vars]. At a binding, [index], [value] and [gate]
    are evaluated as {!eval} evaluates them, and the store writes
    [(slot, index, value)] where [index] is an integer, [value] is not
    [`Invalid] and [gate] is [true]; an [`Invalid] index or value, or a gate
    that fails, writes nothing. Ranges a store does not run inside, such as
    those only an {!Tolk_next.Op.End} lists, add no binding.

    Raises [Invalid_argument] as {!eval} does, or if a store's destination is
    not an index of storage by one integer. *)
