(** Values of scalar graphs.

    A scalar graph is a node over constants, scalar parameters and leaves of
    index arithmetic (variables, ranges and hardware indices), built with the
    arithmetic operations, casts and bit reinterpretations. Its value is
    computed node by node, each operation rounding its result to its type
    ({!Tolk_next.Ops.exec_alu}), so that it is the value compiled code computes,
    rounding included.

    The value of an index where its gate fails ({!Tolk_next.Ops.valid}) is
    [`Invalid], which poisons what reads it: an operation of an [`Invalid]
    operand, and a selection by an [`Invalid] condition, is [`Invalid]. A
    selection picks its branch, [`Invalid] included. *)

open Tolk_next

val name : Ops.t -> string option
(** [name u] is the name that binds [u] in {!eval}: a variable's name, a
    hardware index's name, and [r] and the identity of a range
    ({!Tolk_next.Ops.range_str}), as [r0] or [r1_2]. It is [None] for any other
    node. *)

val eval :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  Ops.t ->
  Dtype.const
(** [eval ~vars ~params u] is the value of [u], each variable, range and
    hardware index replaced by the value that [vars] binds to its {!name}, a
    variable [vars] does not bind by the value it is bound to, and each other
    scalar parameter of slot [s] by the value that [params] gives [s] (both
    default [[]]). A weak integer operand of an operation on a committed integer
    type takes that type and wraps, as compiled code commits it. A cast converts
    as {!Dtype.const} then wraps to the target type ({!Dtype.truncate}); a bit
    reinterpretation is {!Dtype.bitcast}. Both branches of a selection are
    evaluated.

    Raises [Invalid_argument] if [u] reads an unbound variable, a range or a
    hardware index that [vars] does not bind, or a parameter that [params] does
    not give, or holds an operation other than a constant, a scalar leaf, an
    arithmetic operation, a cast or a bit reinterpretation. *)

val overflows :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  Ops.t ->
  bool
(** [overflows ~vars ~params u] is [true] iff an arithmetic operation, a cast or
    the commitment of a weak integer operand in [u], its leaves and parameters
    bound as {!eval} binds them, gives a committed integer type a value outside
    that type, which {!eval} then wraps. Where it is [false], {!eval} computes
    what exact integer arithmetic does.

    Raises [Invalid_argument] as {!eval} does. *)
