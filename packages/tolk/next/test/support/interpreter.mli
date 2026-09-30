(** Values of scalar graphs.

    A scalar graph is a node over constants, variables and scalar parameters,
    built with the arithmetic operations, casts and bit reinterpretations. Its
    value is computed node by node, each operation rounding its result to its
    type ({!Tolk_next.Ops.exec_alu}), so that it is the value compiled code
    computes, rounding included. *)

open Tolk_next

val eval :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  Ops.t ->
  Dtype.value
(** [eval ~vars ~params u] is the value of [u], each variable
    ({!Tolk_next.Ops.variable}) replaced by the value that [vars] binds to its
    name, and each other scalar parameter of slot [s] by the value that [params]
    gives [s] (both default [[]]). A cast converts as {!Dtype.const} then wraps
    to the target type ({!Dtype.truncate}); a bit reinterpretation is
    {!Dtype.bitcast}. Both branches of a selection are evaluated.

    Raises [Invalid_argument] if [u] reads a variable that [vars] does not bind
    or a parameter that [params] does not give, or holds an operation other than
    a constant, a scalar parameter, an arithmetic operation, a cast or a bit
    reinterpretation. *)
