(** What kernel graphs write.

    A kernel graph holds storage and calls ({!Tolk_next.Op.Call}) of kernels,
    each a {!Tolk_next.Op.Sink} whose storage is its parameters. A call's
    argument names storage in a state: the storage as it is given, or, through
    an {!Tolk_next.Op.After}, as the calls it lists leave it. A kernel's storage
    parameter of slot [k] stands for the storage of the call's [k]th argument in
    the state that argument names, and a scalar parameter of slot [k] for the
    value a variable passed as that argument is bound to. A graph that is the
    body of a call reads a scalar argument of that call as a scalar parameter
    its kernels hold without passing it. The kernel writes what its stores write
    ({!Interpreter.writes}) into the storage of the arguments it stores into. So
    a kernel reads what its arguments name, and the order the kernels would run
    in is the scheduler's concern. *)

open Tolk_next

val writes :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  ?buffers:(int * Dtype.value array) list ->
  Ops.t ->
  (int * int * Dtype.value) list
(** [writes ~vars ~params ~buffers sink] is what the kernel graph [sink] leaves
    in the storage it is given, its parameters and buffers, in the states the
    sink's sources name: each element [(s, i, v)] of memory [s] that a kernel
    wrote, with the last value [v] written, sorted by [s], then [i]. Memory [s]
    holds the array that [buffers] (default [[]]) gives [s], a variable passed
    unbound takes the value that [vars] (default [[]]) gives its name, and a
    scalar parameter of slot [k] that no call passes the value that [params]
    (default [[]]) gives [k]. Call-local storage ({!Tolk_next.Op.Alloc}) is
    memory of its own, apart from the numbering of the others, and scratch: its
    writes are left out. Memory that [buffers] does not give holds a value that
    no computation on small integers gives: NaN, the type's least integer, or
    [false].

    Raises [Invalid_argument] if a call's argument is neither storage on one
    device, possibly after effects, nor a variable with a value, or as
    {!Interpreter.writes} does. *)

val linear_writes :
  ?vars:(string * Dtype.value) list ->
  ?params:(int * Dtype.value) list ->
  ?buffers:(int * Dtype.value array) list ->
  Ops.t ->
  (int * int * Dtype.value) list
(** [linear_writes ~vars ~params ~buffers linear] is what running the calls of
    the schedule [linear], an {!Tolk_next.Op.Linear}, in order leaves in the
    storage it is given, as {!writes} states it: each kernel reads its
    arguments' storage as the calls before it left it.

    Raises [Invalid_argument] if a call is not of a kernel, or as {!writes}
    does. *)
