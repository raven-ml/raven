(** What kernel graphs write.

    A kernel graph holds storage and calls ({!Tolk_next.Op.Call}) of kernels,
    each a {!Tolk_next.Op.Sink} whose storage is its parameters. A call's
    argument names storage in a state: the storage as it is given, or, through
    an {!Tolk_next.Op.After}, as the calls it lists leave it. A kernel's storage
    parameter of slot [k] stands for the storage of the call's [k]th argument in
    the state that argument names, and a scalar parameter of slot [k] for the
    value a variable passed as that argument is bound to; the kernel writes what
    its stores write ({!Interpreter.writes}) into the storage of the arguments
    it stores into. So a kernel reads what its arguments name, and the order the
    kernels would run in is the scheduler's concern. *)

open Tolk_next

val writes :
  ?buffers:(int * Dtype.value array) list ->
  Ops.t ->
  (int * int * Dtype.value) list
(** [writes ~buffers sink] is what the kernel graph [sink] leaves in the storage
    it is given, its parameters and buffers, in the states the sink's sources
    name: each element [(s, i, v)] of memory [s] that a kernel wrote, with the
    last value [v] written, sorted by [s], then [i]. Memory [s] holds the array
    that [buffers] (default [[]]) gives [s]. Call-local storage
    ({!Tolk_next.Op.Alloc}) is memory of its own, apart from the numbering of
    the others, and scratch: its writes are left out. Memory that [buffers] does
    not give holds a value that no computation on small integers gives: NaN, the
    type's least integer, or [false].

    Raises [Invalid_argument] if a call's argument is neither storage on one
    device, possibly after effects, nor a bound variable, or as
    {!Interpreter.writes} does. *)
