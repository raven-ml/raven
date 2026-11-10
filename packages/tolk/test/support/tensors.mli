(** Values of tensor graphs, and what they write.

    A tensor graph computes whole tensors of concrete shape ({!Tolk.Shape.shape}),
    stored row-major. Its value is computed node by node, as the operations
    define it:

    - storage ({!Tolk.Op.Param}, {!Tolk.Op.Buffer}) of slot [s] and size [n],
      [1] for a scalar, holds the elements [0] to [n - 1] of memory [s].
      Call-local storage ({!Tolk.Op.Alloc}) is memory of its own, apart from the
      numbering of the others, and holds [`Invalid] until it is written;
    - a bound variable is the value it is bound to;
    - a movement rearranges its source's elements: a reshape keeps their order,
      an expand broadcasts, a pad adds zeros, a shrink keeps a box, a permute
      and a flip reorder;
    - an arithmetic operation, a cast or a bit reinterpretation computes each
      element from its sources' elements, broadcast to its shape as numpy
      broadcasts, each element rounded to its type ({!Tolk.Ops.exec_alu});
      [`Invalid] poisons what reads it, and a selection picks its branch. A bit
      reinterpretation into a type of another size reads the bytes of each row
      of the last axis, little-endian, as the new type;
    - a reduction ({!Tolk.Op.Reduce}) of [k] axes folds its operation over the
      leading [k] axes of its source, from its identity, the last of them
      varying fastest;
    - a data {!Tolk.Op.Stack} stacks its sources along a new leading axis, and a
      {!Tolk.Op.Stage} or a {!Tolk.Op.Copy} copies its source to new storage.

    Graphs have effects on memory. A {!Tolk.Op.Store} writes its value,
    broadcast to its destination's shape, into the memory its destination
    views. Storage is read as the memory held before any store. An
    {!Tolk.Op.After} runs its first source, then its other sources, and holds
    the latest of the state its first source names, the states its sources'
    writes of its storage leave and the states of its storage they read, as
    memory was then; a join
    ({!Tolk.Ops.is_join}) holds the value its first source has. An effect runs
    after the state of each {!Tolk.Op.After} that lists it and whose storage
    it writes, unless that state lists it already. A {!Tolk.Op.Call} runs its
    body with each parameter of slot [k] standing for the call's [k]th
    argument, reshaped to the parameter's shape; an argument it writes is
    memory as it is once every argument has run. An
    {!Tolk.Op.End} of a call over a range runs the call once for each of the
    range's values, each trip reading the storage the loop writes, that its
    call writes ({!Tolk.Shape.written_params}) or, where its body does not
    show it, that of the {!Tolk.Op.After}s that list it, as the trips before
    left it, where the
    call is given the latest state of it, and an older state as it was. Each
    node runs once. *)

open Tolk

val eval :
  ?buffers:(int * Dtype.value array) list -> Ops.t -> Dtype.const array
(** [eval ~buffers u] is the value of [u]: its elements, row-major, where
    memory [s] holds the array that [buffers] (default [[]]) gives [s], and
    memory that [buffers] does not give holds [`Invalid].

    Raises [Invalid_argument] if [u] reads memory outside its array, has a
    shape that is not concrete, reads a variable, or holds an operation other
    than those above, a constant, and the operations that only order or name their first source
    ({!Tolk.Op.Detach}, {!Tolk.Op.Contiguous_backward}). *)

val writes :
  ?buffers:(int * Dtype.value array) list ->
  Ops.t ->
  (int * int * Dtype.value) list
(** [writes ~buffers u] is what running [u] leaves in memory, from the memory
    {!eval} starts from: each element [(s, i, v)] of memory [s], of parameters
    and buffers, that a store wrote, with the last value [v] it wrote, sorted by
    [s], then [i]. An element whose last value is [`Invalid] is left out, and so
    is call-local storage, which is scratch.

    Raises [Invalid_argument] as {!eval} does. *)
