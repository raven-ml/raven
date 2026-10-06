(** Values of tensor graphs, and what they write.

    A tensor graph computes whole tensors of concrete shape ({!Tolk.Shape.shape}),
    stored row-major. Its value is computed node by node, as the operations
    define it:

    - storage ({!Tolk.Op.Param}, {!Tolk.Op.Buffer}) of slot [s] and size [n],
      [1] for a scalar, holds the elements [0] to [n - 1] of memory [s]; on
      several devices, device [k] holds the elements [k * n] to [k * n + n - 1].
      Call-local storage ({!Tolk.Op.Alloc}) is memory of its own, apart from the
      numbering of the others, and holds [`Invalid] until it is written;
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
      {!Tolk.Op.Stage} copies its source to new storage;
    - on several devices a value has an element array per device: a
      {!Tolk.Op.Mselect} is one device's, a {!Tolk.Op.Mstack} gathers one
      device's value from each source, a {!Tolk.Op.Copy} places its source, or
      the first device's value of a source on several devices, on each of its
      target's devices, an operation combines device [k] of its sources on
      device [k], a value on one device standing for every device, and an
      {!Tolk.Op.Allreduce} folds its operation over its source's devices,
      element by element, and places the result on each of its target's devices;
    - a movement whose argument holds a device range
      ({!Tolk.Ops.Axis_type.Device}) moves device [k]'s value with the range
      standing for [k];
    - an {!Tolk.Op.Unshard} is the whole value that its source's devices hold
      the parts of: along each sharded axis, device [k] holds the part at the
      position that the axis's range takes on [k].

    Graphs have effects on memory. A {!Tolk.Op.Store} writes its value,
    broadcast to its destination's shape, into the memory its destination views,
    each element on device [k] taking device [k]'s value. Storage is read as the
    memory held before any store; an {!Tolk.Op.After} reads its first source
    once its other sources have run. A {!Tolk.Op.Call} runs its body with each
    parameter of slot [k] standing for the call's [k]th argument, reshaped to
    the parameter's shape, or for its parts if it is sharded. Each node runs
    once. *)

open Tolk

val eval :
  ?buffers:(int * Dtype.value array) list -> Ops.t -> Dtype.const array list
(** [eval ~buffers u] is the value of [u]: its elements, row-major, one array
    per device, where memory [s] holds the array that [buffers] (default [[]])
    gives [s], and memory that [buffers] does not give holds [`Invalid].

    Raises [Invalid_argument] if [u] reads memory outside its array, has a shape
    that is not concrete, reads a variable, shards a value over a range that is
    not a device range, or holds an operation other than those above, a
    constant, and the operations that only order or name their first source
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
