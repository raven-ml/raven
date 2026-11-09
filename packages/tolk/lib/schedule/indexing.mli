(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Ranges: from whole tensors to indexed elements.

    A tensor graph computes whole tensors: an operation reads every element of
    its sources. Scheduling needs the element each operation reads, as an index
    built from loop variables ({!Op.Range}). This module assigns to each node
    the ranges that index its elements, and rewrites the graph so that every
    read of a tensor names its element: movements become arithmetic on indices,
    and the values that must be stored whole are marked for storage. *)

val is_gather : Ops.t -> bool
(** [is_gather u] is whether [u] is a gather: an {!Op.Index} of a tensor by a
    value with axes, which reads the tensor at the row each element of the value
    names. *)

val apply_movement_op :
  Ops.sint list -> Ops.movement -> Ops.t list -> Ops.t list
(** [apply_movement_op in_shape m idxs] is the index, into a source of shape
    [in_shape], of the element that the movement [m] of that source places at
    the index [idxs]: one node per axis of the source, from one per axis of the
    result.

    - [Shrink] adds each axis' start;
    - [Permute] puts each index back on its source axis;
    - [Expand] drops the indices of the axes it adds in front;
    - [Pad] subtracts each padded axis' start, and the index is valid
      ({!Shape.valid}) only where it falls within the source;
    - [Reshape] flattens the index in row-major order and splits it along
      [in_shape], simplified with the symbolic rules and the validity rules
      ({!Shape.val-symbolic}, {!Symbolic.pm_simplify_valid}).

    The validity a [Pad] gives an index lasts only as long as the index
    expression carries it. A later movement whose simplification makes the index
    constant on an axis drops it: a [Reshape] to an axis of one element, an
    [Expand] that drops the leading index, a [Shrink] onto an axis of one
    element. A caller that reads through the index masks the padded values
    itself, as {!run_rangeify} turns each [Pad] into a selection. *)

val stack_select : Ops.t -> Ops.t list -> Ops.t
(** [stack_select i lanes] is the lane of [lanes] at the index [i], a tree of
    selections on the bits of [i], the most significant first: of two lanes,
    [where (i mod 2 < 1) l0 l1]. An [i] outside the lanes selects one of them.

    Raises [Invalid_argument] if [lanes] is empty. *)

val run_rangeify : Ops.t -> Ops.t
(** [run_rangeify sink] is the tensor graph [sink] with its elements
    indexed by ranges. Walking from [sink] to its leaves, each node gets the
    ranges that index its output:

    - a node that is stored whole gets new ranges, one per axis of size other
      than [1], where an axis of size [1] is indexed by [0] and an axis whose
      size is a range by that range;
    - so does every axis of a node whose consumers index it differently on some
      axis, validity aside;
    - so does every axis of a reduction or elementwise node below a broadcast,
      which would otherwise be recomputed for each element the broadcast adds. A
      node is below a broadcast when it reaches one through consumers not stored
      whole: an {!Op.Expand} whose sizes are not ranges, or an operation that
      broadcasts a source. An elementwise node that an operation broadcasts
      directly is not below it, a reduction is;
    - the index source of a gather, an {!Op.Index} whose index has axes, takes
      the gather's leading ranges, one per axis of the index;
    - any other node takes the ranges of its consumers, the validity of an index
      being the disjunction of theirs; a node without indexed consumers gets
      none.

    New ranges are numbered from [0] in the order the walk creates them, and
    are {!Ops.Axis_type.Weak}: a range is a reduction's when the reduction
    closes it. A node is stored whole when it is a {!Op.Store}, an argument
    of a call compiled on its own (a kernel given as code, or a precompiled
    call) that the call reads and that is neither storage, a state of it, a
    call's result, a load, a scalar of a precompiled call, nor a row of
    storage that loops ({!Ops.Axis_type.Loop}) move along (an {!Op.Shrink}
    whose elements on each trip are a run), or a gathered source that is not
    storage. A
    value stored into a destination it reads is not: within a kernel,
    {!Rangeify.get_kernel_graph} stages a value that
    reads its destination at other elements than it writes, and across
    kernels, {!Schedule.create_schedule} orders the reads before the write,
    or copies a state where no order does.

    The graph is then rewritten from the bottom up:

    - a source stored whole is read through an {!Op.Stage} of it over its
      ranges, indexed by its consumer's ranges; the stage lives on the source's
      device, and later passes may inline it back unless the source is storage
      or feeds a kernel given as code. A stored {!Op.Store} is instead closed by
      an {!Op.End} over its ranges;
    - storage read by an indexed node is indexed by that node's ranges;
    - a gather reads its gathered source at the element its index holds,
      followed by the gather's trailing ranges;
    - a reduction of leading axes becomes a reduction over their ranges,
      those of one element left out, and its own;
    - a {!Op.Pad} becomes a selection of its source where its index is valid,
      and of [0] elsewhere;
    - a {!Op.Stack} becomes a selection of its sources by its leading range:
      {!stack_select}, a tree of selections on the bits of the range, whose
      depth is logarithmic in the number of sources;
    - movements are removed;
    - a stage without a device gets the device of the nearest node above it
      that has one, where the kernel that reads it computes: read on several
      devices, it is a stage on each. One that no such node reads, or that
      only a store to a disk reads, gets [sink]'s.

    Calls and {!Op.After}s get no ranges, and the walk does not enter kernels
    ({!Ops.gate_kernel_sink}).

    Raises [Invalid_argument] if a reduction of leading axes gets no ranges, if
    a gather has more than one index, or if a call compiled on its own writes
    an argument ({!Shape.written_params}; each argument, where its body does
    not show its writes) that would be stored whole, since the call would
    write the copy, or reads one that is a constant, which needs no storage,
    or another value loops move along, which a store before the call cannot
    hold. An argument passed as it is must start where its parameter
    assumes ({!Shape.storage_phase}, {!Ops.type-param_arg}): one that does
    not is stored whole, and it raises where the call writes it, where loops
    move it, as rows 20 bytes apart for a parameter on a 16-byte boundary,
    and where the parameter assumes a start that new storage
    ({!Shape.allocated_start}) lacks. Storage the call also writes, only read
    off its parameter's start, is passed as it is: {!Schedule.create_schedule}
    reads it through a copy, made each trip in a loop. In a loop
    ({!Ops.Axis_type.Loop}), a read of the state the call writes, or a later
    one, that the call takes as neither storage nor a state of it, nor a row
    moving along it, raises unless it is a run of the storage itself
    ({!Schedule.create_linear_with_vars}): a store would be made once, before
    the loop, and hold no trip's writes. A read of an older state reads it as
    it was. *)

(**/**)

val store_checks : unit -> int
(** [store_checks ()] is the number of times {!run_rangeify} has checked
    whether two stores can share their loops, in every domain, since the
    program started. Two stores are checked at most once per pair of their
    groups, however many nodes they share. *)
