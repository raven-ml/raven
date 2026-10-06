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
    - [Flip] counts a reversed axis of size [s] from its end, [s - 1 - i];
    - [Expand] drops the indices of the axes it adds in front;
    - [Pad] subtracts each padded axis' start, and the index is valid
      ({!Shape.valid}) only where it falls within the source;
    - [Reshape] flattens the index in row-major order and splits it along
      [in_shape], simplified with the symbolic rules and the validity rules
      ({!Shape.symbolic}, {!Symbolic.pm_simplify_valid}).

    The validity a [Pad] gives an index lasts only as long as the index
    expression carries it. A later movement whose simplification makes the index
    constant on an axis drops it: a [Reshape] to an axis of one element, an
    [Expand] that drops the leading index, a [Shrink] onto an axis of one
    element. A caller that reads through the index masks the padded values
    itself, as {!run_rangeify} turns each [Pad] into a selection. *)

val run_rangeify : ?debug:bool -> Ops.t -> Ops.t
(** [run_rangeify ~debug sink] is the tensor graph [sink] with its elements
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

    New ranges are numbered from [0] in the order the walk creates them; they
    are {!Ops.Axis_type.Weak}, and those of the axes a reduction reduces are
    {!Ops.Axis_type.Reduce}. A node is stored whole when it is a {!Op.Store}, a
    source of an {!Op.Mselect} or an {!Op.Mstack} that is not storage, a source
    of a call to a kernel given as code that is not storage, a gathered source
    that is not storage, or a value stored into a destination it reads.

    The graph is then rewritten from the bottom up:

    - a source stored whole is read through an {!Op.Stage} of it over its
      ranges, indexed by its consumer's ranges; the stage lives on the source's
      device, and later passes may inline it back unless the source is storage
      or feeds a kernel given as code. A stored {!Op.Store} is instead closed by
      an {!Op.End} over its ranges;
    - storage read by an indexed node is indexed by that node's ranges;
    - a gather reads its gathered source at the element its index holds,
      followed by the gather's trailing ranges;
    - a reduction of leading axes becomes a reduction over its ranges;
    - a {!Op.Pad} becomes a selection of its source where its index is valid,
      and of [0] elsewhere;
    - a {!Op.Stack} becomes a selection of its sources by its leading range: a
      chain of comparisons with each index for at most [8] sources, and above
      that a comparison with the middle index choosing between the selections of
      each half, the whole guarded so that a negative index selects the last
      source. Its depth is logarithmic in the number of sources;
    - movements are removed;
    - a stage without a device gets [sink]'s.

    Calls, {!Op.After}s, {!Op.Mselect}s and {!Op.Mstack}s get no ranges, and the
    walk does not enter kernels ({!Ops.gate_kernel_sink}). With [debug] (default
    [false]), each node's ranges are printed on standard output as they are
    assigned.

    Raises [Invalid_argument] if a reduction of leading axes gets no ranges, or
    if a gather has more than one index. *)
