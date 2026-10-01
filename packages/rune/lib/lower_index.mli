(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Assembly, indexed access and windows as UOps.

    Each function takes the nodes of an operation's operands, of the shapes and
    dtypes nx gives them, and is the node that computes its result as nx
    documents it. An element that an operation moves keeps its bits: [-0.] stays
    [-0.], and a NaN keeps its payload. Where windows overlap or updates meet,
    the elements are summed as nx sums: floats at [float32] or wider from [+0.],
    rounded once, so that a sum that is exactly zero is [+0.]; integers
    modularly; booleans as whether any holds. Updates that meet under a maximum
    or a minimum keep the bits of the operand that is the extreme, as nx.cpu
    does.

    Indices are [int64] nodes, as nx's are. A scatter compares them with the
    positions of an axis of at most [2{^ 31}] elements in [int32], after every
    index outside the axis is sent to [-1], so that no truncation brings one
    inside it. A window's starts lie within [x] and are cast exactly. *)

open Tolk

val pad : (int * int) array -> Dtype.const -> Ops.t -> Ops.t
(** [pad padding fill x] is [x] with [fst padding.(i)] elements [fill] before it
    and [snd padding.(i)] after it along each axis [i]. *)

val cat : int -> Ops.t -> Ops.t list -> Ops.t
(** [cat axis x xs] is the nodes [x :: xs] one after the other along [axis].
    They have one shape but along [axis]. *)

val gather : int -> Ops.t -> Ops.t -> Ops.t
(** [gather axis indices x] is, at each position of the index node [indices],
    [x]'s element at that position with its [axis] component replaced by the
    index there. An index outside \[[0], [n]), [n] being [x]'s size along
    [axis], reads zero. *)

(** {1:regions Regions}

    A write into a value that a program consumes can be stored in place, over
    only the elements it writes. A region is a view of the written value,
    through movements and gathers, and the value of the write there: storing
    each region's value into its view of the written value's storage leaves
    that storage holding the write's result. *)

type region = {
  dest : Ops.t;  (** The view of the written value it stores into. *)
  value : Ops.t;  (** The value stored into it. *)
}

val scatter :
  mode:Nx_backend.scatter ->
  unique:bool ->
  axis:int ->
  indices:Ops.t ->
  updates:Ops.t ->
  Ops.t ->
  Ops.t * region list
(** [scatter ~mode ~unique ~axis ~indices ~updates x] is [x] with each element
    of [updates] combined by [mode] into the element at the position of the
    index node [indices] at the same index along [axis], and its regions. A
    position that no update reaches is [x]'s element.
    - Under [`Set] the last of duplicate positions in row-major order wins, and
      with [unique], which asserts that the positions are distinct, one of them
      does.
    - Under [`Add] the element and its updates are summed.
    - Under [`Max] and [`Min] the position takes the bits of the first of its
      element and its updates, in that order, that is the extreme: a NaN result
      is the element's or the first NaN update's, with its payload.

    An update at an index outside \[[0], [n]), [n] being [x]'s size along
    [axis], is dropped.

    The value combines, at each position of [x], the updates that reach it:
    with [k] updates along [axis], an [n x k] comparison, which fuses with what
    reads the value. When [unique] or [k <= n], and shapes are static, the one
    region is an indexed store of the updates, which computes fewer values in
    place: each update's value combines every update at its index, a [k x k]
    comparison without [unique], and the last of them stores it; a dropped
    update's index is Invalid. Otherwise there is no region. *)

val update : Ops.t -> starts:Ops.t -> Ops.t -> Ops.t
(** [update x ~starts v] is [x] with [v] at the window whose corner is the index
    vector [starts] and whose extent is [v]'s shape, a window within [x]. *)

val update_region : Ops.t -> starts:Ops.t -> Ops.t -> region option
(** [update_region x ~starts v] is the one region of [update x ~starts v], a
    window of [x], or [None] if a size is symbolic or [x] is split along an axis
    the window narrows. *)

val unfold :
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  Ops.t ->
  Ops.t
(** [unfold ~kernel_size ~stride ~dilation ~padding x] is the windows over the
    last [k] axes of [x], [k] being the length of [kernel_size], zero-padded by
    [padding]: of shape [(leading..., product kernel_size, l)], [l] being the
    number of windows, the kernel's positions and the windows each in row-major
    order. *)

val fold :
  output_size:int array ->
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  Ops.t ->
  Ops.t
(** [fold ~output_size ~kernel_size ~stride ~dilation ~padding x] is the windows
    [x], of shape [(leading..., product kernel_size, l)], put back in place into
    a node of shape [(leading..., output_size...)], summed where they overlap:
    the transpose of the {!unfold} with these parameters. *)
