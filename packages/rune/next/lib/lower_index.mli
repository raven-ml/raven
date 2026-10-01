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
    modularly; booleans as whether any holds.

    Indices are [int64] nodes, as nx's are. A scatter compares them with the
    positions of an axis of at most [2{^ 31}] elements in [int32], after every
    index outside the axis is sent to [-1], so that no truncation brings one
    inside it. A window's starts lie within [x] and are cast exactly. *)

open Tolk_next

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

val scatter :
  mode:[ `Set | `Add ] ->
  unique:bool ->
  axis:int ->
  indices:Ops.t ->
  updates:Ops.t ->
  Ops.t ->
  Ops.t
(** [scatter ~mode ~unique ~axis ~indices ~updates x] is [x] with each element
    of [updates] set ([`Set]) or added ([`Add]) at the position of the index
    node [indices] at the same index along [axis]. A position that no update
    reaches is [x]'s element. Under [`Set] the last of duplicate positions in
    row-major order wins, and with [unique], which asserts that the positions
    are distinct, one of them does. An update at an index outside \[[0], [n]),
    [n] being [x]'s size along [axis], is dropped. *)

val update : Ops.t -> starts:Ops.t -> Ops.t -> Ops.t
(** [update x ~starts v] is [x] with [v] at the window whose corner is the index
    vector [starts] and whose extent is [v]'s shape, a window within [x]. *)

(** {1:regions Regions}

    A write into a value that a program consumes can be stored in place, over
    only the elements it writes. A region is a window of the written value, of
    static extent at an offset that may be a node, and the value of the write
    there: storing each region's value into its window of the written value's
    storage leaves that storage holding the write's result. A region with a
    [padding] is a window of the written value padded first: what it stores into
    the padding is dropped. *)

type region = {
  padding : (Ops.sint * Ops.sint) option list option;
      (** The padding, before and after each axis, of the value its window is
          of, if any. *)
  bounds : (Ops.sint * Ops.sint) option list;
      (** The window: the start and end of each axis it narrows. *)
  value : Ops.t;  (** The value stored into the window. *)
}

val scatter_rows :
  axis:int -> indices:Ops.t -> updates:Ops.t -> Ops.t -> region list option
(** [scatter_rows ~axis ~indices ~updates x] is the regions of
    [scatter ~mode:`Set ~unique:true ~axis ~indices ~updates x], one per row of
    [updates] along [axis], when [indices] is one index per row broadcast along
    every other axis, [updates] spans [x] along those axes, and there are at
    most 16 rows: each row is its own store. A window is on one device, so [x]
    split along [axis] has none. Each is a window of [x] padded by one row after
    [axis], at the row's index where it lies within [x] and at the padding row
    otherwise, so that the row is dropped. [None] otherwise. *)

val update_region : Ops.t -> starts:Ops.t -> Ops.t -> region option
(** [update_region x ~starts v] is the one region of [update x ~starts v], or
    [None] if a size is symbolic or [x] is split along an axis the window
    narrows. *)

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
