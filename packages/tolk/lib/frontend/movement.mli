(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Shape movement.

    These operations rearrange or resize a tensor's axes without touching the
    stored values: they build views. Every one returns its input unchanged when
    the requested shape equals the current shape.

    Axis arguments accept negative indices, counted from the last axis. Shape
    arguments on the plain entry points are concrete integers; the [symbolic_*]
    entry points take dimension nodes instead and accept symbolic values.
    {!reshape}, {!expand}, {!broadcast_to}, {!pad}, {!squeeze}, {!unsqueeze},
    {!flatten}, {!transpose}, {!permute}, and {!flip} also operate on tensors
    whose current shape is symbolic. The remaining operations ({!pool},
    {!unfold}, {!split}, {!repeat}, {!unflatten}, and strided indexing) require
    concrete shapes and raise [Invalid_argument] on a symbolic dimension. *)

val reshape : Tensor.t -> int list -> Tensor.t
(** [reshape t dims] returns a view of [t] with shape [dims], preserving the
    element order and total count. At most one dimension may be [-1], whose
    size is inferred so that the total count is preserved.

    @raise Invalid_argument
      if more than one [-1] is given, or if the element counts do not match. *)

val symbolic_reshape : Tensor.t -> Tolk_uop.Uop.t list -> Tensor.t
(** [symbolic_reshape t dims] is {!reshape} over dimension nodes, with no [-1]
    inference. The element count must be preserved provably.

    @raise Invalid_argument if the element counts cannot be shown equal. *)

val expand : Tensor.t -> int list -> Tensor.t
(** [expand t dims] broadcasts [t] to shape [dims]. A [-1] entry keeps the
    corresponding input size. [t] is first left-aligned with [1]s, so [expand]
    may add leading axes. Each broadcast axis must have input size [1].

    @raise Invalid_argument if a non-[1] axis is asked to change size. *)

val broadcast_to : Tensor.t -> int list -> Tensor.t
(** [broadcast_to t dims] is [expand] to an explicit target shape, with no
    [-1] handling: [t] is left-aligned with [1]s and every axis must already
    match [dims] or have size [1]. *)

val symbolic_broadcast_to : Tensor.t -> Tolk_uop.Uop.t list -> Tensor.t
(** [symbolic_broadcast_to t dims] is {!broadcast_to} over dimension nodes. A
    symbolic axis matches when it carries the same expression in [t] and
    [dims], or when [t] has size [1] there. *)

val permute : Tensor.t -> int list -> Tensor.t
(** [permute t order] reorders the axes of [t] so that output axis [k] is
    input axis [List.nth order k]. [order] must be a permutation of the axes.

    @raise Invalid_argument if [order] is not a permutation. *)

val flip : Tensor.t -> int list -> Tensor.t
(** [flip t axes] reverses [t] along each axis in [axes].

    @raise Invalid_argument if an axis is repeated. *)

val pad : Tensor.t -> (int * int) list -> Tensor.t
(** [pad t padding] pads [t] with zeros. [padding] gives, per axis in order,
    the number of elements to add before and after. Every axis must be listed.

    @raise Invalid_argument if [padding] length differs from [ndim t]. *)

val shrink : Tensor.t -> (int * int) list -> Tensor.t
(** [shrink t bounds] slices [t]. [bounds] gives, per axis in order, a
    half-open [(start, stop)] range. Every axis must be listed.

    @raise Invalid_argument if [bounds] length differs from [ndim t]. *)

val symbolic_shrink :
  Tensor.t -> (Tolk_uop.Uop.t * Tolk_uop.Uop.t) option list -> Tensor.t
(** [symbolic_shrink t bounds] is {!shrink} over dimension nodes: each bound is
    a half-open [(start, stop)] pair of possibly symbolic values, whose axis
    size becomes [stop - start], or [None] to leave the axis unchanged. The
    upper bound of every slice must provably stay inside the input axis.

    @raise Invalid_argument
      if [bounds] length differs from [ndim t] or a slice can exceed its
      axis. *)

val squeeze : ?dim:int -> Tensor.t -> Tensor.t
(** [squeeze t] removes all size-[1] axes. [squeeze ~dim t] removes only axis
    [dim], and returns [t] unchanged if that axis is not size [1]. *)

val unsqueeze : Tensor.t -> int -> Tensor.t
(** [unsqueeze t dim] inserts a new size-[1] axis at position [dim]. *)

val transpose : ?dim0:int -> ?dim1:int -> Tensor.t -> Tensor.t
(** [transpose t] swaps axes [dim0] and [dim1] (default the last two). *)

val flatten : ?start_dim:int -> ?end_dim:int -> Tensor.t -> Tensor.t
(** [flatten t] collapses the axes from [start_dim] to [end_dim] inclusive
    (default: all axes) into a single axis. *)

val unflatten : Tensor.t -> int -> int list -> Tensor.t
(** [unflatten t dim sizes] splits axis [dim] into the axes [sizes]. The
    product of [sizes] must equal the size of [dim]. *)

val repeat : Tensor.t -> int list -> Tensor.t
(** [repeat t repeats] tiles [t], repeating it [List.nth repeats k] times
    along axis [k]. [repeats] may be longer than [ndim t], adding leading
    axes. *)

val shrink_to : Tensor.t -> int option list -> Tensor.t
(** [shrink_to t dims] shrinks each axis to size [dims], keeping [0] as the
    start. [None] leaves an axis unchanged. Every axis must be listed. *)

val pad_to : Tensor.t -> int option list -> Tensor.t
(** [pad_to t dims] pads each axis on the right up to size [dims]. [None]
    leaves an axis unchanged. Every axis must be listed. *)

val pool : Tensor.t -> k:int list -> ?stride:int list -> ?dilation:int list ->
  unit -> Tensor.t
(** [pool t ~k ()] forms sliding windows over the last [List.length k] axes of
    [t] for a convolution or pooling. Each of those axes is replaced by an
    output-position axis and a kernel axis: an input of shape
    [(..., i0, i1, ...)] with kernel [k], strides [stride] (default all [1]),
    and dilations [dilation] (default all [1]) becomes
    [(..., o0, o1, ..., k0, k1, ...)] where [oj = ceil((ij - dj*(kj-1)) / sj)].
    A reduction over the trailing kernel axes then realises the window
    operation. [stride] and [dilation] may be a single-element list, which is
    broadcast to every kernel axis.

    @raise Invalid_argument
      if a kernel is larger than its input axis, or if the argument lengths
      are inconsistent. *)

val unfold : Tensor.t -> int -> size:int -> step:int -> Tensor.t
(** [unfold t dim ~size ~step] replaces axis [dim] with overlapping windows: a
    new axis of window count [(shape.(dim) - size) / step + 1] followed by a
    trailing axis of length [size]. The window at index [w] holds
    [t]'s slice [\[w*step, w*step + size)] along [dim].

    @raise Invalid_argument
      if [size] is negative, [step] is not positive, or [size] exceeds
      [shape.(dim)]. *)

val split : ?dim:int -> Tensor.t -> int -> Tensor.t list
(** [split t size] cuts [t] along [dim] (default [0]) into consecutive chunks of
    [size] elements; the final chunk is smaller when [size] does not divide the
    axis.

    @raise Invalid_argument if [size] is not positive. *)

(** {1 Indexing}

    An [index] describes how one axis (or a freshly inserted one) is selected.
    A list of them addresses a tensor from the outermost axis inward; the whole
    selection is realised by {!Op.getitem}. *)

type index =
  | I of int  (** Pick a single position, dropping the axis. Negative counts from the end. *)
  | R of int option * int option * int option
      (** A [start:stop:step] slice; each bound omitted with [None]. A negative
          bound counts from the end and a negative [step] reverses direction. *)
  | All  (** Keep the whole axis, equivalent to [R (None, None, None)]. *)
  | New  (** Insert a new size-[1] axis at this position. *)
  | Ellipsis  (** Stand in for as many {!All} axes as needed to cover the rank. *)
  | T of Tensor.t
      (** Advanced indexing: an integer tensor whose elements select positions
          along the axis. Negative elements count from the end. *)

type resolved = Newaxis | View | Advanced of Tensor.t
(** How a parsed index acts on the tensor: {!Newaxis} injects an axis,
    {!View} is a pure shrink/flip/stride of an existing axis, and
    {!Advanced} gathers along it with an index tensor. *)

type parsed = {
  size : int;  (** Length of the resulting axis (before any collapse). *)
  boundary : int * int;  (** Half-open [\[lo, hi)] window kept from the axis. *)
  stride : int;  (** Step through the window; negative reverses it. *)
  collapse_dim : bool;  (** Whether the axis is dropped, as for an integer index. *)
  resolved : resolved;
}
(** A single index resolved against a concrete axis size. *)

val parse_view_index : index -> int -> parsed
(** [parse_view_index index size] resolves a non-advanced [index] against an
    axis of length [size].

    @raise Invalid_argument
      if an integer index is out of bounds or a slice step is zero. *)

val normalize_indices : Tensor.t -> index list -> index list
(** [normalize_indices t indices] expands a single {!Ellipsis} (or, if there is
    none, a virtual one at the end) into enough {!All} entries that every real
    axis of [t] is addressed.

    @raise Invalid_argument
      if there is more than one ellipsis or more real indices than axes. *)

val apply_view_ops : Tensor.t -> parsed list -> Tensor.t
(** [apply_view_ops t parsed] applies the shrink, flip, and stride of each
    parsed view index to the matching axis of [t], in order. Advanced and
    new-axis entries must already be filtered out. *)
