(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Strided tensor views.

    A view describes how a linear buffer is interpreted as an n-dimensional
    tensor through shape, strides, and offset. View operations are metadata
    transformations: they do not copy element storage. *)

type t
(** The type for tensor views. *)

(** {1:constructors Construction} *)

val create : ?offset:int -> ?strides:int array -> int array -> t
(** [create ?offset ?strides shape] is a view over [shape].

    Defaults:
    - [offset] defaults to [0].
    - [strides] defaults to C-contiguous strides derived from [shape].

    If [shape] has a zero-size dimension, the resulting view has [offset = 0].

    Raises [Invalid_argument] if explicit [strides] length does not match
    [Array.length shape]. *)

(** {1:accessors Accessors} *)

val shape : t -> int array
(** [shape v] is [v]'s shape. *)

val strides : t -> int array
(** [strides v] is [v]'s stride vector. *)

val offset : t -> int
(** [offset v] is [v]'s linear base offset. *)

val ndim : t -> int
(** [ndim v] is [Array.length (shape v)]. *)

val numel : t -> int
(** [numel v] is the product of dimensions in [shape v].

    [numel] of a scalar ([ndim v = 0]) is [1]. *)

val dim : int -> t -> int
(** [dim axis v] is dimension [axis] of [v].

    Raises [Invalid_argument] if [axis] is outside [[0; ndim v - 1]]. *)

val stride : int -> t -> int
(** [stride axis v] is stride [axis] of [v].

    Raises [Invalid_argument] if [axis] is outside [[0; ndim v - 1]]. *)

val is_c_contiguous : t -> bool
(** [is_c_contiguous v] is [true] iff [v] is recognized as C-contiguous. *)

(** {1:transform Transformations} *)

val reshape : t -> int array -> t
(** [reshape v new_shape] returns a view over the same storage with [new_shape]
    when stride-compatible.

    Supported cases include:
    - C-contiguous reshape.
    - Reshape by adding/removing singleton dimensions.
    - Certain merge/split patterns on compatible strided layouts.
    - All-zero-stride broadcast layouts.

    Raises [Invalid_argument] if reshape cannot be represented, including size
    mismatches (except zero-size special cases) or incompatible stride
    patterns. *)

val expand : t -> int array -> t
(** [expand v new_shape] broadcasts singleton dimensions to [new_shape] by
    setting corresponding strides to [0].

    Scalars ([ndim v = 0]) may expand to any rank.

    Raises [Invalid_argument] if ranks are incompatible for non-scalars, or if a
    non-singleton dimension would need expansion. *)

val permute : t -> int array -> t
(** [permute v axes] reorders dimensions according to [axes].

    Raises [Invalid_argument] if [axes] is not a valid permutation of
    [[0; ndim v - 1]]. *)

val shrink : t -> (int * int) array -> t
(** [shrink v bounds] restricts [v] to per-axis half-open intervals
    [(start, end)].

    Bounds must satisfy [0 <= start < end <= size] for each dimension.

    Raises [Invalid_argument] if bounds are malformed or rank mismatches. *)

val flip : t -> bool array -> t
(** [flip v axes_to_flip] reverses selected axes by negating strides and
    shifting offset.

    Raises [Invalid_argument] if [axes_to_flip] rank mismatches [ndim v]. *)
