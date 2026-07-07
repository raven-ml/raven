(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Type conversions.

    Numeric casts change the element type, converting values with the usual
    rounding and range rules. Bitcasts reinterpret the underlying bits and
    require matching element sizes. *)

val cast : Tensor.t -> Tolk_uop.Dtype.t -> Tensor.t
(** [cast t dt] converts [t] to element type [dt]. Returns [t] unchanged when
    its dtype is already [dt]. *)

val bitcast : Tensor.t -> Tolk_uop.Dtype.t -> Tensor.t
(** [bitcast t dt] reinterprets the bits of [t] as [dt] without conversion.

    @raise Invalid_argument
      if [dt] and [t]'s dtype have different element sizes. This module does
      not implement the multi-element repacking used for size-changing
      bitcasts. *)

val is_floating_point : Tensor.t -> bool
(** [is_floating_point t] is [true] iff [t]'s scalar dtype is a floating-point
    type. *)

val element_size : Tensor.t -> int
(** [element_size t] is the size in bytes of one element of [t]. *)

val float : Tensor.t -> Tensor.t
(** [float t] casts [t] to 32-bit float. *)

val half : Tensor.t -> Tensor.t
(** [half t] casts [t] to 16-bit float. *)

val int : Tensor.t -> Tensor.t
(** [int t] casts [t] to 32-bit signed integer. *)

val bool : Tensor.t -> Tensor.t
(** [bool t] casts [t] to boolean. *)

val double : Tensor.t -> Tensor.t
(** [double t] casts [t] to 64-bit float. *)

val long : Tensor.t -> Tensor.t
(** [long t] casts [t] to 64-bit signed integer. *)
