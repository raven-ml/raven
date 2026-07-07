(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tensor creation.

    These build constant-valued tensors. By default the constant is
    materialized into a fresh buffer on realization, so the result owns
    storage that in-place assignment (see {!Op.assign}) can later write to.
    With [~buffer:false] the result is instead a pure broadcast view of a
    single scalar constant: no storage is ever allocated and the value folds
    into its consumers. When [dtype] is omitted, it follows the fill value's
    variant (see {!Tensor.scalar}). *)

val full :
  ?dtype:Tolk_uop.Dtype.Val.t -> ?buffer:bool -> int list -> Tensor.scalar ->
  Tensor.t
(** [full shape v] is a tensor of shape [shape] with every element equal to
    [v]. *)

val zeros : ?dtype:Tolk_uop.Dtype.Val.t -> ?buffer:bool -> int list -> Tensor.t
(** [zeros shape] is a tensor of shape [shape] filled with zeros, defaulting
    to the default float dtype. *)

val ones : ?dtype:Tolk_uop.Dtype.Val.t -> ?buffer:bool -> int list -> Tensor.t
(** [ones shape] is a tensor of shape [shape] filled with ones, defaulting to
    the default float dtype. *)

val const_like : Tensor.t -> Tensor.scalar -> Tensor.t
(** [const_like t v] is a broadcast constant with the shape and dtype of [t],
    with every element equal to [v] coerced to [t]'s dtype. No storage is
    allocated. *)

val full_like :
  ?dtype:Tolk_uop.Dtype.Val.t -> ?buffer:bool -> Tensor.t -> Tensor.scalar ->
  Tensor.t
(** [full_like t v] is [full] with the shape of [t] and, unless overridden,
    the dtype of [t]. *)

val zeros_like :
  ?dtype:Tolk_uop.Dtype.Val.t -> ?buffer:bool -> Tensor.t -> Tensor.t
(** [zeros_like t] is a zero-filled tensor shaped like [t]. *)

val ones_like :
  ?dtype:Tolk_uop.Dtype.Val.t -> ?buffer:bool -> Tensor.t -> Tensor.t
(** [ones_like t] is a one-filled tensor shaped like [t]. *)
