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
    into its consumers. When [dtype] is omitted, scalar integers and floats
    start weak (see {!Tensor.scalar}). Materialization chooses a concrete dtype
    from the value's bounds with {!Tolk_uop.Uop.commit_dtype}. *)

val empty :
  ?dtype:Tolk_uop.Dtype.t -> ?device:Tolk_uop.Uop.device -> int list ->
  Tensor.t
(** [empty shape] is a tensor of shape [shape] over fresh storage whose
    contents are unspecified until something writes them. [dtype] defaults to
    the default float dtype. The storage is placed on [device]; without one it
    is placed by whatever consumes it.

    Raises [Invalid_argument] if [dtype] is weak. *)

val clone : ?device:Tolk_uop.Uop.device -> Tensor.t -> Tensor.t
(** [clone t] is [t]'s value in fresh storage: a new buffer written by one
    kernel that computes [t] into it. Writes into the clone leave [t]'s own
    storage alone. The buffer is placed on [device], which defaults to [t]'s
    device, and [t] is copied across when it lives on another one. A constant
    [t] has no device, and without [device] its clone is placed by whatever
    consumes it. Weak inputs commit to a concrete dtype according to
    {!Tolk_uop.Uop.commit_dtype}; exact integers outside all storage types
    raise [Invalid_argument]. *)

val full :
  ?dtype:Tolk_uop.Dtype.t -> ?buffer:bool -> int list -> Tensor.scalar ->
  Tensor.t
(** [full shape v] is a tensor of shape [shape] with every element equal to
    [v]. *)

val zeros : ?dtype:Tolk_uop.Dtype.t -> ?buffer:bool -> int list -> Tensor.t
(** [zeros shape] is a tensor of shape [shape] filled with zeros, defaulting
    to the default float dtype. *)

val ones : ?dtype:Tolk_uop.Dtype.t -> ?buffer:bool -> int list -> Tensor.t
(** [ones shape] is a tensor of shape [shape] filled with ones, defaulting to
    the default float dtype. *)

val const_like :
  ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.scalar -> Tensor.t
(** [const_like t v] is a broadcast constant with the shape of [t] and, unless
    [dtype] overrides it, the dtype of [t]; every element equals [v] coerced to
    that dtype. No storage is allocated. *)

val full_like :
  ?dtype:Tolk_uop.Dtype.t -> ?buffer:bool -> Tensor.t -> Tensor.scalar ->
  Tensor.t
(** [full_like t v] is [full] with the shape of [t] and, unless overridden,
    the dtype of [t]. *)

val zeros_like :
  ?dtype:Tolk_uop.Dtype.t -> ?buffer:bool -> Tensor.t -> Tensor.t
(** [zeros_like t] is a zero-filled tensor shaped like [t]. *)

val ones_like :
  ?dtype:Tolk_uop.Dtype.t -> ?buffer:bool -> Tensor.t -> Tensor.t
(** [ones_like t] is a one-filled tensor shaped like [t]. *)
