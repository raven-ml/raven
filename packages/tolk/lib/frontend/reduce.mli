(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Reductions.

    Each reduction collapses one or more axes. [axis] selects the axes to
    reduce (default: all of them) and accepts negative indices. With
    [~keepdim:true] the reduced axes are kept as size [1] instead of being
    removed. *)

val sum :
  ?axis:int list -> ?keepdim:bool -> ?dtype:Tolk_uop.Dtype.Val.t -> Tensor.t ->
  Tensor.t
(** [sum t] sums the elements of [t]. The accumulation is done at a widened
    dtype to limit overflow; pass [dtype] to fix it. Narrow-float inputs are
    cast back to their own dtype unless [dtype] is given. *)

val prod :
  ?axis:int list -> ?keepdim:bool -> ?dtype:Tolk_uop.Dtype.Val.t -> Tensor.t ->
  Tensor.t
(** [prod t] multiplies the elements of [t]. Pass [dtype] to set the
    accumulation dtype. *)

val max : ?axis:int list -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [max t] is the largest element of [t] along the reduced axes. *)

val min : ?axis:int list -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [min t] is the smallest element of [t] along the reduced axes. *)

val any : ?axis:int list -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [any t] is true where at least one element along the reduced axes is
    non-zero. *)

val all : ?axis:int list -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [all t] is true where every element along the reduced axes is non-zero. *)
