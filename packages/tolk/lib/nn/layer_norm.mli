(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Layer normalization over the last axis. *)

open Tolk_frontend

type t = {
  weight : Tensor.t option;  (** Per-feature scale, shape [(dim)]. *)
  bias : Tensor.t option;  (** Per-feature shift, shape [(dim)]. *)
  eps : float;
  dim : int;
}
(** A layer-norm layer. [weight] and [bias] are [None] when the layer was
    created without an elementwise affine transform. *)

val create : ?eps:float -> ?elementwise_affine:bool -> int -> t
(** [create dim] is a layer norm over a last axis of size [dim]. [eps]
    (default [1e-5]) stabilises the variance denominator.
    [elementwise_affine] (default [true]) adds a learned per-feature scale
    (initialised to ones) and shift (initialised to zeros). *)

val apply : t -> Tensor.t -> Tensor.t
(** [apply ln x] normalises [x] along its last axis to zero mean and unit
    variance (biased estimator), then applies the affine transform when
    present.

    @raise Invalid_argument if the last axis of [x] does not have size [dim]. *)
