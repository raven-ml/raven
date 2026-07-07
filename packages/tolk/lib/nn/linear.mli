(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Affine transformation of the last axis. *)

open Tolk_frontend

type t = {
  weight : Tensor.t;  (** Shape [(out_features, in_features)]. *)
  bias : Tensor.t option;  (** Shape [(out_features)], when present. *)
}
(** A linear layer. *)

val create : ?bias:bool -> int -> int -> t
(** [create in_features out_features] is a linear layer mapping the last axis
    from [in_features] to [out_features], zero-initialised. [bias] (default
    [true]) selects whether an additive bias is included. Load trained values
    with {!State.load_state_dict}. *)

val apply : t -> Tensor.t -> Tensor.t
(** [apply l x] is [x @ weight^T + bias], contracting the last axis of [x]
    with [in_features]. *)
