(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Multi-device sharding transformations. *)

val multi_pm :
  Tolk_ir.Tensor.builder ->
  shapes:int list option array ->
  devices:Tolk_ir.Tensor.device option array ->
  Tolk_ir.Tensor.t ->
  Tolk_ir.Tensor.view ->
  Tolk_ir.Tensor.view option
(** [multi_pm b ~shapes ~devices program view] is the combined multi-device
    pattern matcher. *)
