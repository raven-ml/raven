(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

val create :
  Tensor.t -> (Tolk_uop.Uop.t list -> string option -> Tensor.t) -> Tensor.t
(** [create t make] constructs values with the shape and placement of [t].
    [make shape device] is called once for a single or replicated tensor and
    once per device with the local shape for a sharded tensor. *)
