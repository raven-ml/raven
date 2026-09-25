(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Binding compiled schedules to owned storage. *)

val run :
  resolve:(Tolk_uop.Uop.t -> Device.Buffer.t) ->
  ?allow_cache:bool -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [run ~resolve ?allow_cache linear] allocates tagged placeholders, resolves
    device addresses and applies constant initialization stores outside call
    bodies. The result retains every allocation whose address it embeds.
    Untagged parameters stay bound at execution time. [lt_input] placeholders
    bind through [resolve] and disable caching. Other links are cached weakly
    by [linear] unless [allow_cache] is [false]. Queue submissions record the
    device instance IDs used to bind native addresses; relinking a submission
    after any recorded owner was replaced raises [Invalid_argument]. *)
