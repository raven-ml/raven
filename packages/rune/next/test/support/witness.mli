(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Storage identity: whether a compiled call's result took a consumed
    argument's memory. *)

val addresses : ('a, 'b) Nx.t -> nativeint list
(** [addresses x] is the address of each device's buffer of [x]'s storage, in
    the order of [x]'s placement's devices: equal for a result that took a
    consumed value's storage and that value, read before the call.

    Raises [Invalid_argument] if [x] is traced or consumed. *)
