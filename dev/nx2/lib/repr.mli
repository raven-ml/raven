(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values from arrays and arrays from values: the implementation of [Nx.Repr].
    Errors name [by]. *)

val of_array :
  by:string -> 'd Devices.t -> ('v, 's) Nx_array.t -> ('v, 's, 'd) Value.t

val array : ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t option

val of_shards :
  by:string ->
  'd Devices.placement ->
  ('v, 's) Nx_array.t iarray ->
  ('v, 's, 'd) Value.t

val shards : ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t iarray option
