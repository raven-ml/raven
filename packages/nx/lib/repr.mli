(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values from arrays and arrays from values: the implementation of [Nx.Repr].
    Errors name [by]. This module lies above {!Prim}. *)

val of_array :
  by:string -> 'd Devices.t -> ('v, 's) Nx_array.t -> ('v, 's, 'd) Value.t

val unsafe_of_array :
  by:string -> 'd Devices.t -> ('v, 's) Nx_array.t -> ('v, 's, 'd) Value.t

val array : by:string -> ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t option

val of_shards :
  by:string ->
  'd Devices.placement ->
  ('v, 's) Nx_array.t iarray ->
  ('v, 's, 'd) Value.t

val unsafe_of_shards :
  by:string ->
  'd Devices.placement ->
  ('v, 's) Nx_array.t iarray ->
  ('v, 's, 'd) Value.t

val shards : by:string -> ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t iarray

val layouts : ('v, 's, 'd) Value.t -> Nx_array.Layout.t iarray
(** Raises [Invalid_argument] naming [Nx.Repr.layouts] for a traced value. *)
