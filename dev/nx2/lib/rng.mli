(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [Nx.Rng]: keys, samplers and the scope. Each draw is one map whose elements
    compute their Threefry blocks from their positions. The scope's keys are
    host keys, computed when drawn and received by no interpretation; a
    keyless draw with a parameter makes its key where the parameter lies, from
    the two words of the scope's next key. *)

type +'d key = private (int32, Nx_array.Dtype.int32_elt, 'd) Value.t

val key : int -> Devices.host key
val of_tensor : (int32, Nx_array.Dtype.int32_elt, 'd) Value.t -> 'd key
val to_tensor : 'd key -> (int32, Nx_array.Dtype.int32_elt, 'd) Value.t
val place : 'e Devices.placement -> 'd key -> 'e key
val walk : ('a, 'b, 'd, 'e) Ptree.Walk.cursor -> 'd key -> 'e key
val split : ?n:int -> 'd key -> 'd key array
val split_batch : n:int -> 'd key -> 'd key
val fold_in : 'd key -> int -> 'd key

val fold_in_tensor :
  'd key -> (int32, Nx_array.Dtype.int32_elt, 'd) Value.t -> 'd key

val bits :
  ?key:Devices.host key ->
  int array ->
  (int32, Nx_array.Dtype.int32_elt, Devices.host) Value.t

val uniform :
  ?key:Devices.host key ->
  (float, 's) Nx_array.Dtype.t ->
  int array ->
  (float, 's, Devices.host) Value.t

val normal :
  ?key:Devices.host key ->
  (float, 's) Nx_array.Dtype.t ->
  int array ->
  (float, 's, Devices.host) Value.t

val exponential :
  ?key:Devices.host key ->
  (float, 's) Nx_array.Dtype.t ->
  int array ->
  (float, 's, Devices.host) Value.t

val gumbel :
  ?key:Devices.host key ->
  (float, 's) Nx_array.Dtype.t ->
  int array ->
  (float, 's, Devices.host) Value.t

val randint :
  ?key:Devices.host key ->
  ?low:int ->
  high:int ->
  int array ->
  (int32, Nx_array.Dtype.int32_elt, Devices.host) Value.t

val bernoulli :
  ?key:'d key ->
  (float, 's, 'd) Value.t ->
  (bool, Nx_array.Dtype.bool_elt, 'd) Value.t

val gamma : ?key:'d key -> (float, 's, 'd) Value.t -> (float, 's, 'd) Value.t

val beta :
  ?key:'d key ->
  (float, 's, 'd) Value.t ->
  (float, 's, 'd) Value.t ->
  (float, 's, 'd) Value.t

val von_mises :
  ?key:'d key -> (float, 's, 'd) Value.t -> (float, 's, 'd) Value.t

val poisson :
  ?key:'d key ->
  (float, 's, 'd) Value.t ->
  (int32, Nx_array.Dtype.int32_elt, 'd) Value.t

val binomial :
  ?key:'d key ->
  (int32, Nx_array.Dtype.int32_elt, 'd) Value.t ->
  (float, 's, 'd) Value.t ->
  (int32, Nx_array.Dtype.int32_elt, 'd) Value.t

val truncated_normal :
  ?key:'d key ->
  (float, 's, 'd) Value.t ->
  (float, 's, 'd) Value.t ->
  (float, 's, 'd) Value.t

val dirichlet :
  ?key:'d key -> (float, 's, 'd) Value.t -> (float, 's, 'd) Value.t

val categorical :
  ?key:'d key ->
  ?axis:int ->
  (float, 's, 'd) Value.t ->
  (int64, Nx_array.Dtype.int64_elt, 'd) Value.t

val permutation :
  ?key:Devices.host key ->
  int ->
  (int64, Nx_array.Dtype.int64_elt, Devices.host) Value.t

val with_key : Devices.host key -> (unit -> 'a) -> 'a
val refusing : string -> (unit -> 'a) -> 'a
val next_key : unit -> Devices.host key
