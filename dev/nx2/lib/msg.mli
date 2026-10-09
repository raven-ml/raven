(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Messages: what nx raises for misuse, in one format.

    A message starts with the function the user called, then the reason, then,
    where they matter, the operands as [dtype shape on set]:
    [Nx.add: shapes [2; 3] and [4] differ (float32 [2; 3], float32 [4])]. A
    value on the host, or a constant, prints no [on] part. *)

val fail : by:string -> ('a, Format.formatter, unit, 'b) format4 -> 'a
(** [fail ~by fmt …] raises [Invalid_argument] of [by], [": "] and the formatted
    reason. *)

val operand :
  Format.formatter ->
  ('v, 's) Nx_array.Dtype.t * int array * 'd Devices.placement ->
  unit
(** [operand ppf (dt, shape, p)] formats [float32 [2; 3]], followed by
    [ on set 2 [CUDA:0]] where [p]'s set is not the host's. *)

val declined :
  by:string ->
  kind:string ->
  kernels:string ->
  Rig.t ->
  Nx_array.Dtype.any list ->
  'a
(** [declined ~by ~kind ~kernels d dts] raises for a core case a kernel library
    declined: [Nx.exp: nx.metal does not compute Exp on float64 (METAL:0)]. *)

val no_kernels : by:string -> op:string -> 'd Devices.t -> 'a
(** [no_kernels ~by ~op s] raises for an eager operation on a set without
    kernels, naming the set, the operation and the remedies: place the value on
    a set with kernels, mint the set with kernels, or apply the function where a
    compiler stages it. *)
