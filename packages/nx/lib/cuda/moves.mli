(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Moves and their adjoints: a call's launches of moves.cu.

    A plan takes a call nx.cuda found nothing to refuse in, its buffers
    live, and writes its launches' parameters as kernels.h lays them out,
    as calls each submitted as one submission: every call's launches that
    share the workspace are one. A submission's refs follow its kernels
    alone, and the sequences of kernels the plans make are fewer than 100.
    A launch's grid follows its work alone and changes no bit. *)

type index := (int64, Nx_array.Dtype.int64_elt) Nx_array.t

val gather :
  axis:int ->
  dst:('v, 's) Nx_array.t ->
  index ->
  ('v, 's) Nx_array.t ->
  Bytes.t Launch.plan
(** [gather ~axis ~dst idx x] plans {!Nx_kernel.S.gather}: one launch. *)

val gather_ranges :
  dst:('v, 's) Nx_array.t ->
  starts:index ->
  ends:index ->
  ('v, 's) Nx_array.t ->
  Bytes.t Launch.plan
(** [gather_ranges ~dst ~starts ~ends x] plans
    {!Nx_kernel.S.gather_ranges}: the runs' running maxima of their ends in
    the workspace, 24 bytes a run, then the gather. *)

val assemble :
  fill:'v ->
  dst:('v, 's) Nx_array.t ->
  (Nx_array.Move.range iarray * ('v, 's) Nx_array.t) iarray ->
  Bytes.t Launch.plan
(** [assemble ~fill ~dst pieces] plans {!Nx_kernel.S.assemble}: a fill
    where the pieces may not cover [dst], then the pieces in order, a
    launch for each group of them its parameters hold. *)

val fold :
  Nx_kernel.pad ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Bytes.t Launch.plan
(** [fold p ~dst x] plans {!Nx_kernel.S.fold}: one launch. *)

val scatter :
  Nx_kernel.combine ->
  unique:bool ->
  axis:int ->
  dst:('v, 's) Nx_array.t ->
  into:('v, 's) Nx_array.t ->
  index ->
  ('v, 's) Nx_array.t ->
  Bytes.t Launch.plan
(** [scatter c ~unique ~axis ~dst ~into idx u] plans
    {!Nx_kernel.S.scatter}: a copy of [into] unless it is [dst], then one
    pass, two for [Set], or a float [Add]'s position tree in the workspace.
    It is [Error Declined] where that tree's keys pass 64 bits. *)
