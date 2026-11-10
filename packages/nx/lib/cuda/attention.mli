(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Attention's launches of attention.cu.

    A plan takes a call nx.cuda found nothing to refuse in, chooses its
    path by the dtypes and the extents (attention.cu's header), cuts its
    keys into parts where its units are few, and lowers its pair program
    to the tapes the kernel runs on planes it fills. A program whose tapes
    do not fit a launch computes in part through {!Loop.map} first, into
    temporaries the kernel reads as loads. *)

val attention :
  Nx_kernel.Prog.t ->
  scale:float ->
  o:('w, 'r) Nx_array.t ->
  lse:(float, 'c) Nx_array.t ->
  ?lengths:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  ( ('v, 's) Nx_array.t,
    (int64, Nx_array.Dtype.int64_elt) Nx_array.t )
  Nx_kernel.kv ->
  Nx_array.any iarray ->
  Bytes.t Launch.plan
(** [attention p ~scale ~o ~lse ?lengths q kv loads] plans
    {!Nx_kernel.S.attention} of a call whose dtypes and shapes fit: maps
    into temporaries of their own, then calls of attention.cu, whose parts'
    states lie in the workspace between a call's launches. It is an
    [Error] only where {!Loop.map} is.

    Raises {!Rig.Out_of_memory} where the device has no memory for its
    temporaries. *)
