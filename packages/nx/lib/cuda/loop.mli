(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A loop's launches of loop.cu.

    A plan takes a call nx.cuda found nothing to refuse in, and lowers
    it to the parameter words of its launches (kernels.h, Loops): its walk
    over the loop's axes, its operands' descriptors and its program's tape.
    A call too wide for one launch runs in parts, each a submission of its
    own: its outputs or its reductions split, its program cut at nodes
    mapped into temporaries. A launch's association follows the call's
    extents and dtypes alone, so a result's bits depend on its terms'
    values alone, whatever the parts. *)

val map :
  Nx_kernel.Prog.t ->
  dsts:Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  Bytes.t Launch.plan
(** [map p ~dsts loads] plans {!Nx_kernel.S.map}, a part's temporaries in
    buffers of their own. It is [Error Declined] only where a part that
    copies one load into a temporary does not fit a launch, which no load
    of at most {!Nx_array.Layout.max_rank} axes reaches. *)

val reduce :
  Nx_kernel.Prog.t ->
  shape:int iarray ->
  axes:int iarray ->
  (Nx_kernel.reduction * int) iarray ->
  dsts:Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  Bytes.t Launch.plan
(** [reduce p ~shape ~axes rs ~dsts loads] plans {!Nx_kernel.S.reduce} of
    outputs of at least one term, each reduction in its turn. It is
    [Error Declined] as {!map} is. *)

val scan :
  ?carry:Nx_array.any ->
  Nx_kernel.Prog.t ->
  axis:int ->
  Nx_kernel.reduction * int ->
  dsts:Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  Bytes.t Launch.plan
(** [scan ?carry p ~axis r ~dsts loads] plans {!Nx_kernel.S.scan}, and
    is [Error Declined] as {!map} is. *)

val destinations : Nx_kernel.reduction -> int
(** [destinations r] is the destinations a reduction by [r] stores: two for
    [Arg] and [Moments], one for the others. *)

(** {1:tapes Tapes of other kernels}

    A program lowered to loop.cu's tape for a kernel that runs it on
    planes it fills itself, as attention's pair program: each [In] and
    [Coord] node it reads lies in planes the kernel fills before a run, and
    the tape computes the rest. *)

type tape
(** The type for tapes as they grow: their constants, their bodies and
    their planes. *)

val tape : unit -> tape
(** [tape ()] is a tape of no instruction and no plane. *)

val given_plane : tape -> Nx_array.Dtype.any -> int
(** [given_plane t dt] is the first of the planes a value of [dt] takes,
    taken for good: the kernel fills them. *)

val seal : tape -> unit
(** [seal t] takes every plane below the highest taken yet, so that a body
    lowered later never writes a plane an earlier one reads, nor a
    constant's. *)

val planes_taken : tape -> int
(** [planes_taken t] is one past the highest plane [t] takes. *)

val lower_given :
  tape ->
  Nx_kernel.Prog.t ->
  roots:int list ->
  given:(Nx_kernel.Prog.node -> int option) ->
  out:(int -> int -> unit) ->
  int * int
(** [lower_given t p ~roots ~given ~out] lowers the nodes of [p] that the
    outputs [roots] reach into [t]'s constants and a body of its own, and
    is that body's first word and its words. Each [In] and [Coord] node [n]
    it reads holds its value in the planes from [given n], which must be
    [Some]. [out k q] is called with each output [k] among [roots] and the
    first plane of its value, which no instruction of the body writes after
    it. Raises [Too_wide] past 4096 planes. *)

val prologue : tape -> Bytes.t
(** [prologue t] is the words of [t]'s constants, little-endian, run once
    before any body. *)

val body : tape -> Bytes.t
(** [body t] is the words of [t]'s bodies, little-endian. *)

exception Too_wide
(** The exception for a tape past the planes an instruction names. *)
