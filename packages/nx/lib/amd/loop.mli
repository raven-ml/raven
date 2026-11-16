(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A loop's launches of loop.hip.

    A plan takes a call nx.amd found nothing to refuse in, and lowers it to
    the words of its launches (kernels.h, Loops): its walk over the loop's
    axes, its operands' descriptors and its program's tape, one submission
    in all. A call too wide for one launch runs in parts: its outputs or
    its reductions split, its program cut at nodes mapped into temporaries
    in the workspace. A launch's association follows the call's extents
    and dtypes alone, so a result's bits depend on its terms' values alone,
    whatever the parts. Each answers [None] for a loop of no element. *)

val map :
  Nx_kernel.Prog.t ->
  dsts:Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  (Plan.t option, Nx_array.answer) result
(** [map p ~dsts loads] plans {!Nx_kernel.S.map}. It is [Error Declined]
    only where a part that copies one load into a temporary does not fit a
    launch, which no load of at most {!Nx_array.Layout.max_rank} axes
    reaches. *)

val reduce :
  Nx_kernel.Prog.t ->
  shape:int iarray ->
  axes:int iarray ->
  (Nx_kernel.reduction * int) iarray ->
  dsts:Nx_array.any iarray ->
  Nx_kernel.load iarray ->
  (Plan.t option, Nx_array.answer) result
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
  (Plan.t option, Nx_array.answer) result
(** [scan ?carry p ~axis r ~dsts loads] plans {!Nx_kernel.S.scan}, and is
    [Error Declined] as {!map} is. *)

val destinations : Nx_kernel.reduction -> int
(** [destinations r] is the destinations a reduction [r] stores: two for
    [Arg] and [Moments], one for the others. *)
