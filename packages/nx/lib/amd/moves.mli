(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gathers, scatters, assemblies and folds: a call's launches of moves.hip.

    Each plans a call that fits and that nx.amd found nothing to refuse in, as
    one call ({!Plan}); [None] where it has no element to write. A launch's grid
    follows its work alone and changes no bit. *)

type index := (int64, Nx_array.Dtype.int64_elt) Nx_array.t
type planned := (Plan.t option, Nx_array.answer) result

val gather :
  axis:int -> dst:('v, 's) Nx_array.t -> index -> ('v, 's) Nx_array.t -> planned
(** [gather ~axis ~dst idx x] plans {!Nx_kernel.S.gather}: one launch. *)

val gather_ranges :
  dst:('v, 's) Nx_array.t ->
  starts:index ->
  ends:index ->
  ('v, 's) Nx_array.t ->
  planned
(** [gather_ranges ~dst ~starts ~ends x] plans {!Nx_kernel.S.gather_ranges}: the
    runs' running maxima of their ends in the workspace, 24 bytes a run and 8
    for each 4096 runs of a row, then the gather. *)

val assemble :
  fill:'v ->
  dst:('v, 's) Nx_array.t ->
  (Nx_array.Move.range iarray * ('v, 's) Nx_array.t) iarray ->
  planned
(** [assemble ~fill ~dst pieces] plans {!Nx_kernel.S.assemble}: a fill where the
    pieces may not cover [dst], then the pieces in order, a launch for each
    group of them its words hold. *)

val fold :
  Nx_kernel.pad -> dst:('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> planned
(** [fold p ~dst x] plans {!Nx_kernel.S.fold}: one launch. *)

val scatter :
  Nx_kernel.combine ->
  unique:bool ->
  axis:int ->
  dst:('v, 's) Nx_array.t ->
  into:('v, 's) Nx_array.t ->
  index ->
  ('v, 's) Nx_array.t ->
  planned
(** [scatter c ~unique ~axis ~dst ~into idx u] plans {!Nx_kernel.S.scatter}: a
    copy of [into] unless it is [dst], then one pass; for [Set], a zeroing and
    two passes; for a complex [Max] or [Min], a zeroing and two passes for
    complex64, three for complex128; or a float [Add]'s position tree: one
    launch along an axis of one tile, else, where the axis has at most a
    tile's targets, the tiles' launch and the packing of their entries, a
    tally and a pack; then a radix sort of the entries by target, a count and
    a move a key byte; then a launch for each level of the runs' sums. *)

val bound_scatters :
  parts:int -> groups:int -> whole:int -> wide:bool -> unit
(** [bound_scatters ~parts ~groups ~whole ~wide] makes a float [Add]'s sort
    by target cut its entries into at most [parts] parts, in place of 256,
    and its launches take at most [groups] groups, in place of
    {!Plan.stride_groups}; its runs of at most [whole] entries, in place of
    64, summed whole, none where [0]; and with [wide] its items wide,
    whatever their bits, so that tests reach parts of several tiles, groups
    of several parts, runs summed by levels and wide items with small
    arrays. It applies to calls planned after it returns.

    Raises [Invalid_argument] if [parts] is not from 1 to 256, [groups]
    not from 1 to {!Plan.stride_groups} or [whole] not from 0 to 64. *)
