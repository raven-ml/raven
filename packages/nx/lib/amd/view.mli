(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Contractions with their axes grouped.

    A view groups a contraction's axes into one of each kind, where every
    operand lays out each group's axes as one run of strides, as a matrix
    product reads them. A group's axes of extent [1] drop out, and an axis
    merges into the one before it where every operand of the group steps
    along it by its stride times its extent there. *)

type t
(** The type for views. *)

(** The type for a contraction's operands and its result. *)
type operand = A | B | Init | Dst

(** The type for a view's axes. [A] has [Batch], [Row] and [Contracted]; [B]
    has [Batch], [Column] and [Contracted]; [Init] and [Dst] have [Batch],
    [Row] and [Column]. *)
type axis = Batch | Row | Column | Contracted

val v :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  (t, Nx_array.answer) result
(** [v ~batch ~contracting ~dst ~init a b] is the view of the axes of [a],
    [b], [init] and [dst], as {!Nx_kernel.S.contract} takes them. It is
    [Error Shape_mismatch] where {!Nx_kernel.Shape.contract} answers [Error]
    for [a] and [b] or [dst] and [init] do not have the shape it gives, and
    otherwise [Error Declined] if a group does not merge into one axis. *)

val extent : t -> axis -> int
(** [extent v x] is [v]'s extent along [x]: [1] for a group of no axis. *)

val offset : t -> operand -> int
(** [offset v o] is [o]'s first element, in elements from its buffer's first
    byte; [0] for [Init] in a view without it. *)

val stride : t -> operand -> axis -> int
(** [stride v o x] is [o]'s stride along [x], in elements: [0] along an axis
    of at most one element, along an axis [o] does not have, and for [Init]
    in a view without it. *)
