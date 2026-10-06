(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Adaptive partitions of the unit cube.

    Each lane partitions [[0, 1]^d] into at most [budget] boxes, kept in
    [budget] slots: a box is [index / 2^level] to [(index + 1) / 2^level] along
    each axis, as integers, so its ends are exact at any depth and a derivative
    through them reaches the problem's ends. {!refine} bisects each lane's box
    of largest error until the lane settles. Shapes are static: every array has
    the slots in front, [[budget] @ lanes]. *)

type ('d, 'b) t = {
  level : (int32, Nx.int32_elt) Nx.t;
      (** Each box's level along each axis, [[budget] @ lanes @ [d]]. *)
  index : (int64, Nx.int64_elt) Nx.t;  (** Each box's index, as [level]. *)
  axis : (int64, Nx.int64_elt) Nx.t;
      (** The axis each box would split along, [[budget] @ lanes]. *)
  data : 'd;  (** Each box's data, leaves of shape [[budget] @ lanes @ _]. *)
  error : (float, 'b) Nx.t;
      (** Each box's error, [[budget] @ lanes]; the largest is bisected. *)
  used : (int32, Nx.int32_elt) Nx.t;
      (** The slots in use per lane, the first ones. *)
  status : (int32, Nx.int32_elt) Nx.t;  (** Each lane's status. *)
  evaluations : (int32, Nx.int32_elt) Nx.t;  (** Each lane's points evaluated. *)
}
(** The type for partitions with box data ['d] and errors of dtype ['b]. *)

val fractions :
  (float, 'b) Nx.dtype ->
  (int32, Nx.int32_elt) Nx.t ->
  (int64, Nx.int64_elt) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t
(** [fractions dtype level index] are the ends [index / 2^level] and
    [(index + 1) / 2^level]. *)

val refine :
  'd Nx.Ptree.t ->
  budget:int ->
  lanes:int array ->
  dims:int ->
  cost:int ->
  evaluate:
    ((int32, Nx.int32_elt) Nx.t ->
    (int64, Nx.int64_elt) Nx.t ->
    'd * (float, 'b) Nx.t * (int64, Nx.int64_elt) Nx.t) ->
  point:((int64, Nx.int64_elt) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  verdict:
    (('d, 'b) t ->
    (bool, Nx.bool_elt) Nx.t ->
    (bool, Nx.bool_elt) Nx.t * (bool, Nx.bool_elt) Nx.t) ->
  ('d, 'b) t
(** [refine s ~budget ~lanes ~dims ~cost ~evaluate ~point ~verdict] is the
    partition of [[0, 1]^dims] per lane of shape [lanes] at which each lane
    settled. [evaluate level index] is the data, error and split axis of [k]
    boxes per lane, of shapes [[k] @ lanes @ _]; each box costs [cost] points.
    [point axis t] is the problem's coordinate of fraction [t] along [axis], per
    lane. [verdict p live] is, per lane, whether [p] is not finite and whether
    it meets its tolerance, given {!in_use}'s [live].

    A lane settles, in this order: [Not_finite] or [Converged] by [verdict];
    [Stalled] when its worst box is at level 62 along its split axis, or no
    float lies strictly between its ends there; [Budget_spent] when its [budget]
    slots are used. A bisection puts the left half in the worst box's slot and
    the right half in the next free one. *)

val in_use : ('d, 'b) t -> (bool, Nx.bool_elt) Nx.t
(** [in_use p] is whether each slot holds a box, [[budget] @ lanes]. *)

val sum : (bool, Nx.bool_elt) Nx.t -> (float, 'c) Nx.t -> (float, 'c) Nx.t
(** [sum live v] is the sum of [v], [[budget] @ lanes @ _], over the slots
    {!in_use} marks [live]. *)

val worst :
  ('d, 'b) t ->
  point:((int64, Nx.int64_elt) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  (float, 'b) Nx.t * (float, 'b) Nx.t
(** [worst p ~point] is, per lane, the ends of the box of largest error along
    its split axis, mapped by [point] as {!refine} maps them. *)

val integrate :
  ('d, 'b) t ->
  ((int32, Nx.int32_elt) Nx.t -> (int64, Nx.int64_elt) Nx.t -> (float, 'b) Nx.t) ->
  (float, 'b) Nx.t
(** [integrate p f] is the sum of [f level index] over the boxes in use, [f]
    taking a chunk of [k] boxes per lane and returning [[k] @ lanes]. The chunks
    run under a {!Rune.scan}, so reverse mode keeps one chunk's values; an
    unused slot evaluates [[0, 1]^d] with weight zero, so its points lie inside
    the problem. *)
