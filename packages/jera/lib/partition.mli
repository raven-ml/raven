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

type ('q, 'b, 'd) t = {
  level : 'd Nx.int32_t;
      (** Each box's level along each axis, [[budget] @ lanes @ [d]]. *)
  index : 'd Nx.int64_t;  (** Each box's index, as [level]. *)
  axis : 'd Nx.int64_t;
      (** The axis each box would split along, [[budget] @ lanes]. *)
  data : 'q;  (** Each box's data, leaves of shape [[budget] @ lanes @ _]. *)
  error : (float, 'b, 'd) Nx.t;
      (** Each box's error, [[budget] @ lanes]; the largest is bisected. *)
  used : 'd Nx.int32_t;  (** The slots in use per lane, the first ones. *)
  status : 'd Nx.int32_t;  (** Each lane's status. *)
  evaluations : 'd Nx.int32_t;  (** Each lane's points evaluated. *)
}
(** The type for partitions with box data ['q] and errors of dtype ['b]. *)

val fractions :
  (float, 'b) Nx.dtype ->
  'd Nx.int32_t ->
  'd Nx.int64_t ->
  (float, 'b, 'd) Nx.t * (float, 'b, 'd) Nx.t
(** [fractions dtype level index] are the ends [index / 2^level] and
    [(index + 1) / 2^level]. *)

val refine :
  ('q, 'd) Nx.Ptree.t ->
  at:'d Nx.Placement.t ->
  budget:int ->
  lanes:int array ->
  dims:int ->
  cost:int ->
  evaluate:
    ('d Nx.int32_t ->
    'd Nx.int64_t ->
    'q * (float, 'b, 'd) Nx.t * 'd Nx.int64_t) ->
  point:('d Nx.int64_t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t) ->
  verdict:(('q, 'b, 'd) t -> 'd Nx.bool_t -> 'd Nx.bool_t * 'd Nx.bool_t) ->
  ('q, 'b, 'd) t
(** [refine s ~at ~budget ~lanes ~dims ~cost ~evaluate ~point ~verdict] is the
    partition of [[0, 1]^dims] per lane of shape [lanes] at which each lane
    settled, its arrays made at [at], which must cut nothing (a placement
    {!Nx.beside} gives). [evaluate level index] is the data, error and split
    axis of [k] boxes per lane, of shapes [[k] @ lanes @ _]; each box costs
    [cost] points. [point axis t] is the problem's coordinate of fraction [t]
    along [axis], per lane. [verdict p live] is, per lane, whether [p] is not
    finite and whether it meets its tolerance, given {!in_use}'s [live].

    A lane settles, in this order: [Not_finite] or [Converged] by [verdict];
    [Stalled] when its worst box is at level 62 along its split axis, or no
    float lies strictly between its ends there; [Budget_spent] when its [budget]
    slots are used. A bisection puts the left half in the worst box's slot and
    the right half in the next free one. *)

val in_use : ('q, 'b, 'd) t -> 'd Nx.bool_t
(** [in_use p] is whether each slot holds a box, [[budget] @ lanes]. *)

val sum : 'd Nx.bool_t -> (float, 'c, 'd) Nx.t -> (float, 'c, 'd) Nx.t
(** [sum live v] is the sum of [v], [[budget] @ lanes @ _], over the slots
    {!in_use} marks [live]. *)

val worst :
  ('q, 'b, 'd) t ->
  point:('d Nx.int64_t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t) ->
  (float, 'b, 'd) Nx.t * (float, 'b, 'd) Nx.t
(** [worst p ~point] is, per lane, the ends of the box of largest error along
    its split axis, mapped by [point] as {!refine} maps them. *)

val integrate :
  ('q, 'b, 'd) t ->
  ('d Nx.int32_t -> 'd Nx.int64_t -> (float, 'b, 'd) Nx.t) ->
  (float, 'b, 'd) Nx.t
(** [integrate p f] is the sum of [f level index] over the boxes in use, [f]
    taking a chunk of [k] boxes per lane and returning [[k] @ lanes]. The chunks
    run under a {!Rune.scan}, so reverse mode keeps one chunk's values; an
    unused slot evaluates [[0, 1]^d] with weight zero, so its points lie inside
    the problem. *)
