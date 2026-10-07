(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Newton-type searches share: vectors over lanes, the contraction
    estimate and the backtracking line search.

    A vector over lanes is a tensor of shape [lanes @ [n]]: one vector of [n]
    components for each index of [lanes]. A problem that is one system has no
    lane axis, [lanes = [||]]. Sums over a vector add pairwise in a fixed order,
    so eager and compiled searches take the same decisions. *)

val dot : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [dot u v] is each lane's inner product, of the lanes' shape. *)

val norm : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [norm v] is each lane's Euclidean norm, of the lanes' shape. *)

val hold :
  (bool, Nx.bool_elt) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t
(** [hold m next last] is [next] in the lanes where [m], [last] elsewhere, for
    tensors whose shape starts with [m]'s. *)

val accepted :
  Tol.t -> e:(float, 'b) Nx.t -> y:(float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t
(** [accepted tol ~e ~y] is [true] in each lane whose error [e] at [y] meets
    [tol]: the root mean square of the scaled components is at most [1]. *)

val contraction : (float, 'b) Nx.t -> q:(float, 'b) Nx.t -> (float, 'b) Nx.t
(** [contraction delta ~q] is the error of an iteration whose undamped step is
    [delta] and whose undamped map contracts by [q] per lane:
    [|delta| q / (1 − q)] per component. It is infinite unless [0 ≤ q < 1], and
    zero where [delta] is. *)

val backtrack :
  'p Nx.Ptree.t ->
  (float, 'b) Nx.dtype ->
  trials:int ->
  running:(bool, Nx.bool_elt) Nx.t ->
  accept:((float, 'b) Nx.t -> (float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t) ->
  shrink:((float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  ((float, 'b) Nx.t -> 'p * (float, 'b) Nx.t) ->
  'p ->
  'p * (bool, Nx.bool_elt) Nx.t * (int32, Nx.int32_elt) Nx.t
(** [backtrack p dtype ~trials ~running ~accept ~shrink trial last] searches
    from the step length [α = 1] of [dtype] in each running lane: [trial α] is a
    point's payload of structure [p], whose tensors' shapes start with the
    lanes', and its merit [φ]. A lane ends at the first [α] with [accept α φ],
    and otherwise moves to [shrink α φ], which is at most [α / 2], for at most
    [trials] trials. The result is the accepted payload, or [last] in a lane
    that accepted none, whether each lane accepted, and the trials each lane
    took. *)

val armijo :
  phi0:(float, 'b) Nx.t ->
  slope0:(float, 'b) Nx.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t)
  * ((float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t)
(** [armijo ~phi0 ~slope0] is the sufficient-decrease test of a merit [φ] with
    [φ 0 = phi0] and [φ' 0 = slope0], [φ α ≤ φ 0 + c α φ' 0] with [c = 10⁻⁴],
    and its step: the minimum of the quadratic through [φ 0], [φ' 0] and [φ α],
    kept in [[α / 10, α / 2]]. *)
