(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Splitting methods for separable Hamiltonians.

    A Hamiltonian [H (q, p) = T p + V q] is the sum of two parts whose flows the
    caller computes exactly: the {e kick}, which moves momenta by [−∇V] over a
    duration, and the {e drift}, which moves positions by [∇T]. A splitting
    composes them over a step [h] as
    [K(a₁h) D(b₁h) K(a₂h) … D(b_m h) K(a_(m+1) h)]. Every scheme here is
    palindromic, so each step is symmetric and the march time-reversible; with
    exact flows of a Hamiltonian it is symplectic, and its energy error stays
    bounded over long times instead of growing.

    A flow may be any exact flow of its part: a Wisdom–Holman drift is a Kepler
    step.

    {b Error.} A scheme of order [p] has global error [O(h^p)] in the state.
    {b Cost.} One kick per element of [kick], one fewer once adjacent kicks
    merge in a march. {b Derivative.} A march is a composition of the flows, so
    its derivative is theirs. *)

type t
(** The type for splittings: the coefficients [a] of the kicks and [b] of the
    drifts. [a] has one more element than [b], each sums to [1], and the
    sequence is palindromic. *)

val leapfrog : t
(** [leapfrog] is [K(h/2) D(h) K(h/2)], Störmer–Verlet: order 2. *)

val mclachlan : t
(** [mclachlan] is [K(λh) D(h/2) K((1 − 2λ)h) D(h/2) K(λh)] with
    [λ = 0.1931833275037836], the two-stage scheme of least error (Omelyan,
    Mryglod and Folk, 2002): order 2. *)

val yoshida4 : t
(** [yoshida4] is Yoshida's (1990) composition of three leapfrogs: order 4. *)

val yoshida6 : t
(** [yoshida6] is Yoshida's (1990) solution A, seven leapfrogs: order 6. *)

val yoshida8 : t
(** [yoshida8] is Yoshida's (1990) solution D, fifteen leapfrogs: order 8. *)

val v : kick:float array -> drift:float array -> t
(** [v ~kick ~drift] is the splitting [K(kick.(0) h) D(drift.(0) h) …].

    Raises [Invalid_argument] unless every coefficient is finite, [drift] is not
    empty, [kick] has one more element than [drift], each sums to [1] within
    rounding, and the sequence is palindromic: each array equals its reverse. *)

type ('s, 'b) flow = (float, 'b) Nx.t -> 's -> 's
(** The type for flows: [flow h s] is the state [s] moved by its part over the
    duration [h]: a scalar, or in {!step} a tensor that broadcasts against the
    state's leaves. *)

val step :
  t -> kick:('s, 'b) flow -> drift:('s, 'b) flow -> (float, 'b) Nx.t -> 's -> 's
(** [step m ~kick ~drift h s] is [s] after one step [h] of [m]. A negative [h]
    steps back: [step m ~kick ~drift (−h) (step m ~kick ~drift h s)] is [s] up
    to rounding.

    [h] may hold one duration per batch of the state: an [h] of shape
    [[chains; 1]] against leaves of shape [[chains; d]] steps each chain by its
    own duration, as stepping each alone with its scalar would. The flows
    receive [h] times each coefficient, of [h]'s shape. *)

val march :
  's Nx.Ptree.t ->
  t ->
  steps:int ->
  kick:('s, 'b) flow ->
  drift:('s, 'b) flow ->
  at:(float, 'b) Nx.t ->
  's ->
  's
(** [march s m ~steps ~kick ~drift ~at s0] is the state at each time of [at],
    stacked on a new leading axis of each leaf, [s0] first at [at.(0)]. Each
    interval of [at] takes [steps] equal steps, its adjacent kicks merged into
    one: a {!leapfrog} interval makes [steps + 1] kicks. Reverse mode keeps one
    state per time of [at] and recomputes each interval while it reverses it.

    Raises [Invalid_argument] if [steps < 1], if [at] is not a non-empty 1-D
    tensor, or if [at] is not strictly monotone. *)
