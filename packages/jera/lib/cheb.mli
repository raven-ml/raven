(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Piecewise Chebyshev series held as coefficient tensors of shape
    [[pieces; degree + 1] @ rest]: on piece [p], at [u ∈ [−1, 1]], the value is
    [Σ_k c.(p).(k) T_k(u)]. {!Piecewise} and {!Grid} share these operations;
    [rest] holds a value's axes, and a grid's other axes. *)

(** The type for what a series is outside its breaks. *)
type extension =
  | Bounded  (** Outside the breaks, a finite point raises. *)
  | Hold  (** Outside, the value at the nearest end. *)
  | Polynomial  (** Outside, the end pieces' series continued. *)

type ('v, 'b) series = {
  s : 'v Nx.Ptree.t;  (** The values' structure. *)
  breaks : (float, 'b) Nx.t;
      (** [[pieces + 1]] breaks, non-decreasing, the first below the last. *)
  coefficients : 'v;  (** Leaves of shape [[pieces; degree + 1] @ value]. *)
  extension : extension;
}
(** The type for piecewise series with values of structure ['v]: the
    representation of {!Piecewise.t}, which an ODE's path builds too. *)

val locate :
  string ->
  extension ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (int64, Nx.int64_elt) Nx.t * (float, 'b) Nx.t
(** [locate fn e breaks x] is the piece and the local coordinate of each point
    of the 1-D [x]: a point on a break lies in the last piece of positive width
    that ends there, or the first piece at the first break. NaN gives NaN's
    coordinate. An infinite point raises [Invalid_argument] naming [fn] through
    {!Nx.check}, and under [Bounded] so does a finite point outside. *)

val nodes : int -> float array
(** [nodes n] is the [n + 1] Chebyshev points of the second kind on [[−1, 1]] in
    increasing order, ends included: [[|0.|]] for [n = 0]. *)

val fit : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [fit v] is the coefficients of the series of degree [n] that interpolates
    the values [v] at the {!nodes} [n] of each piece, [v] of shape
    [[pieces; n + 1] @ rest]. *)

val hermite :
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [hermite y0 y1 d0 d1] is the cubic series of each piece whose values at
    [u = −1] and [u = 1] are [y0] and [y1] and whose derivatives in [u] there
    are [d0] and [d1], each of shape [[pieces] @ rest]. *)

val clenshaw : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [clenshaw u g] is the series [g] of shape [[q; n + 1] @ rest] at [u] of
    shape [[q]], by Clenshaw's recurrence: of shape [[q] @ rest]. *)

val derivative : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [derivative widths c] is the derivative in [x] of the series [c] on pieces
    of [widths], of degree one less, and of degree [0] for degree [0]. *)

val integral : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [integral widths c] is the antiderivative of [c] in [x], of degree one more,
    continuous across pieces and zero at the first piece's start. *)
