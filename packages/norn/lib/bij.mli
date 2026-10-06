(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Bijectors: maps from unconstrained coordinates onto a support.

    A bijector maps a tensor of coordinates, any real numbers, to a value in its
    support and back. Samplers and optimisers move in coordinates; a model's
    density over its values is pulled back to them by adding the log-determinant
    of the map's Jacobian.

    A bijector acts on {e units} of its trailing axes: one element for the
    elementwise bijectors, one vector for {!simplex}, {!ordered} and
    {!sum_to_zero}, one matrix for {!cholesky_corr}. Leading axes are batch
    axes, and the log-determinant has one element per unit.

    A unit's log-determinant is that of the map onto its free components, the
    ones that determine the rest: all but the last for {!simplex} and
    {!sum_to_zero}, the strict lower triangle for {!cholesky_corr}.

    {b Totality.} For every finite coordinate, {!forward} lands in the open
    support at the dtype's precision: [exp] never returns [0] nor [inf], and
    [interval] never returns a bound. Beyond the points where the map saturates,
    it returns the saturated value, the nearest value of the support, with a
    log-determinant of [-inf], so the pulled-back density counts no value twice:
    its mass is the target's mass between the saturation points. *)

type 'f t = 'f Bijection.t
(** The type for bijectors over tensors of element type ['f]. *)

(** {1:constructors Constructors} *)

val identity : 'f t
(** [identity] maps [u] to [u], onto the reals. *)

val exp : 'f t
(** [exp] maps [u] to [exp u], onto [(0, inf)]. It saturates at the dtype's
    smallest positive normal number and at its largest finite one. *)

val greater : low:(float, 'f) Nx.t -> 'f t
(** [greater ~low] maps [u] to [low + exp u], onto [(low, inf)]. [low]
    broadcasts against the coordinates. *)

val interval : low:(float, 'f) Nx.t -> high:(float, 'f) Nx.t -> 'f t
(** [interval ~low ~high] maps [u] to [low + (high - low) sigmoid u], onto
    [(low, high)]. The bounds broadcast against the coordinates. *)

val affine : loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> 'f t
(** [affine ~loc ~scale] maps [u] to [loc + scale u], onto the reals. The
    parameters broadcast against the coordinates; [scale] must have no zero
    element. *)

val affine_tril : loc:(float, 'f) Nx.t -> scale_tril:(float, 'f) Nx.t -> 'f t
(** [affine_tril ~loc ~scale_tril] maps a vector [u] to [loc + L u], [L] the
    lower triangle of [scale_tril], onto the reals. [scale_tril]'s last two axes
    are a matrix, its leading axes batch axes; its diagonal must have no zero
    element. Its log-determinant is the sum of the logarithms of [|L(i,i)|]. *)

val simplex : 'f t
(** [simplex] maps a vector of [K - 1] coordinates to a vector of [K] positive
    components summing to one, by the isometric log-ratio: the coordinates are
    the value's centred logarithms in an orthonormal basis of the vectors
    summing to zero. *)

val ordered : 'f t
(** [ordered] maps a vector [u] of [K] coordinates to the strictly increasing
    vector [x] with [x0 = u0] and [xk = x(k-1) + exp uk]. *)

val cholesky_corr : 'f t
(** [cholesky_corr] maps a vector of [n (n - 1) / 2] coordinates to the Cholesky
    factor of an [n × n] correlation matrix. Coordinate [k] is the hyperbolic
    arctangent of the canonical partial correlation of the [k]-th element of the
    strict lower triangle, read row by row. *)

val sum_to_zero : 'f t
(** [sum_to_zero] maps a vector of [K - 1] coordinates to a vector of [K]
    components summing to zero, by an orthonormal basis of that subspace. *)

val compose : 'f t -> 'f t -> 'f t
(** [compose b c] maps [u] to [b]'s image of [c]'s image of [u]. Its
    log-determinant is the sum of theirs, the one with finer units summed to the
    other's units. *)

(** {1:maps Maps} *)

val forward : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t * (float, 'f) Nx.t
(** [forward b u] is [(x, ld)]: the value [x] that [b] maps the coordinates [u]
    to, and [log |det J|] of the map at [u], one element per unit. Both come
    from one computation. *)

val inverse : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t
(** [inverse b x] is the coordinates that [b] maps to [x], for [x] in the
    support. *)

val shape : 'f t -> int array -> int array
(** [shape b s] is the shape of the coordinates of a value of shape [s]: [s] for
    the elementwise bijectors and {!ordered}, [s] with its last axis one shorter
    for {!simplex} and {!sum_to_zero}, and [s] with its last two axes of [n]
    replaced by one of [n (n - 1) / 2] for {!cholesky_corr}.

    Raises [Invalid_argument] if [s] has fewer axes than a unit, or a last axis
    of length [0] where coordinates have one fewer element. *)

val pp : Format.formatter -> 'f t -> unit
(** [pp ppf b] formats [b]'s name, such as [exp] or [compose(exp, affine)]. *)
