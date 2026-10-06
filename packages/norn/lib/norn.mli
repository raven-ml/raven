(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Probabilistic inference.

    Norn turns a log density over a structure of the caller's own type into
    draws and their diagnostics. A {e position} is a value of a structure ['u]
    ({!Nx.Ptree.t}) whose every tensor has a leading axis, the {e chain axis},
    of one length [c]; one chain is [c = 1]. A {!type-density} maps a position
    to the normalised log density of each chain.

    Every algorithm takes the structure, the density, then the key. Gradient
    samplers differentiate the density with {!Rune}; a density with a gradient
    of its own states it with {!with_gradient}. Samplers move in unconstrained
    coordinates that a bijector ({!Bij}) maps onto a value's support. *)

(** {1:densities Densities} *)

type ('u, 'f) density = 'u -> (float, 'f) Nx.t
(** The type for log densities over positions of ['u]. For a position of [c]
    chains the result has shape [[c]], each element finite or [-inf], and its
    row [i] depends only on row [i] of the position. *)

val with_gradient :
  'u Nx.Ptree.t -> ('u -> (float, 'f) Nx.t * 'u) -> ('u, 'f) density
(** [with_gradient u f] is the density [fun x -> fst (f x)] whose gradient, row
    by row, is [snd (f x)]: a tangent [dx] moves chain [i]'s log density by the
    inner product of row [i] of the gradient with row [i] of [dx], over every
    float tensor. [f] receives values with no derivative attached, so code
    outside nx, such as an adjoint solver, may read them. {!Rune.grad} and
    {!Rune.jvp} both follow it. *)

(** {1:modules Modules} *)

module Bij = Bij
