(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Slice sampling along a direction, for many walkers in lock step.

    Each walker draws a level [Exp (1)] below its log density and samples
    uniformly from the slice of the line through it along its direction above
    that level (Neal 2003): a bracket of {!width} placed at random around it
    doubles toward a random side while an end lies in the slice, at most as many
    times as the dtype's significand has bits, then shrinks toward the walker
    until a point lies in the slice and passes Neal's acceptance check. A
    bracket narrower than the dtype's resolution of {!width} leaves the walker
    in place. *)

val width : float
(** [width] is the bracket's initial width, [4 sqrt (2 / π)]: the expected width
    of a standard normal's slice at a point of it. A direction of unit length in
    a Gaussian's whitened coordinates gives a bracket that fits a Gaussian
    target. *)

val move :
  string ->
  'u Nx.Ptree.t ->
  'a Nx.Ptree.t ->
  (Nx.Rng.t -> 'u -> (float, 'f) Nx.t * 'a) ->
  Nx.Rng.t ->
  direction:'u ->
  'u ->
  (float, 'f) Nx.t ->
  'a ->
  'u * (float, 'f) Nx.t * 'a * Nx.int32_t
(** [move context u a eval keys ~direction x lp aux] moves the walkers from [x],
    at log density [lp] with the values [aux], along [direction], and returns
    their positions, log densities, values and each walker's number of
    evaluations. [eval ks y] is the log density at the position [y] of every
    walker and the values carried with it, [ks] a key per walker; one call per
    trip evaluates every walker, a walker that has finished at its last
    evaluated position. [keys] holds a key per walker.

    Raises through {!Nx.check} if [eval] is NaN or [+inf] at a walker. *)

val unit_directions : 'u Nx.Ptree.t -> Nx.Rng.t -> (float, 'f) Nx.t -> 'u -> 'u
(** [unit_directions u keys lp like] is a direction per row of [like], uniform
    on the unit sphere of its float elements, row [i]'s drawn from [keys.(i)];
    norms are taken at [lp]'s dtype. *)

val hit_and_run :
  string ->
  'u Nx.Ptree.t ->
  'a Nx.Ptree.t ->
  (Nx.Rng.t -> 'u -> (float, 'f) Nx.t * 'a) ->
  Nx.Rng.t ->
  ('u, 'f) Gaussian.t ->
  'u ->
  (float, 'f) Nx.t ->
  'a ->
  'u * (float, 'f) Nx.t * 'a * Nx.int32_t
(** [hit_and_run context u a eval keys g x lp aux] is {!move} along a direction
    per walker of unit length in [g]'s whitened coordinates, drawn from
    [fold_in keys.(i) 5]. *)
