(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stochastic differential equations.

    A problem is a drift [f t y], the deterministic part of the derivative, a
    diffusion [g t y], applied to a Brownian increment as [diffusion t y dw],
    linear in [dw], and a Brownian path. The method fixes the calculus: Itô or
    Stratonovich. *)

(** Brownian paths with their space–time Lévy area.

    A path is a virtual tree over [[t0, t1]] (Foster, Lyons and Oberhauser,
    2020; Jelinčič et al., 2024): each query descends [depth] levels of
    bisections, each drawing the midpoint's increments and areas from their
    distribution given the interval's, under a key that is a pure function of
    the path's key and the node. So a path is one function of time, whatever the
    queries: marches with different steps see one path. Steps finer than
    [(t1 − t0) / 2^depth] see, inside each finest interval, the path's mean
    given that interval's increment and area, a quadratic, and lose their stated
    order. *)
module Brownian : sig
  type 'b t
  (** The type for Brownian paths of dtype ['b]. *)

  val v :
    Nx.Rng.t ->
    (float, 'b) Nx.dtype ->
    shape:int array ->
    t0:float ->
    t1:float ->
    depth:int ->
    'b t
  (** [v key dtype ~shape ~t0 ~t1 ~depth] is a Brownian path of independent
      standard components of shape [shape] over [[t0, t1]], resolved to
      [(t1 − t0) / 2^depth].

      Raises [Invalid_argument] if [t0] or [t1] is not finite, if [t1 <= t0], if
      [depth] is not in [[0, 30]], or if a dimension of [shape] is negative. *)

  val increment :
    'b t ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t * (float, 'b) Nx.t
  (** [increment w s t] is [W t − W s] and the space–time Lévy area over
      [[s, t]],
      [H = (1 / (t − s)) ∫_s^t (W r − W s − (r − s) / (t − s) (W t − W s)) dr],
      each of the path's shape; [H] is zero when [s = t]. [s] and [t] are
      scalars. Increments over adjacent intervals compose by Chen's relation. A
      query costs [2 depth] normal draws of the path's shape at each end; a
      derivative in a time through the path has no meaning.

      Raises [Invalid_argument] through {!Nx.check} if [s] or [t] lies outside
      [[t0, t1]]. *)

  val ptree : (float, 'b) Nx.dtype -> 'b t Nx.Ptree.t
  (** [ptree dtype] is the structure of paths of [dtype]: the key at [key], the
      shape's dimensions reported at [shape], the interval's ends at [t0] and
      [t1], and the depth reported at [depth]. A path draws from its key, so a
      compiled function takes it as an argument of this structure; a path it
      captures is a constant, whose draws it refuses ({!Rune.Jit_error}). *)
end

(** {1:methods Methods} *)

type t
(** The type for methods. *)

val euler_maruyama : t
(** [euler_maruyama] is the Euler–Maruyama method, Itô, for any noise: strong
    order 1/2. One drift and one diffusion evaluation per step. *)

val milstein : t
(** [milstein] is the derivative-free Milstein method (Kloeden and Platen, 1992,
    §11.1), Itô: strong order 1 for diagonal noise, where component [i] of the
    diffusion depends on component [i] of the state only, and 1/2 otherwise. The
    Brownian path has the state's shape. One drift and three diffusion
    evaluations per step. *)

val sra1 : t
(** [sra1] is Rößler's (2010) SRA1, for additive noise, where the diffusion does
    not depend on the state: strong order 3/2, reading the Lévy area; 1/2
    otherwise. Two drift and two diffusion evaluations per step. *)

val reversible_heun : t
(** [reversible_heun] is the reversible Heun method (Kidger, Foster, Li and
    Lyons, 2021), Stratonovich: strong order 1/2, and 1 for additive noise. It
    carries a second state, so each step costs one drift and two diffusion
    evaluations. *)

(** {1:marches Marches} *)

val march :
  'y Nx.Ptree.t ->
  t ->
  steps:int ->
  drift:((float, 't) Nx.t -> 'y -> 'y) ->
  diffusion:((float, 't) Nx.t -> 'y -> (float, 't) Nx.t -> 'y) ->
  't Brownian.t ->
  at:(float, 't) Nx.t ->
  'y ->
  'y
(** [march y m ~steps ~drift ~diffusion w ~at y0] is the state at each time of
    [at], stacked on a new leading axis of each leaf, [y0] first, along the path
    [w]. Each interval of [at] takes [steps] equal steps. [drift t y] is the
    deterministic field; [diffusion t y dw] is [g t y] applied to [dw], of [w]'s
    shape, and must be linear in [dw]. Reverse mode keeps one state per time of
    [at] and recomputes each interval while it reverses it; the derivative is
    the composition's, in the initial state and every tracked value the drift
    and diffusion read. A compiled function takes [w] as an argument of
    {!Brownian.ptree}'s structure.

    Raises [Invalid_argument] if [steps < 1], if [at] is not a non-empty 1-D
    tensor, through {!Nx.check} if [at] is not strictly increasing or leaves
    [w]'s interval, and if the drift or the diffusion returns a value of another
    structure, dtype or shape than its state. *)
