(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Models as generative functions.

    A model is one OCaml function that draws every random variable with
    {!sample} and returns them all: the latent ones as a value of the caller's
    structure ['p], the observed ones as a value of ['y]. Every inference task
    interprets that function: scoring it gives densities over coordinates
    ({!log_density}), running it forward simulates data ({!simulate}), and
    answering its latent sites with draws maps draws back to values
    ({!constrain}).

    {[
    type 'a schools = { mu : 'a; tau : 'a; theta : 'a }

    let eight_schools ~sigma =
      Norn_model.v Nx.float64 schools Nx.Ptree.tensor @@ fun () ->
      let mu = Norn_model.sample (D.normal ~loc:(f64 0.) ~scale:(f64 5.)) in
      let tau = Norn_model.sample (D.half_cauchy ~scale:(f64 5.)) in
      let theta =
        Norn_model.sample (D.iid [| 8 |] (D.normal ~loc:mu ~scale:tau))
      in
      let y = Norn_model.sample (D.normal ~loc:theta ~scale:sigma) in
      ({ mu; tau; theta }, y)
    ]}

    {b The result rule.} A {e site} is one call of {!sample}. Its value must
    reach the result unchanged: the interpreter hands out a fresh tensor at each
    site and finds where it landed, by physical identity, when the function
    returns. A site is named by the path of its value in that result, [theta]
    above; a site at the root of ['p] or ['y] is named [p] or [y]. Every
    interpreter checks the rule on every run, so a model that returns
    [Nx.exp tau] where it sampled [tau] raises naming the site.

    {b Static models.} The sites, their order, families and shapes do not depend
    on random values, so draws of all sites stack into one ['p] and one compiled
    program serves every run. Every interpreter checks each site against what
    {!v} learned. A [sample] inside a loop body or inside a map within the
    function never reaches the result unchanged, and raises naming the site.

    {b Interpreters} install their handler when applied, so they run inside
    [Rune.grad], [Rune.vmap] and [Rune.jit] and leave nothing in a compiled
    program. *)

(** {1:models Models} *)

type ('p, 'y, 'f) t
(** The type for models with latent values ['p], observed values ['y] and log
    densities of element type ['f]. *)

val v :
  (float, 'f) Nx.dtype ->
  'p Nx.Ptree.t ->
  'y Nx.Ptree.t ->
  (unit -> 'p * 'y) ->
  ('p, 'y, 'f) t
(** [v dtype latent observed gen] is the model [gen] describes. [gen] performs
    {!sample} once per random variable and returns every sampled value
    unchanged: the latent ones in the first component, of structure [latent],
    the observed ones in the second, of structure [observed]. Each site's log
    density is cast to [dtype].

    [v] runs [gen] once, answering each continuous site with its bijector's
    image of zero coordinates, and checks shapes but no domain. An
    [Invalid_argument] or [Failure] that [gen] raises is raised again as
    [Norn_model.v: the model raised after N sites (...): message], naming the
    sites learned so far; any other exception propagates unchanged, with its
    backtrace. Raises [Invalid_argument] naming the site if a sampled value is
    missing from the result or appears twice, if a result tensor is not a
    sampled value, or if a latent site is discrete. *)

(** {1:effects Effects} *)

val sample : ('x, 'f) Norn.Dist.t -> 'x
(** [sample d] is a value of [d] at a new site. Only an interpreter answers it;
    outside one it raises [Effect.Unhandled]. *)

val factor : (float, 'f) Nx.t -> unit
(** [factor l] adds [l] to the log likelihood. Axis 0 of [l] indexes its points;
    a scalar is one point. Interpreters that simulate ignore it, so its data
    take no part in {!simulate} and {!predict}. *)

(** {1:coords Coordinates} *)

type 'p coords = private 'p
(** The type for unconstrained latent values: one tensor per site, sized by its
    degrees of freedom (a simplex of [K] components has [K - 1]). A site's
    coordinates are its value under its distribution's bijector
    ({!Norn.Dist.coords}), or the one {!reparam} gives. [(c :> 'p)] reads
    coordinates as a structure. *)

val coords : ('p, 'y, 'f) t -> 'p coords Nx.Ptree.t
(** [coords m] is the structure of [m]'s coordinates: [m]'s latent structure. *)

val constrain : ('p, 'y, 'f) t -> 'p coords -> 'p
(** [constrain m c] is the latent values at the coordinates [c]. It runs the
    model, so a bijector that depends on other sites sees their values. *)

val unconstrain : ('p, 'y, 'f) t -> 'p -> 'p coords
(** [unconstrain m p] is the coordinates of the latent values [p]. *)

(** {1:batched Batched interpreters}

    These take and give positions: values with a leading chain axis. Each maps
    the single-instance interpretation with [Rune.vmap] over that axis, so the
    rows of a density are independent by construction. *)

val log_density : ('p, 'y, 'f) t -> 'y -> ('p coords, 'f) Norn.density
(** [log_density m y] is the log density of the coordinates conditioned on [y]:
    the prior of the latent values, each site's log-determinant and the
    likelihood of [y], factors included.

    Raises [Invalid_argument] at once if [y] does not match the model, naming
    the first site that differs, as in
    [Norn_model.log_density: the observations do not match the model: at counts,
     data of shape [111], the site's shape [112]]. A parameter outside its
    domain raises naming the site and the chain, as in
    [Norn_model.log_density: site theta, chain 17: normal: scale at [3] is -1,
     not in (0, inf)], except where the log density accumulated before the site
    is already [-inf]: there the site's term is [-inf]. A term that is NaN or
    [+inf] raises naming the site and the chain. *)

val log_prior : ('p, 'y, 'f) t -> ('p coords, 'f) Norn.density
(** [log_prior m] is the prior density of the coordinates: the latent sites'
    terms and log-determinants. Observed sites are answered with zeros: a latent
    site whose distribution depends on an observed value has its prior evaluated
    at zero data. *)

val log_likelihood : ('p, 'y, 'f) t -> 'y -> ('p coords, 'f) Norn.density
(** [log_likelihood m y] is the log likelihood of [y], factors included:
    [log_density m y c = log_prior m c + log_likelihood m y c].

    Raises as {!log_density} does. *)

val from_prior : ('p, 'y, 'f) t -> n:int -> Nx.Rng.t -> 'p coords
(** [from_prior m ~n k] is [n] prior draws of the coordinates, stacked on a
    leading axis. Draw [i] has the key of row [i] of [Nx.Rng.split_batch ~n k],
    so it does not depend on [n].

    Raises [Invalid_argument] if [n < 1]. *)

(** Initialisation strategies. *)
module Init : sig
  type 'p t
  (** The type for strategies over latent values ['p]. *)

  val uniform : 'p t
  (** [uniform] draws each coordinate uniformly in [(-2, 2)]. *)

  val prior : 'p t
  (** [prior] draws coordinates from the prior. *)

  val near : 'p -> 'p t
  (** [near p] is the coordinates of the values [p], each moved uniformly in
      [(-0.1, 0.1)]. *)
end

val init :
  ?from:'p Init.t -> ('p, 'y, 'f) t -> 'y -> chains:int -> Nx.Rng.t -> 'p coords
(** [init ?from m y ~chains k] is a start of [chains] chains with a finite log
    density. Chain [i] has the key of row [i] of
    [Nx.Rng.split_batch ~n:chains k], so its start does not depend on [chains];
    it draws 100 candidates from [from] (default {!Init.uniform}) and keeps the
    first whose log density is finite.

    Raises [Invalid_argument] if [chains < 1], and if a chain has no finite
    candidate, naming the first site whose term is not finite at the last
    candidate and listing every term, as in
    [Norn_model.init: no finite log density in 100 candidates; at the last, site
     counts has log density -inf: element 17 of its factors is outside the
     support of poisson, {0, 1, 2, ...}; the terms are rate -1.2, counts -inf].
*)

(** {1:single Single-instance interpreters}

    These take values with no chain axis. They reach draws through
    {!Norn.Draws.map} and {!Norn.Draws.simulate}. *)

val log_joint : ('p, 'y, 'f) t -> 'y -> 'p -> (float, 'f) Nx.t
(** [log_joint m y p] is the log density of the values [p] and the observations
    [y], with no log-determinant: the argument's type says which. *)

val pointwise : ('p, 'y, 'f) t -> 'y -> 'p -> (float, 'f) Nx.t
(** [pointwise m y p] is the log likelihood of every point, in the order the
    observed sites and factors are performed: the points of a site are axis 0 of
    its factors ({!Norn.Dist.factors}), and a scalar is one point. It sums to
    the likelihood that {!log_likelihood} gives at [unconstrain m p]. It reads
    values, so a model rebuilt on new covariates takes a posterior's values. *)

val simulate : ('p, 'y, 'f) t -> Nx.Rng.t -> 'p * 'y
(** [simulate m k] is a draw of every site from the model: latent values and
    data. Site [i] draws from [Nx.Rng.fold_in k i]. *)

val predict : ('p, 'y, 'f) t -> Nx.Rng.t -> 'p -> 'y
(** [predict m k p] is data drawn given the latent values [p]. *)

(** {1:transformations Transformations of a model} *)

val reparam :
  ('p -> (float, 'f) Nx.t) ->
  (((float, 'f) Nx.t, 'f) Norn.Dist.t -> 'f Norn.Bij.t) ->
  ('p, 'y, 'f) t ->
  ('p, 'y, 'f) t
(** [reparam sel b m] is [m] whose site at [sel] has the coordinates its value
    has under [b d], [d] being the site's distribution on each run. The density
    of the values is unchanged.

    [sel] is a selector: a projection of the latent values, [fun p -> p.theta].
    It is applied to fresh tensors and must return one of them. Raises
    [Invalid_argument] if it computes, or if the site's dtype is not [m]'s. *)

val noncentre : ('p -> (float, 'f) Nx.t) -> ('p, 'y, 'f) t -> ('p, 'y, 'f) t
(** [noncentre sel m] is [reparam sel Norn.Dist.standardize m]. *)

val fix :
  ('p -> ('a, 'b) Nx.t) -> ('a, 'b) Nx.t -> ('p, 'y, 'f) t -> ('p, 'y, 'f) t
(** [fix sel v m] is [m] whose site at [sel] is the constant [v], with no
    density term. Its coordinates have no element: shape [[0]].

    Raises [Invalid_argument] if [sel] computes, or if [v]'s dtype or shape is
    not the site's. *)

(** {1:sites Sites} *)

val terms :
  ('p, 'y, 'f) t ->
  'y ->
  'p coords ->
  (string * (float, 'f) Nx.t) list * (float, 'f) Nx.t
(** [terms m y c] is each site's term of [log_density m y] at the position [c],
    of shape [[chain]], by name in the order the model performs them, then the
    factors' sum, zeros when there is none. *)

val pp : Format.formatter -> ('p, 'y, 'f) t -> unit
(** [pp ppf m] formats [m]'s site table: each site's name, role, family, shape,
    points and coordinates. *)
