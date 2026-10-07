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
    of its own states it with {!with_gradient}. Continuous values live in
    supports ({!Support}), and samplers move in unconstrained coordinates that a
    bijector ({!Bij}) maps onto them. *)

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

module Support : sig
  (** Sets of values a distribution gives positive density.

      A support names the set; it carries the bounds that do not depend on a
      tensor. Each continuous support has one bijector ({!Norn.Bij}) whose
      coordinates cover it. *)

  (** The type for supports. *)
  type t =
    | Real  (** The real numbers. *)
    | Greater of float  (** The reals above a bound, exclusive. *)
    | Interval of float * float  (** The reals between two bounds, exclusive. *)
    | Simplex of int  (** Vectors of [n] positive components that sum to one. *)
    | Ordered  (** Vectors whose components strictly increase. *)
    | Correlation_cholesky of int
        (** Lower-triangular [n × n] matrices with a positive diagonal and rows
            of unit norm: Cholesky factors of correlation matrices. *)
    | Sum_to_zero  (** Vectors whose components sum to zero. *)
    | Integers_from of int  (** The integers from a bound, inclusive. *)
    | Integer_interval of int * int
        (** The integers between two bounds, inclusive. *)
    | Boolean  (** [false] and [true]. *)

  val equal : t -> t -> bool
  (** [equal s s'] is [true] iff [s] and [s'] are the same constructor with
      equal arguments, floats compared with [Float.equal]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf s] formats [s] as a set: [(-inf, inf)], [(0, inf)], [(0, 1)],
      [{0, 1, 2, ...}], [{0, 1, ..., 10}], [{false, true}], or a phrase such as
      [simplex of 3]. *)
end

module Bij : sig
  (** Bijectors: maps from unconstrained coordinates onto a support.

      A bijector maps a tensor of coordinates, any real numbers, to a value in
      its support and back. Samplers and optimisers move in coordinates; a
      model's density over its values is pulled back to them by adding the
      log-determinant of the map's Jacobian.

      A bijector acts on {e units} of its trailing axes: one element for the
      elementwise bijectors, one vector for {!simplex}, {!ordered} and
      {!sum_to_zero}, one matrix for {!cholesky_corr}. Leading axes are batch
      axes, and the log-determinant has one element per unit.

      A unit's log-determinant is that of the map onto its free components, the
      ones that determine the rest: all but the last for {!simplex} and
      {!sum_to_zero}, the strict lower triangle for {!cholesky_corr}.

      {b Totality.} For every finite coordinate, {!forward} lands in the open
      support at the dtype's precision: [exp] never returns [0] nor [inf], and
      [interval] never returns a bound. Beyond the points where the map
      saturates, it returns the saturated value, the nearest value of the
      support, with a log-determinant of [-inf], so the pulled-back density
      counts no value twice: its mass is the target's mass between the
      saturation points. *)

  type 'f t
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
      lower triangle of [scale_tril], onto the reals. [scale_tril]'s last two
      axes are a matrix, its leading axes batch axes; its diagonal must have no
      zero element. Its log-determinant is the sum of the logarithms of
      [|L(i,i)|]. *)

  val simplex : 'f t
  (** [simplex] maps a vector of [K - 1] coordinates to a vector of [K] positive
      components summing to one, by the isometric log-ratio: the coordinates are
      the value's centred logarithms in an orthonormal basis of the vectors
      summing to zero. *)

  val ordered : 'f t
  (** [ordered] maps a vector [u] of [K] coordinates to the strictly increasing
      vector [x] with [x0 = u0] and [xk = x(k-1) + exp uk]. *)

  val cholesky_corr : 'f t
  (** [cholesky_corr] maps a vector of [n (n - 1) / 2] coordinates to the
      Cholesky factor of an [n × n] correlation matrix. Coordinate [k] is the
      hyperbolic arctangent of the canonical partial correlation of the [k]-th
      element of the strict lower triangle, read row by row. *)

  val sum_to_zero : 'f t
  (** [sum_to_zero] maps a vector of [K - 1] coordinates to a vector of [K]
      components summing to zero, by an orthonormal basis of that subspace. *)

  val compose : 'f t -> 'f t -> 'f t
  (** [compose b c] maps [u] to [b]'s image of [c]'s image of [u]. Its
      log-determinant is the sum of theirs, the one with finer units summed to
      the other's units. *)

  (** {1:maps Maps} *)

  val forward : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t * (float, 'f) Nx.t
  (** [forward b u] is [(x, ld)]: the value [x] that [b] maps the coordinates
      [u] to, and [log |det J|] of the map at [u], one element per unit. Both
      come from one computation. *)

  val inverse : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t
  (** [inverse b x] is the coordinates that [b] maps to [x], for [x] in the
      support. *)

  val shape : 'f t -> int array -> int array
  (** [shape b s] is the shape of the coordinates of a value of shape [s]: [s]
      for the elementwise bijectors and {!ordered}, [s] with its last axis one
      shorter for {!simplex} and {!sum_to_zero}, and [s] with its last two axes
      of [n] replaced by one of [n (n - 1) / 2] for {!cholesky_corr}.

      Raises [Invalid_argument] if [s] has fewer axes than a unit, or a last
      axis of length [0] where coordinates have one fewer element. *)

  val pp : Format.formatter -> 'f t -> unit
  (** [pp ppf b] formats [b]'s name, such as [exp] or [compose(exp, affine)]. *)
end

module Stats : sig
  (** Statistics of Hamiltonian transitions.

      Each field holds one element per chain, with the axes of the chains it
      describes: a sampler's state holds [[chain]], and its draws
      [[chain; draw]]. *)

  type 'f t = {
    lp : (float, 'f) Nx.t;  (** The log density at the transition's end. *)
    acceptance : (float, 'f) Nx.t;
        (** The acceptance statistic warmup aims for, of
            [min (1, exp (H0 - H))], [H0] the Hamiltonian after the momentum
            draw: its mean over the trajectory for {!Norn.Nuts}, its value at
            the trajectory's end, the Metropolis probability, for {!Norn.Hmc}.
        *)
    step_size : (float, 'f) Nx.t;  (** The step size the transition took. *)
    n_steps : Nx.int32_t;  (** The leapfrog steps it took. *)
    diverging : Nx.bool_t;
        (** Whether the energy error exceeded 1000 or a position was not finite.
        *)
    saturated : Nx.bool_t;
        (** Whether the trajectory reached its maximum length before it turned.
        *)
    energy : (float, 'f) Nx.t;
        (** The Hamiltonian after the momentum draw, [H0]. *)
  }
  (** The type for statistics of transitions over element type ['f]. *)

  val ptree : (float, 'f) Nx.dtype -> 'f t Nx.Ptree.t
  (** [ptree dtype] is the structure of statistics at [dtype], with tensors at
      [lp], [acceptance], [step_size], [n_steps], [diverging], [saturated] and
      [energy]. *)
end

module Draws : sig
  (** Equal-weight draws from chains.

      Draws of a structure ['u] are a value of ['u] whose every tensor has two
      leading axes, [[chain; draw]], of one length each across tensors. A
      sampler returns them, and {!v} checks a value for them. Chain diagnostics
      ({!Norn.Diag}) take draws, so they apply to equal-weight chains only. *)

  type 'u t = private 'u
  (** The type for draws of ['u]. [(d :> 'u)] reads draws as a value. *)

  val v : 'u Nx.Ptree.t -> 'u -> 'u t
  (** [v u x] is [x] as draws.

      Raises [Invalid_argument] if a tensor of [x] has fewer than two axes, or
      if two tensors differ in the length of one of their first two axes, naming
      the path. *)

  val ptree : 'u Nx.Ptree.t -> 'u t Nx.Ptree.t
  (** [ptree u] is the structure of draws of [u]: [u]'s, with the same paths. *)

  val map : 'a Nx.Ptree.t -> 'b Nx.Ptree.t -> ('a -> 'b) -> 'a t -> 'b t
  (** [map a b f d] is [f] applied to every draw of [d], a value of ['a] without
      the two leading axes, the results stacked on them. [f] runs under
      {!Rune.val-vmap}, once for all draws. *)

  val simulate :
    'a Nx.Ptree.t ->
    'b Nx.Ptree.t ->
    (Nx.Rng.t -> 'a -> 'b) ->
    Nx.Rng.t ->
    'a t ->
    'b t
  (** [simulate a b f k d] is [map a b] of [f k_ij] over [d], where the draw [j]
      of chain [i] has the key [Nx.Rng.fold_in k (i * n + j)], [n] being the
      number of draws per chain. *)

  val append : 'u Nx.Ptree.t -> 'u t -> 'u t -> 'u t
  (** [append u d d'] is the draws of [d] followed, in every chain, by those of
      [d'].

      Raises [Invalid_argument] if [d] and [d'] differ in their visits
      ({!Nx.Ptree.visits}) or in their number of chains. *)

  val thin : 'u Nx.Ptree.t -> every:int -> 'u t -> 'u t
  (** [thin u ~every d] is every [every]-th draw of each chain of [d], from the
      first.

      Raises [Invalid_argument] if [every < 1]. *)
end

module Diag : sig
  (** Diagnostics of draws.

      A diagnostic of draws of ['u] is a value of ['u] holding one number per
      element, at the element's dtype: [(rhat schools post).tau] is the R-hat of
      [tau]. Diagnostics read their draws on the host and take float tensors.

      The chain diagnostics follow Vehtari, Gelman, Simpson, Carpenter and
      Bürkner (2021). Each splits every chain into its two halves, dropping the
      middle draw of an odd count, and ranks the pooled draws, ties at their
      average rank [r]; the normal scores [ndtri ((r - 3/8) / (S + 1/4))], [S]
      the number of draws, make them insensitive to the draws' scale and tails.
      An element whose draws are all equal, or not all finite, or a chain of
      fewer than four draws, has no diagnostic: it is NaN.

      Raises [Invalid_argument] naming the path for a tensor that is not of a
      float dtype. *)

  (** {1:convergence Convergence} *)

  val rhat : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
  (** [rhat u d] is the rank-normalised split R-hat of each element: the larger
      of the R-hat of the split chains' normal scores (bulk) and of the normal
      scores of their distance to the median (tails). Near [1] the chains agree;
      above [1.01] they have not mixed. *)

  val nested_rhat : 'u Nx.Ptree.t -> superchains:int -> 'u Draws.t -> 'u
  (** [nested_rhat u ~superchains d] is the rank-normalised nested R-hat
      (Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari and Gelman 2024) of
      each element, for many short chains: the chains form [superchains]
      consecutive groups of equal size, each started from one point, and the
      statistic compares the variance between groups with the variance within
      them, chains split and scored as for {!rhat}.

      Raises [Invalid_argument] if [superchains < 2] or if it does not divide
      the number of chains. *)

  val ess_bulk : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
  (** [ess_bulk u d] is the effective sample size of each element's normal
      scores, by Geyer's initial monotone sequence over the autocorrelations of
      the split chains. It measures how well the draws estimate the centre of
      the distribution. *)

  val ess_tail : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
  (** [ess_tail u d] is the smaller of the effective sample sizes of the
      indicators of each element being at most its 5% and its 95% quantile. It
      measures how well the draws estimate the tails. *)

  val mcse_mean : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
  (** [mcse_mean u d] is the Monte Carlo standard error of each element's mean:
      its standard deviation over the square root of the effective sample size
      of the split chains' draws. *)

  (** {1:hamiltonian Hamiltonian transitions} *)

  val ebfmi : 'f Stats.t Draws.t -> (float, 'f) Nx.t
  (** [ebfmi s] is each chain's energy Bayesian fraction of missing information,
      shape [[chain]]: the mean squared change of the energy between draws over
      the energy's variance. Below [0.3] the momentum resampling explores the
      energy poorly.

      Raises [Invalid_argument] if the chains have fewer than two draws. *)

  val divergent_shift : 'u Nx.Ptree.t -> 'f Stats.t Draws.t -> 'u Draws.t -> 'u
  (** [divergent_shift u s d] is, for each element, the mean of the draws whose
      transition diverged minus the mean of all draws, in standard deviations of
      all draws. Where divergences gather, it is far from [0]; with no
      divergence it is NaN.

      Raises [Invalid_argument] naming both shapes if [s] and [d] differ in
      their chains or draws. *)

  (** {1:calibration Calibration} *)

  val rank : 'u Nx.Ptree.t -> truth:'u -> 'u Draws.t -> 'u
  (** [rank u ~truth d] is, for each element, the number of draws of [d] below
      the element of [truth]: from [0] to the number of draws. With [truth]
      drawn from the prior, data simulated from it and [d] exact posterior
      draws, each rank is uniform (Talts, Betancourt, Simpson, Vehtari and
      Gelman 2018).

      Raises [Invalid_argument] naming the path if there are more draws than the
      element's dtype holds exact integers. *)

  val rank_uniformity : 'u Nx.Ptree.t -> draws:int -> 'u list -> 'u
  (** [rank_uniformity u ~draws ranks] is, for each element, the simultaneous
      p-value of its ranks against the uniform distribution on [0], ..., [draws]
      (Säilynoja, Bürkner and Vehtari 2022), computed exactly, with no
      simulation: the probability, under uniformity, that the count of ranks
      below some point [i / (draws + 1)] strays from its expectation at least as
      far, in pointwise p-value, as the ranks' farthest count. Below [alpha]
      rejects uniformity at level [alpha].

      Raises [Invalid_argument] if [ranks] is empty, if [draws < 1], or if a
      rank is not an integer in [0], ..., [draws]. *)
end

module Summary : sig
  (** Summaries of draws, with their findings.

      A summary has a row per element of the draws, such as [theta[3]], and a
      column per statistic: the mean, the standard deviation, the 5%, 50% and
      95% quantiles, the Monte Carlo standard error of the mean, the bulk and
      tail effective sample sizes and R-hat ({!Norn.Diag}). Its findings are the
      problems those numbers and the transitions' statistics show, as data:
      there are no warnings.

      {[
      Format.printf "%a@." Norn.Summary.pp (Norn.Summary.v schools ~stats post)
      ]} *)

  type t
  (** The type for summaries. *)

  (** The type for findings. A finding about an element names it by the path of
      its tensor and its index in the tensor. *)
  type finding =
    | Rhat_high of { path : Nx.Ptree.Path.t; index : int array; rhat : float }
        (** R-hat above [1.01]: split R-hat without superchains, nested R-hat
            with them. *)
    | Ess_low of {
        path : Nx.Ptree.Path.t;
        index : int array;
        bulk : float;
        tail : float;
      }
        (** A bulk or tail effective sample size below [100] per chain, or below
            [400] in total with superchains. *)
    | Not_finite of { path : Nx.Ptree.Path.t; index : int array; count : int }
        (** [count] of the element's draws are NaN or infinite: it has no
            diagnostic. *)
    | Constant of { path : Nx.Ptree.Path.t; index : int array }
        (** Every draw of the element is equal: it has no R-hat nor effective
            size. *)
    | Divergent of { count : int; total : int; regions : region list }
        (** [count] of the [total] transitions diverged. *)
    | Saturated of { count : int; total : int }
        (** [count] of the [total] transitions reached their maximum length
            before they turned. *)
    | Ebfmi_low of { chain : int; ebfmi : float }
        (** A chain's E-BFMI is below [0.3]. *)

  and region = { path : Nx.Ptree.Path.t; index : int array; shift : float }
  (** The type for where divergences gather: an element whose draws from
      transitions that diverged have a mean [shift] standard deviations from its
      mean, by more than one ({!Norn.Diag.divergent_shift}). *)

  val v :
    'u Nx.Ptree.t ->
    ?stats:'f Stats.t Draws.t ->
    ?superchains:int ->
    'u Draws.t ->
    t
  (** [v u ?stats ?superchains d] summarises the draws [d] of [u]. [stats], the
      transitions' statistics, adds the findings on divergences, depth and
      E-BFMI. [superchains] takes the chains as that many consecutive groups for
      {!Norn.Diag.nested_rhat}, which then fills the [rhat] column.

      The statistics are computed at float64, whatever the draws' dtype.

      Raises [Invalid_argument] if the chains have fewer than 4 draws, if
      [superchains] does not divide the chains, and as {!Norn.Diag} does for a
      tensor that is not of a float dtype. *)

  val concat : t list -> t
  (** [concat ss] is the rows and findings of [ss], in order: derived quantities
      summarised beside parameters. *)

  val findings : t -> finding list
  (** [findings s] is [s]'s findings, the elements' in row order, then the
      transitions'. *)

  val labels : t -> string array
  (** [labels s] is [s]'s row labels: an element's path, then its index in
      brackets for a tensor that is not a scalar, as [theta[3]] or [b[1,0]]. At
      the root path, a tensor is labelled by its index alone, and a scalar
      [value]. *)

  val columns : t -> (string * Nx.float64_t) list
  (** [columns s] is [s]'s columns, each a vector with one element per row:
      [mean], [sd], [q5], [median], [q95], [mcse], [ess_bulk], [ess_tail] and
      [rhat]. *)

  val pp_finding : Format.formatter -> finding -> unit
  (** [pp_finding ppf f] formats [f] as a sentence. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf s] formats [s] as a table, then its findings, one per line. *)
end

module Dist : sig
  (** Probability distributions over tensors.

      A distribution describes one whole tensor value: its log density is a
      scalar, and {!iid} adds independent copies. Each family has one
      parameterisation, named by labels; parameters are tensors and broadcast.
      Event axes belong to the family: {!normal} has none, {!dirichlet}'s and
      {!mvn}'s value is a vector along the last axis.

      {[
      let prior = Norn.Dist.normal ~loc:(f64 0.) ~scale:(f64 5.)
      let lp = Norn.Dist.log_density prior (f64 1.2)
      ]}

      {b Validation on use.} Constructors are total: they check shapes, raising
      [Invalid_argument] naming the function, and record their parameters'
      domain checks. Every eliminator runs them with {!Nx.check}, eagerly at
      once and in a compiled function when the call returns: a parameter outside
      its domain raises [Invalid_argument] naming its index and value, as in
      [Norn.Dist.log_density: normal: scale at [3] is -1, not in (0, inf)]. A
      value outside the support has density [-inf] and raises nothing. {!valid}
      reads the same checks as data, for a density that must be total:

      {[
      let ok = Norn.Dist.valid d in
      let d = Norn.Dist.check ~unless:(Nx.scalar Nx.bool true) "" d in
      Nx.where ok (Norn.Dist.log_density d x) (Nx.scalar dt Float.neg_infinity)
      ]} *)

  type ('x, 'f) t
  (** The type for distributions over values ['x] whose log densities have
      element type ['f]. ['x] is [(float, 'f) Nx.t] for continuous families,
      {!Nx.int32_t} for counts, {!Nx.int64_t} for categories and {!Nx.bool_t}
      for {!bernoulli}. *)

  (** The type for the values of a distribution: a witness of ['x]. *)
  type ('x, 'f) kind =
    | Continuous : ((float, 'f) Nx.t, 'f) kind  (** Float tensors. *)
    | Counts : (Nx.int32_t, 'f) kind  (** Counts, from [0]. *)
    | Categories : (Nx.int64_t, 'f) kind  (** Category indices, from [0]. *)
    | Booleans : (Nx.bool_t, 'f) kind  (** Booleans. *)

  (** {1:continuous Continuous families} *)

  val normal :
    loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [normal ~loc ~scale] has density [exp (-z²/2) / (scale sqrt (2π))],
      [z = (x - loc) / scale]. [scale] is in [(0, inf)] and [loc] finite. *)

  val half_normal : scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [half_normal ~scale] is the distribution of [|x|], [x] normal with
      location [0], on [(0, inf)]. *)

  val lognormal :
    loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [lognormal ~loc ~scale] is the distribution of [exp x], [x] normal. *)

  val student_t :
    df:(float, 'f) Nx.t ->
    loc:(float, 'f) Nx.t ->
    scale:(float, 'f) Nx.t ->
    ((float, 'f) Nx.t, 'f) t
  (** [student_t ~df ~loc ~scale] is Student's t with [df] degrees of freedom,
      in [(0, inf)], located and scaled. *)

  val cauchy :
    loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [cauchy ~loc ~scale] has density [1 / (π scale (1 + z²))]. *)

  val half_cauchy : scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [half_cauchy ~scale] is the distribution of [|x|], [x] Cauchy with
      location [0], on [(0, inf)]. *)

  val laplace :
    loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [laplace ~loc ~scale] has density [exp (-|z|) / (2 scale)]. *)

  val logistic :
    loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [logistic ~loc ~scale] has density [exp (-z) / (scale (1 + exp (-z))²)].
  *)

  val exponential : rate:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [exponential ~rate] has density [rate exp (-rate x)] on [(0, inf)]. *)

  val gamma :
    concentration:(float, 'f) Nx.t ->
    rate:(float, 'f) Nx.t ->
    ((float, 'f) Nx.t, 'f) t
  (** [gamma ~concentration ~rate] has density proportional to
      [x^(α - 1) exp (-rate x)] on [(0, inf)], [α] the concentration. *)

  val inverse_gamma :
    concentration:(float, 'f) Nx.t ->
    scale:(float, 'f) Nx.t ->
    ((float, 'f) Nx.t, 'f) t
  (** [inverse_gamma ~concentration ~scale] is the distribution of [1 / x], [x]
      gamma with rate [scale]. *)

  val beta :
    a:(float, 'f) Nx.t -> b:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [beta ~a ~b] has density proportional to [x^(a - 1) (1 - x)^(b - 1)] on
      [(0, 1)]. *)

  val uniform :
    low:(float, 'f) Nx.t -> high:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [uniform ~low ~high] has density [1 / (high - low)] on [(low, high)]. Each
      element of [high - low] is in [(0, inf)]. *)

  val dirichlet : concentration:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
  (** [dirichlet ~concentration] is over vectors along the last axis of
      [concentration], of positive components summing to one, with density
      proportional to [prod_k x_k^(α_k - 1)]. A matrix of concentrations is
      independent rows.

      Raises [Invalid_argument] if the last axis has fewer than two components.
  *)

  val mvn :
    loc:(float, 'f) Nx.t ->
    scale_tril:(float, 'f) Nx.t ->
    ((float, 'f) Nx.t, 'f) t
  (** [mvn ~loc ~scale_tril] is the multivariate normal of vectors along the
      last axis, with mean [loc] and covariance [L Lᵀ], [L] the lower triangle
      of [scale_tril]. [L] is finite, its diagonal in [(0, inf)].

      Raises [Invalid_argument] if [scale_tril]'s last two axes are not square
      or not of [loc]'s length. *)

  (** {1:discrete Discrete families} *)

  val bernoulli : logits:(float, 'f) Nx.t -> (Nx.bool_t, 'f) t
  (** [bernoulli ~logits] is [true] with probability [sigmoid logits]. A logit
      may be infinite. *)

  val poisson : rate:(float, 'f) Nx.t -> (Nx.int32_t, 'f) t
  (** [poisson ~rate] is the count with probability [rate^x exp (-rate) / x!],
      [rate] non-negative and finite. *)

  val neg_binomial :
    mean:(float, 'f) Nx.t -> dispersion:(float, 'f) Nx.t -> (Nx.int32_t, 'f) t
  (** [neg_binomial ~mean ~dispersion] is the count of mean [mean] and variance
      [mean + mean² / dispersion], a Poisson whose rate is gamma with
      concentration [dispersion]. [mean] is non-negative and finite,
      [dispersion] in [(0, inf)]. *)

  val categorical : logits:(float, 'f) Nx.t -> (Nx.int64_t, 'f) t
  (** [categorical ~logits] is the index [k] with probability [softmax logits]
      along the last axis. A logit may be [-inf], a category of probability
      zero.

      Raises [Invalid_argument] if [logits] is a scalar or its last axis is
      empty. *)

  (** {1:combinators Combinators} *)

  val iid : int array -> ('x, 'f) t -> ('x, 'f) t
  (** [iid s d] is [s] independent draws of [d], its value of shape [s] followed
      by [d]'s. *)

  val sorted : int -> ((float, 'f) Nx.t, 'f) t -> ((float, 'f) Nx.t, 'f) t
  (** [sorted n d] is [n] independent draws of the scalar [d] sorted along a new
      last axis, with the exact density [n!] times the product of [d]'s on
      nondecreasing vectors.

      Raises [Invalid_argument] if [d] is not over scalars or [n < 1]. *)

  val transform :
    'f Bij.t -> ((float, 'f) Nx.t, 'f) t -> ((float, 'f) Nx.t, 'f) t
  (** [transform b d] is the distribution of [b]'s image of a draw of [d], with
      the density of [d] at the preimage less [b]'s log-determinant there. A
      value outside [b]'s image has density [-inf]. *)

  val mixture : logits:(float, 'f) Nx.t -> ('x, 'f) t -> ('x, 'f) t
  (** [mixture ~logits d] draws one component [k] with probability
      [softmax logits] for the whole value, then the value from [d_k], the slice
      of [d] at [k] on its leading axis: the log density is
      [logsumexp_k (log_softmax logits_k + log_density d_k x)]. A component per
      point is [iid [| n |] (mixture ~logits d)]. Its coordinates map onto the
      hull of its components' supports, read on the host.

      Raises [Invalid_argument] if [logits] is not a vector or [d]'s leading
      axis is not its length. *)

  (** {1:eliminators Eliminators} *)

  val log_density : ('x, 'f) t -> 'x -> (float, 'f) Nx.t
  (** [log_density d x] is [Nx.sum (factors d x)], a scalar. *)

  val factors : ('x, 'f) t -> 'x -> (float, 'f) Nx.t
  (** [factors d x] is the log density of each independent unit of [x]: one per
      element for the elementwise families, per row for {!dirichlet} and {!mvn},
      one for {!sorted} and {!mixture}, with {!iid}'s axes in front. [x]
      broadcasts against the parameters. A unit outside the support has factor
      [-inf]. *)

  val sample : Nx.Rng.t -> ('x, 'f) t -> 'x
  (** [sample k d] is a draw of [d] from the key [k]. It is reparameterised,
      differentiable in the parameters, wherever rune differentiates the
      {!Nx.Rng} sampler it uses. *)

  val quantile :
    ((float, 'f) Nx.t, 'f) t -> (float, 'f) Nx.t -> (float, 'f) Nx.t
  (** [quantile d p] is the value whose CDF is [p], elementwise.

      Raises [Invalid_argument] for a family whose quantile nx does not compute:
      {!student_t}, {!gamma}, {!inverse_gamma}, {!beta}, the vector families and
      the combinators but {!iid}. *)

  val coords : ((float, 'f) Nx.t, 'f) t -> 'f Bij.t
  (** [coords d] is the bijector onto [d]'s support, from coordinates in which
      samplers and optimisers move: {!Bij.identity} on the reals, {!Bij.exp} on
      [(0, inf)], {!Bij.interval} for {!beta} and {!uniform}, {!Bij.simplex} for
      {!dirichlet}, and for {!sorted} [d]'s bijector after {!Bij.ordered}. *)

  val standardize : ((float, 'f) Nx.t, 'f) t -> 'f Bij.t
  (** [standardize d] is the bijector whose coordinates are [d]'s draws
      standardised: [(x - loc) / scale] for {!normal}, {!student_t}, {!cauchy},
      {!laplace} and {!logistic}, [(log x - loc) / scale] for {!lognormal},
      [L⁻¹ (x - loc)] for {!mvn}, and the same for {!iid} of these.

      Raises [Invalid_argument] naming another family. *)

  val mixture_membership : ('x, 'f) t -> 'x -> (float, 'f) Nx.t
  (** [mixture_membership d x] is each component's posterior probability given
      [x], of shape [[k]] for a {!mixture} of [k] components and [s @ [k]] for
      [iid s] of one.

      Raises [Invalid_argument] if [d] is not one of those. *)

  (** {1:observers Observers} *)

  val family : ('x, 'f) t -> string
  (** [family d] is [d]'s family, such as ["normal"]. {!iid} keeps its
      argument's. *)

  val kind : ('x, 'f) t -> ('x, 'f) kind
  (** [kind d] is the type of [d]'s values. *)

  val dtype : ('x, 'f) t -> (float, 'f) Nx.dtype
  (** [dtype d] is the dtype of [d]'s log densities, and of its values for a
      continuous [d]. *)

  val shape : ('x, 'f) t -> int array
  (** [shape d] is the shape of [d]'s values. *)

  val support : ('x, 'f) t -> Support.t
  (** [support d] is the set of [d]'s values. Bounds given by parameters are
      read on the host, as the interval holding every element's support: inside
      a compiled or mapped function, a bound that depends on the function's
      arguments raises, {!Rune.Jit_error} or [Invalid_argument]. {!bounds} reads
      nothing. *)

  val bounds : ('x, 'f) t -> (float, 'f) Nx.t * (float, 'f) Nx.t
  (** [bounds d] is the least and greatest value each element of [d]'s values
      may take, of [d]'s shape: [-inf] and [inf] where unbounded, the support's
      interval for an element of a vector. It reads nothing on the host, so it
      serves inside compiled and mapped functions. *)

  val pp : Format.formatter -> ('x, 'f) t -> unit
  (** [pp ppf d] formats [d]'s family and its parameters' dtypes and shapes, as
      [normal(loc: float64 [], scale: float64 [8])]. *)

  (** {1:checks Checks} *)

  val check : ?unless:Nx.bool_t -> string -> ('x, 'f) t -> ('x, 'f) t
  (** [check ?unless what d] is [d] whose parameters are checked now, and whose
      eliminators then check none: a caller that computes parameters names
      itself in the refusal. A parameter outside its domain raises
      [Invalid_argument] with the message
      [what: family: param at [i] is v, not in domain]. Where [unless] holds, a
      parameter is taken whatever its value: [unless] broadcasts against the
      parameters. *)

  val valid : ('x, 'f) t -> Nx.bool_t
  (** [valid d] is whether every element of every parameter of [d] is in its
      domain, a scalar over all of [d]'s parameters: under {!Rune.val-vmap}, one
      per lane. NaN is in no domain. It reads the checks {!check} runs, raises
      nothing and ignores whether [d] was checked. *)
end

module Gaussian : sig
  (** Gaussians over structures, diagonal plus low rank.

      A Gaussian over a structure ['u] has a mean [m], a positive scale [S] per
      element, orthonormal directions [u_j] and variances [v_j > 0] along them:
      its covariance is [S (I + Σ_j (v_j - 1) u_j u_jᵀ) S], with the elements of
      every float tensor of ['u] taken together as one vector. No matrix over
      all the elements exists: each product is leafwise, then summed over
      leaves, at [O(k d)] for [k] directions and [d] elements.

      A Gaussian describes one position, so its tensors have no chain axis;
      {!sample} adds one and {!log_density} reads one. A sampler whose geometry
      differs per chain stacks Gaussians on the chain axis and applies chain
      [i]'s inside the map over chains.

      {b Preconditioning is a pullback.} The map [color z = m + S A z],
      [A = I + Σ_j (sqrt v_j - 1) u_j u_jᵀ], takes standard normal coordinates
      [z] to the Gaussian. A kernel moving with unit metric on
      [fun z -> lp (color z) + log |det|] moves with covariance [Σ] on [lp]. *)

  type ('u, 'f) t
  (** The type for Gaussians over ['u] whose log densities have element type
      ['f]. *)

  (** {1:constructors Constructors} *)

  val diagonal :
    'u Nx.Ptree.t -> (float, 'f) Nx.dtype -> mean:'u -> scale:'u -> ('u, 'f) t
  (** [diagonal u dtype ~mean ~scale] is the Gaussian of mean [mean] and
      standard deviation [scale], element by element. [scale]'s elements are
      positive. *)

  val low_rank :
    'u Nx.Ptree.t ->
    mean:'u ->
    scale:'u ->
    directions:'u ->
    variances:(float, 'f) Nx.t ->
    ('u, 'f) t
  (** [low_rank u ~mean ~scale ~directions ~variances] is the Gaussian whose
      directions are [directions], each tensor with a leading axis of [k], from
      [0] to the number of elements, orthonormalised, and whose variances along
      them are [variances], of shape [[k]].

      Raises [Invalid_argument] if [variances] is not of shape [[k]], and
      through {!Nx.check} if one is not positive and finite. *)

  val of_precision :
    'u Nx.Ptree.t ->
    (float, 'f) Nx.dtype ->
    ?low_rank:'u * (float, 'f) Nx.t ->
    mean:'u ->
    'u ->
    ('u, 'f) t
  (** [of_precision u dtype ?low_rank ~mean p] is the Gaussian of mean [mean]
      whose precision is the diagonal [p] plus, with [~low_rank:(w, l)], the sum
      of [l_j w_j w_jᵀ] over the directions [w_j], the leading axis of [w]'s
      tensors. The form is closed under inversion: a Laplace approximation is
      the precision at a mode.

      Raises [Invalid_argument] naming the path at an element of [p] that is not
      positive and finite, or at an [l_j] that is negative or not finite. *)

  (** {1:eliminators Eliminators} *)

  val sample : 'u Nx.Ptree.t -> Nx.Rng.t -> n:int -> ('u, 'f) t -> 'u
  (** [sample u k ~n g] is [n] draws of [g] on a new leading axis.

      Raises [Invalid_argument] if [n < 0]. *)

  val log_density : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u -> (float, 'f) Nx.t
  (** [log_density u g] is [g]'s log density at a position: one value per row of
      the leading axis. *)

  val mean : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u
  (** [mean u g] is [g]'s mean. *)

  val variance : 'u Nx.Ptree.t -> ('u, 'f) t -> 'u
  (** [variance u g] is the diagonal of [g]'s covariance, element by element. *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t
  (** [ptree u] is the structure of Gaussians over [u]: [mean], [scale] and
      [directions], each walked as [u], and [variances]. *)

  val pp : 'u Nx.Ptree.t -> Format.formatter -> ('u, 'f) t -> unit
  (** [pp u ppf g] formats [g]'s dimension and rank. *)
end

module Nuts : sig
  (** The No-U-Turn sampler over many chains.

      Multinomial NUTS with biased progressive sampling and the generalised
      U-turn criterion (Betancourt 2017), checked around each merged subtree and
      between its two halves, as Stan has since 2019. A transition diverges when
      the energy error exceeds [1000] or a position is not finite.

      Every chain has its own step size and geometry ({!Gaussian}), and moves
      with unit metric in its geometry's whitened coordinates. The chains move
      in lock step: one loop over leapfrog steps, each one density evaluation
      over all chains, until every chain has turned, diverged or filled its
      depth. The tree is built iteratively (Phan, Pradhan and Jankowiak 2019),
      with a checkpoint per level. A chain that has stopped is held: the density
      is evaluated again at its last point and its results are discarded, so no
      density is evaluated at a point a chain did not reach.

      {b Keys.} A transition gives chain [i] row [i] of
      [Nx.Rng.split_batch ~n:chains k], so a chain's randomness does not depend
      on the chain count. {!warmup} and {!sample} fold the state's draw counter
      into their key and advance it, so [a + b] draws are [a] draws then [b],
      and no two transitions of a run share a key. *)

  type ('u, 'f) state = private {
    position : 'u;  (** Each chain's position. *)
    lp : (float, 'f) Nx.t;  (** The log density there, shape [[chain]]. *)
    grad : 'u;  (** Its gradient there. *)
    step_size : (float, 'f) Nx.t;  (** Each chain's step size. *)
    geometry : ('u, 'f) Gaussian.t;
        (** Each chain's Gaussian, stacked on the chain axis. *)
    stats : 'f Stats.t;  (** The last transition's statistics. *)
    draw : Nx.int32_t;  (** The number of transitions taken. *)
    accept : (float, 'f) Nx.t;  (** The acceptance warmup aims for. *)
    max_depth : int;  (** The deepest tree a transition builds. *)
  }
  (** The type for states of chains over positions ['u]. *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
  (** [ptree u] is the structure of states over [u]. It reports [max_depth]. *)

  val stats : (float, 'f) Nx.dtype -> 'f Stats.t Nx.Ptree.t
  (** [stats dtype] is {!Stats.ptree}[ dtype]. *)

  val init :
    'u Nx.Ptree.t ->
    ?max_depth:int ->
    ?accept:float ->
    ?rank:int ->
    ?geometry:('u, 'f) Gaussian.t ->
    ('u -> (float, 'f) Nx.t) ->
    'u ->
    ('u, 'f) state
  (** [init u ?max_depth ?accept ?rank ?geometry lp start] is the state of
      chains at [start], a position. [max_depth] defaults to [10], [accept], the
      acceptance warmup aims for, to [0.8], and [rank], the directions warmup
      fits beside the diagonal, to [2]. Two directions gain correlated
      posteriors several times their effective draws per gradient, at a cost
      linear in the dimension; a rank above a chain's number of float elements
      is that number. Each chain's geometry is [geometry], which seeds every
      chain, or a diagonal whose variance is the inverse of the absolute
      gradient at the start; its step size is [1].

      Raises [Invalid_argument] if [max_depth < 1], [accept] is not in [(0, 1)],
      [rank < 0], or if [lp] does not return one log density per chain, as in
      [Norn.Nuts.init: the density returned shape [4; 8] for a position of 4
       chains; a density returns one log density per chain, shape [4]]. With
      several chains it evaluates [lp] again on the chains reversed, and raises
      naming the first row whose log density differs: a chain's density reads
      only its own row. *)

  val step :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [step u lp k s] is one transition of every chain from [s].

      Raises [Invalid_argument] if [lp] is NaN or [+inf] at a finite position,
      naming the chain. *)

  val warmup :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    steps:int ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [warmup u lp k ~steps s] is [s] after [steps] transitions that tune each
      chain's step size and geometry, by Stan's schedule scaled to [steps]: an
      initial buffer, windows that double in length, and a final buffer. Dual
      averaging (Nesterov 2009; Hoffman and Gelman 2014) sets each chain's step
      size for an acceptance of [s.accept], restarting at each window. At each
      window's end each chain's geometry is refitted to its window's draws and
      gradients by the Fisher divergence (Seyboldt, Carlson and Carpenter 2026):
      a diagonal scale [sqrt (sd x / sd score)] per element. Transition [n] of
      the run, counted by [s.draw], has the key [Nx.Rng.fold_in k n], as in
      {!sample}, so warmup and sampling may share a key.

      Raises [Invalid_argument] if [steps < 0]. *)

  val sample :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    draws:int ->
    ('u, 'f) state ->
    ('u, 'f) state * 'u Draws.t * 'f Stats.t Draws.t
  (** [sample u lp k ~draws s] is the state after [draws] transitions, their
      positions and their statistics. Transition [n] of the run, counted by
      [s.draw], has the key [Nx.Rng.fold_in k n].

      Raises [Invalid_argument] if [draws < 1]. *)
end

module Hmc : sig
  (** Hamiltonian Monte Carlo over many chains, with one trajectory length.

      Every chain shares one step size, one trajectory length and one geometry
      ({!Gaussian}), and moves with unit metric in the geometry's whitened
      coordinates. A transition draws each chain's momentum, takes [n] leapfrog
      steps, and accepts the trajectory's end with the Metropolis probability
      [min (1, exp (H0 - H))]. [n] is the length, jittered uniformly in [(0, 2)]
      times its value by one draw the chains share, over the step size, rounded
      up and at most [1024], so every chain takes the same number of steps
      (Sountsov, Carroll and Hoffman 2024). A chain whose step leaves the reals
      or whose energy error exceeds [1000] diverges: it is held at its last
      point, the density evaluated again there, and rejected.

      {b Keys.} Transition [n] of a run has the key [k = Nx.Rng.fold_in run n].
      Chain [i] has row [i] of
      [Nx.Rng.split_batch ~n:chains (Nx.Rng.fold_in k 0)], so its randomness
      does not depend on the chain count; the jitter draws from
      [Nx.Rng.fold_in k 1]. {!warmup} and {!sample} fold the state's draw
      counter into their key and advance it, so [a + b] draws are [a] draws then
      [b], and no two transitions of a run share a key. *)

  type ('u, 'f) state = private {
    position : 'u;  (** Each chain's position. *)
    lp : (float, 'f) Nx.t;  (** The log density there, shape [[chain]]. *)
    grad : 'u;  (** Its gradient there. *)
    step_size : (float, 'f) Nx.t;  (** The step size, a scalar. *)
    length : (float, 'f) Nx.t;
        (** The trajectory length in whitened coordinates, a scalar. *)
    geometry : ('u, 'f) Gaussian.t;  (** The chains' Gaussian. *)
    stats : 'f Stats.t;  (** The last transition's statistics. *)
    draw : Nx.int32_t;  (** The number of transitions taken. *)
    accept : (float, 'f) Nx.t;  (** The acceptance warmup aims for. *)
  }
  (** The type for states of chains over positions ['u]. *)

  type 'f stats = 'f Stats.t
  (** The type for statistics of transitions. [acceptance] is the Metropolis
      probability of the trajectory's end, [0] at a divergence, and [saturated]
      is [false]. *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
  (** [ptree u] is the structure of states over [u]. *)

  val stats : (float, 'f) Nx.dtype -> 'f stats Nx.Ptree.t
  (** [stats dtype] is {!Stats.ptree}[ dtype]. *)

  val init :
    'u Nx.Ptree.t ->
    ?accept:float ->
    ?rank:int ->
    ?geometry:('u, 'f) Gaussian.t ->
    ('u -> (float, 'f) Nx.t) ->
    'u ->
    ('u, 'f) state
  (** [init u ?accept ?rank ?geometry lp start] is the state of chains at
      [start], a position. [accept], the harmonic mean acceptance warmup aims
      for, defaults to [0.8], and [rank], the directions warmup fits beside the
      diagonal, to [2]; a rank above a chain's number of float elements is that
      number. The geometry is [geometry], or a diagonal centred on the chains'
      mean whose variance is the inverse of the gradient's root mean square over
      the chains at the start. The step size and the length are [1].

      Raises [Invalid_argument] if [accept] is not in [(0, 1)], [rank < 0], or
      if [lp] does not return one log density per chain, as in
      [Norn.Hmc.init: the density returned shape [4; 8] for a position of 4
       chains; a density returns one log density per chain, shape [4]]. With
      several chains it evaluates [lp] again on the chains reversed, and raises
      naming the first row whose log density differs. *)

  val step :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [step u lp k s] is one transition of every chain from [s].

      Raises [Invalid_argument] if [lp] is NaN or [+inf] at a finite position,
      naming the chain. *)

  val warmup :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    steps:int ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [warmup u lp k ~steps s] is [s] after [steps] transitions that tune the
      step size, the length and the geometry, by Stan's schedule scaled to
      [steps]: an initial buffer, windows that double in length, and a final
      buffer.

      - Dual averaging (Nesterov 2009; Hoffman and Gelman 2014) sets the step
        size for a harmonic mean acceptance over the chains of [s.accept],
        restarting at each window. A chain that never accepts drives the mean to
        [0], so the step shrinks until every chain moves.
      - The length maximises the change in the chains' expected squared jumped
        distance (ChEES; Hoffman, Radul and Sountsov 2021) in whitened
        coordinates: each transition estimates the criterion's gradient in log
        length from the chains, weighted by acceptance, and Adam takes a step of
        [0.025] along it. The final length is the iterates' polynomial average.
      - At each window's end the geometry is refitted to every chain's draws and
        gradients of the window by the Fisher divergence (Seyboldt, Carlson and
        Carpenter 2026): a diagonal scale [sqrt (sd x / sd score)] per element,
        then its directions.

      Transition [n] of the run, counted by [s.draw], has the key
      [Nx.Rng.fold_in k n], as in {!sample}, so warmup and sampling may share a
      key.

      Raises [Invalid_argument] if [steps < 0]. *)

  val sample :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    draws:int ->
    ('u, 'f) state ->
    ('u, 'f) state * 'u Draws.t * 'f stats Draws.t
  (** [sample u lp k ~draws s] is the state after [draws] transitions, their
      positions and their statistics. Transition [n] of the run, counted by
      [s.draw], has the key [Nx.Rng.fold_in k n].

      Raises [Invalid_argument] if [draws < 1]. *)
end

module Ensemble : sig
  (** Ensemble sampling by the stretch move, without derivatives.

      Walkers lie on the chain axis in independent ensembles of equal size,
      consecutive on the axis. A transition splits each ensemble at random into
      two halves, which mixes faster than fixed halves, and updates them in
      turn: a walker [x] of the moving half draws a walker [w] of the other half
      and a stretch [z] of density proportional to [1 / sqrt z] on [[1 / 2, 2]],
      and moves to [w + z (x - w)] with probability
      [min (1, z^(d - 1) p(y) / p(x))], [d] its float elements (Goodman and
      Weare 2010). A walker's move depends only on the half that does not move,
      as detailed balance requires, and on no scale: the move is unchanged by an
      affine map of the coordinates, so a correlated posterior costs what an
      isotropic one does. Nothing is tuned.

      The density is evaluated once per transition on the moving half of every
      ensemble, a call of [walkers / 2] rows when ensembles have an even size.

      The walkers move along lines through each other, so they explore the
      region their spread spans. Walkers that start far from a narrow,
      correlated posterior, spread over a region of another shape, come back to
      it only over many transitions: start them where a warmed sampler or the
      posterior's approximation leaves them.

      {b Keys.} Transition [n] of a run has the key [k = Nx.Rng.fold_in run n],
      and walker [i] row [i] of [Nx.Rng.split_batch ~n:walkers k]; an ensemble's
      split draws from its first walker's key. So with ensembles of a fixed size
      a walker's randomness does not depend on the number of ensembles.
      {!warmup} and {!sample} fold the state's draw counter into their key and
      advance it, so [a + b] draws are [a] draws then [b]. *)

  type 'f stats = {
    lp : (float, 'f) Nx.t;  (** The log density at the transition's end. *)
    acceptance : (float, 'f) Nx.t;
        (** The probability with which the walker's move was accepted. *)
  }
  (** The type for statistics of transitions, one element per walker. *)

  type ('u, 'f) state = private {
    position : 'u;  (** Each walker's position. *)
    lp : (float, 'f) Nx.t;  (** The log density there, shape [[walker]]. *)
    stats : 'f stats;  (** The last transition's statistics. *)
    draw : Nx.int32_t;  (** The number of transitions taken. *)
    ensembles : int;  (** The number of ensembles. *)
  }
  (** The type for states of walkers over positions ['u]. *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
  (** [ptree u] is the structure of states over [u]. It reports [ensembles]. *)

  val stats : (float, 'f) Nx.dtype -> 'f stats Nx.Ptree.t
  (** [stats dtype] is the structure of statistics at [dtype]. *)

  val init :
    'u Nx.Ptree.t ->
    ?ensembles:int ->
    ('u -> (float, 'f) Nx.t) ->
    'u ->
    ('u, 'f) state
  (** [init u ?ensembles lp start] is the state of walkers at [start], a
      position, in [ensembles] ensembles, by default [1].

      Raises [Invalid_argument] if [ensembles < 1], if the walkers do not split
      into [ensembles] ensembles of equal size, if an ensemble has fewer walkers
      than twice a walker's float elements, or if [lp] does not return one log
      density per walker, each read from its own row. *)

  val step :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [step u lp k s] is one transition of every walker from [s].

      Raises [Invalid_argument] if [lp] is NaN or [+inf] at a walker. *)

  val warmup :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    steps:int ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [warmup u lp k ~steps s] is [s] after [steps] transitions, whose draws are
      discarded. Transition [n] of the run, counted by [s.draw], has the key
      [Nx.Rng.fold_in k n], as in {!sample}.

      Raises [Invalid_argument] if [steps < 0]. *)

  val sample :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    draws:int ->
    ('u, 'f) state ->
    ('u, 'f) state * 'u Draws.t * 'f stats Draws.t
  (** [sample u lp k ~draws s] is the state after [draws] transitions, the
      walkers' positions and their statistics, each walker a chain of the draws.
      A walker's moves read the other walkers of its ensemble, so the chains of
      one ensemble are dependent and those of two independent: their R-hat is
      {!Diag.nested_rhat} with a superchain per ensemble ({!Summary.v}'s
      [superchains] at [s.ensembles]). Transition [n] of the run, counted by
      [s.draw], has the key [Nx.Rng.fold_in k n].

      Raises [Invalid_argument] if [draws < 1]. *)
end

module Weighted : sig
  (** Weighted draws.

      Weighted draws of a structure ['u] are a value of ['u] whose every tensor
      has one leading axis of [n] draws, and a log weight per draw. The weights
      need not be normalised, and a draw of log weight [-inf] has none.
      Importance sampling, sequential Monte Carlo and nested sampling return
      them. *)

  type ('u, 'f) t = {
    values : 'u;  (** The draws, on a leading axis of [n]. *)
    log_weights : (float, 'f) Nx.t;  (** Their log weights, shape [[n]]. *)
  }
  (** The type for weighted draws of ['u]. *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t
  (** [ptree u] is the structure of weighted draws of [u]: [values], walked as
      [u], and [log_weights]. *)

  val ess : ('u, 'f) t -> (float, 'f) Nx.t
  (** [ess w] is Kish's effective sample size [(Σ w)² / Σ w²], a scalar from [1]
      to [n]. It is NaN if no weight is positive. *)

  val resample : 'u Nx.Ptree.t -> Nx.Rng.t -> n:int -> ('u, 'f) t -> 'u
  (** [resample u k ~n w] is [n] equal-weight draws of [w], by systematic
      resampling: one uniform [v] places the points [(i + v) / n] on the
      cumulative normalised weights, so a draw of normalised weight [w_i] is
      copied [floor (n w_i)] or [ceil (n w_i)] times.

      Raises [Invalid_argument] if [n < 1]. *)
end

module Evidence : sig
  (** Evidence: the normalising constant of a posterior, with its draws.

      For a prior [π] and a likelihood [L] the evidence is [Z = ∫ L π], the
      posterior's normaliser and the probability of the data under the model.
      {!Norn.Nested} and {!Norn.Smc} estimate it, and their particles weighted
      by the posterior are its {!sample}. *)

  type ('u, 'f) t
  (** The type for evidence estimates over positions ['u]. *)

  val log_evidence : ('u, 'f) t -> (float, 'f) Nx.t
  (** [log_evidence z] is the estimate of [ln Z], a scalar. *)

  val error : ('u, 'f) t -> (float, 'f) Nx.t
  (** [error z] is the estimate's standard error in [ln Z], a scalar. *)

  val information : ('u, 'f) t -> (float, 'f) Nx.t
  (** [information z] is [H], the posterior's Kullback-Leibler divergence from
      the prior in nats, [E_post (ln L) - ln Z]: how much the data narrowed the
      prior. *)

  val sample : ('u, 'f) t -> ('u, 'f) Weighted.t
  (** [sample z] is draws weighted by the posterior, their log weights
      normalised. *)

  val expectation :
    'u Nx.Ptree.t ->
    ('u -> (float, 'f) Nx.t) ->
    ('u, 'f) t ->
    (float, 'f) Nx.t * (float, 'f) Nx.t
  (** [expectation u f z] is the posterior expectation of [f], a map of one draw
      to a tensor, under [z]'s {!sample}, and its standard error, element by
      element. The error counts the noise of the weights over the shrinkage
      sequences {!error} simulates, and the sampling error of the draws in
      independent groups: for nested sampling the draws that descend from one
      starting point ({!Norn.Nested.type-state}'s [lineage]), for tempering each
      chain. It under-covers when slice moves leave the draws within a
      replacement correlated: by 1.5 to 1.9 times on a hierarchical posterior's
      hyperparameters. For an error to trust on a hard posterior, run
      independent replicates and take their spread. [f] runs under
      {!Rune.val-vmap}, once for all draws. *)

  (** {1:stop Why a run stopped} *)

  (** The type for the reasons a run stopped. A spent budget is a result, not an
      error: the estimate so far, with how far it fell short. *)
  type stop =
    | Converged
        (** The run finished: nested sampling's live points could add less than
            its tolerance to [ln Z], tempering reached [β = 1]. *)
    | Remaining of float
        (** Nested sampling's budget ran out while its live points could still
            add up to this many nats to [ln Z]. *)
    | Temperature of float
        (** Tempering's budget ran out at this temperature [β < 1]: the estimate
            is of [ln ∫ L^β π], the sample of its tempered posterior. *)

  val stop : ('u, 'f) t -> stop
  (** [stop z] is why the run that gave [z] stopped. It reads [z]'s tensors, so
      it waits for a computation that produces them. *)

  (** {1:structure Structure and formatting} *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t
  (** [ptree u] is the structure of evidence estimates over [u], so a compiled
      function may return one. *)

  val pp : Format.formatter -> ('u, 'f) t -> unit
  (** [pp] formats an estimate as [ln Z = -412.31 ± 0.08, H = 5.20 nats],
      followed, for a spent budget, by [, budget spent with 0.52 nats left] or
      [, budget spent at β = 0.73]. *)
end

module Nested : sig
  (** Nested sampling, by slice moves on the prior.

      Nested sampling (Skilling 2006) estimates the evidence [Z = ∫ L π] as a
      sum over shells of prior volume. It keeps [n] live points drawn from the
      prior; each step deletes the [batch] of lowest likelihood and replaces
      each by a draw from the prior above the highest deleted likelihood. Here a
      replacement starts at a surviving live point drawn at random and takes
      three hit-and-run slice moves (Neal 2003) per float element of a point on
      the prior, constrained above that likelihood, along directions of unit
      length in the coordinates whitened by the survivors' mean and covariance
      (nested slice sampling; Yallup et al. 2026). Fewer moves leave the
      replacements correlated with their starts: on a Gaussian in ten dimensions
      one move per element biases [ln Z] by [+0.39] nats, two by [+0.14], and
      three by no measurable amount. Likelihoods compare with a rank, uniform
      per point, as a second key, so ties keep the live set uniform (Fowlie,
      Handley and Su 2021).

      {b Volumes.} Deleting the [k] lowest of [n] live points shrinks the prior
      volume by the order statistics of [n] uniforms: the [j]-th deleted, from
      the lowest, lowers [ln X] by [1 / (n - j + 1)] in expectation. When the
      live points' largest possible contribution, the highest live likelihood
      times the volume left, would change [ln Z] by less than [tolerance], the
      run stops and adds them as dead points with their order statistics'
      volumes, the highest taking all the volume left. {!Evidence.error} is the
      spread of [ln Z] over 100 simulated shrinkage sequences; [sqrt (H / n)]
      (Skilling 2006) is its limit at a constant count.

      {b Densities.} [prior] and [likelihood] are densities over positions of
      the same structure, evaluated on the [batch] points being replaced. The
      prior is normalised, so the evidence is the data's probability.

      {b Keys.} {!run}'s step [s] has the key
      [Nx.Rng.fold_in (Nx.Rng.fold_in k 0) s], and its {!evidence} the key
      [Nx.Rng.fold_in k 1]. A step's replacement [i] has row [i] of
      [Nx.Rng.split_batch ~n:batch] of the step's key. *)

  type ('u, 'f) state = private {
    live : 'u;  (** The live points, on a leading axis of [n]. *)
    prior : (float, 'f) Nx.t;  (** Their prior log densities, shape [[n]]. *)
    likelihood : (float, 'f) Nx.t;  (** Their log likelihoods. *)
    rank : (float, 'f) Nx.t;  (** Their ranks, in [(0, 1)]. *)
    lineage : Nx.int32_t;
        (** The starting point each descends from, by the slice moves that
            replaced its ancestors: an index of the initial live points. *)
    dead : 'u;
        (** The dead points, lowest first, on a leading axis of
            [budget * batch]; the rows past [deaths] are padding. *)
    dead_likelihood : (float, 'f) Nx.t;
        (** Their log likelihoods, [-inf] in the padding. *)
    shrinkage : (float, 'f) Nx.t;
        (** The expected fall in [ln X] at each, [0] in the padding. *)
    dead_lineage : Nx.int32_t;  (** Their lineages. *)
    deaths : Nx.int32_t;  (** The number of dead points. *)
    log_volume : (float, 'f) Nx.t;  (** The expected [ln X] left. *)
    log_evidence : (float, 'f) Nx.t;  (** [ln Z] of the dead points. *)
    steps : Nx.int32_t;
        (** The steps taken, at most [budget]: at [budget] the dead buffer is
            full and {!step} leaves the state as it is. *)
    tolerance : (float, 'f) Nx.t;  (** The stopping tolerance in [ln Z]. *)
    batch : int;  (** The points a step deletes. *)
    budget : int;  (** The steps the dead buffer holds. *)
  }
  (** The type for states of nested sampling over positions ['u]. *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
  (** [ptree u] is the structure of states over [u]. It reports [batch] and
      [budget]. *)

  val init :
    'u Nx.Ptree.t ->
    ?batch:int ->
    ?tolerance:float ->
    budget:int ->
    prior:('u -> (float, 'f) Nx.t) ->
    likelihood:('u -> (float, 'f) Nx.t) ->
    'u ->
    ('u, 'f) state
  (** [init u ?batch ?tolerance ~budget ~prior ~likelihood live] is the state
      with the live points [live], [n] independent draws of the prior. [batch]
      defaults to [n / 2] and [tolerance] to [1e-3]; the dead buffer holds
      [budget] steps.

      Raises [Invalid_argument] if [batch < 1] or [batch >= n], [budget < 1],
      [tolerance] is not positive, or if a density does not return one log
      density per point, each read from its own row. *)

  val step :
    'u Nx.Ptree.t ->
    prior:('u -> (float, 'f) Nx.t) ->
    likelihood:('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [step u ~prior ~likelihood k s] deletes [s.batch] live points and replaces
      them, or is [s] if [s] has taken [s.budget] steps.

      Raises [Invalid_argument] through {!Nx.check} if [likelihood] is NaN or
      [+inf] at a point. *)

  val evidence :
    'u Nx.Ptree.t -> Nx.Rng.t -> ('u, 'f) state -> ('u, 'f) Evidence.t
  (** [evidence u k s] is the evidence of the dead points of [s] and of its live
      points added as dead, [k] simulating the shrinkage sequences of its error.
      Its {!Evidence.sample} is every dead point, padding included at weight
      zero, and its {!Evidence.stop} is [Converged], or [Remaining r] if the
      live points could still add [r] nats, the tolerance or more, to [ln Z]. *)

  val run :
    'u Nx.Ptree.t ->
    ?batch:int ->
    ?tolerance:float ->
    budget:int ->
    prior:('u -> (float, 'f) Nx.t) ->
    likelihood:('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    'u ->
    ('u, 'f) Evidence.t
  (** [run u ?batch ?tolerance ~budget ~prior ~likelihood k live] steps from
      [init] until the live points' largest possible contribution changes [ln Z]
      by less than [tolerance], or for [budget] steps, and is the evidence then.
      At the default batch a step lowers [ln X] by about [0.69], so a posterior
      [H] nats from the prior needs about [H / 0.69] steps and a few more.

      Raises [Invalid_argument] as {!init} does. *)
end

module Smc : sig
  (** Sequential Monte Carlo by tempering.

      The particles move from the prior to the posterior through the tempered
      densities [prior + β likelihood], [β] rising from [0] to [1], and the
      product of their mean incremental weights estimates the evidence.

      {b Temperatures.} From [β] the next temperature is [1] if the incremental
      weights [exp ((1 - β) L)] keep an effective sample size of half the
      particles; otherwise it is the [β'] above [β] and below [1] where they
      keep exactly half, found by {!Jera.Root.bracket}.

      {b Moves.} Each step is waste-free (Dau and Chopin 2022): it resamples
      [resampled] ancestors, [M], from the [N] particles systematically by their
      incremental weights and runs [N / M - 1] moves from each on the tempered
      density, keeping every state, so the [N] particles are [M] chains of
      [N / M] states, equally weighted. Moves are in the coordinates whitened by
      the reweighted particles' mean and covariance, by one of two kernels
      ({!move}):

      - [Hmc], Hamiltonian transitions of one step size and one length, the
        length jittered uniformly in [(0, 2)] times its value, as {!Hmc.step}
        takes them. Both start at [1] and are tuned between temperatures, which
        are close, and fixed within one. The step size takes one Robbins-Monro
        step on its log toward a mean acceptance of [0.8]: [log eps] gains the
        mean acceptance less [0.8]. The length takes one Adam step of the ChEES
        criterion per move, as {!Hmc.warmup} tunes it, one search across the
        run, each gradient estimated from the [M] chains. The densities need a
        gradient.
      - [Slice], hit-and-run slice steps (Neal 2003) along directions of unit
        length; nothing is tuned and the densities need no derivative.

      {b Error.} Each temperature's mean incremental weight has a variance
      estimated from its [M] chains read as one stationary chain: their pooled
      autocovariances summed by Geyer's initial monotone sequence (Dau and
      Chopin 2022). {!Evidence.error} is the root of the relative variances
      summed over temperatures, whose errors are nearly independent when the
      chains are long beside their number.

      {b Densities.} [prior] and [likelihood] are densities over positions of
      the same structure, evaluated on the [M] chains of a move. The prior is
      normalised, so the evidence is the data's probability.

      {b Keys.} {!run}'s step [s] has the key [k_s = Nx.Rng.fold_in k s]: it
      resamples from [Nx.Rng.fold_in k_s 0], and its moves draw from
      [k_m = Nx.Rng.fold_in k_s 1]. Hamiltonian move [j] is the transition of
      key [Nx.Rng.fold_in k_m j], as {!Hmc.step} draws from its key; chain [i]'s
      slice move [j] draws from [Nx.Rng.fold_in] of row [i] of
      [Nx.Rng.split_batch ~n:M k_m] with [j]. *)

  (** The type for the kernels that move the particles. *)
  type move =
    | Hmc  (** Hamiltonian transitions; the densities need a gradient. *)
    | Slice  (** Hit-and-run slice steps; no derivative. *)

  type ('u, 'f) state = private {
    particles : 'u;  (** The particles, on a leading axis of [N]. *)
    prior : (float, 'f) Nx.t;  (** Their prior log densities, shape [[N]]. *)
    likelihood : (float, 'f) Nx.t;  (** Their log likelihoods. *)
    beta : (float, 'f) Nx.t;  (** The temperature, a scalar in [[0, 1]]. *)
    log_evidence : (float, 'f) Nx.t;  (** [ln Z] of the tempered density. *)
    variance : (float, 'f) Nx.t;  (** The variance of [log_evidence]. *)
    steps : Nx.int32_t;  (** The temperatures taken. *)
    step_size : (float, 'f) Nx.t;
        (** The Hamiltonian moves' step size in whitened coordinates, a scalar.
        *)
    length : (float, 'f) Nx.t;
        (** Their trajectory length in whitened coordinates, a scalar. *)
    length_moment : (float, 'f) Nx.t;
        (** The second moment of the length's Adam search, a scalar. *)
    resampled : int;  (** [M], the chains of a step. *)
    move : move;  (** The kernel that moves the particles. *)
  }
  (** The type for states of tempering over positions ['u]. *)

  val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t
  (** [ptree u] is the structure of states over [u]. It reports [resampled]. *)

  val init :
    'u Nx.Ptree.t ->
    ?move:move ->
    ?resampled:int ->
    prior:('u -> (float, 'f) Nx.t) ->
    likelihood:('u -> (float, 'f) Nx.t) ->
    'u ->
    ('u, 'f) state
  (** [init u ?move ?resampled ~prior ~likelihood start] is the state at [β = 0]
      of the particles [start], [N] independent draws of the prior. [move]
      defaults to [Hmc]. [resampled] defaults to the largest divisor of [N] not
      above [sqrt N]: the evidence's error is consistent when the chains are
      long beside their number, [M] small beside [sqrt N] (Dau and Chopin 2022,
      theorem 2), and a thousand particles make 25 chains of 40.

      Raises [Invalid_argument] if [resampled] does not divide [N] into chains
      of two states or more, or if a density does not return one log density per
      particle, each read from its own row. *)

  val step :
    'u Nx.Ptree.t ->
    prior:('u -> (float, 'f) Nx.t) ->
    likelihood:('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    ('u, 'f) state ->
    ('u, 'f) state
  (** [step u ~prior ~likelihood k s] moves the particles of [s] to the next
      temperature. At [β = 1] it moves them on the posterior and leaves the
      evidence as it is.

      Raises [Invalid_argument] through {!Nx.check} if a density is NaN or
      [+inf] at a particle. *)

  val evidence : ('u, 'f) state -> ('u, 'f) Evidence.t
  (** [evidence s] is the evidence of the tempered density at [s.beta], the
      posterior's at [β = 1], with the particles as its sample. Its
      {!Evidence.stop} is [Converged] at [β = 1], else [Temperature s.beta]. *)

  val run :
    'u Nx.Ptree.t ->
    ?move:move ->
    ?resampled:int ->
    budget:int ->
    prior:('u -> (float, 'f) Nx.t) ->
    likelihood:('u -> (float, 'f) Nx.t) ->
    Nx.Rng.t ->
    'u ->
    ('u, 'f) Evidence.t
  (** [run u ?move ?resampled ~budget ~prior ~likelihood k start] steps from
      [init] until [β = 1], or for [budget] temperatures, and is the evidence
      then.

      Raises [Invalid_argument] as {!init} does, and if [budget < 1]. *)
end
