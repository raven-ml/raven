(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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
    [Invalid_argument] naming the function, and record their parameters' domain
    checks. Every eliminator runs them with {!Nx.check}, eagerly at once and in
    a compiled function when the call returns: a parameter outside its domain
    raises [Invalid_argument] naming its index and value, as in
    [Norn.Dist.log_density: normal: scale at [3] is -1, not in (0, inf)]. A
    value outside the support has density [-inf] and raises nothing. {!valid}
    reads the same checks as data, for a density that must be total:

    {[
    let ok = Norn.Dist.valid d in
    let d = Norn.Dist.check ~unless:(Nx.scalar Nx.bool true) "" d in
    Nx.where ok (Norn.Dist.log_density d x) (Nx.scalar dt Float.neg_infinity)
    ]} *)

type ('x, 'f) t
(** The type for distributions over values ['x] whose log densities have element
    type ['f]. ['x] is [(float, 'f) Nx.t] for continuous families, {!Nx.int32_t}
    for counts, {!Nx.int64_t} for categories and {!Nx.bool_t} for {!bernoulli}.
*)

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
(** [half_normal ~scale] is the distribution of [|x|], [x] normal with location
    [0], on [(0, inf)]. *)

val lognormal :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
(** [lognormal ~loc ~scale] is the distribution of [exp x], [x] normal. *)

val student_t :
  df:(float, 'f) Nx.t ->
  loc:(float, 'f) Nx.t ->
  scale:(float, 'f) Nx.t ->
  ((float, 'f) Nx.t, 'f) t
(** [student_t ~df ~loc ~scale] is Student's t with [df] degrees of freedom, in
    [(0, inf)], located and scaled. *)

val cauchy :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
(** [cauchy ~loc ~scale] has density [1 / (π scale (1 + z²))]. *)

val half_cauchy : scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
(** [half_cauchy ~scale] is the distribution of [|x|], [x] Cauchy with location
    [0], on [(0, inf)]. *)

val laplace :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
(** [laplace ~loc ~scale] has density [exp (-|z|) / (2 scale)]. *)

val logistic :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
(** [logistic ~loc ~scale] has density [exp (-z) / (scale (1 + exp (-z))²)]. *)

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

val beta : a:(float, 'f) Nx.t -> b:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t
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

    Raises [Invalid_argument] if the last axis has fewer than two components. *)

val mvn :
  loc:(float, 'f) Nx.t ->
  scale_tril:(float, 'f) Nx.t ->
  ((float, 'f) Nx.t, 'f) t
(** [mvn ~loc ~scale_tril] is the multivariate normal of vectors along the last
    axis, with mean [loc] and covariance [L Lᵀ], [L] the lower triangle of
    [scale_tril]. [L] is finite, its diagonal in [(0, inf)].

    Raises [Invalid_argument] if [scale_tril]'s last two axes are not square or
    not of [loc]'s length. *)

(** {1:discrete Discrete families} *)

val bernoulli : logits:(float, 'f) Nx.t -> (Nx.bool_t, 'f) t
(** [bernoulli ~logits] is [true] with probability [sigmoid logits]. A logit may
    be infinite. *)

val poisson : rate:(float, 'f) Nx.t -> (Nx.int32_t, 'f) t
(** [poisson ~rate] is the count with probability [rate^x exp (-rate) / x!],
    [rate] non-negative and finite. *)

val neg_binomial :
  mean:(float, 'f) Nx.t -> dispersion:(float, 'f) Nx.t -> (Nx.int32_t, 'f) t
(** [neg_binomial ~mean ~dispersion] is the count of mean [mean] and variance
    [mean + mean² / dispersion], a Poisson whose rate is gamma with
    concentration [dispersion]. [mean] is non-negative and finite, [dispersion]
    in [(0, inf)]. *)

val categorical : logits:(float, 'f) Nx.t -> (Nx.int64_t, 'f) t
(** [categorical ~logits] is the index [k] with probability [softmax logits]
    along the last axis. A logit may be [-inf], a category of probability zero.

    Raises [Invalid_argument] if [logits] is a scalar or its last axis is empty.
*)

(** {1:combinators Combinators} *)

val iid : int array -> ('x, 'f) t -> ('x, 'f) t
(** [iid s d] is [s] independent draws of [d], its value of shape [s] followed
    by [d]'s. *)

val sorted : int -> ((float, 'f) Nx.t, 'f) t -> ((float, 'f) Nx.t, 'f) t
(** [sorted n d] is [n] independent draws of the scalar [d] sorted along a new
    last axis, with the exact density [n!] times the product of [d]'s on
    nondecreasing vectors.

    Raises [Invalid_argument] if [d] is not over scalars or [n < 1]. *)

val transform : 'f Bij.t -> ((float, 'f) Nx.t, 'f) t -> ((float, 'f) Nx.t, 'f) t
(** [transform b d] is the distribution of [b]'s image of a draw of [d], with
    the density of [d] at the preimage less [b]'s log-determinant there. A value
    outside [b]'s image has density [-inf]. *)

val mixture : logits:(float, 'f) Nx.t -> ('x, 'f) t -> ('x, 'f) t
(** [mixture ~logits d] draws one component [k] with probability
    [softmax logits] for the whole value, then the value from [d_k], the slice
    of [d] at [k] on its leading axis: the log density is
    [logsumexp_k (log_softmax logits_k + log_density d_k x)]. A component per
    point is [iid [| n |] (mixture ~logits d)]. Its coordinates map onto the
    hull of its components' supports, read on the host.

    Raises [Invalid_argument] if [logits] is not a vector or [d]'s leading axis
    is not its length. *)

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
    differentiable in the parameters, wherever rune differentiates the {!Nx.Rng}
    sampler it uses. *)

val quantile : ((float, 'f) Nx.t, 'f) t -> (float, 'f) Nx.t -> (float, 'f) Nx.t
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
(** [family d] is [d]'s family, such as ["normal"]. {!iid} keeps its argument's.
*)

val kind : ('x, 'f) t -> ('x, 'f) kind
(** [kind d] is the type of [d]'s values. *)

val dtype : ('x, 'f) t -> (float, 'f) Nx.dtype
(** [dtype d] is the dtype of [d]'s log densities, and of its values for a
    continuous [d]. *)

val shape : ('x, 'f) t -> int array
(** [shape d] is the shape of [d]'s values. *)

val support : ('x, 'f) t -> Support.t
(** [support d] is the set of [d]'s values. Bounds given by parameters are read
    on the host, as the interval holding every element's support: inside a
    compiled or mapped function, a bound that depends on the function's
    arguments raises, {!Rune.Jit_error} or [Invalid_argument]. {!bounds} reads
    nothing. *)

val bounds : ('x, 'f) t -> (float, 'f) Nx.t * (float, 'f) Nx.t
(** [bounds d] is the least and greatest value each element of [d]'s values may
    take, of [d]'s shape: [-inf] and [inf] where unbounded, the support's
    interval for an element of a vector. It reads nothing on the host, so it
    serves inside compiled and mapped functions. *)

val pp : Format.formatter -> ('x, 'f) t -> unit
(** [pp ppf d] formats [d]'s family and its parameters' dtypes and shapes, as
    [normal(loc: float64 [], scale: float64 [8])]. *)

(** {1:checks Checks} *)

val check : ?unless:Nx.bool_t -> string -> ('x, 'f) t -> ('x, 'f) t
(** [check ?unless what d] is [d] whose parameters are checked now, and whose
    eliminators then check none: a caller that computes parameters names itself
    in the refusal. A parameter outside its domain raises [Invalid_argument]
    with the message [what: family: param at [i] is v, not in domain]. Where
    [unless] holds, a parameter is taken whatever its value: [unless] broadcasts
    against the parameters. *)

val valid : ('x, 'f) t -> Nx.bool_t
(** [valid d] is whether every element of every parameter of [d] is in its
    domain, a scalar over all of [d]'s parameters: under {!Rune.val-vmap}, one
    per lane. NaN is in no domain. It reads the checks {!check} runs, raises
    nothing and ignores whether [d] was checked. *)
