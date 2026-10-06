(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Diagnostics of draws.

    A diagnostic of draws of ['u] is a value of ['u] holding one number per
    element, at the element's dtype: [(rhat schools post).tau] is the R-hat of
    [tau]. Diagnostics read their draws on the host and take float tensors.

    The chain diagnostics follow Vehtari, Gelman, Simpson, Carpenter and Bürkner
    (2021). Each splits every chain into its two halves, dropping the middle
    draw of an odd count, and ranks the pooled draws, ties at their average rank
    [r]; the normal scores [ndtri ((r - 3/8) / (S + 1/4))], [S] the number of
    draws, make them insensitive to the draws' scale and tails. An element whose
    draws are all equal, or not all finite, or a chain of fewer than four draws,
    has no diagnostic: it is NaN.

    Raises [Invalid_argument] naming the path for a tensor that is not of a
    float dtype. *)

(** {1:convergence Convergence} *)

val rhat : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
(** [rhat u d] is the rank-normalised split R-hat of each element: the larger of
    the R-hat of the split chains' normal scores (bulk) and of the normal scores
    of their distance to the median (tails). Near [1] the chains agree; above
    [1.01] they have not mixed. *)

val nested_rhat : 'u Nx.Ptree.t -> superchains:int -> 'u Draws.t -> 'u
(** [nested_rhat u ~superchains d] is the rank-normalised nested R-hat
    (Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari and Gelman 2024) of
    each element, for many short chains: the chains form [superchains]
    consecutive groups of equal size, each started from one point, and the
    statistic compares the variance between groups with the variance within
    them, chains split and scored as for {!rhat}.

    Raises [Invalid_argument] if [superchains < 2] or if it does not divide the
    number of chains. *)

val ess_bulk : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
(** [ess_bulk u d] is the effective sample size of each element's normal scores,
    by Geyer's initial monotone sequence over the autocorrelations of the split
    chains. It measures how well the draws estimate the centre of the
    distribution. *)

val ess_tail : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
(** [ess_tail u d] is the smaller of the effective sample sizes of the
    indicators of each element being at most its 5% and its 95% quantile. It
    measures how well the draws estimate the tails. *)

val mcse_mean : 'u Nx.Ptree.t -> 'u Draws.t -> 'u
(** [mcse_mean u d] is the Monte Carlo standard error of each element's mean:
    its standard deviation over the square root of the effective sample size of
    the split chains' draws. *)

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
    all draws. Where divergences gather, it is far from [0]; with no divergence
    it is NaN.

    Raises [Invalid_argument] naming both shapes if [s] and [d] differ in their
    chains or draws. *)

(** {1:calibration Calibration} *)

val rank : 'u Nx.Ptree.t -> truth:'u -> 'u Draws.t -> 'u
(** [rank u ~truth d] is, for each element, the number of draws of [d] below the
    element of [truth]: from [0] to the number of draws. With [truth] drawn from
    the prior, data simulated from it and [d] exact posterior draws, each rank
    is uniform (Talts, Betancourt, Simpson, Vehtari and Gelman 2018).

    Raises [Invalid_argument] naming the path if there are more draws than the
    element's dtype holds exact integers. *)

val rank_uniformity : 'u Nx.Ptree.t -> draws:int -> 'u list -> 'u
(** [rank_uniformity u ~draws ranks] is, for each element, the simultaneous
    p-value of its ranks against the uniform distribution on [0], ..., [draws]
    (Säilynoja, Bürkner and Vehtari 2022), computed exactly, with no simulation:
    the probability, under uniformity, that the count of ranks below some point
    [i / (draws + 1)] strays from its expectation at least as far, in pointwise
    p-value, as the ranks' farthest count. Below [alpha] rejects uniformity at
    level [alpha].

    Raises [Invalid_argument] if [ranks] is empty, if [draws < 1], or if a rank
    is not an integer in [0], ..., [draws]. *)
