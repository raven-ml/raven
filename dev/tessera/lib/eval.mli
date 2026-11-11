(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Measures of embeddings: a linear probe and RankMe.

    A {e linear probe} scores how much of a labelling a linear map can read from
    the embeddings: multinomial logistic regression fitted on the embeddings of
    a training set's pixels, then scored on others. TESSERA reports a probe's
    accuracy and weighted F1 on crop types (v1 §5). {e RankMe} (Garrido et al.
    2023) measures how many directions a set of embeddings fills, without
    labels, as the entropy of its singular values; TESSERA reports it divided by
    its greatest value, so a collapsed embedding scores near [0] and one that
    fills every direction alike near [1].

    Every function computes on the host in float64. *)

(** {1:probe Linear probes} *)

type probe = {
  w : Nx.host Nx.float64_t;  (** [[d; classes]] *)
  b : Nx.host Nx.float64_t;  (** [[classes]] *)
}
(** The type for probes: the class of [x] is the greatest of [x · w + b]. *)

val fit : c:float -> (float, 's, Nx.host) Nx.t -> Nx.host Nx.int64_t -> probe
(** [fit ~c x y] is the probe that minimises
    [Σᵢ −log softmax (xᵢ · w + b)_{yᵢ} + |w|² / (2c)] over the rows [xᵢ] of [x]
    ([[n; d]]) and their labels [y] ([[n]]), its bias free of the penalty. The
    classes are [0] to the greatest label, and each must have a row: a class
    without one would send its bias to [−∞]. The objective is convex, and
    strictly convex in [w] and in the differences of [b]'s entries, which alone
    the probabilities read; its minimum is the one whose [b] sums to [0]. L-BFGS
    ([Jera.Minimize.lbfgs], memory 10) finds it from zeros, to the tolerance
    [Jera.Tol.v ~rel:1e-6 ~abs:1e-6] in [Jera.Minimize]'s sense, in at most
    [10000] iterations.

    Raises [Invalid_argument] if [c] is not positive, if [x] is not [[n; d]]
    with [n > 0] or [y] not [[n]], if a label is negative, or if a class below
    the greatest label has no row; and [Failure] with the search's report if the
    search ends otherwise than converged: its budget spent, stalled, or at a
    value that is not finite. *)

val predict : probe -> (float, 's, Nx.host) Nx.t -> Nx.host Nx.int64_t
(** [predict p x] is the class of each row of [x], the first of the greatest
    scores. *)

type scores = {
  accuracy : float;  (** The share of rows whose class is their label. *)
  f1 : float;
      (** The mean over labels of their F1 scores, each weighted by the label's
          share of the rows. A label's F1 is [2 t / (2 t + f + m)], [t] its rows
          predicted as it, [f] the other rows predicted as it, [m] its rows
          predicted as another, and [0] where [t] is [0]. *)
}
(** The type for a probe's scores on labelled rows. *)

val score : predicted:Nx.host Nx.int64_t -> Nx.host Nx.int64_t -> scores
(** [score ~predicted y] is the scores of the classes [predicted] against the
    labels [y].

    Raises [Invalid_argument] if [predicted] and [y] differ in shape or are
    empty. *)

(** {1:rankme RankMe} *)

val rankme : (float, 's, 'd) Nx.t -> float
(** [rankme z] is the entropy of [z]'s singular values [σ], normalised as a
    distribution, divided by its greatest value: [−Σ pᵢ log (pᵢ + ε) / log k]
    with [pᵢ = σᵢ / (Σ σ + ε)], [ε = 1e-7], for the [k = min n d] singular
    values of [z] ([[n; d]]). [z] is read on the host.

    Raises [Invalid_argument] if [z] is not [[n; d]] with [min n d ≥ 2]. *)
