(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Weighted draws.

    Weighted draws of a structure ['u] are a value of ['u] whose every tensor
    has one leading axis of [n] draws, and a log weight per draw. The weights
    need not be normalised, and a draw of log weight [-inf] has none. Importance
    sampling, sequential Monte Carlo and nested sampling return them. *)

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
