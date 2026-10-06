(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Evidence: the normalising constant of a posterior, with its draws.

    For a prior [π] and a likelihood [L] the evidence is [Z = ∫ L π], the
    posterior's normaliser and the probability of the data under the model.
    {!Norn.Nested} and {!Norn.Smc} estimate it, and their particles weighted by
    the posterior are its {!sample}. *)

type ('u, 'f) t
(** The type for evidence estimates over positions ['u]. *)

val log_evidence : ('u, 'f) t -> (float, 'f) Nx.t
(** [log_evidence z] is the estimate of [ln Z], a scalar. *)

val error : ('u, 'f) t -> (float, 'f) Nx.t
(** [error z] is the estimate's standard error in [ln Z], a scalar. *)

val information : ('u, 'f) t -> (float, 'f) Nx.t
(** [information z] is [H], the posterior's Kullback-Leibler divergence from the
    prior in nats, [E_post (ln L) - ln Z]: how much the data narrowed the prior.
*)

val sample : ('u, 'f) t -> ('u, 'f) Weighted.t
(** [sample z] is draws weighted by the posterior, their log weights normalised.
*)

(** {1:stop Why a run stopped} *)

(** The type for the reasons a run stopped. A spent budget is a result, not an
    error: the estimate so far, with how far it fell short. *)
type stop =
  | Converged
      (** The run finished: nested sampling's live points could add less than
          its tolerance to [ln Z], tempering reached [β = 1]. *)
  | Remaining of float
      (** Nested sampling's budget ran out while its live points could still add
          up to this many nats to [ln Z]. *)
  | Temperature of float
      (** Tempering's budget ran out at this temperature [β < 1]: the estimate
          is of [ln ∫ L^β π], the sample of its tempered posterior. *)

val stop : ('u, 'f) t -> stop
(** [stop z] is why the run that gave [z] stopped. It reads [z]'s tensors, so it
    waits for a computation that produces them. *)

(** {1:structure Structure and formatting} *)

val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t
(** [ptree u] is the structure of evidence estimates over [u], so a compiled
    function may return one. *)

val pp : Format.formatter -> ('u, 'f) t -> unit
(** [pp] formats an estimate as [ln Z = -412.31 ± 0.08, H = 5.20 nats],
    followed, for a spent budget, by [, budget spent with 0.52 nats left] or
    [, budget spent at β = 0.73]. *)

(**/**)

val v :
  sample:('u, 'f) Weighted.t ->
  log_evidence:(float, 'f) Nx.t ->
  error:(float, 'f) Nx.t ->
  information:(float, 'f) Nx.t ->
  stop:Nx.int32_t ->
  reached:(float, 'f) Nx.t ->
  ('u, 'f) t
(* [v ... ~stop ~reached] is an estimate that stopped as [stop] codes it:
   {!converged}, {!remaining} or {!temperature}, whose payload is [reached], a
   scalar. *)

val converged : int32
val remaining : int32
val temperature : int32
