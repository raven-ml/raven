(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** MCMC sampling with automatic gradients.

    Norn provides Markov chain Monte Carlo samplers that leverage {!Rune}'s
    automatic differentiation. The core abstraction is the {!type-kernel}: a
    composable [{init; step}] record that any algorithm produces and any
    sampling loop consumes.

    {b Quick start.}
    {[
      let result = Norn.nuts ~n:1000 log_prob (Nx.zeros Nx.float64 [| dim |])
    ]}

    For configured usage, construct a kernel and pass it to {!sample}:
    {[
      let result =
        Norn.sample ~n:1000 log_prob init (fun ~step_size ~metric ->
            Norn.nuts_kernel ~step_size ~metric ())
    ]} *)

(** {1:types Types} *)

type state = {
  position : Nx.float64_t;  (** Current sample, shape [[dim]]. *)
  log_density : float;  (** Log-density at {!position}. *)
  grad_log_density : Nx.float64_t;
      (** Gradient of log-density at {!position}, shape [[dim]]. *)
}
(** The type for sampler states. Shared across all gradient-based kernels. *)

type info = {
  acceptance_rate : float;
      (** Metropolis acceptance probability in \[0, 1\]. *)
  is_divergent : bool;  (** [true] when the energy error exceeds 1000. *)
  energy : float;  (** Total Hamiltonian energy of the proposal. *)
  num_integration_steps : int;  (** Leapfrog steps taken this transition. *)
}
(** The type for per-step diagnostics. *)

type kernel = {
  init : Nx.float64_t -> (Nx.float64_t -> Nx.float64_t) -> state;
      (** [init position log_density_fn] is the initial state at [position]. *)
  step : state -> (Nx.float64_t -> Nx.float64_t) -> state * info;
      (** [step state log_density_fn] is [(new_state, info)]. *)
}
(** The type for sampling kernels. Constructed by {!hmc_kernel}, {!nuts_kernel},
    etc. The [log_density_fn] argument is not baked in so the same kernel can be
    reused with different targets (e.g. tempering). *)

(** {1:integrators Integrators} *)

type integrator =
  (Nx.float64_t -> Nx.float64_t) ->
  Nx.float64_t ->
  Nx.float64_t ->
  Nx.float64_t ->
  (Nx.float64_t -> float * Nx.float64_t) ->
  float ->
  Nx.float64_t * Nx.float64_t * float * Nx.float64_t
(** The type for symplectic integrators.
    [integrator kinetic_energy_grad position momentum gradient grad_log_prob
     step_size] is [(new_pos, new_mom, new_log_density, new_grad)].

    [kinetic_energy_grad] is [M{^-1} p], the gradient of the kinetic energy with
    respect to momentum. For unit metric this is the identity. The kernel
    provides it from {!type-metric}[.scale]. *)

val leapfrog : integrator
(** [leapfrog] is the velocity Verlet integrator (second-order symplectic). *)

val mclachlan : integrator
(** [mclachlan] is McLachlan's two-stage integrator. Higher acceptance rates
    than {!leapfrog} on challenging posteriors (McLachlan 1995). Two gradient
    evaluations per step. *)

val yoshida : integrator
(** [yoshida] is Yoshida's fourth-order symplectic integrator. More accurate
    than {!leapfrog} at the cost of three gradient evaluations per step. *)

(** {1:metrics Metrics} *)

type metric = {
  sample_momentum : int -> Nx.float64_t;
      (** [sample_momentum dim] draws momentum from the kinetic energy
          distribution. *)
  kinetic_energy : Nx.float64_t -> float;
      (** [kinetic_energy p] is [0.5 * p{^T} M{^-1} p]. *)
  scale : Nx.float64_t -> Nx.float64_t;  (** [scale v] is [M{^-1} v]. *)
  is_turning : Nx.float64_t -> Nx.float64_t -> Nx.float64_t -> bool;
      (** [is_turning left_p right_p momentum_sum] is the U-turn criterion for
          NUTS trajectory termination. *)
}
(** The type for mass matrix metrics. Defines the geometry of the sampling
    space. *)

val unit_metric : int -> metric
(** [unit_metric dim] is the identity metric. Momentum sampled from [N(0, I)].
*)

val diagonal_metric : Nx.float64_t -> metric
(** [diagonal_metric inv_mass_diag] is a diagonal metric with the given inverse
    mass diagonal. *)

val dense_metric : Nx.float64_t -> metric
(** [dense_metric inv_mass_matrix] is a dense metric with the given inverse mass
    matrix. Uses Cholesky decomposition for momentum sampling. *)

(** {1:kernels Kernels} *)

val hmc_kernel :
  ?integrator:integrator ->
  ?num_leapfrog:int ->
  step_size:float ->
  metric:metric ->
  unit ->
  kernel
(** [hmc_kernel ~step_size ~metric ()] is a Hamiltonian Monte Carlo kernel.

    [integrator] defaults to {!leapfrog}. [num_leapfrog] defaults to [20]. *)

val nuts_kernel :
  ?integrator:integrator ->
  ?max_depth:int ->
  step_size:float ->
  metric:metric ->
  unit ->
  kernel
(** [nuts_kernel ~step_size ~metric ()] is a No-U-Turn Sampler kernel.

    NUTS automatically adapts the trajectory length using a binary tree
    expansion with U-turn detection. This eliminates the [num_leapfrog]
    parameter of {!hmc_kernel}.

    [integrator] defaults to {!leapfrog}. [max_depth] defaults to [10]. *)

(** {1:sampling Sampling} *)

type stats = {
  accept_rate : float;  (** Mean acceptance rate during sampling. *)
  step_size : float;  (** Final adapted step size. *)
  num_divergent : int;  (** Number of divergent transitions. *)
}
(** The type for aggregate sampling statistics. *)

type result = {
  samples : Nx.float64_t;  (** Shape [[n; dim]]. *)
  log_densities : Nx.float64_t;  (** Shape [[n]]. *)
  stats : stats;
}
(** The type for sampling results. *)

val sample :
  ?step_size:float ->
  ?target_accept:float ->
  ?num_warmup:int ->
  ?report:(step:int -> state -> info -> unit) ->
  n:int ->
  (Nx.float64_t -> Nx.float64_t) ->
  Nx.float64_t ->
  (step_size:float -> metric:metric -> kernel) ->
  result
(** [sample ~n log_prob init make_kernel] draws [n] samples from the
    distribution with unnormalized log-density [log_prob], starting at [init].

    During [num_warmup] iterations (discarded), step size and mass matrix are
    adapted using Stan-style window adaptation: an initial fast phase (step size
    only), doubling slow windows (step size + mass matrix with regularized
    Welford estimation), and a final fast phase.

    [step_size] defaults to [0.01]. [target_accept] defaults to [0.65].
    [num_warmup] defaults to [n / 2]. [report] is called after each step with
    negative step numbers during warmup. *)

val hmc :
  ?step_size:float ->
  ?target_accept:float ->
  ?num_leapfrog:int ->
  ?num_warmup:int ->
  n:int ->
  (Nx.float64_t -> Nx.float64_t) ->
  Nx.float64_t ->
  result
(** [hmc ~n log_prob init] draws [n] samples using Hamiltonian Monte Carlo with
    window adaptation.

    [step_size] defaults to [0.01]. [target_accept] defaults to [0.65].
    [num_leapfrog] defaults to [20]. [num_warmup] defaults to [n / 2]. *)

val nuts :
  ?step_size:float ->
  ?target_accept:float ->
  ?max_depth:int ->
  ?num_warmup:int ->
  n:int ->
  (Nx.float64_t -> Nx.float64_t) ->
  Nx.float64_t ->
  result
(** [nuts ~n log_prob init] draws [n] samples using the No-U-Turn Sampler with
    window adaptation.

    [step_size] defaults to [0.01]. [target_accept] defaults to [0.80].
    [max_depth] defaults to [10]. [num_warmup] defaults to [n / 2]. *)

(** {1:diagnostics Diagnostics} *)

val ess : Nx.float64_t -> Nx.float64_t
(** [ess samples] is the effective sample size for each parameter. [samples] has
    shape [[n; dim]], returns shape [[dim]]. Computed via autocorrelation with
    the initial monotone sequence estimator. *)

val rhat : Nx.float64_t array -> Nx.float64_t
(** [rhat chains] is the split R-hat convergence diagnostic for each parameter.
    Each chain has shape [[n; dim]], returns shape [[dim]]. Values close to
    [1.0] indicate convergence; above [1.01] suggests the chains have not mixed.
*)
