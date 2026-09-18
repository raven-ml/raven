(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type state = {
  position : Nx.float64_t;
  log_density : float;
  grad_log_density : Nx.float64_t;
}

type info = {
  acceptance_rate : float;
  is_divergent : bool;
  energy : float;
  num_integration_steps : int;
}

type kernel = {
  init : Nx.float64_t -> (Nx.float64_t -> Nx.float64_t) -> state;
  step : state -> (Nx.float64_t -> Nx.float64_t) -> state * info;
}

type integrator =
  (Nx.float64_t -> Nx.float64_t) ->
  Nx.float64_t ->
  Nx.float64_t ->
  Nx.float64_t ->
  (Nx.float64_t -> float * Nx.float64_t) ->
  float ->
  Nx.float64_t * Nx.float64_t * float * Nx.float64_t

type metric = {
  sample_momentum : int -> Nx.float64_t;
  kinetic_energy : Nx.float64_t -> float;
  scale : Nx.float64_t -> Nx.float64_t;
  is_turning : Nx.float64_t -> Nx.float64_t -> Nx.float64_t -> bool;
}

type stats = { accept_rate : float; step_size : float; num_divergent : int }

type result = {
  samples : Nx.float64_t;
  log_densities : Nx.float64_t;
  stats : stats;
}
