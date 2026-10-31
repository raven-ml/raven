(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Diag}. *)

(** {1:convergence Convergence} *)

val rhat : ('u, 'd) Nx.Ptree.t -> 'u Draws.t -> 'u

val nested_rhat : ('u, 'd) Nx.Ptree.t -> superchains:int -> 'u Draws.t -> 'u

val ess_bulk : ('u, 'd) Nx.Ptree.t -> 'u Draws.t -> 'u

val ess_tail : ('u, 'd) Nx.Ptree.t -> 'u Draws.t -> 'u

val mcse_mean : ('u, 'd) Nx.Ptree.t -> 'u Draws.t -> 'u

(** {1:hamiltonian Hamiltonian transitions} *)

val ebfmi : ('f, 'd) Stats.t Draws.t -> (float, 'f, 'd) Nx.t

val divergent_shift : ('u, 'd) Nx.Ptree.t -> ('f, 'd) Stats.t Draws.t -> 'u Draws.t -> 'u

(** {1:calibration Calibration} *)

val rank : ('u, 'd) Nx.Ptree.t -> truth:'u -> 'u Draws.t -> 'u

val rank_uniformity : ('u, 'd) Nx.Ptree.t -> draws:int -> 'u list -> 'u

(** {1:importance Importance sampling} *)

type ('f, 'd) psis = {
  log_weights : (float, 'f, 'd) Nx.t;
  khat : (float, 'f, 'd) Nx.t;
}

val psis : (float, 'f, 'd) Nx.t -> ('f, 'd) psis

type ('f, 'd) loo = {
  elpd : (float, 'f, 'd) Nx.t;
  se : (float, 'f, 'd) Nx.t;
  p : (float, 'f, 'd) Nx.t;
  pointwise : (float, 'f, 'd) Nx.t;
  khat : (float, 'f, 'd) Nx.t;
}

val loo : (float, 'f, 'd) Nx.t Draws.t -> ('f, 'd) loo
