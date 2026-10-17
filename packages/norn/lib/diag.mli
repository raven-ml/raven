(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Diag}. *)

(** {1:convergence Convergence} *)

val rhat : 'u Nx.Ptree.t -> 'u Draws.t -> 'u

val nested_rhat : 'u Nx.Ptree.t -> superchains:int -> 'u Draws.t -> 'u

val ess_bulk : 'u Nx.Ptree.t -> 'u Draws.t -> 'u

val ess_tail : 'u Nx.Ptree.t -> 'u Draws.t -> 'u

val mcse_mean : 'u Nx.Ptree.t -> 'u Draws.t -> 'u

(** {1:hamiltonian Hamiltonian transitions} *)

val ebfmi : 'f Stats.t Draws.t -> (float, 'f) Nx.t

val divergent_shift : 'u Nx.Ptree.t -> 'f Stats.t Draws.t -> 'u Draws.t -> 'u

(** {1:calibration Calibration} *)

val rank : 'u Nx.Ptree.t -> truth:'u -> 'u Draws.t -> 'u

val rank_uniformity : 'u Nx.Ptree.t -> draws:int -> 'u list -> 'u
