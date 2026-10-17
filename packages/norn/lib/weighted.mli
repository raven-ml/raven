(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Weighted}. *)

type ('u, 'f) t = { values : 'u; log_weights : (float, 'f) Nx.t }

val ptree : 'u Nx.Ptree.t -> ('u, 'f) t Nx.Ptree.t

val ess : ('u, 'f) t -> (float, 'f) Nx.t

val resample : 'u Nx.Ptree.t -> Nx.Rng.t -> n:int -> ('u, 'f) t -> 'u
