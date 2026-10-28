(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Weighted}. *)

type ('u, 'f, 'd) t = { values : 'u; log_weights : (float, 'f, 'd) Nx.t }

val ptree : ('u, 'd) Nx.Ptree.t -> (('u, 'f, 'd) t, 'd) Nx.Ptree.t

val ess : ('u, 'f, 'd) t -> (float, 'f, 'd) Nx.t

val resample : ('u, 'd) Nx.Ptree.t -> 'd Nx.Rng.key -> n:int -> ('u, 'f, 'd) t -> 'u
