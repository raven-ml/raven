(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Stats}. *)

type 'f t = {
  lp : (float, 'f) Nx.t;
  acceptance : (float, 'f) Nx.t;
  step_size : (float, 'f) Nx.t;
  n_steps : Nx.int32_t;
  diverging : Nx.bool_t;
  saturated : Nx.bool_t;
  energy : (float, 'f) Nx.t;
}

val ptree : (float, 'f) Nx.dtype -> 'f t Nx.Ptree.t
