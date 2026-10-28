(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Stats}. *)

type ('f, 'd) t = {
  lp : (float, 'f, 'd) Nx.t;
  acceptance : (float, 'f, 'd) Nx.t;
  step_size : (float, 'f, 'd) Nx.t;
  n_steps : 'd Nx.int32_t;
  diverging : 'd Nx.bool_t;
  saturated : 'd Nx.bool_t;
  energy : (float, 'f, 'd) Nx.t;
}

val ptree : (float, 'f) Nx.dtype -> (('f, 'd) t, 'd) Nx.Ptree.t
