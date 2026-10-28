(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('f, 'd) t = {
  lp : (float, 'f, 'd) Nx.t;
  acceptance : (float, 'f, 'd) Nx.t;
  step_size : (float, 'f, 'd) Nx.t;
  n_steps : 'd Nx.int32_t;
  diverging : 'd Nx.bool_t;
  saturated : 'd Nx.bool_t;
  energy : (float, 'f, 'd) Nx.t;
}

type ('f, 'd) stats = ('f, 'd) t

(* The dtype only fixes ['f]: a structure built by an application is not
   polymorphic. *)
let ptree (type f) (_ : (float, f) Nx.dtype) : ((f, 'd) t, 'd) Nx.Ptree.t =
  let walk c (s : (f, 'd) stats) : (f, 'd) stats =
    let open Nx.Ptree.Walk in
    let lp = field c "lp" tensor s.lp in
    let acceptance = field c "acceptance" tensor s.acceptance in
    let step_size = field c "step_size" tensor s.step_size in
    let n_steps = field c "n_steps" tensor s.n_steps in
    let diverging = field c "diverging" tensor s.diverging in
    let saturated = field c "saturated" tensor s.saturated in
    let energy = field c "energy" tensor s.energy in
    { lp; acceptance; step_size; n_steps; diverging; saturated; energy }
  in
  Nx.Ptree.v walk
