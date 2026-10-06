(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'f t = {
  lp : (float, 'f) Nx.t;
  acceptance : (float, 'f) Nx.t;
  step_size : (float, 'f) Nx.t;
  n_steps : Nx.int32_t;
  diverging : Nx.bool_t;
  saturated : Nx.bool_t;
  energy : (float, 'f) Nx.t;
}

type 'f stats = 'f t

(* The dtype only fixes ['f]: a structure built by an application is not
   polymorphic. *)
let ptree (type f) (_ : (float, f) Nx.dtype) : f t Nx.Ptree.t =
  let module S = struct
    type _ t = f stats

    let walk c s =
      let open Nx.Ptree.Walk in
      let lp = field c "lp" tensor s.lp in
      let acceptance = field c "acceptance" tensor s.acceptance in
      let step_size = field c "step_size" tensor s.step_size in
      let n_steps = field c "n_steps" tensor s.n_steps in
      let diverging = field c "diverging" tensor s.diverging in
      let saturated = field c "saturated" tensor s.saturated in
      let energy = field c "energy" tensor s.energy in
      { lp; acceptance; step_size; n_steps; diverging; saturated; energy }
  end in
  Nx.Ptree.nest (module S) Nx.Ptree.unit
