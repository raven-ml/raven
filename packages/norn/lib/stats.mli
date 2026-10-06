(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Statistics of Hamiltonian transitions.

    Each field holds one element per chain, with the axes of the chains it
    describes: a sampler's state holds [[chain]], and its draws [[chain; draw]].
*)

type 'f t = {
  lp : (float, 'f) Nx.t;  (** The log density at the transition's end. *)
  acceptance : (float, 'f) Nx.t;
      (** The trajectory's mean of [min (1, exp (H0 - H))], [H0] the Hamiltonian
          after the momentum draw. *)
  step_size : (float, 'f) Nx.t;  (** The step size the transition took. *)
  n_steps : Nx.int32_t;  (** The leapfrog steps it took. *)
  diverging : Nx.bool_t;
      (** Whether the energy error exceeded 1000 or a position was not finite.
      *)
  saturated : Nx.bool_t;
      (** Whether the trajectory reached its maximum length before it turned. *)
  energy : (float, 'f) Nx.t;
      (** The Hamiltonian after the momentum draw, [H0]. *)
}
(** The type for statistics of transitions over element type ['f]. *)

val ptree : (float, 'f) Nx.dtype -> 'f t Nx.Ptree.t
(** [ptree dtype] is the structure of statistics at [dtype], with tensors at
    [lp], [acceptance], [step_size], [n_steps], [diverging], [saturated] and
    [energy]. *)
