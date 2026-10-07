(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layer's solves. This is the only module that names jera: the inverses
   of the distortion families, and the deprojections of ZPN and AIR. Each
   answer is stated as a zero of its equation, so its derivative is the
   implicit one and no iteration is differentiated. A lane that does not
   converge keeps its last estimate and is reported in the mask. *)

type v = (float, Nx.float64_elt) Nx.t

(* A float64 zero located to its last bits: two units in the last place of
   the answer, and an absolute floor for an answer at zero. *)
let eps = Float.epsilon
let bracket_tol = Jera.Tol.v ~rel:(2. *. eps) ~abs:(4. *. eps)

let bracket f ~lo ~hi =
  let s = Jera.Root.bracket ~tol:bracket_tol f ~lo ~hi in
  (Jera.Solution.best s, Jera.Solution.ok s)

(* Newton's method from a seed within a pixel or a fraction of a degree
   converges quadratically in a few steps; the budget bounds a lane that does
   not. *)
let lanes_budget = 20

(* [lanes ~scale ~jacobian f guess] solves [f u = 0] in each lane of [guess],
   [[...; k]], to [scale] times float64's resolution, [scale] the size of the
   points' coordinates. *)
let lanes ~scale ~jacobian f guess =
  let tol = Jera.Tol.v ~rel:(4. *. eps) ~abs:(4. *. eps *. scale) in
  let s = Jera.System.lanes ~tol ~budget:lanes_budget ~jacobian f guess in
  (Jera.Solution.best s, Jera.Solution.ok s)
