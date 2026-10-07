(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Supernova fits: magnitudes m of covariance C = L Lᵀ against a model μ(θ), by
   Levenberg–Marquardt on the whitened residual L⁻¹ (m − μ(θ)), whose squared
   norm is the χ² of the full covariance.

   The fit's covariance is the data's propagated through the answer: G = ∂θ̂/∂m,
   rune's derivative of the minimum, and G C Gᵀ = (G L)(G L)ᵀ. The Fisher
   matrix's inverse (JᵀJ)⁻¹, with J the whitened residual's Jacobian at the
   answer, is the same where the residual vanishes and differs by the residual's
   curvature elsewhere. *)

open Ymir
open Jera

type vector = (float, Nx.float64_elt) Nx.t

(* theta is the minimum [p]; cov is G C Gᵀ and fisher (JᵀJ)⁻¹, both [p; p]. *)
type fit = {
  theta : vector;
  cov : vector;
  fisher : vector;
  chi2 : float;
  evaluations : int;
}

(* A minimum is determined to about the square root of chi^2's rounding over
   its curvature: 3e-8 of Pantheon+'s Omega_m. *)
let tol = Tol.v ~rel:1e-7 ~abs:1e-9
let budget = 100

let fit ~chol ~model m theta0 =
  let residual m theta = Nx.solve_triangular chol (Nx.sub m (model theta)) in
  let lm = Minimize.levenberg_marquardt Nx.Ptree.tensor ~linear:Linear.dense in
  let solve m =
    Minimize.solve Nx.Ptree.tensor lm ~tol ~budget (residual m) theta0
  in
  let s = solve m in
  let theta = Solution.get s in
  let g = Rune.jacrev' (fun m -> Solution.get (solve m)) m in
  let gl = Nx.matmul g chol in
  let j = Rune.jacfwd' (residual m) theta in
  let r = residual m theta in
  {
    theta;
    cov = Nx.matmul gl (Nx.transpose gl);
    fisher = Nx.inv (Nx.matmul (Nx.transpose j) j);
    chi2 = Nx.item [] (Nx.sum (Nx.square r));
    evaluations = Int32.to_int (Nx.item [] (Solution.evaluations s));
  }

let sigma cov i = Float.sqrt (Nx.item [ i; i ] cov)

(* Models *)

let f64 x = Nx.scalar Nx.float64 x
let km_s_mpc = Unit.(kilo metre / second / mega Units.parsec)

(* Planck 2018 without radiation: T_CMB = 0 removes the photons and every
   neutrino, and dark energy closes the budget. *)
let matter_and_lambda =
  {
    (Cosmology.planck2018 ~codata:Codata.v2022 Nx.float64) with
    t_cmb = Quantity.v Unit.kelvin (f64 0.);
  }

let lcdm ?h0 ~omega_m ~omega_l () =
  let c = matter_and_lambda in
  {
    c with
    h0 = (match h0 with None -> c.h0 | Some h -> Quantity.v km_s_mpc h);
    omega_cb = omega_m;
    omega_k = Nx.sub (Nx.sub (f64 1.) omega_m) omega_l;
  }
