(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ymir
open Jera

type data = {
  z_hd : Nx.float64_t;
  z_hel : Nx.float64_t;
  m_b : Nx.float64_t;
  chol : Nx.float64_t;
}

type model = Curved | Flat

let planck = Cosmology.planck2018 ~codata:Codata.v2022 Nx.float64

(* The 1999 model: T_CMB = 0 removes the photons and every neutrino. *)
let matter_and_lambda =
  { planck with t_cmb = Quantity.v Unit.kelvin (Nx.scalar Nx.float64 0.) }

let at theta i = Nx.get [ i ] theta

let omegas model theta =
  match model with
  | Curved -> (at theta 0, at theta 1, at theta 2)
  | Flat -> (at theta 0, Nx.rsub_s 1. (at theta 0), at theta 1)

let cosmology model theta =
  let omega_m, omega_l, _ = omegas model theta in
  {
    matter_and_lambda with
    omega_cb = omega_m;
    omega_k = Nx.sub (Nx.rsub_s 1. omega_m) omega_l;
  }

let residuals d model theta =
  let _, _, offset = omegas model theta in
  let mu =
    Cosmology.distance_modulus (cosmology model theta) ~observed:d.z_hel d.z_hd
  in
  let r = Nx.sub d.m_b (Nx.add mu offset) in
  Nx.solve_triangular d.chol r

let lm = Minimize.levenberg_marquardt Nx.Ptree.tensor ~linear:Linear.dense

let fit d model theta0 =
  Minimize.solve Nx.Ptree.tensor lm
    ~tol:(Tol.v ~rel:1e-6 ~abs:1e-9)
    ~budget:100 (residuals d model) theta0

let covariance d model theta =
  let j = Rune.jacfwd' (residuals d model) theta in
  Nx.inv (Nx.matmul (Nx.transpose j) j)

let chi2 d model theta =
  let r = residuals d model theta in
  Nx.item [] (Nx.sum (Nx.mul r r))
