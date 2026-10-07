(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The 1999 supernova cosmology fit.

    A supernova's corrected apparent magnitude is its distance modulus plus an
    offset, the absolute magnitude less 5 log₁₀ h, which supernovae alone do not
    separate: H₀ stays at the realisation's value and the offset absorbs both.
    The model has no radiation, so dark energy closes the budget of matter and
    curvature. The fit minimises χ² = rᵀ C⁻¹ r by Levenberg and Marquardt on the
    whitened residuals L⁻¹ r, with C = L Lᵀ. *)

open Ymir

type data = {
  z_hd : Nx.float64_t;  (** The redshift that places each supernova, [[n]]. *)
  z_hel : Nx.float64_t;  (** The redshift its observer measures, [[n]]. *)
  m_b : Nx.float64_t;  (** Its corrected apparent magnitude, [[n]]. *)
  chol : Nx.float64_t;
      (** The lower Cholesky factor of the magnitudes' covariance, [[n; n]]. *)
}

(** The models, each over a vector θ of its parameters. *)
type model =
  | Curved  (** θ = (Ω_m, Ω_Λ, offset), curvature free. *)
  | Flat  (** θ = (Ω_m, offset), Ω_Λ = 1 − Ω_m. *)

val cosmology : model -> Nx.float64_t -> Nx.float64_t Cosmology.t
(** [cosmology m theta] is the cosmology θ describes: Planck 2018's H₀ with no
    radiation, Ω_cb = Ω_m and Ω_k = 1 − Ω_m − Ω_Λ. *)

val residuals : data -> model -> Nx.float64_t -> Nx.float64_t
(** [residuals d m theta] is L⁻¹ (m_b − μ − offset), with μ the distance modulus
    at [z_hd] observed at [z_hel]. *)

val fit : data -> model -> Nx.float64_t -> Nx.float64_t Jera.Solution.t
(** [fit d m theta0] is the θ that minimises χ², from [theta0]. *)

val covariance : data -> model -> Nx.float64_t -> Nx.float64_t
(** [covariance d m theta] is (JᵀJ)⁻¹, with J the Jacobian of {!residuals} at
    [theta]: the covariance of the fit at its answer. *)

val chi2 : data -> model -> Nx.float64_t -> float
(** [chi2 d m theta] is χ² at [theta]. *)
