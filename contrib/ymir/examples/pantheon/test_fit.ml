(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Pantheon+ fit's path on a synthetic sample of 40 supernovae with
   correlated errors: magnitudes made by the model at a known θ are fitted
   back to it; at that zero-residual answer the fit's covariance (JᵀJ)⁻¹ is
   the covariance that the derivative of the answer with respect to the
   magnitudes carries from C; and noisy samples fit within 4σ of the truth. *)

open Windtrap
open Jera
open Pantheon_fit

let f64 = Nx.float64
let n = 40

(* Redshifts from 0.02 to 2.3, observed with a peculiar velocity of up to
   300 km/s, and errors of 0.15 mag with 0.03 mag common to all. *)
let sample () =
  let z_hd = Nx.geomspace f64 0.02 2.3 n in
  let z_hel =
    Nx.add z_hd
      (Nx.mul_s (Nx.sin (Nx.arange_f f64 0. (Float.of_int n) 1.)) 1e-3)
  in
  let c =
    Nx.add
      (Nx.mul_s (Nx.eye f64 n) (0.15 *. 0.15))
      (Nx.full f64 [| n; n |] 9e-4)
  in
  (z_hd, z_hel, c)

let data_at model theta ~noise =
  let z_hd, z_hel, c = sample () in
  let chol = Nx.cholesky c in
  let blank = { Fit.z_hd; z_hel; chol; m_b = Nx.zeros f64 [| n |] } in
  (* Residuals of zero magnitudes are -L⁻¹(μ + offset). *)
  let model_mb = Nx.neg (Nx.matmul chol (Fit.residuals blank model theta)) in
  let m_b = Nx.add model_mb (Nx.matmul chol noise) in
  ({ blank with m_b }, c)

let truth = function
  | Fit.Curved -> Nx.create f64 [| 3 |] [| 0.3; 0.7; -19.35 |]
  | Flat -> Nx.create f64 [| 2 |] [| 0.3; -19.35 |]

let start = function
  | Fit.Curved -> Nx.create f64 [| 3 |] [| 0.4; 0.5; -19. |]
  | Flat -> Nx.create f64 [| 2 |] [| 0.4; -19. |]

let models = [ ("curved", Fit.Curved); ("flat", Fit.Flat) ]
let vec = Nx.to_array

let recovered =
  cases ~name:fst "noise-free magnitudes fit back to the truth" models
    (fun (_, model) ->
      let d, _ = data_at model (truth model) ~noise:(Nx.zeros f64 [| n |]) in
      let theta = Solution.get (Fit.fit d model (start model)) in
      equal (array (float 1e-8)) (vec (truth model)) (vec theta))

let covariance =
  test "the fit's covariance is the derivative of the answer carried from C"
    (fun () ->
      let model = Fit.Curved in
      let d, c = data_at model (truth model) ~noise:(Nx.zeros f64 [| n |]) in
      let answer m_b =
        Solution.best (Fit.fit { d with m_b } model (start model))
      in
      let g = Rune.jacfwd' answer d.m_b in
      let carried = Nx.matmul (Nx.matmul g c) (Nx.transpose g) in
      let fisher = Fit.covariance d model (truth model) in
      equal
        (array (float_rel ~rel:1e-8 ~abs:0.))
        (vec (Nx.flatten fisher))
        (vec (Nx.flatten carried)))

let noisy =
  cases ~name:string_of_int "noisy magnitudes fit within 4 sigma" [ 1; 2; 3 ]
    (fun seed ->
      let model = Fit.Curved in
      let noise = Nx.Rng.normal (Nx.Rng.key seed) f64 [| n |] in
      let d, _ = data_at model (truth model) ~noise in
      let theta = Solution.get (Fit.fit d model (start model)) in
      let sigma = Nx.sqrt (Nx.diagonal (Fit.covariance d model theta)) in
      let pull = Nx.abs (Nx.div (Nx.sub theta (truth model)) sigma) in
      Array.iter (fun p -> less float_exact ~than:4. p) (vec pull))

let () = exit (run "Pantheon fit" [ recovered; covariance; noisy ])
