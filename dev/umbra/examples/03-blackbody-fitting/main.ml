(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differentiable blackbody SED fitting.

   Given broadband photometric measurements in UGRIZ bands, fit the stellar
   effective temperature and luminosity normalization by gradient descent on the
   chi-squared statistic. The Planck function is evaluated as Nx tensor
   operations, making it fully differentiable through Rune.

   Uses Umbra.Const for physical constants. *)

open Nx
open Umbra

let f64 = Nx.float64

(* Central wavelengths of SDSS UGRIZ bands in meters *)
let lambda =
  create f64 [| 5 |] [| 3.551e-7; 4.686e-7; 6.166e-7; 7.480e-7; 8.932e-7 |]

(* Physical constants from Umbra *)
let h_planck = Const.h_si
let c_light = Unit.to_float Const.c
let k_boltz = Const.k_b_si

(* Pre-computed constant tensors *)
let two_hc2 = scalar f64 (2.0 *. h_planck *. c_light *. c_light)
let hc_over_k = scalar f64 (h_planck *. c_light /. k_boltz)
let lam5 = pow_s lambda 5.0

(* Generate synthetic observations from a Sun-like star *)
let true_temp = 5800.0
let true_log_norm = -50.0

let planck_scalar lam_m temp =
  let x = h_planck *. c_light /. (lam_m *. k_boltz *. temp) in
  2.0 *. h_planck *. c_light *. c_light
  /. (lam_m *. lam_m *. lam_m *. lam_m *. lam_m)
  /. (Float.exp x -. 1.0)

let flux_obs =
  let norm = Float.exp true_log_norm in
  let fluxes =
    Array.init 5 (fun i ->
        let lam_m =
          [| 3.551e-7; 4.686e-7; 6.166e-7; 7.480e-7; 8.932e-7 |].(i)
        in
        norm
        *. planck_scalar lam_m true_temp
        *. (1.0 +. (0.02 *. (Float.of_int i -. 2.0))))
  in
  create f64 [| 5 |] fluxes

(* Fractional errors: 5% photometry *)
let flux_err = mul_s flux_obs 0.05
let band_names = [| "U"; "G"; "R"; "I"; "Z" |]

(* Differentiable forward model: Planck function at 5 wavelengths. Parameterized
   in log-space for positivity and gradient conditioning.

   B(lambda, T) = 2hc^2 / lambda^5 / (exp(hc / (lambda * k * T)) - 1) *)
let loss params =
  match params with
  | [ log_temp; log_norm ] ->
      let temp = exp log_temp in
      let norm = exp log_norm in
      let exponent = div hc_over_k (mul lambda temp) in
      let planck =
        div (div two_hc2 lam5) (sub (exp exponent) (scalar f64 1.0))
      in
      let flux_pred = mul norm planck in
      let residual = div (sub flux_pred flux_obs) flux_err in
      sum (square residual)
  | _ -> failwith "expected [log_temp; log_norm]"

let () =
  Printf.printf "Differentiable blackbody SED fitting\n";
  Printf.printf "====================================\n";
  Printf.printf "Fitting temperature and normalization to UGRIZ photometry\n\n";

  Printf.printf "True parameters:\n";
  Printf.printf "  T    = %.0f K\n" true_temp;
  Printf.printf "  logA = %.1f\n\n" true_log_norm;

  Printf.printf "Synthetic observations (5%% errors):\n";
  for i = 0 to 4 do
    Printf.printf "  %s: %.4e +/- %.4e\n" band_names.(i) (item [ i ] flux_obs)
      (item [ i ] flux_err)
  done;
  Printf.printf "\n";

  (* Start from a guess *)
  let algo = Vega.adam (Vega.Schedule.constant 1e-2) in
  let log_temp = ref (scalar f64 (Float.log 5000.0)) in
  let log_norm = ref (scalar f64 (-52.0)) in
  let states = [| Vega.init algo !log_temp; Vega.init algo !log_norm |] in
  let steps = 500 in

  Printf.printf "%5s  %12s  %8s  %10s\n" "step" "chi2" "T (K)" "log_norm";
  Printf.printf "%5s  %12s  %8s  %10s\n" "-----" "------------" "--------"
    "----------";

  let refs = [| log_temp; log_norm |] in
  for i = 0 to steps - 1 do
    let loss_val, grads = Rune.value_and_grads loss [ !log_temp; !log_norm ] in
    List.iteri
      (fun j g ->
        let p, s = Vega.step states.(j) ~grad:g ~param:!(refs.(j)) in
        refs.(j) := p;
        states.(j) <- s)
      grads;
    if i mod 100 = 0 || i = steps - 1 then
      Printf.printf "%5d  %12.4f  %8.1f  %10.3f\n" i (item [] loss_val)
        (Float.exp (item [] !log_temp))
        (item [] !log_norm)
  done;

  Printf.printf "\nFitted parameters:\n";
  Printf.printf "  T    = %.1f K  (true: %.0f K)\n"
    (Float.exp (item [] !log_temp))
    true_temp;
  Printf.printf "  logA = %.3f  (true: %.1f)\n" (item [] !log_norm)
    true_log_norm
