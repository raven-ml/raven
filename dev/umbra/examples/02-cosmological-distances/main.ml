(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differentiable cosmological parameter fitting from Type Ia supernova distance
   moduli.

   Fits H0 (Hubble constant) and Omega_m (matter density fraction) by gradient
   descent on the distance modulus residuals. The forward model uses
   Umbra.Cosmo.distance_modulus directly -- its Gauss-Legendre quadrature,
   luminosity distance, and distance modulus are all Nx tensor operations,
   making them natively differentiable through Rune's autodiff.

   Also demonstrates basic cosmological distance queries: comoving distance,
   luminosity distance, angular diameter distance, lookback time, and the age of
   the universe at various redshifts. *)

open Nx
open Umbra

let f64 = Nx.float64

(* --- Part 1: Distance table for the Planck 2018 cosmology --- *)

let print_distance_table () =
  Printf.printf "Cosmological distances (Planck 2018)\n";
  Printf.printf "====================================\n\n";
  let p = Cosmo.planck18 in
  Printf.printf "  H0      = %.2f km/s/Mpc\n" (item [] (Cosmo.h0 p));
  Printf.printf "  Omega_m = %.4f\n" (item [] (Cosmo.omega_m p));
  Printf.printf "  Omega_L = %.4f\n\n" (item [] (Cosmo.omega_l p));

  Printf.printf "%6s  %10s  %10s  %10s  %8s  %8s\n" "z" "d_C (Mpc)" "d_L (Mpc)"
    "d_A (Mpc)" "mu" "t_lb (Gyr)";
  Printf.printf "%6s  %10s  %10s  %10s  %8s  %8s\n" "------" "----------"
    "----------" "----------" "--------" "----------";
  let redshifts = [| 0.01; 0.1; 0.3; 0.5; 1.0; 2.0; 3.0; 5.0 |] in
  Array.iter
    (fun z ->
      let zv = scalar f64 z in
      let d_c = item [] (Unit.Length.in_mpc (Cosmo.comoving_distance ~p zv)) in
      let d_l =
        item [] (Unit.Length.in_mpc (Cosmo.luminosity_distance ~p zv))
      in
      let d_a =
        item [] (Unit.Length.in_mpc (Cosmo.angular_diameter_distance ~p zv))
      in
      let mu = item [] (Cosmo.distance_modulus ~p zv) in
      let t_lb = item [] (Unit.Time.in_gyr (Cosmo.lookback_time ~p zv)) in
      Printf.printf "%6.2f  %10.1f  %10.1f  %10.1f  %8.2f  %8.2f\n" z d_c d_l
        d_a mu t_lb)
    redshifts;
  Printf.printf "\n";

  (* Age of the universe *)
  let age_now = item [] (Unit.Time.in_gyr (Cosmo.age ~p (scalar f64 0.0))) in
  Printf.printf "  Age of the universe (z=0): %.2f Gyr\n\n" age_now

(* --- Part 2: Fit H0 and Omega_m from SN Ia data --- *)

(* Representative SN Ia data points (z, observed distance modulus). Based on
   Pantheon+ compilation values for flat LCDM with H0 ~ 73, Omega_m ~ 0.3. *)
let z_arr = [| 0.01; 0.03; 0.08; 0.15; 0.25; 0.40; 0.55; 0.70; 0.85; 1.00 |]
let n_sn = Array.length z_arr

let mu_obs =
  [| 33.07; 35.47; 37.62; 39.07; 40.24; 41.42; 42.23; 42.85; 43.34; 43.74 |]

(* Forward model: compute distance modulus for all SNe. The differentiable
   parameters are H0 and Omega_m, which flow through Cosmo.distance_modulus via
   Nx tensor operations. *)
let loss params =
  match params with
  | [ h0; omega_m ] ->
      let p = Cosmo.create_flat_lcdm ~h0 ~omega_m in
      let total = ref (scalar f64 0.0) in
      for i = 0 to n_sn - 1 do
        let z_i = scalar f64 z_arr.(i) in
        let mu_pred = Cosmo.distance_modulus ~p z_i in
        let mu_obs_i = scalar f64 mu_obs.(i) in
        let residual = sub mu_pred mu_obs_i in
        total := add !total (square residual)
      done;
      div_s !total (Float.of_int n_sn)
  | _ -> failwith "expected [h0; omega_m]"

let fit_cosmology () =
  Printf.printf "Fitting H0 and Omega_m from Type Ia supernovae\n";
  Printf.printf "===============================================\n";
  Printf.printf "  Data: %d distance moduli (Pantheon+-like)\n" n_sn;
  Printf.printf "  Method: Adam optimizer, 300 steps\n";
  Printf.printf "  Model: flat LCDM via Cosmo.distance_modulus\n\n";

  let algo = Vega.adam (Vega.Schedule.constant 0.5) in
  let h0 = ref (scalar f64 65.0) in
  let omega_m = ref (scalar f64 0.25) in
  let states = [| Vega.init algo !h0; Vega.init algo !omega_m |] in
  let steps = 300 in

  Printf.printf "%5s  %10s  %8s  %8s\n" "step" "loss" "H0" "Omega_m";
  Printf.printf "%5s  %10s  %8s  %8s\n" "-----" "----------" "--------"
    "--------";

  let refs = [| h0; omega_m |] in
  for i = 0 to steps - 1 do
    let loss_val, grads = Rune.value_and_grads loss [ !h0; !omega_m ] in
    List.iteri
      (fun j g ->
        let p, s = Vega.step states.(j) ~grad:g ~param:!(refs.(j)) in
        refs.(j) := p;
        states.(j) <- s)
      grads;
    if i mod 50 = 0 || i = steps - 1 then
      Printf.printf "%5d  %10.6f  %8.2f  %8.4f\n" i (item [] loss_val)
        (item [] !h0) (item [] !omega_m)
  done;

  Printf.printf "\nFitted parameters:\n";
  Printf.printf "  H0      = %.2f km/s/Mpc\n" (item [] !h0);
  Printf.printf "  Omega_m = %.4f\n" (item [] !omega_m)

let () =
  print_distance_table ();
  fit_cosmology ()
