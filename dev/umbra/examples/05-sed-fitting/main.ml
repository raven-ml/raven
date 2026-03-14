(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differentiable SED fitting: temperature + extinction from photometry.

   Demonstrates the composable differentiable pipeline: Spectrum.blackbody ->
   Extinction.apply -> Photometry.ab_mag

   All operations flow through Nx tensor ops, making the entire pipeline
   differentiable via Rune's autodiff. We fit stellar temperature, dust
   extinction, and flux normalization simultaneously by gradient descent on
   photometric residuals. *)

open Nx
open Umbra

let f64 = Nx.float64

(* Define 5 broadband filters (UGRIZ-like tophats) *)
let n_bp = 100

let band_u =
  Photometry.tophat ~lo:(Unit.Length.m 3.0e-7) ~hi:(Unit.Length.m 4.0e-7)
    ~n:n_bp

let band_g =
  Photometry.tophat ~lo:(Unit.Length.m 4.0e-7) ~hi:(Unit.Length.m 5.5e-7)
    ~n:n_bp

let band_r =
  Photometry.tophat ~lo:(Unit.Length.m 5.5e-7) ~hi:(Unit.Length.m 7.0e-7)
    ~n:n_bp

let band_i =
  Photometry.tophat ~lo:(Unit.Length.m 7.0e-7) ~hi:(Unit.Length.m 8.5e-7)
    ~n:n_bp

let band_z =
  Photometry.tophat ~lo:(Unit.Length.m 8.5e-7) ~hi:(Unit.Length.m 1.0e-6)
    ~n:n_bp

let bands = [ band_u; band_g; band_r; band_i; band_z ]
let band_names = [| "U"; "G"; "R"; "I"; "Z" |]

(* True parameters *)
let true_temp = 6500.0 (* K -- F-type star *)
let true_av = 0.5 (* moderate extinction *)
let true_log_norm = -50.0

(* Generate synthetic observations *)
let rv = Nx.scalar f64 3.1

let obs_mags =
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 true_temp) in
  let av = Nx.scalar f64 true_av in
  let norm = Nx.scalar f64 (Float.exp true_log_norm) in
  let mags =
    List.map
      (fun bp ->
        let bp_wave = Photometry.wavelength bp in
        let sed =
          Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
          |> Spectrum.scale norm
          |> Extinction.apply (Extinction.ccm89 ~rv) ~av
          |> Spectrum.as_flux_density
        in
        Photometry.ab_mag bp sed)
      bands
  in
  (* Add 3% photometric noise *)
  let noise = [| 0.03; -0.02; 0.01; -0.01; 0.02 |] in
  List.mapi (fun i m -> Nx.add_s m noise.(i)) mags

let obs_errs = List.init 5 (fun _ -> Nx.scalar f64 0.05)

(* Forward model: generate magnitudes from parameters *)
let forward_model log_temp av log_norm =
  let temp = Unit.Temperature.of_kelvin (exp log_temp) in
  let norm = exp log_norm in
  List.map
    (fun bp ->
      let bp_wave = Photometry.wavelength bp in
      let sed =
        Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
        |> Spectrum.scale norm
        |> Extinction.apply (Extinction.ccm89 ~rv) ~av
        |> Spectrum.as_flux_density
      in
      Photometry.ab_mag bp sed)
    bands

(* Loss function: chi-squared *)
let loss params =
  match params with
  | [ log_temp; av; log_norm ] ->
      let pred = forward_model log_temp av log_norm in
      List.fold_left2
        (fun acc p (o, e) ->
          let residual = div (sub p o) e in
          add acc (square residual))
        (scalar f64 0.0) pred
        (List.combine obs_mags obs_errs)
  | _ -> failwith "expected [log_temp; av; log_norm]"

let () =
  Printf.printf "Differentiable SED Fitting\n";
  Printf.printf "=========================\n";
  Printf.printf
    "Pipeline: Spectrum.blackbody -> Extinction.ccm89 -> Photometry.ab_mag\n\n";

  Printf.printf "True parameters:\n";
  Printf.printf "  T    = %.0f K\n" true_temp;
  Printf.printf "  A_V  = %.2f mag\n" true_av;
  Printf.printf "  logN = %.1f\n\n" true_log_norm;

  Printf.printf "Observed magnitudes (with noise):\n";
  List.iteri
    (fun i m ->
      Printf.printf "  %s = %.3f +/- %.3f\n" band_names.(i) (item [] m)
        (item [] (List.nth obs_errs i)))
    obs_mags;
  Printf.printf "\n";

  (* Initial guesses *)
  let algo = Vega.adam (Vega.Schedule.constant 1e-3) in
  let log_temp = ref (scalar f64 (Float.log 7000.0)) in
  let av = ref (scalar f64 0.3) in
  let log_norm = ref (scalar f64 (-50.5)) in
  let states =
    [| Vega.init algo !log_temp; Vega.init algo !av; Vega.init algo !log_norm |]
  in
  let steps = 1000 in

  Printf.printf "%5s  %10s  %8s  %8s  %8s\n" "step" "chi2" "T (K)" "A_V"
    "log_norm";
  Printf.printf "%5s  %10s  %8s  %8s  %8s\n" "-----" "----------" "--------"
    "--------" "--------";

  let refs = [| log_temp; av; log_norm |] in
  for i = 0 to steps - 1 do
    let loss_val, grads =
      Rune.value_and_grads loss [ !log_temp; !av; !log_norm ]
    in
    if i mod 200 = 0 || i = steps - 1 then
      Printf.printf "%5d  %10.4f  %8.1f  %8.3f  %8.3f\n" i (item [] loss_val)
        (Float.exp (item [] !log_temp))
        (item [] !av) (item [] !log_norm);
    List.iteri
      (fun j g ->
        let p, s = Vega.step states.(j) ~grad:g ~param:!(refs.(j)) in
        refs.(j) := p;
        states.(j) <- s)
      grads
  done;

  Printf.printf "\nFitted parameters:\n";
  Printf.printf "  T    = %.1f K  (true: %.0f K)\n"
    (Float.exp (item [] !log_temp))
    true_temp;
  Printf.printf "  A_V  = %.3f    (true: %.2f)\n" (item [] !av) true_av;
  Printf.printf "  logN = %.3f    (true: %.1f)\n" (item [] !log_norm)
    true_log_norm;

  (* Show fitted vs observed magnitudes *)
  Printf.printf "\nFitted vs observed magnitudes:\n";
  let fitted_mags = forward_model !log_temp !av !log_norm in
  Printf.printf "%5s  %8s  %8s  %8s\n" "Band" "Observed" "Fitted" "Residual";
  Printf.printf "%5s  %8s  %8s  %8s\n" "-----" "--------" "--------" "--------";
  List.iteri
    (fun i (obs, fit) ->
      let o = item [] obs in
      let f = item [] fit in
      Printf.printf "%5s  %8.3f  %8.3f  %+8.3f\n" band_names.(i) o f (f -. o))
    (List.combine obs_mags fitted_mags)
