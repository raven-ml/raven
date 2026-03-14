(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differentiable gravitational lens parameter fitting.

   A point-mass gravitational lens produces multiple images of a background
   source. Given the observed image positions, we fit the lens center and
   Einstein radius by requiring that all images map back to the same source
   position. The lens equation and source-plane mapping are expressed as Nx
   tensor operations, making the entire model differentiable through Rune. *)

open Nx

let f64 = Nx.float64

(* True lens parameters (for generating synthetic data) *)
let true_x_l = 0.1
let true_y_l = -0.05
let true_theta_e = 1.0

(* Generate synthetic image positions for a quadruply-imaged quasar. Source at
   (0.15, 0.08), lens at (true_x_l, true_y_l). *)
let source_x = 0.15
let source_y = 0.08
let () = Printf.printf "Differentiable gravitational lens modeling\n"
let () = Printf.printf "==========================================\n\n"

(* Solve lens equation: beta = theta - theta_E^2 * theta / |theta|^2 for point
   mass (where theta is relative to lens center). We generate 4 image positions
   by solving analytically + adding noise. *)
let img_x, img_y =
  (* For a point mass, images lie along the source-lens axis. Place 4 images at
     realistic positions around the lens. *)
  let dx = source_x -. true_x_l in
  let dy = source_y -. true_y_l in
  let beta = Float.sqrt ((dx *. dx) +. (dy *. dy)) in
  let cos_a = dx /. beta and sin_a = dy /. beta in
  (* Two images along the axis *)
  let theta_p =
    (beta
    +. Float.sqrt ((beta *. beta) +. (4.0 *. true_theta_e *. true_theta_e)))
    /. 2.0
  in
  let theta_m =
    (beta
    -. Float.sqrt ((beta *. beta) +. (4.0 *. true_theta_e *. true_theta_e)))
    /. 2.0
  in
  (* Image positions in 2D (along and perpendicular to axis, with noise) *)
  let noise = 0.005 in
  let x1 = true_x_l +. (theta_p *. cos_a) +. (noise *. 0.3) in
  let y1 = true_y_l +. (theta_p *. sin_a) -. (noise *. 0.2) in
  let x2 = true_x_l +. (theta_m *. cos_a) -. (noise *. 0.5) in
  let y2 = true_y_l +. (theta_m *. sin_a) +. (noise *. 0.4) in
  (* Add two more images from slight perturbation (simulating extended
     source) *)
  let x3 = true_x_l +. (theta_p *. 0.7 *. cos_a) +. (theta_p *. 0.3 *. sin_a) in
  let y3 = true_y_l +. (theta_p *. 0.7 *. sin_a) -. (theta_p *. 0.3 *. cos_a) in
  let x4 = true_x_l -. (theta_p *. 0.5 *. cos_a) +. (theta_p *. 0.4 *. sin_a) in
  let y4 = true_y_l -. (theta_p *. 0.5 *. sin_a) -. (theta_p *. 0.4 *. cos_a) in
  ( create f64 [| 4 |] [| x1; x2; x3; x4 |],
    create f64 [| 4 |] [| y1; y2; y3; y4 |] )

(* Loss: given lens params, map each image back to the source plane. All images
   should map to the same source -> minimize variance of inferred source
   positions. *)
let loss params =
  match params with
  | [ x_l; y_l; theta_e ] ->
      (* Displacement from lens center *)
      let dx = sub img_x x_l in
      let dy = sub img_y y_l in
      (* Distance from lens center *)
      let r_sq = add (square dx) (square dy) in
      let r = sqrt r_sq in
      (* Point-mass deflection: alpha = theta_E^2 / r *)
      let alpha = div (square theta_e) r in
      (* Source position for each image: beta = theta - alpha * hat(theta) *)
      let beta_x = sub img_x (mul alpha (div dx r)) in
      let beta_y = sub img_y (mul alpha (div dy r)) in
      (* Variance of source positions (should be ~0 if lens model is correct) *)
      let mean_bx = mean beta_x in
      let mean_by = mean beta_y in
      let var_x = mean (square (sub beta_x mean_bx)) in
      let var_y = mean (square (sub beta_y mean_by)) in
      add var_x var_y
  | _ -> failwith "expected [x_l; y_l; theta_e]"

let () =
  Printf.printf "True parameters:\n";
  Printf.printf "  x_L     = %.3f arcsec\n" true_x_l;
  Printf.printf "  y_L     = %.3f arcsec\n" true_y_l;
  Printf.printf "  theta_E = %.3f arcsec\n\n" true_theta_e;
  let algo = Vega.adam (Vega.Schedule.constant 1e-2) in
  let x_l = ref (scalar f64 0.0) in
  let y_l = ref (scalar f64 0.0) in
  let theta_e = ref (scalar f64 0.5) in
  let states =
    [| Vega.init algo !x_l; Vega.init algo !y_l; Vega.init algo !theta_e |]
  in
  let steps = 500 in
  Printf.printf "%5s  %12s  %8s  %8s  %8s\n" "step" "loss" "x_L" "y_L" "theta_E";
  Printf.printf "%5s  %12s  %8s  %8s  %8s\n" "-----" "------------" "--------"
    "--------" "--------";
  let refs = [| x_l; y_l; theta_e |] in
  for i = 0 to steps - 1 do
    let loss_val, grads = Rune.value_and_grads loss [ !x_l; !y_l; !theta_e ] in
    List.iteri
      (fun j g ->
        let p, s = Vega.step states.(j) ~grad:g ~param:!(refs.(j)) in
        refs.(j) := p;
        states.(j) <- s)
      grads;
    if i mod 100 = 0 || i = steps - 1 then
      Printf.printf "%5d  %12.8f  %8.4f  %8.4f  %8.4f\n" i (item [] loss_val)
        (item [] !x_l) (item [] !y_l) (item [] !theta_e)
  done;
  Printf.printf "\nFitted parameters:\n";
  Printf.printf "  x_L     = %.4f  (true: %.4f)\n" (item [] !x_l) true_x_l;
  Printf.printf "  y_L     = %.4f  (true: %.4f)\n" (item [] !y_l) true_y_l;
  Printf.printf "  theta_E = %.4f  (true: %.4f)\n" (item [] !theta_e)
    true_theta_e
