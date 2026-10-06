(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Models whose evidence is known in closed form, for the evidence laws: a prior
   and a likelihood over positions of rows, prior draws, and ln Z. *)

let f64 = Nx.scalar Nx.float64

type model = {
  name : string;
  prior : Nx.float64_t -> Nx.float64_t;
  likelihood : Nx.float64_t -> Nx.float64_t;
  draw : Nx.Rng.t -> int -> Nx.float64_t; (* [n] prior draws *)
  log_z : float;
}

let rows x = (Nx.shape x).(0)

(* A uniform prior on the cube [-5, 5]^d. *)
let cube d =
  let log_volume = float_of_int d *. Float.log 10. in
  let prior x =
    let inside = Nx.all ~axes:[ 1 ] (Nx.less_equal (Nx.abs x) (f64 5.)) in
    Nx.where inside
      (Nx.full Nx.float64 [| rows x |] (-.log_volume))
      (Nx.full Nx.float64 [| rows x |] Float.neg_infinity)
  in
  let draw k n =
    Nx.mul_s (Nx.sub_s (Nx.Rng.uniform k Nx.float64 [| n; d |]) 0.5) 10.
  in
  (prior, draw, log_volume)

(* [normal_at x mu sigma] is the log density of N(mu, sigma² I) at each row. *)
let normal_at x mu sigma =
  let d = float_of_int (Nx.shape x).(1) in
  let r = Nx.sum ~axes:[ 1 ] (Nx.square (Nx.sub x mu)) in
  Nx.sub_s
    (Nx.mul_s r (-0.5 /. (sigma *. sigma)))
    (d *. (Float.log sigma +. (0.5 *. Float.log (2. *. Float.pi))))

(* A Gaussian likelihood of scale 0.5 under the cube in ten dimensions; the cube
   holds all but 1e-20 of its mass, so [Z] is the cube's inverse volume. *)
let gaussian =
  let prior, draw, log_volume = cube 10 in
  {
    name = "a Gaussian under a uniform prior in ten dimensions";
    prior;
    likelihood = (fun x -> normal_at x (Nx.zeros Nx.float64 [| 10 |]) 0.5);
    draw;
    log_z = -.log_volume;
  }

(* An equal mixture of two Gaussians of scale 0.3, at -2 and 2 on the first
   axis, under the square. *)
let mixture =
  let prior, draw, log_volume = cube 2 in
  let at m = Nx.create Nx.float64 [| 2 |] [| m; 0. |] in
  {
    name = "a mixture of two Gaussians";
    prior;
    likelihood =
      (fun x ->
        Nx.sub_s
          (Nx.logsumexp ~axes:[ 0 ]
             (Nx.stack ~axis:0
                [ normal_at x (at (-2.)) 0.3; normal_at x (at 2.) 0.3 ]))
          (Float.log 2.));
    draw;
    log_z = -.log_volume;
  }

(* x ~ N(0, 1), y_i ~ N(x, 1) for the five points y: marginally y ~ N(0, I + 1
   1ᵀ), whose log density is the evidence. *)
let conjugate =
  let y = [| 0.8; 1.9; 1.1; 0.4; 1.5 |] in
  let m = float_of_int (Array.length y) in
  let sum = Array.fold_left ( +. ) 0. y in
  let sq = Array.fold_left (fun s y -> s +. (y *. y)) 0. y in
  (* (I + 1 1ᵀ)⁻¹ = I - 1 1ᵀ / (m + 1), det = m + 1. *)
  let log_z =
    (-0.5 *. (sq -. (sum *. sum /. (m +. 1.))))
    -. (0.5 *. Float.log (m +. 1.))
    -. (0.5 *. m *. Float.log (2. *. Float.pi))
  in
  let yt = Nx.create Nx.float64 [| 1; Array.length y |] y in
  {
    name = "a conjugate normal";
    prior = (fun x -> normal_at x (Nx.zeros Nx.float64 [| 1 |]) 1.);
    likelihood =
      (fun x ->
        let r = Nx.sum ~axes:[ 1 ] (Nx.square (Nx.sub yt x)) in
        Nx.sub_s (Nx.mul_s r (-0.5)) (0.5 *. m *. Float.log (2. *. Float.pi)));
    draw = (fun k n -> Nx.Rng.normal k Nx.float64 [| n; 1 |]);
    log_z;
  }
