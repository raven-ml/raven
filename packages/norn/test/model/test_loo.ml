(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module M = Norn_model
module D = Norn.Dist
module Diag = Norn.Diag
module Draws = Norn.Draws

let t = Nx.Ptree.tensor
let f64 = Nx.scalar Nx.float64
let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* A normal mean of known unit variance under a normal prior of scale [tau]:
   every posterior, with a point or without, is normal, and so is each point's
   leave-one-out predictive. *)

let tau = 3.
let points = 20

let model =
  M.v Nx.float64 t t @@ fun () ->
  let mu = M.sample (D.normal ~loc:(f64 0.) ~scale:(f64 tau)) in
  let y = M.sample (D.iid [| points |] (D.normal ~loc:mu ~scale:(f64 1.))) in
  (mu, y)

(* Seeded data about 1, the last point far out. *)
let y =
  let z = floats (Nx.Rng.normal ~key:(Nx.Rng.key 11) Nx.float64 [| points |]) in
  z.(points - 1) <- 3.5;
  Array.map (fun z -> 1. +. z) z

(* [posterior ys] is the mean and precision of mu given [ys]. *)
let posterior ys =
  let precision = (1. /. (tau *. tau)) +. float_of_int (Array.length ys) in
  (Array.fold_left ( +. ) 0. ys /. precision, precision)

let log_normal x ~loc ~var =
  let d = x -. loc in
  -0.5 *. (Float.log (2. *. Float.pi *. var) +. (d *. d /. var))

(* [exact i] is log p(y_i | y_-i): normal about the posterior mean without
   [y_i], its variance the posterior's plus the point's. *)
let exact i =
  let rest = List.filteri (fun j _ -> j <> i) (Array.to_list y) in
  let rest = Array.of_list rest in
  let m, p = posterior rest in
  log_normal y.(i) ~loc:m ~var:(1. +. (1. /. p))

(* 4 chains of 1000 independent posterior draws, so that each point's
   self-normalised estimate has the Monte Carlo error of independent draws:
   sqrt (Σ w̃_s² (p_s / p̂ - 1)²) in log p̂ (Vehtari, Simpson, Gelman, Yao and
   Gabry 2024, equation 5, through the delta method). Five of those bound a
   false failure at 6e-7 a point under normality; with k-hat below 0.7 the
   smoothing's bias is negligible beside them. *)
let draws =
  let m, p = posterior y in
  let z = Nx.Rng.normal ~key:(Nx.Rng.key 12) Nx.float64 [| 4; 1000 |] in
  Draws.v t (Nx.add_s (Nx.div_s z (Float.sqrt p)) m)

let log_lik =
  let data = Nx.create Nx.float64 [| points |] y in
  Draws.map t t (fun mu -> M.pointwise model data mu) draws

let mcse i =
  let ll = Nx.slice [ Nx.A; Nx.A; Nx.I i ] (log_lik :> Nx.host Nx.float64_t) in
  let ll = floats (Nx.reshape [| -1 |] ll) in
  let s = Array.length ll in
  let lr = Nx.create Nx.float64 [| s |] (Array.map Float.neg ll) in
  let lw = floats (Diag.psis lr).log_weights in
  let sum xs = Array.fold_left ( +. ) 0. xs in
  let z = Array.fold_left Float.max Float.neg_infinity lw in
  let w = Array.map (fun l -> Float.exp (l -. z)) lw in
  let total = sum w in
  let w = Array.map (fun w -> w /. total) w in
  let e = sum (Array.map2 (fun w l -> w *. Float.exp l) w ll) in
  Float.sqrt
    (sum (Array.map2 (fun w l -> (w *. ((Float.exp l /. e) -. 1.)) ** 2.) w ll))

let exact_loo =
  test "PSIS-LOO of a conjugate normal model is its exact leave-one-out"
    (fun () ->
      let r = Diag.loo log_lik in
      equal (array int) [| points |] (Nx.shape r.pointwise);
      let pointwise = floats r.pointwise and khat = floats r.khat in
      let bound = ref 0. in
      for i = 0 to points - 1 do
        less float_exact ~than:0.7 khat.(i);
        let e = mcse i in
        bound := !bound +. e;
        equal ~msg:(Printf.sprintf "point %d" i) (float (5. *. e)) (exact i)
          pointwise.(i)
      done;
      let total = List.fold_left ( +. ) 0. (List.init points exact) in
      equal (float (5. *. !bound)) total (Nx.item [] r.elpd))

let pointwise_shape =
  test "the pointwise likelihood over draws has a point axis after the draws"
    (fun () ->
      equal (array int) [| 4; 1000; points |]
        (Nx.shape (log_lik :> Nx.host Nx.float64_t)))

let () = exit (run "Norn.Diag.loo" [ pointwise_shape; exact_loo ])
