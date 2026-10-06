(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Law 7: simulation agrees with scoring. Simulation-based calibration of a
   conjugate normal model, whose posterior is known: the ranks of simulated
   truths among exact posterior draws, and among NUTS draws, are uniform. Each
   of the two uniformity tests rejects below a p-value of 0.005, holding their
   false alarms at 1%. *)

open Windtrap
module M = Norn_model
module D = Norn.Dist

let f64 = Nx.scalar Nx.float64
let points = 5
let t = Nx.Ptree.tensor

(* mu ~ N(0, 1), y_i ~ N(mu, 1): the posterior of mu is N(Σ y / (n + 1), 1 / (n
   + 1)). *)
let model =
  M.v Nx.float64 t t (fun () ->
      let mu = M.sample (D.normal ~loc:(f64 0.) ~scale:(f64 1.)) in
      (mu, M.sample (D.iid [| points |] (D.normal ~loc:mu ~scale:(f64 1.)))))

let n = float_of_int points

let exact_posterior k y ~draws =
  let mean = Nx.div_s (Nx.sum y) (n +. 1.) in
  let sd = 1. /. Float.sqrt (n +. 1.) in
  let z = Nx.Rng.normal k Nx.float64 [| 1; draws |] in
  Norn.Draws.v t (Nx.add mean (Nx.mul_s z sd))

let replications = 200
let draws = 99
let alpha = 0.005

let exact =
  test "ranks among exact posterior draws are uniform" (fun () ->
      let rank r =
        let k = Nx.Rng.fold_in (Nx.Rng.key 50) r in
        let truth, y = M.simulate model (Nx.Rng.fold_in k 0) in
        Norn.Diag.rank t ~truth (exact_posterior (Nx.Rng.fold_in k 1) y ~draws)
      in
      let u =
        Norn.Diag.rank_uniformity t ~draws (List.init replications rank)
      in
      at_least (float 1e-12) ~than:alpha (Nx.item [] u))

(* A posterior whose variance is twice the truth's is caught. *)
let too_wide =
  test "ranks among overdispersed draws are not uniform" (fun () ->
      let rank r =
        let k = Nx.Rng.fold_in (Nx.Rng.key 51) r in
        let truth, y = M.simulate model (Nx.Rng.fold_in k 0) in
        let d = exact_posterior (Nx.Rng.fold_in k 1) y ~draws in
        let mean = Nx.div_s (Nx.sum y) (n +. 1.) in
        let wide =
          Nx.add mean (Nx.mul_s (Nx.sub (d :> Nx.float64_t) mean) 2.)
        in
        Norn.Diag.rank t ~truth (Norn.Draws.v t wide)
      in
      let u =
        Norn.Diag.rank_uniformity t ~draws (List.init replications rank)
      in
      less (float 1e-12) ~than:alpha (Nx.item [] u))

let nuts =
  slow "ranks among NUTS draws are uniform" (fun () ->
      let u = M.coords model in
      let fit =
        Rune.jit
          Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> returns (Norn.Draws.ptree u))
          (fun k y ->
            let lp = M.log_density model y in
            let start = M.init model y ~chains:1 (Nx.Rng.fold_in k 0) in
            let s = Norn.Nuts.init u lp start in
            let s = Norn.Nuts.warmup u lp (Nx.Rng.fold_in k 1) ~steps:200 s in
            let _, d, _ =
              Norn.Nuts.sample u lp (Nx.Rng.fold_in k 2) ~draws:(draws * 3) s
            in
            Norn.Draws.thin u ~every:3 d)
      in
      let rank r =
        let k = Nx.Rng.fold_in (Nx.Rng.key 52) r in
        let truth, y = M.simulate model (Nx.Rng.fold_in k 0) in
        let d = fit (Nx.Rng.fold_in k 1) y in
        Norn.Diag.rank t
          ~truth:(M.unconstrain model truth :> Nx.float64_t)
          (Norn.Draws.v t (d :> Nx.float64_t))
      in
      let u = Norn.Diag.rank_uniformity t ~draws (List.init 100 rank) in
      at_least (float 1e-12) ~than:alpha (Nx.item [] u))

let () = exit (run "Norn calibration" [ group "SBC" [ exact; too_wide; nuts ] ])
