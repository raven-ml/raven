(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Law 10: reference posteriors. On six posteriordb posteriors, NUTS recovers
   each element's mean and standard deviation within z sqrt (mcse² + mcse_ref²).
   The reference moments and their errors come from gen/posteriordb.py. z holds
   the family-wise false-alarm rate over every comparison at 1%. *)

open Windtrap
module M = Norn_model
module P = Posteriordb

let posteriors = lazy (P.all "../golden/posteriordb.golden")
let chains = 8
let warmup = 300
let draws = 150
let false_alarms = 0.01

let comparisons =
  lazy
    (List.fold_left
       (fun n (P.Posterior p) -> n + (2 * List.length p.refs))
       0 (Lazy.force posteriors))

(* Two-sided, Bonferroni over every comparison. *)
let z =
  lazy
    (let p =
       1. -. (false_alarms /. (2. *. float_of_int (Lazy.force comparisons)))
     in
     Nx.item []
       (Norn.Dist.quantile
          (Norn.Dist.normal ~loc:(Nx.scalar Nx.float64 0.)
             ~scale:(Nx.scalar Nx.float64 1.))
          (Nx.scalar Nx.float64 p)))

let fit (type y) (m : (Nx.float64_t list, y, Nx.float64_elt) M.t) (y : y) =
  let u = M.coords m in
  let lp = M.log_density m y in
  let run =
    Rune.jit
      Nx.Ptree.(Nx.Rng.ptree @-> returns (Norn.Draws.ptree u))
      (fun k ->
        let s = Norn.Nuts.init u lp (M.init m y ~chains k) in
        let s = Norn.Nuts.warmup u lp k ~steps:warmup s in
        let _, d, _ = Norn.Nuts.sample u lp k ~draws s in
        d)
  in
  Norn.Draws.map u P.latent (M.constrain m) (run (Nx.Rng.key 1))

(* [mcse_mean x] is the Monte Carlo standard error of the mean of draws [x] of
   one element, chains by rows. *)
let mcse_mean x =
  let c = Array.length x and n = Array.length x.(0) in
  let d =
    Norn.Draws.v Nx.Ptree.tensor
      (Nx.create Nx.float64 [| c; n |] (Array.concat (Array.to_list x)))
  in
  Nx.item [] (Norn.Diag.mcse_mean Nx.Ptree.tensor d)

(* [moments x] is the mean, the standard deviation and their Monte Carlo
   standard errors of draws [x] of one element, chains by rows. The variance is
   the mean of the squared deviations, so its error is theirs; the standard
   deviation's follows by the delta method. *)
let moments x =
  let all = Array.concat (Array.to_list x) in
  let n = float_of_int (Array.length all) in
  let mean = Array.fold_left ( +. ) 0. all /. n in
  let dev2 = Array.map (Array.map (fun v -> (v -. mean) ** 2.)) x in
  let var =
    Array.fold_left ( +. ) 0. (Array.concat (Array.to_list dev2)) /. n
  in
  let sd = Float.sqrt (var *. n /. (n -. 1.)) in
  (mean, sd, mcse_mean x, mcse_mean dev2 /. (2. *. sd))

let element x index =
  let s = Nx.shape x in
  let c = s.(0) and n = s.(1) in
  let size = Array.fold_left ( * ) 1 (Array.sub s 2 (Array.length s - 2)) in
  let flat = Nx.to_array (Nx.reshape [| c; n; size |] x) in
  Array.init c (fun i ->
      Array.init n (fun j -> flat.((((i * n) + j) * size) + index)))

(* [close ~z what x ~ref ~mcse ~mcse_ref] asserts that [x] is within [z]
   combined standard errors of the reference. *)
let close ~z what x ~ref ~mcse ~mcse_ref =
  let msg = Printf.sprintf "%s is %g, the reference %g" what x ref in
  at_most (float 1e-12)
    ~than:(z *. Float.hypot mcse mcse_ref)
    ~msg
    (Float.abs (x -. ref))

let recovers (P.Posterior p) =
  test p.name (fun () ->
      let d = fit p.model p.y in
      let values = Array.of_list (d :> Nx.float64_t list) in
      let z = Lazy.force z in
      List.iter
        (fun (r : P.reference) ->
          let i =
            Option.get (List.find_index (String.equal r.param) p.params)
          in
          let mean, sd, mcse_mean, mcse_sd =
            moments (element values.(i) r.index)
          in
          let what = Printf.sprintf "%s[%d]'s" r.param r.index in
          close ~z (what ^ " mean") mean ~ref:r.mean ~mcse:mcse_mean
            ~mcse_ref:r.mcse_mean;
          close ~z (what ^ " sd") sd ~ref:r.sd ~mcse:mcse_sd ~mcse_ref:r.mcse_sd)
        p.refs)

let () =
  exit
    (run "Norn posteriordb"
       [ group "NUTS recovers" (List.map recovers (Lazy.force posteriors)) ])
