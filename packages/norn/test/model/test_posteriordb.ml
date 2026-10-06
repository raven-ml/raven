(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Law 10: reference posteriors. On six posteriordb posteriors, NUTS, HMC and
   ensemble slice sampling each recover every element's mean and standard
   deviation within z sqrt (mcse² + mcse_ref²). The reference moments and their
   errors come from gen/posteriordb.py. HMC also recovers Neal's funnel,
   non-centred, at 1024 chains. z holds the family-wise false-alarm rate over
   every comparison at 1%. *)

open Windtrap
module M = Norn_model
module P = Posteriordb

let posteriors = lazy (P.all "../golden/posteriordb.golden")
let warmup = 300
let false_alarms = 0.01

(* Three kernels on every reference, and the funnel's four moments. *)
let kernels = 3
let funnel_comparisons = 4

let comparisons =
  lazy
    (List.fold_left
       (fun n (P.Posterior p) -> n + (2 * kernels * List.length p.refs))
       funnel_comparisons (Lazy.force posteriors))

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

(* A kernel: its chains and a run from a start. *)
type kernel = {
  chains : int;
  run :
    'u.
    'u Nx.Ptree.t -> ('u -> Nx.float64_t) -> Nx.Rng.t -> 'u -> 'u Norn.Draws.t;
}

let nuts =
  {
    chains = 8;
    run =
      (fun u lp k start ->
        let s = Norn.Nuts.init u lp start in
        let s = Norn.Nuts.warmup u lp k ~steps:warmup s in
        let _, d, _ = Norn.Nuts.sample u lp k ~draws:150 s in
        d);
  }

let hmc =
  {
    chains = 64;
    run =
      (fun u lp k start ->
        let s = Norn.Hmc.init u lp start in
        let s = Norn.Hmc.warmup u lp k ~steps:warmup s in
        let _, d, _ = Norn.Hmc.sample u lp k ~draws:50 s in
        d);
  }

(* Two ensembles of 64 walkers: at least twice the coordinates of each
   posterior. A walker far from a narrow, correlated posterior moves along
   directions of the posterior's shape, which lead back to it over hundreds of
   transitions, so the walkers start where HMC leaves them after twice the
   warmup, which on sblrc-blr leaves no chain behind. *)
let ensemble =
  {
    chains = 128;
    run =
      (fun u lp k start ->
        let h = Norn.Hmc.init u lp start in
        let h =
          Norn.Hmc.warmup u lp (Nx.Rng.fold_in k 1) ~steps:(2 * warmup) h
        in
        let s = Norn.Ensemble.init u ~ensembles:2 lp h.position in
        let s = Norn.Ensemble.warmup u lp k ~steps:warmup s in
        let _, d, _ = Norn.Ensemble.sample u lp k ~draws:200 s in
        d);
  }

let fit (type p y) kernel (m : (p, y, Nx.float64_elt) M.t)
    (latent : p Nx.Ptree.t) (y : y) =
  let u = M.coords m in
  let lp = M.log_density m y in
  let run =
    Rune.jit
      Nx.Ptree.(Nx.Rng.ptree @-> returns (Norn.Draws.ptree u))
      (fun k -> kernel.run u lp k (M.init m y ~chains:kernel.chains k))
  in
  Norn.Draws.map u latent (M.constrain m) (run (Nx.Rng.key 1))

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

let recovers kernel (P.Posterior p) =
  test p.name (fun () ->
      let d = fit kernel p.model P.latent p.y in
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

(* Neal's funnel: v ~ N(0, 3), x ~ N(0, exp (v / 2)) in nine dimensions,
   non-centred. Its exact moments: v's mean 0 and sd 3; each x's mean 0 and sd
   exp (9 / 4), since E exp v = exp 4.5. *)

type 'a funnel = { v : 'a; x : 'a }

module Funnel = struct
  type 'a t = 'a funnel

  let walk c { v; x } =
    let open Nx.Ptree.Walk in
    let v = field c "v" leaf v in
    let x = field c "x" leaf x in
    { v; x }
end

let funnel_latent : Nx.float64_t funnel Nx.Ptree.t =
  Nx.Ptree.instantiate (module Funnel)

let funnel =
  M.noncentre
    (fun p -> p.x)
    ( M.v Nx.float64 funnel_latent Nx.Ptree.unit @@ fun () ->
      let f64 = Nx.scalar Nx.float64 in
      let v = M.sample (Norn.Dist.normal ~loc:(f64 0.) ~scale:(f64 3.)) in
      let scale = Nx.exp (Nx.mul_s v 0.5) in
      let x =
        M.sample (Norn.Dist.iid [| 9 |] (Norn.Dist.normal ~loc:(f64 0.) ~scale))
      in
      ({ v; x }, ()) )

let funnel_test =
  slow "HMC recovers a funnel at 1024 chains" (fun () ->
      let kernel = { hmc with chains = 1024 } in
      let d = fit kernel funnel funnel_latent () in
      let { v; x } = (d :> Nx.float64_t funnel) in
      let z = Lazy.force z in
      let check what x ~mean ~sd =
        let m, s, mcse_m, mcse_s = moments (element x 0) in
        close ~z (what ^ "'s mean") m ~ref:mean ~mcse:mcse_m ~mcse_ref:0.;
        close ~z (what ^ "'s sd") s ~ref:sd ~mcse:mcse_s ~mcse_ref:0.
      in
      check "v" v ~mean:0. ~sd:3.;
      check "x[0]" x ~mean:0. ~sd:(Float.exp 2.25))

let () =
  let all = Lazy.force posteriors in
  exit
    (run "Norn posteriordb"
       [
         group "NUTS recovers" (List.map (recovers nuts) all);
         group "HMC recovers" (List.map (recovers hmc) all @ [ funnel_test ]);
         group "ensemble slice sampling recovers"
           (List.map (recovers ensemble) all);
       ])
