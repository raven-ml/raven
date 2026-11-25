(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The recovery law: reference posteriors, as a careful user relies on them. A
   run of NUTS, HMC or ensemble sampling whose diagnostics pass, nested R-hat
   and the summary's findings, recovers every element's mean and standard
   deviation within z sqrt (mcse² + mcse_ref²). SMC and nested sampling run r
   independent replicates and recover within t sqrt (se² + mcse_ref²), se the
   standard error of the replicates' mean and t of Student's t with r - 1
   degrees of freedom, unless a replicate spent its budget. z and t hold the
   family-wise false-alarm rate over every comparison the suites make at 1%: z =
   3.72, and t = 4.88 at 16 replicates, over the suites' 50 comparisons.

   Each run starts from [key]. The suites' runs are sized so that their
   diagnostics pass on every key from 1 to 40, so that a change to the random
   stream or to rounding does not fail a correct sampler. A false finding fails
   a run as a false alarm does: none of keys 1 to 40 gave one, which bounds each
   run's rate of false findings below 7.2% at 95% confidence, a bound beside the
   comparisons' 1% and far looser. *)

open Windtrap
module M = Norn_model
module P = Posteriordb

(* Every run starts from this key. *)
let key = Nx.Rng.key 1

(* [posterior name] is the posteriordb posterior [name], read from norn's test
   directory. *)
let posterior name = P.find "model/support/posteriordb.golden" name

(* The comparisons of the runs norn's suites hold the law on, one family: NUTS
   on dogs (6) and on eight_schools (20), and HMC, ensemble sampling, SMC and
   nested sampling on dogs (6 each). *)
let family = 50
let false_alarms = 0.01

(* The probability of a two-sided error within the bound, per comparison. *)
let tail = 1. -. (false_alarms /. (2. *. float_of_int family))

let normal_quantile p =
  Nx.item []
    (Norn.Dist.quantile
       (Norn.Dist.normal ~loc:(Nx.scalar Nx.float64 0.)
          ~scale:(Nx.scalar Nx.float64 1.))
       (Nx.scalar Nx.float64 p))

(* [t_quantile ~df p] is the quantile of Student's t with [df] degrees of
   freedom, by bisection on its distribution function. With [θ = atan (t / sqrt
   df)], [P(|T| <= t)] is, for odd [df], [2/π (θ + sin θ cos θ Σ a_k cos^2k θ)],
   [a_0 = 1], [a_k = a_(k-1) 2k / (2k + 1)], [k < (df - 1) / 2]; for even [df],
   [sin θ Σ b_k cos^2k θ], [b_0 = 1], [b_k = b_(k-1) (2k - 1) / 2k], [k < df /
   2] (Abramowitz and Stegun 26.7.3 and 26.7.4). *)
let t_quantile ~df p =
  let central t =
    let th = Float.atan (t /. Float.sqrt (float_of_int df)) in
    let c2 = Float.cos th ** 2. in
    let rec sum k n term ratio =
      if k = n then 0.
      else term +. sum (k + 1) n (term *. c2 *. ratio (k + 1)) ratio
    in
    if df mod 2 = 1 then
      2. /. Float.pi
      *. (th
         +. Float.sin th *. Float.cos th
            *. sum 0
                 ((df - 1) / 2)
                 1.
                 (fun k -> float_of_int (2 * k) /. float_of_int ((2 * k) + 1)))
    else
      Float.sin th
      *. sum 0 (df / 2) 1. (fun k ->
          float_of_int ((2 * k) - 1) /. float_of_int (2 * k))
  in
  let target = (2. *. p) -. 1. in
  let rec bracket hi = if central hi < target then bracket (2. *. hi) else hi in
  let rec bisect lo hi n =
    if n = 0 then (lo +. hi) /. 2.
    else
      let mid = (lo +. hi) /. 2. in
      if central mid < target then bisect mid hi (n - 1)
      else bisect lo mid (n - 1)
  in
  bisect 0. (bracket 1.) 200

(* [near ~q what est ~ref ~se ~se_ref] asserts that the estimate [est] lies
   within [q] combined standard errors of its reference [ref]. *)
let near ~q what est ~ref ~se ~se_ref =
  at_most (float 1e-12)
    ~than:(q *. Float.hypot se se_ref)
    ~msg:(Printf.sprintf "%s is %g, the reference %g" what est ref)
    (Float.abs (est -. ref))

(* Chains *)

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

(* [element x index] is the chains of element [index] of the draws [x], of shape
   [[c; n; ...]]. *)
let element x index =
  let s = Nx.shape x in
  let c = s.(0) and n = s.(1) in
  let size = Array.fold_left ( * ) 1 (Array.sub s 2 (Array.length s - 2)) in
  let flat = Nx.to_array (Nx.reshape [| c; n; size |] x) in
  Array.init c (fun i ->
      Array.init n (fun j -> flat.((((i * n) + j) * size) + index)))

let finding = Testable.structural ~pp:Norn.Summary.pp_finding

(* [chains post ~superchains ?stats d] asserts the law on the draws [d] of
   [post]'s latent values: their summary, with nested R-hat over [superchains]
   and the transitions' statistics [stats], has no finding, and every reference
   element's mean and standard deviation is recovered. *)
let chains (P.Posterior p) ~superchains ?stats d =
  let s = Norn.Summary.v P.latent ?stats ~superchains d in
  equal (list finding) [] (Norn.Summary.findings s);
  let q = normal_quantile tail in
  let values = (d : Nx.host Nx.float64_t list Norn.Draws.t :> _ list) in
  List.iter
    (fun (r : P.reference) ->
      let at = Option.get (List.find_index (String.equal r.param) p.params) in
      let mean, sd, mcse_mean, mcse_sd =
        moments (element (List.nth values at) r.index)
      in
      let what = Printf.sprintf "%s › %s[%d]'s" p.name r.param r.index in
      near ~q (what ^ " mean") mean ~ref:r.mean ~se:mcse_mean
        ~se_ref:r.mcse_mean;
      near ~q (what ^ " sd") sd ~ref:r.sd ~se:mcse_sd ~se_ref:r.mcse_sd)
    p.refs

(* Replicates *)

(* A sampler that returns an evidence, run from prior draws. Its particles are
   not chains a convergence diagnostic reads: tempering's chains start from
   resampled, shared ancestors, and nested sampling's dead points from
   survivors. So it runs independent replicates as lanes of one compiled run,
   each replicate's expectations of every element and of its square give its
   mean and standard deviation, and the replicates' spread is their error. *)
type evidence = {
  run :
    'u.
    ('u, Nx.host) Nx.Ptree.t ->
    prior:('u -> Nx.host Nx.float64_t) ->
    likelihood:('u -> Nx.host Nx.float64_t) ->
    Nx.host Nx.Rng.key ->
    'u ->
    ('u, Nx.float64_elt, Nx.host) Norn.Evidence.t;
}

let replicated sampler ~replicates ~draws (P.Posterior p) =
  let m = p.model in
  let u = M.coords m in
  let flat x =
    let v =
      Nx.concatenate ~axis:0
        (List.map (Nx.reshape [| -1 |])
           (M.constrain m x :> Nx.host Nx.float64_t list))
    in
    Nx.concatenate ~axis:0 [ v; Nx.square v ]
  in
  let ez = Norn.Evidence.ptree u in
  let run =
    Rune.jit
      Nx.Ptree.(Nx.Rng.ptree @-> returns (pair ez tensor))
      (fun ks ->
        Rune.vmap
          Nx.Ptree.(Nx.Rng.ptree @-> returns (pair ez tensor))
          (fun k ->
            let z =
              sampler.run u ~prior:(M.log_prior m)
                ~likelihood:(M.log_likelihood m p.y) (Nx.Rng.fold_in k 0)
                (M.from_prior m ~n:draws (Nx.Rng.fold_in k 1))
            in
            (z, fst (Norn.Evidence.expectation u flat z)))
          ks)
  in
  let zs, e = run (Nx.Rng.split_batch ~n:replicates key) in
  for i = 0 to replicates - 1 do
    let z = Nx.Ptree.map ez (fun _ t -> Nx.slice [ Nx.I i ] t) zs in
    match Norn.Evidence.stop z with
    | Converged -> ()
    | Remaining r -> failf "replicate %d spent its budget with %g nats left" i r
    | Temperature b -> failf "replicate %d spent its budget at β = %g" i b
  done;
  let e = Nx.to_array e and k = float_of_int replicates in
  let n = Array.length e / replicates / 2 in
  let at r j = e.((r * 2 * n) + j) in
  (* Each parameter's first element in a replicate's row. *)
  let offsets =
    let one =
      M.constrain m
        (Nx.Ptree.map u
           (fun _ t -> Nx.slice [ Nx.I 0 ] t)
           (M.from_prior m ~n:1 (Nx.Rng.key 0)))
    in
    fst
      (List.fold_left2
         (fun (acc, at) name v -> ((name, at) :: acc, at + Nx.numel v))
         ([], 0) p.params
         (one :> Nx.host Nx.float64_t list))
  in
  let q = t_quantile ~df:(replicates - 1) tail in
  let avg l = List.fold_left ( +. ) 0. l /. k in
  let se l =
    let a = avg l in
    Float.sqrt
      (List.fold_left (fun s v -> s +. ((v -. a) ** 2.)) 0. l /. (k -. 1.) /. k)
  in
  List.iter
    (fun (r : P.reference) ->
      let j = r.index + List.assoc r.param offsets in
      let means = List.init replicates (fun q -> at q j) in
      let sds =
        List.init replicates (fun q ->
            Float.sqrt (Float.max 0. (at q (n + j) -. (at q j ** 2.))))
      in
      let what = Printf.sprintf "%s › %s[%d]'s" p.name r.param r.index in
      near ~q (what ^ " mean") (avg means) ~ref:r.mean ~se:(se means)
        ~se_ref:r.mcse_mean;
      near ~q (what ^ " sd") (avg sds) ~ref:r.sd ~se:(se sds) ~se_ref:r.mcse_sd)
    p.refs

(* [smc ~replicates ~draws post] asserts the law on tempering from [draws] prior
   draws, in [replicates] replicates. *)
let smc ~replicates ~draws post =
  replicated
    {
      run =
        (fun u ~prior ~likelihood k start ->
          Norn.Smc.run u ~budget:200 ~prior ~likelihood k start);
    }
    ~replicates ~draws post

(* [nested ~replicates ~draws post] asserts the law on nested sampling from
   [draws] live points, in [replicates] replicates. *)
let nested ~replicates ~draws post =
  replicated
    {
      run =
        (fun u ~prior ~likelihood k start ->
          Norn.Nested.run u ~budget:200 ~prior ~likelihood k start);
    }
    ~replicates ~draws post
