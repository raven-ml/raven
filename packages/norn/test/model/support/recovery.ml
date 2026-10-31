(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Law 10: reference posteriors, as a careful user relies on them. A run of
   NUTS, HMC or ensemble sampling whose diagnostics pass, nested R-hat and the
   summary's findings, recovers every element's mean and standard deviation
   within z sqrt (mcse² + mcse_ref²); a run that does not recover is flagged by
   them. SMC and nested sampling run r independent replicates and recover within
   t sqrt (se² + mcse_ref²), se the standard error of the replicates' mean and t
   of Student's t with r - 1 degrees of freedom, or are flagged by a replicate
   that spent its budget. z and t hold the family-wise false-alarm rate over
   every comparison of the cells a law is held on at 1%. *)

open Windtrap
module M = Norn_model
module P = Posteriordb

let warmup = 300
let false_alarms = 0.01

(* Every run starts from this key. *)
let key = Nx.Rng.key 1

(* A comparison: an estimate, its reference and their combined standard error. A
   cell's verdict: flagged by its diagnostics, or the comparisons it makes. *)
type comparison = { what : string; est : float; ref : float; se : float }
type verdict = Flagged of string | Compared of comparison list

(* A cell: a sampler's run on one posterior, from a key. [comparisons] counts
   the comparisons it makes when it recovers, and [quantile p] is the multiple
   of a comparison's standard error that a two-sided error lies within with
   probability [2p - 1]. Its run is compiled once, whatever the key. *)
type cell = {
  name : string;
  sampler : string;
  comparisons : int;
  quantile : float -> float;
  verdict : Nx.host Nx.Rng.key -> verdict;
}

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

let stats_ptree = Norn.Draws.ptree (Norn.Stats.ptree Nx.float64)

(* A kernel: its chains, the superchains of its nested R-hat, and a run from a
   start, with the Hamiltonian transitions' statistics. *)
type kernel = {
  label : string;
  chains : int;
  superchains : int;
  run :
    'u.
    ('u, Nx.host) Nx.Ptree.t ->
    ('u -> Nx.host Nx.float64_t) ->
    Nx.host Nx.Rng.key ->
    'u ->
    'u Norn.Draws.t * (Nx.float64_elt, Nx.host) Norn.Stats.t Norn.Draws.t option;
}

let nuts ?accept ~draws () =
  {
    label = "NUTS";
    chains = 8;
    superchains = 4;
    run =
      (fun u lp k start ->
        let s = Norn.Nuts.init u ?accept lp start in
        let s = Norn.Nuts.warmup u lp k ~steps:warmup s in
        let _, d, st = Norn.Nuts.sample u lp k ~draws s in
        (d, Some st));
  }

let hmc ~draws =
  {
    label = "HMC";
    chains = 64;
    superchains = 8;
    run =
      (fun u lp k start ->
        let s = Norn.Hmc.init u lp start in
        let s = Norn.Hmc.warmup u lp k ~steps:warmup s in
        let _, d, st = Norn.Hmc.sample u lp k ~draws s in
        (d, Some st));
  }

(* Ensembles of 32 walkers: at least twice the coordinates of each posterior. A
   walker's moves read the other walkers of its ensemble, so the walkers of one
   ensemble are dependent and the ensembles independent: each ensemble is a
   superchain of nested R-hat. Walkers spread over a region of another shape
   than a narrow, correlated posterior come back to it only over many
   transitions, so they start where HMC leaves them after twice the warmup. *)
let ensemble ~ensembles ~draws =
  {
    label = "ensemble sampling";
    chains = 32 * ensembles;
    superchains = ensembles;
    run =
      (fun u lp k start ->
        let h = Norn.Hmc.init u lp start in
        let h =
          Norn.Hmc.warmup u lp (Nx.Rng.fold_in k 1) ~steps:(2 * warmup) h
        in
        let s = Norn.Ensemble.init u ~ensembles lp h.position in
        let s = Norn.Ensemble.warmup u lp k ~steps:warmup s in
        let _, d, _ = Norn.Ensemble.sample u lp k ~draws s in
        (d, None));
  }

let fit (type p y) kernel (m : (p, y, Nx.float64_elt, Nx.host) M.t)
    (latent : (p, 'd) Nx.Ptree.t) (y : y) =
  let u = M.coords m in
  let lp = M.log_density m y in
  let run =
    Rune.jit
      Nx.Ptree.(
        Nx.Rng.ptree
        @-> returns (pair (Norn.Draws.ptree u) (option stats_ptree)))
      (fun k -> kernel.run u lp k (M.init m y ~chains:kernel.chains k))
  in
  fun k ->
    let d, stats = run k in
    (Norn.Draws.map u latent (M.constrain m) d, stats)

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

let compared what est ~ref ~se ~se_ref =
  { what; est; ref; se = Float.hypot se se_ref }

(* The verdict of chains of draws [d]: flagged by a finding of their summary,
   with nested R-hat over [superchains], else each reference element's mean and
   standard deviation compared. *)
let judge_chains (type l) (latent : (l, 'd) Nx.Ptree.t) ~superchains
    (refs : P.reference list) (value : string -> l -> Nx.host Nx.float64_t) stats
    (d : l Norn.Draws.t) =
  let s = Norn.Summary.v latent ?stats ~superchains d in
  match Norn.Summary.findings s with
  | _ :: _ as fs ->
      let kind : Norn.Summary.finding -> string = function
        | Rhat_high _ -> "R-hat"
        | Ess_low _ -> "ESS"
        | Not_finite _ -> "not finite"
        | Constant _ -> "constant"
        | Divergent _ -> "divergent"
        | Saturated _ -> "saturated"
        | Ebfmi_low _ -> "E-BFMI"
        | Pareto_k _ -> "Pareto k"
      in
      let kinds = List.sort_uniq compare (List.map kind fs) in
      Flagged
        (Printf.sprintf "%s (%s)"
           (Format.asprintf "%a" Norn.Summary.pp_finding (List.hd fs))
           (String.concat ", "
              (List.map
                 (fun k ->
                   Printf.sprintf "%s x%d" k
                     (List.length (List.filter (fun f -> kind f = k) fs)))
                 kinds)))
  | [] ->
      Compared
        (List.concat_map
           (fun (r : P.reference) ->
             let mean, sd, mcse_mean, mcse_sd =
               moments (element (value r.param (d :> l)) r.index)
             in
             let what = Printf.sprintf "%s[%d]'s" r.param r.index in
             [
               compared (what ^ " mean") mean ~ref:r.mean ~se:mcse_mean
                 ~se_ref:r.mcse_mean;
               compared (what ^ " sd") sd ~ref:r.sd ~se:mcse_sd
                 ~se_ref:r.mcse_sd;
             ])
           refs)

(* A posteriordb parameter of draws: the tensor at its position. *)
let param (params : string list) name (values : Nx.host Nx.float64_t list) =
  List.nth values (Option.get (List.find_index (String.equal name) params))

let chain_cell kernel (P.Posterior p) =
  let run = fit kernel p.model P.latent p.y in
  {
    name = kernel.label ^ " › " ^ p.name;
    sampler = kernel.label;
    comparisons = 2 * List.length p.refs;
    quantile = normal_quantile;
    verdict =
      (fun k ->
        let d, stats = run k in
        judge_chains P.latent ~superchains:kernel.superchains p.refs
          (param p.params) stats d);
  }

(* A sampler that returns an evidence, run from prior draws. Its particles are
   not chains a convergence diagnostic reads: tempering's chains start from
   resampled, shared ancestors, and nested sampling's dead points from
   survivors. So it runs independent replicates as lanes of one compiled run,
   each replicate's expectations of every element and of its square give its
   mean and standard deviation, and the replicates' spread is their error. A
   replicate that spent its budget flags the run. *)
type replicated = {
  label : string;
  replicates : int;
  draws : int;
  evidence :
    'u.
    ('u, Nx.host) Nx.Ptree.t ->
    prior:('u -> Nx.host Nx.float64_t) ->
    likelihood:('u -> Nx.host Nx.float64_t) ->
    Nx.host Nx.Rng.key ->
    'u ->
    ('u, Nx.float64_elt, Nx.host) Norn.Evidence.t;
}

(* Tempering from [draws] prior draws, in [replicates] replicates. *)
let smc ~replicates ~draws =
  {
    label = "SMC";
    replicates;
    draws;
    evidence =
      (fun u ~prior ~likelihood k start ->
        Norn.Smc.run u ~budget:200 ~prior ~likelihood k start);
  }

(* Nested sampling from [draws] live points, in [replicates] replicates. *)
let nested ~replicates ~draws =
  {
    label = "nested sampling";
    replicates;
    draws;
    evidence =
      (fun u ~prior ~likelihood k start ->
        Norn.Nested.run u ~budget:200 ~prior ~likelihood k start);
  }

let replicated_cell sampler (P.Posterior p) =
  let m = p.model in
  let u = M.coords m in
  let flat x =
    let v =
      Nx.concatenate ~axis:0
        (List.map (Nx.reshape [| -1 |]) (M.constrain m x :> Nx.host Nx.float64_t list))
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
              sampler.evidence u ~prior:(M.log_prior m)
                ~likelihood:(M.log_likelihood m p.y) (Nx.Rng.fold_in k 0)
                (M.from_prior m ~n:sampler.draws (Nx.Rng.fold_in k 1))
            in
            (z, fst (Norn.Evidence.expectation u flat z)))
          ks)
  in
  let replicates = sampler.replicates in
  {
    name = sampler.label ^ " › " ^ p.name;
    sampler = sampler.label;
    comparisons = 2 * List.length p.refs;
    quantile = t_quantile ~df:(replicates - 1);
    verdict =
      (fun key ->
        let zs, e = run (Nx.Rng.split_batch ~n:replicates key) in
        let spent =
          List.filter_map
            (fun i ->
              let z = Nx.Ptree.map ez (fun _ t -> Nx.slice [ Nx.I i ] t) zs in
              match Norn.Evidence.stop z with
              | Converged -> None
              | Remaining r ->
                  Some
                    (Printf.sprintf
                       "replicate %d spent its budget with %g nats left" i r)
              | Temperature b ->
                  Some
                    (Printf.sprintf "replicate %d spent its budget at β = %g" i
                       b))
            (List.init replicates Fun.id)
        in
        match spent with
        | why :: _ -> Flagged why
        | [] ->
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
                   (fun (acc, at) name v ->
                     ((name, at) :: acc, at + Nx.numel v))
                   ([], 0) p.params
                   (one :> Nx.host Nx.float64_t list))
            in
            Compared
              (List.concat_map
                 (fun (r : P.reference) ->
                   let j = r.index + List.assoc r.param offsets in
                   let means = List.init replicates (fun q -> at q j) in
                   let sds =
                     List.init replicates (fun q ->
                         Float.sqrt
                           (Float.max 0. (at q (n + j) -. (at q j ** 2.))))
                   in
                   let avg l = List.fold_left ( +. ) 0. l /. k in
                   let se l =
                     let a = avg l in
                     Float.sqrt
                       (List.fold_left (fun s v -> s +. ((v -. a) ** 2.)) 0. l
                       /. (k -. 1.) /. k)
                   in
                   let what = Printf.sprintf "%s[%d]'s" r.param r.index in
                   [
                     compared (what ^ " mean") (avg means) ~ref:r.mean
                       ~se:(se means) ~se_ref:r.mcse_mean;
                     compared (what ^ " sd") (avg sds) ~ref:r.sd ~se:(se sds)
                       ~se_ref:r.mcse_sd;
                   ])
                 p.refs));
  }

(* [tail cells] is the two-sided tail whose quantile holds the false-alarm rate
   over all the comparisons of [cells], Bonferroni. *)
let tail cells =
  let comparisons = List.fold_left (fun n c -> n + c.comparisons) 0 cells in
  1. -. (false_alarms /. (2. *. float_of_int comparisons))

let within ~tail c cs =
  let q = c.quantile tail in
  List.iter
    (fun k ->
      at_most (float 1e-12) ~than:(q *. k.se)
        ~msg:
          (Printf.sprintf "%s: %s is %g, the reference %g" c.name k.what k.est
             k.ref)
        (Float.abs (k.est -. k.ref)))
    cs

(* [recovers cells] is the law on [cells], one test each: each must recover
   within its quantile of [tail cells]. *)
let recovers cells =
  let tail = tail cells in
  List.map
    (fun c ->
      test (c.name ^ " recovers the reference") (fun () ->
          match c.verdict key with
          | Flagged why -> failf "%s is flagged: %s" c.name why
          | Compared cs -> within ~tail c cs))
    cells

(* [flagged c] is the law on the one cell [c], which its diagnostics must
   flag. *)
let flagged c =
  test (c.name ^ " is flagged") (fun () ->
      match c.verdict key with
      | Flagged _ -> ()
      | Compared _ -> failf "%s passes its diagnostics" c.name)

(* [law cells] runs every cell once. A cell recovers within its quantile of
   [tail cells], or is flagged. Each sampler of the cells must recover on one of
   them, and one cell must be flagged. *)
let law cells =
  let tail = tail cells in
  let samplers =
    List.sort_uniq String.compare (List.map (fun c -> c.sampler) cells)
  in
  let pp ppf c = Format.pp_print_string ppf c.name in
  prop ~count:0 ~examples:cells
    "a run recovers the reference or its diagnostics flag it"
    (Gen.of_list ~pp cells) (fun c ->
      let verdict = c.verdict key in
      let recovered =
        match verdict with Compared _ -> true | Flagged _ -> false
      in
      List.iter
        (fun s -> cover (s ^ " recovers") (recovered && s = c.sampler))
        samplers;
      cover "flagged" (not recovered);
      match verdict with
      | Flagged why -> classify (c.name ^ " flagged: " ^ why) true
      | Compared cs -> within ~tail c cs)
