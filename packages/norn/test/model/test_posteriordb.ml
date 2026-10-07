(* Law 10: reference posteriors, as a careful user relies on them. On six
   posteriordb posteriors, a run of NUTS, HMC, ensemble sampling or SMC whose
   diagnostics pass, nested R-hat and the summary's findings, recovers every
   element's mean and standard deviation within z sqrt (mcse² + mcse_ref²); a
   run that does not recover is flagged by them. Nested sampling runs four
   independent replicates and recovers within t sqrt (se² + mcse_ref²), se the
   replicates' spread over 2 and t of Student's t with 3 degrees of freedom, or
   is flagged by a replicate that spent its budget. HMC also recovers Neal's
   funnel, non-centred, at 1024 chains. z and t hold the family-wise false-alarm
   rate over every comparison at 1%. The reference moments and their errors come
   from gen/posteriordb.py. Both verdicts are covered. *)

open Windtrap
module M = Norn_model
module P = Posteriordb

let posteriors = lazy (P.all "../golden/posteriordb.golden")
let warmup = 300
let false_alarms = 0.01

(* Four samplers judged by their chains, nested sampling, and the funnel's four
   moments. *)
let kernels = 5
let funnel_comparisons = 4
let replicates = 4

let comparisons =
  lazy
    (List.fold_left
       (fun n (P.Posterior p) -> n + (2 * kernels * List.length p.refs))
       funnel_comparisons (Lazy.force posteriors))

(* The two-sided quantile of each comparison, Bonferroni over every one. *)
let tail =
  lazy (1. -. (false_alarms /. (2. *. float_of_int (Lazy.force comparisons))))

let z =
  lazy
    (Nx.item []
       (Norn.Dist.quantile
          (Norn.Dist.normal ~loc:(Nx.scalar Nx.float64 0.)
             ~scale:(Nx.scalar Nx.float64 1.))
          (Nx.scalar Nx.float64 (Lazy.force tail))))

(* Student's t with 3 degrees of freedom has the distribution function [1/2 +
   (atan u + u / (1 + u²)) / π], [u = t / sqrt 3]; its quantile is found by
   bisection. *)
let t3 =
  lazy
    (let p = Lazy.force tail in
     let cdf t =
       let u = t /. Float.sqrt 3. in
       0.5 +. ((Float.atan u +. (u /. (1. +. (u *. u)))) /. Float.pi)
     in
     let rec bisect lo hi k =
       if k = 0 then (lo +. hi) /. 2.
       else
         let mid = (lo +. hi) /. 2. in
         if cdf mid < p then bisect mid hi (k - 1) else bisect lo mid (k - 1)
     in
     bisect 0. 1e4 200)

(* A cell's verdict: flagged by its diagnostics, or the comparisons it makes,
   each an estimate, its reference and the distance it may lie from it. *)
type comparison = { what : string; est : float; ref : float; allowed : float }
type verdict = Flagged of string | Compared of comparison list
type cell = { name : string; verdict : unit -> verdict }

let stats_ptree = Norn.Draws.ptree (Norn.Stats.ptree Nx.float64)

(* A kernel: its chains, the superchains of its nested R-hat, and a run from a
   start, with the Hamiltonian transitions' statistics. *)
type kernel = {
  chains : int;
  superchains : int;
  run :
    'u.
    'u Nx.Ptree.t ->
    ('u -> Nx.float64_t) ->
    Nx.Rng.t ->
    'u ->
    'u Norn.Draws.t * Nx.float64_elt Norn.Stats.t Norn.Draws.t option;
}

let nuts =
  {
    chains = 8;
    superchains = 4;
    run =
      (fun u lp k start ->
        let s = Norn.Nuts.init u lp start in
        let s = Norn.Nuts.warmup u lp k ~steps:warmup s in
        let _, d, st = Norn.Nuts.sample u lp k ~draws:150 s in
        (d, Some st));
  }

let hmc =
  {
    chains = 64;
    superchains = 8;
    run =
      (fun u lp k start ->
        let s = Norn.Hmc.init u lp start in
        let s = Norn.Hmc.warmup u lp k ~steps:warmup s in
        let _, d, st = Norn.Hmc.sample u lp k ~draws:50 s in
        (d, Some st));
  }

(* Two ensembles of 64 walkers: at least twice the coordinates of each
   posterior. Walkers spread over a region of another shape than a narrow,
   correlated posterior come back to it only over many transitions, so they
   start where HMC leaves them after twice the warmup, which on sblrc-blr leaves
   no chain behind. *)
let ensemble =
  {
    chains = 128;
    superchains = 8;
    run =
      (fun u lp k start ->
        let h = Norn.Hmc.init u lp start in
        let h =
          Norn.Hmc.warmup u lp (Nx.Rng.fold_in k 1) ~steps:(2 * warmup) h
        in
        let s = Norn.Ensemble.init u ~ensembles:2 lp h.position in
        let s = Norn.Ensemble.warmup u lp k ~steps:warmup s in
        let _, d, _ = Norn.Ensemble.sample u lp k ~draws:200 s in
        (d, None));
  }

let fit (type p y) kernel (m : (p, y, Nx.float64_elt) M.t)
    (latent : p Nx.Ptree.t) (y : y) =
  let u = M.coords m in
  let lp = M.log_density m y in
  let run =
    Rune.jit
      Nx.Ptree.(
        Nx.Rng.ptree
        @-> returns (pair (Norn.Draws.ptree u) (option stats_ptree)))
      (fun k -> kernel.run u lp k (M.init m y ~chains:kernel.chains k))
  in
  let d, stats = run (Nx.Rng.key 1) in
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

let compared ~z what est ~ref ~se ~se_ref =
  { what; est; ref; allowed = z *. Float.hypot se se_ref }

(* The verdict of chains of draws [d]: flagged by a finding of their summary,
   with nested R-hat over [superchains], else each reference element's mean and
   standard deviation compared. *)
let judge_chains (type l) (latent : l Nx.Ptree.t) ~superchains
    (refs : (string * int * float * float * float * float) list)
    (value : string -> l -> Nx.float64_t) stats (d : l Norn.Draws.t) =
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
      let z = Lazy.force z in
      Compared
        (List.concat_map
           (fun (param, index, mean_ref, sd_ref, mcse_mean_ref, mcse_sd_ref) ->
             let mean, sd, mcse_mean, mcse_sd =
               moments (element (value param (d :> l)) index)
             in
             let what = Printf.sprintf "%s[%d]'s" param index in
             [
               compared ~z (what ^ " mean") mean ~ref:mean_ref ~se:mcse_mean
                 ~se_ref:mcse_mean_ref;
               compared ~z (what ^ " sd") sd ~ref:sd_ref ~se:mcse_sd
                 ~se_ref:mcse_sd_ref;
             ])
           refs)

let refs refs =
  List.map
    (fun (r : P.reference) ->
      (r.param, r.index, r.mean, r.sd, r.mcse_mean, r.mcse_sd))
    refs

(* A posteriordb parameter of draws: the tensor at its position. *)
let param (params : string list) name (values : Nx.float64_t list) =
  List.nth values (Option.get (List.find_index (String.equal name) params))

let chain_cell label kernel (P.Posterior p) =
  {
    name = label ^ " › " ^ p.name;
    verdict =
      (fun () ->
        let d, stats = fit kernel p.model P.latent p.y in
        judge_chains P.latent ~superchains:kernel.superchains (refs p.refs)
          (param p.params) stats d);
  }

(* Tempering from 1000 prior draws, in 25 chains of 40 states: the particles are
   read as those chains' draws, so their errors count the chains'
   autocorrelation. *)
let smc_cell (P.Posterior p) =
  let particles = 1000 and chains = 25 in
  {
    name = "SMC › " ^ p.name;
    verdict =
      (fun () ->
        let m = p.model in
        let u = M.coords m in
        let run =
          Rune.jit
            Nx.Ptree.(Nx.Rng.ptree @-> returns (Norn.Evidence.ptree u))
            (fun k ->
              Norn.Smc.run u ~budget:200 ~prior:(M.log_prior m)
                ~likelihood:(M.log_likelihood m p.y) (Nx.Rng.fold_in k 0)
                (M.from_prior m ~n:particles (Nx.Rng.fold_in k 1)))
        in
        let ev = run (Nx.Rng.key 1) in
        match Norn.Evidence.stop ev with
        | Temperature b ->
            Flagged (Printf.sprintf "the budget was spent at β = %g" b)
        | Remaining _ | Converged ->
            let x = (Norn.Evidence.sample ev).values in
            (* Particle [i] is state [i / M] of chain [i mod M]. *)
            let chained =
              Nx.Ptree.map u
                (fun _ t ->
                  let s = Nx.shape t in
                  Nx.moveaxis 0 1
                    (Nx.reshape
                       (Array.append
                          [| particles / chains; chains |]
                          (Array.sub s 1 (Array.length s - 1)))
                       t))
                x
            in
            let d =
              Norn.Draws.map u P.latent (M.constrain m) (Norn.Draws.v u chained)
            in
            judge_chains P.latent ~superchains:5 (refs p.refs) (param p.params)
              None d);
  }

(* Nested sampling from 500 prior draws, four replicates as lanes of one
   compiled run. Each replicate's expectations of every element and of its
   square give its mean and standard deviation; the replicates' spread is their
   error. *)
let nested_cell (P.Posterior p) =
  let live = 500 in
  {
    name = "nested sampling › " ^ p.name;
    verdict =
      (fun () ->
        let m = p.model in
        let u = M.coords m in
        let flat x =
          let v =
            Nx.concatenate ~axis:0
              (List.map (Nx.reshape [| -1 |])
                 (M.constrain m x :> Nx.float64_t list))
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
                    Norn.Nested.run u ~budget:200 ~prior:(M.log_prior m)
                      ~likelihood:(M.log_likelihood m p.y) (Nx.Rng.fold_in k 0)
                      (M.from_prior m ~n:live (Nx.Rng.fold_in k 1))
                  in
                  (z, fst (Norn.Evidence.expectation u flat z)))
                ks)
        in
        let zs, e = run (Nx.Rng.split_batch ~n:replicates (Nx.Rng.key 1)) in
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
              | Temperature _ -> None)
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
                   (one :> Nx.float64_t list))
            in
            let t = Lazy.force t3 in
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
                     compared ~z:t (what ^ " mean") (avg means) ~ref:r.mean
                       ~se:(se means) ~se_ref:r.mcse_mean;
                     compared ~z:t (what ^ " sd") (avg sds) ~ref:r.sd
                       ~se:(se sds) ~se_ref:r.mcse_sd;
                   ])
                 p.refs));
  }

let cells =
  lazy
    (let all = Lazy.force posteriors in
     List.map (chain_cell "NUTS" nuts) all
     @ List.map (chain_cell "HMC" hmc) all
     @ List.map (chain_cell "ensemble sampling" ensemble) all
     @ List.map smc_cell all @ List.map nested_cell all)

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

let funnel_cell =
  {
    name = "HMC › a funnel at 1024 chains";
    verdict =
      (fun () ->
        let d, stats =
          fit
            { hmc with chains = 1024; superchains = 8 }
            funnel funnel_latent ()
        in
        let value name (f : Nx.float64_t funnel) =
          if name = "v" then f.v else f.x
        in
        judge_chains funnel_latent ~superchains:8
          [ ("v", 0, 0., 3., 0., 0.); ("x", 0, 0., Float.exp 2.25, 0., 0.) ]
          value stats d);
  }

let law_10 =
  let cells = Lazy.force cells @ [ funnel_cell ] in
  let pp ppf c = Format.pp_print_string ppf c.name in
  prop ~count:0 ~examples:cells
    "a run recovers the reference or its diagnostics flag it"
    (Gen.of_list ~pp cells) (fun c ->
      match c.verdict () with
      | Flagged why ->
          cover "flagged" true;
          cover "recovered" false;
          classify (c.name ^ " flagged: " ^ why) true
      | Compared cs ->
          cover "flagged" false;
          cover "recovered" true;
          List.iter
            (fun k ->
              at_most (float 1e-12) ~than:k.allowed
                ~msg:
                  (Printf.sprintf "%s: %s is %g, the reference %g" c.name k.what
                     k.est k.ref)
                (Float.abs (k.est -. k.ref)))
            cs)

let () = exit (run "Norn posteriordb" [ law_10 ])
