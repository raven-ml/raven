(* Law 10 over every sampler and the six posteriordb posteriors, and HMC on
   Neal's funnel, non-centred, at 1024 chains. The law is Recovery's; the test
   suite holds it on a few cells, this sweep on all 31.

   It is outside runtest by the maintainer's decision of 2026-10-11: runtest
   holds a small calibrated law, test_posteriordb, and this executable the full
   sweep, whose 31 inference runs take seven to twelve minutes, beyond the
   minute a test has. Run it after a change to a sampler or a diagnostic, from
   the repository root:

   cd packages/norn/test/model && dune exec ./posteriordb_sweep.exe *)

open Windtrap
module M = Norn_model
module P = Posteriordb
module R = Recovery

(* The cells, each sized as a careful user would run it: the smallest run, among
   the sizes tried, whose diagnostics pass on the test's key, so that it reaches
   the recovery branch. A cell that the diagnostics flag at every size names its
   pathology. *)

let nuts_cell (P.Posterior p as post) =
  match p.name with
  | "gp_pois_regr-gp_pois_regr" ->
      (* At the default acceptance a transition or two in thousands diverges in
         the hyperparameters' funnel; at 0.95 none does, and 250 draws reach an
         effective size of 400 where 150 reach 300. *)
      R.chain_cell (R.nuts ~accept:0.95 ~draws:250 ()) post
  | _ -> R.chain_cell (R.nuts ~draws:150 ()) post

let hmc_cell (P.Posterior p as post) =
  let draws =
    match p.name with
    | "arK-arK" -> 100 (* at 50 an effective size of 288 *)
    | "gp_pois_regr-gp_pois_regr" ->
        (* R-hat 1.021 at 100 draws, 1.011 at 200, an effective size of 258 at
           400 *)
        800
    | _ -> 50
  in
  (* Known pathologies, flagged at every size: on low_dim_gauss_mix a chain that
     starts at the degenerate solution, both components one wide Gaussian, stays
     there (R-hat 1.03). A NUTS chain does too: NUTS at 64 chains is flagged
     there, and its cell runs 8, which miss that start. On sblrc-blr a few
     chains end warmup far in the tails, where the shared step size and length
     are too short; they drift in, accepting nearly every transition, and their
     E-BFMI is low. NUTS at 64 chains reaches the bulk on every chain. *)
  R.chain_cell (R.hmc ~draws) post

let ensemble_cell (P.Posterior p as post) =
  let draws =
    match p.name with
    | "eight_schools-eight_schools_noncentered" ->
        800 (* an effective size of 371 at 600 *)
    | _ -> 400 (* at 200, R-hat up to 1.016 *)
  in
  (* Known pathology, flagged at every size: on gp_pois_regr the stretch move
     barely moves the Gaussian process's hyperparameters, whose effective size
     stays near 200 from 800 draws to 1600. *)
  R.chain_cell (R.ensemble ~ensembles:4 ~draws) post

(* Neal's funnel: v ~ N(0, 3), x ~ N(0, exp (v / 2)) in nine dimensions,
   non-centred. Its exact moments: v's mean 0 and sd 3; each x's mean 0 and sd
   exp (9 / 4), since E exp v = exp 4.5. *)

type 'a funnel = { v : 'a; x : 'a }

module Funnel = struct
  type ('a, _) t = 'a funnel

  let walk c { v; x } =
    let open Nx.Ptree.Walk in
    let v = field c "v" leaf v in
    let x = field c "x" leaf x in
    { v; x }
end

let funnel_latent : (Nx.host Nx.float64_t funnel, Nx.host) Nx.Ptree.t =
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

let funnel_refs : P.reference list =
  let exact param sd =
    { P.param; index = 0; mean = 0.; sd; mcse_mean = 0.; mcse_sd = 0. }
  in
  [ exact "v" 3.; exact "x" (Float.exp 2.25) ]

let funnel_cell =
  let kernel = { (R.hmc ~draws:50) with chains = 1024; superchains = 8 } in
  let run = R.fit kernel funnel funnel_latent () in
  {
    R.name = "HMC › a funnel at 1024 chains";
    sampler = kernel.label;
    comparisons = 2 * List.length funnel_refs;
    quantile = R.normal_quantile;
    verdict =
      (fun k ->
        let d, stats = run k in
        let value name (f : Nx.host Nx.float64_t funnel) =
          if name = "v" then f.v else f.x
        in
        R.judge_chains funnel_latent ~superchains:kernel.superchains funnel_refs
          value stats d);
  }

let cells =
  let all = P.all "support/posteriordb.golden" in
  List.map nuts_cell all @ List.map hmc_cell all @ List.map ensemble_cell all
  @ List.map (R.replicated_cell (R.smc ~replicates:4 ~draws:1000)) all
  @ List.map (R.replicated_cell (R.nested ~replicates:4 ~draws:500)) all
  @ [ funnel_cell ]

let () = exit (run "Norn posteriordb sweep" [ R.law cells ])
