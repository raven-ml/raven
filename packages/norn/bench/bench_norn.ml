(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Norn.Dist
module M = Norn_model

let f64 = Nx.scalar Nx.float64

(* Eight schools, non-centred *)

type 'a schools = { mu : 'a; tau : 'a; theta : 'a }

module Schools = struct
  type ('a, _) t = 'a schools

  let walk c { mu; tau; theta } =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf mu in
    let tau = field c "tau" leaf tau in
    let theta = field c "theta" leaf theta in
    { mu; tau; theta }
end

let schools : (Nx.host Nx.float64_t schools, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module Schools)

let y = Nx.create Nx.float64 [| 8 |] [| 28.; 8.; -3.; 7.; -1.; 1.; 18.; 12. |]

let sigma =
  Nx.create Nx.float64 [| 8 |] [| 15.; 10.; 16.; 11.; 9.; 11.; 10.; 18. |]

let model =
  M.noncentre
    (fun p -> p.theta)
    ( M.v Nx.float64 schools Nx.Ptree.tensor @@ fun () ->
      let mu = M.sample (D.normal ~loc:(f64 0.) ~scale:(f64 5.)) in
      let tau = M.sample (D.half_cauchy ~scale:(f64 5.)) in
      let theta = M.sample (D.iid [| 8 |] (D.normal ~loc:mu ~scale:tau)) in
      let y = M.sample (D.normal ~loc:theta ~scale:sigma) in
      ({ mu; tau; theta }, y) )

let u = M.coords model

(* Setup runs in the measuring worker: compiling spawns domains, after which a
   worker can no longer be forked. *)

(* A NUTS transition, compiled, from one state with one key: each run does the
   same work. *)
let nuts_step chains =
  let setup () =
    let lp = M.log_density model y in
    let start = M.init model y ~chains (Nx.Rng.key 1) in
    let s = Norn.Nuts.init u lp start in
    let sp = Norn.Nuts.ptree u in
    let step =
      Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
        (Norn.Nuts.step u lp)
    in
    let k = Nx.Rng.key 2 in
    ignore (step k s);
    (step, k, s)
  in
  Thumper.bench_with_setup ~setup (Printf.sprintf "schools/%d" chains)
    (fun (step, k, s) -> step k s)

(* Eight schools as a hand-written log density of a [chains; 10] position (mu,
   log tau and theta's standard coordinates), placed at [at]: no site computes
   eagerly, so a GPU set's init and step both compute under [jit]. *)
let schools_density at =
  let place x = Nx.place at (Nx.reshape [| 1; 8 |] x) in
  let y = place y and sigma = place sigma in
  fun x ->
    let mu = Nx.slice [ Nx.A; Nx.R (0, 1) ] x in
    let s = Nx.slice [ Nx.A; Nx.R (1, 2) ] x in
    let z = Nx.slice [ Nx.A; Nx.R (2, 10) ] x in
    let tau = Nx.exp s in
    let sq a = Nx.mul a a in
    let lmu = Nx.mul_s (sq (Nx.div_s mu 5.)) (-0.5) in
    let ltau = Nx.sub s (Nx.log (Nx.add_s (sq (Nx.div_s tau 5.)) 1.)) in
    let lz = Nx.mul_s (Nx.sum ~axes:[ 1 ] ~keepdims:true (sq z)) (-0.5) in
    let theta = Nx.add mu (Nx.mul tau z) in
    let ly =
      Nx.mul_s
        (Nx.sum ~axes:[ 1 ] ~keepdims:true (sq (Nx.div (Nx.sub y theta) sigma)))
        (-0.5)
    in
    Nx.reshape [| -1 |] (Nx.add (Nx.add lmu ltau) (Nx.add lz ly))

(* A NUTS transition of {!schools_density} at [at], compiled, from one state of
   4 chains with one key, which waits for what it computes. A GPU launches the
   kernels of the transition's loops on every trip, about 500 a transition. *)
let schools_nuts at =
  let u = Nx.Ptree.tensor in
  let lp = schools_density at in
  let uniform = Nx.Rng.uniform ~key:(Nx.Rng.key 1) Nx.float64 [| 4; 10 |] in
  let start = Nx.place at (Nx.sub_s (Nx.mul_s uniform 4.) 2.) in
  let sp = Norn.Nuts.ptree u in
  let init = Rune.jit Nx.Ptree.(u @-> returns sp) (Norn.Nuts.init u lp) in
  let step =
    Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp) (Norn.Nuts.step u lp)
  in
  let s = init start in
  let k = Nx.Rng.place at (Nx.Rng.key 2) in
  fun () -> Nx.Ptree.fold sp (fun _ x () -> Nx.sync x) (step k s) ()

(* Whether CUDA's first GPU opens, asked of a fresh process: a GPU's driver
   must not be initialized before the fork that isolates a case. *)
let cuda_opens () =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--cuda" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

(* A call made at a placement of any set. *)
type call = { at : 'd. 'd Nx.Placement.t -> unit -> unit }

(* [c]'s call at CUDA's first GPU. *)
let on_cuda c =
  match Rig.open_ (Rig_cuda.gpu 0) with
  | Error e -> failwith e
  | Ok gpu ->
      let module D = (val Nx.devices [ gpu ] : Nx.Devices) in
      c.at D.on

let nuts = { at = schools_nuts }

(* A NUTS transition of {!schools_density} on CUDA's first GPU, opened and
   compiled in the worker, from one state with one key. *)
let cuda_nuts_step =
  let setup () =
    let step = on_cuda nuts in
    step ();
    step
  in
  Thumper.bench_with_setup ~setup "schools-density-cuda/4" (fun step -> step ())

(* An HMC transition, compiled, from one state with one key. The default length
   and step size take one or two leapfrog steps. *)
let hmc_step chains =
  let setup () =
    let lp = M.log_density model y in
    let start = M.init model y ~chains (Nx.Rng.key 1) in
    let s = Norn.Hmc.init u lp start in
    let sp = Norn.Hmc.ptree u in
    let step =
      Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
        (Norn.Hmc.step u lp)
    in
    let k = Nx.Rng.key 2 in
    ignore (step k s);
    (step, k, s)
  in
  Thumper.bench_with_setup ~setup (Printf.sprintf "schools/%d" chains)
    (fun (step, k, s) -> step k s)

(* A NUTS and an HMC transition, eager, as a user calls [step] outside [jit],
   from the state [warmed] warmup transitions reach, with one key. *)
let eager_nuts ~warmed name u lp start =
  let setup () =
    let s = Norn.Nuts.init u lp (start ()) in
    Norn.Nuts.warmup u lp (Nx.Rng.key 3) ~steps:warmed s
  in
  Thumper.bench_with_setup ~setup ("eager/" ^ name) (fun s ->
      Norn.Nuts.step u lp (Nx.Rng.key 2) s)

let eager_hmc ~warmed name u lp start =
  let setup () =
    let s = Norn.Hmc.init u lp (start ()) in
    Norn.Hmc.warmup u lp (Nx.Rng.key 3) ~steps:warmed s
  in
  Thumper.bench_with_setup ~setup ("eager/" ^ name) (fun s ->
      Norn.Hmc.step u lp (Nx.Rng.key 2) s)

(* The schools after 50 warmup transitions and the Poisson Gaussian process of
   posteriordb after 20, at 4 chains: NUTS takes 7 leapfrog steps on the
   schools and 2 to 127 on the process, HMC 4 and 8. The process's warmup is
   short to keep its setup within a trial's deadline. *)
let eager_schools step =
  step ~warmed:50 "schools/4" u (M.log_density model y) (fun () ->
      M.init model y ~chains:4 (Nx.Rng.key 1))

let eager_gp step =
  match
    List.find
      (fun (Posteriordb.Posterior p) -> p.name = "gp_pois_regr-gp_pois_regr")
      (Posteriordb.all "../test/model/support/posteriordb.golden")
  with
  | Posteriordb.Posterior p ->
      step ~warmed:20 "gp_pois_regr/4" (M.coords p.model) (M.log_density p.model p.y)
        (fun () -> M.init p.model p.y ~chains:4 (Nx.Rng.key 1))

(* An ensemble transition, compiled, of 64 walkers in one ensemble. *)
let ensemble_step =
  let setup () =
    let lp = M.log_density model y in
    let start = M.init model y ~chains:64 (Nx.Rng.key 1) in
    let s = Norn.Ensemble.init u lp start in
    let sp = Norn.Ensemble.ptree u in
    let step =
      Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
        (Norn.Ensemble.step u lp)
    in
    let k = Nx.Rng.key 2 in
    ignore (step k s);
    (step, k, s)
  in
  Thumper.bench_with_setup ~setup "schools/64" (fun (step, k, s) -> step k s)

(* Pathfinder, eager, from 4 chains: 50 L-BFGS iterates per path, their
   Gaussians and the bounds' draws. *)
let pathfinder =
  let setup () =
    (M.log_density model y, M.init model y ~chains:4 (Nx.Rng.key 1))
  in
  Thumper.bench_with_setup ~setup "schools/4" (fun (lp, start) ->
      Norn.Gaussian.pathfinder u ~steps:50 lp (Nx.Rng.key 2) start)

(* A mean-field fit, eager, of 4 chains in 200 steps, after its search of 50
   steps per step size. *)
let vi_fit =
  let setup () =
    (M.log_density model y, M.init model y ~chains:4 (Nx.Rng.key 1))
  in
  Thumper.bench_with_setup ~setup "schools/4" (fun (lp, start) ->
      Norn.Gaussian.vi u ~steps:200 lp (Nx.Rng.key 2) start)

(* A Gaussian likelihood of scale 0.5 under a uniform prior on [-5, 5]^5, over
   400 points. *)
let cube_prior x =
  let inside =
    Nx.all ~axes:[ 1 ] (Nx.less_equal (Nx.abs x) (Nx.scalar Nx.float64 5.))
  in
  let c = (Nx.shape x).(0) in
  Nx.where inside
    (Nx.full Nx.float64 [| c |] (-5. *. Float.log 10.))
    (Nx.full Nx.float64 [| c |] Float.neg_infinity)

let gaussian x =
  Nx.sub_s
    (Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-2.))
    (5. *. (Float.log 0.5 +. (0.5 *. Float.log (2. *. Float.pi))))

let points k =
  Nx.mul_s (Nx.sub_s (Nx.Rng.uniform ~key:k Nx.float64 [| 400; 5 |]) 0.5) 10.

(* A nested sampling step, compiled: 200 points replaced. *)
let nested_step =
  let t = Nx.Ptree.tensor in
  let setup () =
    let s =
      Norn.Nested.init t ~budget:10 ~prior:cube_prior ~likelihood:gaussian
        (points (Nx.Rng.key 1))
    in
    let sp = Norn.Nested.ptree t in
    let step =
      Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
        (Norn.Nested.step t ~prior:cube_prior ~likelihood:gaussian)
    in
    let k = Nx.Rng.key 2 in
    ignore (step k s);
    (step, k, s)
  in
  Thumper.bench_with_setup ~setup "gaussian5/400" (fun (step, k, s) -> step k s)

(* A nested sampling run, compiled, to its evidence: from 100 of the points,
   about 30 steps of 50 replacements, then the evidence's errors, whose cost
   grows with the dead buffer, [budget * batch] points. *)
let nested_run =
  let t = Nx.Ptree.tensor in
  let setup () =
    let run =
      Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> returns (Norn.Evidence.ptree t))
        (fun k ->
          Norn.Nested.run t ~budget:60 ~prior:cube_prior ~likelihood:gaussian
            (Nx.Rng.fold_in k 0)
            (Nx.slice [ Nx.R (0, 100) ] (points (Nx.Rng.fold_in k 1))))
    in
    let k = Nx.Rng.key 2 in
    ignore (run k);
    (run, k)
  in
  Thumper.bench_with_setup ~setup "run/gaussian5/100" (fun (run, k) -> run k)

(* A tempering step, compiled: 20 chains of 20 states, moved by each kernel. *)
let smc_step move name =
  let t = Nx.Ptree.tensor in
  let setup () =
    let s =
      Norn.Smc.init t ~move ~prior:cube_prior ~likelihood:gaussian
        (points (Nx.Rng.key 1))
    in
    let sp = Norn.Smc.ptree t in
    let step =
      Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
        (Norn.Smc.step t ~prior:cube_prior ~likelihood:gaussian)
    in
    let k = Nx.Rng.key 2 in
    ignore (step k s);
    (step, k, s)
  in
  Thumper.bench_with_setup ~setup (name ^ "/gaussian5/400") (fun (step, k, s) ->
      step k s)

let log_density =
  let setup () =
    let lp = M.log_density model y in
    (lp, M.init model y ~chains:16 (Nx.Rng.key 1))
  in
  Thumper.bench_with_setup ~setup "schools/16" (fun (lp, c) -> lp c)

(* A family's factors at [count] of its draws, 100000 by default, eager, and
   as a compiled value and gradient in one parameter [p]. *)
let rows ?(count = 100_000) name family p =
  let x = D.sample (Nx.Rng.key 6) (D.iid [| count |] (family (f64 p))) in
  let n = Printf.sprintf "%s/%d" name count in
  let eager =
    let setup () = (D.iid [| count |] (family (f64 p)), x) in
    Thumper.bench_with_setup ~setup n (fun (d, x) -> D.factors d x)
  in
  let grad =
    let setup () =
      let lp p = Nx.sum (D.factors (D.iid [| count |] (family p)) x) in
      let grad =
        Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (Rune.value_and_grad Nx.Ptree.tensor lp)
      in
      ignore (grad (f64 p));
      (grad, f64 p)
    in
    Thumper.bench_with_setup ~setup (n ^ "/grad") (fun (grad, p) -> grad p)
  in
  [ eager; grad ]

let count_factors =
  rows "poisson" (fun rate -> D.poisson ~rate) 30.
  @ rows "binomial"
      (fun logits -> D.binomial ~trials:(Nx.scalar Nx.int32 1000l) ~logits)
      (-1.)
  @ rows "neg_binomial"
      (fun mean -> D.neg_binomial ~mean ~dispersion:(f64 4.5))
      30.

let continuous_factors =
  rows "gamma" (fun a -> D.gamma ~concentration:a ~rate:(f64 2.)) 30.
  @ rows "inverse_gamma"
      (fun a -> D.inverse_gamma ~concentration:a ~scale:(f64 2.))
      30.
  @ rows "beta" (fun a -> D.beta ~a ~b:(f64 4.5)) 30.
  @ rows "student_t"
      (fun df -> D.student_t ~df ~loc:(f64 0.) ~scale:(f64 1.))
      4.
  @ rows "dirichlet"
      (fun a ->
        let weights = Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |] in
        D.dirichlet ~concentration:(Nx.mul a weights))
      30.

(* The new families: a truncation in a far tail, von Mises, LKJ of 5 x 5
   factors, and 100 sequences of a 3-state HMM over 100 steps. *)
let new_factors =
  let normal = D.normal ~loc:(f64 0.) ~scale:(f64 1.) in
  rows "truncated_normal" (fun low -> D.truncated ~low normal) 30.
  @ rows "von_mises" (fun k -> D.von_mises ~loc:(f64 0.5) ~concentration:k) 3.
  @ rows ~count:10_000 "lkj_cholesky"
      (fun eta -> D.lkj_cholesky 5 ~concentration:eta)
      2.
  @ rows ~count:100 "hmm"
      (fun p ->
        let stay = Nx.mul_s (Nx.eye Nx.float64 3) 2. in
        let loc = Nx.create Nx.float64 [| 3 |] [| -2.; 0.; 2. |] in
        D.hmm
          ~initial:(Nx.zeros Nx.float64 [| 3 |])
          ~transition:(Nx.mul p stay) 100
          (D.normal ~loc ~scale:(f64 1.)))
      1.

let rhat =
  let setup () =
    Norn.Draws.v Nx.Ptree.tensor
      (Nx.Rng.normal ~key:(Nx.Rng.key 4) Nx.float64 [| 4; 1000; 10 |])
  in
  Thumper.bench_with_setup ~setup "rhat/4x1000x10" (fun d ->
      Norn.Diag.rhat Nx.Ptree.tensor d)

(* The log density of a posteriordb posterior with its gradient, compiled, at 8
   chains. A sampler's wall time per effective draw is this time over its
   effective draws per gradient. *)
let posteriordb =
  List.map
    (fun (Posteriordb.Posterior p) ->
      let setup () =
        let u = M.coords p.model in
        let lp = M.log_density p.model p.y in
        let grad =
          Rune.jit
            Nx.Ptree.(u @-> returns (pair tensor u))
            (Rune.value_and_grad u (fun c -> Nx.sum (lp c)))
        in
        let c = M.init p.model p.y ~chains:8 (Nx.Rng.key 5) in
        ignore (grad c);
        (grad, c)
      in
      Thumper.bench_with_setup ~setup p.name (fun (grad, c) -> grad c))
    (Posteriordb.all "../test/model/support/posteriordb.golden")

(* Cold compiles: each sample runs this executable again as [--cold id] over
   an empty disk cache, so a sample is a fresh process's first NUTS transition
   of {!schools_density}, its init and step traced, lowered and compiled by
   tolk and the kernel compiler on one domain. A warm process cannot hold this
   cost: tolk's caches serve a program after its first compile. The [cuda/]
   case runs on CUDA's first GPU and is left out without one. *)
let cold_cases =
  [
    ("nuts/schools-density-4", fun () -> schools_nuts Nx.Host.on ());
    ("cuda/nuts/schools-density-4", fun () -> on_cuda nuts ());
  ]

let rec remove path =
  if Sys.is_directory path then (
    Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
    Sys.rmdir path)
  else Sys.remove path

(* One sample: a fresh process's first call of [id] over an empty disk cache
   and an empty cache of the GPU driver's binaries. *)
let cold id () =
  let dir = Filename.temp_dir "bench-norn-compile" "" in
  Fun.protect ~finally:(fun () -> remove dir) @@ fun () ->
  let exe = Sys.executable_name in
  let env =
    Array.append
      [|
        "CACHEDB=" ^ Filename.concat dir "cache.db";
        "CUDA_CACHE_PATH=" ^ Filename.concat dir "driver";
        "PARALLEL=1";
      |]
      (Unix.environment ())
  in
  let null = Unix.openfile Filename.null [ Unix.O_WRONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close null) @@ fun () ->
  let pid =
    Unix.create_process_env exe [| exe; "--cold"; id |] env Unix.stdin null
      Unix.stderr
  in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED 0 -> ()
  | _ -> failwith ("bench_norn: the first call of " ^ id ^ " failed")

let compile argv =
  let cuda = cuda_opens () in
  let runs id = cuda || not (String.starts_with ~prefix:"cuda/" id) in
  Thumper.run "compile" ~argv
    ~config:
      Thumper.Config.(
        default |> samples 3 |> warmup 0. |> deadline 300.
        |> metrics [ Thumper.Metric.wall_time ])
    ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
    (List.filter_map
       (fun (id, _) ->
         if runs id then Some (Thumper.bench id (cold id)) else None)
       cold_cases)

let run () =
  let cuda = if cuda_opens () then [ cuda_nuts_step ] else [] in
  Thumper.run "norn"
    [
      Thumper.group "nuts"
        ([
          nuts_step 4;
          nuts_step 64;
          eager_schools eager_nuts;
          eager_gp eager_nuts;
        ]
        @ cuda);
      Thumper.group "hmc"
        [
          hmc_step 4;
          hmc_step 64;
          hmc_step 1024;
          eager_schools eager_hmc;
          eager_gp eager_hmc;
        ];
      Thumper.group "ensemble" [ ensemble_step ];
      Thumper.group "pathfinder" [ pathfinder ];
      Thumper.group "vi" [ vi_fit ];
      Thumper.group "nested" [ nested_step; nested_run ];
      Thumper.group "smc"
        [ smc_step Norn.Smc.Hmc "hmc"; smc_step Norn.Smc.Slice "slice" ];
      Thumper.group "model" [ log_density ];
      Thumper.group "dist" (count_factors @ continuous_factors @ new_factors);
      Thumper.group "diag" [ rhat ];
      Thumper.group "posteriordb" posteriordb;
    ]

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cuda" ] ->
      exit (if Result.is_ok (Rig.open_ (Rig_cuda.gpu 0)) then 0 else 1)
  | [ _; "--cold"; id ] -> (
      match List.assoc_opt id cold_cases with
      | Some f ->
          let t0 = Unix.gettimeofday () in
          f ();
          Printf.printf "%.3f\n" ((Unix.gettimeofday () -. t0) *. 1000.)
      | None ->
          prerr_endline ("bench_norn: no case " ^ id);
          exit 2)
  | exe :: "compile" :: rest -> exit (compile (Array.of_list (exe :: rest)))
  | _ -> exit (run ())
