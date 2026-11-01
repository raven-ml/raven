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

let () =
  Thumper.run "norn"
    [
      Thumper.group "nuts" [ nuts_step 4; nuts_step 64 ];
      Thumper.group "hmc" [ hmc_step 4; hmc_step 64; hmc_step 1024 ];
      Thumper.group "ensemble" [ ensemble_step ];
      Thumper.group "pathfinder" [ pathfinder ];
      Thumper.group "vi" [ vi_fit ];
      Thumper.group "nested" [ nested_step ];
      Thumper.group "smc"
        [ smc_step Norn.Smc.Hmc "hmc"; smc_step Norn.Smc.Slice "slice" ];
      Thumper.group "model" [ log_density ];
      Thumper.group "dist" (count_factors @ continuous_factors @ new_factors);
      Thumper.group "diag" [ rhat ];
      Thumper.group "posteriordb" posteriordb;
    ]
  |> exit
