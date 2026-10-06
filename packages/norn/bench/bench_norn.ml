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
  type 'a t = 'a schools

  let walk c { mu; tau; theta } =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf mu in
    let tau = field c "tau" leaf tau in
    let theta = field c "theta" leaf theta in
    { mu; tau; theta }
end

let schools : Nx.float64_t schools Nx.Ptree.t =
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

let log_density =
  let setup () =
    let lp = M.log_density model y in
    (lp, M.init model y ~chains:16 (Nx.Rng.key 1))
  in
  Thumper.bench_with_setup ~setup "schools/16" (fun (lp, c) -> lp c)

let factors =
  let setup () =
    let d =
      D.iid [| 100_000 |]
        (D.student_t ~df:(f64 4.) ~loc:(f64 0.) ~scale:(f64 1.))
    in
    (d, Nx.Rng.normal (Nx.Rng.key 3) Nx.float64 [| 100_000 |])
  in
  Thumper.bench_with_setup ~setup "student_t/100000" (fun (d, x) ->
      D.factors d x)

let rhat =
  let setup () =
    Norn.Draws.v Nx.Ptree.tensor
      (Nx.Rng.normal (Nx.Rng.key 4) Nx.float64 [| 4; 1000; 10 |])
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
    (Posteriordb.all "../test/golden/posteriordb.golden")

let () =
  Thumper.run "norn"
    [
      Thumper.group "nuts" [ nuts_step 4; nuts_step 64 ];
      Thumper.group "model" [ log_density ];
      Thumper.group "dist" [ factors ];
      Thumper.group "diag" [ rhat ];
      Thumper.group "posteriordb" posteriordb;
    ]
  |> exit
