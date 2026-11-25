(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models
open Nested_runs

let init ?batch ?tolerance ?(budget = budget) m live =
  Norn.Nested.init t ?batch ?tolerance ~budget ~prior:m.prior
    ~likelihood:m.likelihood
    (m.draw (Nx.Rng.key 2) live)

let step m = Norn.Nested.step t ~prior:m.prior ~likelihood:m.likelihood

(* init *)

let refused =
  cases "refuses a batch, budget or tolerance out of range"
    ~id:(fun (name, _, _) -> name)
    [
      ( "a batch of every live point",
        (fun () -> init ~batch:20 conjugate 20),
        "Norn.Nested.init: batch = 20" );
      ( "a batch of none",
        (fun () -> init ~batch:0 conjugate 20),
        "Norn.Nested.init: batch = 0" );
      ( "a budget of none",
        (fun () -> init ~budget:0 conjugate 20),
        "Norn.Nested.init: budget = 0" );
      ( "a tolerance of zero",
        (fun () -> init ~tolerance:0. conjugate 20),
        "Norn.Nested.init: tolerance = 0" );
    ]
    (fun (_, f, sub) -> raises_match (Exn.invalid_arg ~sub) f)

let defaults =
  test "a batch defaults to half the live points, the tolerance to 1e-3"
    (fun () ->
      let s = init conjugate 21 in
      equal int 10 s.batch;
      equal float_exact 1e-3 (Nx.item [] s.tolerance))

let reads_others =
  test "a likelihood that reads the other points is refused" (fun () ->
      (* Centred on the points' mean, which their order leaves. *)
      let likelihood x =
        conjugate.likelihood (Nx.sub x (Nx.mean ~axes:[ 0 ] x))
      in
      raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
          Norn.Nested.init t ~budget ~prior:conjugate.prior ~likelihood
            (conjugate.draw (Nx.Rng.key 2) 20)))

let init_tests = group "init" [ refused; defaults; reads_others ]

(* step *)

(* Births: a dead point's shrinkage is one over the points live when it died, [1
   / (n - j + 1)] for the [j]-th of a step's deletions. A point drawn after [b]
   deaths is live at death [i] when [b <= i] and it has not died before [i], so
   counting them recovers each dead point's live count. *)
let born =
  cases "every dead point's live count is the points born before it"
    ~id:(fun (n, batch, steps) ->
      Printf.sprintf "%d live, batch %d, %d steps" n batch steps)
    [ (10, 5, 4); (10, 3, 5); (7, 1, 9); (12, 11, 2) ]
    (fun (n, batch, steps) ->
      let s =
        List.fold_left
          (fun s i -> step conjugate (Nx.Rng.key i) s)
          (Norn.Nested.init t ~batch ~budget:steps ~prior:conjugate.prior
             ~likelihood:conjugate.likelihood
             (conjugate.draw (Nx.Rng.key 0) n))
          (List.init steps (fun i -> i + 1))
      in
      let deaths = Int32.to_int (Nx.item [] s.deaths) in
      let dead_born = Nx.to_array s.dead_born and born = Nx.to_array s.born in
      let shrinkage = Nx.to_array s.shrinkage in
      for i = 0 to deaths - 1 do
        let live = ref 0 in
        for j = 0 to deaths - 1 do
          if Int32.to_int dead_born.(j) <= i && i <= j then incr live
        done;
        Array.iter (fun b -> if Int32.to_int b <= i then incr live) born;
        equal (float 1e-12)
          ~msg:(Printf.sprintf "dead point %d" i)
          (1. /. float_of_int !live)
          shrinkage.(i)
      done)

let past_budget =
  test "a step past the budget leaves the state" (fun () ->
      let s = init ~budget:1 conjugate 20 in
      let s = step conjugate (Nx.Rng.key 3) s in
      let s' = step conjugate (Nx.Rng.key 4) s in
      equal int32 1l (Nx.item [] s'.steps);
      equal (array float_exact) (Nx.to_array s.live) (Nx.to_array s'.live);
      equal float_exact (Nx.item [] s.log_evidence) (Nx.item [] s'.log_evidence))

(* Live points drawn at or below 0 under a likelihood that is NaN above 0: the
   first step's slice moves, from those points along both directions, reach past
   0. *)
let nan_likelihood =
  test "a NaN likelihood at a point raises" (fun () ->
      let likelihood x =
        let l = conjugate.likelihood x in
        Nx.where
          (Nx.greater_s (Nx.reshape [| -1 |] x) 0.)
          (Nx.full_like l Float.nan) l
      in
      let live = Nx.neg (Nx.abs (conjugate.draw (Nx.Rng.key 2) 20)) in
      let s =
        Norn.Nested.init t ~budget ~prior:conjugate.prior ~likelihood live
      in
      raises_match (Exn.invalid_arg ~sub:"Norn.Nested.step: the density is nan")
        (fun () ->
          Norn.Nested.step t ~prior:conjugate.prior ~likelihood (Nx.Rng.key 3) s))

let step_tests = group "step" [ born; past_budget; nan_likelihood ]

(* evidence *)

let normalised =
  test "its sample's log weights are normalised" (fun () ->
      let w = Norn.Evidence.sample (nest conjugate 50 (Nx.Rng.key 4)) in
      (* The 1550 dead points' log-sum-exp rounds by at most their count's ulps
         of 1, 3.4e-13. *)
      equal (float 1e-12) 0. (Nx.item [] (Nx.logsumexp w.log_weights)))

let expectation_of_sample =
  test "an expectation is the weighted mean of its sample" (fun () ->
      let z = nest conjugate 50 (Nx.Rng.key 4) in
      let w = Norn.Evidence.sample z in
      let p = Nx.exp w.log_weights and x = Nx.reshape [| -1 |] w.values in
      let mean, _ = Norn.Evidence.expectation t (fun x -> x) z in
      equal (float 1e-12)
        (Nx.item [] (Nx.sum (Nx.mul p x)))
        (Nx.item [ 0 ] mean))

let evidence_tests = group "evidence" [ normalised; expectation_of_sample ]

(* run *)

let converges =
  test "a run within its budget converges" (fun () ->
      equal string "converged"
        (stop_name (Norn.Evidence.stop (nest conjugate 50 (Nx.Rng.key 4)))))

let spent =
  test
    "a spent budget stops with the ln Z the live points could add, the \
     tolerance or more" (fun () ->
      match Norn.Evidence.stop (nest ~budget:2 conjugate 50 (Nx.Rng.key 4)) with
      | Remaining r -> at_least float_exact ~than:1e-3 r
      | stop -> failf "stopped as %s" (stop_name stop))

let pp_spent =
  test "pp states the ln Z a spent budget leaves" (fun () ->
      let z = nest ~budget:2 conjugate 50 (Nx.Rng.key 4) in
      match Norn.Evidence.stop z with
      | Remaining r ->
          let f x = Nx.item [] x in
          equal string
            (Printf.sprintf
               "ln Z = %.2f ± %.2f, H = %.2f nats, budget spent with %.2f nats \
                left"
               (lnz z) (err z)
               (f (Norn.Evidence.information z))
               r)
            (Format.asprintf "%a" Norn.Evidence.pp z)
      | stop -> failf "stopped as %s" (stop_name stop))

(* The module's keys: step [s] has the key [fold_in (fold_in k 0) s] and the
   evidence [fold_in k 1]. A run from the conjugate model's 20 prior draws stops
   after a step that leaves the live points' contribution below the tolerance,
   which the stop of the state's evidence reads. *)
let keyed =
  test "a run is init, then steps keyed by their count, then the evidence"
    (fun () ->
      let k = Nx.Rng.key 5 and live = conjugate.draw (Nx.Rng.key 6) 20 in
      let ran =
        Norn.Nested.run t ~budget ~prior:conjugate.prior
          ~likelihood:conjugate.likelihood k live
      in
      let converged s =
        Norn.Evidence.stop (Norn.Nested.evidence t k s) = Converged
      in
      let rec steps s =
        let n = Int32.to_int (Nx.item [] s.Norn.Nested.steps) in
        if converged s || n >= budget then s
        else steps (step conjugate (Nx.Rng.fold_in (Nx.Rng.fold_in k 0) n) s)
      in
      let s =
        steps
          (Norn.Nested.init t ~budget ~prior:conjugate.prior
             ~likelihood:conjugate.likelihood live)
      in
      let stepped = Norn.Nested.evidence t (Nx.Rng.fold_in k 1) s in
      List.iter
        (fun (name, f) ->
          equal float_exact ~msg:name
            (Nx.item [] (f ran))
            (Nx.item [] (f stepped)))
        [
          ("ln Z", Norn.Evidence.log_evidence);
          ("its error", Norn.Evidence.error);
          ("the effective groups", Norn.Evidence.effective_groups);
        ])

(* Nested sampling runs 13 live points: its run of likelihoods is the slowest
   cell of the recovery law, and its time is about linear in the points. Its
   largest error has a median of 2.3 standard errors over keys 1 to 40 and
   reaches 4.1, against the bound of 4.88. It is not held to the shrinkage of ln
   X: one that shrinks it by 1 / (n - j + 2) at the j-th deleted point, one past
   the true 1 / (n - j + 1), fails this test on 6 of keys 1 to 40, its largest
   error of median 3.6. *)
let recovers =
  test "replicates recover the dogs posterior" (fun () ->
      Compiled.here ();
      Recovery.nested ~replicates:16 ~draws:13 (Recovery.posterior "dogs-dogs"))

(* With no point of nonzero likelihood found, the volume left may hold all of
   [Z]: the live points could add any amount. *)
let nothing_found =
  test "a likelihood 0 at every point spends the budget with all of ln Z left"
    (fun () ->
      let likelihood x =
        Nx.full Nx.float64 [| (Nx.shape x).(0) |] Float.neg_infinity
      in
      let z =
        Norn.Nested.run t ~budget:3 ~prior:conjugate.prior ~likelihood
          (Nx.Rng.key 1)
          (conjugate.draw (Nx.Rng.key 2) 20)
      in
      match Norn.Evidence.stop z with
      | Remaining r -> equal float_exact Float.infinity r
      | stop -> failf "stopped as %s" (stop_name stop))

let run_tests =
  group "run" [ converges; spent; pp_spent; keyed; nothing_found; recovers ]

let () =
  exit (run "Norn.Nested" [ init_tests; step_tests; evidence_tests; run_tests ])
