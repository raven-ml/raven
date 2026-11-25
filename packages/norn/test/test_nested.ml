(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models
open Nested_runs

(* [init m live] is the state of [live] prior draws of [m]. *)
let init ?batch ?tolerance ?(budget = budget) m live =
  Norn.Nested.init t ?batch ?tolerance ~budget ~prior:m.prior
    ~likelihood:m.likelihood
    (m.draw (Nx.Rng.key 2) live)

let step m = Norn.Nested.step t ~prior:m.prior ~likelihood:m.likelihood

(* [steps m n s] is [n] steps from [s], keyed 1 to [n]. *)
let steps m n s =
  List.fold_left (fun s i -> step m (Nx.Rng.key i) s) s (List.init n succ)

let ints x = Array.map Int32.to_int (Nx.to_array x)
let deaths s = Int32.to_int (Nx.item [] s.Norn.Nested.deaths)

(* [bad f v x] is [f x] with [v] where the position is above 0. *)
let bad f v x =
  let l = f x in
  Nx.where (Nx.greater_s (Nx.reshape [| -1 |] x) 0.) (Nx.full_like l v) l

(* [leaves s] is each tensor of the state [s] by its path, as floats. *)
let leaves s =
  List.rev
    (Nx.Ptree.fold (Norn.Nested.ptree t)
       (fun p x acc ->
         ( Nx.Ptree.Path.to_string p,
           Nx.to_array (Nx.reshape [| -1 |] (Nx.cast Nx.float64 x)) )
         :: acc)
       s [])

(* ptree *)

let ptree_tests =
  group "ptree"
    [
      test "reports batch and budget" (fun () ->
          let reports =
            List.filter_map
              (function
                | Nx.Ptree.Report (p, Int i) ->
                    Some (Printf.sprintf "%s %d" (Nx.Ptree.Path.to_string p) i)
                | _ -> None)
              (Nx.Ptree.visits (Norn.Nested.ptree t)
                 (init ~batch:3 conjugate 10))
          in
          equal (list string)
            [ "batch 3"; Printf.sprintf "budget %d" budget ]
            reports);
    ]

(* init *)

let init_tests =
  group "init"
    [
      test "a batch defaults to half the live points, the tolerance to 1e-3"
        (fun () ->
          let s = init conjugate 21 in
          equal int 10 s.batch;
          equal float_exact 1e-3 (Nx.item [] s.tolerance));
      cases "a batch outside [1, n) is refused" ~id:string_of_int
        [ 0; -1; 20; 21 ] (fun batch ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
              init ~batch conjugate 20));
      cases "a budget below one is refused" ~id:string_of_int [ 0; -1 ]
        (fun budget ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
              init ~budget conjugate 20));
      cases "a tolerance that is not positive is refused"
        ~id:(Printf.sprintf "%g") [ 0.; -1e-3; Float.nan ] (fun tolerance ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
              init ~tolerance conjugate 20));
      cases
        "a density is refused that does not return one log density per point"
        ~id:fst
        [
          ("one value for all points", fun x -> Nx.sum (conjugate.likelihood x));
          ( "centred on the points' mean",
            fun x -> conjugate.likelihood (Nx.sub x (Nx.mean ~axes:[ 0 ] x)) );
          ( "normalised by the points' largest",
            fun x ->
              let l = conjugate.likelihood x in
              Nx.sub l (Nx.max l) );
          ( "centred on the points' median",
            fun x ->
              conjugate.likelihood (Nx.sub x (Nx.quantile ~axis:0 [| 0.5 |] x))
          );
        ]
        (fun (_, likelihood) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
              Norn.Nested.init t ~budget ~prior:conjugate.prior ~likelihood
                (conjugate.draw (Nx.Rng.key 2) 20)));
      cases "a density NaN or +inf at a point is refused"
        ~id:(fun (which, v) -> Printf.sprintf "%s %g" which v)
        [
          ("likelihood", Float.nan);
          ("likelihood", Float.infinity);
          ("prior", Float.nan);
        ]
        (fun (which, v) ->
          let prior, likelihood =
            if which = "prior" then (bad conjugate.prior v, conjugate.likelihood)
            else (conjugate.prior, bad conjugate.likelihood v)
          in
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
              Norn.Nested.init t ~budget ~prior ~likelihood
                (conjugate.draw (Nx.Rng.key 2) 20)));
      test "live points that hold no tensor are refused" (fun () ->
          let none () = Nx.zeros Nx.float64 [| 1 |] in
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
              Norn.Nested.init Nx.Ptree.unit ~budget ~prior:none
                ~likelihood:none ()));
      test "a start of float tensors of two dtypes is refused" (fun () ->
          (* A float32 tensor beside a float64 one. *)
          let u = Nx.Ptree.(pair tensor tensor) in
          let density (a, b) =
            Nx.neg
              (Nx.add
                 (Nx.sum ~axes:[ 1 ] (Nx.square (Nx.cast Nx.float64 a)))
                 (Nx.sum ~axes:[ 1 ] (Nx.square b)))
          in
          let x =
            (Nx.ones Nx.float32 [| 10; 1 |], Nx.ones Nx.float64 [| 10; 2 |])
          in
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.init") (fun () ->
              Norn.Nested.init u ~budget ~prior:density ~likelihood:density x));
    ]

(* step *)

(* Every step starts from [n] live points, so the [j]-th of its deletions, from
   [j = 1], shrinks ln X by [1 / (n - j + 1)]: dead point [i] by [1 / (n - i mod
   batch)]. The examples are the rows that held the shrinkage when it was
   checked by counting births. *)
let shrinkage_law =
  prop ~count:20
    ~examples:[ (10, 5, 4); (10, 3, 5); (7, 1, 9); (12, 11, 2) ]
    "the j-th of a step's deletions shrinks ln X by 1 / (n - j + 1)"
    Gen.(
      with_pp
        (fun ppf (n, b, k) ->
          Format.fprintf ppf "%d live, batch %d, %d steps" n b k)
        (let* n = int_range 2 12 in
         let* batch = int_range 1 (n - 1) in
         let+ k = int_range 1 6 in
         (n, batch, k)))
    (fun (n, batch, k) ->
      cover "a batch of one" (batch = 1);
      cover "a batch of all but one" (batch = n - 1);
      let s = steps conjugate k (init ~batch ~budget:k conjugate n) in
      let expected =
        Array.init (deaths s) (fun i -> 1. /. float_of_int (n - (i mod batch)))
      in
      equal
        (array (float 1e-15))
        expected
        (Array.sub (Nx.to_array s.shrinkage) 0 (deaths s)))

(* [counted rows f] is [f] recording in [rows] the leading axis of every
   position it reads. *)
let counted rows f x =
  rows := (Nx.shape x).(0) :: !rows;
  f x

let step_tests =
  group "step"
    [
      shrinkage_law;
      test "each replacement is born at the deaths before it" (fun () ->
          let s = steps conjugate 2 (init ~batch:3 conjugate 10) in
          let born = ints s.born in
          equal int ~msg:"live points born at the last step's deaths" 3
            (Array.fold_left
               (fun c b -> if b = deaths s then c + 1 else c)
               0 born);
          Array.iter
            (at_most int ~msg:"a live point's birth" ~than:(deaths s))
            born;
          Array.iteri
            (fun i b ->
              if i < deaths s then
                at_most int ~msg:"a dead point's birth" ~than:i b)
            (ints s.dead_born));
      cases "the dead are lowest first, and every live point lies above them"
        ~id:(fun m -> m.name)
        [ conjugate; indicator ]
        (fun m ->
          let s = steps m 5 (init m 20) in
          let dead = Array.sub (Nx.to_array s.dead_likelihood) 0 (deaths s) in
          Array.iteri
            (fun i l ->
              if i > 0 then
                at_least float_exact ~msg:"a dead point" ~than:dead.(i - 1) l)
            dead;
          let last = dead.(Array.length dead - 1) in
          Array.iter
            (at_least float_exact ~msg:"a live point" ~than:last)
            (Nx.to_array s.likelihood);
          Array.iter
            (fun r ->
              greater float_exact ~msg:"a rank" ~than:0. r;
              less float_exact ~msg:"a rank" ~than:1. r)
            (Nx.to_array s.rank);
          Array.iter
            (fun l ->
              at_least int ~msg:"a lineage" ~than:0 l;
              less int ~msg:"a lineage" ~than:20 l)
            (ints s.lineage));
      test "the densities are evaluated on the batch points being replaced"
        (fun () ->
          let rows = ref [] in
          ignore
            (Norn.Nested.step t
               ~prior:(counted rows conjugate.prior)
               ~likelihood:(counted rows conjugate.likelihood)
               (Nx.Rng.key 3)
               (init ~batch:3 conjugate 10));
          equal (list int) [ 3 ] (List.sort_uniq compare !rows));
      test "a step past the budget leaves the state" (fun () ->
          let s = step conjugate (Nx.Rng.key 3) (init ~budget:1 conjugate 20) in
          let s' = step conjugate (Nx.Rng.key 4) s in
          equal (list (pair string (array float_exact))) (leaves s) (leaves s'));
      (* Live points drawn at or below 0 under a likelihood bad above 0: the
         first step's slice moves, along both directions, reach past 0. *)
      cases "a likelihood NaN or +inf at a point raises"
        ~id:(Printf.sprintf "%g") [ Float.nan; Float.infinity ] (fun v ->
          let likelihood = bad conjugate.likelihood v in
          let live = Nx.neg (Nx.abs (conjugate.draw (Nx.Rng.key 2) 20)) in
          let s =
            Norn.Nested.init t ~budget ~prior:conjugate.prior ~likelihood live
          in
          raises_match (Exn.invalid_arg ~sub:"Norn.Nested.step") (fun () ->
              Norn.Nested.step t ~prior:conjugate.prior ~likelihood
                (Nx.Rng.key 3) s));
    ]

(* evidence *)

let evidence_tests =
  group "evidence"
    [
      test "its sample is every dead point, the padding at weight zero"
        (fun () ->
          let s = steps conjugate 3 (init ~batch:5 ~budget:10 conjugate 10) in
          let w =
            Norn.Evidence.sample (Norn.Nested.evidence t (Nx.Rng.key 1) s)
          in
          let lw = Nx.to_array w.log_weights in
          equal int ~msg:"draws" ((10 * 5) + 10) (Array.length lw);
          (* The dead buffer's rows past the 15 deaths, before the live. *)
          equal (array float_exact) ~msg:"the padding"
            (Array.make 35 Float.neg_infinity)
            (Array.sub lw 15 35));
      test "its sample's log weights are normalised" (fun () ->
          let w = Norn.Evidence.sample (nest conjugate 50 (Nx.Rng.key 4)) in
          (* The 1550 dead points' log-sum-exp rounds by at most their count's
             ulps of 1, 3.4e-13. *)
          equal (float 1e-12) 0. (Nx.item [] (Nx.logsumexp w.log_weights)));
      test "its key moves the error alone" (fun () ->
          let s = steps conjugate 4 (init conjugate 20) in
          let z1 = Norn.Nested.evidence t (Nx.Rng.key 1) s
          and z2 = Norn.Nested.evidence t (Nx.Rng.key 2) s in
          let scalar f z = Nx.item [] (f z) in
          equal float_exact ~msg:"ln Z"
            (scalar Norn.Evidence.log_evidence z1)
            (scalar Norn.Evidence.log_evidence z2);
          equal float_exact ~msg:"H"
            (scalar Norn.Evidence.information z1)
            (scalar Norn.Evidence.information z2);
          let weights z = Nx.to_array (Norn.Evidence.sample z).log_weights in
          equal (array float_exact) ~msg:"the sample's weights" (weights z1)
            (weights z2));
      (* H is the posterior's divergence from the prior, [E_post (ln L) - ln Z],
         an expectation over the sample: a sum of 1550 terms, rounded within
         their count's ulps. *)
      test "information is the sample's mean of ln L less ln Z" (fun () ->
          let z = nest conjugate 50 (Nx.Rng.key 4) in
          let w = Norn.Evidence.sample z in
          let p = Nx.to_array (Nx.exp w.log_weights) in
          let l = Nx.to_array (conjugate.likelihood w.values) in
          let mean = ref 0. in
          Array.iteri
            (fun i p -> if p > 0. then mean := !mean +. (p *. l.(i)))
            p;
          equal
            (float_rel ~rel:1e-12 ~abs:1e-12)
            (!mean -. lnz z)
            (Nx.item [] (Norn.Evidence.information z)));
      test "an expectation is the weighted mean of its sample" (fun () ->
          let z = nest conjugate 50 (Nx.Rng.key 4) in
          let w = Norn.Evidence.sample z in
          let p = Nx.exp w.log_weights and x = Nx.reshape [| -1 |] w.values in
          let mean, _ = Norn.Evidence.expectation t (fun x -> x) z in
          equal (float 1e-12)
            (Nx.item [] (Nx.sum (Nx.mul p x)))
            (Nx.item [ 0 ] mean));
    ]

(* run *)

(* The module's keys: step [s] has the key [fold_in (fold_in k 0) s] and the
   evidence [fold_in k 1]. A run from the conjugate model's 20 prior draws stops
   after a step that leaves the live points' contribution below the tolerance,
   which the stop of the state's evidence reads. *)
let keyed () =
  let k = Nx.Rng.key 5 and live = conjugate.draw (Nx.Rng.key 6) 20 in
  let ran =
    Norn.Nested.run t ~budget ~prior:conjugate.prior
      ~likelihood:conjugate.likelihood k live
  in
  let converged s =
    Norn.Evidence.stop (Norn.Nested.evidence t k s) = Converged
  in
  let rec go s =
    let n = Int32.to_int (Nx.item [] s.Norn.Nested.steps) in
    if converged s || n >= budget then s
    else go (step conjugate (Nx.Rng.fold_in (Nx.Rng.fold_in k 0) n) s)
  in
  let s =
    go
      (Norn.Nested.init t ~budget ~prior:conjugate.prior
         ~likelihood:conjugate.likelihood live)
  in
  let stepped = Norn.Nested.evidence t (Nx.Rng.fold_in k 1) s in
  List.iter
    (fun (name, f) ->
      equal float_exact ~msg:name (Nx.item [] (f ran)) (Nx.item [] (f stepped)))
    [
      ("ln Z", Norn.Evidence.log_evidence);
      ("its error", Norn.Evidence.error);
      ("the effective groups", Norn.Evidence.effective_groups);
    ]

let run_tests =
  group "run"
    [
      test "a run within its budget converges" (fun () ->
          equal stop Converged
            (Norn.Evidence.stop (nest conjugate 50 (Nx.Rng.key 4))));
      test
        "a spent budget stops with the ln Z the live points could add, the \
         tolerance or more" (fun () ->
          match
            Norn.Evidence.stop (nest ~budget:2 conjugate 50 (Nx.Rng.key 4))
          with
          | Remaining r -> at_least float_exact ~than:1e-3 r
          | s -> failf "stopped as %a" (Testable.pp stop) s);
      test "pp states the ln Z a spent budget leaves" (fun () ->
          let z = nest ~budget:2 conjugate 50 (Nx.Rng.key 4) in
          match Norn.Evidence.stop z with
          | Remaining r ->
              equal string
                (Printf.sprintf
                   "ln Z = %.2f ± %.2f, H = %.2f nats, budget spent with %.2f \
                    nats left"
                   (lnz z) (err z)
                   (Nx.item [] (Norn.Evidence.information z))
                   r)
                (Format.asprintf "%a" Norn.Evidence.pp z)
          | s -> failf "stopped as %a" (Testable.pp stop) s);
      (* With no point of nonzero likelihood found, the volume left may hold all
         of [Z]. *)
      test
        "a likelihood 0 at every point spends the budget with all of ln Z left"
        (fun () ->
          let likelihood x =
            Nx.full Nx.float64 [| (Nx.shape x).(0) |] Float.neg_infinity
          in
          let z =
            Norn.Nested.run t ~budget:3 ~prior:conjugate.prior ~likelihood
              (Nx.Rng.key 1)
              (conjugate.draw (Nx.Rng.key 2) 20)
          in
          equal stop (Remaining Float.infinity) (Norn.Evidence.stop z));
      test "a run is init, then steps keyed by their count, then the evidence"
        keyed;
      (* Every point ties, and the volumes telescope: ln Z sums [e^c] times
         volumes that add to 1, within the rounding of a few hundred terms, in
         every simulated sequence alike. *)
      cases "a constant likelihood's ln Z is that constant, with no error"
        ~id:(Printf.sprintf "%g") [ 0.; -1000.; 7.5 ] (fun c ->
          let z = nest (constant c) 20 (Nx.Rng.key 4) in
          equal stop ~msg:"stop" Converged (Norn.Evidence.stop z);
          equal (float_rel ~rel:1e-12 ~abs:1e-12) ~msg:"ln Z" c (lnz z);
          equal (float 1e-12) ~msg:"its error" 0. (err z);
          equal (float 1e-9) ~msg:"H" 0.
            (Nx.item [] (Norn.Evidence.information z)));
      (* The points of likelihood 1 tie, and their ranks keep the live set
         uniform among them. A calibrated error misses by three with probability
         0.27%. *)
      test "a tied likelihood's ln Z is within three errors of the truth"
        (fun () ->
          let z = nest indicator 50 (Nx.Rng.key 4) in
          equal stop ~msg:"stop" Converged (Norn.Evidence.stop z);
          at_most float_exact
            ~than:(3. *. err z)
            ~msg:
              (Printf.sprintf "ln Z = %g, the truth %g" (lnz z) indicator.log_z)
            (Float.abs (lnz z -. indicator.log_z)));
      (* Nested sampling runs 13 live points: its run of likelihoods is the
         slowest cell of the recovery law, and its time is about linear in the
         points. Its largest error has a median of 2.3 standard errors over keys
         1 to 40 and reaches 4.1, against the bound of 4.88. It is not held to
         the shrinkage of ln X: one that shrinks it by 1 / (n - j + 2) at the
         j-th deleted point, one past the true 1 / (n - j + 1), fails this test
         on 6 of keys 1 to 40, its largest error of median 3.6. *)
      test "replicates recover the dogs posterior" (fun () ->
          Compiled.here ();
          Recovery.nested ~replicates:16 ~draws:13
            (Recovery.posterior "dogs-dogs"));
    ]

let () =
  exit
    (run "Norn.Nested"
       [ ptree_tests; init_tests; step_tests; evidence_tests; run_tests ])
