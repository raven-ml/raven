(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models

let t = Nx.Ptree.tensor

(* ln Z and its error. On models whose evidence is known, a run's [ln Z] is
   within three of its errors of the truth, and over [runs] runs the spread of
   [ln Z] is within a factor of two of the mean error. Over 20 runs a calibrated
   error leaves the spread outside that factor with probability about 4e-4, from
   χ² with 19 degrees of freedom; the fixed keys make each outcome the same on
   every run. *)

let particles = 1000
let runs = 20
let budget = 100

let temper ?move m i =
  let k = Nx.Rng.fold_in (Nx.Rng.key 80) i in
  Norn.Smc.run t ?move ~budget ~prior:m.prior ~likelihood:m.likelihood
    (Nx.Rng.fold_in k 0)
    (m.draw (Nx.Rng.fold_in k 1) particles)

(* [runs_of m] is the ln Z law's runs on [m], compiled as one function mapped
   over their keys: eager, the ten-dimensional Gaussian's take ten times as
   long. *)
let runs_of m =
  let ez = Norn.Evidence.ptree t in
  let one k =
    Norn.Smc.run t ~budget ~prior:m.prior ~likelihood:m.likelihood
      (Nx.Rng.fold_in k 0)
      (m.draw (Nx.Rng.fold_in k 1) particles)
  in
  let all =
    Rune.jit
      Nx.Ptree.(Nx.Rng.ptree @-> returns ez)
      (Rune.vmap Nx.Ptree.(Nx.Rng.ptree @-> returns ez) one)
      (Nx.Rng.split_batch ~n:runs (Nx.Rng.key 80))
  in
  List.init runs (fun i ->
      Nx.Ptree.map ez (fun _ x -> Nx.slice [ Nx.I i ] x) all)

let within_errors m =
  test m.name (fun () ->
      let zs = runs_of m in
      let ln z = Nx.item [] (Norn.Evidence.log_evidence z) in
      let err z = Nx.item [] (Norn.Evidence.error z) in
      let z0 = List.hd zs in
      at_most (float 1e-12)
        ~than:(3. *. err z0)
        ~msg:(Printf.sprintf "ln Z = %g, the truth %g" (ln z0) m.log_z)
        (Float.abs (ln z0 -. m.log_z));
      let n = float_of_int runs in
      let values = List.map ln zs in
      let mean = List.fold_left ( +. ) 0. values /. n in
      let spread =
        Float.sqrt
          (List.fold_left (fun s v -> s +. ((v -. mean) ** 2.)) 0. values
          /. (n -. 1.))
      in
      let error = List.fold_left (fun s z -> s +. err z) 0. zs /. n in
      satisfies ~claim:"the spread within a factor of two of the error"
        (float 1e-12)
        (fun r -> r > 0.5 && r < 2.)
        (spread /. error))

(* The default chains: the largest divisor of N not above sqrt N. *)
let chains =
  cases "the default chains"
    ~id:(fun (n, m) -> Printf.sprintf "%d particles, %d chains" n m)
    [ (1000, 25); (12, 3); (16, 4); (13, 1); (2, 1) ]
    (fun (n, m) ->
      let s =
        Norn.Smc.init t ~prior:conjugate.prior ~likelihood:conjugate.likelihood
          (conjugate.draw (Nx.Rng.key 2) n)
      in
      equal int m s.resampled)

(* init *)

let init_tests =
  group "init"
    [
      chains;
      test "the default move is Hamiltonian" (fun () ->
          let s =
            Norn.Smc.init t ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood
              (conjugate.draw (Nx.Rng.key 2) 12)
          in
          match s.move with
          | Norn.Smc.Hmc -> ()
          | Norn.Smc.Slice -> fail "the default move is Slice");
      test "resampled must divide the particles into chains" (fun () ->
          raises
            (Invalid_argument
               "Norn.Smc.init: resampled = 30 does not divide 100 particles \
                into chains of two or more") (fun () ->
              Norn.Smc.init t ~resampled:30 ~prior:conjugate.prior
                ~likelihood:conjugate.likelihood
                (conjugate.draw (Nx.Rng.key 2) 100)));
      test "a density of one value for all particles is refused" (fun () ->
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Norn.Smc.init: the density returned shape [] for a position \
                  of 100 chains") (fun () ->
              Norn.Smc.init t
                ~prior:(fun x -> Nx.sum (conjugate.prior x))
                ~likelihood:conjugate.likelihood
                (conjugate.draw (Nx.Rng.key 2) 100)));
    ]

(* step *)

let step_tests =
  group "step"
    [
      test "a step at β = 1 leaves the evidence" (fun () ->
          let s =
            Norn.Smc.init t ~resampled:10 ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood
              (conjugate.draw (Nx.Rng.key 2) 100)
          in
          let step =
            Norn.Smc.step t ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood
          in
          let rec warm s i =
            if Nx.item [] s.Norn.Smc.beta >= 1. then s
            else warm (step (Nx.Rng.key i) s) (i + 1)
          in
          let s = warm s 0 in
          let s' = step (Nx.Rng.key 99) s in
          equal float_exact
            (Nx.item [] s.log_evidence)
            (Nx.item [] s'.log_evidence));
      test "a NaN likelihood at a particle is refused" (fun () ->
          let likelihood x =
            let l = conjugate.likelihood x in
            Nx.where
              (Nx.greater_s (Nx.reshape [| -1 |] x) 0.)
              (Nx.full_like l Float.nan) l
          in
          let start = Nx.neg (Nx.abs (conjugate.draw (Nx.Rng.key 2) 100)) in
          let s = Norn.Smc.init t ~prior:conjugate.prior ~likelihood start in
          raises_match
            (Exn.invalid_arg ~sub:"Norn.Smc.step: the density is nan")
            (fun () ->
              Norn.Smc.step t ~prior:conjugate.prior ~likelihood (Nx.Rng.key 3)
                s));
      test "a compiled step is the eager one" (fun () ->
          let s =
            Norn.Smc.init t ~prior:gaussian.prior
              ~likelihood:gaussian.likelihood
              (gaussian.draw (Nx.Rng.key 2) 200)
          in
          let sp = Norn.Smc.ptree t in
          let step =
            Norn.Smc.step t ~prior:gaussian.prior
              ~likelihood:gaussian.likelihood
          in
          let compiled =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              step (Nx.Rng.key 3) s
          in
          let eager = step (Nx.Rng.key 3) s in
          equal (float 1e-12) (Nx.item [] eager.beta) (Nx.item [] compiled.beta);
          equal (float 1e-9)
            (Nx.item [] eager.log_evidence)
            (Nx.item [] compiled.log_evidence);
          equal
            (array (float 1e-9))
            (Nx.to_array eager.particles)
            (Nx.to_array compiled.particles));
    ]

(* evidence *)

(* Expectations: the particles are [M] chains of equal mass, so
   [effective_groups] is [M], and x's error is the variance between chains, [M /
   (M - 1) Σ_g (m_g - m)² / M²] for chain means [m_g], its root widened by [t /
   1.96], [t] the 0.975 quantile of Student's t with [M - 1] degrees of freedom:
   12.706205 at one, 2.063899 at 24, to their seven digits. *)
let between_chains =
  cases "an expectation's error is the variance between chains, widened"
    ~id:(fun (m, _) -> Printf.sprintf "%d chains" m)
    [ (2, 12.706205); (25, 2.063899) ]
    (fun (m, t975) ->
      let k = Nx.Rng.key 81 in
      let z =
        Norn.Smc.run t ~resampled:m ~budget ~prior:conjugate.prior
          ~likelihood:conjugate.likelihood (Nx.Rng.fold_in k 0)
          (conjugate.draw (Nx.Rng.fold_in k 1) particles)
      in
      equal (float 1e-9) ~msg:"effective groups" (float_of_int m)
        (Nx.item [] (Norn.Evidence.effective_groups z));
      let x =
        Nx.to_array (Nx.reshape [| -1 |] (Norn.Evidence.sample z).values)
      in
      let n = Array.length x in
      let mean = Array.fold_left ( +. ) 0. x /. float_of_int n in
      let between = ref 0. in
      for g = 0 to m - 1 do
        let s = ref 0. and c = ref 0 in
        Array.iteri
          (fun i v ->
            if i mod m = g then begin
              s := !s +. v;
              incr c
            end)
          x;
        let d = (!s /. float_of_int !c) -. mean in
        between := !between +. (d *. d)
      done;
      let mf = float_of_int m in
      let sd = Float.sqrt (mf /. (mf -. 1.) *. !between /. (mf *. mf)) in
      let e, err = Norn.Evidence.expectation t Fun.id z in
      equal (float 1e-9) ~msg:"the mean" mean (Nx.item [ 0 ] e);
      equal
        (float (2e-6 *. sd *. t975))
        ~msg:"the error"
        (sd *. t975 /. 1.959963984540054)
        (Nx.item [ 0 ] err))

let evidence_tests =
  group "evidence"
    [
      test "short of β = 1 it stops at the temperature" (fun () ->
          let s =
            Norn.Smc.init t ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood
              (conjugate.draw (Nx.Rng.key 2) 12)
          in
          match Norn.Evidence.stop (Norn.Smc.evidence s) with
          | Temperature b -> equal float_exact 0. b
          | Converged -> fail "converged at β = 0"
          | Remaining _ -> fail "stopped as nested sampling does");
      between_chains;
    ]

(* run *)

(* Two temperatures of the ten-dimensional Gaussian, short of beta = 1. *)
let spent () =
  Norn.Smc.run t ~budget:2 ~prior:gaussian.prior ~likelihood:gaussian.likelihood
    (Nx.Rng.key 1)
    (gaussian.draw (Nx.Rng.key 2) particles)

(* Slice moves need no derivative. *)
let slice =
  test "slice moves recover the evidence within three errors" (fun () ->
      let z = temper ~move:Slice conjugate 0 in
      let ln = Nx.item [] (Norn.Evidence.log_evidence z) in
      at_most (float 1e-12)
        ~than:(3. *. Nx.item [] (Norn.Evidence.error z))
        ~msg:(Printf.sprintf "ln Z = %g, the truth %g" ln conjugate.log_z)
        (Float.abs (ln -. conjugate.log_z)))

(* Under the conjugate model x's posterior is N(Σ y / 6, 1 / 6), y's sum 5.7, so
   E x = 0.95 and E x² = 1 / 6 + 0.95². An expectation's error is its standard
   deviation between the 25 chains times t / 1.96, t = 2.063899 at 24 degrees of
   freedom: three errors are 3.16 standard deviations, beyond which Student's t
   with 24 degrees of freedom lies with probability 0.0042, so the two moments
   fail falsely with probability below 0.9%. *)
let moments =
  test "an expectation is within three errors of the posterior's" (fun () ->
      let z = temper conjugate 0 in
      let f x = Nx.concatenate ~axis:0 [ x; Nx.square x ] in
      let e, err = Norn.Evidence.expectation t f z in
      List.iteri
        (fun i (name, truth) ->
          at_most (float 1e-12)
            ~than:(3. *. Nx.item [ i ] err)
            ~msg:name
            (Float.abs (Nx.item [ i ] e -. truth)))
        [ ("E x", 0.95); ("E x²", (1. /. 6.) +. (0.95 *. 0.95)) ])

(* [keyed_run m k start] runs as [run] is documented to: [init], then step [s]
   keyed [fold_in k s] until [β = 1], then the evidence. *)
let keyed_run m k start =
  let step = Norn.Smc.step t ~prior:m.prior ~likelihood:m.likelihood in
  let rec go s =
    if Nx.item [] s.Norn.Smc.beta >= 1. then s
    else go (step (Nx.Rng.fold_in k (Int32.to_int (Nx.item [] s.steps))) s)
  in
  Norn.Smc.evidence
    (go (Norn.Smc.init t ~prior:m.prior ~likelihood:m.likelihood start))

let run_tests =
  group "run"
    [
      test "a spent budget stops at the temperature reached" (fun () ->
          match Norn.Evidence.stop (spent ()) with
          | Temperature b ->
              greater float_exact ~than:0. b;
              less float_exact ~than:1. b
          | Converged -> fail "converged in two temperatures"
          | Remaining _ -> fail "stopped as nested sampling does");
      test "pp states the temperature a spent budget reached" (fun () ->
          let z = spent () in
          match Norn.Evidence.stop z with
          | Temperature b ->
              let f x = Nx.item [] x in
              equal string
                (Printf.sprintf
                   "ln Z = %.2f ± %.2f, H = %.2f nats, budget spent at β = %.2f"
                   (f (Norn.Evidence.log_evidence z))
                   (f (Norn.Evidence.error z))
                   (f (Norn.Evidence.information z))
                   b)
                (Format.asprintf "%a" Norn.Evidence.pp z)
          | _ -> fail "the run did not stop at a temperature");
      test "a run within its budget converges" (fun () ->
          match Norn.Evidence.stop (temper conjugate 0) with
          | Converged -> ()
          | _ -> fail "the run stopped short of beta = 1");
      test "a budget below one is refused" (fun () ->
          raises (Invalid_argument "Norn.Smc.run: budget = 0 is not positive")
            (fun () ->
              Norn.Smc.run t ~budget:0 ~prior:conjugate.prior
                ~likelihood:conjugate.likelihood (Nx.Rng.key 1)
                (conjugate.draw (Nx.Rng.key 2) 100)));
      test "a run is init, then steps keyed by their count" (fun () ->
          let k = Nx.Rng.key 3 and start = conjugate.draw (Nx.Rng.key 2) 100 in
          let z =
            Norn.Smc.run t ~budget ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood k start
          in
          let z' = keyed_run conjugate k start in
          equal float_exact
            (Nx.item [] (Norn.Evidence.log_evidence z'))
            (Nx.item [] (Norn.Evidence.log_evidence z));
          equal float_exact
            (Nx.item [] (Norn.Evidence.error z'))
            (Nx.item [] (Norn.Evidence.error z));
          equal (array float_exact)
            (Nx.to_array (Norn.Evidence.sample z').values)
            (Nx.to_array (Norn.Evidence.sample z).values));
      slice;
      moments;
      Recovery.recovers "replicates recover the dogs posterior" (fun () ->
          Recovery.replicated_cell
            (Recovery.smc ~replicates:16 ~draws:100)
            (Recovery.posterior "dogs-dogs"));
    ]

let () =
  exit
    (run "Norn.Smc"
       [
         init_tests;
         step_tests;
         evidence_tests;
         run_tests;
         group "ln Z" (List.map within_errors [ conjugate; gaussian; mixture ]);
       ])
