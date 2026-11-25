(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models

let t = Nx.Ptree.tensor
let particles = 1000
let budget = 100

let stop =
  Testable.structural ~pp:(fun ppf -> function
    | Norn.Evidence.Converged -> Format.pp_print_string ppf "Converged"
    | Remaining r -> Format.fprintf ppf "Remaining %.17g" r
    | Temperature b -> Format.fprintf ppf "Temperature %.17g" b)

let ln_z z = Nx.item [] (Norn.Evidence.log_evidence z)
let error z = Nx.item [] (Norn.Evidence.error z)
let information z = Nx.item [] (Norn.Evidence.information z)

(* [start m n] is the state at β = 0 of [n] prior draws of [m]. *)
let start ?move ?resampled m n =
  Norn.Smc.init t ?move ?resampled ~prior:m.prior ~likelihood:m.likelihood
    (m.draw (Nx.Rng.key 2) n)

let step m = Norn.Smc.step t ~prior:m.prior ~likelihood:m.likelihood

(* [temper m] is a run on [m] from 1000 prior draws. *)
let temper ?move m =
  let k = Nx.Rng.key 80 in
  Norn.Smc.run t ?move ~budget ~prior:m.prior ~likelihood:m.likelihood
    (Nx.Rng.fold_in k 0)
    (m.draw (Nx.Rng.fold_in k 1) particles)

(* [bad f v x] is [f x] with [v] where the position is above 0. *)
let bad f v x =
  let l = f x in
  Nx.where (Nx.greater_s (Nx.reshape [| -1 |] x) 0.) (Nx.full_like l v) l

(* The conjugate model's prior and likelihood, one of them NaN or +inf above
   0. *)
let bad_densities =
  [
    ("likelihood NaN", conjugate.prior, bad conjugate.likelihood Float.nan);
    ("likelihood +inf", conjugate.prior, bad conjugate.likelihood Float.infinity);
    ("prior NaN", bad conjugate.prior Float.nan, conjugate.likelihood);
  ]

(* ptree *)

let ptree_tests =
  group "ptree"
    [
      test "a state crosses a compiled function with its resampled and move"
        (fun () ->
          Compiled.here ();
          let s = start ~move:Slice ~resampled:4 conjugate 20 in
          let u = Norn.Smc.ptree t in
          let s' = Rune.jit Nx.Ptree.(u @-> returns u) Fun.id s in
          equal int 4 s'.resampled;
          match s'.move with
          | Slice -> ()
          | Hmc -> fail "the state came back with Hamiltonian moves");
    ]

(* init *)

let init_tests =
  group "init"
    [
      cases "the default chains are the largest divisor of N not above sqrt N"
        ~id:(fun (n, m) -> Printf.sprintf "%d particles, %d chains" n m)
        [ (1000, 25); (12, 3); (16, 4); (13, 1); (2, 1) ]
        (fun (n, m) -> equal int m (start conjugate n).resampled);
      test "the default move is Hamiltonian" (fun () ->
          match (start conjugate 12).move with
          | Hmc -> ()
          | Slice -> fail "the default move is Slice");
      test "the step size and the length start at 1" (fun () ->
          let s = start conjugate 12 in
          equal float_exact 1. (Nx.item [] s.step_size);
          equal float_exact 1. (Nx.item [] s.length));
      test "chains of two states are allowed" (fun () ->
          equal int 50 (start ~resampled:50 conjugate 100).resampled);
      cases
        "a resampled that leaves a chain of fewer than two states is refused"
        ~id:(fun (n, m) ->
          match m with
          | Some m -> Printf.sprintf "%d particles, resampled %d" n m
          | None -> Printf.sprintf "%d particle, the default" n)
        [ (100, Some 30); (100, Some 0); (100, Some 100); (1, None) ]
        (fun (n, resampled) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Smc.init") (fun () ->
              start ?resampled conjugate n));
      cases
        "a density is refused that does not return one log density per particle"
        ~id:fst
        [
          ( "one value for all particles",
            fun x -> Nx.sum (conjugate.likelihood x) );
          ( "centred on the particles' mean",
            fun x -> conjugate.likelihood (Nx.sub x (Nx.mean ~axes:[ 0 ] x)) );
          ( "normalised by the particles' largest",
            fun x ->
              let l = conjugate.likelihood x in
              Nx.sub l (Nx.max l) );
          ( "centred on the particles' median",
            fun x ->
              conjugate.likelihood (Nx.sub x (Nx.quantile ~axis:0 [| 0.5 |] x))
          );
        ]
        (fun (_, likelihood) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Smc.init") (fun () ->
              Norn.Smc.init t ~prior:conjugate.prior ~likelihood
                (conjugate.draw (Nx.Rng.key 2) 100)));
      cases "a density NaN or +inf at a particle is refused"
        ~id:(fun (name, _, _) -> name)
        bad_densities
        (fun (_, prior, likelihood) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Smc.init") (fun () ->
              Norn.Smc.init t ~prior ~likelihood
                (conjugate.draw (Nx.Rng.key 2) 100)));
      test "a start that holds no tensor is refused" (fun () ->
          let none () = Nx.zeros Nx.float64 [| 1 |] in
          raises_match (Exn.invalid_arg ~sub:"Norn.Smc.init") (fun () ->
              Norn.Smc.init Nx.Ptree.unit ~prior:none ~likelihood:none ()));
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
            (Nx.ones Nx.float32 [| 12; 1 |], Nx.ones Nx.float64 [| 12; 2 |])
          in
          raises_match (Exn.invalid_arg ~sub:"Norn.Smc.init") (fun () ->
              Norn.Smc.init u ~prior:density ~likelihood:density x));
    ]

(* step *)

(* The next temperature from [s] keeps half the effective size of the smallest
   step, Kish's [(Σ w)² / Σ w²] of [w = exp (db L)] over the particles whose
   likelihood is not 0: half their count. [β'] is found to a relative 1e-3: the
   solve's estimate is an end of a bracket around the root of half-width at most
   1e-3 [β'] (jera.mli, Root.bracket), so the size passes half within 2e-3 [β']
   of it. *)
let keeps_half s s' =
  let l =
    List.filter Float.is_finite
      (Array.to_list (Nx.to_array s.Norn.Smc.likelihood))
  in
  let beta = Nx.item [] s.beta and beta' = Nx.item [] s'.Norn.Smc.beta in
  let ess b =
    let w = List.map (fun l -> Float.exp ((b -. beta) *. l)) l in
    let sum = List.fold_left ( +. ) 0. w in
    sum *. sum /. List.fold_left (fun a w -> a +. (w *. w)) 0. w
  in
  let half = float_of_int (List.length l) /. 2. in
  if beta' >= 1. then at_least float_exact ~than:half (ess 1.)
  else begin
    at_least float_exact ~than:half (ess (beta' *. (1. -. 2e-3)));
    at_most float_exact ~than:half (ess (beta' *. (1. +. 2e-3)))
  end

(* [counted rows f] is [f] recording in [rows] the leading axis of every
   position it reads. *)
let counted rows f x =
  rows := (Nx.shape x).(0) :: !rows;
  f x

let step_tests =
  group "step"
    [
      cases "the next temperature keeps half the smallest step's effective size"
        ~id:(fun m -> m.name)
        [ conjugate; gaussian; truncated ]
        (fun m ->
          let s0 = start m particles in
          let s1 = step m (Nx.Rng.key 3) s0 in
          keeps_half s0 s1;
          keeps_half s1 (step m (Nx.Rng.key 4) s1));
      cases "the densities are evaluated on the M chains of a move"
        ~id:(function Norn.Smc.Hmc -> "Hmc" | Slice -> "Slice")
        [ Norn.Smc.Hmc; Slice ]
        (fun move ->
          let s = start ~move conjugate 100 in
          let rows = ref [] in
          ignore
            (Norn.Smc.step t
               ~prior:(counted rows conjugate.prior)
               ~likelihood:(counted rows conjugate.likelihood)
               (Nx.Rng.key 3) s);
          equal (list int) [ s.resampled ] (List.sort_uniq compare !rows));
      test "a step draws from its key alone" (fun () ->
          let s = start conjugate 100 in
          let particles k =
            Nx.to_array (step conjugate (Nx.Rng.key k) s).particles
          in
          equal (array float_exact) ~msg:"one key" (particles 3) (particles 3);
          not_equal (array float_exact) ~msg:"two keys" (particles 3)
            (particles 4));
      (* [log ε] gains the mean acceptance less 0.8, from ε = 1: within [-0.8,
         0.2]. *)
      test "a Hamiltonian step moves log ε by the mean acceptance less 0.8"
        (fun () ->
          let s = step conjugate (Nx.Rng.key 3) (start conjugate 100) in
          let log_eps = Float.log (Nx.item [] s.step_size) in
          at_least float_exact ~than:(-0.8) log_eps;
          at_most float_exact ~than:0.2 log_eps);
      (* On a flat density a trajectory keeps its energy, so every move is
         accepted: [log ε] gains [1 - 0.8], within an ulp of 0.2. *)
      test "at an acceptance of 1, log ε gains 0.2" (fun () ->
          let flat x = Nx.zeros Nx.float64 [| (Nx.shape x).(0) |] in
          let s =
            Norn.Smc.step t ~prior:flat ~likelihood:flat (Nx.Rng.key 3)
              (Norn.Smc.init t ~prior:flat ~likelihood:flat
                 (conjugate.draw (Nx.Rng.key 2) 100))
          in
          equal
            (float_rel ~rel:1e-15 ~abs:0.)
            (1. -. 0.8)
            (Float.log (Nx.item [] s.step_size)));
      test "a slice step tunes nothing" (fun () ->
          let s =
            step conjugate (Nx.Rng.key 3) (start ~move:Slice conjugate 100)
          in
          equal float_exact 1. (Nx.item [] s.step_size);
          equal float_exact 1. (Nx.item [] s.length));
      test "a step at β = 1 leaves the evidence" (fun () ->
          let rec warm s i =
            if Nx.item [] s.Norn.Smc.beta >= 1. then s
            else warm (step conjugate (Nx.Rng.key i) s) (i + 1)
          in
          let s = warm (start ~resampled:10 conjugate 100) 0 in
          let s' = step conjugate (Nx.Rng.key 99) s in
          equal float_exact ~msg:"ln Z"
            (Nx.item [] s.log_evidence)
            (Nx.item [] s'.log_evidence);
          equal float_exact ~msg:"its variance" (Nx.item [] s.variance)
            (Nx.item [] s'.variance));
      (* Particles drawn at or below 0, under a density bad above 0: the first
         step's moves reach past 0. *)
      cases "a density NaN or +inf at a particle is refused"
        ~id:(fun (name, _, _) -> name)
        bad_densities
        (fun (_, prior, likelihood) ->
          let s =
            Norn.Smc.init t ~prior ~likelihood
              (Nx.neg (Nx.abs (conjugate.draw (Nx.Rng.key 2) 100)))
          in
          raises_match (Exn.invalid_arg ~sub:"Norn.Smc.step") (fun () ->
              Norn.Smc.step t ~prior ~likelihood (Nx.Rng.key 3) s));
    ]

(* evidence *)

(* H is the divergence of the tempered posterior [L^β π / Z_β] from the prior,
   [β E_β (ln L) - ln Z_β], with the particles as the posterior's draws; at [β =
   0], [L^0 = 1] even where [L = 0]. A sum of 1000 terms rounds within 1000 ulps
   of its largest, 1e-12 relatively. *)
let tempered_information m s =
  let z = Norn.Smc.evidence s in
  let beta = match Norn.Evidence.stop z with Temperature b -> b | _ -> 1. in
  let l = m.likelihood (Norn.Evidence.sample z).values in
  let expected =
    if beta = 0. then 0. else (beta *. Nx.item [] (Nx.mean l)) -. ln_z z
  in
  equal (float_rel ~rel:1e-12 ~abs:1e-12) expected (information z)

(* The particles are [M] chains of equal mass, so [effective_groups] is [M], and
   x's error is the variance between chains, [M / (M - 1) Σ_g (m_g - m)² / M²]
   for chain means [m_g], its root widened by [t / 1.96], [t] the 0.975 quantile
   of Student's t with [M - 1] degrees of freedom: 12.706205 at one, 2.063899 at
   24, to their seven digits. *)
let between_chains (m, t975) =
  let k = Nx.Rng.key 81 in
  let z =
    Norn.Smc.run t ~resampled:m ~budget ~prior:conjugate.prior
      ~likelihood:conjugate.likelihood (Nx.Rng.fold_in k 0)
      (conjugate.draw (Nx.Rng.fold_in k 1) particles)
  in
  equal (float 1e-9) ~msg:"effective groups" (float_of_int m)
    (Nx.item [] (Norn.Evidence.effective_groups z));
  let x = Nx.to_array (Nx.reshape [| -1 |] (Norn.Evidence.sample z).values) in
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
    (Nx.item [ 0 ] err)

let evidence_tests =
  group "evidence"
    [
      test "short of β = 1 it stops at the temperature" (fun () ->
          equal stop (Temperature 0.)
            (Norn.Evidence.stop (Norn.Smc.evidence (start conjugate 12))));
      test "information at β = 0 is 0, the prior's divergence from itself"
        (fun () ->
          tempered_information conjugate (start conjugate 100);
          tempered_information truncated (start truncated 100));
      test "information short of β = 1 is the tempered posterior's" (fun () ->
          let s = step gaussian (Nx.Rng.key 1) (start gaussian particles) in
          less float_exact ~than:1. (Nx.item [] s.beta);
          tempered_information gaussian s);
      test "information at β = 1 is the posterior's" (fun () ->
          let s = step conjugate (Nx.Rng.key 1) (start conjugate 100) in
          let s = step conjugate (Nx.Rng.key 2) s in
          equal float_exact 1. (Nx.item [] s.beta);
          tempered_information conjugate s);
      test "one chain is one group, and an expectation's error is inf"
        (fun () ->
          let z = Norn.Smc.evidence (start ~resampled:1 conjugate 12) in
          equal float_exact 1. (Nx.item [] (Norn.Evidence.effective_groups z));
          let _, err = Norn.Evidence.expectation t Fun.id z in
          equal float_exact Float.infinity (Nx.item [ 0 ] err));
      test "an expectation has f's shape, and f runs once on one draw"
        (fun () ->
          let z = Norn.Smc.evidence (start conjugate 12) in
          let seen = ref [] in
          let f x =
            seen := Nx.shape x :: !seen;
            Nx.broadcast_to [| 2; 3 |] (Nx.reshape [| 1; 1 |] x)
          in
          let e, err = Norn.Evidence.expectation t f z in
          equal (list (array int)) [ [| 1 |] ] !seen;
          equal (array int) [| 2; 3 |] (Nx.shape e);
          equal (array int) [| 2; 3 |] (Nx.shape err));
      cases "an expectation's error is the variance between chains, widened"
        ~id:(fun (m, _) -> Printf.sprintf "%d chains" m)
        [ (2, 12.706205); (25, 2.063899) ]
        between_chains;
    ]

(* run *)

(* Two temperatures of the ten-dimensional Gaussian, short of β = 1. *)
let spent () =
  Norn.Smc.run t ~budget:2 ~prior:gaussian.prior ~likelihood:gaussian.likelihood
    (Nx.Rng.key 1)
    (gaussian.draw (Nx.Rng.key 2) particles)

(* [keyed_run m k x] runs as [run] is documented to: [init], then step [s] keyed
   [fold_in k s] until [β = 1], then the evidence. *)
let keyed_run m k x =
  let rec go s =
    if Nx.item [] s.Norn.Smc.beta >= 1. then s
    else go (step m (Nx.Rng.fold_in k (Int32.to_int (Nx.item [] s.steps))) s)
  in
  Norn.Smc.evidence
    (go (Norn.Smc.init t ~prior:m.prior ~likelihood:m.likelihood x))

(* The ln Z laws read [runs] runs on a model, compiled as one function mapped
   over their keys: eager, the ten-dimensional Gaussian's take ten times as
   long. *)
let runs = 20

let runs_of m =
  fixture (fun () ->
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
          Nx.Ptree.map ez (fun _ x -> Nx.slice [ Nx.I i ] x) all))

let mean xs = List.fold_left ( +. ) 0. xs /. float_of_int (List.length xs)

let spread xs =
  let m = mean xs in
  Float.sqrt
    (List.fold_left (fun s v -> s +. ((v -. m) ** 2.)) 0. xs
    /. float_of_int (List.length xs - 1))

(* The mean of 20 runs' ln Z lies within three of its standard errors, their
   spread over sqrt 20, of the truth: beyond lies Student's t with 19 degrees of
   freedom, with probability 0.74%. *)
let mean_near (m, runs) =
  test m.name (fun () ->
      Compiled.here ();
      let values = List.map ln_z (runs ()) in
      at_most float_exact
        ~than:
          (3. *. spread values /. Float.sqrt (float_of_int (List.length values)))
        ~msg:(Printf.sprintf "mean ln Z %g, the truth %g" (mean values) m.log_z)
        (Float.abs (mean values -. m.log_z)))

(* The spread of 20 runs' ln Z lies within a factor of two of their mean error:
   a calibrated error falls outside with probability 3.9e-4, from χ² with 19
   degrees of freedom. *)
let spread_is_error (m, runs) =
  test m.name (fun () ->
      Compiled.here ();
      let zs = runs () in
      let ratio = spread (List.map ln_z zs) /. mean (List.map error zs) in
      let msg = "the spread of ln Z over the mean error" in
      greater float_exact ~than:0.5 ~msg ratio;
      less float_exact ~than:2. ~msg ratio)

let law_runs =
  List.map (fun m -> (m, runs_of m)) [ conjugate; gaussian; mixture; truncated ]

let run_tests =
  group "run"
    [
      test "a budget below one is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Smc.run") (fun () ->
              Norn.Smc.run t ~budget:0 ~prior:conjugate.prior
                ~likelihood:conjugate.likelihood (Nx.Rng.key 1)
                (conjugate.draw (Nx.Rng.key 2) 100)));
      test "a run is init, then steps keyed by their count" (fun () ->
          let k = Nx.Rng.key 3 and x = conjugate.draw (Nx.Rng.key 2) 100 in
          let z =
            Norn.Smc.run t ~budget ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood k x
          in
          let z' = keyed_run conjugate k x in
          equal float_exact ~msg:"ln Z" (ln_z z') (ln_z z);
          equal float_exact ~msg:"its error" (error z') (error z);
          equal (array float_exact) ~msg:"the sample"
            (Nx.to_array (Norn.Evidence.sample z').values)
            (Nx.to_array (Norn.Evidence.sample z).values));
      test "a run within its budget converges" (fun () ->
          equal stop Converged (Norn.Evidence.stop (temper conjugate)));
      test "a spent budget stops at the temperature reached" (fun () ->
          match Norn.Evidence.stop (spent ()) with
          | Temperature b ->
              greater float_exact ~than:0. b;
              less float_exact ~than:1. b
          | s -> failf "stopped as %a" (Testable.pp stop) s);
      test "pp states the temperature a spent budget reached" (fun () ->
          let z = spent () in
          match Norn.Evidence.stop z with
          | Temperature b ->
              equal string
                (Printf.sprintf
                   "ln Z = %.2f ± %.2f, H = %.2f nats, budget spent at β = %.2f"
                   (ln_z z) (error z) (information z) b)
                (Format.asprintf "%a" Norn.Evidence.pp z)
          | s -> failf "stopped as %a" (Testable.pp stop) s);
      test "pp of a converged run states ln Z, its error and H" (fun () ->
          let z = temper conjugate in
          equal string
            (Printf.sprintf "ln Z = %.2f ± %.2f, H = %.2f nats" (ln_z z)
               (error z) (information z))
            (Format.asprintf "%a" Norn.Evidence.pp z));
      (* Every step's weights are equal, [e^(db c)], so ln Z sums the steps' [db
         c] to [c], within the rounding of means of 1000 equal terms. *)
      cases "a constant likelihood's ln Z is that constant, with no error"
        ~id:(fun c -> Printf.sprintf "%g" c)
        [ 0.; -1000.; 7.5 ]
        (fun c ->
          let z = temper (constant c) in
          equal stop ~msg:"stop" Converged (Norn.Evidence.stop z);
          equal (float_rel ~rel:1e-12 ~abs:1e-12) ~msg:"ln Z" c (ln_z z);
          equal (float 1e-12) ~msg:"its error" 0. (error z));
      (* The likelihood runs as a callback, which has no derivative: a
         differentiation through it raises (rune.mli, Callbacks). Hamiltonian
         moves differentiate it, slice moves do not. *)
      test "slice moves run a density that cannot be differentiated" (fun () ->
          let likelihood x =
            Rune.callback
              Nx.Ptree.(tensor @-> returns tensor)
              ~like:(Nx.zeros Nx.float64 [| (Nx.shape x).(0) |])
              conjugate.likelihood x
          in
          let run move =
            Norn.Smc.run t ~move ~budget ~prior:conjugate.prior ~likelihood
              (Nx.Rng.key 1)
              (conjugate.draw (Nx.Rng.key 2) 100)
          in
          raises_match (Exn.invalid_arg ~sub:"Rune.callback") (fun () ->
              run Hmc);
          equal stop Converged (Norn.Evidence.stop (run Slice)));
      (* A calibrated error misses by three with probability 0.27%. *)
      test "slice moves recover the evidence within three errors" (fun () ->
          let z = temper ~move:Slice conjugate in
          at_most float_exact
            ~than:(3. *. error z)
            ~msg:
              (Printf.sprintf "ln Z = %g, the truth %g" (ln_z z) conjugate.log_z)
            (Float.abs (ln_z z -. conjugate.log_z)));
      (* Under the conjugate model x's posterior is N(Σ y / 6, 1 / 6), y's sum
         5.7, so E x = 0.95 and E x² = 1 / 6 + 0.95². An expectation's error is
         its standard deviation between the 25 chains times t / 1.96, t =
         2.063899 at 24 degrees of freedom: three errors are 3.16 standard
         deviations, beyond which Student's t with 24 degrees of freedom lies
         with probability 0.0042, so the two moments fail falsely with
         probability below 0.9%. *)
      test "an expectation is within three errors of the posterior's" (fun () ->
          let z = temper conjugate in
          let f x = Nx.concatenate ~axis:0 [ x; Nx.square x ] in
          let e, err = Norn.Evidence.expectation t f z in
          List.iteri
            (fun i (name, truth) ->
              at_most float_exact
                ~than:(3. *. Nx.item [ i ] err)
                ~msg:name
                (Float.abs (Nx.item [ i ] e -. truth)))
            [ ("E x", 0.95); ("E x²", (1. /. 6.) +. (0.95 *. 0.95)) ]);
      group "ln Z's mean over 20 runs is within three standard errors of"
        (List.map mean_near law_runs);
      group "ln Z's spread over 20 runs is its error, on"
        (List.map spread_is_error law_runs);
      test "replicates recover the dogs posterior" (fun () ->
          Compiled.here ();
          Recovery.smc ~replicates:16 ~draws:100
            (Recovery.posterior "dogs-dogs"));
    ]

let () =
  exit
    (run "Norn.Smc"
       [ ptree_tests; init_tests; step_tests; evidence_tests; run_tests ])
