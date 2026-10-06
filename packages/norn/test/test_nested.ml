(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models

let t = Nx.Ptree.tensor

let stop_name = function
  | Norn.Evidence.Converged -> "converged"
  | Remaining r -> Printf.sprintf "Remaining %g" r
  | Temperature b -> Printf.sprintf "Temperature %g" b

(* Law 13: evidence. On models whose evidence is known, a run's [ln Z] is within
   three of its errors of the truth, and over [runs] runs the spread of [ln Z]
   is within a factor of two of the mean error. Over 20 runs the spread of a
   correct error falls in [0.5, 2] with probability 0.995. A run on the Gaussian
   in ten dimensions takes seven seconds, so it is held to the first claim. *)

let live = 200
let budget = 200

let nest m i =
  let k = Nx.Rng.fold_in (Nx.Rng.key 70) i in
  Norn.Nested.run t ~budget ~prior:m.prior ~likelihood:m.likelihood
    (Nx.Rng.fold_in k 0)
    (m.draw (Nx.Rng.fold_in k 1) live)

let law_13 ~runs m =
  slow m.name (fun () ->
      let zs = List.init runs (nest m) in
      let ln z = Nx.item [] (Norn.Evidence.log_evidence z) in
      let err z = Nx.item [] (Norn.Evidence.error z) in
      let z0 = List.hd zs in
      at_most (float 1e-12)
        ~than:(3. *. err z0)
        ~msg:(Printf.sprintf "ln Z = %g, the truth %g" (ln z0) m.log_z)
        (Float.abs (ln z0 -. m.log_z));
      if runs > 1 then begin
        let n = float_of_int runs in
        let values = List.map ln zs in
        let mean = List.fold_left ( +. ) 0. values /. n in
        let spread =
          Float.sqrt
            (List.fold_left (fun s v -> s +. ((v -. mean) ** 2.)) 0. values
            /. (n -. 1.))
        in
        let error = List.fold_left (fun s z -> s +. err z) 0. zs /. n in
        let msg = "the spread of ln Z over the mean error" in
        greater (float 1e-12) ~than:0.5 ~msg (spread /. error);
        less (float 1e-12) ~than:2. ~msg (spread /. error)
      end)

(* The weighted sample is the posterior: under the conjugate model x's posterior
   is N(Σ y / (m + 1), 1 / (m + 1)). *)
let posterior =
  test "the weighted sample has the posterior's moments" (fun () ->
      let w = Norn.Evidence.sample (nest conjugate 0) in
      let p = Nx.exp w.log_weights in
      let x = Nx.reshape [| -1 |] w.values in
      let mean = Nx.item [] (Nx.sum (Nx.mul p x)) in
      let var = Nx.item [] (Nx.sum (Nx.mul p (Nx.square (Nx.sub_s x mean)))) in
      equal (float 0.05) (5.7 /. 6.) mean;
      equal (float 0.03) (1. /. 6.) var)

(* Two steps of the ten-dimensional Gaussian, far from its tolerance. *)
let spent () =
  Norn.Nested.run t ~budget:2 ~prior:gaussian.prior
    ~likelihood:gaussian.likelihood (Nx.Rng.key 1)
    (gaussian.draw (Nx.Rng.key 2) live)

let runs_and_refusals =
  group "runs"
    [
      test "a spent budget stops with the ln Z the live points could add"
        (fun () ->
          match Norn.Evidence.stop (spent ()) with
          | Remaining r -> at_least float_exact ~than:1e-3 r
          | stop -> failf "stopped as %s" (stop_name stop));
      test "pp states the ln Z a spent budget leaves" (fun () ->
          let z = spent () in
          match Norn.Evidence.stop z with
          | Remaining r ->
              let f x = Nx.item [] x in
              equal string
                (Printf.sprintf
                   "ln Z = %.2f ± %.2f, H = %.2f nats, budget spent with %.2f \
                    nats left"
                   (f (Norn.Evidence.log_evidence z))
                   (f (Norn.Evidence.error z))
                   (f (Norn.Evidence.information z))
                   r)
                (Format.asprintf "%a" Norn.Evidence.pp z)
          | stop -> failf "stopped as %s" (stop_name stop));
      test "a run within its budget converges" (fun () ->
          equal string "converged"
            (stop_name (Norn.Evidence.stop (nest conjugate 0))));
      test "a step past the budget leaves the state" (fun () ->
          let s =
            Norn.Nested.init t ~budget:1 ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood
              (conjugate.draw (Nx.Rng.key 2) 20)
          in
          let step =
            Norn.Nested.step t ~prior:conjugate.prior
              ~likelihood:conjugate.likelihood
          in
          let s = step (Nx.Rng.key 3) s in
          let s' = step (Nx.Rng.key 4) s in
          equal int32 1l (Nx.item [] s'.steps);
          equal (array float_exact) (Nx.to_array s.live) (Nx.to_array s'.live);
          equal float_exact
            (Nx.item [] s.log_evidence)
            (Nx.item [] s'.log_evidence));
      test "a batch of every live point is refused" (fun () ->
          raises
            (Invalid_argument "Norn.Nested.init: batch = 20 is not in [1, 20)")
            (fun () ->
              Norn.Nested.init t ~batch:20 ~budget:1 ~prior:conjugate.prior
                ~likelihood:conjugate.likelihood
                (conjugate.draw (Nx.Rng.key 2) 20)));
      test "a run repeated with one key gives the same evidence" (fun () ->
          equal float_exact
            (Nx.item [] (Norn.Evidence.log_evidence (nest conjugate 3)))
            (Nx.item [] (Norn.Evidence.log_evidence (nest conjugate 3))));
      test "a compiled step is the eager one" (fun () ->
          let s =
            Norn.Nested.init t ~budget:2 ~prior:gaussian.prior
              ~likelihood:gaussian.likelihood
              (gaussian.draw (Nx.Rng.key 2) 50)
          in
          let sp = Norn.Nested.ptree t in
          let step =
            Norn.Nested.step t ~prior:gaussian.prior
              ~likelihood:gaussian.likelihood
          in
          let compiled =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> sp @-> returns sp)
              step (Nx.Rng.key 3) s
          in
          let eager = step (Nx.Rng.key 3) s in
          equal (float 1e-9)
            (Nx.item [] eager.log_evidence)
            (Nx.item [] compiled.log_evidence);
          equal
            (array (float 1e-9))
            (Nx.to_array eager.live)
            (Nx.to_array compiled.live));
      posterior;
    ]

let () =
  exit
    (run "Norn.Nested"
       [
         runs_and_refusals;
         group "evidence"
           [
             law_13 ~runs:20 conjugate;
             law_13 ~runs:1 gaussian;
             law_13 ~runs:20 mixture;
           ];
       ])
