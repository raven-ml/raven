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
   correct error falls in [0.5, 2] with probability 0.995, over 200 with
   probability 1 - 1e-20. The Gaussian in ten dimensions is held to the first
   claim.

   Both claims are on multiples of the errors, which grow as the live points
   fall, so their odds of a miss hold at any count whose errors stay
   calibrated. The conjugate model shares Law 14's 200 runs at 50 live points.
   The mixture takes 100 live points, at two thirds of 200's time: over 40
   runs at 100 its [ln Z] lay 0.00 errors from the truth on average and spread
   1.10 of the mean error (-0.03 and 1.03 at 200). The Gaussian keeps 200:
   over 10 runs at 100 its [ln Z] lay 0.32 ± 0.25 errors above the truth (0.12
   at 200), and a shift of 0.32 would raise the first claim's odds of a miss
   from 0.27% to 0.41%. *)

let live = 200

(* Every run here converges within 40 steps; the dead buffer's size sets the
   cost of the errors. *)
let budget = 60
let ez = Norn.Evidence.ptree t

(* [batch_of ?live m n] is [n] runs on [m] from [live] prior draws, the [i]-th
   from row [i] of [Nx.Rng.split_batch ~n (Nx.Rng.key 70)], compiled as one
   function mapped over the keys: eager, a run of the ten-dimensional Gaussian
   takes twenty times as long. [runs_of] is them one by one. *)
let runs = Hashtbl.create 8

let batch_of ?(live = live) m n =
  match Hashtbl.find_opt runs (m.name, live, n) with
  | Some all -> all
  | None ->
      let one k =
        Norn.Nested.run t ~budget ~prior:m.prior ~likelihood:m.likelihood
          (Nx.Rng.fold_in k 0)
          (m.draw (Nx.Rng.fold_in k 1) live)
      in
      let all =
        Rune.jit
          Nx.Ptree.(Nx.Rng.ptree @-> returns ez)
          (Rune.vmap Nx.Ptree.(Nx.Rng.ptree @-> returns ez) one)
          (Nx.Rng.split_batch ~n (Nx.Rng.key 70))
      in
      Hashtbl.replace runs (m.name, live, n) all;
      all

let lanes n all =
  List.init n (fun i -> Nx.Ptree.map ez (fun _ x -> Nx.slice [ Nx.I i ] x) all)

let runs_of ?live m n = lanes n (batch_of ?live m n)

let nest m = List.hd (runs_of m 1)

let law_13 m zs =
  slow
  @@ test m.name (fun () ->
      let zs = zs () in
      let runs = List.length zs in
      List.iter
        (fun z ->
          equal string "converged" (stop_name (Norn.Evidence.stop z)))
        zs;
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
      let w = Norn.Evidence.sample (nest conjugate) in
      let p = Nx.exp w.log_weights in
      let x = Nx.reshape [| -1 |] w.values in
      let mean = Nx.item [] (Nx.sum (Nx.mul p x)) in
      let var = Nx.item [] (Nx.sum (Nx.mul p (Nx.square (Nx.sub_s x mean)))) in
      equal (float 0.05) (5.7 /. 6.) mean;
      equal (float 0.03) (1. /. 6.) var)

(* Law 14: expectations. Of [calibration] = 200 independent runs, the number
   whose interval of [± 1.96] errors holds the truth is [Bin (200, p)], [p]
   the coverage, 0.95 when the errors are right. Each model's count is at
   least 175: at [p = 0.95] a moment falls short with probability 9e-6, at
   the lowest coverage seen, 0.935, with 6e-4, and an error that covers 68%
   passes with probability 1e-10. Over six key sets the conjugate model's
   runs covered 0.945 to 0.975 of each moment, the banana's 0.97 to 0.99.

   The errors widen by Student's t, so a run's error over [t / 1.96], [t] the
   0.975 quantile at [K - 1] degrees of freedom, [K] the run's
   [Norn.Evidence.effective_groups], is the pivot's scale, and the truth lies
   within [t_0.75] of those scales, the quantile at 0.75, in half the runs: [Bin
   (200, 0.5)]. The conjugate model's count lies in [81, 119], outside with
   probability 5.7e-3 per moment. An error 1.4 times too wide, as the one that
   counted the volumes' noise twice, gives counts near 131 and passes with
   probability 0.06 per moment; over six key sets, the new spread times 1.42
   failed in each. The banana's errors are wider than the pivot's at its centre,
   where few lineages make the multiples a mixture: its count lay in 123 to 141,
   so it is held to the first claim only.

   Under the conjugate model x's posterior has mean [5.7 / 6] and second
   moment [1 / 6 + (5.7 / 6)²]. On the banana x1's posterior is the standard
   normal cut to [-4, 4], of mean 0 and second moment
   [1 - 8 φ(4) / P(|x1| <= 4)], and slice moves leave the draws of a lineage
   correlated: an error that takes the draws of different starts as
   independent covered 59% to 68%. *)

let calibration = 200

(* Law 14's runs, from 50 live points. *)
let small_runs m = runs_of ~live:50 m calibration

let cut =
  let inside = 1. -. Float.erfc (4. /. Float.sqrt 2.) in
  let phi4 = Float.exp (-8.) /. Float.sqrt (2. *. Float.pi) in
  1. -. (8. *. phi4 /. inside)

let first_moments x =
  let x1 = Nx.slice [ Nx.I 0 ] x in
  Nx.stack ~axis:0 [ x1; Nx.square x1 ]

(* The normal's 0.975 quantile. *)
let z975 = 1.959963984540054

(* [t_quantile p df] is Student's t's [p] quantile, [p > 1/2], at each of the
   degrees of freedom [df]: its two tails beyond [t] hold
   [I_x (df / 2, 1 / 2)], [x = df / (df + t²)], which bisection on [x] over
   64 halvings solves for [2 (1 - p)]. *)
let t_quantile p df =
  let df = Nx.create Nx.float64 [| Array.length df |] df in
  let a = Nx.mul_s df 0.5 and half = Nx.full_like df 0.5 in
  let tails = Nx.full_like df (2. *. (1. -. p)) in
  let lo = ref (Nx.zeros_like df) and hi = ref (Nx.ones_like df) in
  for _ = 1 to 64 do
    let mid = Nx.mul_s (Nx.add !lo !hi) 0.5 in
    let inside = Nx.less (Nx.betainc a half mid) tails in
    lo := Nx.where inside mid !lo;
    hi := Nx.where inside !hi mid
  done;
  let x = Nx.mul_s (Nx.add !lo !hi) 0.5 in
  Nx.to_array (Nx.sqrt (Nx.div (Nx.mul df (Nx.rsub_s 1. x)) x))

let law_14 ~centre m truth =
  slow
  @@ test m.name (fun () ->
      let zs = small_runs m in
      List.iter
        (fun z ->
          equal string "converged" (stop_name (Norn.Evidence.stop z)))
        zs;
      (* The runs' expectations, compiled over them as their runs are. *)
      let m', e' =
        let pt = Nx.Ptree.(pair tensor tensor) in
        Rune.jit
          Nx.Ptree.(ez @-> returns pt)
          (Rune.vmap
             Nx.Ptree.(ez @-> returns pt)
             (Norn.Evidence.expectation t first_moments))
          (batch_of ~live:50 m calibration)
      in
      let df =
        Array.of_list
          (List.map
             (fun z -> Nx.item [] (Norn.Evidence.effective_groups z) -. 1.)
             zs)
      in
      let t975 = t_quantile 0.975 df and t75 = t_quantile 0.75 df in
      Array.iteri
        (fun j truth ->
          let multiples =
            Array.init calibration (fun i ->
                (Nx.item [ i; j ] m' -. truth) /. Nx.item [ i; j ] e')
          in
          let count within =
            Array.fold_left ( + ) 0
              (Array.mapi
                 (fun i x -> if Float.abs x <= within i then 1 else 0)
                 multiples)
          in
          let msg =
            Printf.sprintf "runs whose interval holds moment %d" (j + 1)
          in
          at_least int ~than:175 ~msg (count (fun _ -> z975));
          if centre then begin
            (* An error is [t975 / z975] pivot scales. *)
            let central = count (fun i -> t75.(i) *. z975 /. t975.(i)) in
            let msg =
              Printf.sprintf "runs whose central interval holds moment %d"
                (j + 1)
            in
            at_least int ~than:81 ~msg central;
            at_most int ~than:119 ~msg central
          end)
        truth)

let law_14s =
  group "expectations"
    [
      law_14 ~centre:true conjugate
        [| 5.7 /. 6.; (1. /. 6.) +. ((5.7 /. 6.) ** 2.) |];
      law_14 ~centre:false banana [| 0.; cut |];
    ]

(* A run on the conjugate model from 20 prior draws. *)
let small_run k =
  Norn.Nested.run t ~budget ~prior:conjugate.prior
    ~likelihood:conjugate.likelihood (Nx.Rng.fold_in k 0)
    (conjugate.draw (Nx.Rng.fold_in k 1) 20)

let small i = small_run (Nx.Rng.key i)

(* Births: a dead point's shrinkage is one over the points live when it died,
   [1 / (n - j + 1)] for the [j]-th of a step's deletions. A point drawn after
   [b] deaths is live at death [i] when [b <= i] and it has not died before
   [i], so counting them recovers each dead point's live count. *)
let born =
  cases "every dead point's live count is the points born before it"
    ~id:(fun (n, batch, steps) ->
      Printf.sprintf "%d live, batch %d, %d steps" n batch steps)
    [ (10, 5, 4); (10, 3, 5); (7, 1, 9); (12, 11, 2) ]
    (fun (n, batch, steps) ->
      let step =
        Norn.Nested.step t ~prior:gaussian.prior ~likelihood:gaussian.likelihood
      in
      let s =
        List.fold_left
          (fun s i -> step (Nx.Rng.key i) s)
          (Norn.Nested.init t ~batch ~budget:steps ~prior:gaussian.prior
             ~likelihood:gaussian.likelihood
             (gaussian.draw (Nx.Rng.key 0) n))
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
          (1. /. float_of_int !live) shrinkage.(i)
      done)

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
            (stop_name
               (Norn.Evidence.stop
                  (List.hd (small_runs conjugate)))));
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
            (Nx.item [] (Norn.Evidence.log_evidence (small 3)))
            (Nx.item [] (Norn.Evidence.log_evidence (small 3))));
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
      test "a compiled run is the eager one" (fun () ->
          let compiled =
            Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns ez) small_run
              (Nx.Rng.key 5)
          in
          let eager = small_run (Nx.Rng.key 5) in
          let field f z = Nx.item [] (f z) in
          List.iter
            (fun (name, f) ->
              equal (float 1e-9) ~msg:name (field f eager) (field f compiled))
            [
              ("ln Z", Norn.Evidence.log_evidence);
              ("its error", Norn.Evidence.error);
              ("the effective groups", Norn.Evidence.effective_groups);
            ];
          let moments z =
            Nx.to_array (snd (Norn.Evidence.expectation t first_moments z))
          in
          equal (array (float 1e-9)) (moments eager) (moments compiled));
      born;
      posterior;
    ]

let () =
  exit
    (run "Norn.Nested"
       [
         runs_and_refusals;
         group "evidence"
           [
             law_13 conjugate (fun () -> small_runs conjugate);
             law_13 gaussian (fun () -> runs_of gaussian 1);
             law_13 mixture (fun () -> runs_of ~live:100 mixture 20);
           ];
         law_14s;
       ])
