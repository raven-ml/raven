(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models
open Nested_runs

(* Expectations, over [calibration] = 200 independent runs of 50 live points.

   Under the conjugate model x's posterior has mean [5.7 / 6] and second moment
   [1 / 6 + (5.7 / 6)²]. On the banana x1's posterior is the standard normal cut
   to [-4, 4], of mean 0 and second moment [1 - 8 φ(4) / P(|x1| <= 4)], and
   slice moves leave the draws of a lineage correlated: an error that takes the
   draws of different starts as independent covered 59% to 68%. *)

let calibration = 200
let conjugate_runs = fixture (fun () -> runs ~live:50 conjugate calibration)
let banana_runs = fixture (fun () -> runs ~live:50 banana calibration)

let cut =
  let inside = 1. -. Float.erfc (4. /. Float.sqrt 2.) in
  let phi4 = Float.exp (-8.) /. Float.sqrt (2. *. Float.pi) in
  1. -. (8. *. phi4 /. inside)

let conjugate_moments = [| 5.7 /. 6.; (1. /. 6.) +. ((5.7 /. 6.) ** 2.) |]
let banana_moments = [| 0.; cut |]

(* The normal's 0.975 quantile. *)
let z975 = 1.959963984540054

(* [t_quantile p df] is Student's t's [p] quantile, [p > 1/2], at each of the
   degrees of freedom [df]: its two tails beyond [t] hold [I_x (df / 2, 1 / 2)],
   [x = df / (df + t²)], so [x] is the beta's [2 (1 - p)] quantile and [t² = df
   (1 - x) / x]. *)
let t_quantile p df =
  let df = Nx.create Nx.float64 [| Array.length df |] df in
  let x =
    Nx.betaincinv (Nx.mul_s df 0.5) (Nx.full_like df 0.5)
      (Nx.full_like df (2. *. (1. -. p)))
  in
  Nx.to_array (Nx.sqrt (Nx.div (Nx.mul df (Nx.rsub_s 1. x)) x))

(* [counts runs truth ~within] is, per moment [j], the number of runs whose
   estimate lies within [within i] errors of [truth.(j)], [i] the run. *)
let counts runs truth ~within =
  Array.mapi
    (fun j truth ->
      List.fold_left ( + ) 0
        (List.mapi
           (fun i (_, (e, err)) ->
             let m = (Nx.item [ j ] e -. truth) /. Nx.item [ j ] err in
             if Float.abs m <= within i then 1 else 0)
           runs))
    truth

(* The number of runs whose interval of [± 1.96] errors holds the truth is [Bin
   (200, p)], [p] the coverage, 0.95 when the errors are right. Each model's
   count is at least 175: at [p = 0.95] a moment falls short with probability
   9e-6, at the lowest coverage seen, 0.935, with 6e-4, and an error that covers
   68% passes with probability 1e-10. Over six key sets the conjugate model's
   runs covered 0.945 to 0.975 of each moment, the banana's 0.97 to 0.99. *)
let interval_holds m runs truth =
  test m.name (fun () ->
      Compiled.here ();
      let runs = runs () in
      List.iter (fun (z, _) -> equal stop Converged (Norn.Evidence.stop z)) runs;
      Array.iteri
        (fun j count ->
          at_least int ~than:175
            ~msg:(Printf.sprintf "runs whose interval holds moment %d" (j + 1))
            count)
        (counts runs truth ~within:(fun _ -> z975)))

(* The errors widen by Student's t, so a run's error over [t / 1.96], [t] the
   0.975 quantile at [K - 1] degrees of freedom, [K] the run's
   [Norn.Evidence.effective_groups], is the pivot's scale, and the truth lies
   within [t_0.75] of those scales, the quantile at 0.75, in half the runs: [Bin
   (200, 0.5)]. The count lies in [81, 119], outside with probability 5.7e-3 per
   moment. An error 1.4 times too wide, as the one that counted the volumes'
   noise twice, gives counts near 131 and passes with probability 0.06 per
   moment; over six key sets, the new spread times 1.42 failed in each. The
   banana's errors are wider than the pivot's at its centre, where few lineages
   make the multiples a mixture: its count lay in 123 to 141, so it is held to
   the interval alone. *)
let centre_holds m runs truth =
  test m.name (fun () ->
      Compiled.here ();
      let runs = runs () in
      let df =
        Array.of_list
          (List.map
             (fun (z, _) -> Nx.item [] (Norn.Evidence.effective_groups z) -. 1.)
             runs)
      in
      let t975 = t_quantile 0.975 df and t75 = t_quantile 0.75 df in
      Array.iteri
        (fun j count ->
          let msg =
            Printf.sprintf "runs whose central interval holds moment %d" (j + 1)
          in
          at_least int ~than:81 ~msg count;
          at_most int ~than:119 ~msg count)
        (counts runs truth ~within:(fun i -> t75.(i) *. z975 /. t975.(i))))

let expectation_tests =
  group "expectation"
    [
      group "the 95% interval holds the truth in at least 175 of 200 runs, on"
        [
          interval_holds conjugate conjugate_runs conjugate_moments;
          interval_holds banana banana_runs banana_moments;
        ];
      group
        "the pivot's central half holds the truth in 81 to 119 of 200 runs, on"
        [ centre_holds conjugate conjugate_runs conjugate_moments ];
    ]

let () = exit (run "Norn.Nested expectations" [ expectation_tests ])
