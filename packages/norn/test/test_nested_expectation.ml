(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models
open Nested_runs

(* Expectations. Of [calibration] = 200 independent runs, the number whose
   interval of [± 1.96] errors holds the truth is [Bin (200, p)], [p] the
   coverage, 0.95 when the errors are right. Each model's count is at least 175:
   at [p = 0.95] a moment falls short with probability 9e-6, at the lowest
   coverage seen, 0.935, with 6e-4, and an error that covers 68% passes with
   probability 1e-10. Over six key sets the conjugate model's runs covered 0.945
   to 0.975 of each moment, the banana's 0.97 to 0.99.

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

   Under the conjugate model x's posterior has mean [5.7 / 6] and second moment
   [1 / 6 + (5.7 / 6)²]. On the banana x1's posterior is the standard normal cut
   to [-4, 4], of mean 0 and second moment [1 - 8 φ(4) / P(|x1| <= 4)], and
   slice moves leave the draws of a lineage correlated: an error that takes the
   draws of different starts as independent covered 59% to 68%. *)

let calibration = 200

let cut =
  let inside = 1. -. Float.erfc (4. /. Float.sqrt 2.) in
  let phi4 = Float.exp (-8.) /. Float.sqrt (2. *. Float.pi) in
  1. -. (8. *. phi4 /. inside)

(* The normal's 0.975 quantile. *)
let z975 = 1.959963984540054

(* [t_quantile p df] is Student's t's [p] quantile, [p > 1/2], at each of the
   degrees of freedom [df]: its two tails beyond [t] hold [I_x (df / 2, 1 / 2)],
   [x = df / (df + t²)], which bisection on [x] over 64 halvings solves for [2
   (1 - p)]. *)
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

let covers ~centre m truth =
  test m.name (fun () ->
      Compiled.here ();
      let rs = runs ~live:50 m calibration in
      List.iter
        (fun (z, _) ->
          equal string "converged" (stop_name (Norn.Evidence.stop z)))
        rs;
      let estimates = Array.of_list (List.map snd rs) in
      let df =
        Array.of_list
          (List.map
             (fun (z, _) -> Nx.item [] (Norn.Evidence.effective_groups z) -. 1.)
             rs)
      in
      let t975 = t_quantile 0.975 df and t75 = t_quantile 0.75 df in
      Array.iteri
        (fun j truth ->
          let multiples =
            Array.init calibration (fun i ->
                let e, err = estimates.(i) in
                (Nx.item [ j ] e -. truth) /. Nx.item [ j ] err)
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

let expectation_tests =
  group "expectation"
    [
      covers ~centre:true conjugate
        [| 5.7 /. 6.; (1. /. 6.) +. ((5.7 /. 6.) ** 2.) |];
      covers ~centre:false banana [| 0.; cut |];
    ]

let () = exit (run "Norn.Nested expectations" [ expectation_tests ])
