(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models
open Nested_runs

(* ln Z and its error, on models whose evidence is known. Both claims are on
   multiples of the errors, which grow as the live points fall, so their odds of
   a false failure hold at any count whose errors stay calibrated. The conjugate
   model runs the expectation law's 200 runs at 50 live points. The mixture
   takes 100 live points: over 40 runs its [ln Z] lay 0.00 errors from the truth
   on average and spread 1.10 of the mean error (-0.03 and 1.03 at 200). The
   ten-dimensional Gaussian keeps 200 live points: at 100 its [ln Z] lay 0.32 ±
   0.25 errors above the truth. *)

let conjugate_runs = fixture (fun () -> runs ~live:50 conjugate 200)
let mixture_runs = fixture (fun () -> runs ~live:100 mixture 20)
let gaussian_runs = fixture (fun () -> runs ~live:200 gaussian 16)
let mean xs = List.fold_left ( +. ) 0. xs /. float_of_int (List.length xs)

let spread xs =
  let m = mean xs in
  Float.sqrt
    (List.fold_left (fun s v -> s +. ((v -. m) ** 2.)) 0. xs
    /. float_of_int (List.length xs - 1))

(* The mean of [R] runs' ln Z lies within three of its standard errors, their
   spread over sqrt R, of the truth: beyond lies Student's t with R - 1 degrees
   of freedom, with probability 0.30% at 200 runs, 0.74% at 20 and 0.89% at
   16. *)
let mean_near m runs =
  test m.name (fun () ->
      Compiled.here ();
      let zs = List.map fst (runs ()) in
      List.iter (fun z -> equal stop Converged (Norn.Evidence.stop z)) zs;
      let values = List.map lnz zs in
      let n = float_of_int (List.length values) in
      at_most float_exact
        ~than:(3. *. spread values /. Float.sqrt n)
        ~msg:(Printf.sprintf "mean ln Z %g, the truth %g" (mean values) m.log_z)
        (Float.abs (mean values -. m.log_z)))

(* The spread of [R] runs' ln Z lies within a factor of two of their mean error:
   a calibrated error falls outside with probability 3.9e-4 at 20 runs, from χ²
   with 19 degrees of freedom, and 1e-20 at 200. *)
let spread_is_error m runs =
  test m.name (fun () ->
      Compiled.here ();
      let zs = List.map fst (runs ()) in
      let ratio = spread (List.map lnz zs) /. mean (List.map err zs) in
      let msg = "the spread of ln Z over the mean error" in
      greater float_exact ~than:0.5 ~msg ratio;
      less float_exact ~than:2. ~msg ratio)

let run_tests =
  group "run"
    [
      group "ln Z's mean over the runs is within three standard errors of"
        [
          mean_near conjugate conjugate_runs;
          mean_near gaussian gaussian_runs;
          mean_near mixture mixture_runs;
        ];
      group "ln Z's spread over the runs is its error, on"
        [
          spread_is_error conjugate conjugate_runs;
          spread_is_error gaussian gaussian_runs;
          spread_is_error mixture mixture_runs;
        ];
    ]

let () = exit (run "Norn.Nested ln Z" [ run_tests ])
