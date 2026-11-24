(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Evidence_models
open Nested_runs

(* ln Z and its error. On models whose evidence is known, a run's [ln Z] is
   within three of its errors of the truth, and over several runs the spread of
   [ln Z] is within a factor of two of the mean error. A calibrated error misses
   the first claim with probability 0.27%. Over 20 runs the spread of a correct
   error falls in [0.5, 2] with probability 0.995, over 200 with probability 1 -
   1e-20. The Gaussian in ten dimensions is held to the first claim.

   Both claims are on multiples of the errors, which grow as the live points
   fall, so their odds of a miss hold at any count whose errors stay calibrated.
   The conjugate model runs the expectation law's 200 runs at 50 live points.
   The mixture takes 100 live points, at two thirds of 200's time: over 40 runs
   at 100 its [ln Z] lay 0.00 errors from the truth on average and spread 1.10
   of the mean error (-0.03 and 1.03 at 200). The Gaussian keeps 200: over 10
   runs at 100 its [ln Z] lay 0.32 ± 0.25 errors above the truth (0.12 at 200),
   and a shift of 0.32 would raise the first claim's odds of a miss from 0.27%
   to 0.41%. *)
let within_errors m zs =
  List.iter
    (fun z -> equal string "converged" (stop_name (Norn.Evidence.stop z)))
    zs;
  let z0 = List.hd zs in
  at_most (float 1e-12)
    ~than:(3. *. err z0)
    ~msg:(Printf.sprintf "ln Z = %g, the truth %g" (lnz z0) m.log_z)
    (Float.abs (lnz z0 -. m.log_z));
  let n = float_of_int (List.length zs) in
  if n > 1. then begin
    let values = List.map lnz zs in
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
  end

let log_evidence_tests =
  group "ln Z"
    [
      test conjugate.name (fun () ->
          Compiled.here ();
          within_errors conjugate (List.map fst (runs ~live:50 conjugate 200)));
      test gaussian.name (fun () ->
          (* Eager: compiling this run costs more than running it. *)
          let k = Nx.Rng.split_batch ~n:1 (Nx.Rng.key 70) in
          let k = Nx.Ptree.map Nx.Rng.ptree (fun _ k -> Nx.slice [ I 0 ] k) k in
          within_errors gaussian [ nest gaussian 200 k ]);
      test mixture.name (fun () ->
          Compiled.here ();
          within_errors mixture (List.map fst (runs ~live:100 mixture 20)));
    ]

let () = exit (run "Norn.Nested ln Z" [ log_evidence_tests ])
