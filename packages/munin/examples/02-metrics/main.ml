(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Metrics with goals and an x axis of their own, then their series and
   statistics read back. An iterative solver's residual shrinks each epoch. *)

open Munin

let solve s =
  let epoch = Session.metric s "epoch" in
  let residual = Session.metric s ~goal:`Minimize "residual" in
  let rate = Session.metric s ~step_metric:epoch "convergence_rate" in
  let r = ref 1.0 in
  for e = 1 to 20 do
    let ratio = 0.7 +. Random.float 0.1 in
    r := !r *. ratio;
    (* Several metrics at one step go in one record. *)
    Session.log_metrics s ~step:(e * 25)
      [ (residual, !r); (rate, ratio); (epoch, Float.of_int e) ]
  done;
  Session.id s

let () =
  let store = Store.open_ () in
  let id =
    Session.within
      (Session.start ~store ~experiment:"solver" ~name:"conjugate-gradient"
         ~params:[ ("tolerance", `Float 1e-6); ("max_iter", `Int 500) ]
         ~steps:500 ())
      solve
  in
  let h = History.read (Option.get (Store.run store id)) in
  Printf.printf "run: %s\n" id;
  List.iter
    (fun key ->
      let goal =
        match History.goal h key with
        | Some `Minimize -> "minimize"
        | Some `Maximize -> "maximize"
        | None -> "-"
      in
      let axis = Option.value ~default:"step" (History.step_metric h key) in
      let st = Option.get (History.stats h key) in
      Printf.printf "%s: goal %s, x axis %s, %d samples, last %g\n" key goal
        axis st.count st.last.value)
    (History.metrics h);
  match History.best h "residual" with
  | Some b -> Printf.printf "best residual: %.3e at step %d\n" b.value b.step
  | None -> ()
