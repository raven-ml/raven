(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A sweep: one run that drives three configurations, each a run whose parent is
   the sweep, compared by their best error once they have ended. *)

open Munin

let configs =
  [
    ("aggressive", 0.1, 50);
    ("moderate", 0.01, 100);
    ("conservative", 0.001, 200);
  ]

let trial store ~parent (name, step_size, max_iter) =
  Session.within
    (Session.start ~store ~experiment:"optimisation" ~name ~parent
       ~params:[ ("step_size", `Float step_size); ("max_iter", `Int max_iter) ]
       ~steps:max_iter ())
  @@ fun s ->
  let error = Session.metric s ~goal:`Minimize "error" in
  let e = ref 10.0 in
  for step = 1 to max_iter do
    e := (!e *. (1.0 -. step_size)) +. Random.float 0.01;
    if step mod 10 = 0 then Session.log error ~step !e
  done

let () =
  let store = Store.open_ () in
  let sweep =
    Session.within
      (Session.start ~store ~experiment:"optimisation" ~name:"sweep-1" ())
    @@ fun s ->
    List.iter (trial store ~parent:(Session.id s)) configs;
    Session.id s
  in
  let trials =
    List.filter (fun r -> Run.parent r = Some sweep) (Store.runs store)
  in
  Printf.printf "sweep %s\n" sweep;
  List.iter
    (fun r ->
      (* An ended run's statistics come from the hint beside its log. *)
      let best =
        match List.assoc_opt "error" (History.stats_of r) with
        | Some { goal = Some goal; stats; _ } -> Metric.best goal stats
        | _ -> None
      in
      let step_size =
        Option.fold ~none:"-"
          ~some:(Format.asprintf "%a" Value.pp)
          (List.assoc_opt "step_size" (Run.params r))
      in
      Printf.printf "%-13s step_size %-6s best error %s\n"
        (Option.value ~default:"-" (Run.name r))
        step_size
        (Option.fold ~none:"-"
           ~some:(fun (b : Metric.sample) ->
             Printf.sprintf "%.6f at step %d" b.value b.step)
           best))
    trials
