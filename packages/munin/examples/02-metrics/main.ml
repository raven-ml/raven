(** Metric declarations and rich scalar logging.

    Demonstrates Session.metric with summaries, goals, and step_metric for
    custom x-axes. Simulates an iterative solver converging over epochs. *)

open Munin

let () =
  let store = Store.open_ ~root:"_munin" () in
  let session =
    Session.start ~store ~experiment:"solver" ~name:"conjugate-gradient"
      ~params:[ ("tolerance", `Float 1e-6); ("max_iter", `Int 500) ]
      ()
  in
  (* Declare how metrics should be summarised and compared. The epoch axis comes
     first: metrics plotted against it refer to its handle. *)
  let epoch_metric = Session.metric session "epoch" in
  let residual_metric = Session.metric session ~goal:`Minimize "residual" in
  let rate_metric =
    Session.metric session ~summary:`Mean ~step_metric:epoch_metric
      "convergence_rate"
  in

  (* Simulate an iterative solver: residual shrinks, rate stabilises. *)
  let residual = ref 1.0 in
  for epoch = 1 to 20 do
    let rate = 0.7 +. Random.float 0.1 in
    residual := !residual *. rate;
    let step = epoch * 25 in
    Session.log_metrics session ~step
      [
        (residual_metric, !residual);
        (rate_metric, rate);
        (epoch_metric, Float.of_int epoch);
      ]
  done;

  Session.set_summary session [ ("final_residual", `Float !residual) ];
  Session.finish session;

  (* Read back and print. *)
  let run = Session.run session in
  Printf.printf "run: %s\n" (Session.id session);
  Printf.printf "metric keys: %s\n" (String.concat ", " (Run.metric_keys run));

  let defs = Run.metric_defs run in
  List.iter
    (fun (key, (def : Metric.def)) ->
      let goal =
        match def.goal with
        | Some `Minimize -> "minimize"
        | Some `Maximize -> "maximize"
        | None -> "none"
      in
      Printf.printf "  %s: summary=%s goal=%s\n" key
        (match def.summary with
        | `Min -> "min"
        | `Max -> "max"
        | `Mean -> "mean"
        | `Last -> "last"
        | `None -> "none")
        goal)
    defs;

  let history = Run.metric_history run "residual" in
  Printf.printf "residual: %d samples, final=%.2e\n" (List.length history)
    (List.nth history (List.length history - 1)).value
