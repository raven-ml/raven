(** Metric definitions and rich scalar logging.

    Demonstrates define_metric with summaries, goals, and step_metric for custom
    x-axes. Simulates an iterative solver converging over epochs. *)

open Munin

let () =
  let root = "_munin" in
  let session =
    Session.start ~root ~experiment:"solver" ~name:"conjugate-gradient"
      ~params:[ ("tolerance", `Float 1e-6); ("max_iter", `Int 500) ]
      ()
  in
  (* Declare how metrics should be summarised and compared. *)
  Session.define_metric session "residual" ~summary:`Min ~goal:`Minimize ();
  Session.define_metric session "convergence_rate" ~summary:`Mean
    ~step_metric:"epoch" ();

  (* Simulate an iterative solver: residual shrinks, rate stabilises. *)
  let residual = ref 1.0 in
  for epoch = 1 to 20 do
    let rate = 0.7 +. Random.float 0.1 in
    residual := !residual *. rate;
    let step = epoch * 25 in
    Session.log_metrics session ~step
      [
        ("residual", !residual);
        ("convergence_rate", rate);
        ("epoch", Float.of_int epoch);
      ]
  done;

  Session.set_summary session [ ("final_residual", `Float !residual) ];
  Session.finish session ();

  (* Read back and print. *)
  let run = Session.run session in
  Printf.printf "run: %s\n" (Run.id run);
  Printf.printf "metric keys: %s\n" (String.concat ", " (Run.metric_keys run));

  let defs = Run.metric_defs run in
  List.iter
    (fun (key, (def : Run.metric_def)) ->
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
