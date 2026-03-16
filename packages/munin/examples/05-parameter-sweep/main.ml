(** Parameter sweep with grouped runs.

    Runs the same computation with different configurations, grouped under a
    single sweep. Compares results at the end via Store queries. *)

open Munin

let () =
  let root = "_munin" in
  let store = Store.open_ ~root () in
  let group = "sweep-1" in

  (* Sweep over three configurations. *)
  let configs =
    [
      ("aggressive", 0.1, 50);
      ("moderate", 0.01, 100);
      ("conservative", 0.001, 200);
    ]
  in
  List.iter
    (fun (name, step_size, max_iter) ->
      let session =
        Session.start ~root ~experiment:"optimisation" ~name ~group
          ~params:
            [ ("step_size", `Float step_size); ("max_iter", `Int max_iter) ]
          ()
      in
      Session.define_metric session "error" ~summary:`Min ~goal:`Minimize ();

      (* Simulate convergence: smaller steps converge slower but lower. *)
      let error = ref 10.0 in
      for i = 1 to max_iter do
        error := (!error *. (1.0 -. step_size)) +. Random.float 0.01;
        if i mod 10 = 0 then Session.log_metric session ~step:i "error" !error
      done;
      Session.finish session ())
    configs;

  (* Compare: list all runs in the group and print a results table. *)
  let runs = Store.list_runs store ~experiment:"optimisation" ~group () in
  Printf.printf "%-15s  %-10s  %-10s  %-12s\n" "name" "step_size" "max_iter"
    "final_error";
  Printf.printf "%s\n" (String.make 52 '-');
  List.iter
    (fun run ->
      let name = Option.value ~default:"?" (Run.name run) in
      let step_size =
        match Run.find_param run "step_size" with
        | Some (`Float f) -> Printf.sprintf "%g" f
        | _ -> "?"
      in
      let max_iter =
        match Run.find_param run "max_iter" with
        | Some v -> Format.asprintf "%a" Value.pp v
        | None -> "?"
      in
      let latest = Run.latest_metrics run in
      let error =
        match List.assoc_opt "error" latest with
        | Some m -> Printf.sprintf "%.6f" m.value
        | None -> "?"
      in
      Printf.printf "%-15s  %-10s  %-10s  %-12s\n" name step_size max_iter error)
    runs
