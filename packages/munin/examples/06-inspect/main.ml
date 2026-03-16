(** Querying and inspecting past runs.

    The "notebook" use case: open a store, browse experiments, filter runs,
    examine provenance, and extract metric histories. Assumes earlier examples
    have been run to populate the store. *)

open Munin

let () =
  let root = "_munin" in
  let store = Store.open_ ~root () in

  (* List all experiments. *)
  let experiments = Store.list_experiments store in
  Printf.printf "experiments: %s\n\n" (String.concat ", " experiments);

  (* List runs, optionally filtering. *)
  let all_runs = Store.list_runs store () in
  Printf.printf "total runs: %d\n" (List.length all_runs);
  let finished = Store.list_runs store ~status:`finished () in
  Printf.printf "finished runs: %d\n\n" (List.length finished);

  (* Find the latest run and inspect it. *)
  (match Store.latest_run store () with
  | None -> Printf.printf "no runs found\n"
  | Some run ->
      Printf.printf "latest run: %s\n" (Run.id run);
      Printf.printf "  experiment: %s\n" (Run.experiment_name run);
      Printf.printf "  name: %s\n"
        (Option.value ~default:"(none)" (Run.name run));
      Printf.printf "  status: %s\n"
        (match Run.status run with
        | `running -> "running"
        | `finished -> "finished"
        | `failed -> "failed"
        | `killed -> "killed");
      Printf.printf "  tags: [%s]\n" (String.concat ", " (Run.tags run));

      (* Provenance. *)
      let prov = Run.provenance run in
      Printf.printf "  hostname: %s\n" (Option.value ~default:"?" prov.hostname);
      Printf.printf "  git: %s%s\n"
        (Option.value ~default:"?" prov.git_commit)
        (match prov.git_dirty with Some true -> " (dirty)" | _ -> "");

      (* Params. *)
      let params = Run.params run in
      if params <> [] then (
        Printf.printf "  params:\n";
        List.iter
          (fun (k, v) ->
            Printf.printf "    %s = %s\n" k (Format.asprintf "%a" Value.pp v))
          params);

      (* Metrics. *)
      let keys = Run.metric_keys run in
      Printf.printf "  metrics: %s\n" (String.concat ", " keys);
      List.iter
        (fun key ->
          let history = Run.metric_history run key in
          let n = List.length history in
          if n > 0 then
            let last = (List.nth history (n - 1)).value in
            Printf.printf "    %s: %d samples, last=%.4g\n" key n last)
        keys);

  (* List artifacts. *)
  let artifacts = Store.list_artifacts store () in
  Printf.printf "\nartifacts: %d\n" (List.length artifacts);
  List.iter
    (fun a ->
      Printf.printf "  %s v%s (%d bytes)\n" (Artifact.name a)
        (Artifact.version a) (Artifact.size_bytes a))
    artifacts
