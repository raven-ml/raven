let pp_kv pairs =
  List.iter
    (fun (key, value) ->
      Printf.printf "  %s: %s\n" key (Format.asprintf "%a" Value.pp value))
    pairs

let pp_string_list items = Printf.printf "  %s\n" (String.concat ", " items)

let string_of_status = function
  | `running -> "running"
  | `finished -> "finished"
  | `failed -> "failed"
  | `killed -> "killed"

let string_of_kind = function
  | `dataset -> "dataset"
  | `model -> "model"
  | `checkpoint -> "checkpoint"
  | `file -> "file"
  | `dir -> "dir"
  | `other -> "other"

let string_of_payload = function `file -> "file" | `dir -> "dir"

let value_to_float = function
  | `Float f -> Some f
  | `Int n -> Some (Float.of_int n)
  | `String s -> Float.of_string_opt s
  | `Bool _ -> None

let sorted_unique_keys runs f =
  let seen = Hashtbl.create 16 in
  List.iter
    (fun run -> List.iter (fun (k, _) -> Hashtbl.replace seen k ()) (f run))
    runs;
  Hashtbl.to_seq_keys seen |> List.of_seq |> List.sort String.compare

let run_root root f =
  let store = Store.open_ ?root () in
  f store

let runs_cmd =
  let doc = "List tracked runs" in
  let experiment =
    let doc = "Limit the listing to a single experiment." in
    Cmdliner.Arg.(
      value & opt (some string) None & info [ "experiment" ] ~docv:"NAME" ~doc)
  in
  Cmdliner.Cmd.v
    (Cmdliner.Cmd.info "runs" ~doc)
    Cmdliner.Term.(
      const (fun root experiment ->
          run_root root (fun store ->
              Store.list_runs store ?experiment ()
              |> List.iter (fun run ->
                  Printf.printf "%s\t%s\t%s\t%s\t%s\t%s\n" (Run.id run)
                    (Run.experiment_name run)
                    (string_of_status (Run.status run))
                    (Option.value (Run.parent_id run) ~default:"-")
                    (Option.value (Run.name run) ~default:"-")
                    (Option.value (Run.provenance run).git_commit ~default:"-"))))
      $ Cmdliner.Arg.(
          value & opt (some string) None & info [ "root" ] ~docv:"DIR")
      $ experiment)

let show_cmd =
  let doc = "Show one run" in
  let run_id =
    Cmdliner.Arg.(required & pos 0 (some string) None & info [] ~docv:"RUN_ID")
  in
  Cmdliner.Cmd.v
    (Cmdliner.Cmd.info "show" ~doc)
    Cmdliner.Term.(
      const (fun root run_id ->
          run_root root (fun store ->
              match Store.find_run store run_id with
              | None ->
                  Printf.eprintf "munin: run not found: %s\n" run_id;
                  exit 1
              | Some run ->
                  Printf.printf "id: %s\n" (Run.id run);
                  Printf.printf "experiment: %s\n" (Run.experiment_name run);
                  Printf.printf "name: %s\n"
                    (Option.value (Run.name run) ~default:"-");
                  Printf.printf "parent: %s\n"
                    (Option.value (Run.parent_id run) ~default:"-");
                  Printf.printf "status: %s\n"
                    (string_of_status (Run.status run));
                  Printf.printf "started_at: %.0f\n" (Run.started_at run);
                  Option.iter
                    (Printf.printf "ended_at: %.0f\n")
                    (Run.ended_at run);
                  Printf.printf "resumable: %b\n" (Run.resumable run);
                  Printf.printf "notes: %s\n"
                    (Option.value (Run.notes run) ~default:"-");
                  let prov = Run.provenance run in
                  Printf.printf "command: %s\n" (String.concat " " prov.command);
                  Printf.printf "cwd: %s\n" prov.cwd;
                  Printf.printf "hostname: %s\n"
                    (Option.value prov.hostname ~default:"-");
                  Printf.printf "pid: %d\n" prov.pid;
                  Printf.printf "git_commit: %s\n"
                    (Option.value prov.git_commit ~default:"-");
                  Printf.printf "git_dirty: %s\n"
                    (match prov.git_dirty with
                    | None -> "-"
                    | Some true -> "true"
                    | Some false -> "false");
                  Printf.printf "env:\n";
                  List.iter
                    (fun (key, value) -> Printf.printf "  %s=%s\n" key value)
                    prov.env;
                  Printf.printf "tags:\n";
                  List.iter (Printf.printf "  %s\n") (Run.tags run);
                  Printf.printf "params:\n";
                  pp_kv (Run.params run);
                  Printf.printf "summary:\n";
                  pp_kv (Run.summary run);
                  Printf.printf "metric_keys:\n";
                  pp_string_list (Run.metric_keys run);
                  Printf.printf "latest_metrics:\n";
                  List.iter
                    (fun (key, (metric : Run.metric)) ->
                      Printf.printf "  %s: step=%d value=%g\n" key metric.step
                        metric.value)
                    (Run.latest_metrics run);
                  Printf.printf "children:\n";
                  List.iter
                    (fun child -> Printf.printf "  %s\n" (Run.id child))
                    (Run.children run);
                  Printf.printf "output_artifacts:\n";
                  List.iter
                    (fun artifact ->
                      Printf.printf "  %s %s aliases=[%s] consumers=[%s]\n"
                        (Artifact.name artifact)
                        (Artifact.version artifact)
                        (String.concat "," (Artifact.aliases artifact))
                        (String.concat "," (Artifact.consumer_run_ids artifact)))
                    (Run.output_artifacts run);
                  Printf.printf "input_artifacts:\n";
                  List.iter
                    (fun artifact ->
                      Printf.printf "  %s %s producer=%s\n"
                        (Artifact.name artifact)
                        (Artifact.version artifact)
                        (Option.value
                           (Artifact.producer_run_id artifact)
                           ~default:"-"))
                    (Run.input_artifacts run)))
      $ Cmdliner.Arg.(
          value & opt (some string) None & info [ "root" ] ~docv:"DIR")
      $ run_id)

let artifacts_cmd =
  let doc = "List stored artifacts" in
  let name =
    let doc = "Limit the listing to a single artifact name." in
    Cmdliner.Arg.(
      value & opt (some string) None & info [ "name" ] ~docv:"NAME" ~doc)
  in
  Cmdliner.Cmd.v
    (Cmdliner.Cmd.info "artifacts" ~doc)
    Cmdliner.Term.(
      const (fun root name ->
          run_root root (fun store ->
              Store.list_artifacts store ?name ()
              |> List.iter (fun artifact ->
                  Printf.printf "%s\t%s\t%s\t%s\t%d\t%s\t%s\n"
                    (Artifact.name artifact)
                    (Artifact.version artifact)
                    (string_of_kind (Artifact.kind artifact))
                    (string_of_payload (Artifact.payload artifact))
                    (Artifact.size_bytes artifact)
                    (Option.value
                       (Artifact.producer_run_id artifact)
                       ~default:"-")
                    (String.concat "," (Artifact.consumer_run_ids artifact)))))
      $ Cmdliner.Arg.(
          value & opt (some string) None & info [ "root" ] ~docv:"DIR")
      $ name)

let watch_cmd =
  let doc = "Launch the live experiment dashboard" in
  let experiment =
    let doc = "Limit to runs in a single experiment." in
    Cmdliner.Arg.(
      value & opt (some string) None & info [ "experiment" ] ~docv:"NAME" ~doc)
  in
  let runs =
    Cmdliner.Arg.(value & pos_all string [] & info [] ~docv:"RUN_ID")
  in
  Cmdliner.Cmd.v
    (Cmdliner.Cmd.info "watch" ~doc)
    Cmdliner.Term.(
      const (fun root experiment runs ->
          Munin_tui.run ?root ?experiment ~runs ())
      $ Cmdliner.Arg.(
          value & opt (some string) None & info [ "root" ] ~docv:"DIR")
      $ experiment $ runs)

let compare_cmd =
  let doc = "Compare runs side by side" in
  let root =
    Cmdliner.Arg.(value & opt (some string) None & info [ "root" ] ~docv:"DIR")
  in
  let run_ids =
    Cmdliner.Arg.(non_empty & pos_all string [] & info [] ~docv:"RUN_ID")
  in
  Cmdliner.Cmd.v
    (Cmdliner.Cmd.info "compare" ~doc)
    Cmdliner.Term.(
      const (fun root run_ids ->
          run_root root (fun store ->
              let runs =
                List.filter_map (fun id -> Store.find_run store id) run_ids
              in
              if runs = [] then (
                Printf.eprintf "munin: no runs found\n";
                exit 1);
              let run_label run =
                Option.value (Run.name run) ~default:(Run.id run)
              in
              (* Header *)
              Printf.printf "key";
              List.iter (fun run -> Printf.printf "\t%s" (run_label run)) runs;
              Printf.printf "\n";
              (* Params *)
              let param_keys = sorted_unique_keys runs Run.params in
              List.iter
                (fun key ->
                  Printf.printf "%s" key;
                  List.iter
                    (fun run ->
                      let v =
                        match List.assoc_opt key (Run.params run) with
                        | Some v -> Format.asprintf "%a" Value.pp v
                        | None -> "-"
                      in
                      Printf.printf "\t%s" v)
                    runs;
                  Printf.printf "\n")
                param_keys;
              (* Summaries *)
              let summary_keys = sorted_unique_keys runs Run.summary in
              (* Collect goals from metric_defs *)
              let goals = Hashtbl.create 8 in
              List.iter
                (fun run ->
                  List.iter
                    (fun (key, (def : Run.metric_def)) ->
                      match def.goal with
                      | Some g -> Hashtbl.replace goals key g
                      | None -> ())
                    (Run.metric_defs run))
                runs;
              List.iter
                (fun key ->
                  Printf.printf "%s" key;
                  let values =
                    List.map
                      (fun run ->
                        Run.find_summary run key
                        |> Fun.flip Option.bind value_to_float)
                      runs
                  in
                  (* Find best index *)
                  let best_idx =
                    match Hashtbl.find_opt goals key with
                    | None -> None
                    | Some goal ->
                        let compare =
                          match goal with
                          | `Minimize -> fun a b -> Float.compare a b
                          | `Maximize -> fun a b -> Float.compare b a
                        in
                        let best = ref None in
                        List.iteri
                          (fun i v ->
                            match (v, !best) with
                            | Some v, None -> best := Some (i, v)
                            | Some v, Some (_, bv) ->
                                if compare v bv < 0 then best := Some (i, v)
                            | None, _ -> ())
                          values;
                        Option.map fst !best
                  in
                  List.iteri
                    (fun i _ ->
                      let s =
                        match List.nth values i with
                        | Some v ->
                            let s = Printf.sprintf "%g" v in
                            if Some i = best_idx then s ^ "*" else s
                        | None -> "-"
                      in
                      Printf.printf "\t%s" s)
                    runs;
                  Printf.printf "\n")
                summary_keys))
      $ root $ run_ids)

let metrics_cmd =
  let doc = "Show metric history" in
  let root =
    Cmdliner.Arg.(value & opt (some string) None & info [ "root" ] ~docv:"DIR")
  in
  let run_id =
    Cmdliner.Arg.(required & pos 0 (some string) None & info [] ~docv:"RUN_ID")
  in
  let key =
    let doc = "Metric key to dump history for." in
    Cmdliner.Arg.(
      value & opt (some string) None & info [ "key" ] ~docv:"KEY" ~doc)
  in
  let format =
    let doc = "Output format: tsv (default), csv, or json." in
    Cmdliner.Arg.(
      value
      & opt (enum [ ("tsv", `Tsv); ("csv", `Csv); ("json", `Json) ]) `Tsv
      & info [ "format" ] ~docv:"FORMAT" ~doc)
  in
  Cmdliner.Cmd.v
    (Cmdliner.Cmd.info "metrics" ~doc)
    Cmdliner.Term.(
      const (fun root run_id key format ->
          run_root root (fun store ->
              match Store.find_run store run_id with
              | None ->
                  Printf.eprintf "munin: run not found: %s\n" run_id;
                  exit 1
              | Some run -> (
                  match key with
                  | None ->
                      (* Listing mode *)
                      Printf.printf "key\tlatest_value\tlatest_step\tcount\n";
                      List.iter
                        (fun (key, (m : Run.metric)) ->
                          let count =
                            List.length (Run.metric_history run key)
                          in
                          Printf.printf "%s\t%g\t%d\t%d\n" key m.value m.step
                            count)
                        (Run.latest_metrics run)
                  | Some key -> (
                      let history = Run.metric_history run key in
                      match format with
                      | `Tsv ->
                          Printf.printf "step\ttimestamp\tvalue\n";
                          List.iter
                            (fun (m : Run.metric) ->
                              Printf.printf "%d\t%.6f\t%g\n" m.step m.timestamp
                                m.value)
                            history
                      | `Csv ->
                          Printf.printf "step,timestamp,value\n";
                          List.iter
                            (fun (m : Run.metric) ->
                              Printf.printf "%d,%.6f,%g\n" m.step m.timestamp
                                m.value)
                            history
                      | `Json ->
                          Printf.printf "[";
                          List.iteri
                            (fun i (m : Run.metric) ->
                              if i > 0 then Printf.printf ",";
                              Printf.printf
                                "{\"step\":%d,\"timestamp\":%.6f,\"value\":%g}"
                                m.step m.timestamp m.value)
                            history;
                          Printf.printf "]\n"))))
      $ root $ run_id $ key $ format)

let () =
  exit
    (Cmdliner.Cmd.eval
       (Cmdliner.Cmd.group
          (Cmdliner.Cmd.info "munin" ~doc:"Local experiment tracking for Raven")
          [
            runs_cmd;
            show_cmd;
            artifacts_cmd;
            watch_cmd;
            compare_cmd;
            metrics_cmd;
          ]))
