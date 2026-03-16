open Windtrap

(* Helpers *)

let with_temp_dir f =
  let base = Filename.temp_file "munin" "test" in
  Sys.remove base;
  Unix.mkdir base 0o755;
  Fun.protect
    ~finally:(fun () -> ignore (Sys.command ("rm -rf " ^ Filename.quote base)))
    (fun () -> f base)

let write_text path text =
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc text)

let make_file root name text =
  let path = Filename.concat root name in
  write_text path text;
  path

let make_dir root name files =
  let dir = Filename.concat root name in
  Unix.mkdir dir 0o755;
  List.iter
    (fun (fname, text) -> write_text (Filename.concat dir fname) text)
    files;
  dir

(* Session lifecycle *)

let test_start_creates_run_dir () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  let run = Session.run session in
  is_true ~msg:"run dir exists" (Sys.file_exists (Run.dir run));
  is_true ~msg:"manifest exists"
    (Sys.file_exists (Filename.concat (Run.dir run) "run.json"));
  is_true ~msg:"status is running" (Run.status run = `running);
  Session.finish session ()

let test_finish_sets_status () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  let run = Session.run session in
  is_true ~msg:"status finished" (Run.status run = `finished);
  is_true ~msg:"ended_at set" (Option.is_some (Run.ended_at run))

let test_finish_failed () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish ~status:`failed session ();
  is_true ~msg:"status failed" (Run.status (Session.run session) = `failed)

let test_finish_killed () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish ~status:`killed session ();
  is_true ~msg:"status killed" (Run.status (Session.run session) = `killed)

let test_finish_idempotent () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  Session.finish ~status:`failed session ();
  is_true ~msg:"still finished" (Run.status (Session.run session) = `finished)

let test_with_run_success () =
  with_temp_dir @@ fun root ->
  let result =
    Session.with_run ~root ~experiment:"exp" (fun session ->
        Session.log_metric session ~step:1 "x" 1.0;
        42)
  in
  equal ~msg:"return value" int 42 result;
  let store = Store.open_ ~root () in
  let run = Option.get (Store.latest_run store ()) in
  is_true ~msg:"status finished" (Run.status run = `finished)

let test_with_run_exception () =
  with_temp_dir @@ fun root ->
  let raised =
    try
      ignore
        (Session.with_run ~root ~experiment:"exp" (fun _session ->
             failwith "boom"));
      false
    with Failure _ -> true
  in
  is_true ~msg:"exception re-raised" raised;
  let store = Store.open_ ~root () in
  let run = Option.get (Store.latest_run store ()) in
  is_true ~msg:"status failed" (Run.status run = `failed)

let test_resume () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  let run = Session.run session in
  is_true ~msg:"resumable before finish" (Run.resumable run);
  let resumed = Session.resume run in
  Session.log_metric resumed ~step:2 "x" 0.5;
  Session.finish resumed ();
  let final = Session.run resumed in
  equal ~msg:"history length" int 2 (List.length (Run.metric_history final "x"));
  is_false ~msg:"not resumable after finish" (Run.resumable final)

let test_resume_finished_raises () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  let run = Session.run session in
  raises_invalid_arg "Munin.Session.resume: run is not resumable" (fun () ->
      ignore (Session.resume run))

let test_ops_after_finish_ignored () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  Session.finish session ();
  Session.log_metric session ~step:2 "x" 2.0;
  Session.set_notes session (Some "late");
  Session.add_tags session [ "late" ];
  Session.set_summary session [ ("late", `Float 1.0) ];
  let run = Session.run session in
  equal ~msg:"only one metric" int 1 (List.length (Run.metric_history run "x"));
  is_true ~msg:"no late note" (Run.notes run = None);
  equal ~msg:"no tags" (list string) [] (Run.tags run)

let lifecycle =
  [
    test "start creates run dir" test_start_creates_run_dir;
    test "finish sets status" test_finish_sets_status;
    test "finish failed" test_finish_failed;
    test "finish killed" test_finish_killed;
    test "finish is idempotent" test_finish_idempotent;
    test "with_run success" test_with_run_success;
    test "with_run exception" test_with_run_exception;
    test "resume" test_resume;
    test "resume finished raises" test_resume_finished_raises;
    test "ops after finish ignored" test_ops_after_finish_ignored;
  ]

(* Scalars *)

let test_log_metric () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "train/loss" 1.5;
  Session.finish session ();
  let run = Session.run session in
  let m =
    match List.assoc_opt "train/loss" (Run.latest_metrics run) with
    | Some m -> m
    | None -> failwith "missing"
  in
  equal ~msg:"step" int 1 m.step;
  equal ~msg:"value" (float 0.0) 1.5 m.value

let test_log_metrics_batch () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metrics session ~step:1 [ ("train/loss", 1.0); ("val/acc", 0.6) ];
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"two keys" int 2 (List.length (Run.metric_keys run))

let test_metric_history_chronological () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 3.0;
  Session.log_metric session ~step:2 "x" 2.0;
  Session.log_metric session ~step:3 "x" 1.0;
  Session.finish session ();
  let run = Session.run session in
  let history = Run.metric_history run "x" in
  equal ~msg:"length" int 3 (List.length history);
  let values = List.map (fun (m : Run.metric) -> m.value) history in
  equal ~msg:"order" (list (float 0.0)) [ 3.0; 2.0; 1.0 ] values

let test_latest_metrics () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  Session.log_metric session ~step:2 "x" 2.0;
  Session.finish session ();
  let run = Session.run session in
  let latest =
    match List.assoc_opt "x" (Run.latest_metrics run) with
    | Some m -> m
    | None -> failwith "missing"
  in
  equal ~msg:"latest step" int 2 latest.step;
  equal ~msg:"latest value" (float 0.0) 2.0 latest.value

let test_metric_keys_sorted () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "z/loss" 1.0;
  Session.log_metric session ~step:1 "a/acc" 0.5;
  Session.log_metric session ~step:1 "m/lr" 0.01;
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"sorted keys" (list string)
    [ "a/acc"; "m/lr"; "z/loss" ]
    (Run.metric_keys run)

let test_explicit_timestamp () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 ~timestamp:42.0 "x" 1.0;
  Session.finish session ();
  let run = Session.run session in
  let m =
    match List.assoc_opt "x" (Run.latest_metrics run) with
    | Some m -> m
    | None -> failwith "missing"
  in
  equal ~msg:"timestamp" (float 0.0) 42.0 m.timestamp

let test_missing_metric_history_empty () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"empty history" (list int) []
    (List.map
       (fun (m : Run.metric) -> m.step)
       (Run.metric_history run "nonexistent"))

let scalars =
  [
    test "log_metric" test_log_metric;
    test "log_metrics batch" test_log_metrics_batch;
    test "history chronological" test_metric_history_chronological;
    test "latest_metrics" test_latest_metrics;
    test "metric_keys sorted" test_metric_keys_sorted;
    test "explicit timestamp" test_explicit_timestamp;
    test "missing key history empty" test_missing_metric_history_empty;
  ]

(* Metric definitions *)

let test_define_metric_summary_and_goal () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "train/loss" ~summary:`Min ~goal:`Minimize ();
  Session.define_metric session "val/acc" ~summary:`Max ~goal:`Maximize ();
  Session.finish session ();
  let run = Session.run session in
  let defs = Run.metric_defs run in
  equal ~msg:"two defs" int 2 (List.length defs);
  let loss_def = List.assoc "train/loss" defs in
  is_true ~msg:"loss summary min" (loss_def.summary = `Min);
  is_true ~msg:"loss goal minimize" (loss_def.goal = Some `Minimize);
  let acc_def = List.assoc "val/acc" defs in
  is_true ~msg:"acc summary max" (acc_def.summary = `Max);
  is_true ~msg:"acc goal maximize" (acc_def.goal = Some `Maximize)

let test_define_metric_all_summaries () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "a" ~summary:`Min ();
  Session.define_metric session "b" ~summary:`Max ();
  Session.define_metric session "c" ~summary:`Mean ();
  Session.define_metric session "d" ~summary:`Last ();
  Session.define_metric session "e" ~summary:`None ();
  Session.finish session ();
  let run = Session.run session in
  let defs = Run.metric_defs run in
  equal ~msg:"five defs" int 5 (List.length defs);
  is_true ~msg:"a=Min" ((List.assoc "a" defs).summary = `Min);
  is_true ~msg:"b=Max" ((List.assoc "b" defs).summary = `Max);
  is_true ~msg:"c=Mean" ((List.assoc "c" defs).summary = `Mean);
  is_true ~msg:"d=Last" ((List.assoc "d" defs).summary = `Last);
  is_true ~msg:"e=None" ((List.assoc "e" defs).summary = `None)

let test_define_metric_step_metric () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "train/loss" ~step_metric:"epoch" ();
  Session.finish session ();
  let run = Session.run session in
  let def = List.assoc "train/loss" (Run.metric_defs run) in
  equal ~msg:"step_metric" (option string) (Some "epoch") def.step_metric

let test_define_metric_default_summary () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "x" ();
  Session.finish session ();
  let run = Session.run session in
  let def = List.assoc "x" (Run.metric_defs run) in
  is_true ~msg:"default summary is Last" (def.summary = `Last)

let test_metric_defs_sorted () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "z" ();
  Session.define_metric session "a" ();
  Session.define_metric session "m" ();
  Session.finish session ();
  let run = Session.run session in
  let keys = List.map fst (Run.metric_defs run) in
  equal ~msg:"sorted" (list string) [ "a"; "m"; "z" ] keys

let metric_definitions =
  [
    test "summary and goal" test_define_metric_summary_and_goal;
    test "all summary modes" test_define_metric_all_summaries;
    test "step_metric" test_define_metric_step_metric;
    test "default summary is Last" test_define_metric_default_summary;
    test "defs sorted" test_metric_defs_sorted;
  ]

(* Metadata *)

let test_set_notes () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.set_notes session (Some "hello");
  Session.finish session ();
  equal ~msg:"notes set" (option string) (Some "hello")
    (Run.notes (Session.run session))

let test_set_notes_replace () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.set_notes session (Some "first");
  Session.set_notes session (Some "second");
  Session.finish session ();
  equal ~msg:"notes replaced" (option string) (Some "second")
    (Run.notes (Session.run session))

let test_clear_notes () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" ~notes:"initial" () in
  Session.set_notes session None;
  Session.finish session ();
  equal ~msg:"notes cleared" (option string) None
    (Run.notes (Session.run session))

let test_set_summary_merge () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.set_summary session [ ("a", `Float 1.0) ];
  Session.set_summary session [ ("b", `Float 2.0); ("a", `Float 3.0) ];
  Session.finish session ();
  let run = Session.run session in
  let summary = Run.summary run in
  equal ~msg:"two keys" int 2 (List.length summary);
  is_true ~msg:"a replaced"
    (Value.to_float (Option.get (Run.find_summary run "a")) = Some 3.0);
  is_true ~msg:"b present"
    (Value.to_float (Option.get (Run.find_summary run "b")) = Some 2.0)

let test_find_summary_missing () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  equal ~msg:"missing key" (option unit) None
    (Option.map ignore (Run.find_summary (Session.run session) "nope"))

let test_add_tags () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" ~tags:[ "initial" ] () in
  Session.add_tags session [ "added" ];
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"tags" (list string) [ "initial"; "added" ] (Run.tags run)

let test_add_tags_dedup () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" ~tags:[ "a" ] () in
  Session.add_tags session [ "a"; "b" ];
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"deduped" (list string) [ "a"; "b" ] (Run.tags run)

let test_add_tags_empty_noop () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.add_tags session [];
  Session.finish session ();
  equal ~msg:"no tags" (list string) [] (Run.tags (Session.run session))

let test_params_immutable () =
  with_temp_dir @@ fun root ->
  let session =
    Session.start ~root ~experiment:"exp"
      ~params:[ ("lr", `Float 0.001); ("bs", `Int 32) ]
      ()
  in
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"param count" int 2 (List.length (Run.params run))

let metadata =
  [
    test "set_notes" test_set_notes;
    test "set_notes replace" test_set_notes_replace;
    test "clear notes" test_clear_notes;
    test "set_summary merge" test_set_summary_merge;
    test "find_summary missing" test_find_summary_missing;
    test "add_tags" test_add_tags;
    test "add_tags dedup" test_add_tags_dedup;
    test "add_tags empty noop" test_add_tags_empty_noop;
    test "params immutable" test_params_immutable;
  ]

(* Provenance *)

let test_provenance_fields () =
  with_temp_dir @@ fun root ->
  let session =
    Session.start ~root ~experiment:"exp" ~command:[ "ocaml"; "train.ml" ]
      ~cwd:root ~hostname:"node0" ~pid:42 ~git_commit:"abc123" ~git_dirty:true
      ()
  in
  Session.finish session ();
  let prov = Run.provenance (Session.run session) in
  equal ~msg:"command" (list string) [ "ocaml"; "train.ml" ] prov.command;
  equal ~msg:"cwd" string root prov.cwd;
  equal ~msg:"hostname" (option string) (Some "node0") prov.hostname;
  equal ~msg:"pid" int 42 prov.pid;
  equal ~msg:"git_commit" (option string) (Some "abc123") prov.git_commit;
  equal ~msg:"git_dirty" (option bool) (Some true) prov.git_dirty

let test_capture_env () =
  with_temp_dir @@ fun root ->
  Unix.putenv "MUNIN_TEST_A" "alpha";
  let session =
    Session.start ~root ~experiment:"exp" ~capture_env:[ "MUNIN_TEST_A" ] ()
  in
  Session.finish session ();
  let prov = Run.provenance (Session.run session) in
  equal ~msg:"env captured"
    (list (pair string string))
    [ ("MUNIN_TEST_A", "alpha") ]
    prov.env

let test_capture_env_missing () =
  with_temp_dir @@ fun root ->
  (* Use a variable name that won't exist *)
  let session =
    Session.start ~root ~experiment:"exp"
      ~capture_env:[ "MUNIN_NONEXISTENT_9999" ]
      ()
  in
  Session.finish session ();
  let prov = Run.provenance (Session.run session) in
  equal ~msg:"missing env omitted" (list (pair string string)) [] prov.env

let test_explicit_env () =
  with_temp_dir @@ fun root ->
  let session =
    Session.start ~root ~experiment:"exp" ~env:[ ("KEY", "value") ] ()
  in
  Session.finish session ();
  let prov = Run.provenance (Session.run session) in
  equal ~msg:"explicit env"
    (list (pair string string))
    [ ("KEY", "value") ]
    prov.env

let provenance_tests =
  [
    test "all fields round-trip" test_provenance_fields;
    test "capture_env" test_capture_env;
    test "capture_env missing var" test_capture_env_missing;
    test "explicit env" test_explicit_env;
  ]

(* Run loading *)

let test_run_load () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  let id = Run.id (Session.run session) in
  is_some ~msg:"load existing" (Run.load ~root ~experiment:"exp" ~id)

let test_run_load_missing () =
  with_temp_dir @@ fun root ->
  let _store = Store.open_ ~root () in
  is_none ~msg:"load missing"
    (Run.load ~root ~experiment:"exp" ~id:"nonexistent")

let test_run_list_sorted_descending () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp" () in
  Session.finish s2 ();
  let runs = Run.list ~root ~experiment:"exp" () in
  equal ~msg:"count" int 2 (List.length runs);
  let ids = List.map Run.id runs in
  (* Sorted descending by id *)
  is_true ~msg:"descending order"
    (String.compare (List.nth ids 0) (List.nth ids 1) > 0)

let test_run_list_filter_status () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp" () in
  Session.finish ~status:`failed s2 ();
  equal ~msg:"finished only" int 1
    (List.length (Run.list ~root ~experiment:"exp" ~status:`finished ()));
  equal ~msg:"failed only" int 1
    (List.length (Run.list ~root ~experiment:"exp" ~status:`failed ()))

let test_run_list_filter_tag () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp" ~tags:[ "train" ] () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp" ~tags:[ "eval" ] () in
  Session.finish s2 ();
  equal ~msg:"train tagged" int 1
    (List.length (Run.list ~root ~experiment:"exp" ~tag:"train" ()));
  equal ~msg:"eval tagged" int 1
    (List.length (Run.list ~root ~experiment:"exp" ~tag:"eval" ()))

let test_run_list_filter_parent () =
  with_temp_dir @@ fun root ->
  let parent = Session.start ~root ~experiment:"exp" () in
  Session.finish parent ();
  let parent_run = Session.run parent in
  let child = Session.start ~root ~experiment:"exp" ~parent:parent_run () in
  Session.finish child ();
  let _other = Session.start ~root ~experiment:"exp" () in
  Session.finish _other ();
  equal ~msg:"one child" int 1
    (List.length
       (Run.list ~root ~experiment:"exp" ~parent:(Run.id parent_run) ()))

let test_run_children () =
  with_temp_dir @@ fun root ->
  let parent = Session.start ~root ~experiment:"exp" () in
  Session.finish parent ();
  let parent_run = Session.run parent in
  let c1 = Session.start ~root ~experiment:"exp" ~parent:parent_run () in
  Session.finish c1 ();
  let c2 = Session.start ~root ~experiment:"exp" ~parent:parent_run () in
  Session.finish c2 ();
  let children = Run.children parent_run in
  equal ~msg:"two children" int 2 (List.length children);
  List.iter
    (fun child ->
      equal ~msg:"parent_id" (option string)
        (Some (Run.id parent_run))
        (Run.parent_id child))
    children

let test_run_name () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp" ~name:"baseline" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp" () in
  Session.finish s2 ();
  equal ~msg:"named" (option string) (Some "baseline")
    (Run.name (Session.run s1));
  equal ~msg:"unnamed" (option string) None (Run.name (Session.run s2))

let test_experiment_name () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"my-exp" () in
  Session.finish session ();
  equal ~msg:"experiment name" string "my-exp"
    (Run.experiment_name (Session.run session))

let run_loading =
  [
    test "load existing" test_run_load;
    test "load missing" test_run_load_missing;
    test "list sorted descending" test_run_list_sorted_descending;
    test "list filter status" test_run_list_filter_status;
    test "list filter tag" test_run_list_filter_tag;
    test "list filter parent" test_run_list_filter_parent;
    test "children" test_run_children;
    test "name" test_run_name;
    test "experiment_name" test_experiment_name;
  ]

(* Artifacts *)

let test_file_artifact () =
  with_temp_dir @@ fun root ->
  let src = make_file root "weights.bin" "model weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  let artifact =
    Session.log_artifact session ~name:"model" ~kind:`checkpoint ~path:src ()
  in
  Session.finish session ();
  equal ~msg:"name" string "model" (Artifact.name artifact);
  equal ~msg:"version" string "v1" (Artifact.version artifact);
  is_true ~msg:"kind" (Artifact.kind artifact = `checkpoint);
  is_true ~msg:"payload file" (Artifact.payload artifact = `file);
  is_true ~msg:"size positive" (Artifact.size_bytes artifact > 0);
  is_true ~msg:"path exists" (Sys.file_exists (Artifact.path artifact));
  is_true ~msg:"digest is sha256 hex"
    (String.length (Artifact.digest artifact) = 64)

let test_dir_artifact () =
  with_temp_dir @@ fun root ->
  let dir =
    make_dir root "dataset" [ ("a.txt", "data a"); ("b.txt", "data b") ]
  in
  let session = Session.start ~root ~experiment:"exp" () in
  let artifact =
    Session.log_artifact session ~name:"data" ~kind:`dataset ~path:dir ()
  in
  Session.finish session ();
  is_true ~msg:"payload dir" (Artifact.payload artifact = `dir);
  is_true ~msg:"kind dataset" (Artifact.kind artifact = `dataset);
  is_true ~msg:"blob dir exists" (Sys.file_exists (Artifact.path artifact))

let test_artifact_versioning () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "v1 weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  let a1 =
    Session.log_artifact session ~name:"model" ~kind:`model ~path:src ()
  in
  write_text src "v2 weights";
  let a2 =
    Session.log_artifact session ~name:"model" ~kind:`model ~path:src ()
  in
  Session.finish session ();
  equal ~msg:"first version" string "v1" (Artifact.version a1);
  equal ~msg:"second version" string "v2" (Artifact.version a2)

let test_artifact_aliases () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  let artifact =
    Session.log_artifact session ~name:"model" ~kind:`model ~path:src
      ~aliases:[ "best"; "latest" ] ()
  in
  Session.finish session ();
  is_true ~msg:"has best alias" (Artifact.has_alias artifact "best");
  is_true ~msg:"has latest alias" (Artifact.has_alias artifact "latest");
  is_false ~msg:"no nope alias" (Artifact.has_alias artifact "nope")

let test_artifact_alias_resolution () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  ignore
    (Session.log_artifact session ~name:"model" ~kind:`model ~path:src
       ~aliases:[ "latest" ] ());
  write_text src "v2 weights";
  ignore
    (Session.log_artifact session ~name:"model" ~kind:`model ~path:src
       ~aliases:[ "latest" ] ());
  Session.finish session ();
  let resolved = Artifact.load ~root ~name:"model" ~version:"latest" in
  is_true ~msg:"alias resolves to v2"
    (match resolved with Some a -> Artifact.version a = "v2" | None -> false);
  let v1 = Artifact.load ~root ~name:"model" ~version:"v1" in
  is_some ~msg:"explicit v1 loads" v1

let test_artifact_metadata () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  let artifact =
    Session.log_artifact session ~name:"model" ~kind:`model ~path:src
      ~metadata:[ ("framework", `String "rune") ]
      ()
  in
  Session.finish session ();
  equal ~msg:"metadata count" int 1 (List.length (Artifact.metadata artifact))

let test_artifact_lineage () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "weights" in
  let producer = Session.start ~root ~experiment:"exp" ~name:"producer" () in
  let artifact =
    Session.log_artifact producer ~name:"model" ~kind:`model ~path:src ()
  in
  let consumer = Session.start ~root ~experiment:"exp" ~name:"consumer" () in
  Session.use_artifact consumer artifact;
  Session.finish producer ();
  Session.finish consumer ();
  let producer_run = Session.run producer in
  let consumer_run = Session.run consumer in
  equal ~msg:"producer_run_id" (option string)
    (Some (Run.id producer_run))
    (Artifact.producer_run_id artifact);
  (* Reload to see consumer linkage *)
  let reloaded =
    match Artifact.load ~root ~name:"model" ~version:"v1" with
    | Some a -> a
    | None -> failwith "missing artifact"
  in
  equal ~msg:"consumer_run_ids" (list string)
    [ Run.id consumer_run ]
    (Artifact.consumer_run_ids reloaded);
  equal ~msg:"output artifacts" int 1
    (List.length (Run.output_artifacts producer_run));
  equal ~msg:"input artifacts" int 1
    (List.length (Run.input_artifacts consumer_run))

let test_log_artifact_closed_raises () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  raises_failure "Munin.Session.log_artifact: closed session" (fun () ->
      ignore
        (Session.log_artifact session ~name:"model" ~kind:`model ~path:src ()))

let test_log_artifact_missing_path_raises () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  raises_invalid_arg
    "Munin.Session.log_artifact: path does not exist: /nonexistent/path"
    (fun () ->
      ignore
        (Session.log_artifact session ~name:"model" ~kind:`model
           ~path:"/nonexistent/path" ()))

let test_artifact_content_dedup () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "same content" in
  let session = Session.start ~root ~experiment:"exp" () in
  let a1 =
    Session.log_artifact session ~name:"model" ~kind:`model ~path:src ()
  in
  let a2 =
    Session.log_artifact session ~name:"backup" ~kind:`model ~path:src ()
  in
  Session.finish session ();
  equal ~msg:"same digest" string (Artifact.digest a1) (Artifact.digest a2)

let artifacts =
  [
    test "file artifact" test_file_artifact;
    test "directory artifact" test_dir_artifact;
    test "versioning" test_artifact_versioning;
    test "aliases" test_artifact_aliases;
    test "alias resolution" test_artifact_alias_resolution;
    test "metadata" test_artifact_metadata;
    test "lineage" test_artifact_lineage;
    test "closed session raises" test_log_artifact_closed_raises;
    test "missing path raises" test_log_artifact_missing_path_raises;
    test "content dedup" test_artifact_content_dedup;
  ]

(* Store *)

let test_store_open () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  equal ~msg:"root" string root (Store.root store);
  is_true ~msg:"experiments dir exists"
    (Sys.file_exists (Filename.concat root "experiments"));
  is_true ~msg:"artifacts dir exists"
    (Sys.file_exists (Filename.concat root "artifacts"))

let test_store_list_experiments () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"mnist" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"cifar" () in
  Session.finish s2 ();
  let store = Store.open_ ~root () in
  let exps = Store.list_experiments store in
  equal ~msg:"two experiments" int 2 (List.length exps);
  is_true ~msg:"has mnist" (List.mem "mnist" exps);
  is_true ~msg:"has cifar" (List.mem "cifar" exps)

let test_store_list_runs_all () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp1" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp2" () in
  Session.finish s2 ();
  let store = Store.open_ ~root () in
  equal ~msg:"all runs" int 2 (List.length (Store.list_runs store ()))

let test_store_list_runs_experiment () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp1" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp2" () in
  Session.finish s2 ();
  let store = Store.open_ ~root () in
  equal ~msg:"exp1 runs" int 1
    (List.length (Store.list_runs store ~experiment:"exp1" ()));
  equal ~msg:"exp2 runs" int 1
    (List.length (Store.list_runs store ~experiment:"exp2" ()))

let test_store_find_run () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  let id = Run.id (Session.run session) in
  let store = Store.open_ ~root () in
  is_some ~msg:"found" (Store.find_run store id)

let test_store_find_run_missing () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  is_none ~msg:"not found" (Store.find_run store "nonexistent")

let test_store_latest_run () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" ~name:"only" () in
  Session.finish session ();
  let store = Store.open_ ~root () in
  let latest = Store.latest_run store () in
  is_true ~msg:"latest returns the run"
    (match latest with Some run -> Run.name run = Some "only" | None -> false)

let test_store_latest_run_with_filter () =
  with_temp_dir @@ fun root ->
  let s1 =
    Session.start ~root ~experiment:"exp" ~tags:[ "train" ] ~name:"a" ()
  in
  Session.finish s1 ();
  Unix.sleepf 0.01;
  let s2 =
    Session.start ~root ~experiment:"exp" ~tags:[ "eval" ] ~name:"b" ()
  in
  Session.finish s2 ();
  let store = Store.open_ ~root () in
  let latest = Store.latest_run store ~tag:"train" () in
  is_true ~msg:"latest with tag filter"
    (match latest with Some run -> Run.name run = Some "a" | None -> false)

let test_store_delete_run () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  let run = Session.run session in
  let store = Store.open_ ~root () in
  is_true ~msg:"run dir exists before" (Sys.file_exists (Run.dir run));
  Store.delete_run store run;
  is_false ~msg:"run dir removed" (Sys.file_exists (Run.dir run))

let test_store_delete_run_cleans_experiment () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"temp" () in
  Session.finish session ();
  let run = Session.run session in
  let store = Store.open_ ~root () in
  Store.delete_run store run;
  let exp_dir = Filename.concat (Filename.concat root "experiments") "temp" in
  is_false ~msg:"experiment dir removed" (Sys.file_exists exp_dir)

let test_store_gc () =
  with_temp_dir @@ fun root ->
  (* Use directory artifact so blob is a directory (gc uses list_dirs) *)
  let dir = make_dir root "dataset" [ ("a.txt", "data") ] in
  let session = Session.start ~root ~experiment:"exp" () in
  let artifact =
    Session.log_artifact session ~name:"data" ~kind:`dataset ~path:dir ()
  in
  Session.finish session ();
  let store = Store.open_ ~root () in
  (* Blob is referenced, gc should remove 0 *)
  let removed_before = Store.gc store in
  equal ~msg:"no unreferenced before" int 0 removed_before;
  is_true ~msg:"blob exists" (Sys.file_exists (Artifact.path artifact));
  (* Remove artifact manifest to make blob unreferenced *)
  let version_dir =
    Filename.concat
      (Filename.concat
         (Filename.concat (Filename.concat root "artifacts") "data")
         "versions")
      "v1"
  in
  ignore (Sys.command ("rm -rf " ^ Filename.quote version_dir));
  let removed_after = Store.gc store in
  equal ~msg:"one blob removed" int 1 removed_after

let test_store_find_artifact () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  ignore (Session.log_artifact session ~name:"model" ~kind:`model ~path:src ());
  Session.finish session ();
  let store = Store.open_ ~root () in
  is_some ~msg:"found" (Store.find_artifact store ~name:"model" ~version:"v1")

let test_store_list_artifacts () =
  with_temp_dir @@ fun root ->
  let src = make_file root "model.bin" "weights" in
  let session = Session.start ~root ~experiment:"exp" () in
  ignore (Session.log_artifact session ~name:"model" ~kind:`model ~path:src ());
  ignore (Session.log_artifact session ~name:"data" ~kind:`dataset ~path:src ());
  Session.finish session ();
  let store = Store.open_ ~root () in
  equal ~msg:"all artifacts" int 2 (List.length (Store.list_artifacts store ()));
  equal ~msg:"filter by kind" int 1
    (List.length (Store.list_artifacts store ~kind:`model ()));
  equal ~msg:"filter by name" int 1
    (List.length (Store.list_artifacts store ~name:"data" ()))

let store_tests =
  [
    test "open creates dirs" test_store_open;
    test "list_experiments" test_store_list_experiments;
    test "list_runs all" test_store_list_runs_all;
    test "list_runs by experiment" test_store_list_runs_experiment;
    test "find_run" test_store_find_run;
    test "find_run missing" test_store_find_run_missing;
    test "latest_run" test_store_latest_run;
    test "latest_run with filter" test_store_latest_run_with_filter;
    test "delete_run" test_store_delete_run;
    test "delete_run cleans experiment" test_store_delete_run_cleans_experiment;
    test "gc" test_store_gc;
    test "find_artifact" test_store_find_artifact;
    test "list_artifacts" test_store_list_artifacts;
  ]

(* Run monitor *)

let test_monitor_poll () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "train/loss" 1.0;
  Session.log_metric session ~step:2 "train/loss" 0.5;
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  let metrics = Run_monitor.metrics monitor in
  equal ~msg:"one key" int 1 (List.length metrics);
  let latest = List.assoc "train/loss" metrics in
  equal ~msg:"latest step" int 2 latest.step;
  Run_monitor.close monitor;
  Session.finish session ()

let test_monitor_incremental () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  equal ~msg:"one point after first poll" int 1
    (List.length (Run_monitor.history monitor "x"));
  Session.log_metric session ~step:2 "x" 2.0;
  Run_monitor.poll monitor;
  equal ~msg:"two points after second poll" int 2
    (List.length (Run_monitor.history monitor "x"));
  Run_monitor.close monitor;
  Session.finish session ()

let test_monitor_history_chronological () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 3.0;
  Session.log_metric session ~step:2 "x" 1.0;
  Session.log_metric session ~step:3 "x" 2.0;
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  let history = Run_monitor.history monitor "x" in
  let steps = List.map fst history in
  equal ~msg:"chronological steps" (list int) [ 1; 2; 3 ] steps;
  Run_monitor.close monitor;
  Session.finish session ()

let test_monitor_metric_defs () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "loss" ~summary:`Min ~goal:`Minimize ();
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  let defs = Run_monitor.metric_defs monitor in
  equal ~msg:"one def" int 1 (List.length defs);
  let def = List.assoc "loss" defs in
  is_true ~msg:"summary min" (def.summary = `Min);
  Run_monitor.close monitor;
  Session.finish session ()

let test_monitor_best_minimize () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "loss" ~goal:`Minimize ();
  Session.log_metric session ~step:1 "loss" 1.0;
  Session.log_metric session ~step:2 "loss" 0.3;
  Session.log_metric session ~step:3 "loss" 0.7;
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  let best = Run_monitor.best monitor "loss" in
  is_true ~msg:"best is 0.3"
    (match best with Some m -> m.value = 0.3 | None -> false);
  Run_monitor.close monitor;
  Session.finish session ()

let test_monitor_best_maximize () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "acc" ~goal:`Maximize ();
  Session.log_metric session ~step:1 "acc" 0.5;
  Session.log_metric session ~step:2 "acc" 0.9;
  Session.log_metric session ~step:3 "acc" 0.7;
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  let best = Run_monitor.best monitor "acc" in
  is_true ~msg:"best is 0.9"
    (match best with Some m -> m.value = 0.9 | None -> false);
  Run_monitor.close monitor;
  Session.finish session ()

let test_monitor_best_loss_heuristic () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "train/loss" 1.0;
  Session.log_metric session ~step:2 "train/loss" 0.2;
  Session.log_metric session ~step:3 "train/loss" 0.5;
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  let best = Run_monitor.best monitor "train/loss" in
  is_true ~msg:"loss heuristic picks minimum"
    (match best with Some m -> m.value = 0.2 | None -> false);
  Run_monitor.close monitor;
  Session.finish session ()

let test_monitor_live_status () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  is_true ~msg:"live during session" (Run_monitor.live_status monitor = `Live);
  Session.finish session ();
  Run_monitor.poll monitor;
  is_true ~msg:"done after finish"
    (match Run_monitor.live_status monitor with `Done _ -> true | _ -> false);
  Run_monitor.close monitor

let test_monitor_best_nonexistent () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  let run = Session.run session in
  let monitor = Run_monitor.start run in
  Run_monitor.poll monitor;
  is_none ~msg:"no best for missing key" (Run_monitor.best monitor "nope");
  Run_monitor.close monitor;
  Session.finish session ()

let run_monitor =
  [
    test "poll reads metrics" test_monitor_poll;
    test "incremental polling" test_monitor_incremental;
    test "history chronological" test_monitor_history_chronological;
    test "metric_defs" test_monitor_metric_defs;
    test "best minimize" test_monitor_best_minimize;
    test "best maximize" test_monitor_best_maximize;
    test "best loss heuristic" test_monitor_best_loss_heuristic;
    test "live_status" test_monitor_live_status;
    test "best nonexistent key" test_monitor_best_nonexistent;
  ]

(* Robustness *)

let test_partial_log () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  let run = Session.run session in
  (* Append truncated JSON *)
  let oc =
    open_out_gen
      [ Open_append; Open_creat ]
      0o644
      (Filename.concat (Run.dir run) "events.jsonl")
  in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc "{\"type\":\"metric\"");
  let run = Session.run session in
  equal ~msg:"partial line ignored" int 1
    (List.length (Run.metric_history run "x"));
  Session.finish session ()

let test_malformed_line () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  let run = Session.run session in
  let oc =
    open_out_gen
      [ Open_append; Open_creat ]
      0o644
      (Filename.concat (Run.dir run) "events.jsonl")
  in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc "this is not json at all\n");
  let run = Session.run session in
  equal ~msg:"malformed line skipped" int 1
    (List.length (Run.metric_history run "x"));
  Session.finish session ()

let test_empty_event_log () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  let run = Session.run session in
  (* No events logged *)
  equal ~msg:"no metrics" int 0 (List.length (Run.metric_keys run));
  equal ~msg:"no tags" (list string) [] (Run.tags run);
  is_true ~msg:"running" (Run.status run = `running);
  Session.finish session ()

let test_mixed_valid_invalid () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_metric session ~step:1 "x" 1.0;
  let run = Session.run session in
  let oc =
    open_out_gen
      [ Open_append; Open_creat ]
      0o644
      (Filename.concat (Run.dir run) "events.jsonl")
  in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      output_string oc "garbage\n";
      output_string oc "{\"bad\":true}\n");
  Session.log_metric session ~step:2 "x" 2.0;
  let run = Session.run session in
  equal ~msg:"only valid metrics" int 2
    (List.length (Run.metric_history run "x"));
  Session.finish session ()

let robustness =
  [
    test "partial log" test_partial_log;
    test "malformed line" test_malformed_line;
    test "empty event log" test_empty_event_log;
    test "mixed valid and invalid" test_mixed_valid_invalid;
  ]

(* Auto-computed summaries *)

let summary_float run key =
  match Run.find_summary run key with
  | Some v -> Value.to_float v
  | None -> None

let test_auto_summary_min () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "loss" ~summary:`Min ();
  Session.log_metric session ~step:1 "loss" 1.0;
  Session.log_metric session ~step:2 "loss" 0.3;
  Session.log_metric session ~step:3 "loss" 0.7;
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"min summary"
    (option (float 0.0))
    (Some 0.3) (summary_float run "loss")

let test_auto_summary_max () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "acc" ~summary:`Max ();
  Session.log_metric session ~step:1 "acc" 0.5;
  Session.log_metric session ~step:2 "acc" 0.9;
  Session.log_metric session ~step:3 "acc" 0.7;
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"max summary"
    (option (float 0.0))
    (Some 0.9) (summary_float run "acc")

let test_auto_summary_mean () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "x" ~summary:`Mean ();
  Session.log_metric session ~step:1 "x" 1.0;
  Session.log_metric session ~step:2 "x" 2.0;
  Session.log_metric session ~step:3 "x" 3.0;
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"mean summary"
    (option (float 0.0))
    (Some 2.0) (summary_float run "x")

let test_auto_summary_last () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "x" ~summary:`Last ();
  Session.log_metric session ~step:1 "x" 1.0;
  Session.log_metric session ~step:2 "x" 2.0;
  Session.log_metric session ~step:3 "x" 3.0;
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"last summary"
    (option (float 0.0))
    (Some 3.0) (summary_float run "x")

let test_auto_summary_none () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "x" ~summary:`None ();
  Session.log_metric session ~step:1 "x" 1.0;
  Session.finish session ();
  let run = Session.run session in
  is_none ~msg:"no auto-summary" (Run.find_summary run "x")

let test_explicit_summary_wins () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.define_metric session "loss" ~summary:`Min ();
  Session.log_metric session ~step:1 "loss" 1.0;
  Session.log_metric session ~step:2 "loss" 0.3;
  Session.set_summary session [ ("loss", `Float 999.0) ];
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"explicit wins"
    (option (float 0.0))
    (Some 999.0) (summary_float run "loss")

let auto_summaries =
  [
    test "auto-summary min" test_auto_summary_min;
    test "auto-summary max" test_auto_summary_max;
    test "auto-summary mean" test_auto_summary_mean;
    test "auto-summary last" test_auto_summary_last;
    test "auto-summary none" test_auto_summary_none;
    test "explicit summary wins" test_explicit_summary_wins;
  ]

(* Grouping *)

let test_group_round_trip () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" ~group:"sweep-lr" () in
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"group" (option string) (Some "sweep-lr") (Run.group run)

let test_group_none_by_default () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  equal ~msg:"no group" (option string) None (Run.group (Session.run session))

let test_run_list_filter_group () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp" ~group:"a" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp" ~group:"b" () in
  Session.finish s2 ();
  let s3 = Session.start ~root ~experiment:"exp" () in
  Session.finish s3 ();
  equal ~msg:"group a" int 1
    (List.length (Run.list ~root ~experiment:"exp" ~group:"a" ()));
  equal ~msg:"group b" int 1
    (List.length (Run.list ~root ~experiment:"exp" ~group:"b" ()));
  equal ~msg:"all" int 3 (List.length (Run.list ~root ~experiment:"exp" ()))

let test_store_list_runs_group () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp" ~group:"sweep" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp" () in
  Session.finish s2 ();
  let store = Store.open_ ~root () in
  equal ~msg:"sweep only" int 1
    (List.length (Store.list_runs store ~group:"sweep" ()))

let test_store_latest_run_group () =
  with_temp_dir @@ fun root ->
  let s1 = Session.start ~root ~experiment:"exp" ~group:"a" () in
  Session.finish s1 ();
  let s2 = Session.start ~root ~experiment:"exp" ~group:"b" () in
  Session.finish s2 ();
  let store = Store.open_ ~root () in
  let latest = Store.latest_run store ~group:"a" () in
  is_true ~msg:"latest in group a"
    (match latest with Some r -> Run.group r = Some "a" | None -> false)

let grouping =
  [
    test "group round-trip" test_group_round_trip;
    test "group none by default" test_group_none_by_default;
    test "run list filter group" test_run_list_filter_group;
    test "store list_runs group" test_store_list_runs_group;
    test "store latest_run group" test_store_latest_run_group;
  ]

(* Media *)

let test_log_media_copies_file () =
  with_temp_dir @@ fun root ->
  let src = make_file root "pred.png" "image data" in
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_media session ~step:1 ~key:"predictions" ~kind:`Image ~path:src;
  Session.finish session ();
  let run = Session.run session in
  let entries = Run.media_history run "predictions" in
  equal ~msg:"one entry" int 1 (List.length entries);
  let entry = List.hd entries in
  is_true ~msg:"file exists" (Sys.file_exists entry.path);
  is_true ~msg:"kind is image" (entry.kind = `Image);
  equal ~msg:"step" int 1 entry.step

let test_log_media_nested_key () =
  with_temp_dir @@ fun root ->
  let src = make_file root "img.png" "pixels" in
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_media session ~step:5 ~key:"train/predictions" ~kind:`Image
    ~path:src;
  Session.finish session ();
  let run = Session.run session in
  let entries = Run.media_history run "train/predictions" in
  equal ~msg:"one entry" int 1 (List.length entries);
  let entry = List.hd entries in
  is_true ~msg:"file in subdir"
    (let parts = String.split_on_char '/' entry.path in
     List.exists (String.equal "train") parts)

let test_log_media_multiple_steps () =
  with_temp_dir @@ fun root ->
  let src = make_file root "img.png" "pixels" in
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_media session ~step:1 ~key:"pred" ~kind:`Image ~path:src;
  Session.log_media session ~step:5 ~key:"pred" ~kind:`Image ~path:src;
  Session.log_media session ~step:10 ~key:"pred" ~kind:`Image ~path:src;
  Session.finish session ();
  let run = Session.run session in
  let entries = Run.media_history run "pred" in
  equal ~msg:"three entries" int 3 (List.length entries);
  let steps = List.map (fun (e : Run.media_entry) -> e.step) entries in
  equal ~msg:"chronological" (list int) [ 1; 5; 10 ] steps

let test_log_media_missing_path_raises () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  raises_invalid_arg
    "Munin.Session.log_media: path does not exist: /no/such/file" (fun () ->
      Session.log_media session ~step:1 ~key:"x" ~kind:`File
        ~path:"/no/such/file");
  Session.finish session ()

let test_log_media_closed_ignored () =
  with_temp_dir @@ fun root ->
  let src = make_file root "img.png" "pixels" in
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  Session.log_media session ~step:1 ~key:"pred" ~kind:`Image ~path:src;
  let run = Session.run session in
  equal ~msg:"no media" int 0 (List.length (Run.media_keys run))

let test_log_table () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_table session ~step:1 ~key:"confusion" ~columns:[ "cat"; "dog" ]
    ~rows:[ [ `Int 90; `Int 10 ]; [ `Int 5; `Int 95 ] ];
  Session.finish session ();
  let run = Session.run session in
  let entries = Run.media_history run "confusion" in
  equal ~msg:"one entry" int 1 (List.length entries);
  let entry = List.hd entries in
  is_true ~msg:"kind is table" (entry.kind = `Table);
  is_true ~msg:"json file exists" (Sys.file_exists entry.path)

let test_media_keys_sorted () =
  with_temp_dir @@ fun root ->
  let src = make_file root "f.bin" "data" in
  let session = Session.start ~root ~experiment:"exp" () in
  Session.log_media session ~step:1 ~key:"z/output" ~kind:`File ~path:src;
  Session.log_media session ~step:1 ~key:"a/input" ~kind:`File ~path:src;
  Session.log_media session ~step:1 ~key:"m/middle" ~kind:`File ~path:src;
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"sorted" (list string)
    [ "a/input"; "m/middle"; "z/output" ]
    (Run.media_keys run)

let test_media_empty_history () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  Session.finish session ();
  let run = Session.run session in
  equal ~msg:"no media keys" (list string) [] (Run.media_keys run);
  equal ~msg:"empty history" int 0
    (List.length (Run.media_history run "nonexistent"))

let media =
  [
    test "log_media copies file" test_log_media_copies_file;
    test "log_media nested key" test_log_media_nested_key;
    test "log_media multiple steps" test_log_media_multiple_steps;
    test "log_media missing path raises" test_log_media_missing_path_raises;
    test "log_media closed ignored" test_log_media_closed_ignored;
    test "log_table" test_log_table;
    test "media_keys sorted" test_media_keys_sorted;
    test "media empty history" test_media_empty_history;
  ]

(* System monitor *)

let test_system_monitor_logs_metrics () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  let monitor = Munin_sys.start session ~interval:0.1 () in
  Thread.delay 0.35;
  Munin_sys.stop monitor;
  Session.finish session ();
  let run = Session.run session in
  let keys = Run.metric_keys run in
  is_true ~msg:"has sys/cpu_user" (List.mem "sys/cpu_user" keys);
  is_true ~msg:"has sys/mem_used_pct" (List.mem "sys/mem_used_pct" keys);
  is_true ~msg:"has sys/proc_mem_mb" (List.mem "sys/proc_mem_mb" keys)

let test_system_monitor_defines_metrics () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  let monitor = Munin_sys.start session ~interval:100.0 () in
  Munin_sys.stop monitor;
  Session.finish session ();
  let run = Session.run session in
  let defs = Run.metric_defs run in
  let has_def key =
    match List.assoc_opt key defs with
    | Some d -> d.summary = `Last
    | None -> false
  in
  is_true ~msg:"cpu_user def" (has_def "sys/cpu_user");
  is_true ~msg:"mem_used_pct def" (has_def "sys/mem_used_pct");
  is_true ~msg:"proc_mem_mb def" (has_def "sys/proc_mem_mb")

let test_system_monitor_stop_idempotent () =
  with_temp_dir @@ fun root ->
  let session = Session.start ~root ~experiment:"exp" () in
  let monitor = Munin_sys.start session ~interval:100.0 () in
  Munin_sys.stop monitor;
  Munin_sys.stop monitor;
  Session.finish session ()

let system_monitor_tests =
  [
    test "logs metrics" test_system_monitor_logs_metrics;
    test "defines metrics" test_system_monitor_defines_metrics;
    test "stop idempotent" test_system_monitor_stop_idempotent;
  ]

(* Suite *)

let suite =
  [
    group "Session lifecycle" lifecycle;
    group "Scalars" scalars;
    group "Metric definitions" metric_definitions;
    group "Metadata" metadata;
    group "Provenance" provenance_tests;
    group "Run loading" run_loading;
    group "Artifacts" artifacts;
    group "Store" store_tests;
    group "Run monitor" run_monitor;
    group "Robustness" robustness;
    group "Auto-computed summaries" auto_summaries;
    group "Grouping" grouping;
    group "Media" media;
    group "System monitor" system_monitor_tests;
  ]

let () = run "Munin" suite
