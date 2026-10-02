(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Munin

(* [Session.start] defaults its provenance to [Provenance.detect ()], which runs
   git twice: in this repository, most of the suite's time. Sessions here record
   [provenance] unless a test passes its own. *)
let provenance =
  {
    Provenance.command = [ "test_sys" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    git_commit = None;
    git_dirty = None;
    env = [];
  }

module Session = struct
  include Session

  let start ?(provenance = provenance) = start ~provenance
end

let rec rm_rf path =
  if Sys.is_directory path then begin
    Array.iter (fun e -> rm_rf (Filename.concat path e)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let with_temp_dir f =
  let base = Filename.temp_file "munin" "test" in
  Sys.remove base;
  Unix.mkdir base 0o755;
  Fun.protect ~finally:(fun () -> rm_rf base) (fun () -> f base)

let test_system_monitor_logs_metrics () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:0.1 session in
  let rec sampled () =
    if not (List.mem "sys/cpu_user" (Run.metric_keys (Session.run session)))
    then begin
      Thread.delay 0.01;
      sampled ()
    end
  in
  sampled ();
  Munin_sys.stop monitor;
  Session.finish session;
  let run = Session.run session in
  let keys = Run.metric_keys run in
  is_true ~msg:"has sys/cpu_user" (List.mem "sys/cpu_user" keys);
  is_true ~msg:"has sys/mem_used_pct" (List.mem "sys/mem_used_pct" keys);
  is_true ~msg:"has sys/proc_mem_mb" (List.mem "sys/proc_mem_mb" keys)

let test_system_monitor_defines_metrics () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:100.0 session in
  Munin_sys.stop monitor;
  Session.finish session;
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
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:100.0 session in
  Munin_sys.stop monitor;
  Munin_sys.stop monitor;
  Session.finish session

let system_monitor_tests =
  [
    test "logs metrics" test_system_monitor_logs_metrics;
    test "defines metrics" test_system_monitor_defines_metrics;
    test "stop idempotent" test_system_monitor_stop_idempotent;
  ]

let () = exit (run "Munin_sys" [ group "System monitor" system_monitor_tests ])
