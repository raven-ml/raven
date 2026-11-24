(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* No compile waits on another live process ({!Rig_metal.section-code}). Metal
   keeps the code it compiles in a cache directory per application bundle, whose
   files a process's first compile locks until the process exits. A helper
   process holds those files in the directory Metal gives this executable's
   bundle, then a process of its own builds a pipeline through rig and reports.
   The helper lets go only after that report, so a build that waits on it hangs
   until the test's limit ends the test. The executable links
   rig-metal-cache-tests.plist, which gives it a bundle and the directory of its
   own. The suite never locks or writes another program's cache, nor the one
   that processes without a bundle share, and each of its processes checks its
   bundle before it touches Metal. Each exits once its input ends, which the
   suite closes or which ends with the suite's process, and each test kills
   every process it started by its pid as it ends. Every run of the suite, from
   any tree, uses the bundle's one cache, so each test holds the suite's lock
   while it runs, and two runs take turns. *)

open Windtrap
module S = Rig_metal_support

external bundle_id : unit -> string = "rig_metal_cache_test_bundle"
external user_cache : unit -> string = "rig_metal_cache_test_user_cache"
external metal_first : unit -> bool = "rig_metal_cache_test_device"
external flock : Unix.file_descr -> bool = "rig_metal_cache_test_flock"

(* The bundle rig-metal-cache-tests.plist names. *)
let bundle = "rig-metal-cache-tests"

let mine () =
  let id = bundle_id () in
  if id <> bundle then
    failf "the process's bundle is %S, not %S: it would use another's cache" id
      bundle

(* The directory of this executable's bundle in the user's cache. *)
let bundle_dir () = Filename.concat (user_cache ()) bundle

(* Metal's directory for this executable's bundle. *)
let metal_dir () = Filename.concat (bundle_dir ()) "com.apple.metal"

(* [alone f] is [f ()] while this process holds the suite's lock, a file of the
   bundle's directory. It waits for the lock while another run of the suite
   holds it. *)
let alone f =
  let dir = bundle_dir () in
  (try Unix.mkdir dir 0o755 with Unix.Unix_error (EEXIST, _, _) -> ());
  let fd =
    Unix.openfile
      (Filename.concat dir "suite.lock")
      [ O_RDWR; O_CREAT; O_CLOEXEC ]
      0o644
  in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let rec lock () =
    try Unix.lockf fd F_LOCK 0 with Unix.Unix_error (EINTR, _, _) -> lock ()
  in
  lock ();
  f ()

(* The data files of Metal's shader cache in [metal_dir ()]. Metal keeps its
   functions in pairs of files, functions<n>.list and functions<n>.data. While
   another process holds every data file and no list, as here, a process's first
   compile waits until that process lets go. With the lists held too, Metal
   makes a new pair and does not wait. *)
let data_files () =
  let dir = metal_dir () in
  let entries d = try Sys.readdir d with Sys_error _ -> [||] in
  let data f =
    String.starts_with ~prefix:"functions" f && Filename.check_suffix f ".data"
  in
  Array.to_list (entries dir)
  |> List.concat_map (fun sub ->
      let sub = Filename.concat dir sub in
      if not (Sys.is_directory sub) then []
      else
        Array.to_list (entries sub)
        |> List.filter data
        |> List.map (Filename.concat sub))

(* How many files of [paths] this process locks, as Metal does, until it exits.
   A file another live process holds is skipped. *)
let hold paths =
  List.fold_left
    (fun n p ->
      let fd = Unix.openfile p [ O_RDWR; O_CLOEXEC ] 0 in
      if flock fd then n + 1
      else (
        Unix.close fd;
        n))
    0 paths

(* Builds a pipeline through rig: [Ok ()] once its entry is made. *)
let build () =
  let { S.g; _ } = S.open_ () in
  match Rig_metal.image g (S.fixture ~dir:"fixtures" "threadgroup") with
  | Error why -> Error why
  | Ok (Rig_edge.Place _) -> Error "Metal asked to place its code"
  | Ok (Loaded i) -> (
      match S.entry i "small" with
      | Entry _ -> Ok ()
      | Absent -> Error "the function is absent"
      | No_memory n -> Error (Printf.sprintf "no memory for %d bytes" n))

(* Processes of the suite *)

(* Ends the process once its input ends, whether the suite closed it or the
   suite's process died, even while the process waits on Metal. *)
let watch () =
  ignore (In_channel.input_all stdin : string);
  Unix._exit 0

(* A process of the suite, which reports one line on its output. [--hold] locks
   the data files of Metal's cache, reports how many, and lives until its input
   ends. [--build] builds a pipeline, reports [built] and exits. [--first] first
   makes a Metal device, before rig opens one, which leaves the process in
   Metal's cache for its bundle, then does as [--build]. *)
let child mode =
  let watcher = Thread.create watch () in
  let builds = mode <> "--hold" in
  let report () =
    mine ();
    if mode = "--first" && not (metal_first ()) then
      Error "Metal made no device"
    else if builds then Result.map (fun () -> "built") (build ())
    else Ok (string_of_int (hold (data_files ())))
  in
  match try report () with e -> Error (Printexc.to_string e) with
  | Error why ->
      prerr_endline why;
      exit 1
  | Ok line ->
      print_endline line;
      if not builds then Thread.join watcher;
      exit 0

(* The suite's side *)

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> "killed by " ^ Sys.signal_to_string n
  | Unix.WSTOPPED n -> "stopped by " ^ Sys.signal_to_string n

(* A process the suite started: its pid, the suite's end of its input and its
   output. *)
type process = {
  what : string;
  pid : int;
  input : Unix.file_descr;
  output : in_channel;
  mutable open_input : bool;
  mutable reaped : bool;
}

let rec reap pid =
  match Unix.waitpid [] pid with
  | _, st -> st
  | exception Unix.Unix_error (EINTR, _, _) -> reap pid

(* Ends [p]'s input, which ends [p] unless it builds. *)
let release p =
  if p.open_input then (
    p.open_input <- false;
    Unix.close p.input)

(* Ends [p]'s input and checks that [p] then exits 0. *)
let finish p =
  release p;
  let st = reap p.pid in
  p.reaped <- true;
  equal ~msg:(p.what ^ ", as it ends") string "exited 0" (status st)

(* The line [p] reports, however long it takes. *)
let report p =
  match In_channel.input_line p.output with
  | Some line -> line
  | None -> failf "%s: ended without a report" p.what

(* [with_process what mode f] is [f p] for [p] this executable run with [mode].
   However [f] ends, [p] is then killed by its pid unless it was reaped. *)
let with_process what mode f =
  let input_r, input = Unix.pipe ~cloexec:true () in
  let output, output_w = Unix.pipe ~cloexec:true () in
  let exe = Sys.executable_name in
  let pid =
    Fun.protect
      ~finally:(fun () ->
        Unix.close input_r;
        Unix.close output_w)
      (fun () ->
        Unix.create_process exe [| exe; mode |] input_r output_w Unix.stderr)
  in
  let output = Unix.in_channel_of_descr output in
  let p = { what; pid; input; output; open_input = true; reaped = false } in
  let stop () =
    release p;
    close_in_noerr p.output;
    if not p.reaped then (
      p.reaped <- true;
      Unix.kill p.pid Sys.sigkill;
      ignore (reap p.pid : Unix.process_status))
  in
  Fun.protect ~finally:stop (fun () -> f p)

let gpu () =
  if not (S.present ()) then
    skip ~reason:"the machine has no METAL GPU rig drives" ();
  mine ()

let built p =
  equal ~msg:p.what string "built" (report p);
  finish p

(* The helper holds the data files, the build reports, and only then does the
   helper let go. *)
let no_wait () =
  gpu ();
  alone @@ fun () ->
  with_process "a build in a process where Metal made a device first" "--first"
    built;
  with_process "a process that holds Metal's cache" "--hold" @@ fun helper ->
  let held =
    require_some ~msg:"the number of files held"
      (int_of_string_opt (report helper))
  in
  greater ~msg:"the data files of Metal's cache held" int ~than:0 held;
  with_process "a build while another process holds Metal's cache" "--build"
    built;
  finish helper

let () =
  (match Sys.argv with
  | [| _; ("--hold" | "--build" | "--first") as mode |] -> child mode
  | _ -> ());
  exit
    (run "rig_metal_cache"
       [ test "no compile waits on another live process" no_wait ])
