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
   until the test's limit ends the test. The suite also reads the directories
   that live processes of its own take. The executable links
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

(* The directory that holds the numbered directories of rig's processes. *)
let rig_dir () = Filename.concat (bundle_dir ()) "rig.metal"

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

(* [path] with no empty component, as Metal writes [//] into its own. *)
let normal path =
  String.split_on_char '/' path |> List.filter (( <> ) "") |> String.concat "/"

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

(* Opens a device, starts this executable with [--idle], a program that never
   touches Metal and lives until its input, this process's, ends, and reports
   Metal's directory for the process and the program's pid. *)
let spawn () =
  ignore (S.open_ () : S.t);
  let exe = Sys.executable_name in
  let idle =
    Unix.create_process exe [| exe; "--idle" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  Ok (Printf.sprintf "%s\n%d" (Rig_metal.shader_cache ()) idle)

(* A process of the suite, which reports one line on its output. [--hold] locks
   the data files of Metal's cache and reports how many. [--build] builds a
   pipeline and reports [built]. [--live] opens a device and reports Metal's
   directory for the process. [--first] and [--first-live] first make a Metal
   device, before rig opens one, which leaves the process in Metal's cache for
   its bundle, then do as [--build] and [--live]. [--spawn] does as {!spawn} and
   reports a second line. A build and a spawner exit once they report; the
   others live until their input ends. *)
let child mode =
  if mode = "--idle" then watch ();
  let watcher = Thread.create watch () in
  let first = mode = "--first" || mode = "--first-live" in
  let builds = mode = "--build" || mode = "--first" in
  let stays = not (builds || mode = "--spawn") in
  let report () =
    mine ();
    if first && not (metal_first ()) then Error "Metal made no device"
    else if mode = "--hold" then Ok (string_of_int (hold (data_files ())))
    else if builds then Result.map (fun () -> "built") (build ())
    else if mode = "--spawn" then spawn ()
    else (
      ignore (S.open_ () : S.t);
      Ok (Rig_metal.shader_cache ()))
  in
  match try report () with e -> Error (Printexc.to_string e) with
  | Error why ->
      prerr_endline why;
      exit 1
  | Ok line ->
      print_endline line;
      if stays then Thread.join watcher;
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

(* Checks that [p] exits 0, its input left as it is. *)
let exits p =
  let st = reap p.pid in
  p.reaped <- true;
  equal ~msg:(p.what ^ ", as it ends") string "exited 0" (status st)

(* Ends [p]'s input and checks that [p] then exits 0. *)
let finish p =
  release p;
  exits p

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

(* The number a directory of the processes' own is named by. *)
let slot dir =
  require_some
    ~msg:(dir ^ ", named by a number")
    (int_of_string_opt (Filename.basename dir))

(* The numbers of the directories two live processes take, checked to lie in
   [rig_dir ()]. *)
let two () =
  with_process "the first live process" "--live" @@ fun a ->
  with_process "the second live process" "--live" @@ fun b ->
  let dir_a = report a in
  let dir_b = report b in
  List.iter
    (fun d ->
      equal ~msg:(d ^ ", in rig.metal") string
        (normal (rig_dir ()))
        (normal (Filename.dirname d)))
    [ dir_a; dir_b ];
  finish a;
  finish b;
  (slot dir_a, slot dir_b)

let apart () =
  gpu ();
  alone @@ fun () ->
  let a, b = two () in
  not_equal ~msg:"their directories" int a b

(* Once both have exited, a third process takes the lesser of their numbers: the
   least that no live process holds. *)
let reused () =
  gpu ();
  alone @@ fun () ->
  let a, b = two () in
  with_process "a live process after the two" "--live" @@ fun c ->
  equal ~msg:"its directory's number" int (min a b) (slot (report c));
  finish c

(* A program that a process starts after its first open holds the process's
   directory while it lives, though it never touches Metal, and gives it back as
   it exits. The program writes nothing, so the spawner's output ends as the
   program exits. *)
let pinned () =
  gpu ();
  alone @@ fun () ->
  with_process "a process that starts a program, then exits" "--spawn"
  @@ fun spawner ->
  let k = slot (report spawner) in
  let program =
    require_some ~msg:"the pid of the program it started"
      (int_of_string_opt (report spawner))
  in
  let ended = ref false in
  let stop () =
    if not !ended then
      try Unix.kill program Sys.sigkill
      with Unix.Unix_error (ESRCH, _, _) -> ()
  in
  Fun.protect ~finally:stop @@ fun () ->
  exits spawner;
  with_process "a live process while the program lives" "--live" (fun p ->
      not_equal ~msg:"its directory's number" int k (slot (report p));
      finish p);
  release spawner;
  equal ~msg:"what the program wrote" string ""
    (In_channel.input_all spawner.output);
  ended := true;
  with_process "a live process once the program exited" "--live" @@ fun p ->
  equal ~msg:"its directory's number" int k (slot (report p));
  finish p

let refused () =
  gpu ();
  alone @@ fun () ->
  with_process "a process where Metal made a device first" "--first-live"
  @@ fun p ->
  equal ~msg:"its directory" string (normal (metal_dir ())) (normal (report p));
  finish p

(* The arguments that make this executable a process of the suite. *)
let modes =
  [
    "--hold";
    "--build";
    "--first";
    "--live";
    "--first-live";
    "--spawn";
    "--idle";
  ]

let () =
  (match Sys.argv with
  | [| _; mode |] when List.mem mode modes -> child mode
  | _ -> ());
  exit
    (run "rig_metal_cache"
       [
         test "no compile waits on another live process" no_wait;
         test "two live processes take two directories in rig.metal" apart;
         test "a directory a process held is taken again once it exits" reused;
         test "a program a process starts holds its directory while it lives"
           pinned;
         test "a process where Metal made a device first keeps Metal's cache"
           refused;
       ])
