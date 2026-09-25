(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Persistent jit compile cache, across processes.

   [Tolk.Diskcache] resolves its cache directory from the environment once, at
   module initialization, so every cache-touching scenario runs in a child
   process spawned from this executable with [XDG_CACHE_HOME] pointing at a
   per-scenario temporary directory (the pattern of tolk's test_diskcache).
   Children select a role via [RUNE_JITCACHE_ROLE], print results as hex floats
   on stdout (bit-exact comparison), and run with [RUNE_JIT_DEBUG=1] so the
   parent observes cache hits and misses on stderr. *)

open Windtrap
open Rune_test_support.Support

let role_var = "RUNE_JITCACHE_ROLE"

(* Child side *)

(* An elementwise chain plus a reduction: several kernels, one capture. *)
let capture =
  lazy
    (Nx.create f32 [| 6 |]
       (Array.init 6 (fun i -> float_of_int (i + 1) /. 11.0)))

let f x =
  let y = Nx.tanh (Nx.add (Nx.mul x x) x) in
  Nx.sum (Nx.add y (Lazy.force capture)) ~axes:[ 1 ]

let input () =
  Nx.create f32 [| 4; 6 |]
    (Array.init 24 (fun i -> float_of_int (i + 7) /. 3.0))

let print_result t =
  to_arr t |> Array.to_list
  |> List.map (Printf.sprintf "%h")
  |> String.concat " " |> print_endline

let child_once () = print_result (Rune.jit' f (input ()))

let child_twice () =
  print_result (Rune.jit' f (input ()));
  print_result (Rune.jit' f (input ()))

let child_pmap () =
  let x = input () in
  let expect = Rune.jit' f x in
  let g =
    Rune.pmap
      ~devices:[ Rune.device "CPU:1"; Rune.device "CPU:2" ]
      Nx.Ptree.(tensor @-> returns tensor)
      f
  in
  let got = g x in
  let e = to_arr expect and a = to_arr got in
  if e <> a then failwith "pmap result differs from jit";
  print_result got

let run_role = function
  | "once" -> child_once ()
  | "twice" -> child_twice ()
  | "pmap" -> child_pmap ()
  | role -> failwith ("unknown role: " ^ role)

(* Parent side *)

let fresh_dir =
  let counter = ref 0 in
  fun () ->
    incr counter;
    let dir =
      Filename.concat
        (Filename.get_temp_dir_name ())
        (Printf.sprintf "rune-jitcache-test-%d-%d" (Unix.getpid ()) !counter)
    in
    Unix.mkdir dir 0o755;
    dir

let child_env ?(unset = []) extra =
  let names = unset @ List.map fst extra in
  let keep binding =
    match String.index_opt binding '=' with
    | Some i -> not (List.mem (String.sub binding 0 i) names)
    | None -> true
  in
  let base = Unix.environment () |> Array.to_list |> List.filter keep in
  Array.of_list (base @ List.map (fun (k, v) -> k ^ "=" ^ v) extra)

let drain fd =
  let buf = Buffer.create 256 in
  let chunk = Bytes.create 4096 in
  let rec loop () =
    let n = Unix.read fd chunk 0 (Bytes.length chunk) in
    if n > 0 then begin
      Buffer.add_subbytes buf chunk 0 n;
      loop ()
    end
  in
  loop ();
  Unix.close fd;
  Buffer.contents buf

(* Run [exe] with the given role and cache dir; return (stdout, cache events).
   Cache events are the "rune.jit: compile cache <event> <key>" lines, reduced
   to their <event> word, in order. *)
(* A child's text-mode stdout ends lines with CRLF on Windows. *)
let strip_cr s = String.concat "" (String.split_on_char '\r' s)

let run_child ?(exe = Sys.executable_name) ?unset ?(extra = []) ~cache role =
  let env =
    child_env ?unset
      ([ ("XDG_CACHE_HOME", cache); (role_var, role); ("RUNE_JIT_DEBUG", "1") ]
      @ extra)
  in
  let out_read, out_write = Unix.pipe ~cloexec:false () in
  let err_read, err_write = Unix.pipe ~cloexec:false () in
  let pid =
    Unix.create_process_env exe [| exe |] env Unix.stdin out_write err_write
  in
  Unix.close out_write;
  Unix.close err_write;
  let out = strip_cr (drain out_read) in
  let errs = strip_cr (drain err_read) in
  let _, status = Unix.waitpid [] pid in
  (match status with
  | Unix.WEXITED 0 -> ()
  | _ -> fail (Printf.sprintf "child failed; stderr:\n%s" errs));
  let prefix = "rune.jit: compile cache " in
  let events =
    String.split_on_char '\n' errs
    |> List.filter_map (fun line ->
        if String.starts_with ~prefix line then
          let rest =
            String.sub line (String.length prefix)
              (String.length line - String.length prefix)
          in
          match String.index_opt rest ' ' with
          | Some i -> Some (String.sub rest 0 i)
          | None -> Some rest
        else None)
  in
  (out, events)

let table_dir cache = Filename.concat (Filename.concat cache "tolk") "rune_jit"

let entry_files cache =
  let dir = table_dir cache in
  if Sys.file_exists dir then
    Sys.readdir dir |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".cache")
    |> List.map (Filename.concat dir)
  else []

(* Tests *)

let cold_miss_then_warm_hit () =
  let cache = fresh_dir () in
  (* Cold process: the first closure misses and stores, the second closure (same
     trace, same process) hits; both produce identical bytes. *)
  let out, events = run_child ~cache "twice" in
  equal (list string) ~msg:"cold events" [ "miss"; "store"; "hit" ] events;
  (match String.split_on_char '\n' (String.trim out) with
  | [ a; b ] -> equal string ~msg:"hit matches miss bit-for-bit" a b
  | _ -> fail "expected two result lines");
  equal int ~msg:"one entry written" 1 (List.length (entry_files cache));
  (* Warm process: pure hit, same bytes as the cold run. *)
  let warm_out, warm_events = run_child ~cache "once" in
  equal (list string) ~msg:"warm events" [ "hit" ] warm_events;
  equal string ~msg:"warm output matches cold output"
    (List.hd (String.split_on_char '\n' (String.trim out)))
    (String.trim warm_out)

let corrupt_entry_is_a_miss_and_rewritten () =
  let cache = fresh_dir () in
  let cold_out, _ = run_child ~cache "once" in
  let path =
    match entry_files cache with [ p ] -> p | _ -> fail "expected one entry"
  in
  let oc = open_out_bin path in
  output_string oc "not a marshalled entry";
  close_out oc;
  let out, events = run_child ~cache "once" in
  equal (list string) ~msg:"corrupt entry misses and is rewritten"
    [ "miss"; "store" ] events;
  equal string ~msg:"output unchanged" (String.trim cold_out) (String.trim out);
  let ic = open_in_bin path in
  let size = in_channel_length ic in
  close_in ic;
  is_true ~msg:"entry rewritten" (size > String.length "not a marshalled entry")

(* The key digests the executable, so the same trace compiled by a different
   binary must miss. Append a byte to a copy of this executable: same code runs,
   different fingerprint. *)
let different_exe_fingerprint_is_a_miss () =
  let cache = fresh_dir () in
  ignore (run_child ~cache "once");
  let dir = fresh_dir () in
  let exe = Filename.concat dir "test_jit_cache_altered.exe" in
  let ic = open_in_bin Sys.executable_name in
  let n = in_channel_length ic in
  let bytes = really_input_string ic n in
  close_in ic;
  let oc = open_out_bin exe in
  output_string oc bytes;
  output_string oc "\x00";
  close_out oc;
  Unix.chmod exe 0o755;
  let _, events = run_child ~exe ~cache "once" in
  equal (list string) ~msg:"altered binary misses" [ "miss"; "store" ] events;
  equal int ~msg:"a second entry appears" 2 (List.length (entry_files cache))

let jitcache_zero_disables () =
  let cache = fresh_dir () in
  let _, events = run_child ~extra:[ ("JITCACHE", "0") ] ~cache "once" in
  equal (list string) ~msg:"no cache events" [] events;
  is_true ~msg:"no rune_jit table created"
    (not (Sys.file_exists (table_dir cache)));
  (* And a warmed cache is ignored when disabled. *)
  ignore (run_child ~cache "once");
  let _, events = run_child ~extra:[ ("JITCACHE", "0") ] ~cache "once" in
  equal (list string) ~msg:"warm cache ignored" [] events

(* A trace stored under one value of a setting that changes what it compiles to
   must not be served under another. The second child runs with [name] and
   [also] removed from this process's environment. *)
let misses_without ?(also = []) name value =
  let cache = fresh_dir () in
  let _, events = run_child ~extra:[ (name, value) ] ~cache "once" in
  equal (list string) ~msg:(name ^ " stores") [ "miss"; "store" ] events;
  let _, events = run_child ~unset:(name :: also) ~cache "once" in
  equal (list string) ~msg:(name ^ " unset misses") [ "miss"; "store" ] events

(* Profiling compiles queues with timestamps; LATE_ALLREDUCE is a setting
   scheduling reads, for when sharded traces are cached. *)
let schedule_settings_are_part_of_the_key () =
  misses_without ~also:[ "DEBUG" ] "PROFILE" "1";
  misses_without "LATE_ALLREDUCE" "0"

(* Each changes the kernels compiled from one schedule. *)
let program_settings_are_part_of_the_key () =
  misses_without "TC_SELECT" "0";
  misses_without "MV" "0";
  misses_without "DMC" "1"

let pmap_bails () =
  let cache = fresh_dir () in
  let _, events = run_child ~cache "pmap" in
  (* The single-device jit inside the role caches; the pmap compilations must
     not. One trace -> one miss/store pair and no more. *)
  equal (list string) ~msg:"only the jit trace touches the cache"
    [ "miss"; "store" ] events;
  equal int ~msg:"one entry (the jit trace)" 1 (List.length (entry_files cache))

let () =
  match Sys.getenv_opt role_var with
  | Some role -> run_role role
  | None ->
      run "rune.jit_cache"
        [
          group "persistent compile cache"
            [
              test "cold process misses, warm process hits, bytes identical"
                cold_miss_then_warm_hit;
              test "corrupt entry is a silent miss and is rewritten"
                corrupt_entry_is_a_miss_and_rewritten;
              test "a different executable fingerprint misses"
                different_exe_fingerprint_is_a_miss;
              test "JITCACHE=0 disables the cache and touches no disk"
                jitcache_zero_disables;
              test
                "a trace is not served under other queue or schedule settings"
                schedule_settings_are_part_of_the_key;
              test "a trace is not served under other program settings"
                program_settings_are_part_of_the_key;
              test "pmap compilations bail and still work" pmap_bails;
            ];
        ]
