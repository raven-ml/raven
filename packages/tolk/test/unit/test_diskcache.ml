(* Diskcache behavior tests.

   [Diskcache] resolves its cache directory from the environment once, at
   module initialization. To keep these tests hermetic, the parent process
   never touches the cache itself: every cache operation runs in a child
   process spawned from this executable with [XDG_CACHE_HOME] pointing at a
   per-run temporary directory. Children select a role via the
   [TOLK_DISKCACHE_ROLE] environment variable. *)

open Windtrap
open Tolk

let role_var = "TOLK_DISKCACHE_ROLE"
let table_var = "TOLK_DISKCACHE_TABLE"
let key_var = "TOLK_DISKCACHE_KEY"
let value_var = "TOLK_DISKCACHE_VALUE"
let fills_var = "TOLK_DISKCACHE_FILLS"
let size_var = "TOLK_DISKCACHE_SIZE"
let count_var = "TOLK_DISKCACHE_COUNT"

let getenv name =
  match Sys.getenv_opt name with
  | Some v -> v
  | None -> failwith (name ^ " not set")

(* Child roles. Each prints a single line and exits. *)

let child_put () =
  Diskcache.put ~table:(getenv table_var) ~key:(getenv key_var)
    (getenv value_var);
  print_string "ok"

let child_get () =
  match Diskcache.get ~table:(getenv table_var) ~key:(getenv key_var) with
  | Some (v : string) -> print_string ("hit:" ^ v)
  | None -> print_string "miss"

(* Repeatedly write a uniform payload and read the entry back. A read must
   observe either a miss (torn file rejected by [get]) or a complete payload
   written by exactly one of the concurrent writers. *)
let child_spam () =
  let table = getenv table_var in
  let key = getenv key_var in
  let fills = getenv fills_var in
  let size = int_of_string (getenv size_var) in
  let count = int_of_string (getenv count_var) in
  let fill = fills.[0] in
  let payload = String.make size fill in
  let valid v =
    String.length v = size
    && String.exists (fun c -> Char.equal c v.[0]) fills
    && String.for_all (fun c -> Char.equal c v.[0]) v
  in
  let torn = ref false in
  for _ = 1 to count do
    Diskcache.put ~table ~key payload;
    match Diskcache.get ~table ~key with
    | None -> ()
    | Some (v : string) -> if not (valid v) then torn := true
  done;
  print_string (if !torn then "torn" else "ok")

let run_role = function
  | "put" -> child_put ()
  | "get" -> child_get ()
  | "spam" -> child_spam ()
  | role -> failwith ("unknown role: " ^ role)

(* Parent-side driver. *)

let cache_root =
  lazy
    (let dir =
       Filename.concat
         (Filename.get_temp_dir_name ())
         (Printf.sprintf "tolk-diskcache-test-%d" (Unix.getpid ()))
     in
     Unix.mkdir dir 0o755;
     dir)

let child_env extra =
  let names = List.map fst extra in
  let keep binding =
    match String.index_opt binding '=' with
    | Some i -> not (List.mem (String.sub binding 0 i) names)
    | None -> true
  in
  let base =
    Unix.environment () |> Array.to_list |> List.filter keep
  in
  Array.of_list (base @ List.map (fun (k, v) -> k ^ "=" ^ v) extra)

let spawn_child extra =
  let env =
    child_env (("XDG_CACHE_HOME", Lazy.force cache_root) :: extra)
  in
  let out_read, out_write = Unix.pipe ~cloexec:false () in
  let pid =
    Unix.create_process_env Sys.executable_name
      [| Sys.executable_name |]
      env Unix.stdin out_write Unix.stderr
  in
  Unix.close out_write;
  (pid, out_read)

let finish_child (pid, out_read) =
  let buf = Buffer.create 64 in
  let chunk = Bytes.create 4096 in
  let rec drain () =
    let n = Unix.read out_read chunk 0 (Bytes.length chunk) in
    if n > 0 then begin
      Buffer.add_subbytes buf chunk 0 n;
      drain ()
    end
  in
  drain ();
  Unix.close out_read;
  let _, status = Unix.waitpid [] pid in
  (status, Buffer.contents buf)

let run_child extra =
  let status, output = finish_child (spawn_child extra) in
  (match status with
   | Unix.WEXITED 0 -> ()
   | _ -> fail "child process failed");
  output

(* Mirrors [Diskcache.cache_path] so the parent can inspect entries without
   loading the cache module's directory resolution. *)
let entry_path ~table ~key =
  Filename.concat
    (Filename.concat (Filename.concat (Lazy.force cache_root) "tolk") table)
    (Digest.to_hex (Digest.string key) ^ ".cache")

let read_entry ~table ~key =
  let ic = open_in_bin (entry_path ~table ~key) in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let version : int = Marshal.from_channel ic in
      let value : string = Marshal.from_channel ic in
      (version, value))

(* Tests. *)

let roundtrip_across_processes () =
  let extra = [ (table_var, "rt"); (key_var, "k") ] in
  equal string "ok" (run_child ((role_var, "put") :: (value_var, "hello") :: extra));
  equal string "hit:hello" (run_child ((role_var, "get") :: extra))

let missing_key_is_a_miss () =
  equal string "miss"
    (run_child [ (role_var, "get"); (table_var, "rt"); (key_var, "absent") ])

let corrupt_entry_is_a_miss () =
  let extra = [ (table_var, "corrupt"); (key_var, "k") ] in
  equal string "ok"
    (run_child ((role_var, "put") :: (value_var, "payload") :: extra));
  let path = entry_path ~table:"corrupt" ~key:"k" in
  (* Truncate the entry mid-payload. *)
  let size = (Unix.stat path).Unix.st_size in
  Unix.truncate path (size / 2);
  equal string "miss" (run_child ((role_var, "get") :: extra));
  (* Replace the entry with garbage bytes. *)
  let oc = open_out_bin path in
  output_string oc "not a marshalled entry";
  close_out oc;
  equal string "miss" (run_child ((role_var, "get") :: extra))

let concurrent_writers_never_tear () =
  let table = "spam" and key = "k" in
  let fills = "abcd" in
  let size = 1 lsl 20 and count = 20 in
  let extra fill =
    [
      (role_var, "spam");
      (table_var, table);
      (key_var, key);
      (fills_var, String.make 1 fill ^ fills);
      (size_var, string_of_int size);
      (count_var, string_of_int count);
    ]
  in
  let children =
    List.init (String.length fills) (fun i -> spawn_child (extra fills.[i]))
  in
  List.iter
    (fun child ->
      let status, output = finish_child child in
      (match status with
       | Unix.WEXITED 0 -> ()
       | _ -> fail "spam child failed");
      equal string ~msg:"child observed only complete entries" "ok" output)
    children;
  (* The final entry must be one writer's complete payload. *)
  let version, value = read_entry ~table ~key in
  equal int ~msg:"entry version" 1 version;
  equal int ~msg:"entry size" size (String.length value);
  is_true ~msg:"payload written by a single writer"
    (String.exists (fun c -> Char.equal c value.[0]) fills
     && String.for_all (fun c -> Char.equal c value.[0]) value);
  (* Renames leave no temporary files behind. *)
  let dir = Filename.dirname (entry_path ~table ~key) in
  let leftovers =
    Sys.readdir dir |> Array.to_list
    |> List.filter (fun f -> not (Filename.check_suffix f ".cache"))
  in
  equal (list string) ~msg:"no leftover temp files" [] leftovers

let () =
  match Sys.getenv_opt role_var with
  | Some role -> run_role role
  | None ->
      run "tolk.diskcache"
        [
          group "Diskcache"
            [
              test "round-trips a value across processes"
                roundtrip_across_processes;
              test "missing key is a miss" missing_key_is_a_miss;
              test "corrupt or truncated entry is a miss"
                corrupt_entry_is_a_miss;
              test "concurrent writers never tear an entry"
                concurrent_writers_never_tear;
            ];
        ]
