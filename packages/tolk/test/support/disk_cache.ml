let role = "TOLK_TEST_ROLE"

let play parts =
  match Sys.getenv_opt role with
  | None -> ()
  | Some part -> (
      match List.assoc_opt part parts with
      | Some f ->
          f ();
          exit 0
      | None -> failwith ("unknown part " ^ part))

let caches = ref 0

let fresh () =
  incr caches;
  Filename.concat (Sys.getcwd ()) (Printf.sprintf "disk_cache_%d" !caches)

type child = int * string * string

let start ?(env = []) ~cachedb part =
  let set = (role, part) :: ("CACHEDB", cachedb) :: env in
  let unchanged binding =
    match String.index_opt binding '=' with
    | Some i -> not (List.mem_assoc (String.sub binding 0 i) set)
    | None -> true
  in
  let environment =
    Array.of_list
      (List.map (fun (k, v) -> k ^ "=" ^ v) set
      @ List.filter unchanged (Array.to_list (Unix.environment ())))
  in
  let out = Filename.temp_file part ".out"
  and err = Filename.temp_file part ".err" in
  let fd file = Unix.openfile file [ Unix.O_WRONLY ] 0 in
  let out_fd = fd out and err_fd = fd err in
  let pid =
    Unix.create_process_env Sys.executable_name [| Sys.executable_name |]
      environment Unix.stdin out_fd err_fd
  in
  Unix.close out_fd;
  Unix.close err_fd;
  (pid, out, err)

let finish (pid, out, err) =
  let _, status = Unix.waitpid [] pid in
  let read file = In_channel.with_open_bin file In_channel.input_all in
  match status with Unix.WEXITED 0 -> Ok (read out) | _ -> Error (read err)

let child ?env ~cachedb part = finish (start ?env ~cachedb part)

let contains s part =
  let n = String.length part in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = part || at (i + 1))
  in
  at 0

let outcome =
  Windtrap.Testable.make
    ~pp:(fun ppf -> function
      | Ok out -> Format.fprintf ppf "Ok:@.%s" out
      | Error err -> Format.fprintf ppf "Error: %s" err)
    ~equal:(fun r0 r1 ->
      match (r0, r1) with
      | Ok o0, Ok o1 -> String.equal o0 o1
      | Error e0, Error e1 -> contains e0 e1 || contains e1 e0
      | _ -> false)

let rec files dir =
  Array.to_list (Sys.readdir dir)
  |> List.concat_map (fun f ->
      let path = Filename.concat dir f in
      if Sys.is_directory path then files path else [ path ])

(* A table's entries are in the directory named by the digest of its name. *)
let entries ?table cachedb =
  let in_table path =
    match table with
    | None -> true
    | Some t ->
        String.starts_with
          ~prefix:(Digest.to_hex (Digest.string t))
          (Filename.basename (Filename.dirname path))
  in
  if Sys.file_exists cachedb then List.filter in_table (files cachedb) else []

let damage ?table cachedb f =
  List.iter
    (fun path ->
      let contents = In_channel.with_open_bin path In_channel.input_all in
      Out_channel.with_open_bin path (fun oc -> output_string oc (f contents)))
    (entries ?table cachedb)

let truncated e = String.sub e 0 (String.length e / 2)

(* An entry is "<key length> <value length>\n<key><value>". *)
let header e = String.index e '\n' + 1

let not_a_graph e =
  let n = String.length e in
  let last = String.rindex_from e (n - 2) '\n' in
  String.sub e 0 (last + 1) ^ String.make (n - last - 1) ' '

let of_another_build e =
  let h = header e in
  String.sub e 0 h ^ String.make 32 '0'
  ^ String.sub e (h + 32) (String.length e - h - 32)
