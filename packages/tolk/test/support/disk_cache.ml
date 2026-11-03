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

let fresh () = Windtrap.temp_dir ~prefix:"disk_cache" ()

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
  let out = Windtrap.temp_file ~suffix:".out" ()
  and err = Windtrap.temp_file ~suffix:".err" () in
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
  (* in text mode: on Windows a child's lines end in CRLF *)
  let read file = In_channel.with_open_text file In_channel.input_all in
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

(* The entries are the files of the cache's directories, a table's in the one
   named by the digest of its name. *)
let entries ?table cachedb =
  let in_table path =
    match table with
    | None -> true
    | Some t ->
        String.starts_with
          ~prefix:(Digest.to_hex (Digest.string t))
          (Filename.basename (Filename.dirname path))
  in
  if not (Sys.file_exists cachedb) then []
  else
    Array.to_list (Sys.readdir cachedb)
    |> List.map (Filename.concat cachedb)
    |> List.filter Sys.is_directory
    |> List.concat_map files |> List.filter in_table

let damage ?table cachedb f =
  List.iter
    (fun path ->
      let contents = In_channel.with_open_bin path In_channel.input_all in
      Out_channel.with_open_bin path (fun oc -> output_string oc (f contents)))
    (entries ?table cachedb)

let truncated e = String.sub e 0 (String.length e / 2)

(* An entry is "<key length> <value length> <digest>\n<key><value>", the digest
   of the key and the value. *)
let header e = String.index e '\n' + 1

let parts e =
  let h = header e in
  let k = int_of_string (List.hd (String.split_on_char ' ' e)) in
  (String.sub e h k, String.sub e (h + k) (String.length e - h - k))

let entry key value =
  Printf.sprintf "%d %d %s\n%s%s" (String.length key) (String.length value)
    (Digest.to_hex (Digest.string (key ^ value)))
    key value

let not_a_graph e =
  let key, v = parts e in
  let n = String.length v in
  let last = String.rindex_from v (n - 2) '\n' in
  entry key (String.sub v 0 (last + 1) ^ String.make (n - last - 1) ' ')

let restated f e =
  let key, v = parts e in
  match String.split_on_char ' ' (String.sub e 0 (header e - 1)) with
  | _ :: _ :: rest ->
      let v = f v in
      let lengths = [ String.length key; String.length v ] in
      String.concat " " (List.map string_of_int lengths @ rest)
      ^ "\n" ^ key ^ v
  | _ -> invalid_arg "Disk_cache.restated: not an entry"

let of_another_build e =
  let key, v = parts e in
  entry (String.make 32 '0' ^ String.sub key 32 (String.length key - 32)) v
