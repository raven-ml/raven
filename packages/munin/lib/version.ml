(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  name : string;
  version : int;
  digest : string;
  path : string;
  size_bytes : int;
  metadata : (string * Value.t) list;
  run : string;
  step : int option;
  created_at : float;
}

(* Paths *)

let algorithm = "blake2b-256"
let blob_dir root = Filename.concat (Filename.concat root "blobs") algorithm
let lock_path root = Filename.concat root "lock"
let name_dir root name = Filename.concat (Filename.concat root "artifacts") name

let version_file root name n =
  Filename.concat (name_dir root name) (string_of_int n ^ ".json")

let names ~root =
  File.entries (Filename.concat root "artifacts") |> List.filter File.is_segment

(* "<n>.json", n a decimal from 1 without leading zeros. *)
let number_of_file f =
  match Filename.chop_suffix_opt ~suffix:".json" f with
  | Some n
    when n <> ""
         && n.[0] <> '0'
         && String.for_all (function '0' .. '9' -> true | _ -> false) n ->
      int_of_string_opt n
  | _ -> None

let numbers ~root name =
  File.entries (name_dir root name)
  |> List.filter_map number_of_file
  |> List.sort compare

(* Version files *)

(* [hex_of_digest d] is the 64 lowercase hexadecimal digits of the digest
   [d], as the format defines one, or [None]: nothing else names a blob, so
   a version file cannot point outside [blobs/]. *)
let hex_of_digest d =
  let prefix = algorithm ^ ":" in
  let n = String.length prefix in
  if
    String.length d = n + 64
    && String.starts_with ~prefix d
    && String.for_all
         (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
         (String.sub d n 64)
  then Some (String.sub d n 64)
  else None

let ( let* ) = Option.bind

let decode ~root ~name ~version s =
  match Json.of_string s with
  | Error _ -> `Missing
  | Ok j -> (
      match Json.mem "deleted" j with
      | Some (Json.Bool true) -> `Deleted
      | _ -> (
          let str k = Json.field k Json.string j in
          let v =
            let* format = Json.field "format" Json.int j in
            if format <> Store_format.version then None
            else
              let* created_at = Json.field "time" Json.float j in
              let* run = str "run" in
              let* digest = str "digest" in
              let* hex = hex_of_digest digest in
              let* size_bytes = Json.field "size" Json.int j in
              let* metadata = Json.field "metadata" Json.values j in
              let step = Json.field "step" Json.int j in
              Some
                {
                  name;
                  version;
                  digest;
                  path = Filename.concat (blob_dir root) hex;
                  size_bytes;
                  metadata;
                  run;
                  step;
                  created_at;
                }
          in
          match v with Some v -> `Version v | None -> `Missing))

let load ~root name n =
  match File.read_opt (version_file root name n) with
  | None -> `Missing
  | Some s -> decode ~root ~name ~version:n s
  | exception Sys_error _ -> `Missing

let encode a =
  let b = Buffer.create 256 in
  let str s b = Json.add_string b s in
  Json.add_object b
    ([
       ("format", fun b -> Json.add_int b Store_format.version);
       ("time", fun b -> Json.add_float b a.created_at);
       ("run", str a.run);
     ]
    @ (match a.step with
      | None -> []
      | Some s -> [ ("step", fun b -> Json.add_int b s) ])
    @ [
        ("digest", str a.digest);
        ("size", fun b -> Json.add_int b a.size_bytes);
        ( "metadata",
          fun b ->
            Json.add_object b
              (List.map
                 (fun (k, v) -> (k, fun b -> Json.add_value b v))
                 a.metadata) );
      ]);
  Buffer.add_char b '\n';
  Buffer.contents b

let tombstone ~root a =
  let b = Buffer.create 64 in
  Json.add_object b
    [
      ("format", fun b -> Json.add_int b Store_format.version);
      ("time", fun b -> Json.add_float b (Unix.gettimeofday ()));
      ("deleted", fun b -> Buffer.add_string b "true");
    ];
  Buffer.add_char b '\n';
  File.replace (version_file root a.name a.version) (Buffer.contents b)

(* Blobs *)

let sys_error path e = raise (Sys_error (path ^ ": " ^ Unix.error_message e))

let linked_dir path =
  invalid_arg
    (Printf.sprintf "Session.log_artifact: %s links to a directory" path)

let kind_of path =
  match Unix.stat path with
  | { Unix.st_kind = Unix.S_DIR; _ } -> (
      match Unix.lstat path with
      | { Unix.st_kind = Unix.S_LNK; _ } -> linked_dir path
      | _ -> `Dir)
  | { Unix.st_kind = Unix.S_REG; _ } -> `File
  | _ -> raise (Sys_error (path ^ ": not a file or directory"))
  | exception Unix.Unix_error (e, _, _) -> sys_error path e

(* The relative paths under [root], files and directories, in byte order. *)
let rec walk root rel acc =
  let dir = if rel = "" then root else Filename.concat root rel in
  Sys.readdir dir
  |> Array.fold_left
       (fun acc n ->
         let r = if rel = "" then n else rel ^ "/" ^ n in
         match kind_of (Filename.concat root r) with
         | `Dir -> walk root r ((r, `Dir) :: acc)
         | `File -> (r, `File) :: acc)
       acc

(* [tree ?dst src] is the digest of the file or directory [src] and its size, a
   directory's digest being that of its listing (Store_format, section blobs).
   With [dst], it also copies [src] to [dst] and syncs the copy. *)
let tree ?dst src =
  match kind_of src with
  | `File -> File.copy ?dst src
  | `Dir ->
      let entries = List.sort compare (walk src "" []) in
      let under r = Option.map (fun d -> Filename.concat d r) dst in
      let mkdir d =
        try Unix.mkdir d 0o777 with Unix.Unix_error (e, _, _) -> sys_error d e
      in
      Option.iter mkdir dst;
      let listing = Buffer.create 256 in
      let size =
        List.fold_left
          (fun size (r, k) ->
            match k with
            | `Dir ->
                Option.iter mkdir (under r);
                Printf.bprintf listing "dir:%s\n" r;
                size
            | `File ->
                let hex, n = File.copy ?dst:(under r) (Filename.concat src r) in
                Printf.bprintf listing "file:%s:%s\n" r hex;
                size + n)
          0 entries
      in
      List.iter
        (fun (r, k) -> if k = `Dir then Option.iter File.sync_dir (under r))
        entries;
      Option.iter File.sync_dir dst;
      (File.digest_string (Buffer.contents listing), size)

let with_lock root kind f =
  File.mkdirs root;
  let path = lock_path root in
  let fd =
    try Unix.openfile path [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_CLOEXEC ] 0o666
    with Unix.Unix_error (e, _, _) ->
      raise (Sys_error (path ^ ": " ^ Unix.error_message e))
  in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  File.lock fd kind;
  f ()

let publish ~root ~run ~step ~metadata ~name path =
  with_lock root `Shared @@ fun () ->
  let blobs = blob_dir root in
  File.mkdirs blobs;
  let tmp = File.temp_name blobs in
  let hex, size_bytes =
    try tree ~dst:tmp path
    with e ->
      File.remove_tree tmp;
      raise e
  in
  let blob = Filename.concat blobs hex in
  (if Sys.file_exists blob then File.remove_tree tmp
   else
     match Unix.rename tmp blob with
     | () -> File.sync_dir blobs
     | exception Unix.Unix_error ((Unix.EEXIST | Unix.ENOTEMPTY), _, _) ->
         File.remove_tree tmp
     | exception Unix.Unix_error (e, _, _) ->
         File.remove_tree tmp;
         raise (Sys_error (blob ^ ": " ^ Unix.error_message e)));
  let dir = name_dir root name in
  File.mkdirs dir;
  let a =
    {
      name;
      version = 0;
      digest = algorithm ^ ":" ^ hex;
      path = blob;
      size_bytes;
      metadata;
      run;
      step;
      created_at = Unix.gettimeofday ();
    }
  in
  let tmp = File.temp_name dir in
  let fd =
    try
      Unix.openfile tmp
        [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_EXCL; Unix.O_CLOEXEC ]
        0o666
    with Unix.Unix_error (e, _, _) -> sys_error tmp e
  in
  (try
     File.write_all fd (encode a);
     File.sync fd;
     Unix.close fd
   with e ->
     Unix.close fd;
     raise e);
  (* A link claims a number: it fails when the name exists, so the next number
     is tried. *)
  let rec claim n =
    match Unix.link tmp (version_file root name n) with
    | () -> n
    | exception Unix.Unix_error (Unix.EEXIST, _, _) -> claim (n + 1)
    | exception Unix.Unix_error (e, _, _) -> sys_error dir e
  in
  let first =
    match List.rev (numbers ~root name) with n :: _ -> n + 1 | [] -> 1
  in
  let n =
    Fun.protect
      ~finally:(fun () -> try Unix.unlink tmp with Unix.Unix_error _ -> ())
      (fun () -> claim first)
  in
  File.sync_dir dir;
  { a with version = n }
