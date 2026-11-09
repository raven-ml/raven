(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let error path e = raise (Sys_error (path ^ ": " ^ Unix.error_message e))
let wrap path f = try f () with Unix.Unix_error (e, _, _) -> error path e

(* Names *)

let is_text s =
  s <> "" && Json.utf8 s
  && String.for_all (fun c -> Char.code c >= 0x20 && c <> '\x7F') s

let utf8 s =
  if String.is_valid_utf_8 s then s
  else
    let b = Buffer.create (String.length s + 8) in
    let rec go i =
      if i < String.length s then begin
        let d = String.get_utf_8_uchar s i in
        Buffer.add_utf_8_uchar b
          (if Uchar.utf_decode_is_valid d then Uchar.utf_decode_uchar d
           else Uchar.rep);
        go (i + Uchar.utf_decode_length d)
      end
    in
    go 0;
    Buffer.contents b

let is_tag s = is_text s && not (String.contains s ',')
let is_segment s = is_text s && s.[0] <> '.' && not (String.contains s '/')
let is_key s = s <> "" && List.for_all is_segment (String.split_on_char '/' s)

(* The system *)

external pid : unit -> int = "caml_munin_getpid"
external monotonic : unit -> float = "caml_munin_monotonic"
external uname : unit -> string * string = "caml_munin_uname"
external sysctl_string : string -> string option = "caml_munin_sysctl_string"
external darwin_boot_time : unit -> float option = "caml_munin_boot_time"
external platform_uuid : unit -> string option = "caml_munin_platform_uuid"

(* [shared f] is [f ()], tried again while Windows refuses it with EACCES
   because another handle holds the file. Windows renames no file over one that
   another handle has open, a reader's included: MoveFileEx, which OCaml's
   [Unix.rename] calls with MOVEFILE_REPLACE_EXISTING, fails until that handle
   closes, an open can fail while a rename replaces its file, and a removal can
   fail while another handle or a rename holds the file. A reader holds a file
   for one read and a rename is one call, so [f] is tried again after 1,
   2, 4 ... 512 ms: at most 11 tries over about one second, then the failure is
   raised. Elsewhere [f] is tried once. *)
let shared_tries = 11

let shared f =
  let rec go k delay =
    match f () with
    | v -> v
    | exception Unix.Unix_error (Unix.EACCES, _, _)
      when Sys.win32 && k < shared_tries ->
        Unix.sleepf delay;
        go (k + 1) (2. *. delay)
  in
  go 1 0.001

(* Every file munin opens is opened share-delete, so that on Windows another
   process can rename or remove it meanwhile, as a deletion renames a log a
   reader has open. *)
let open_flags = [ Unix.O_RDONLY; Unix.O_CLOEXEC; Unix.O_SHARE_DELETE ]

let create_flags =
  Unix.[ O_WRONLY; O_CREAT; O_EXCL; O_CLOEXEC; O_SHARE_DELETE ]

let open_read path =
  wrap path (fun () -> shared (fun () -> Unix.openfile path open_flags 0))

let open_read_opt path =
  match shared (fun () -> Unix.openfile path open_flags 0) with
  | fd -> Some fd
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> None
  | exception Unix.Unix_error (e, _, _) -> error path e

(* [fill fd b off] reads into [b] from [off] until [b] is full or the file ends,
   and is the offset reached. *)
let rec fill fd b off =
  if off >= Bytes.length b then off
  else
    match Unix.read fd b off (Bytes.length b - off) with
    | 0 -> off
    | n -> fill fd b (off + n)

(* Whole files, read to their end into one buffer of the size [fstat] gives, so
   that a small file costs its bytes alone. A /proc file reports a size of 0,
   and a file may grow after the [fstat]: the buffer doubles until a read finds
   the end. *)
let read_fd path fd =
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  wrap path @@ fun () ->
  let size = (Unix.fstat fd).Unix.st_size in
  let probe = Bytes.create 1 in
  let rec go b off =
    let off = fill fd b off in
    if off < Bytes.length b then Bytes.sub_string b 0 off
    else
      match Unix.read fd probe 0 1 with
      | 0 -> Bytes.unsafe_to_string b
      | _ ->
          let b = Bytes.extend b 0 (Int.max 4096 (Bytes.length b)) in
          Bytes.set b off (Bytes.get probe 0);
          go b (off + 1)
  in
  go (Bytes.create size) 0

let read_all path = read_fd path (open_read path)

(* Only the open's own ENOENT says the file does not exist: a test of its
   existence after a failed open would race with a writer that renames the
   file into place meanwhile. *)
let read_opt path = Option.map (read_fd path) (open_read_opt path)

(* [read_small p] is [p]'s content, or [None] if it cannot be read. *)
let read_small path = try read_opt path with Sys_error _ -> None

let lines path =
  match read_small path with
  | None -> []
  | Some s -> String.split_on_char '\n' s

(* "model name : <cpu>" in /proc/cpuinfo on x86; arm64 kernels name none. *)
let cpu_model () =
  match sysctl_string "machdep.cpu.brand_string" with
  | Some _ as m -> m
  | None ->
      List.find_map
        (fun l ->
          match String.index_opt l ':' with
          | Some i when String.trim (String.sub l 0 i) = "model name" ->
              Some
                (String.trim (String.sub l (i + 1) (String.length l - i - 1)))
          | _ -> None)
        (lines "/proc/cpuinfo")

let boot_id =
  let id =
    lazy
      (match read_small "/proc/sys/kernel/random/boot_id" with
      | Some s when String.trim s <> "" -> Some (String.trim s)
      | _ -> sysctl_string "kern.bootsessionuuid")
  in
  fun () -> Lazy.force id

(* Linux: the "btime" line of /proc/stat, seconds since the epoch. *)
let boot_time () =
  match darwin_boot_time () with
  | Some _ as t -> t
  | None ->
      List.find_map
        (fun l ->
          match String.split_on_char ' ' l with
          | [ "btime"; t ] -> Option.map Float.of_int (int_of_string_opt t)
          | _ -> None)
        (lines "/proc/stat")

(* BLAKE2b *)

external blake2_size : unit -> int = "caml_munin_blake2_size"
external blake2_init : bytes -> int -> string -> unit = "caml_munin_blake2_init"

external blake2_update : bytes -> bytes -> int -> int -> unit
  = "caml_munin_blake2_update"

external blake2_final : bytes -> int -> string = "caml_munin_blake2_final"

type hasher = bytes

let digest_bytes = 32

let hasher ?(key = "") () =
  let h = Bytes.create (blake2_size ()) in
  blake2_init h digest_bytes key;
  h

let feed h b off len =
  if off < 0 || len < 0 || off > Bytes.length b - len then
    invalid_arg "File.feed";
  blake2_update h b off len

let to_hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let hex h = to_hex (blake2_final h digest_bytes)

let digest_string s =
  let h = hasher () in
  feed h (Bytes.unsafe_of_string s) 0 (String.length s);
  hex h

(* The key that makes the machine identity munin's own, as machine-id(5) asks of
   an application. *)
let machine_key = "munin"

let machine_id =
  let id =
    lazy
      (let text =
         match read_small "/etc/machine-id" with
         | Some s ->
             let s = String.trim s in
             if String.length s = 32 then Some s else None
         | None -> platform_uuid ()
       in
       Option.map
         (fun t ->
           let h = hasher ~key:machine_key () in
           feed h (Bytes.of_string t) 0 (String.length t);
           hex h)
         text)
  in
  fun () -> Lazy.force id

(* Locks *)

external lock_stub : Unix.file_descr -> int -> bool -> bool = "caml_munin_lock"

let kind = function `Exclusive -> 0 | `Shared -> 1

let try_lock fd k =
  try lock_stub fd (kind k) false
  with Unix.Unix_error (e, _, _) -> error "lock" e

let lock fd k =
  try ignore (lock_stub fd (kind k) true)
  with Unix.Unix_error (e, _, _) -> error "lock" e

let unlock fd =
  try ignore (lock_stub fd 2 false)
  with Unix.Unix_error (e, _, _) -> error "unlock" e

let with_lock path k f =
  let fd =
    try
      Unix.openfile path
        [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_CLOEXEC; Unix.O_SHARE_DELETE ]
        0o666
    with Unix.Unix_error (e, _, _) -> error path e
  in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  lock fd k;
  f ()

(* A refused exclusive lock is a writer's or a reader's probe. A shared lock
   tells them apart: readers' shared locks grant it, a writer's exclusive lock
   refuses it. A probe holds its lock for two system calls, so the loop ends at
   the first gap between probes. *)
let take_writer fd =
  let rec go () =
    if try_lock fd `Exclusive then true
    else if try_lock fd `Shared then (
      unlock fd;
      Thread.yield ();
      go ())
    else false
  in
  go ()

let probe fd =
  if try_lock fd `Shared then (
    unlock fd;
    false)
  else true

(* [repeat key l] is the first [key x] of an [x] of [l] that an earlier one has,
   found without allocating: the lists are a call's few arguments. *)
let repeat key l =
  let rec among k n = function
    | x :: rest when n > 0 -> key x = k || among k (n - 1) rest
    | _ -> false
  in
  let rec go i = function
    | [] -> None
    | x :: rest ->
        let k = key x in
        if among k i l then Some k else go (i + 1) rest
  in
  go 0 l

(* Files *)

external full_fsync : Unix.file_descr -> unit = "caml_munin_full_fsync"

let sync fd =
  try full_fsync fd with Unix.Unix_error (e, _, _) -> error "fsync" e

let sync_dir dir =
  if not Sys.win32 then
    wrap dir @@ fun () ->
    let fd = Unix.openfile dir [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 in
    Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
    (* Some file systems refuse to sync a directory; their entries reach the
       storage with the files that name them. *)
    try Unix.fsync fd
    with Unix.Unix_error ((Unix.EINVAL | Unix.EBADF), _, _) -> ()

let fstat fd = wrap "fstat" (fun () -> Unix.fstat fd)

let pread fd ~pos ~len =
  wrap "read" @@ fun () ->
  let b = Bytes.create len in
  ignore (Unix.lseek fd pos Unix.SEEK_SET);
  let n = fill fd b 0 in
  if n = len then Bytes.unsafe_to_string b else Bytes.sub_string b 0 n

let write_sub fd b off len =
  wrap "write" @@ fun () ->
  let rec go off left =
    if left > 0 then
      let n = Unix.write fd b off left in
      go (off + n) (left - n)
  in
  go off len

let write_all fd s =
  wrap "write" @@ fun () ->
  let b = Bytes.unsafe_of_string s in
  let rec go off =
    if off < Bytes.length b then
      go (off + Unix.write fd b off (Bytes.length b - off))
  in
  go 0

let buf_size = 65536

(* [copy ?dst src] is the hexadecimal digest of [src]'s bytes and their count.
   With [dst], it also copies them to the new file [dst], digesting the bytes it
   writes, and syncs it. *)
let copy ?dst src =
  let h = hasher () in
  let buf = Bytes.create buf_size in
  let ic = open_read src in
  Fun.protect ~finally:(fun () -> Unix.close ic) @@ fun () ->
  let oc =
    Option.map
      (fun dst ->
        try
          Unix.openfile dst create_flags 0o666
        with Unix.Unix_error (e, _, _) -> error dst e)
      dst
  in
  Fun.protect ~finally:(fun () -> Option.iter Unix.close oc) @@ fun () ->
  let rec go total =
    match Unix.read ic buf 0 buf_size with
    | 0 -> total
    | n ->
        Option.iter (fun oc -> write_sub oc buf 0 n) oc;
        feed h buf 0 n;
        go (total + n)
    | exception Unix.Unix_error (e, _, _) -> error src e
  in
  let total = go 0 in
  Option.iter sync oc;
  (hex h, total)

let rec mkdirs dir =
  if not (Sys.file_exists dir) then begin
    let parent = Filename.dirname dir in
    if parent <> dir then mkdirs parent;
    try Unix.mkdir dir 0o777 with
    | Unix.Unix_error (Unix.EEXIST, _, _) -> ()
    | Unix.Unix_error (e, _, _) -> error dir e
  end

let mkdir dir =
  try Unix.mkdir dir 0o777 with
  | Unix.Unix_error (Unix.EEXIST, _, _) -> ()
  | Unix.Unix_error (e, _, _) -> error dir e

let entries dir =
  match Sys.readdir dir with
  | exception Sys_error _ -> []
  | names ->
      Array.to_list names
      |> List.filter (fun n -> n <> "" && n.[0] <> '.')
      |> List.sort String.compare

let rand = Random.State.make_self_init ()
let rand_lock = Mutex.create ()

let temp_name dir =
  let r = Mutex.protect rand_lock (fun () -> Random.State.bits rand) in
  Filename.concat dir (Printf.sprintf ".tmp-%d-%08x" (pid ()) r)

let replace ?(parents = true) path data =
  let dir = Filename.dirname path in
  if parents then mkdirs dir;
  let tmp = temp_name dir in
  wrap path @@ fun () ->
  let fd = Unix.openfile tmp create_flags 0o666 in
  (match
     write_all fd data;
     sync fd
   with
  | () -> Unix.close fd
  | exception e ->
      Unix.close fd;
      (try Unix.unlink tmp with Unix.Unix_error _ -> ());
      raise e);
  try shared (fun () -> Unix.rename tmp path)
  with e ->
    (try Unix.unlink tmp with Unix.Unix_error _ -> ());
    raise e

(* Windows refuses to remove a read-only file, which a POSIX system removes
   whatever its mode. Clearing the attribute first gives Windows that rule. *)
let unlink path =
  try Unix.unlink path
  with Unix.Unix_error (Unix.EACCES, _, _) when Sys.win32 ->
    Unix.chmod path 0o666;
    Unix.unlink path

(* An entry gone between the listing and its removal was removed by another
   process, as gc and a deletion may both remove what a deletion left, or
   another domain, as two writers of one run's notes may. *)
let gone path f =
  try shared f with
  | Unix.Unix_error (Unix.ENOENT, _, _) -> ()
  | Unix.Unix_error (e, _, _) -> error path e

let rec remove_tree path =
  match Unix.lstat path with
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ()
  | exception Unix.Unix_error (e, _, _) -> error path e
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter
        (fun n -> remove_tree (Filename.concat path n))
        (try Sys.readdir path with Sys_error _ -> [||]);
      gone path (fun () -> Unix.rmdir path)
  | _ -> gone path (fun () -> unlink path)

let stamp path =
  match Unix.stat path with
  | st -> Some (st.Unix.st_mtime, st.Unix.st_size, st.Unix.st_ino)
  | exception Unix.Unix_error _ -> None

let mtime path = Option.map (fun (t, _, _) -> t) (stamp path)

(* Processes *)

let output argv =
  match Unix.open_process_args_full argv.(0) argv (Unix.environment ()) with
  | exception Unix.Unix_error _ -> None
  | (out, inp, err) as p -> (
      close_out_noerr inp;
      (* Drain stderr on its own so that a chatty program cannot block on a full
         pipe while we read stdout. *)
      let drain =
        Thread.create
          (fun () ->
            try ignore (In_channel.input_all err) with Sys_error _ -> ())
          ()
      in
      let s = try In_channel.input_all out with Sys_error _ -> "" in
      Thread.join drain;
      match Unix.close_process_full p with
      | Unix.WEXITED 0 -> Some s
      | _ -> None)
