(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every function may be called from any domain. The descriptor table is guarded
   by [lock]; a copy pins its file's descriptor under it and moves its bytes
   outside it, so copies of several domains run at once and a pinned descriptor
   is never closed under a transfer. *)

let strf = Printf.sprintf

type pages =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external open_path : string -> int -> int -> int * int * int
  = "caml_device_disk_open"

external close : int -> unit = "caml_device_disk_close"
external identity : int -> int * int * int * int = "caml_device_disk_identity"
external sync : int -> int = "caml_device_disk_sync"
external error : int -> string = "caml_device_disk_error"

external read :
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "caml_device_disk_read_byte" "caml_device_disk_read"

external write :
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "caml_device_disk_write_byte" "caml_device_disk_write"

external map : int -> int -> bool -> int * pages = "caml_device_disk_map"
external advise : int -> int -> int -> unit = "caml_device_disk_advise"
external msync : pages -> int = "caml_device_disk_msync"

(* The codes [open_path] answers besides the system's, and its modes. *)
let not_regular = -1
let too_many = -2
let read_mode = 0
let write_mode = 1
let create_mode = 2

(* Files *)

(* A file is named by its path and its identity: its device, its number there
   and when it last changed, which its own writes advance. *)
type identity = { dev : int; ino : int; changed : int }

type file = {
  path : string;
  writable : bool;
  size : int;
  mutable identity : identity;
  mutable fd : int; (* [-1] while closed *)
  mutable users : int; (* copies using [fd] *)
  mutable used : int; (* when [fd] was last used, by [clock] *)
  mutable pages : pages option; (* a writable file's shared mapping *)
}

let same i i' = i.dev = i'.dev && i.ino = i'.ino && i.changed = i'.changed

let identify fd =
  match identity fd with
  | 0, dev, ino, changed -> Ok { dev; ino; changed }
  | code, _, _, _ -> Error code

let sys_error f why = raise (Sys_error (strf "%s: %s" f.path why))

(* Descriptors *)

(* The table holds the open descriptors: at most [max_open] unpinned, the least
   recently used closed to open another. *)
let max_open = 64
let lock = Mutex.create ()
let opened : file list ref = ref []
let clock = ref 0

let touch f =
  incr clock;
  f.used <- !clock

let close_fd f =
  if f.fd >= 0 then begin
    close f.fd;
    f.fd <- -1;
    opened := List.filter (fun f' -> f' != f) !opened
  end

let unpinned () = List.filter (fun f -> f.users = 0) !opened

let admit f fd =
  (match unpinned () with
  | o :: rest when List.length !opened >= max_open ->
      close_fd
        (List.fold_left (fun o f -> if f.used < o.used then f else o) o rest)
  | _ -> ());
  f.fd <- fd;
  touch f;
  opened := f :: !opened

(* Opens [path], closing every unpinned descriptor and trying once more if the
   process has too many open. *)
let open_retrying path mode n =
  match open_path path mode n with
  | code, _, _ when code = too_many && unpinned () <> [] ->
      List.iter close_fd (unpinned ());
      open_path path mode n
  | r -> r

(* [f]'s descriptor, opened again by its path if it was closed, which must still
   name [f]'s file, unchanged. *)
let reopen f =
  match
    open_retrying f.path (if f.writable then write_mode else read_mode) 0
  with
  | 0, fd, _ -> (
      match identify fd with
      | Ok i when same i f.identity -> admit f fd
      | Ok _ ->
          close fd;
          sys_error f "the file changed since its buffers opened it"
      | Error code ->
          close fd;
          sys_error f (error code))
  | code, _, _ when code = too_many -> sys_error f "too many open files"
  | code, _, _ -> sys_error f (error code)

let pin f =
  Mutex.protect lock @@ fun () ->
  if f.fd < 0 then reopen f else touch f;
  f.users <- f.users + 1;
  f.fd

let unpin f = Mutex.protect lock @@ fun () -> f.users <- f.users - 1

(* [using f fn] is [fn fd] with [f]'s descriptor [fd] pinned. Raises [Sys_error]
   naming [f] if it cannot be reopened. *)
let using f fn =
  let fd = pin f in
  Fun.protect ~finally:(fun () -> unpin f) (fun () -> fn fd)

(* The disk as an io device *)

module Io = struct
  type t = unit
  type region = file

  exception Fault of string

  let region_key : file Type.Id.t = Type.Id.make ()
  let budget () = max_int

  let alloc () _ =
    invalid_arg
      "Device_core.Buffer.create: DISK makes no memory: open a file with \
       Device_disk.of_file or Device_disk.create_file"

  let free () f =
    f.pages <- None;
    Mutex.protect lock (fun () -> close_fd f)

  let read () f ~at ~dst ~len =
    using f @@ fun fd ->
    let k = read fd at dst len in
    if k < 0 then sys_error f (error (-k));
    if k < len then
      sys_error f
        (strf "the file ends at byte %d, before byte %d" (at + k) (at + len))

  let write () f ~at ~src ~len =
    if not f.writable then
      invalid_arg
        (strf "Device_core.Buffer.copy: %s was opened for reading" f.path);
    using f @@ fun fd ->
    let k = write fd at src len in
    if k < 0 then sys_error f (error (-k));
    (* The write changed the file: its reopens must still take it for its
       own. *)
    match identify fd with
    | Ok i -> f.identity <- i
    | Error _ -> ()

  (* A file opened for writing maps shared, so the mapping is the file; one
     opened for reading maps copy-on-write. *)
  let pages () f =
    if f.size = 0 then None
    else
      match using f (fun fd -> map fd f.size f.writable) with
      | 0, ba ->
          if f.writable then f.pages <- Some ba;
          Some ba
      | _ -> None
      | exception Sys_error _ -> None

  (* A file that cannot be reopened gets no advice; its pages, mapped already,
     stay valid. *)
  let prefetch () f ~at ~len =
    match using f (fun fd -> advise fd at len) with
    | () -> ()
    | exception Sys_error _ -> ()

  let stop () = ()
end

(* The disk *)

(* The first open of its name, by a device that is never lost: it cannot
   fail. *)
let device =
  match Device_core.open_io (module Io) ~name:"DISK" (fun () -> Ok ()) with
  | Ok d -> d
  | Error why -> failwith (strf "Device_disk: cannot open DISK: %s" why)

let open_file path mode n =
  let writable = mode <> read_mode in
  let opened () =
    match open_retrying path mode n with
    | 0, fd, size -> (
        match identify fd with
        | Ok identity ->
            let f =
              {
                path;
                writable;
                size;
                identity;
                fd = -1;
                users = 0;
                used = 0;
                pages = None;
              }
            in
            admit f fd;
            Ok f
        | Error code ->
            close fd;
            Error (strf "%s: %s" path (error code)))
    | code, _, _ when code = not_regular ->
        Error (strf "%s: not a regular file" path)
    | code, _, _ when code = too_many ->
        Error (strf "%s: too many open files" path)
    | code, _, _ -> Error (strf "%s: %s" path (error code))
  in
  if String.contains path '\000' then
    Error (strf "%S: a path has no NUL byte" path)
  else
    Mutex.protect lock opened
    |> Result.map (fun f ->
        Device_core.Buffer.of_io device Io.region_key f f.size)

let of_file path = open_file path read_mode 0

let create_file path n =
  if n < 0 then
    invalid_arg (strf "Device_disk.create_file: %d bytes is negative" n);
  open_file path create_mode n

let barrier b =
  match Device_core.Buffer.io b Io.region_key with
  | None -> invalid_arg "Device_disk.barrier: the buffer is not on DISK"
  | Some f when not f.writable -> ()
  | Some f ->
      let synced code = if code <> 0 then sys_error f (error code) in
      (* Writes through a shared mapping reach the file by the mapping's own
         flush, which the descriptor's sync does not cover. *)
      Option.iter (fun pages -> synced (msync pages)) f.pages;
      using f @@ fun fd -> synced (sync fd)
