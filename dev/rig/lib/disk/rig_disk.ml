(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every function may be called from any domain. The descriptor table is guarded
   by one lock ([locked]); a copy pins its file's descriptor under it and moves
   its bytes outside it, so copies of several domains run at once and a pinned
   descriptor is never closed under a transfer. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type pages =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external open_path : string -> int -> int -> int * int * int * string
  = "caml_rig_disk_open"

external close : int -> unit = "caml_rig_disk_close"
external sync : int -> int = "caml_rig_disk_sync"
external error : int -> string = "caml_rig_disk_error"

external pread :
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "caml_rig_disk_read_byte" "caml_rig_disk_read"

external pwrite :
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "caml_rig_disk_write_byte" "caml_rig_disk_write"

external map : int -> int -> bool -> int * pages option = "caml_rig_disk_map"

external advise : int -> pages option -> int -> int -> unit
  = "caml_rig_disk_advise"

external msync : pages -> int = "caml_rig_disk_msync"
external watch_forks : unit -> unit = "caml_rig_disk_watch_forks"
external forks : unit -> int = "caml_rig_disk_forks" [@@noalloc]

(* The codes [open_path] answers besides the system's, and its modes. *)
let not_regular = -1
let too_many = -2
let unmappable = -3
let read_mode = 0
let write_mode = 1
let create_mode = 2

(* [why code] is what the failure [code] of an open says. *)
let why code =
  if code = not_regular then "not a regular file"
  else if code = too_many then "too many open files"
  else error code

(* Files *)

(* A file is named by its path and its identity, bytes that name it exactly and
   that no other file has, or [""] where the system gives none. A file without
   an identity keeps its descriptor until it is freed: nothing could tell its
   path's file apart at a reopen. *)
type file = {
  path : string;
  writable : bool;
  size : int;
  identity : string;
  mutable fd : int; (* [-1] while closed *)
  mutable users : int; (* copies using [fd] *)
  mutable newer : file; (* links in the idle ring, [f] itself out of it *)
  mutable older : file;
  mutable pages : pages option; (* its mapping, once a borrow made it *)
}

(* A file not yet open, out of the idle ring. *)
let file ~path ~writable ~size ~identity =
  let rec f =
    {
      path;
      writable;
      size;
      identity;
      fd = -1;
      users = 0;
      newer = f;
      older = f;
      pages = None;
    }
  in
  f

let sys_error f why = raise (Sys_error (strf "%s: %s" f.path why))

(* Descriptors *)

(* The table holds the descriptors of files with an identity: at most
   [max_open], more only while copies pin them, the least recently used unpinned
   one closed to open another. The unpinned ones are idle, in a ring through
   [idle] from the most recently used ([idle.newer]) to the least
   ([idle.older]), so an open or a copy takes the same few steps and allocates
   nothing however full the table is. *)
let max_open = 64
let in_table = ref 0

(* The table's lock and the forks it was made after. A forked child makes the
   lock anew at its first use: a thread of its parent may have held it. *)
let lock =
  watch_forks ();
  Atomic.make (forks (), Mutex.create ())

let rec table_lock () =
  let ((made, m) as l) = Atomic.get lock and n = forks () in
  if made = n then m
  else begin
    ignore (Atomic.compare_and_set lock l (n, Mutex.create ()));
    table_lock ()
  end

let locked f = Mutex.protect (table_lock ()) f

(* The ring's ends: a file of no path, never open. *)
let idle = file ~path:"" ~writable:false ~size:0 ~identity:""

let unlink f =
  f.older.newer <- f.newer;
  f.newer.older <- f.older;
  f.newer <- f;
  f.older <- f

(* Puts the unpinned [f] at the ring's most recent end, if its descriptor is in
   the table. *)
let rest f =
  if f.identity <> "" && f.fd >= 0 then begin
    f.older <- idle;
    f.newer <- idle.newer;
    idle.newer.older <- f;
    idle.newer <- f
  end

let close_fd f =
  if f.fd >= 0 then begin
    close f.fd;
    f.fd <- -1;
    if f.identity <> "" then begin
      decr in_table;
      unlink f
    end
  end

(* Closes the least recently used idle descriptors while more than [max_open]
   are open: the table comes back to its bound once a burst of pins ends. *)
let trim () =
  while !in_table > max_open && idle.older != idle do
    close_fd idle.older
  done

let admit f fd =
  f.fd <- fd;
  if f.identity <> "" then begin
    incr in_table;
    trim ()
  end

(* Opens [path], closing every unpinned descriptor and trying once more if the
   process has too many open. *)
let open_retrying path mode n =
  match open_path path mode n with
  | code, _, _, _ when code = too_many && idle.older != idle ->
      while idle.older != idle do
        close_fd idle.older
      done;
      open_path path mode n
  | r -> r

(* [f]'s descriptor, opened again by its path if it was closed, which must still
   name [f]'s file. *)
let reopen f =
  match
    open_retrying f.path (if f.writable then write_mode else read_mode) 0
  with
  | 0, fd, _, identity when String.equal identity f.identity -> admit f fd
  | 0, fd, _, _ ->
      close fd;
      sys_error f "the path names another file since its buffers opened it"
  | code, _, _, _ -> sys_error f (why code)

let pin f =
  locked @@ fun () ->
  if f.fd < 0 then reopen f else if f.users = 0 then unlink f;
  f.users <- f.users + 1;
  f.fd

let unpin f =
  locked @@ fun () ->
  f.users <- f.users - 1;
  if f.users = 0 then begin
    rest f;
    trim ()
  end

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
      "Rig.Buffer.create: DISK makes no memory: open a file with \
       Rig_disk.of_file or Rig_disk.create_file"

  (* Nothing reaches [f]: no pin is held. *)
  let free () f = locked (fun () -> close_fd f)

  let read () f ~at ~dst ~len =
    using f @@ fun fd ->
    let k = pread fd at dst len in
    if k < 0 then sys_error f (error (-k));
    if k < len then
      sys_error f
        (strf "the file ends at byte %d, before byte %d" (at + k) (at + len))

  let write () f ~at ~src ~len =
    if not f.writable then
      invalid_argf "Rig.Buffer.copy: %s was opened for reading" f.path;
    using f @@ fun fd ->
    let k = pwrite fd at src len in
    if k < 0 then sys_error f (error (-k))

  (* A file opened for writing maps shared, so the mapping is the file; one
     opened for reading maps copy-on-write. *)
  (* A failure that may pass, such as a file that cannot be reopened for the
     moment, raises; [None] says the file system can never map the file. *)
  let pages () f =
    match using f (fun fd -> map fd f.size f.writable) with
    | 0, pages ->
        f.pages <- pages;
        pages
    | code, _ when code = unmappable -> None
    | code, _ -> sys_error f (error code)

  (* A file that cannot be reopened gets no advice; its pages, mapped already,
     stay valid. *)
  let prefetch () f ~at ~len =
    try using f (fun fd -> advise fd f.pages at len) with Sys_error _ -> ()

  let stop () = ()
end

(* The disk *)

(* The first open of its name, by a device that is never lost: it cannot
   fail. *)
let device =
  match Rig.open_io (module Io) ~name:"DISK" (fun () -> Ok ()) with
  | Ok d -> d
  | Error why -> failwith (strf "Rig_disk: cannot open DISK: %s" why)

let open_file path mode n =
  if String.contains path '\000' then
    Error (strf "%s: the path holds a NUL byte" path)
  else
    let writable = mode <> read_mode in
    let opened () =
      match open_retrying path mode n with
      | 0, fd, size, identity ->
          let f = file ~path ~writable ~size ~identity in
          admit f fd;
          rest f;
          Ok f
      | code, _, _, _ -> Error (strf "%s: %s" path (why code))
    in
    locked opened
    |> Result.map (fun f -> Rig.Buffer.of_io device Io.region_key f f.size)

let of_file path = open_file path read_mode 0

let create_file path n =
  if n < 0 then invalid_argf "Rig_disk.create_file: %d bytes is negative" n;
  open_file path create_mode n

let barrier b =
  Option.iter
    (invalid_argf "Rig_disk.barrier: the buffer is dead: %s")
    (Rig.Buffer.dead b);
  match Rig.Buffer.io b Io.region_key with
  | None when Rig.equal (Rig.Buffer.device b) device ->
      (* A buffer of no bytes, which no file holds. *)
      ()
  | None -> invalid_arg "Rig_disk.barrier: the buffer is not on DISK"
  | Some f when not f.writable -> ()
  | Some f ->
      (* A device's work writing through a borrow writes the file once done: the
         barrier orders after it, as a copy reading [b] would. *)
      Rig.Buffer.wait b Read;
      let synced code = if code <> 0 then sys_error f (error code) in
      (* Writes through a shared mapping reach the file by the mapping's own
         flush, which the descriptor's sync does not cover. *)
      Option.iter (fun pages -> synced (msync pages)) f.pages;
      using f (fun fd -> synced (sync fd));
      (* Until the descriptor is unpinned: collecting [b] would close it. *)
      ignore (Sys.opaque_identity b)
