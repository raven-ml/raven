(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every function may be called from any domain. The descriptor table is guarded
   by one lock ([locked]). A copy pins its file's descriptor, taking the lock
   only to reopen it (Pins), and moves its bytes outside it, so copies of
   several domains run at once and a pinned descriptor is never closed under a
   transfer. *)

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
  mutable users : int; [@atomic] (* copies using [fd], [-1] while it closes *)
  mutable newer : file; (* links in the table's ring, [f] itself out of it *)
  mutable older : file;
  mutable pages : pages option; (* its mapping, once a borrow made it *)
}

(* A file not yet open, out of the ring. *)
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
   one closed to open another. They are in a ring through [ring], from the most
   recently used ([ring.newer]) to the least ([ring.older]). A pin leaves its
   file where it is and [trim] passes over pinned files, so a copy of the most
   recently used file changes no link, and an open or a copy takes the same few
   steps and allocates nothing however full the table is. *)
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

(* [locked g x] is [g x] under the table's lock. [g] is a toplevel function, so
   a call makes no closure. *)
let locked g x =
  let m = table_lock () in
  Mutex.lock m;
  match g x with
  | r ->
      Mutex.unlock m;
      r
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      Mutex.unlock m;
      Printexc.raise_with_backtrace e bt

(* The ring's ends: a file of no path, never open. *)
let ring = file ~path:"" ~writable:false ~size:0 ~identity:""

(* Whether the table holds [f]'s descriptor while it is open. *)
let listed f = f.identity <> ""

let unlink f =
  f.older.newer <- f.newer;
  f.newer.older <- f.older;
  f.newer <- f;
  f.older <- f

(* Puts [f], whose descriptor is in the table, at the ring's most recent end. *)
let touch f =
  if ring.newer != f then begin
    unlink f;
    f.older <- ring;
    f.newer <- ring.newer;
    ring.newer.older <- f;
    ring.newer <- f
  end

(* Closes [f]'s descriptor, which no copy pins, and ends an eviction's claim on
   [users]. The file reads as closed before the system's call, which may raise
   an asynchronous exception. *)
let close_fd f =
  let fd = f.fd in
  if fd >= 0 then begin
    f.fd <- -1;
    if listed f then begin
      decr in_table;
      unlink f
    end;
    f.users <- 0;
    close fd
  end

(* Closes [f]'s descriptor unless a copy pins it. It claims [users] at 0, so no
   pin starts meanwhile. *)
let evict f =
  if Atomic.Loc.compare_and_set [%atomic.loc f.users] 0 (-1) then close_fd f

(* Closes the least recently used unpinned descriptors while more than [keep]
   are open. *)
let close_down keep =
  let f = ref ring.older in
  while !in_table > keep && !f != ring do
    let g = !f in
    f := g.newer;
    if g.users = 0 then evict g
  done

(* Brings the table back to its bound, once the pins that took it beyond end. *)
let trim () = close_down max_open

let admit f fd =
  f.fd <- fd;
  if listed f then begin
    incr in_table;
    touch f
  end

(* Opens [path], closing every unpinned descriptor and trying once more if the
   process has too many open. *)
let open_retrying path mode n =
  let ((code, _, _, _) as r) = open_path path mode n and before = !in_table in
  if code <> too_many then r
  else begin
    close_down 0;
    if !in_table < before then open_path path mode n else r
  end

(* Opens [f]'s path again, which must still name [f]'s file. *)
let reopen f =
  match
    open_retrying f.path (if f.writable then write_mode else read_mode) 0
  with
  | 0, fd, _, identity when String.equal identity f.identity -> admit f fd
  | 0, fd, _, _ ->
      close fd;
      sys_error f "the path names another file since its buffers opened it"
  | code, _, _, _ -> sys_error f (why code)

(* Pins

   A copy pins its file's descriptor, which no eviction closes while [users] is
   above 0. An open descriptor is pinned and unpinned by one step on [users],
   without the table's lock: an eviction closes a descriptor only by claiming
   [users] from 0, and a pin that finds the file closed reopens it under the
   lock. An unpin takes the lock only to make its file the most recently used or
   to bring the table back to its bound. *)

(* [f]'s descriptor, opened again by its path if it was closed, pinned. Holds
   the table's lock. *)
let pinned f =
  if f.fd < 0 then reopen f;
  Atomic.Loc.incr [%atomic.loc f.users];
  trim ();
  f.fd

(* The end of a pin that left [f] unpinned. Holds the table's lock. *)
let unpinned f =
  if f.users = 0 && f.fd >= 0 && listed f then begin
    touch f;
    trim ()
  end

(* [pin f] is [f]'s descriptor, which no eviction closes until [unpin f]. *)
let rec pin f =
  let n = f.users in
  if n < 0 then locked pinned f
  else if not (Atomic.Loc.compare_and_set [%atomic.loc f.users] n (n + 1)) then
    pin f
  else
    let fd = f.fd in
    if fd >= 0 then fd
    else begin
      Atomic.Loc.decr [%atomic.loc f.users];
      locked pinned f
    end

let unpin f =
  if
    Atomic.Loc.fetch_and_add [%atomic.loc f.users] (-1) = 1
    && listed f
    && (ring.newer != f || !in_table > max_open)
  then locked unpinned f

(* [using f fn] is [fn fd] with [f]'s descriptor [fd] pinned. Raises [Sys_error]
   naming [f] if it cannot be reopened. *)
let using f fn =
  let fd = pin f in
  match fn fd with
  | r ->
      unpin f;
      r
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      unpin f;
      Printexc.raise_with_backtrace e bt

(* [moved f transfer at addr len] is [transfer fd at addr len] with [f]'s
   descriptor [fd] pinned: [using] without a closure, for copies. *)
let moved f transfer at addr len =
  let fd = pin f in
  match transfer fd at addr len with
  | k ->
      unpin f;
      k
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      unpin f;
      Printexc.raise_with_backtrace e bt

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
  let free () f = locked close_fd f

  let read () f ~at ~dst ~len =
    let k = moved f pread at dst len in
    if k < 0 then sys_error f (error (-k));
    if k < len then
      sys_error f
        (strf "the file ends at byte %d, before byte %d" (at + k) (at + len))

  (* Rig writes no file opened for reading: its memory admits only reads
     ([Rig.Buffer.access]). *)
  let write () f ~at ~src ~len =
    let k = moved f pwrite at src len in
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
          trim ();
          Ok f
      | code, _, _, _ -> Error (strf "%s: %s" path (why code))
    in
    match locked opened () with
    | Error _ as e -> e
    | Ok f -> (
        let access = if writable then Rig.Buffer.Read_write else Read in
        match Rig.Buffer.of_io device Io.region_key f ~access f.size with
        | b -> Ok b
        | exception e ->
            (* A drain [of_io] ran raised, which no buffer of the file survives:
               the file goes as a failed create's does. *)
            let bt = Printexc.get_raw_backtrace () in
            locked close_fd f;
            (if mode = create_mode then
               try Sys.remove path with Sys_error _ -> ());
            Printexc.raise_with_backtrace e bt)

let of_file path = open_file path read_mode 0

let create_file path n =
  if n < 0 then invalid_argf "Rig_disk.create_file: %d bytes is negative" n;
  open_file path create_mode n

let barrier b =
  (* The message is formatted only on failure: a partial application of a format
     builds its closure at every call. *)
  (match Rig.Buffer.dead b with
  | Some why -> invalid_argf "Rig_disk.barrier: the buffer is dead: %s" why
  | None -> ());
  match Rig.Buffer.io b Io.region_key with
  | None when Rig.equal (Rig.Buffer.device b) device ->
      (* A buffer of no bytes, which no file holds. *)
      ()
  | None -> invalid_arg "Rig_disk.barrier: the buffer is not on DISK"
  | Some f when (not f.writable) || f.size = 0 ->
      (* Nothing writes a file opened for reading or a file of no bytes. *)
      ()
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
