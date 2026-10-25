(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every function may be called from any domain. The descriptor table is guarded
   by one lock ([locked]). A copy pins its file's descriptor, taking the lock
   only to reopen it (Pins), and moves its bytes outside it, so copies of
   several domains run at once and a pinned descriptor is never closed under a
   transfer.

   Locks nest in one order: a file's sync lock, then the table's lock. Each
   counts among the locks its thread holds ([take]), so a call of rig's or of
   the disk's from a signal handler, finaliser or Gc.Memprof callback inside
   one raises [Invalid_argument] as it starts: rig checks the calls that reach
   [Io], and [of_file], [create_file] and [barrier] check themselves
   ([guarded]).

   The table's lock is held across the system's open and close of a
   descriptor, and a file's sync lock across its sync, which waits for the
   disk: a descriptor half opened, or a sync that two barriers ran at once,
   would lose a failure the file must keep. A call of rig's from a callback
   during one raises. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type pages =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* The stubs raise [Unix.Unix_error] where a system call fails. *)

external open_path : string -> int -> int -> int * int * string
  = "caml_rig_disk_open"

external close : int -> unit = "caml_rig_disk_close"
external sync : int -> unit = "caml_rig_disk_sync"

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

external descr : int -> Unix.file_descr = "caml_rig_disk_descr"

external advise : int -> pages option -> int -> int -> unit
  = "caml_rig_disk_advise"

external msync : pages -> unit = "caml_rig_disk_msync"
external watch_forks : unit -> unit = "caml_rig_disk_watch_forks" [@@noalloc]
external forks : unit -> int = "caml_rig_disk_forks" [@@noalloc]

(* Locks *)

(* The locks the calling thread holds, and a count of one more and one fewer
   ([rig_holds], [rig_hold] and [rig_unhold] in [rig_edge.h]). *)
external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]

external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]
external reraise : exn -> 'a = "%reraise"

(* The calling thread's signals blocked and given back ([rig_mask] and
   [rig_unmask] in [rig_edge.h]). *)
external mask : unit -> unit = "caml_rig_edge_mask" [@@noalloc]
external unmask : unit -> unit = "caml_rig_edge_unmask"

(* A function a caller outside rig makes, and that may take a lock, runs as
   [guarded fn f]: inside a lock it raises [Invalid_argument] naming [fn],
   having done nothing, since the call comes from a signal handler,
   finaliser or Gc.Memprof callback inside that lock and would wait for it or
   see its section half done. Otherwise it runs [f] with the thread's signals
   blocked ([rig_mask] and [rig_unmask] in [rig_edge.h]), which only the
   driver's waits unblock. *)
let guarded fn f =
  if holds () <> 0 then
    invalid_arg
      ("Rig_disk." ^ fn
     ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
        inside a lock of the calling thread's");
  mask ();
  match f () with
  | r ->
      unmask ();
      r
  | exception e ->
      unmask ();
      reraise e

(* [take m] takes [m], counted from before the take, so that a signal handler
   that runs inside a contended take sees it held. An exception that ends the
   take gives the count back first, having taken nothing: one a handler
   raised, or the error-checking [Mutex]'s [Sys_error] for a take by the
   thread that holds [m], which only a call that skipped its check makes and
   which raises [Invalid_argument]. *)
let take m =
  hold ();
  match Mutex.lock m with
  | () -> ()
  | exception e -> (
      unhold ();
      match e with
      | Sys_error why when String.starts_with ~prefix:"Mutex.lock:" why ->
          invalid_arg
            "Rig_disk: a lock taken again by the thread that holds it, from a \
             signal handler, finaliser or Gc.Memprof callback"
      | e -> reraise e)

(* Gives [m] back, then its count, with nothing between that allocates or
   polls. *)
let give m =
  Mutex.unlock m;
  unhold ()

(* [protect m g x] is [g x] holding [m], taken as the last step before the
   handler and given back as the first step of both exits. [g] is a toplevel
   function, so a call makes no closure. *)
let protect m g x =
  take m;
  match g x with
  | r ->
      give m;
      r
  | exception e ->
      give m;
      reraise e

(* [open_path]'s modes, and the descriptor it answers for a path that names
   no regular file. *)
let read_mode = 0
let write_mode = 1
let create_mode = 2
let not_regular = -1

(* Files *)

(* A file is named by its path and its identity, bytes that name it exactly and
   that no other file has, or [""] where the system gives none. A file without
   an identity keeps its descriptor until it is freed: nothing could tell its
   path's file apart at a reopen.

   A sync's failure is the file's for good. Linux reports a write it could not
   make to one sync and may then drop the bytes, so a later sync would return
   though they are lost. It may also forget the failure once no descriptor holds
   the file, so a descriptor closed for others' sake syncs the copies' writes
   first ([evict]). Syncs of one file take turns under its sync lock
   ([sync_file]): the system reports the failure to the first only. *)
type file = {
  path : string;
  writable : bool;
  size : int;
  identity : string;
  mutable fd : int; [@atomic] (* [-1] while closed *)
  mutable users : int; [@atomic] (* copies using [fd], [-1] while it closes *)
  mutable newer : file; [@atomic]
      (* links in the table's ring, [f] itself out of it *)
  mutable older : file;
  mutable pages : pages option; (* its mapping, once a borrow made it *)
  mutable unsynced : bool; [@atomic] (* written since its last sync *)
  mutable failure : exn; [@atomic]
      (* its first sync's [Unix_error], [No_failure] before one *)
  mutable syncing : int * Mutex.t; [@atomic]
      (* its sync lock and the forks it was made after, [no_lock] before *)
}

exception No_failure

let no_lock = (-1, Mutex.create ())

(* The ring's ends: a file of no path, never open. *)
let rec ring =
  {
    path = "";
    writable = false;
    size = 0;
    identity = "";
    fd = -1;
    users = 0;
    newer = ring;
    older = ring;
    pages = None;
    unsynced = false;
    failure = No_failure;
    syncing = no_lock;
  }

(* A file not yet open, out of the ring. Its links are set to itself after it
   is made, since a [let rec] would allocate the record twice. *)
let file ~path ~writable ~size ~identity =
  let f =
    {
      path;
      writable;
      size;
      identity;
      fd = -1;
      users = 0;
      newer = ring;
      older = ring;
      pages = None;
      unsynced = false;
      failure = No_failure;
      syncing = no_lock;
    }
  in
  f.newer <- f;
  f.older <- f;
  f

let sys_error f why = raise (Sys_error (strf "%s: %s" f.path why))

let unix_error f e = sys_error f (Unix.error_message e)

(* Keeps [x], the [Unix_error] a sync of [f] raised, unless an earlier one is
   kept. It allocates nothing, so no asynchronous exception falls between the
   failure and its keeping. *)
let keep f x =
  ignore (Atomic.Loc.compare_and_set [%atomic.loc f.failure] No_failure x : bool)

(* [f]'s sync lock, made at its first use, and anew in a forked child: a thread
   of its parent may have held it. It is taken before the table's lock, never
   after. *)
let rec sync_lock f =
  let ((made, m) as l) = f.syncing and n = forks () in
  if made = n then m
  else begin
    ignore
      (Atomic.Loc.compare_and_set [%atomic.loc f.syncing] l
         (n, Mutex.create ())
        : bool);
    sync_lock f
  end

(* Descriptors *)

(* The table holds the descriptors of files with an identity: at most
   [max_open], more only while copies pin them, the least recently used unpinned
   one closed to open another. They are in a ring through [ring], from the most
   recently used ([ring.newer]) to the least ([ring.older]). A pin leaves its
   file where it is and [trim] passes over pinned files, so a copy of the most
   recently used file changes no link, and an open or a copy takes the same few
   steps and allocates nothing however full the table is. *)
let max_open = 64
let in_table = Atomic.make 0

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

(* [locked g x] is [g x] under the table's lock. *)
let locked g x = protect (table_lock ()) g x

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
      ignore (Atomic.fetch_and_add in_table (-1));
      unlink f
    end;
    f.users <- 0;
    close fd
  end

(* Closes [f]'s descriptor unless a copy pins it. It claims [users] at 0, so no
   pin, and so no write, starts meanwhile. A file written since its last sync is
   synced first and its failure kept, under the table's lock. It is marked
   synced only then, so a barrier that finds it synced finds the failure. An
   asynchronous exception in the sync leaves it open and unsynced. *)
let evict f =
  if Atomic.Loc.compare_and_set [%atomic.loc f.users] 0 (-1) then
    match
      if f.unsynced then try sync f.fd with Unix.Unix_error _ as x -> keep f x
    with
    | () ->
        f.unsynced <- false;
        close_fd f
    | exception e ->
        f.users <- 0;
        Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* Closes the least recently used unpinned descriptors while more than [keep]
   are open. *)
let close_down keep =
  let f = ref ring.older in
  while Atomic.get in_table > keep && !f != ring do
    let g = !f in
    f := g.newer;
    if g.users = 0 then evict g
  done

(* Brings the table back to its bound, once the pins that took it beyond end. *)
let trim () = close_down max_open

let admit f fd =
  f.fd <- fd;
  if listed f then begin
    ignore (Atomic.fetch_and_add in_table 1);
    touch f
  end

(* Opens [path], closing every unpinned descriptor and trying once more if the
   process or the system has too many open. *)
let open_retrying path mode n =
  let before = Atomic.get in_table in
  match open_path path mode n with
  | r -> r
  | exception (Unix.Unix_error ((EMFILE | ENFILE), _, _) as e) ->
      close_down 0;
      if Atomic.get in_table < before then open_path path mode n else raise e

(* Opens [f]'s path again, which must still name [f]'s file. *)
let reopen f =
  match
    open_retrying f.path (if f.writable then write_mode else read_mode) 0
  with
  | fd, _, _ when fd = not_regular -> sys_error f "not a regular file"
  | fd, _, identity when String.equal identity f.identity -> admit f fd
  | fd, _, _ ->
      close fd;
      sys_error f "the path names another file since its buffers opened it"
  | exception Unix.Unix_error (e, _, _) -> unix_error f e

(* Pins

   A copy pins its file's descriptor, which no eviction closes while [users] is
   above 0. An open descriptor is pinned and unpinned by one step on [users],
   without the table's lock: an eviction closes a descriptor only by claiming
   [users] from 0, and a pin that finds the file closed reopens it under the
   lock. An unpin takes the lock only to make its file the most recently used or
   to bring the table back to its bound. What a pin or an unpin reads without
   the lock ([fd], [ring.newer], [in_table]) is atomic, as [users] is: an
   unpin's read of [in_table] and a trim's read of [users] are never both
   stale, so the table never stays beyond its bound once its pins end. *)

(* [f]'s descriptor, opened again by its path if it was closed, pinned. Holds
   the table's lock. A trim that raises ends the pin, which no caller would
   end: the exception leaves the call before its caller's handler. *)
let pinned f =
  if f.fd < 0 then reopen f;
  ignore (Atomic.Loc.fetch_and_add [%atomic.loc f.users] 1);
  match trim () with
  | () -> f.fd
  | exception e ->
      ignore (Atomic.Loc.fetch_and_add [%atomic.loc f.users] (-1));
      raise e

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
      ignore (Atomic.Loc.fetch_and_add [%atomic.loc f.users] (-1));
      locked pinned f
    end

let unpin f =
  if
    Atomic.Loc.fetch_and_add [%atomic.loc f.users] (-1) = 1
    && listed f
    && (ring.newer != f || Atomic.get in_table > max_open)
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
      "Rig.Buffer.create: DISK makes no memory; open a file with \
       Rig_disk.of_file or Rig_disk.create_file"

  (* Nothing reaches [f]: no pin is held. *)
  let free () f = locked close_fd f

  let read () f ~at ~dst ~len =
    let k =
      try moved f pread at dst len
      with Unix.Unix_error (e, _, _) -> unix_error f e
    in
    if k < len then
      sys_error f
        (strf "the file ends at byte %d, before byte %d" (at + k) (at + len))

  (* Rig writes no file opened for reading: its memory admits only reads
     ([Rig.Buffer.access]). The write marks [f] unsynced while it pins the
     descriptor, so an eviction after it syncs it, and a barrier's sync that
     began before it leaves it marked. *)
  let write () f ~at ~src ~len =
    let fd = pin f in
    match pwrite fd at src len with
    | (_ : int) ->
        f.unsynced <- true;
        unpin f
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        f.unsynced <- true;
        unpin f;
        begin match e with
        | Unix.Unix_error (e, _, _) -> unix_error f e
        | e -> Printexc.raise_with_backtrace e bt
        end

  (* Whether a mapping's failure says the file system can never map the file,
     whatever the moment: no mapping for its kind of file, or a file too large
     for the address space. Others, such as a lack of memory or of mappings, may
     pass. [Unix] has no error for macOS's ENOTSUP (45), and gives Windows's
     ERROR_NOT_SUPPORTED (50) and ERROR_FILE_INVALID (1006) negated. *)
  let never_maps = function
    | Unix.ENODEV | EINVAL | EOVERFLOW | EOPNOTSUPP -> true
    | EUNKNOWNERR (45 | -50 | -1006) -> true
    | _ -> false

  (* A file opened for writing maps shared, so the mapping is the file; one
     opened for reading maps copy-on-write. A file that another process cut
     below [f.size] grows back to it if opened for writing, and raises
     otherwise. *)
  let map f fd =
    let open Bigarray in
    Unix.map_file (descr fd) char c_layout f.writable [| f.size |]

  (* A failure that may pass, such as a file that cannot be reopened for the
     moment, raises; [None] says the file system can never map the file. *)
  let pages () f =
    match using f (map f) with
    | pages ->
        f.pages <- Some (Bigarray.array1_of_genarray pages);
        f.pages
    | exception Unix.Unix_error (e, _, _) when never_maps e -> None
    | exception Unix.Unix_error (e, _, _) -> unix_error f e

  (* A file that cannot be reopened gets no advice; its pages, mapped already,
     stay valid. *)
  let prefetch () f ~at ~len =
    try using f (fun fd -> advise fd f.pages at len) with Sys_error _ -> ()

  let stop () = ()
end

(* The disk *)

(* The first open of its name, as the library starts: it answers [Error] only in
   a process that failed before ([Rig.fail]). *)
let device =
  match
    Rig.open_io { name = "DISK"; io = (module Io); make = (fun () -> Ok ()) }
  with
  | Ok d -> d
  | Error why -> failwith (strf "Rig_disk.device: cannot open DISK: %s" why)

let open_file path mode n =
  if String.contains path '\000' then
    Error (strf "%s: the path holds a NUL byte" path)
  else
    let writable = mode <> read_mode in
    let opened () =
      match open_retrying path mode n with
      | fd, _, _ when fd = not_regular ->
          Error (strf "%s: not a regular file" path)
      | fd, size, identity ->
          let f = file ~path ~writable ~size ~identity in
          admit f fd;
          trim ();
          Ok f
      | exception Unix.Unix_error (e, _, _) ->
          Error (strf "%s: %s" path (Unix.error_message e))
    in
    match locked opened () with
    | Error _ as e -> e
    | Ok f -> (
        let access = if writable then Rig.Read_write else Read in
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

let of_file path =
  guarded "of_file" @@ fun () ->
  open_file path read_mode 0

let create_file path n =
  guarded "create_file" @@ fun () ->
  if n < 0 then invalid_argf "Rig_disk.create_file: %d bytes is negative" n;
  open_file path create_mode n

(* Syncs [f] and keeps a failure before it unpins the descriptor. It marks [f]
   synced while it pins the descriptor, so no eviction falls between the mark
   and the sync. Writes through a shared mapping reach the file by the mapping's
   own flush, which the descriptor's sync does not cover. *)
let synced f =
  let fd = pin f in
  f.unsynced <- false;
  match
    (try
       Option.iter msync f.pages;
       sync fd
     with Unix.Unix_error _ as x -> keep f x)
  with
  | () -> unpin f
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      unpin f;
      Printexc.raise_with_backtrace e bt

(* [synced f] under [f]'s sync lock, unless nothing wrote [f] since its last
   sync. The system reports a failure to one sync of a descriptor, so a
   barrier's sync would return while another's, or an eviction's, has yet to
   keep the failure: it waits for that one to end. A file with a mapping always
   syncs, since nothing marks writes through it. *)
let sync_due f = if f.unsynced || Option.is_some f.pages then synced f
let sync_file f = protect (sync_lock f) sync_due f

let barrier b =
  guarded "barrier" @@ fun () ->
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
      sync_file f;
      (* Until the descriptor is unpinned: collecting [b] would close it. *)
      ignore (Sys.opaque_identity b);
      match f.failure with Unix.Unix_error (e, _, _) -> unix_error f e | _ -> ()
