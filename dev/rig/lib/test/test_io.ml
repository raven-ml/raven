(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Io memory that drivers' devices reach through its pages. Pages is an io
   library over page-aligned host memory that logs its calls. *)

open Windtrap
module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let page_bytes = 1 lsl 16

module Pages = struct
  (* [during] runs as a read or a write starts; [mapped] is how [pages] answers.
     With [read_only], writes are refused and a region's pages are the process's
     own copy, as a file opened for reading maps. *)
  type t = {
    lock : Mutex.t;
    mutable calls : string list;
    mutable during : unit -> unit;
    mutable mapped : [ `Pages | `None | `Fails ];
    mutable read_only : bool;
  }

  (* Its bytes, and the host buffer that holds them on a page. *)
  type region = {
    bytes :
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
    keep : B.t;
  }

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let note t call = Mutex.protect t.lock (fun () -> t.calls <- call :: t.calls)
  let budget _ = max_int

  let alloc t n =
    note t "alloc";
    let keep = B.create C.host (Int.max n page_bytes) in
    Some
      { bytes = Bigarray.Array1.sub (B.bigarray Bigarray.char keep) 0 n; keep }

  let free t _ = note t "free"
  let base r = B.address r.keep

  let read t r ~at ~dst ~len =
    note t "read";
    t.during ();
    Support.move ~dst ~src:(base r + at) len

  let write t r ~at ~src ~len =
    note t "write";
    if t.read_only then invalid_arg "Pages.write: the region is for reading";
    t.during ();
    Support.move ~dst:(base r + at) ~src len

  let pages t r =
    note t "pages";
    match t.mapped with
    | `Pages when t.read_only ->
        let n = Bigarray.Array1.dim r.bytes in
        let own =
          Bigarray.Array1.sub
            (B.bigarray Bigarray.char (B.create C.host (Int.max n page_bytes)))
            0 n
        in
        Bigarray.Array1.blit r.bytes own;
        Some own
    | `Pages -> Some r.bytes
    | `None -> None
    | `Fails -> raise (Sys_error "too many open files")

  let prefetch t _ ~at:_ ~len:_ = note t "prefetch"
  let stop _ = ()
end

let opened = Atomic.make 0

let open_pages () =
  let t =
    {
      Pages.lock = Mutex.create ();
      calls = [];
      during = (fun () -> ());
      mapped = `Pages;
      read_only = false;
    }
  in
  let name = Printf.sprintf "io:pages-%d" (Atomic.fetch_and_add opened 1) in
  let io =
    require_ok ~pp:Format.pp_print_string
      (C.open_io (module Pages) ~name (fun () -> Ok t))
  in
  (io, t)

let count call (t : Pages.t) =
  Mutex.protect t.lock (fun () ->
      List.length (List.filter (( = ) call) t.calls))

let filled n c =
  let b = B.create C.host n in
  Bigarray.Array1.fill (B.bigarray Bigarray.char b) c;
  b

let bytes b =
  let ba = B.bigarray Bigarray.char b in
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

(* A copy on [d] from [src] into [dst], left queued. *)
let queued_copy d ~src ~dst =
  let part =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  ignore (C.submit (Sub.make ~reads:0 ~writes:0 ~waits:0 d [| part |]))

(* A driver's device borrows io memory through its pages, as host memory, mapped
   once whatever its borrows. *)
let test_borrow () =
  let io, t = open_pages () in
  let d, p = P.open_ "io:borrower" in
  let m = B.create io page_bytes in
  for _ = 1 to 3 do
    ignore (require_some (B.borrow d m))
  done;
  equal ~msg:"host mappings" (list int) [ page_bytes ] (P.host_maps p);
  equal ~msg:"pages asked" int 1 (count "pages" t)

(* Memory whose io device maps no pages borrows nowhere, and is asked once. *)
let test_no_pages () =
  let io, t = open_pages () in
  t.mapped <- `None;
  let d, _ = P.open_ "io:pageless" in
  let m = B.create io page_bytes in
  equal ~msg:"a first borrow" bool true (Option.is_none (B.borrow d m));
  equal ~msg:"a second borrow" bool true (Option.is_none (B.borrow d m));
  equal ~msg:"pages asked" int 1 (count "pages" t)

(* A failure of the memory alone when its pages are asked raises from the borrow
   and loses nothing: a later borrow asks again. *)
let test_pages_fail () =
  let io, t = open_pages () in
  t.mapped <- `Fails;
  let d, _ = P.open_ "io:pages-fail" in
  let m = B.create io page_bytes in
  raises_match
    (function Sys_error _ -> true | _ -> false)
    (fun () -> B.borrow d m);
  equal ~msg:"the io device" (option string) None (C.lost io);
  t.mapped <- `Pages;
  ignore (require_some (B.borrow d m));
  equal ~msg:"pages asked" int 2 (count "pages" t)

(* A borrow by a device other than the host asks the io device to read ahead;
   the host's does not. *)
let test_prefetch () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:prefetcher" in
  let m = B.create io page_bytes in
  ignore (require_some (B.borrow C.host m));
  equal ~msg:"after the host's borrow" int 0 (count "prefetch" t);
  ignore (require_some (B.borrow d m));
  at_least ~msg:"after a device's borrow" int ~than:1 (count "prefetch" t)

(* A copy out of io memory follows a device's write through its pages, and a
   copy into it follows a device's read of it. *)
let test_copy_order () =
  let io, _ = open_pages () in
  let d, p = P.open_ "io:copier" in
  let m = B.create io page_bytes in
  let on_d = require_some (B.borrow d m) in
  let staged = B.create d page_bytes in
  B.copy ~src:(filled page_bytes 'w') ~dst:staged;
  queued_copy d ~src:staged ~dst:on_d;
  let out = filled page_bytes '0' in
  B.copy ~src:m ~dst:out;
  equal ~msg:"read after the device's write" string
    (String.make page_bytes 'w')
    (bytes out);
  queued_copy d ~src:on_d ~dst:staged;
  B.copy ~src:(filled page_bytes 'h') ~dst:m;
  equal ~msg:"device work left" int 0 (P.queued p);
  let back = filled page_bytes '0' in
  B.copy ~src:staged ~dst:back;
  equal ~msg:"the device read before the write" string
    (String.make page_bytes 'w')
    (bytes back)

(* A host access of io memory waits for the work of a device that borrowed its
   pages. *)
let test_wait () =
  let io, _ = open_pages () in
  let d, p = P.open_ "io:waiter" in
  let m = B.create io page_bytes in
  let staged = B.create d page_bytes in
  queued_copy d ~src:staged ~dst:(require_some (B.borrow d m));
  equal ~msg:"queued" int 1 (P.queued p);
  B.wait m B.Read;
  equal ~msg:"after the wait" int 0 (P.queued p)

(* Io memory returns to its library once unreachable and its uses reached. *)
let test_free () =
  let io, t = open_pages () in
  let d, p = P.open_ "io:user" in
  (fun () ->
    let m = B.create io page_bytes in
    let s = Sub.make ~reads:1 ~writes:0 ~waits:0 d [||] in
    Sub.read s 0 (require_some (B.borrow d m));
    ignore (C.submit s))
    ();
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create io 0);
    ignore (B.create ~memory:Pinned d 8)
  in
  drain ();
  equal ~msg:"while its use is unreached" int 0 (count "free" t);
  ignore (P.run p);
  drain ();
  equal ~msg:"once it is reached" int 1 (count "free" t)

(* A host buffer of [c] bytes whose collection sets [collected]. *)
let watched collected n c =
  let b = filled n c in
  Gc.finalise (fun _ -> collected := true) b;
  b

(* Makes [t]'s reads and writes collect and drain [io] as they start, and [seen]
   what they saw: whether [collected] was set, and [io]'s frees so far. *)
let collecting t io collected seen =
  t.Pages.during <-
    (fun () ->
      Gc.full_major ();
      Gc.full_major ();
      ignore (B.create io 0);
      seen :=
        [
          Printf.sprintf "collected: %b" !collected;
          Printf.sprintf "frees: %d" (count "free" t);
        ])

(* Between io memory that maps no pages and a device's memory the host does not
   address, the bytes go through the staging memory, a piece at a time, both
   ways: every byte lands, across the pieces' edges. *)
let test_staged_device () =
  let io, t = open_pages () in
  t.mapped <- `None;
  let d, _ = P.open_ ~host_visible:false "io:staged-device" in
  let n = (100 lsl 20) + 4096 in
  let byte i = Char.unsafe_chr ((i + ((i lsr 16) * 13)) land 255) in
  let h = B.create C.host n in
  let ba = B.bigarray Bigarray.char h in
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set ba i (byte i)
  done;
  let m = B.create io n and dev = B.create d n and back = B.create io n in
  B.copy ~src:h ~dst:m;
  B.copy ~src:m ~dst:dev;
  B.copy ~src:dev ~dst:back;
  let out = B.create C.host n in
  B.copy ~src:back ~dst:out;
  let got = B.bigarray Bigarray.char out in
  let wrong = ref 0 in
  for i = 0 to n - 1 do
    if Bigarray.Array1.unsafe_get got i <> byte i then incr wrong
  done;
  equal ~msg:"bytes that differ" int 0 !wrong

(* No device's work reaches io memory itself, only a borrow of its pages: a slot
   refuses an io device's buffer and takes the borrow. *)
let test_slots () =
  let io, _ = open_pages () in
  let d, _ = P.open_ "io:slots" in
  let m = B.create io page_bytes in
  let s = Sub.make ~reads:1 ~writes:1 ~waits:0 d [||] in
  raises_match Exn.invalid_arg (fun () -> Sub.read s 0 m);
  raises_match Exn.invalid_arg (fun () -> Sub.write s 0 m);
  Sub.read s 0 (require_some (B.borrow d m));
  Sub.write s 0 (B.create d 8);
  ignore (C.submit s)

(* The minor words [f ()] allocates. *)
let minor_words f =
  let before = Gc.minor_words () in
  f ();
  int_of_float (Gc.minor_words () -. before)

(* Draining one collected io memory returns it to its library for few words,
   this library's io release included, beyond what an idle drain costs. *)
let test_release_words () =
  let io, t = open_pages () in
  let drain () = ignore (Sys.opaque_identity (B.create io 0)) in
  drain ();
  let idle = minor_words drain in
  let[@inline never] dropped () =
    ignore (Sys.opaque_identity (B.create io 64))
  in
  dropped ();
  Gc.full_major ();
  let release = minor_words drain in
  equal ~msg:"freed" int 1 (count "free" t);
  at_most ~msg:"a release's words over an idle drain's" int ~than:40
    (release - idle)

(* A copy into io memory keeps both buffers until the write returned: a
   collection during it frees neither the source nor the io memory. *)
let test_write_keeps () =
  let io, t = open_pages () in
  let collected = ref false and seen = ref [] in
  collecting t io collected seen;
  B.copy ~src:(watched collected page_bytes 'k') ~dst:(B.create io page_bytes);
  equal (list string) [ "collected: false"; "frees: 0" ] !seen

(* A copy from io memory keeps both buffers until the read returned. *)
let test_read_keeps () =
  let io, t = open_pages () in
  let collected = ref false and seen = ref [] in
  let src = B.create io page_bytes in
  B.copy ~src:(filled page_bytes 'k') ~dst:src;
  collecting t io collected seen;
  B.copy ~src ~dst:(watched collected page_bytes 'x');
  equal (list string) [ "collected: false"; "frees: 0" ] !seen;
  ignore (Sys.opaque_identity src)

(* Faults and refusals *)

let lost d = function C.Lost (d', _) -> C.equal d d' | _ -> false

(* A fault an io read raises loses the io device; a failure of the memory alone,
   such as a file truncated since it was opened, reaches the caller and loses
   nothing. *)
let test_faults () =
  let io, t = open_pages () in
  let m = B.create io page_bytes and out = filled page_bytes '0' in
  t.during <- (fun () -> raise (Sys_error "truncated"));
  raises (Sys_error "truncated") (fun () -> B.copy ~src:m ~dst:out);
  equal ~msg:"after Sys_error" (option string) None (C.lost io);
  t.during <- (fun () -> ());
  B.copy ~src:m ~dst:out;
  t.during <- (fun () -> raise (Pages.Fault "the link went down"));
  raises_match (lost io) (fun () -> B.copy ~src:out ~dst:m);
  equal ~msg:"after Fault" (option string) (Some "the link went down")
    (C.lost io)

(* A region an io library made is a buffer of its io device only, by its
   library's key, never exclusive. *)
let test_of_io () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:not-io" in
  let r = Option.get (Pages.alloc t page_bytes) in
  let other : Pages.region Type.Id.t = Type.Id.make () in
  raises_match Exn.invalid_arg (fun () -> B.of_io d Pages.region_key r 8);
  raises_match Exn.invalid_arg (fun () -> B.of_io io other r 8);
  raises_match Exn.invalid_arg (fun () -> B.of_io io Pages.region_key r (-1));
  let b = B.of_io io Pages.region_key r page_bytes in
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool false (C.Claim.exclusive c b))

(* Io memory has no address or object of a driver, and loads no code. *)
let test_io_refusals () =
  let io, _ = open_pages () in
  let m = B.create io 8 in
  raises_match Exn.invalid_arg (fun () -> B.address m);
  raises_match Exn.invalid_arg (fun () -> B.handle m);
  raises_match Exn.invalid_arg (fun () -> C.Program.load io "code:8")

(* A copy reads and writes io memory through its device's reads and writes,
   never through pages of the process's own: into memory held for reading it
   raises the device's refusal, and out of it, it reads the memory. *)
let test_copy_read_only () =
  let io, t = open_pages () in
  let d, _ = P.open_ ~host_visible:false "io:read-only" in
  let m = B.create io page_bytes in
  B.copy ~src:(filled page_bytes 'm') ~dst:m;
  t.read_only <- true;
  let into = B.create d page_bytes in
  B.copy ~src:(filled page_bytes 'd') ~dst:into;
  raises_match Exn.invalid_arg (fun () -> B.copy ~src:into ~dst:m);
  let own = require_some (B.borrow C.host m) in
  Bigarray.Array1.fill (B.bigarray Bigarray.char own) 'o';
  let out = B.create d page_bytes and back = B.create C.host page_bytes in
  B.copy ~src:m ~dst:out;
  B.copy ~src:out ~dst:back;
  equal string (String.make page_bytes 'm') (bytes back)

let tests =
  [
    group ~timeout "io memory"
      [
        test "a device borrows io memory through its pages, mapped once"
          test_borrow;
        test "memory whose device maps no pages borrows nowhere, asked once"
          test_no_pages;
        test "a failure asking for pages raises and is asked again"
          test_pages_fail;
        test "a device's borrow of io memory reads ahead, the host's does not"
          test_prefetch;
        test "copies of io memory follow devices' work through its pages"
          test_copy_order;
        test "a host access of io memory waits for a borrower's work" test_wait;
        test "io memory returns once unreachable and its uses reached" test_free;
        test "a collected io memory's release costs few words"
          test_release_words;
        test "a slot refuses io memory and takes a borrow of its pages"
          test_slots;
        test "io memory and a device's memory copy through staging, both ways"
          test_staged_device;
        test "a fault of io loses its device, a failure of its memory nothing"
          test_faults;
        test "a region an io library made is a buffer of its device alone"
          test_of_io;
        test "io memory has no driver address or object and loads no code"
          test_io_refusals;
        test "a copy into io memory keeps its buffers until written"
          test_write_keeps;
        test "a copy from io memory keeps its buffers until read"
          test_read_keeps;
        xfail
          ~reason:
            "a copy run on a device's queue reaches io memory through its \
             pages, which for memory held for reading are the process's own \
             copy"
          (test
             "a copy reads and writes io memory held for reading through its \
              device"
             test_copy_read_only);
      ];
  ]

let () = exit (run "rig.io" tests)
