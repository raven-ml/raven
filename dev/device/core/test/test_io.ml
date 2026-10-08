(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Io memory that drivers' devices reach through its pages. Pages is an io
   library over page-aligned host memory that logs its calls. *)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled
module Support = Device_core_support

let timeout = 60.
let page_bytes = 1 lsl 16

module Pages = struct
  type t = { lock : Mutex.t; mutable calls : string list }

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
    Support.move ~dst ~src:(base r + at) len

  let write t r ~at ~src ~len =
    note t "write";
    Support.move ~dst:(base r + at) ~src len

  let pages t r =
    note t "pages";
    Some r.bytes

  let prefetch t _ ~at:_ ~len:_ = note t "prefetch"
  let stop _ = ()
end

let opened = Atomic.make 0

let open_pages () =
  let t = { Pages.lock = Mutex.create (); calls = [] } in
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

let tests =
  [
    group ~timeout "io memory"
      [
        test "a device borrows io memory through its pages, mapped once"
          test_borrow;
        test "a device's borrow of io memory reads ahead, the host's does not"
          test_prefetch;
        test "copies of io memory follow devices' work through its pages"
          test_copy_order;
        test "io memory returns once unreachable and its uses reached" test_free;
      ];
  ]

let () = exit (run "device_core.io" tests)
