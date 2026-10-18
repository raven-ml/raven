(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Io memory that drivers' devices reach through its pages. Pages is an io
   library over page-aligned host memory that logs its calls. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits (Sub.Run.make s) ~buffers

let timeout = 60.
let page_bytes = 1 lsl 16

module Pages = struct
  (* [during] runs as a read or a write starts; [mapped] is how [pages]
     answers. *)
  type t = {
    lock : Mutex.t;
    mutable calls : string list;
    mutable during : unit -> unit;
    mutable mapped : [ `Pages | `None | `Fails ];
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
    let keep = B.create Rig.host (Int.max n page_bytes) in
    Some
      { bytes = Bigarray.Array1.sub (B.export keep Bigarray.char) 0 n; keep }

  let free t _ = note t "free"
  let base r = Support.Reader.host r.keep

  let read t r ~at ~dst ~len =
    note t "read";
    t.during ();
    Support.move ~dst ~src:(base r + at) len

  let write t r ~at ~src ~len =
    note t "write";
    t.during ();
    Support.move ~dst:(base r + at) ~src len

  let pages t r =
    note t "pages";
    match t.mapped with
    | `Pages -> Some r.bytes
    | `None -> None
    | `Fails -> raise (Sys_error "too many open files")

  let prefetch t _ ~at:_ ~len:_ = note t "prefetch"
  let stop _ = ()
end

let opened = Atomic.make 0

let pages () =
  {
    Pages.lock = Mutex.create ();
    calls = [];
    during = (fun () -> ());
    mapped = `Pages;
  }

let open_pages () =
  let t = pages () in
  let name = Printf.sprintf "io:pages-%d" (Atomic.fetch_and_add opened 1) in
  let io =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_io { name; io = (module Pages); make = (fun () -> Ok t) })
  in
  (io, t)

let count call (t : Pages.t) =
  Mutex.protect t.lock (fun () ->
      List.length (List.filter (( = ) call) t.calls))

let filled n c =
  let b = B.create Rig.host n in
  B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a c);
  b

let bytes b =
  B.read b Bigarray.char (fun ba ->
      String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba))

(* A copy on [d] from [src] into [dst], left queued. *)
let queued_copy d ~src ~dst =
  let part =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  ignore (submit (Sub.make d [| part |]))

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

(* Looking for overlap maps no pages. Once borrowed by the host, an io region
   and bigarrays over its pages share the same byte ranges. *)
let test_overlaps () =
  let io, t = open_pages () in
  let m = B.create io page_bytes and other = B.create Rig.host 32 in
  equal ~msg:"before any borrow" (pair bool bool) (false, false)
    (B.overlaps m other, B.overlaps other m);
  equal ~msg:"inspection maps no pages" int 0 (count "pages" t);
  let host = require_some (B.borrow Rig.host m) in
  let ba = B.read host Bigarray.char Fun.id in
  let wrapped = B.of_bigarray (Bigarray.Array1.sub ba 8 16) in
  let shared = B.view m ~first:12 ~length:8 in
  let disjoint = B.view m ~first:24 ~length:8 in
  equal ~msg:"a bigarray and io view over shared bytes" (pair bool bool)
    (true, true)
    (B.overlaps shared wrapped, B.overlaps wrapped shared);
  equal ~msg:"adjacent bytes" (pair bool bool) (false, false)
    (B.overlaps disjoint wrapped, B.overlaps wrapped disjoint);
  raises_match ~msg:"claims reject the same overlap"
    (Exn.invalid_arg ~substring:"a donated buffer overlaps another") (fun () ->
      Rig.Claim.with_ ~read:[ wrapped ] ~donate:[ shared ] (fun _ -> ()));
  Rig.Claim.with_ ~read:[ wrapped ] ~donate:[ disjoint ] (fun _ -> ());
  equal ~msg:"inspection reuses the mapping" int 1 (count "pages" t)

(* rig_buffers_overlap of two byte ranges answers as Buffer.overlaps of the
   views over them, over buffers of every kind of place: host memory, a view
   of it and a device's borrow of it; io memory, its views, a device's borrow,
   the host borrow that maps its pages and a bigarray over those pages; memory
   a free killed and memory made after it; an empty view. Besides any two
   buffers, pairs put two ranges in one buffer, a borrow beside its memory,
   the dead buffer beside any, or the bigarray beside a buffer of the io
   memory, whose overlap only the cached host address shows. *)
let bigarray_at = 0
let io_family = [| 1; 2; 3; 4; 5 |]
let freed_at = 9

(* The buffers, made at the first case: the bigarray at [bigarray_at], the io
   memory's buffers at [io_family], the dead one at [freed_at]. *)
let universe =
  lazy
    (let io, _ = open_pages () in
     let d, _ = P.open_ "io:overlap" in
     let h = B.create Rig.host page_bytes and m = B.create io page_bytes in
     let mapped = require_some (B.borrow Rig.host m) in
     let pages = B.read mapped Bigarray.char Fun.id in
     let freed = B.create Rig.host 64 in
     ignore (B.free freed : bool);
     [|
       B.of_bigarray (Bigarray.Array1.sub pages 8 16);
       m;
       mapped;
       B.view m ~first:12 ~length:8;
       B.view m ~first:24 ~length:8;
       require_some (B.borrow d m);
       h;
       B.view h ~first:16 ~length:32;
       require_some (B.borrow d h);
       freed;
       B.create Rig.host 64;
       B.view h ~first:0 ~length:0;
     |])

let universe_size = 12

(* The borrows beside their memory: [(borrow, memory)]. *)
let borrows = [ (2, 1); (5, 1); (8, 6) ]

(* A range of at most 32 bytes from the first 32 of [b]. *)
let range b (first, bytes) =
  let first = Int.min first (B.length b) in
  (first, Int.min bytes (B.length b - first))

(* Whether two ranges share bytes, by their buffers' spans, for a dead
   buffer, whose views raise. The dead buffer is freed host memory, whose
   place no io page shares, as the pages are live host memory of their own:
   the ranges share bytes iff they intersect in one space. *)
let spans_overlap a (af, an) b (bf, bn) =
  let sa, fa = Support.Reader.span a and sb, fb = Support.Reader.span b in
  an > 0 && bn > 0 && sa = sb
  && fa + af < fb + bf + bn
  && fb + bf < fa + af + an

(* Whether two ranges share bytes, by Buffer.overlaps of their views where
   both buffers are live. *)
let reference a ra b rb =
  let view b (first, bytes) = B.view b ~first ~length:bytes in
  if B.dead a <> None || B.dead b <> None then spans_overlap a ra b rb
  else B.overlaps (view a ra) (view b rb)

let range_gen = Gen.pair (Gen.int_range 0 32) (Gen.int_range 0 32)

(* Pairs whose second buffer holds the first's bytes [shift] bytes in: one
   buffer twice, but the empty view; a borrow beside its memory; the bigarray
   beside the io memory, its host borrow and its device's borrow, which hold its
   bytes from 8. *)
let aligned =
  let pair shift (i, j) = Gen.constant (i, j, shift) in
  [
    Gen.map (fun k -> (k, k, 0)) (Gen.int_range 0 (universe_size - 2));
    Gen.one_of (List.map (pair 0) borrows);
    Gen.one_of (List.map (fun j -> pair 8 (bigarray_at, j)) [ 1; 2; 5 ]);
  ]

(* An aligned pair and two non-empty ranges within its first 8 bytes, which
   every buffer but the empty view holds: the second over the first's bytes, or
   after them. Ranges drawn independently reach these in few cases. *)
let aligned_case =
  Gen.bind
    (Gen.pair (Gen.one_of aligned) (Gen.int_range 0 6))
    (fun ((i, j, shift), f) ->
      Gen.bind (Gen.int_range 1 (7 - f)) (fun n ->
          Gen.map
            (fun (after, m) ->
              let second =
                if after then (f + n + shift, m) else (f + shift, n)
              in
              ((i, j), ((f, n), second)))
            (Gen.pair Gen.bool (Gen.int_range 1 (8 - f - n)))))

let overlap_case =
  let any = Gen.int_range 0 (universe_size - 1) in
  let io = Gen.map (fun k -> io_family.(k)) (Gen.int_range 0 4) in
  let beside_pages = Gen.pair (Gen.constant bigarray_at) io in
  let one_buffer = Gen.map (fun k -> (k, k)) any in
  let borrow = Gen.one_of (List.map Gen.constant borrows) in
  let dead = Gen.pair (Gen.constant freed_at) any in
  let drawn =
    Gen.pair
      (Gen.one_of [ Gen.pair any any; one_buffer; borrow; dead; beside_pages ])
      (Gen.pair range_gen range_gen)
  in
  Gen.frequency [ (1, drawn); (1, aligned_case) ]

let overlap_law ((i, j), (ri, rj)) =
  let all = Lazy.force universe in
  let a = all.(i) and b = all.(j) in
  let ((af, an) as ra) = range a ri and ((bf, bn) as rb) = range b rj in
  let expected = reference a ra b rb in
  cover "overlapping" expected;
  cover "disjoint" (not expected);
  cover "disjoint slices of one buffer"
    (i = j && (not expected) && an > 0 && bn > 0);
  cover "overlapping slices of one buffer" (i = j && expected);
  cover "a borrow beside its memory, overlapping"
    (expected && List.mem (i, j) borrows);
  cover "a bigarray over io pages beside the io memory, overlapping"
    (expected && i = bigarray_at && Array.mem j io_family);
  cover "a part of a dead buffer"
    (i = freed_at && B.dead a <> None && an < B.length a);
  cover "an empty range" (an = 0 || bn = 0);
  equal ~msg:"rig_buffers_overlap" bool expected
    (Support.Reader.overlap a af an b bf bn)

(* Ranges at a buffer's bounds lie within it: an empty range at its end
   overlaps nothing, its last byte overlaps itself, and two halves of it share
   no byte. A range past the bounds aborts the process, which no test in the
   process can observe. *)
let test_overlap_bounds () =
  let a = B.create Rig.host 64 in
  let overlap af an bf bn = Support.Reader.overlap a af an a bf bn in
  equal ~msg:"an empty range at the end" bool false (overlap 64 0 0 64);
  equal ~msg:"the last byte" bool true (overlap 63 1 63 1);
  equal ~msg:"two halves" bool false (overlap 0 32 32 32);
  equal ~msg:"the whole buffer" bool true (overlap 0 64 0 64)

(* Publishing a host mapping cannot move the coordinates of a span already read
   by an overlap check. Views retain their offsets in the same space. *)
let test_span_stable () =
  let io, t = open_pages () in
  let m = B.create io page_bytes in
  let space, first = Support.Reader.span m in
  let host = require_some (B.borrow Rig.host m) in
  let view = B.view m ~first:12 ~length:8 in
  equal ~msg:"publishing pages preserves the memory's coordinates"
    (pair int int)
    (space, first + 12)
    (Support.Reader.span view);
  equal ~msg:"a borrow has its memory's coordinates" (pair int int)
    (space, first)
    (Support.Reader.span host);
  equal ~msg:"one mapping" int 1 (count "pages" t)

(* A borrow back on the io device is its memory: no driver object, no host or
   device address, not borrowed, in the memory's span. *)
let test_borrow_back () =
  let io, _ = open_pages () in
  let m = B.create io page_bytes in
  let back = require_some (B.borrow io (require_some (B.borrow Rig.host m))) in
  let answer f = try Some (f ()) with Invalid_argument _ -> None in
  let coordinates b =
    ( answer (fun () -> B.handle b),
      Support.Reader.host b,
      Support.Reader.span b,
      B.is_borrowed b )
  in
  equal ~msg:"device" bool true (Rig.equal io (B.device back));
  equal
    (quad (option nativeint) int (pair int int) bool)
    (coordinates m) (coordinates back)

(* Nine donations use the sorted check. Each io memory keeps its identity span
   as well as the host range through which a bigarray aliases it. *)
let test_sorted_overlaps () =
  let io, t = open_pages () in
  let memories = List.init 9 (fun _ -> B.create io page_bytes) in
  let target = List.hd memories in
  let host = require_some (B.borrow Rig.host target) in
  let ba = B.read host Bigarray.char Fun.id in
  let wrapped first = B.of_bigarray (Bigarray.Array1.sub ba first 8) in
  let donate = List.map (fun m -> B.view m ~first:8 ~length:16) memories in
  raises_match ~msg:"sorted claims reject an aliased host range"
    (Exn.invalid_arg ~substring:"a donated buffer overlaps another") (fun () ->
      Rig.Claim.with_ ~read:[ wrapped 12 ] ~donate (fun _ -> ()));
  Rig.Claim.with_ ~read:[ wrapped 24 ] ~donate (fun _ -> ());
  equal ~msg:"the sorted check maps no other memory" int 1 (count "pages" t)

let test_empty_host_borrow () =
  let io, t = open_pages () in
  let region = require_some (Pages.alloc t 0) in
  let m = B.of_io io Pages.region_key region ~access:Read_write 0 in
  let host = require_some (B.borrow Rig.host m) in
  not_equal ~msg:"the empty host borrow keeps a host address" int 0
    (Support.Reader.host host);
  equal ~msg:"no bytes overlap" bool false (B.overlaps m host);
  equal ~msg:"an empty borrow maps no pages" int 0 (count "pages" t)

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
  equal ~msg:"the io device" (option string) None (Rig.lost io);
  t.mapped <- `Pages;
  ignore (require_some (B.borrow d m));
  equal ~msg:"pages asked" int 2 (count "pages" t)

(* A borrow by a device other than the host asks the io device to read ahead;
   the host's does not. *)
let test_prefetch () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:prefetcher" in
  let m = B.create io page_bytes in
  ignore (require_some (B.borrow Rig.host m));
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
  B.wait m Rig.Read;
  equal ~msg:"after the wait" int 0 (P.queued p)

(* Io memory returns to its library once unreachable and its uses reached. *)
let test_free () =
  let io, t = open_pages () in
  let d, p = P.open_ "io:user" in
  (fun () ->
    let m = B.create io page_bytes in
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    ignore (submit s ~buffers:[| require_some (B.borrow d m) |]))
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
  let byte i =
    Char.unsafe_chr ((i + ((i lsr 16) * 13) + (i lsr 24)) land 255)
  in
  let h = B.create Rig.host n in
  B.write h Bigarray.char (fun ba ->
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set ba i (byte i)
      done);
  let m = B.create io n and dev = B.create d n and back = B.create io n in
  B.copy ~src:h ~dst:m;
  B.copy ~src:m ~dst:dev;
  B.copy ~src:dev ~dst:back;
  let out = B.create Rig.host n in
  B.copy ~src:back ~dst:out;
  let got = B.export out Bigarray.char in
  let wrong = ref 0 in
  for i = 0 to n - 1 do
    if Bigarray.Array1.unsafe_get got i <> byte i then incr wrong
  done;
  equal ~msg:"bytes that differ" int 0 !wrong

(* No device's work reaches io memory itself, only a borrow of its pages: a run
   refuses an io device's buffer and takes the borrow. *)
let test_run_borrows () =
  let io, _ = open_pages () in
  let d, _ = P.open_ "io:slots" in
  let m = B.create io page_bytes in
  let s = Sub.make d [| Support.touching [| Sub.read; Sub.write |] |] in
  let own = B.create d 8 and borrowed = require_some (B.borrow d m) in
  raises_match Exn.invalid_arg (fun () ->
      ignore (submit s ~buffers:[| m; own |]));
  raises_match Exn.invalid_arg (fun () ->
      ignore (submit s ~buffers:[| borrowed; m |]));
  ignore (submit s ~buffers:[| borrowed; own |])

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

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* A fault an io read raises loses the io device; a failure of the memory alone,
   such as a file truncated since it was opened, reaches the caller and loses
   nothing. *)
let test_faults () =
  let io, t = open_pages () in
  let m = B.create io page_bytes and out = filled page_bytes '0' in
  t.during <- (fun () -> raise (Sys_error "truncated"));
  raises (Sys_error "truncated") (fun () -> B.copy ~src:m ~dst:out);
  equal ~msg:"after Sys_error" (option string) None (Rig.lost io);
  t.during <- (fun () -> ());
  B.copy ~src:m ~dst:out;
  t.during <- (fun () -> raise (Pages.Fault "the link went down"));
  raises_match (lost io) (fun () -> B.copy ~src:out ~dst:m);
  equal ~msg:"after Fault" (option string) (Some "the link went down")
    (Rig.lost io)

(* A copy from or into a borrow of io memory's pages goes through the io
   library's read or write, never the pages: toward host memory, toward a
   device's memory, where the device maps none of the pages, and back, and so
   do the blits and a fill, and a copy between two views of the one
   allocation. *)
let test_borrow_copies () =
  let io, t = open_pages () in
  let d, p = P.open_ ~host_visible:false "io:borrow-copies" in
  let m = B.create io page_bytes in
  let hb = require_some (B.borrow Rig.host m) in
  let calls () = (count "read" t, count "write" t) in
  let calls_of f =
    let r, w = calls () in
    f ();
    let r', w' = calls () in
    (r' - r, w' - w)
  in
  let reads_writes = pair int int in
  equal ~msg:"from the host into the borrow" reads_writes (0, 1)
    (calls_of (fun () -> B.copy ~src:(filled page_bytes 'a') ~dst:hb));
  let out = filled page_bytes '0' in
  equal ~msg:"from the borrow into the host" reads_writes (1, 0)
    (calls_of (fun () -> B.copy ~src:hb ~dst:out));
  equal ~msg:"the host's bytes" string (String.make page_bytes 'a') (bytes out);
  let on_d = B.create d page_bytes in
  equal ~msg:"from the borrow into the device" reads_writes (1, 0)
    (calls_of (fun () -> B.copy ~src:hb ~dst:on_d));
  equal ~msg:"from the device into the borrow" reads_writes (0, 1)
    (calls_of (fun () -> B.copy ~src:on_d ~dst:hb));
  equal ~msg:"the device's maps of the pages" bool false
    (List.mem page_bytes (P.host_maps p));
  let into = Bytes.create page_bytes in
  equal ~msg:"a blit out of the borrow" reads_writes (1, 0)
    (calls_of (fun () -> B.blit_to_bytes hb 0 into 0 page_bytes));
  equal ~msg:"a blit into the borrow" reads_writes (0, 1)
    (calls_of (fun () ->
         B.blit_from_string (Bytes.to_string into) 0 hb 0 page_bytes));
  equal ~msg:"a fill of the borrow" reads_writes (0, 1)
    (calls_of (fun () -> B.fill hb 'c'));
  let half = page_bytes / 2 in
  B.copy ~src:(filled half 'b') ~dst:(B.view hb ~first:0 ~length:half);
  equal ~msg:"within the allocation" reads_writes (1, 1)
    (calls_of (fun () ->
         B.copy
           ~src:(B.view hb ~first:0 ~length:half)
           ~dst:(B.view hb ~first:half ~length:half)));
  B.copy ~src:m ~dst:out;
  equal ~msg:"the moved bytes" string (String.make page_bytes 'b') (bytes out)

(* Io memory and a borrow of its pages copy to and from a device of another
   machine, through the host's staging memory: the io library reads and
   writes, and every byte lands. *)
let test_far_copies () =
  let io, t = open_pages () in
  let machine = Support.machine "io-far" in
  let far =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine
         {
           name = "io:far";
           driver = (module P);
           make =
             (fun ~word ->
               Ok (P.bind ~word (P.make ~host_visible:false ~peers:false ())));
         })
  in
  let m = B.create io page_bytes in
  let hb = require_some (B.borrow Rig.host m) in
  let on_far = B.create far page_bytes in
  let reads = count "read" t and writes = count "write" t in
  B.copy ~src:(filled page_bytes 'f') ~dst:on_far;
  B.copy ~src:on_far ~dst:m;
  B.copy ~src:(filled page_bytes 'g') ~dst:hb;
  B.copy ~src:hb ~dst:on_far;
  let out = filled page_bytes '0' in
  B.copy ~src:on_far ~dst:out;
  equal ~msg:"from the borrow, through the far device" string
    (String.make page_bytes 'g') (bytes out);
  B.copy ~src:(filled page_bytes 'h') ~dst:on_far;
  B.copy ~src:on_far ~dst:hb;
  B.copy ~src:m ~dst:on_far;
  B.copy ~src:on_far ~dst:out;
  equal ~msg:"into the borrow, then from the io memory" string
    (String.make page_bytes 'h') (bytes out);
  equal ~msg:"reads and writes of the io library" (pair int int) (2, 3)
    (count "read" t - reads, count "write" t - writes)

(* A fault of an io read under a copy from a borrow of its pages loses the io
   device, which owns the memory, and leaves the host. *)
let test_borrow_fault () =
  let io, t = open_pages () in
  let hb = require_some (B.borrow Rig.host (B.create io page_bytes)) in
  t.during <- (fun () -> raise (Pages.Fault "the link went down"));
  raises_match (lost io) (fun () ->
      B.copy ~src:hb ~dst:(filled page_bytes '0'));
  equal ~msg:"the io device" (option string) (Some "the link went down")
    (Rig.lost io);
  equal ~msg:"the host" (option string) None (Rig.lost Rig.host)

(* A region an io library made is a buffer of its io device only, by its
   library's key, never exclusive. *)
let test_of_io () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:not-io" in
  let r = Option.get (Pages.alloc t page_bytes) in
  let other : Pages.region Type.Id.t = Type.Id.make () in
  raises_match Exn.invalid_arg (fun () ->
      B.of_io d Pages.region_key r ~access:Read_write 8);
  raises_match Exn.invalid_arg (fun () ->
      B.of_io io other r ~access:Read_write 8);
  raises_match Exn.invalid_arg (fun () ->
      B.of_io io Pages.region_key r ~access:Read_write (-1));
  let b = B.of_io io Pages.region_key r ~access:Read_write page_bytes in
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      equal bool false (Rig.Claim.exclusive c b))

(* A closed io device makes no buffer of a region, also once a later device
   took its slot and its record may be gone. A memory device's last value,
   which the io device's slot keeps and the Polled device after it starts
   from, shows that the slot passed from one to the next. *)
let test_of_io_closed () =
  let m = Result.get_ok (Rig.memory_device "io:before-closed") in
  let p =
    Rig.submit (Rig.Submission.Run.make (Rig.Submission.make m [||]))
      ~buffers:[||]
  in
  Rig.Point.wait p;
  Rig.close m;
  Rig.free_cache m;
  let io, t = open_pages () in
  let r = Option.get (Pages.alloc t page_bytes) in
  Rig.close io;
  Rig.free_cache io;
  let later, _ = P.open_ "io:after-closed" in
  equal ~msg:"the slot passed on" int (Rig.Point.value p) (Rig.submitted later);
  raises_match (lost io) (fun () ->
      B.of_io io Pages.region_key r ~access:Read_write page_bytes);
  Gc.full_major ();
  Rig.close later

(* Io memory has no address or object of a driver, and loads no code. *)
let test_io_refusals () =
  let io, _ = open_pages () in
  let m = B.create io 8 in
  raises_match Exn.invalid_arg (fun () -> B.aligned m ~at:0 8);
  raises_match Exn.invalid_arg (fun () -> B.handle m);
  raises_match Exn.invalid_arg (fun () -> Rig.Image.load io "code:8")

let pp_access ppf a =
  Format.pp_print_string ppf
    (match a with Rig.Read -> "Read" | Read_write -> "Read_write")

let access = Testable.make ~pp:pp_access ~equal:( = )

(* A device's work never writes memory that admits only reads: a run that writes
   it, a copy part into it, a run's copy into it and fixed memory written are
   refused, and reading it is not. *)
let test_read_runs () =
  let io, t = open_pages () in
  let d, _ = P.open_ ~host_visible:false "io:read-runs" in
  let r = Option.get (Pages.alloc t page_bytes) in
  let m = B.of_io io Pages.region_key r ~access:Read page_bytes in
  let on_d = require_some (B.borrow d m) and own = B.create d page_bytes in
  let s = Sub.make d [| Support.touching [| Sub.read; Sub.write |] |] in
  ignore (submit s ~buffers:[| on_d; own |]);
  raises_match ~msg:"a run's write" Exn.invalid_arg (fun () ->
      ignore (submit s ~buffers:[| own; on_d |]));
  let unordered = Sub.make d [| Support.touching [| Sub.unordered |] |] in
  raises_match ~msg:"a run's unordered use" Exn.invalid_arg (fun () ->
      ignore (submit unordered ~buffers:[| on_d |]));
  let copy src dst =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  ignore (Sub.make d [| copy on_d own |]);
  raises_match ~msg:"a copy part's destination" Exn.invalid_arg (fun () ->
      ignore (Sub.make d [| copy own on_d |]));
  let slots = Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |] in
  ignore (submit slots ~buffers:[| on_d; own |]);
  raises_match ~msg:"a run's copy into it" Exn.invalid_arg (fun () ->
      ignore (submit slots ~buffers:[| own; on_d |]));
  let fixed access = Sub.make ~fixed:[ (on_d, access) ] d [||] in
  ignore (fixed Rig.Read);
  raises_match ~msg:"fixed memory written" Exn.invalid_arg (fun () ->
      ignore (fixed Rig.Read_write))

(* A host claim for writing is refused on memory that admits only reads, and one
   for reading is had. *)
let test_read_claims () =
  let module R = Support.Reader in
  let answer = Testable.make ~pp:R.pp_answer ~equal:( = ) in
  let io, t = open_pages () in
  let r = Option.get (Pages.alloc t page_bytes) in
  let m = B.of_io io Pages.region_key r ~access:Read page_bytes in
  equal ~msg:"for writing" answer R.Read_only (R.claim m Read_write);
  equal ~msg:"for reading" answer R.Claimed (R.claim m Read);
  R.release m

(* The blits and a fill reach io memory, which the host does not address, at
   the offset of their view: the region holds what they wrote. *)
let test_blits () =
  let io, t = open_pages () in
  let r = Option.get (Pages.alloc t 8) in
  let m = B.of_io io Pages.region_key r ~access:Read_write 8 in
  let region () = String.init 8 (Bigarray.Array1.get r.bytes) in
  B.blit_from_string "abcdefgh" 0 m 0 8;
  equal ~msg:"a blit into it" string "abcdefgh" (region ());
  B.fill (B.view m ~first:2 ~length:4) 'f';
  equal ~msg:"a fill of a view" string "abffffgh" (region ());
  Bigarray.Array1.fill r.bytes 'r';
  let got = Bytes.make 8 '.' in
  B.blit_to_bytes m 1 got 2 4;
  equal ~msg:"a blit out of it" string "..rrrr.." (Bytes.to_string got)

(* A fill of io memory of more than its writes take at a time, 1 MiB, at an
   offset: every byte of the view and no other. *)
let test_large_fill () =
  let io, t = open_pages () in
  let n = (2 lsl 20) + 7 in
  let r = Option.get (Pages.alloc t (n + 6)) in
  Bigarray.Array1.fill r.bytes 'r';
  let m = B.of_io io Pages.region_key r ~access:Read_write (n + 6) in
  B.fill (B.view m ~first:3 ~length:n) 'f';
  let expected = "rrr" ^ String.make n 'f' ^ "rrr" in
  let got = String.init (n + 6) (Bigarray.Array1.get r.bytes) in
  equal ~msg:"bytes that differ" int 0
    (List.length
       (List.filter
          (fun i -> got.[i] <> expected.[i])
          (List.init (n + 6) Fun.id)))

(* A blit from a string refuses memory that admits only reads; a blit to bytes
   reads it. *)
let test_read_blits () =
  let io, t = open_pages () in
  let r = Option.get (Pages.alloc t page_bytes) in
  Bigarray.Array1.fill r.bytes 'r';
  let m = B.of_io io Pages.region_key r ~access:Read page_bytes in
  let on_host = require_some (B.borrow Rig.host m) in
  raises_match Exn.invalid_arg (fun () -> B.blit_from_string "w" 0 on_host 0 1);
  let got = Bytes.create 2 in
  B.blit_to_bytes on_host 0 got 0 2;
  equal string "rr" (Bytes.to_string got)

(* Memory admits the accesses it was made with, through its views and borrows,
   and a copy into memory that admits only reads raises, even of no bytes: it
   neither reads its source nor writes the region. *)
let test_read_memory () =
  let io, t = open_pages () in
  let d, _ = P.open_ ~host_visible:false "io:read" in
  let r = Option.get (Pages.alloc t page_bytes) in
  Bigarray.Array1.fill r.bytes 'r';
  let m = B.of_io io Pages.region_key r ~access:Read page_bytes in
  let on_d = require_some (B.borrow d m) in
  equal ~msg:"made, viewed, borrowed" (list access) [ Read; Read; Read ]
    [ B.access m; B.access (B.view m ~first:1 ~length:2); B.access on_d ];
  equal ~msg:"created, of a bigarray, of a writable region" (list access)
    [ Read_write; Read_write; Read_write ]
    [
      B.access (B.create io 8);
      B.access (B.of_bigarray (B.export (B.create Rig.host 8) Bigarray.char));
      B.access (B.of_io io Pages.region_key r ~access:Read_write page_bytes);
    ];
  let from_d = B.create d page_bytes in
  B.copy ~src:(filled page_bytes 'd') ~dst:from_d;
  raises_match ~msg:"from a device" Exn.invalid_arg (fun () ->
      B.copy ~src:from_d ~dst:m);
  raises_match ~msg:"through a borrow" Exn.invalid_arg (fun () ->
      B.copy ~src:from_d ~dst:on_d);
  raises_match ~msg:"from the host" Exn.invalid_arg (fun () ->
      B.copy ~src:(filled page_bytes 'h') ~dst:m);
  raises_match ~msg:"of no bytes" Exn.invalid_arg (fun () ->
      B.copy ~src:(filled 0 'h') ~dst:(B.view m ~first:0 ~length:0));
  equal ~msg:"the region" string
    (String.make page_bytes 'r')
    (String.init page_bytes (Bigarray.Array1.get r.bytes));
  equal ~msg:"read and written" (pair int int) (0, 0)
    (count "read" t, count "write" t)

(* Pages whose device faults as rig reads its budget. *)
module Budgetless = struct
  include Pages

  let budget _ = raise (Fault "the link is down")
  let stop t = note t "stop"
end

(* A fault reading an io device's budget is the open's [Error], naming the
   device: its handle is stopped, and the name opens again. *)
let test_budget_fault () =
  let name = "io:budget-fault" and t = pages () in
  equal
    (result unit string)
    (Error "io:budget-fault: the link is down")
    (Result.map ignore
       (Rig.open_io
          { name; io = (module Budgetless); make = (fun () -> Ok t) }));
  equal ~msg:"stops" int 1 (count "stop" t);
  is_ok
    (Rig.open_io
       { name; io = (module Pages); make = (fun () -> Ok (pages ())) })

let tests =
  [
    group ~timeout "io memory"
      [
        test "a device borrows io memory through its pages, mapped once"
          test_borrow;
        test "a fault reading an io device's budget is the open's Error"
          test_budget_fault;
        test "overlaps uses cached io pages without mapping" test_overlaps;
        test "an io span stays stable when its pages are published"
          test_span_stable;
        prop ~count:300
          "rig_buffers_overlap of two ranges is Buffer.overlaps of their views"
          overlap_case overlap_law;
        test "rig_buffers_overlap of ranges at a buffer's bounds"
          test_overlap_bounds;
        test "sorted claims check cached io aliases" test_sorted_overlaps;
        test "a borrow back on the io device is its memory" test_borrow_back;
        test "an empty io borrow keeps its host address without mapping"
          test_empty_host_borrow;
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
        test "a run refuses io memory and takes a borrow of its pages"
          test_run_borrows;
        test "io memory and a device's memory copy through staging, both ways"
          test_staged_device;
        test "a fault of io loses its device, a failure of its memory nothing"
          test_faults;
        test
          "a copy from or into a borrow of io memory goes through its io \
           library"
          test_borrow_copies;
        test "a fault under a copy from a borrow loses the io device"
          test_borrow_fault;
        test
          "io memory and a borrow of it copy with another machine's device"
          test_far_copies;
        test "a region an io library made is a buffer of its device alone"
          test_of_io;
        test "a closed io device makes no buffer" test_of_io_closed;
        test "io memory has no driver address or object and loads no code"
          test_io_refusals;
        test "a copy into io memory keeps its buffers until written"
          test_write_keeps;
        test "a copy from io memory keeps its buffers until read"
          test_read_keeps;
        test "a device's work never writes memory that admits only reads"
          test_read_runs;
        test
          "memory admits its accesses through views and borrows, and a copy \
           into read memory raises"
          test_read_memory;
        test "a host claim for writing refuses memory that admits only reads"
          test_read_claims;
        test "the blits and a fill reach io memory at their view's offset"
          test_blits;
        test "a fill of io memory over 2 MiB sets every byte of its view"
          test_large_fill;
        test "a blit from a string refuses memory that admits only reads"
          test_read_blits;
      ];
  ]

let () = exit (run "rig.io" tests)
