(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Io memory: buffers of an io device over an io library's regions
   (Rig.Buffer.of_io), which drivers' devices reach through its pages. Pages is
   an io library over page-aligned host memory that logs its calls. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1
let page_bytes = B.paged_from
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

let pp_buffer ppf b =
  Format.fprintf ppf "%d bytes on %s" (B.length b) (Rig.name (B.device b))

module Pages = struct
  (* The log of a region's calls: [during] runs as a read or a write starts;
     [mapped] is how [pages] answers; [reads] holds each read's runs, as their
     host addresses and lengths, the last first. *)
  type t = {
    lock : Mutex.t;
    mutable calls : string list;
    mutable reads : (int * int) list list;
    mutable during : unit -> unit;
    mutable mapped : [ `Pages | `None | `Fails ];
  }

  (* Its bytes, the host buffer that holds them on a page, and its log. *)
  type region = {
    bytes :
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
    keep : B.t;
    log : t;
  }

  let note t call = Mutex.protect t.lock (fun () -> t.calls <- call :: t.calls)

  let make log n =
    let keep = B.create Rig.host (Int.max n page_bytes) in
    { bytes = Bigarray.Array1.sub (B.export keep Bigarray.char) 0 n; keep; log }

  let free r = note r.log "free"
  let base r = Support.Reader.host r.keep

  let read r ~at ~dst ~len =
    note r.log "read";
    let runs = List.combine (Array.to_list dst) (Array.to_list len) in
    Mutex.protect r.log.lock (fun () -> r.log.reads <- runs :: r.log.reads);
    r.log.during ();
    Array.iteri
      (fun i at -> Support.move ~dst:dst.(i) ~src:(base r + at) len.(i))
      at

  let write r ~at ~src ~len =
    note r.log "write";
    r.log.during ();
    Support.move ~dst:(base r + at) ~src len

  let pages r =
    note r.log "pages";
    match r.log.mapped with
    | `Pages -> Some r.bytes
    | `None -> None
    | `Fails -> raise (Sys_error "too many open files")

  let prefetch r ~at:_ ~len:_ = note r.log "prefetch"
end

(* Another io library, whose regions have no pages. *)
module Other = struct
  type region = unit

  let free () = ()
  let read () ~at:_ ~dst:_ ~len:_ = ()
  let write () ~at:_ ~src:_ ~len:_ = ()
  let pages () = None
  let prefetch () ~at:_ ~len:_ = ()
end

let opened = Atomic.make 0

let pages () =
  {
    Pages.lock = Mutex.create ();
    calls = [];
    reads = [];
    during = (fun () -> ());
    mapped = `Pages;
  }

let open_pages () =
  let name = Printf.sprintf "io:pages-%d" (Atomic.fetch_and_add opened 1) in
  (require_ok ~pp:Format.pp_print_string (Rig.open_io name), pages ())

(* A buffer of [io] over [n] new bytes of a region that logs to [t]. *)
let create ?(access = Rig.Read_write) io t n =
  B.of_io io (module Pages) (Pages.make t n) ~access n

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

let region (r : Pages.region) =
  String.init (Bigarray.Array1.dim r.bytes) (Bigarray.Array1.get r.bytes)

(* The index of the first byte where [a] and [b] differ, [None] if equal: a
   failure names a byte, not two strings of megabytes. *)
let first_difference a b =
  if String.equal a b then None
  else
    let n = Int.min (String.length a) (String.length b) in
    let rec go i = if i = n || a.[i] <> b.[i] then Some i else go (i + 1) in
    go 0

(* A copy on [d] from [src] into [dst], left queued. *)
let queued_copy d ~src ~dst =
  let part = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  ignore (submit (Sub.make d [| part |]))

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* create *)

(* rig.mli, create: raises if [n > 0] and [d] is an io device, which makes no
   memory. *)
let test_create () =
  let io, _ = open_pages () in
  raises_match ~msg:"a buffer of a byte" Exn.invalid_arg (fun () ->
      B.create io 1);
  equal ~msg:"an empty buffer's length" int 0 (B.length (B.create io 0))

(* of_io *)

let test_of_io_refused () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:not-io" in
  let r = Pages.make t page_bytes in
  raises_match ~msg:"a driver's device" Exn.invalid_arg (fun () ->
      B.of_io d (module Pages) r ~access:Read_write 8);
  raises_match ~msg:"a negative length" Exn.invalid_arg (fun () ->
      B.of_io io (module Pages) r ~access:Read_write (-1))

let test_of_io_device () =
  let io, t = open_pages () in
  let b = create io t 16 in
  let other = B.of_io io (module Other) () ~access:Read_write 16 in
  equal (list device) [ io; io ] [ B.device b; B.device other ]

(* A closed io device makes no buffer of a region, also once a later device took
   its slot and its record may be gone. A software device's last value, which
   the io device's slot keeps and the Polled device after it starts from, shows
   that the slot passed from one to the next. *)
let test_of_io_closed () =
  let m = Result.get_ok (Rig.software "io:before-closed") in
  let p = Rig.submit (Rig.Submission.make m [||]) ~buffers:[||] ~args:[||] in
  Rig.Point.wait p;
  Rig.close m;
  Rig.free_cache m;
  let io, t = open_pages () in
  let r = Pages.make t page_bytes in
  Rig.close io;
  Rig.free_cache io;
  let later, _ = P.open_ "io:after-closed" in
  equal ~msg:"the slot passed on" int (Rig.Point.value p) (Rig.submitted later);
  raises_match (lost io) (fun () ->
      B.of_io io (module Pages) r ~access:Read_write page_bytes);
  Gc.full_major ();
  Rig.close later

let test_free () =
  let io, t = open_pages () in
  let d, p = P.open_ "io:user" in
  (fun () ->
    let m = create io t page_bytes in
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    ignore (submit ~args:one s ~buffers:[| require_some (B.borrow d m) |]))
    ();
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create io 0);
    ignore (B.create d 0)
  in
  drain ();
  equal ~msg:"frees while its use is unreached" int 0 (count "free" t);
  ignore (P.run p);
  drain ();
  equal ~msg:"frees once it is reached" int 1 (count "free" t)

(* The minor words [f ()] allocates. *)
let minor_words f =
  let before = Gc.minor_words () in
  f ();
  int_of_float (Gc.minor_words () -. before)

(* Draining one collected io memory returns it to its library for few words,
   this library's io release included, beyond what an idle drain costs: a give
   that built a closure cost 6 words a file. 40 is a baseline bound. *)
let test_release_words () =
  let io, t = open_pages () in
  let drain () = ignore (Sys.opaque_identity (B.create io 0)) in
  drain ();
  let idle = minor_words drain in
  let[@inline never] [@local never] dropped () =
    ignore (Sys.opaque_identity (create io t 64))
  in
  dropped ();
  Gc.full_major ();
  let release = minor_words drain in
  equal ~msg:"freed" int 1 (count "free" t);
  at_most ~msg:"a release's words over an idle drain's" int ~than:40
    (release - idle)

(* borrow *)

(* The Buffers preamble: a borrow that needs a mapping holds the device's
   mapping of the allocation, which every live borrow of it on that device
   shares. *)
let test_borrow () =
  let io, t = open_pages () in
  let d, p = P.open_ "io:borrower" in
  let m = create io t page_bytes in
  for _ = 1 to 3 do
    ignore (require_some (B.borrow d m))
  done;
  equal ~msg:"host mappings" (list int) [ page_bytes ] (P.host_maps p);
  equal ~msg:"pages asked" int 1 (count "pages" t)

let test_no_pages () =
  let io, t = open_pages () in
  t.mapped <- `None;
  let d, _ = P.open_ "io:pageless" in
  let m = create io t page_bytes in
  is_none ~pp:pp_buffer ~msg:"a first borrow" (B.borrow d m);
  is_none ~pp:pp_buffer ~msg:"a second borrow" (B.borrow d m);
  is_none ~pp:pp_buffer ~msg:"the host's borrow" (B.borrow Rig.host m);
  equal ~msg:"pages asked" int 1 (count "pages" t)

(* rig.mli, borrow: raises [Sys_error] if asking an io library for [b]'s pages
   fails transiently. *)
let test_pages_fail () =
  let io, t = open_pages () in
  t.mapped <- `Fails;
  let d, _ = P.open_ "io:pages-fail" in
  let m = create io t page_bytes in
  raises (Sys_error "too many open files") (fun () -> B.borrow d m);
  equal ~msg:"the io device's loss" (option string) None (Rig.lost io);
  t.mapped <- `Pages;
  ignore (require_some (B.borrow d m));
  equal ~msg:"pages asked" int 2 (count "pages" t)

(* rig_edge.mli, Io.prefetch: rig asks it as a device other than the host
   borrows the region. *)
let test_prefetch () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:prefetcher" in
  let m = create io t page_bytes in
  ignore (require_some (B.borrow Rig.host m));
  equal ~msg:"after the host's borrow" int 0 (count "prefetch" t);
  ignore (require_some (B.borrow d m));
  at_least ~msg:"after a device's borrow" int ~than:1 (count "prefetch" t)

(* The Buffers preamble: empty memory of an io device borrows on every other
   device. *)
let test_empty_host_borrow () =
  let io, t = open_pages () in
  let m = B.of_io io (module Pages) (Pages.make t 0) ~access:Read_write 0 in
  let host = require_some (B.borrow Rig.host m) in
  not_equal ~msg:"the borrow's host address" int 0 (Support.Reader.host host);
  equal ~msg:"pages asked" int 0 (count "pages" t)

(* A borrow back on the io device is its memory: no driver object, no host or
   device address, not borrowed, in the memory's span. *)
let test_borrow_back () =
  let io, t = open_pages () in
  let m = create io t page_bytes in
  let back = require_some (B.borrow io (require_some (B.borrow Rig.host m))) in
  let answer f = try Some (f ()) with Invalid_argument _ -> None in
  let coordinates b =
    ( answer (fun () -> B.handle b),
      Support.Reader.host b,
      Support.Reader.span b,
      B.is_borrowed b )
  in
  equal ~msg:"device" device io (B.device back);
  equal
    (quad (option nativeint) int (pair int int) bool)
    (coordinates m) (coordinates back)

(* The Buffers preamble: a device's work addresses only buffers on that device,
   and another device's memory through a borrow of it. *)
let test_run_borrows () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:slots" in
  let m = create io t page_bytes in
  let s = Sub.make d [| Support.touching [| Sub.read; Sub.write |] |] in
  let own = B.create d 8 and borrowed = require_some (B.borrow d m) in
  raises_match ~msg:"io memory read" Exn.invalid_arg (fun () ->
      ignore (submit ~args:one s ~buffers:[| m; own |]));
  raises_match ~msg:"io memory written" Exn.invalid_arg (fun () ->
      ignore (submit ~args:one s ~buffers:[| borrowed; m |]));
  Rig.Point.wait (submit ~args:one s ~buffers:[| borrowed; own |])

(* overlaps *)

(* rig.mli, overlaps: buffers overlap iff they share memory; the check maps no
   pages, and once the host borrowed them a bigarray over the pages is memory of
   its own that holds the same bytes. *)
let test_overlaps () =
  let io, t = open_pages () in
  let m = create io t page_bytes and other = B.create Rig.host 32 in
  equal ~msg:"io memory and host memory, before any borrow" (pair bool bool)
    (false, false)
    (B.overlaps m other, B.overlaps other m);
  equal ~msg:"pages asked by the check" int 0 (count "pages" t);
  let host = require_some (B.borrow Rig.host m) in
  let ba = B.read host Bigarray.char Fun.id in
  let wrapped = B.of_bigarray (Bigarray.Array1.sub ba 8 16) in
  let shared = B.view m ~first:12 ~length:8 in
  let disjoint = B.view m ~first:24 ~length:8 in
  equal ~msg:"a bigarray and an io view over shared bytes" (pair bool bool)
    (true, true)
    (B.overlaps shared wrapped, B.overlaps wrapped shared);
  equal ~msg:"a bigarray and an io view over adjacent bytes" (pair bool bool)
    (false, false)
    (B.overlaps disjoint wrapped, B.overlaps wrapped disjoint);
  equal ~msg:"pages asked" int 1 (count "pages" t)

(* The Buffers preamble: empty memory of an io device borrows on every other
   device; rig.mli, overlaps: an empty buffer overlaps nothing. *)
let test_empty_overlaps () =
  let io, t = open_pages () in
  let m = B.of_io io (module Pages) (Pages.make t 0) ~access:Read_write 0 in
  let host = require_some (B.borrow Rig.host m) in
  equal (pair bool bool) (false, false) (B.overlaps m host, B.overlaps host m)

(* access *)

let pp_access ppf a =
  Format.pp_print_string ppf
    (match a with Rig.Read -> "Read" | Read_write -> "Read_write")

let access = Testable.make ~pp:pp_access ~equal:( = )

(* rig.mli, access: what [b]'s memory admits, which its views and borrows admit
   too: [Read_write] for memory create and of_bigarray make, what of_io was
   given. *)
let test_access () =
  let io, t = open_pages () in
  let d, _ = P.open_ ~host_visible:false "io:access" in
  let m = create ~access:Read io t page_bytes in
  equal
    (list (pair string access))
    [
      ("of_io ~access:Read", Read);
      ("a view of it", Read);
      ("a device's borrow of it", Read);
      ("of_io ~access:Read_write", Read_write);
      ("create", Read_write);
      ("of_bigarray", Read_write);
    ]
    [
      ("of_io ~access:Read", B.access m);
      ("a view of it", B.access (B.view m ~first:1 ~length:2));
      ("a device's borrow of it", B.access (require_some (B.borrow d m)));
      ("of_io ~access:Read_write", B.access (create io t page_bytes));
      ("create", B.access (B.create Rig.host 8));
      ( "of_bigarray",
        B.access (B.of_bigarray (B.export (B.create Rig.host 8) Bigarray.char))
      );
    ]

(* rig.mli, access: rig never writes through a [Read] buffer, and refuses a copy
   or work that would. *)
let test_read_runs () =
  let io, t = open_pages () in
  let d, _ = P.open_ ~host_visible:false "io:read-runs" in
  let m = create ~access:Read io t page_bytes in
  let on_d = require_some (B.borrow d m) and own = B.create d page_bytes in
  let s = Sub.make d [| Support.touching [| Sub.read; Sub.write |] |] in
  ignore (submit ~args:one s ~buffers:[| on_d; own |]);
  raises_match ~msg:"a submit's write" Exn.invalid_arg (fun () ->
      ignore (submit ~args:one s ~buffers:[| own; on_d |]));
  let unordered = Sub.make d [| Support.touching [| Sub.unordered |] |] in
  raises_match ~msg:"a submit's unordered use" Exn.invalid_arg (fun () ->
      ignore (submit ~args:one unordered ~buffers:[| on_d |]));
  let copy src dst = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  ignore (Sub.make d [| copy on_d own |]);
  raises_match ~msg:"a copy part's destination" Exn.invalid_arg (fun () ->
      ignore (Sub.make d [| copy own on_d |]));
  let slots = Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |] in
  ignore (submit slots ~buffers:[| on_d; own |]);
  raises_match ~msg:"a submit's copy into it" Exn.invalid_arg (fun () ->
      ignore (submit slots ~buffers:[| own; on_d |]));
  let fixed access = Sub.make ~fixed:[ (on_d, access) ] d [||] in
  ignore (fixed Rig.Read);
  raises_match ~msg:"fixed memory written" Exn.invalid_arg (fun () ->
      ignore (fixed Rig.Read_write))

(* wait *)

(* rig.mli, wait: an io device's memory has work to wait for only where a device
   borrowed it. *)
let test_wait () =
  let io, t = open_pages () in
  let d, p = P.open_ "io:waiter" in
  let m = create io t page_bytes in
  let staged = B.create d page_bytes in
  queued_copy d ~src:staged ~dst:(require_some (B.borrow d m));
  equal ~msg:"queued" int 1 (P.queued p);
  B.wait m Rig.Read;
  equal ~msg:"queued after the wait" int 0 (P.queued p)

(* read *)

(* Host access: the host addresses a borrow on the host, which for io memory is
   its pages. *)
let test_host_read () =
  let io, t = open_pages () in
  let r = Pages.make t 16 in
  Bigarray.Array1.fill r.bytes 'p';
  let m = B.of_io io (module Pages) r ~access:Read_write 16 in
  equal string (String.make 16 'p') (bytes (require_some (B.borrow Rig.host m)))

(* Copying *)

(* A copy out of io memory follows a device's write through its pages, and a
   copy into it follows a device's read of it. *)
let test_copy_order () =
  let io, t = open_pages () in
  let d, p = P.open_ "io:copier" in
  let m = create io t page_bytes in
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

(* Copying: memory an io device owns, its own or a borrow of its pages, moves
   only through its io library's read or write, and no device maps the pages for
   it: toward host memory, toward a device's memory and back, the blits and a
   fill, and a copy between two views of the one allocation. *)
let test_borrow_copies () =
  let io, t = open_pages () in
  let d, p = P.open_ ~host_visible:false "io:borrow-copies" in
  let m = create io t page_bytes in
  let hb = require_some (B.borrow Rig.host m) in
  let calls_of f =
    let r = count "read" t and w = count "write" t in
    f ();
    (count "read" t - r, count "write" t - w)
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
  equal ~msg:"the device's maps of the pages" (list int) []
    (List.filter (( = ) (Support.at hb)) (P.host_mapped_at p));
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

(* Copying, staging memory: between io memory that maps no pages and a device's
   memory the host does not address, the bytes go through staging memory, in
   allocations of at most 32 MiB: 100 MiB and a page cross three of their
   edges. *)
let test_staged_device () =
  let io, t = open_pages () in
  t.mapped <- `None;
  let d, _ = P.open_ ~host_visible:false "io:staged-device" in
  let n = (100 lsl 20) + 4096 in
  let s =
    String.init n (fun i ->
        Char.unsafe_chr ((i + ((i lsr 16) * 13) + (i lsr 24)) land 255))
  in
  let m = create io t n and dev = B.create d n and back = create io t n in
  B.blit_from_string s 0 m 0 n;
  B.copy ~src:m ~dst:dev;
  B.copy ~src:dev ~dst:back;
  let got = Bytes.create n in
  B.blit_to_bytes back 0 got 0 n;
  equal ~msg:"the first byte that differs" (option int) None
    (first_difference s (Bytes.unsafe_to_string got))

(* Io memory and a borrow of its pages copy to and from a device of another
   machine, through the host's staging memory: the io library reads and writes,
   and every byte lands. *)
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
  let m = create io t page_bytes in
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
    (String.make page_bytes 'g')
    (bytes out);
  B.copy ~src:(filled page_bytes 'h') ~dst:on_far;
  B.copy ~src:on_far ~dst:hb;
  B.copy ~src:m ~dst:on_far;
  B.copy ~src:on_far ~dst:out;
  equal ~msg:"into the borrow, then from the io memory" string
    (String.make page_bytes 'h')
    (bytes out);
  equal ~msg:"reads and writes of the io library" (pair int int) (2, 3)
    (count "read" t - reads, count "write" t - writes)

(* rig.mli, submit_copy: raises what an io library's read or write raises, such
   as a file truncated since it was opened. *)
let test_failures () =
  let io, t = open_pages () in
  let m = create io t page_bytes and out = filled page_bytes '0' in
  t.during <- (fun () -> raise (Sys_error "truncated"));
  raises ~msg:"a read" (Sys_error "truncated") (fun () ->
      B.copy ~src:m ~dst:out);
  raises ~msg:"a write" (Sys_error "truncated") (fun () ->
      B.copy ~src:out ~dst:m);
  equal ~msg:"the io device's loss" (option string) None (Rig.lost io);
  t.during <- (fun () -> ());
  B.copy ~src:m ~dst:out;
  B.copy ~src:out ~dst:m

(* A failure of an io read under a copy from a borrow of its pages reaches the
   caller and loses neither the io device, which owns the memory, nor the
   host. *)
let test_borrow_failure () =
  let io, t = open_pages () in
  let hb = require_some (B.borrow Rig.host (create io t page_bytes)) in
  t.during <- (fun () -> raise (Sys_error "truncated"));
  raises (Sys_error "truncated") (fun () ->
      B.copy ~src:hb ~dst:(filled page_bytes '0'));
  equal ~msg:"the losses of the io device and the host"
    (pair (option string) (option string))
    (None, None)
    (Rig.lost io, Rig.lost Rig.host)

(* submit_copy *)

(* A copy of [boxes] out of io memory into a destination of 4096 '.', as (each
   read's runs, the destination's bytes after it). *)
let read_rows boxes =
  let io, t = open_pages () in
  let src = create io t page_bytes in
  let s = String.init page_bytes (fun i -> Char.chr (i * 7 land 255)) in
  B.blit_from_string s 0 src 0 page_bytes;
  let dst = filled 4096 '.' in
  t.reads <- [];
  ignore (B.submit_copy ~src ~dst boxes);
  let expected = Bytes.make 4096 '.' in
  Array.iter
    (fun (x : B.box) ->
      for i = 0 to x.rows - 1 do
        Bytes.blit_string s
          (x.src_offset + (i * x.src_pitch))
          expected
          (x.dst_offset + (i * x.dst_pitch))
          x.width
      done)
    boxes;
  (List.rev t.reads, Bytes.to_string expected, bytes dst)

(* rig_edge.mli, Io.read: rig calls it with at least one run, every run
   positive, and no two runs' bytes in host memory in common. *)
let runs_apart runs =
  let rec apart = function
    | (a, n) :: ((b, _) :: _ as rest) -> a + n <= b && apart rest
    | _ -> true
  in
  runs <> []
  && List.for_all (fun (_, n) -> n > 0) runs
  && apart (List.sort compare runs)

let reads_apart ?msg reads =
  satisfies ?msg ~claim:"each read's runs are positive and share no byte"
    (list (list (pair int int)))
    (List.for_all runs_apart) reads

let test_rows_one_read () =
  let apart =
    Array.init 40 (fun i ->
        B.row ~src_offset:(i * 997 mod 60000) ~dst_offset:((39 - i) * 100) 90)
  and box_and_row =
    [|
      {
        B.src_offset = 5;
        dst_offset = 4000;
        width = 9;
        rows = 1;
        src_pitch = 9;
        dst_pitch = 9;
      };
      {
        B.src_offset = 300;
        dst_offset = 0;
        width = 8;
        rows = 30;
        src_pitch = 1000;
        dst_pitch = 12;
      };
    |]
  in
  List.iter
    (fun (what, boxes) ->
      let reads, expected, got = read_rows boxes in
      equal ~msg:(what ^ ": reads") int 1 (List.length reads);
      reads_apart ~msg:what reads;
      equal ~msg:(what ^ ": bytes") string expected got)
    [
      ("rows apart, in decreasing order", apart);
      ("a box of rows beside a row", box_and_row);
    ]

(* Rows that write a common byte read in calls whose runs share none, and a byte
   two rows write holds the later's (rig.mli, submit_copy). *)
let test_rows_overlapping () =
  let overlapping =
    Array.init 10 (fun i ->
        B.row ~src_offset:(i * 1301) ~dst_offset:(i * 11) 64)
  in
  let reads, expected, got = read_rows overlapping in
  reads_apart reads;
  equal ~msg:"bytes" string expected got

(* Io.read takes at least one run, every run positive. *)
let test_rows_none () =
  List.iter
    (fun (what, boxes) ->
      let reads, _, got = read_rows boxes in
      equal ~msg:(what ^ ": reads") int 0 (List.length reads);
      equal ~msg:(what ^ ": bytes") string (String.make 4096 '.') got)
    [
      ("no box", [||]);
      ("boxes of no bytes", [| B.row 0; { (B.row 8) with rows = 0 } |]);
    ]

let test_io_and_borrow () =
  let io, t = open_pages () in
  let m = create io t page_bytes in
  let hb = require_some (B.borrow Rig.host m) in
  let from_m = B.view m ~first:0 ~length:64
  and into_hb = B.view hb ~first:64 ~length:64 in
  raises_match ~msg:"from the io memory" Exn.invalid_arg (fun () ->
      B.submit_copy ~src:from_m ~dst:into_hb [| B.row 64 |]);
  raises_match ~msg:"into the io memory" Exn.invalid_arg (fun () ->
      B.submit_copy ~src:into_hb ~dst:from_m [| B.row 64 |]);
  equal ~msg:"reads and writes" (pair int int) (0, 0)
    (count "read" t, count "write" t)

(* rig.mli, submit_copy: raises if [dst] admits only [Read], which the check
   finds before the copy reads its source or writes the region. *)
let test_read_destination () =
  let io, t = open_pages () in
  let d, _ = P.open_ ~host_visible:false "io:read" in
  let r = Pages.make t page_bytes in
  Bigarray.Array1.fill r.bytes 'r';
  let m = B.of_io io (module Pages) r ~access:Read page_bytes in
  let on_d = require_some (B.borrow d m) in
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
  equal ~msg:"the region" string (String.make page_bytes 'r') (region r);
  equal ~msg:"reads and writes" (pair int int) (0, 0)
    (count "read" t, count "write" t)

(* copy *)

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

(* rig.mli, copy: it holds its read claims until it returns, so a collection
   during the io library's call frees neither side. *)
let test_write_keeps () =
  let io, t = open_pages () in
  let collected = ref false and seen = ref [] in
  collecting t io collected seen;
  B.copy ~src:(watched collected page_bytes 'k') ~dst:(create io t page_bytes);
  equal (list string) [ "collected: false"; "frees: 0" ] !seen

let test_read_keeps () =
  let io, t = open_pages () in
  let collected = ref false and seen = ref [] in
  let src = create io t page_bytes in
  B.copy ~src:(filled page_bytes 'k') ~dst:src;
  collecting t io collected seen;
  B.copy ~src ~dst:(watched collected page_bytes 'x');
  equal (list string) [ "collected: false"; "frees: 0" ] !seen;
  ignore (Sys.opaque_identity src)

(* blit_from_string *)

let test_blit_from_string () =
  let io, t = open_pages () in
  let r = Pages.make t 8 in
  Bigarray.Array1.fill r.bytes 'r';
  let m = B.of_io io (module Pages) r ~access:Read_write 8 in
  B.blit_from_string "abcdef" 1 (B.view m ~first:2 ~length:6) 1 3;
  equal string "rrrbcdrr" (region r)

let test_blit_from_string_read () =
  let io, t = open_pages () in
  let m = create ~access:Read io t page_bytes in
  let on_host = require_some (B.borrow Rig.host m) in
  raises_match ~msg:"into the io memory" Exn.invalid_arg (fun () ->
      B.blit_from_string "w" 0 m 0 1);
  raises_match ~msg:"into its borrow" Exn.invalid_arg (fun () ->
      B.blit_from_string "w" 0 on_host 0 1)

(* blit_to_bytes *)

let test_blit_to_bytes () =
  let io, t = open_pages () in
  let r = Pages.make t 8 in
  String.iteri (Bigarray.Array1.set r.bytes) "abcdefgh";
  let m = B.of_io io (module Pages) r ~access:Read 8 in
  let got = Bytes.make 8 '.' in
  B.blit_to_bytes (B.view m ~first:2 ~length:6) 1 got 2 4;
  equal ~msg:"out of the io memory" string "..defg.." (Bytes.to_string got);
  let on_host = require_some (B.borrow Rig.host m) in
  let got = Bytes.make 8 '.' in
  B.blit_to_bytes on_host 1 got 2 4;
  equal ~msg:"out of its borrow on the host" string "..bcde.."
    (Bytes.to_string got)

(* fill *)

(* View lengths about the 1 MiB that one of rig's writes into an io library
   takes at a time: below one write, exactly one, and two and a tail. *)
let fill_lengths =
  [ ("4 bytes", 4); ("1 MiB", 1 lsl 20); ("2 MiB and 7 bytes", (2 lsl 20) + 7) ]

(* A fill of a view of io memory at an offset sets every byte of the view and no
   other. *)
let test_fill (_, n) =
  let io, t = open_pages () in
  let r = Pages.make t (n + 6) in
  Bigarray.Array1.fill r.bytes 'r';
  let m = B.of_io io (module Pages) r ~access:Read_write (n + 6) in
  B.fill (B.view m ~first:3 ~length:n) 'f';
  equal ~msg:"the first byte that differs" (option int) None
    (first_difference ("rrr" ^ String.make n 'f' ^ "rrr") (region r))

(* take *)

(* rig.mli, of_io: its memory is outside the claims, which take finds. *)
let test_take () =
  let io, t = open_pages () in
  let b = create io t page_bytes in
  is_none ~pp:(Format.pp_print_list pp_buffer) (B.take ~why:"taken" [ b ]);
  equal ~msg:"dead" (option string) None (B.dead b)

(* rig.h *)

(* rig.h, rig_buffer_span: a buffer's span stays where it was read, also when
   its pages are first mapped, and views and borrows lie in its space. *)
let test_span_stable () =
  let io, t = open_pages () in
  let m = create io t page_bytes in
  let space, first = Support.Reader.span m in
  let host = require_some (B.borrow Rig.host m) in
  let view = B.view m ~first:12 ~length:8 in
  equal ~msg:"a view's span, once the pages are mapped" (pair int int)
    (space, first + 12)
    (Support.Reader.span view);
  equal ~msg:"the borrow's span" (pair int int) (space, first)
    (Support.Reader.span host);
  equal ~msg:"pages asked" int 1 (count "pages" t)

(* rig.h, rig_buffers_overlap: whether two byte ranges share a byte, over
   buffers of every kind of place: host memory, a view of it and a device's
   borrow of it; io memory, its views, a device's borrow, the host borrow that
   maps its pages and a bigarray over those pages; memory a free killed and
   memory made after it; an empty view. Besides any two buffers, pairs put two
   ranges in one buffer, a borrow beside its memory, the dead buffer beside any,
   or the bigarray beside a buffer of the io memory, whose overlap only the
   cached host address shows. Where they share a byte, Buffer.overlaps of the
   views over them holds. *)
let bigarray_at = 0
let io_family = [| 1; 2; 3; 4; 5 |]
let freed_at = 9

let sharing =
  Testable.make
    ~pp:(fun ppf o ->
      Format.pp_print_string ppf (if o then "share" else "apart"))
    ~equal:Bool.equal

(* The buffers, made at the first case: the bigarray at [bigarray_at], the io
   memory's buffers at [io_family], the dead one at [freed_at]. *)
let universe =
  lazy
    (let io, t = open_pages () in
     let d, _ = P.open_ "io:overlap" in
     let h = B.create Rig.host page_bytes and m = create io t page_bytes in
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

(* Whether two ranges share bytes, by their buffers' spans, for a dead buffer,
   whose views raise. The dead buffer is freed host memory, whose place no io
   page shares, as the pages are live host memory of their own: the ranges share
   bytes iff they intersect in one space. *)
let spans_overlap a (af, an) b (bf, bn) =
  let sa, fa = Support.Reader.span a and sb, fb = Support.Reader.span b in
  an > 0 && bn > 0 && sa = sb
  && fa + af < fb + bf + bn
  && fb + bf < fa + af + an

(* Whether two ranges share bytes: by their spans in one space, and across
   spaces by the host addresses of their bytes. A buffer outside the host's
   space has one where it lies in the io memory whose pages [mapped], the host
   borrow, maps: the address of [mapped]'s first byte, moved by the distance
   between the two spans. *)
let reference mapped a ra b rb =
  let sa, _ = Support.Reader.span a and sb, _ = Support.Reader.span b in
  if sa = sb || B.dead a <> None || B.dead b <> None then
    spans_overlap a ra b rb
  else
    let address b =
      let s, f = Support.Reader.span b in
      let sm, fm = Support.Reader.span mapped in
      if s = 0 then f
      else if s = sm then Support.Reader.host mapped + (f - fm)
      else 0
    in
    let x = address a and y = address b in
    let (af, an), (bf, bn) = (ra, rb) in
    x <> 0 && y <> 0 && an > 0 && bn > 0
    && x + af < y + bf + bn
    && y + bf < x + af + an

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
      Gen.bind
        (Gen.int_range 1 (7 - f))
        (fun n ->
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
  let expected = reference all.(2) a ra b rb in
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
  let got = Support.Reader.overlap a af an b bf bn in
  let msg =
    Printf.sprintf "rig_buffers_overlap of %d [%d,+%d] and %d [%d,+%d]" i af an
      j bf bn
  in
  equal ~msg sharing expected got;
  (* Bytes in common are a word in common. *)
  if got && B.dead a = None && B.dead b = None then
    equal ~msg:"Buffer.overlaps of the views" sharing true
      (B.overlaps
         (B.view a ~first:af ~length:an)
         (B.view b ~first:bf ~length:bn))

(* rig.h, rig_buffer_borrowed: whether the buffer is a borrow, as
   Buffer.is_borrowed answers, over host memory, a bigarray and a view of it, io
   memory, the host's borrow of its pages and a view of that, a device's borrow
   of host memory, and a borrow back on the io device. *)
let test_borrowed_in_c () =
  let io, t = open_pages () in
  let d, _ = P.open_ "io:borrowed-in-c" in
  let m = create io t page_bytes and h = B.create Rig.host page_bytes in
  let ba =
    B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 64)
  in
  let pages = require_some (B.borrow Rig.host m) in
  let buffers =
    [
      ("host memory", h);
      ("a bigarray", ba);
      ("a view of a bigarray", B.view ba ~first:8 ~length:8);
      ("io memory", m);
      ("the host's borrow of io memory", pages);
      ("a view of that borrow", B.view pages ~first:4096 ~length:64);
      ("a device's borrow of host memory", require_some (B.borrow d h));
      ("a borrow back on the io device", require_some (B.borrow io pages));
    ]
  in
  let answers f = List.map (fun (what, b) -> (what, f b)) buffers in
  equal
    (list (pair string bool))
    (answers B.is_borrowed)
    (answers Support.Reader.borrowed);
  equal ~msg:"borrowed among them" int 5
    (List.length (List.filter (fun (_, b) -> B.is_borrowed b) buffers))

(* rig.h, rig_buffer_claim: RIG_READ_ONLY for a claim for writing of memory that
   admits only reads. *)
let test_read_claims () =
  let module R = Support.Reader in
  let answer = Testable.make ~pp:R.pp_answer ~equal:( = ) in
  let io, t = open_pages () in
  let m = create ~access:Read io t page_bytes in
  equal ~msg:"for writing" answer R.Read_only (R.claim m Read_write);
  equal ~msg:"for reading" answer R.Claimed (R.claim m Read);
  R.release m

(* aligned *)

let test_aligned () =
  let io, t = open_pages () in
  let m = create io t 8 in
  raises_match Exn.invalid_arg (fun () -> B.aligned m ~at:0 8)

(* handle *)

let test_handle () =
  let io, t = open_pages () in
  let m = create io t 8 in
  raises_match Exn.invalid_arg (fun () -> B.handle m)

(* Image.load *)

let test_load () =
  let io, _ = open_pages () in
  raises_match Exn.invalid_arg (fun () -> Rig.Image.load io "code:8")

let tests =
  [
    group "create"
      [ test "an io device makes no memory but an empty buffer" test_create ];
    group "of_io"
      [
        test
          "raises Invalid_argument for a device that is no io device or a \
           negative length"
          test_of_io_refused;
        test "a region is a buffer of its io device, beside another library's"
          test_of_io_device;
        test
          "a closed io device makes no buffer, also once a later device took \
           its slot"
          test_of_io_closed;
        test
          "a region returns to its library once unreachable and its uses \
           reached"
          test_free;
        test "a collected region returns to its library for few words"
          test_release_words;
      ];
    group "borrow"
      [
        test "a device borrows io memory through its pages, mapped once"
          test_borrow;
        test
          "memory whose library gives no pages borrows on no device, its \
           library asked once"
          test_no_pages;
        test "a failure asking for pages raises Sys_error, and is asked again"
          test_pages_fail;
        test
          "a device's borrow asks its library to read ahead, the host's does \
           not"
          test_prefetch;
        test "empty io memory borrows on the host, mapping nothing"
          test_empty_host_borrow;
        test "a borrow back on the io device is its memory" test_borrow_back;
        test "a submit refuses io memory and takes a borrow of its pages"
          test_run_borrows;
      ];
    group "overlaps"
      [
        test
          "a bigarray over io pages the host borrowed overlaps the views over \
           those bytes, mapping nothing"
          test_overlaps;
        test "empty io memory overlaps nothing, its borrow on the host included"
          test_empty_overlaps;
      ];
    group "access"
      [
        test "access is what the memory admits, through views and borrows"
          test_access;
        test "a device's work never writes memory that admits only Read"
          test_read_runs;
      ];
    group "wait"
      [ test "a wait of io memory follows a borrower's work" test_wait ];
    group "read"
      [
        test "the host reads io memory's pages through its borrow on the host"
          test_host_read;
      ];
    group "copying"
      [
        test "copies of io memory follow devices' work through its pages"
          test_copy_order;
        test
          "memory an io device owns, a borrow of its pages included, moves \
           only through its library's reads and writes"
          test_borrow_copies;
        test
          "io memory and a device's memory copy through staging memory, both \
           ways"
          test_staged_device;
        test "io memory and a borrow of it copy with another machine's device"
          test_far_copies;
        test
          "a failure of an io read or write reaches the caller and loses \
           nothing"
          test_failures;
        test
          "a failure under a copy from a borrow of the pages reaches the \
           caller and loses nothing"
          test_borrow_failure;
      ];
    group "submit_copy"
      [
        test
          "an io library reads every box's rows in one call where no two write \
           a common byte"
          test_rows_one_read;
        test
          "rows that write a common byte read in runs that share none, the \
           later box's byte winning"
          test_rows_overlapping;
        test "a copy of no bytes makes no read" test_rows_none;
        test "a copy between io memory and a borrow of its pages raises"
          test_io_and_borrow;
        test
          "a copy into memory that admits only Read raises, of no bytes too, \
           reading and writing nothing"
          test_read_destination;
      ];
    group "copy"
      [
        test "a copy into io memory keeps its buffers until written"
          test_write_keeps;
        test "a copy from io memory keeps its buffers until read"
          test_read_keeps;
      ];
    group "blit_from_string"
      [
        test "a blit into io memory writes its view at the blit's offset"
          test_blit_from_string;
        test "a blit raises on memory that admits only Read"
          test_blit_from_string_read;
      ];
    group "blit_to_bytes"
      [
        test
          "a blit reads io memory from its view at the blit's offset, Read \
           memory included"
          test_blit_to_bytes;
      ];
    group "fill"
      [
        cases "a fill of io memory sets every byte of its view and no other"
          ~id:fst fill_lengths test_fill;
      ];
    group "take" [ test "io memory is never taken" test_take ];
    group "rig.h"
      [
        test "an io span stays where it was when its pages are mapped"
          test_span_stable;
        prop ~count:300
          "rig_buffers_overlap of two ranges is whether they share a byte"
          overlap_case overlap_law;
        test "rig_buffer_borrowed answers Rig.Buffer.is_borrowed"
          test_borrowed_in_c;
        test
          "rig_buffer_claim answers RIG_READ_ONLY for writing memory that \
           admits only Read"
          test_read_claims;
      ];
    group "aligned" [ test "io memory has no driver address" test_aligned ];
    group "handle" [ test "io memory has no driver object" test_handle ];
    group "Image.load" [ test "an io device loads no code" test_load ];
  ]

let () = exit (run "rig.io" tests)
