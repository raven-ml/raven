(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Buffers: mappings, making buffers, their properties, host access and the low
   level. Copying is test_copy's, io memory test_io's. Polled logs what its
   driver is asked. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1
let kib = 1024
let memory name = Result.get_ok (Rig.software name)
let lost = function Rig.Lost _ -> true | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

let buffer =
  Testable.make
    ~pp:(fun ppf b ->
      Format.fprintf ppf "a buffer of %d bytes on %a, %d into its memory"
        (B.length b) Rig.pp (B.device b) (B.offset b))
    ~equal:( == )

let bytes b =
  B.read b Bigarray.char (fun ba ->
      String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba))

let filled n c =
  let b = B.create Rig.host n in
  B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a c);
  b

(* [b]'s bytes, through a blocking copy into host memory. *)
let contents b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.unsafe_to_string s

(* Collects, then drains [d], as an allocation on it does first. *)
let collect d =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create d 0))

(* A submission on [d] of one part that writes its buffer. *)
let writes d = Sub.make d [| Support.touching [| Sub.write |] |]
let reads d = Sub.make d [| Support.touching [| Sub.read |] |]

(* The host's page size, read as rig_heap.c reads it. *)
let page = Rig_gpu_support.Host.page

(* Mappings *)

(* A host buffer a device maps is mapped once, whatever its borrows. *)
let test_maps_once () =
  let d, p = P.open_ "buffer:map-once" in
  let h = B.create Rig.host B.paged_from in
  for _ = 1 to 3 do
    ignore (Sys.opaque_identity (B.borrow d h))
  done;
  equal ~msg:"host mappings" int 1 (count "map_host" p)

(* Borrows on one device of one allocation share one mapping, which goes once
   the last of them goes. A borrow of a view ends as a borrow of the bytes it
   was made over. *)
let test_borrows_share () =
  let d, p = P.open_ "buffer:borrows-share" in
  let h = B.create Rig.host B.paged_from in
  let b = require_some (B.borrow d h) in
  let part = require_some (B.borrow d (B.view h ~first:4096 ~length:4096)) in
  equal ~msg:"one mapping" (pair int int) (1, 1) (count "map_host" p, P.mapped p);
  equal ~msg:"the borrow of a view ends" bool true (B.unborrow part);
  ignore (B.create d 0);
  equal ~msg:"the other holds it" int 1 (P.mapped p);
  equal ~msg:"the last ends" bool true (B.unborrow b);
  ignore (B.create d 0);
  equal ~msg:"the last gone" (pair int int) (1, 0) (count "unmap" p, P.mapped p)

(* A borrow's mapping lives as long as the borrow: collected while its memory
   lives, the borrow gives its mapping back once the device ran the work
   submitted before, and a borrow made again maps again. *)
let test_remade () =
  let d, p = P.open_ "buffer:remade" in
  let h = B.create Rig.host B.paged_from in
  ignore (Sys.opaque_identity (B.borrow d h));
  collect d;
  equal ~msg:"once collected: unmaps, mappings" (pair int int) (1, 0)
    (count "unmap" p, P.mapped p);
  ignore (require_some (B.borrow d h));
  equal ~msg:"made again: maps, mappings" (pair int int) (2, 1)
    (count "map_host" p, P.mapped p)

(* A mapping lasts while its memory lives, and is released with it once the work
   the mapper was handed until then is done. *)
let test_released () =
  let d, p = P.open_ "buffer:released" in
  (fun () ->
    let h = B.create Rig.host B.paged_from in
    ignore
      (submit ~args:one (reads d) ~buffers:[| require_some (B.borrow d h) |]))
    ();
  let drain () =
    collect Rig.host;
    ignore (B.create d 0)
  in
  drain ();
  equal ~msg:"unmaps while its work is unrun" int 0 (count "unmap" p);
  ignore (P.run p);
  drain ();
  equal ~msg:"unmaps once it ran" int 1 (count "unmap" p)

(* Borrowed bytes return to their keeper only once every device's mapping of
   them is released: a device whose unmap waits for its other work keeps them
   from the collector until it ran. *)
let test_bytes_after_unmap () =
  let d, p = P.open_ "buffer:bytes-after-unmap" in
  let collected = ref false in
  (fun () ->
    let paged = B.export (B.create Rig.host B.paged_from) Bigarray.char in
    let ba = Bigarray.Array1.sub paged 0 B.paged_from in
    Gc.finalise (fun _ -> collected := true) ba;
    let h = B.of_bigarray ba in
    ignore
      (submit ~args:one (reads d) ~buffers:[| require_some (B.borrow d h) |]);
    ignore (P.run p))
    ();
  ignore (submit (Sub.make d [||]));
  let drain () =
    collect Rig.host;
    ignore (B.create d 0);
    Gc.full_major ();
    Gc.full_major ()
  in
  drain ();
  equal ~msg:"while d's other work is unrun: unmaps, bytes collected"
    (pair int bool) (0, false)
    (count "unmap" p, !collected);
  ignore (P.run p);
  drain ();
  equal ~msg:"once it ran" (pair int bool) (1, true)
    (count "unmap" p, !collected)

(* Empty memory and the device that borrows it, with no mapping: on a driver's
   device, on one that maps no host memory, and on the host, of memory a
   software device and a driver's device hold. *)
let empty_borrows =
  let empty_paged () =
    let paged = B.export (B.create Rig.host B.paged_from) Bigarray.char in
    B.of_bigarray (Bigarray.Array1.sub paged 0 0)
  in
  let on_driver ?maps_host name =
    let d, p = P.open_ ?maps_host name in
    (d, Some p)
  in
  [
    ( "empty host memory on a page, on a driver's device",
      fun () -> (empty_paged (), on_driver "buffer:empty-borrow") );
    ( "empty host memory, on a device that maps no host memory",
      fun () ->
        (B.create Rig.host 0, on_driver ~maps_host:false "buffer:empty-mapless")
    );
    ( "a software device's empty memory, on the host",
      fun () -> (B.create (memory "buffer:empty-software") 0, (Rig.host, None))
    );
    ( "a driver's device's empty memory, on the host",
      fun () ->
        let d, _ = P.open_ "buffer:empty-driver" in
        (B.create d 0, (Rig.host, None)) );
  ]

(* Empty memory borrows on every device of its machine, and the borrow names no
   memory: a driver's device maps nothing for it. *)
let test_empty_borrow (_, make) =
  let m, (d, p) = make () in
  let b = require_some (B.borrow d m) in
  equal ~msg:"its length" int 0 (B.length b);
  Option.iter
    (fun p ->
      equal ~msg:"host mappings" (list int) [] (P.host_maps p);
      equal ~msg:"its address" int 0
        (Sub.address (Sub.make ~fixed:[ (b, Rig.Read) ] d [||]) b))
    p

(* t *)

(* Two buffers that are not [==], made fresh, and whether they are on one
   device. *)
type devices = One_device | Two_devices

let distinct =
  let two f = (f (), f ()) in
  let viewed () =
    let b = B.create Rig.host 16 in
    (B.view b ~first:0 ~length:8, B.view b ~first:0 ~length:8)
  in
  let borrowed () =
    let h = B.create Rig.host 16 in
    (h, require_some (B.borrow (memory "buffer:compare-borrow") h))
  in
  (* An empty buffer, which holds no memory, of a device of another machine,
     closed, and one of the device opened again under its name, which took its
     slot. *)
  let slot_again () =
    let far =
      Result.get_ok
        (Rig.open_host ~machine:"buffer-compare-far"
           (P.opener "buffer:compare-far-host" (P.make ())))
    in
    let beside () =
      Result.get_ok
        (Rig.open_ ~machine:far (P.opener "buffer:compare-far" (P.make ())))
    in
    let d = beside () in
    let a = B.create d 0 in
    let reached = submit (Sub.make d [||]) in
    Rig.Point.wait reached;
    Rig.close d;
    Gc.full_major ();
    let b = B.create (beside ()) 0 in
    is_none ~msg:"the closed device's slot passed on" (Rig.Point.device reached);
    (a, b)
  in
  let bigarray () =
    let a = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4 in
    Bigarray.Array1.fill a 'b';
    B.of_bigarray a
  in
  [
    ( "two host buffers",
      One_device,
      fun () -> two (fun () -> B.create Rig.host 16) );
    ( "two empty host buffers",
      One_device,
      fun () -> two (fun () -> B.create Rig.host 0) );
    ( "two host buffers once the host filled one",
      One_device,
      fun () ->
        let b = B.create Rig.host 16 in
        Rig.Point.wait (B.submit_fill b 'f');
        (b, B.create Rig.host 16) );
    ( "a buffer and its view",
      One_device,
      fun () ->
        let b = B.create Rig.host 16 in
        (b, B.view b ~first:0 ~length:16) );
    ("two views of one window", One_device, viewed);
    ( "two buffers of a device",
      One_device,
      fun () ->
        let d = memory "buffer:compare" in
        two (fun () -> B.create d 64) );
    ( "buffers of two devices",
      Two_devices,
      fun () ->
        (B.create Rig.host 64, B.create (memory "buffer:compare-other") 64) );
    ("a buffer and its borrow", Two_devices, borrowed);
    ( "buffers of two devices of one slot of another machine",
      Two_devices,
      slot_again );
    ( "two strings' buffers",
      One_device,
      fun () -> two (fun () -> B.of_string "abc") );
    ("two bigarrays' buffers", One_device, fun () -> two bigarray);
  ]

let test_compare (_, _, make) =
  let a, b = make () in
  let sign x = Int.compare x 0 in
  equal ~msg:"compare a b against compare b a" int
    (sign (compare a b))
    (-sign (compare b a));
  equal ~msg:"compare of lists" int
    (sign (compare a b))
    (sign (compare [ a ] [ b ]))

let test_equality (_, devices, make) =
  let a, b = make () in
  raises_match ~msg:"= on itself" Exn.invalid_arg (fun () -> a = a);
  match devices with
  | One_device -> raises_match ~msg:"=" Exn.invalid_arg (fun () -> a = b)
  | Two_devices -> equal ~msg:"=" bool false (a = b)

(* kind *)

let allocs =
  Testable.make
    ~pp:
      (Format.pp_print_list (fun ppf (k, n, ok) ->
           Format.fprintf ppf "%s %d %b"
             (match k with
             | B.Device -> "device"
             | B.Pinned -> "pinned"
             | B.Mapped -> "mapped")
             n ok))
    ~equal:( = )

let flushes p = List.length (List.filter (String.equal "flush") (P.log p))

(* A device whose memory the host addresses has no pinned memory: a create of it
   raises, asking its driver nothing, on the host, a software device and a
   driver's device that declares [host_addresses], of no bytes too. *)
let test_pinned_own () =
  let d, p = P.open_ "buffer:pinned-own" in
  let m = memory "buffer:pinned-own-memory" in
  let refused = Exn.invalid_arg ~sub:"Buffer.borrow" in
  List.iter
    (fun (msg, d, n) ->
      raises_match ~msg refused (fun () -> B.create ~memory:Pinned d n))
    [
      ("the host", Rig.host, 4 * kib);
      ("the host, no bytes", Rig.host, 0);
      ("a software device", m, 4 * kib);
      ("a driver's device the host addresses", d, 4 * kib);
      ("one of no bytes", d, 0);
    ];
  equal ~msg:"asked of the driver" allocs [] (P.allocs p)

(* Where the host addresses a device's memory, Mapped memory is its Device
   memory: its driver is asked for Device memory, which the host reads, and no
   write of it flushes, whether or not its driver makes Mapped memory. *)
let test_mapped_own () =
  List.iter
    (fun mapped ->
      let msg = if mapped then "with Mapped memory" else "without" in
      let d, p =
        P.open_ ~mapped (if mapped then "buffer:mapped-own" else "buffer:own")
      in
      let b = B.create ~memory:Mapped d (4 * kib) in
      equal ~msg allocs [ (B.Device, 4 * kib, true) ] (P.allocs p);
      B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a 'm');
      equal ~msg string (String.make 4 'm')
        (B.read b Bigarray.char (fun a -> String.init 4 (Bigarray.Array1.get a)));
      equal ~msg:(msg ^ ": flushes") int 0 (flushes p))
    [ true; false ]

let window = 4 * kib

let window_pattern k =
  String.init window (fun i -> Char.chr (((i * k) + 1) land 255))

(* Each of rig's calls that writes Mapped memory of a device whose memory the
   host does not address, and the bytes it leaves in a window that held
   [window_pattern 1]. *)
let window_writes =
  [
    ( "write",
      (fun m ->
        B.write m Bigarray.char (fun a ->
            String.iteri
              (fun i c -> Bigarray.Array1.set a i c)
              (window_pattern 3))),
      window_pattern 3 );
    ( "blit_from_string",
      (fun m -> B.blit_from_string (window_pattern 5) 7 m 7 (window - 9)),
      String.sub (window_pattern 1) 0 7
      ^ String.sub (window_pattern 5) 7 (window - 9)
      ^ String.sub (window_pattern 1) (window - 2) 2 );
    ( "a copy in",
      (fun m -> B.copy ~src:(B.of_string (window_pattern 7)) ~dst:m),
      window_pattern 7 );
    ("fill", (fun m -> B.fill m 'f'), String.make window 'f');
    ( "submit_fill",
      (fun m -> Rig.Point.wait (B.submit_fill m 's')),
      String.make window 's' );
    ( "a write that raises after a store",
      (fun m ->
        raises ~msg:"its exception" (Failure "after a store") (fun () ->
            B.write m Bigarray.char (fun a ->
                Bigarray.Array1.set a 0 'r';
                failwith "after a store"))),
      "r" ^ String.sub (window_pattern 1) 1 (window - 1) );
  ]

(* Each write of Mapped memory calls the driver's flush once before it returns,
   and later work, a copy out on the device, reads what it wrote. *)
let test_window_write (name, write, want) =
  let slug = String.map (fun c -> if c = ' ' then '-' else c) name in
  let d, p = P.open_ ~host_visible:false ("buffer:mapped-window-" ^ slug) in
  let m = B.create ~memory:Mapped d window in
  equal ~msg:"asked of the driver" allocs
    [ (B.Mapped, window, true) ]
    (P.allocs p);
  B.copy ~src:(B.of_string (window_pattern 1)) ~dst:m;
  let before = flushes p in
  write m;
  equal ~msg:"flushes" int (before + 1) (flushes p);
  equal ~msg:"the window a copy out reads" string want (contents m)

(* A copy out of Mapped memory runs on its device's copy engine, as one out of
   Device memory does, and a copy in from memory the host addresses on the
   host. *)
let test_window_copies () =
  let d, p = P.open_ ~host_visible:false "buffer:mapped-copies" in
  let m = B.create ~memory:Mapped d window in
  B.copy ~src:(B.of_string (window_pattern 3)) ~dst:m;
  equal ~msg:"device copies of a copy in" int 0 (List.length (P.copies p));
  ignore (contents m);
  equal ~msg:"device copies of a copy out" int 1 (List.length (P.copies p))

(* The host neither reads Mapped memory of a device whose memory it does not
   address, borrows it, nor gives C its address. *)
let test_window_unaddressed () =
  let d, _ = P.open_ ~host_visible:false "buffer:mapped-unaddressed" in
  let m = B.create ~memory:Mapped d window in
  raises_match ~msg:"Buffer.read" Exn.invalid_arg (fun () ->
      B.read m Bigarray.char Bigarray.Array1.dim);
  is_none ~msg:"a borrow on the host" ~pp:(Testable.pp buffer)
    (B.borrow Rig.host m);
  equal ~msg:"rig_buffer_host" int 0 (Support.Reader.host m)

let out_of_memory d n = function
  | Rig.Out_of_memory (d', n') -> Rig.equal d d' && n = n'
  | _ -> false

(* Mapped memory the window cannot hold raises Out_of_memory once reclamation
   ran and emptied the device's cache, asking for no other kind. *)
let test_window_room () =
  let d, p = P.open_ ~host_visible:false ~window "buffer:window" in
  let at = (fun () -> Support.at (B.create d window)) () in
  collect d;
  raises_match
    (out_of_memory d (2 * window))
    (fun () -> B.create ~memory:Mapped d (2 * window));
  mem ~msg:"the cached memory among the frees" int at (List.map fst (P.frees p));
  let asked kind =
    List.filter (fun (k, n, _) -> n = 2 * window && kind k) (P.allocs p)
  in
  equal ~msg:"other kinds asked of the driver" allocs []
    (asked (fun k -> k <> B.Mapped));
  greater ~msg:"Mapped memory asked of the driver" int ~than:0
    (List.length (asked (fun k -> k = B.Mapped)))

(* paged_from *)

let test_paged_from () =
  equal ~msg:"64 KiB, or four of the host's pages" int
    (Int.max (64 * kib) (4 * page))
    B.paged_from

(* create *)

(* A host buffer that [create] or [of_string] makes. *)
type maker = Create | Of_string

let maker =
  Gen.of_list
    ~pp:(fun ppf m ->
      Format.pp_print_string ppf
        (match m with Create -> "create" | Of_string -> "of_string"))
    [ Create; Of_string ]

let host_buffer m n =
  match m with
  | Create -> B.create Rig.host n
  | Of_string -> B.of_string (String.make n 'p')

(* A host buffer of 128 bytes to fewer than paged_from starts on 128 bytes, as
   [create] and [of_string] place it. *)
let line_case =
  Gen.pair maker
    (Gen.with_pp
       (fun ppf n -> Format.fprintf ppf "%d bytes" n)
       (Gen.int_range 128 (B.paged_from - 1)))

let line_law (m, n) =
  cover "create" (m = Create);
  cover "of_string" (m = Of_string);
  equal ~msg:"aligned on 128" bool true (B.aligned (host_buffer m n) ~at:0 128)

(* A host buffer of paged_from bytes or more starts on a page. *)
let page_case =
  Gen.pair maker
    (Gen.with_pp
       (fun ppf n -> Format.fprintf ppf "%d bytes" n)
       (Gen.int_range B.paged_from (4 * B.paged_from)))

let page_law (m, n) =
  cover "create" (m = Create);
  cover "of_string" (m = Of_string);
  equal ~msg:"aligned on a page" bool true
    (B.aligned (host_buffer m n) ~at:0 page)

(* An empty buffer allocates nothing: its driver is asked for nothing, and the
   work of its device addresses it at 0. *)
let test_empty () =
  let d, p = P.open_ "buffer:empty" in
  let b = B.create d 0 in
  equal ~msg:"asked of the driver" allocs [] (P.allocs p);
  equal ~msg:"its address" int 0
    (Sub.address (Sub.make ~fixed:[ (b, Rig.Read) ] d [||]) b)

(* A non-empty buffer's memory runs on to whole words, through bytes that belong
   to no other memory: a write of every word that holds a byte of a view, at the
   end or at an offset, or of a borrow changes no byte outside those words, and
   a driver is asked for whole words. Sizes lie on either side of the host's
   line and page placements. *)
let test_whole_words () =
  let d = memory "buffer:tail" in
  let p, polled = P.open_ ~alignment:256 "buffer:tail-polled" in
  let hidden, unseen =
    P.open_ ~alignment:256 ~host_visible:false "buffer:tail-hidden"
  in
  let w = Rig.word in
  let sizes = [ 1; 3; 15; 17; 63; 65; 127; 129; 255; 257; 65537; 262145 ] in
  let pattern n seed =
    String.init n (fun i -> Char.chr ((seed + i) land 255))
  in
  let made =
    List.concat_map
      (fun (d, kinds) ->
        List.concat_map
          (fun memory -> List.map (fun n -> B.create ~memory d n) sizes)
          kinds)
      B.
        [
          (Rig.host, [ Device; Mapped ]);
          (d, [ Device; Mapped ]);
          (p, [ Device; Mapped ]);
          (hidden, [ Pinned ]);
        ]
  in
  (* The bytes of [b] whose words are written, as [(first, length, borrow)]: its
     last byte, a view from an offset off a word to before its end, and on the
     host, its first bytes through a borrow by [d] where [b] starts on a
     page. *)
  let ranges b =
    let n = B.length b in
    let last = [ (n - 1, 1, false) ] in
    let inner = if n < 7 then [] else [ (1, n - 6, false) ] in
    let first =
      if B.device b == Rig.host && n >= B.paged_from then [ (0, 5, true) ]
      else []
    in
    last @ inner @ first
  in
  List.iteri
    (fun i b ->
      let n = B.length b in
      B.blit_from_string (pattern n i) 0 b 0 n)
    made;
  let marks = B.of_string (String.make 8 '\xee') in
  (* Writes every word that holds a byte of [v], through its host address. *)
  let mark v =
    let a = Support.Reader.host v and n = B.length v in
    let rec go at =
      if at < (a + n + w - 1) / w * w then begin
        Support.move ~dst:at ~src:(Support.Reader.host marks) w;
        go (at + w)
      end
    in
    go (a / w * w)
  in
  List.iter
    (fun b ->
      List.iter
        (fun (first, length, borrow) ->
          let v = B.view b ~first ~length in
          mark (if borrow then require_some (B.borrow d v) else v))
        (ranges b))
    made;
  List.iteri
    (fun i b ->
      let n = B.length b in
      let expected = Bytes.of_string (pattern n i) in
      List.iter
        (fun (first, length, _) ->
          let lo = first / w * w and hi = (first + length + w - 1) / w * w in
          Bytes.fill expected lo (Int.min hi n - lo) '\xee')
        (ranges b);
      equal
        ~msg:(Printf.sprintf "%s, %d bytes" (Rig.name (B.device b)) n)
        string (Bytes.to_string expected) (contents b))
    made;
  List.iter
    (fun (_, n, _) ->
      equal ~msg:(Printf.sprintf "%d bytes asked" n) int 0 (n mod w))
    (P.allocs polled @ P.allocs unseen);
  Gc.full_major ()

let test_negative () =
  raises_match Exn.invalid_arg (fun () -> B.create Rig.host (-1))

(* of_string *)

let of_string_law s =
  cover "no bytes" (s = "");
  equal string s (bytes (B.of_string s))

(* of_bigarray *)

type kind = Kind : string * ('a, 'b) Bigarray.kind * (int -> 'a) -> kind

let kinds =
  Bigarray.
    [
      Kind ("float16", float16, float_of_int);
      Kind ("float32", float32, float_of_int);
      Kind ("float64", float64, float_of_int);
      Kind ("int8_signed", int8_signed, Fun.id);
      Kind ("int8_unsigned", int8_unsigned, Fun.id);
      Kind ("char", char, Char.chr);
      Kind ("int16_signed", int16_signed, Fun.id);
      Kind ("int16_unsigned", int16_unsigned, Fun.id);
      Kind ("int32", int32, Int32.of_int);
      Kind ("int64", int64, Int64.of_int);
      Kind ("int", int, Fun.id);
      Kind ("nativeint", nativeint, Nativeint.of_int);
      Kind
        ( "complex32",
          complex32,
          fun i -> { Complex.re = float_of_int i; im = 0. } );
      Kind
        ( "complex64",
          complex64,
          fun i -> { Complex.re = float_of_int i; im = 0. } );
    ]

(* A bigarray's buffer holds its bytes without a copy: writes to the bigarray
   after the wrap show through [export] of the buffer. *)
let test_kind (Kind (_, k, v)) =
  List.iter
    (fun n ->
      let ba = Bigarray.Array1.create k Bigarray.c_layout n in
      let b = B.of_bigarray ba in
      equal ~msg:"bytes" int (n * Bigarray.kind_size_in_bytes k) (B.length b);
      let back = B.export b k in
      for i = 0 to n - 1 do
        ba.{i} <- v i
      done;
      for i = 0 to n - 1 do
        if back.{i} <> v i then failf "element %d of %d differs" i n
      done)
    [ 0; 1; 17 ]

(* view *)

let test_view_bytes () =
  let b = filled 8 'a' in
  let v = B.view b ~first:4 ~length:4 in
  B.write v Bigarray.char (fun a -> Bigarray.Array1.fill a 'b');
  equal string "aaaabbbb" (bytes b)

(* A view of a buffer of [size] bytes, from [first], of [length] bytes, near and
   past the bounds. *)
let view_case =
  let open Gen in
  let* size = int_range 0 32 in
  let near n = frequency [ (4, int_range 0 n); (1, int_range (-2) (n + 2)) ] in
  let* first = near size in
  let+ length = near (Int.max 0 (size - first)) in
  (size, first, length)

let view_case =
  Gen.with_pp
    (fun ppf (size, first, length) ->
      Format.fprintf ppf "[%d,+%d] of %d bytes" first length size)
    view_case

(* A view is the bytes it names, [length] from [first] into its buffer's memory,
   or raises where they do not lie inside the buffer's. *)
let view_law (size, first, length) =
  let inside = first >= 0 && length >= 0 && first + length <= size in
  cover "inside" inside;
  cover "outside" (not inside);
  cover "an empty buffer" (size = 0);
  cover "past the end by one"
    (first >= 0 && length >= 0 && first + length = size + 1);
  let b = B.view (B.create Rig.host (size + 8)) ~first:8 ~length:size in
  let got =
    match B.view b ~first ~length with
    | v -> Some (B.length v, B.offset v - B.offset b)
    | exception Invalid_argument _ -> None
  in
  equal ~msg:"length and first byte"
    (option (pair int int))
    (if inside then Some (length, first) else None)
    got

let view_examples =
  [
    (0, 0, 0);
    (0, 0, 1);
    (0, 1, 0);
    (16, 12, 8);
    (16, -1, 1);
    (16, 0, -1);
    (16, 17, 0);
  ]

(* borrow *)

(* A borrow lies over the memory it maps: on its device, at the host address of
   the memory, which it overlaps. *)
let test_borrow () =
  let d = memory "buffer:borrow" in
  let h = filled 32 'h' in
  let b = require_some (B.borrow d h) in
  equal ~msg:"its device" device d (B.device b);
  equal ~msg:"its host address" int (Support.Reader.host h)
    (Support.Reader.host b);
  equal ~msg:"its bytes" string (String.make 32 'h') (bytes b)

let test_borrow_own () =
  let d, _ = P.open_ "buffer:own" in
  let b = B.create d 64 in
  satisfies ~claim:"is the buffer itself" (option buffer)
    (function Some b' -> b' == b | None -> false)
    (B.borrow d b)

let test_off_page () =
  let d, _ = P.open_ "buffer:small" in
  let paged = B.create Rig.host B.paged_from in
  let off_page = Bigarray.Array1.sub (B.export paged Bigarray.char) 8 16 in
  is_none ~pp:(Testable.pp buffer) (B.borrow d (B.of_bigarray off_page))

(* Polled that notes each host mapping rig asks of it off a page. *)
let off_page_maps = ref []

module Paged = struct
  include P

  let map_host d p n =
    if p mod page <> 0 then off_page_maps := p :: !off_page_maps;
    P.map_host d p n
end

(* rig asks a driver to map host memory only from the start of a page: memory of
   a software device that starts elsewhere is none that a driver's device
   borrows, though it reaches the device. *)
let test_mapped_from_page () =
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_
         {
           name = "buffer:paged";
           driver = (module Paged);
           make = (fun ~word -> Ok (P.bind (P.make ()) ~word));
         })
  in
  let m = memory "buffer:paged-memory" in
  let rec off () =
    let b = B.create m 64 in
    if Support.at b mod page <> 0 then b else off ()
  in
  let b = off () in
  equal ~msg:"reaches" bool true (Rig.reaches d m);
  is_none ~msg:"borrow" ~pp:(Testable.pp buffer) (B.borrow d b);
  equal ~msg:"mappings off a page" (list int) [] !off_page_maps

(* A device that maps no host memory reaches none, and a borrow of host memory
   that starts on a page is [None], with no mapping asked of its driver. *)
let test_maps_no_host () =
  let d, p = P.open_ ~maps_host:false "buffer:no-host-maps" in
  let h = B.create Rig.host B.paged_from in
  equal ~msg:"reaches" bool false (Rig.reaches d Rig.host);
  is_none ~msg:"borrow" ~pp:(Testable.pp buffer) (B.borrow d h);
  equal ~msg:"maps" (list int) [] (P.host_maps p)

(* A device that maps no host memory reaches no software device's memory, which
   a borrow maps as host memory: [reaches] answers as the borrow does. *)
let test_mapless_software () =
  let d, _ = P.open_ ~maps_host:false "buffer:mapless-memory" in
  let m = memory "buffer:mapless-memory-cpu" in
  let b = B.create m B.paged_from in
  equal ~msg:"reaches as borrows" bool
    (Option.is_some (B.borrow d b))
    (Rig.reaches d m)

(* A device that maps no host memory maps a peer's memory the host addresses,
   through its driver's peer mapping. *)
let test_peer_no_host () =
  let owner, _ = P.open_ "buffer:peer-visible" in
  let d, p = P.open_ ~maps_host:false "buffer:peer-no-host" in
  ignore (require_some ~msg:"borrowed" (B.borrow d (B.create owner 64)));
  equal ~msg:"host maps" (list int) [] (P.host_maps p)

let test_borrow_peer () =
  let d, _ = P.open_ ~host_visible:false "buffer:owner" in
  let e, pe = P.open_ ~host_visible:false "buffer:peer" in
  let m = B.create d 64 in
  let b = require_some (B.borrow e m) in
  equal ~msg:"peer mappings" int 1 (count "map_peer" pe);
  equal ~msg:"its device" device e (B.device b);
  equal ~msg:"overlaps its memory" bool true (B.overlaps b m)

(* A borrow of a borrow maps the memory under it. *)
let test_borrow_of_borrow () =
  let d, _ = P.open_ "buffer:first" in
  let e, pe = P.open_ "buffer:second" in
  let h = B.create Rig.host B.paged_from in
  let b = require_some (B.borrow e (require_some (B.borrow d h))) in
  equal ~msg:"the second device's host mappings" (list int) [ B.paged_from ]
    (P.host_maps pe);
  equal ~msg:"its host address" int (Support.Reader.host h)
    (Support.Reader.host b)

(* Where a buffer's memory lies, as each reader states it: its driver object,
   its host address, the address its device's work uses, [None] where a reader
   refuses, and whether it is borrowed. *)
let coordinates b =
  let answer f = try Some (f ()) with Invalid_argument _ -> None in
  let address () =
    Sub.address (Sub.make ~fixed:[ (b, Rig.Read) ] (B.device b) [||]) b
  in
  ( answer (fun () -> B.handle b),
    Support.Reader.host b,
    answer address,
    B.is_borrowed b )

(* Memory of each kind of owner, and a device of other coordinates that borrows
   it. A hidden device's memory has no host address; one without addresses names
   its memory by handle alone. *)
let owners =
  let hidden name = fst (P.open_ ~host_visible:false name) in
  let peer () = hidden "buffer:back-peer" in
  let paged () = B.export (B.create Rig.host B.paged_from) Bigarray.char in
  [
    ("device memory", fun () -> (B.create (hidden "buffer:back") 64, peer ()));
    ( "memory named by handle",
      fun () ->
        let d, _ =
          P.open_ ~host_visible:false ~addresses:false "buffer:back-handle"
        in
        (B.create d 64, peer ()) );
    ( "pinned memory",
      fun () ->
        (B.create ~memory:Pinned (hidden "buffer:back-pinned") 64, peer ()) );
    ("no bytes", fun () -> (B.create (hidden "buffer:back-empty") 0, peer ()));
    ("host memory", fun () -> (B.create Rig.host B.paged_from, peer ()));
    ("a bigarray", fun () -> (B.of_bigarray (paged ()), peer ()));
  ]

(* A borrow back on the device whose memory it is lies where that memory does,
   by every reader. *)
let test_borrow_back (_, owner) =
  let m, via = owner () in
  let on_via = require_some (B.borrow via m) in
  let back = require_some (B.borrow (B.device m) on_via) in
  equal ~msg:"device" device (B.device m) (B.device back);
  equal ~msg:"coordinates"
    (quad (option nativeint) int (option int) bool)
    (coordinates m) (coordinates back);
  equal ~msg:"overlaps" bool (B.length m > 0) (B.overlaps m back)

(* A borrow of a lost device's buffer raises its loss, empty or not. *)
let test_borrow_lost () =
  let d, p = P.open_ "buffer:borrow-lost" in
  let e, _ = P.open_ "buffer:borrow-lost-on" in
  let m = B.create d 64 and empty = B.create d 0 in
  P.fail p;
  (try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ());
  raises_match ~msg:"64 bytes" lost (fun () -> B.borrow e m);
  raises_match ~msg:"empty" lost (fun () -> B.borrow e empty)

let host_mappings p =
  List.filter (fun c -> c = "map_host" || c = "unmap") (P.log p)

(* A mapping a driver has no room for returns memory once and asks again, and a
   borrow that finds room then maps. The memory returned first includes the
   mappings of borrows collected on any device of the machine, once their work
   ran, and the cached Pinned memory of every device. *)
let test_room () =
  let d, p = P.open_ "buffer:borrow-room" in
  let e, pe = P.open_ ~host_visible:false "buffer:borrow-room-other" in
  let n = B.paged_from in
  (* A borrow whose work is queued, unrun, dropped: its mapping waits for the
     work, which only the room's wait runs. *)
  (fun () ->
    let b0 = require_some (B.borrow d (B.create Rig.host n)) in
    ignore (submit ~args:one (reads d) ~buffers:[| b0 |]))
    ();
  ignore (Sys.opaque_identity (B.borrow e (B.create Rig.host n)));
  ignore (Sys.opaque_identity (B.create ~memory:Pinned e n));
  collect e;
  P.fail_at p 1 (`Refuse 1);
  ignore
    (require_some ~msg:"room at the second ask"
       (B.borrow d (B.create Rig.host n)));
  equal ~msg:"the collected borrow's mapping went before the second ask"
    (list string)
    [ "map_host"; "map_host"; "unmap"; "map_host" ]
    (host_mappings p);
  equal ~msg:"the other device's unmaps and frees" (pair int int) (1, 1)
    (count "unmap" pe, count "free" pe)

(* A mapping a driver refuses room at both asks raises Out_of_memory for the
   memory's bytes. *)
let test_no_room () =
  let d, p = P.open_ "buffer:borrow-no-room" in
  let n = B.paged_from in
  let h = B.create Rig.host n in
  P.fail_at p 1 (`Refuse 2);
  raises_match (out_of_memory d n) (fun () -> B.borrow d h);
  equal ~msg:"asks" int 2 (count "map_host" p)

(* A mapping the driver never makes asks once, and the borrow is [None]. *)
let test_never_mapped () =
  let d, p = P.open_ "buffer:never-mapped" in
  P.never_map p true;
  is_none ~msg:"the borrow" ~pp:(Testable.pp buffer)
    (B.borrow d (B.create Rig.host B.paged_from));
  equal ~msg:"asks" int 1 (count "map_host" p)

(* The staging memory a room's return gives back is unmapped before the second
   ask, even while the device that maps it has other work queued: the return
   waits for that work. *)
let test_room_staging () =
  let d, p = P.open_ ~host_visible:false "buffer:borrow-room-staging" in
  let n = B.paged_from in
  B.copy ~src:(B.create Rig.host n) ~dst:(B.create d n);
  ignore (submit ~args:one (reads d) ~buffers:[| B.create d 64 |]);
  P.fail_at p 1 (`Refuse 1);
  ignore (require_some (B.borrow d (B.create Rig.host n)));
  equal ~msg:"staging mapped, refused, unmapped, then the borrow mapped"
    (list string)
    [ "map_host"; "map_host"; "unmap"; "map_host" ]
    (host_mappings p)

(* free *)

(* A freed buffer's memory returns once the work using it is done, while the
   buffer is still reachable, dead with the reason "freed": an allocation of its
   size reuses it then, and not before. *)
let test_free () =
  let d, p = P.open_ "buffer:free" in
  let b = B.create d (4 * kib) in
  ignore (submit ~args:one (writes d) ~buffers:[| b |]);
  let at = Support.at b in
  equal ~msg:"freed" bool true (B.free b);
  equal ~msg:"its death" (option string) (Some "freed") (B.dead b);
  let other = B.create d (4 * kib) in
  not_equal ~msg:"another allocation while its work is queued" int at
    (Support.at other);
  ignore (P.run p);
  equal ~msg:"an allocation once its work ran" int at
    (Support.at (B.create d (4 * kib)));
  ignore (Sys.opaque_identity (b, other))

(* A free given a reason kills the buffer with it, and every buffer over its
   memory. *)
let test_free_why () =
  let d, _ = P.open_ "buffer:free-why" in
  let b = B.create d 64 in
  let v = B.view b ~first:0 ~length:8 in
  equal ~msg:"freed" bool true (B.free ~why:"donated to f" b);
  equal ~msg:"its death and its view's"
    (pair (option string) (option string))
    (Some "donated to f", Some "donated to f")
    (B.dead b, B.dead v)

(* Freed host memory that a device borrowed returns once the device's work on it
   ran and the collector has run, while the buffer is still reachable. *)
let test_free_host () =
  let d, p = P.open_ "buffer:free-host" in
  let n = 1 lsl 20 in
  let h = B.create Rig.host n in
  ignore
    (submit ~args:one (writes d) ~buffers:[| require_some (B.borrow d h) |]);
  let settle () =
    collect Rig.host;
    Gc.full_major ();
    Rig.allocated Rig.host
  in
  let before = settle () in
  equal ~msg:"freed" bool true (B.free h);
  at_least ~msg:"while the work on it is queued" int ~than:before (settle ());
  ignore (P.run p);
  at_most ~msg:"once the work ran" int ~than:(before - n) (settle ());
  ignore (Sys.opaque_identity h)

(* A free where a take would be [None] is [false] and changes nothing: of a part
   of the memory, of memory a claim reads or a keep holds, of memory outside the
   claims, and of memory that admits only [Read]. *)
let test_free_refused () =
  let d, _ = P.open_ "buffer:free-refused" in
  let kept msg b =
    equal
      ~msg:(msg ^ ": freed and its death")
      (pair bool (option string))
      (false, None)
      (B.free b, B.dead b)
  in
  let b = B.create d 64 in
  kept "a view" (B.view b ~first:0 ~length:8);
  Support.Reader.read b;
  kept "read by a claim" b;
  Support.Reader.release b;
  let k = B.keep [ b ] in
  kept "kept" b;
  ignore (Sys.opaque_identity k);
  let shared = B.create Rig.host 64 in
  B.share shared;
  kept "shared" shared;
  kept "a bigarray's"
    (B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8));
  kept "read-only memory"
    (Support.io_buffer ~access:Read (Support.io "buffer:free-read-only") 8)

let test_free_dead () =
  let b = B.create Rig.host 64 in
  equal ~msg:"a live buffer" bool true (B.free b);
  raises_match (Exn.invalid_arg ~sub:"freed") (fun () -> B.free b)

(* A free of a borrow returns its memory, as a free of any buffer that spans it
   does: also of the buffer a take of the borrow gave. Every buffer over the
   memory is then dead, and its mappings go once its work is done. *)
let test_free_borrow () =
  let d, p = P.open_ "buffer:borrow-free" in
  let h = B.create Rig.host B.paged_from in
  let b = require_some (B.borrow d h) in
  let b' = require_some (Support.donated b) in
  equal ~msg:"freed" bool true (B.free b');
  equal ~msg:"its memory's death" (option string) (Some "freed") (B.dead h);
  ignore (B.create Rig.host 0);
  ignore (B.create d 0);
  equal ~msg:"its mapping gone" (pair int int) (1, 0)
    (count "unmap" p, P.mapped p)

(* Two domains: a wait held at its device's point while a free on the other
   kills its buffer, and the memory returns once the work ran. The wait walks
   the stamps it began on, which live while the dead buffer is reachable, so it
   returns. Run with MallocScribble=1 (macOS) or under AddressSanitizer, a walk
   of freed stamps reads garbage. *)
let test_free_beside_wait () =
  let d, p = P.open_ "buffer:free-wait" in
  let h = B.create Rig.host B.paged_from in
  ignore
    (submit ~args:one (writes d) ~buffers:[| require_some (B.borrow d h) |]);
  P.gate p;
  let waited =
    Domain.spawn (fun () ->
        match B.wait h Rig.Read_write with
        | () -> "returned"
        | exception e -> Printexc.to_string e)
  in
  Support.await "the wait at the gate" (fun () -> P.sleepers p = 1);
  equal ~msg:"freed" bool true (B.free h);
  ignore (P.run p);
  for _ = 1 to 2 do
    ignore (Sys.opaque_identity (B.create Rig.host 0));
    Rig.free_cache d
  done;
  P.open_gate p;
  equal ~msg:"the wait" string "returned" (Domain.join waited);
  ignore (P.run p)

(* Two domains: a borrow held while it maps, and a free on the other. The borrow
   claims the memory while it maps it, so the free changes nothing then, and
   succeeds once the borrow returned. *)
let test_free_beside_borrow () =
  let d, p = P.open_ "buffer:free-borrow" in
  let h = B.create Rig.host B.paged_from in
  P.gate_locates p;
  let borrowed =
    Domain.spawn (fun () ->
        match B.borrow d h with
        | Some _ -> "borrowed"
        | None -> "not mapped"
        | exception e -> Printexc.to_string e)
  in
  Support.await "the borrow mapping" (fun () -> P.locaters p = 1);
  equal ~msg:"a free while the borrow maps" bool false (B.free h);
  P.open_locates p;
  equal ~msg:"the borrow" string "borrowed" (Domain.join borrowed);
  equal ~msg:"a free once it returned" bool true (B.free h)

(* unborrow *)

(* A borrow's end ends the borrow alone: its buffers and their views are dead
   with its reason, its memory lives, and its mapping goes back once the device
   ran the work submitted before. A borrow made again maps again. Only a borrow
   ends, whole. *)
let test_unborrow () =
  let d, p = P.open_ "buffer:unborrow" in
  let h = filled B.paged_from 'f' in
  let b = require_some (B.borrow d h) in
  let v = B.view b ~first:8 ~length:8 in
  raises_match ~msg:"a view of the borrow" Exn.invalid_arg (fun () ->
      B.unborrow v);
  raises_match ~msg:"memory of its own" Exn.invalid_arg (fun () -> B.unborrow h);
  equal ~msg:"ended" bool true (B.unborrow ~why:"done with it" b);
  equal ~msg:"the borrow and its view dead"
    (pair (option string) (option string))
    (Some "done with it", Some "done with it")
    (B.dead b, B.dead v);
  equal ~msg:"its memory live" (option string) None (B.dead h);
  ignore (B.create d 0);
  equal ~msg:"its mapping gone" (pair int int) (1, 0)
    (count "unmap" p, P.mapped p);
  let b' = require_some (B.borrow d h) in
  equal ~msg:"made again" (pair int int) (2, 1) (count "map_host" p, P.mapped p);
  equal ~msg:"the bytes through it" char 'f'
    (B.read b' Bigarray.char (fun a -> Bigarray.Array1.get a 0));
  raises_match ~msg:"an ended borrow" Exn.invalid_arg (fun () -> B.unborrow b)

(* A keep through a borrow keeps it from ending, as a prepared submission that
   fixes it does for its life, since the submission's addresses are made, until
   the keep is released, or collected and a call that moves collected memory
   ran; a keep of the memory through another buffer does not. *)
let test_unborrow_keeps () =
  let d, _ = P.open_ "buffer:unborrow-keeps" in
  let h = B.create Rig.host B.paged_from in
  let b = require_some (B.borrow d h) in
  let k = B.keep [ b ] in
  equal ~msg:"under a keep through it" bool false (B.unborrow b);
  B.release k;
  let s = Sub.make ~fixed:[ (b, Rig.Read) ] d [||] in
  equal ~msg:"under a prepared submission" bool false (B.unborrow b);
  ignore (Sys.opaque_identity s);
  let other = require_some (B.borrow d h) in
  let owner = B.keep [ h ] in
  equal ~msg:"under a keep of the memory itself" bool true (B.unborrow other);
  B.release owner;
  let released = require_some (B.borrow d h) in
  B.release (B.keep [ released ]);
  equal ~msg:"once a keep through it is released" bool true
    (B.unborrow released);
  let collected = require_some (B.borrow d h) in
  ignore (Sys.opaque_identity (B.keep [ collected ]));
  equal ~msg:"unreachable, before a call that moves memory" bool false
    (B.unborrow collected);
  collect d;
  equal ~msg:"once a keep through it is collected" bool true
    (B.unborrow collected)

(* A claim through a borrow keeps it from ending; claims of its memory through
   other buffers neither keep it nor see its end. *)
let test_unborrow_claims () =
  let d, _ = P.open_ "buffer:unborrow-claims" in
  let h = B.create Rig.host B.paged_from in
  let b = require_some (B.borrow d h) in
  Support.Reader.read b;
  equal ~msg:"under a claim through it" bool false (B.unborrow b);
  Support.Reader.release b;
  B.read h Bigarray.char (fun _ ->
      equal ~msg:"under a claim of the memory itself" bool true (B.unborrow b);
      B.write h Bigarray.char (fun a -> Bigarray.Array1.fill a 'c'));
  equal ~msg:"the memory's buffer" char 'c'
    (B.read h Bigarray.char (fun a -> Bigarray.Array1.get a 0))

(* A borrow of memory outside the claims, a bigarray's, ends as any borrow does,
   its bigarray living on. *)
let test_unborrow_outside () =
  let d, p = P.open_ "buffer:unborrow-outside" in
  let paged = B.export (B.create Rig.host B.paged_from) Bigarray.char in
  let h = B.of_bigarray paged in
  let b = require_some (B.borrow d h) in
  equal ~msg:"ended" bool true (B.unborrow b);
  ignore (B.create d 0);
  equal ~msg:"its mapping gone" int 0 (P.mapped p);
  equal ~msg:"the bigarray's buffer" (option string) None (B.dead h)

(* Borrows that hold no mapping, each a memory and the device that borrows it: a
   software device's borrow of another's memory, the host's of a software
   device's, and a software device's of host memory of no bytes. *)
let unmapped_borrows =
  [
    ( "a software device's borrow of another's memory",
      fun () ->
        (B.create (memory "buffer:unborrow-of") 64, memory "buffer:unborrow-on")
    );
    ( "the host's borrow of a software device's memory",
      fun () -> (B.create (memory "buffer:unborrow-host") 64, Rig.host) );
    ( "a software device's borrow of empty host memory",
      fun () -> (B.create Rig.host 0, memory "buffer:unborrow-empty") );
  ]

let c_answer = Testable.make ~pp:Support.Reader.pp_answer ~equal:( = )

(* A borrow that holds no mapping ends as one that holds one does: a claim or a
   keep through it keeps it until the claim ends, the keep is released, or the
   keep is collected and a call that moves collected memory ran, and a claim or
   a keep of its memory through the memory's own buffer does not. Once ended, it
   and its views are dead with its reason, as C reads them, and its memory
   lives. *)
let test_unborrow_unmapped (_, make) =
  let m, d = make () in
  let borrow () = require_some (B.borrow d m) in
  let b = borrow () in
  let n = B.length b in
  let v = B.view b ~first:(n / 2) ~length:(n - (n / 2)) in
  raises_match ~msg:"the memory's own buffer" Exn.invalid_arg (fun () ->
      B.unborrow m);
  (* A borrow of no bytes has no view of part. *)
  if n > 0 then
    raises_match ~msg:"a view of part of it" Exn.invalid_arg (fun () ->
        B.unborrow v);
  Support.Reader.read b;
  equal ~msg:"under a claim through it" bool false (B.unborrow b);
  Support.Reader.release b;
  let k = B.keep [ b ] in
  equal ~msg:"under a keep through it" bool false (B.unborrow b);
  B.release k;
  equal ~msg:"once the keep is released" bool true
    (B.unborrow ~why:"done with it" b);
  equal ~msg:"the borrow and its view dead"
    (pair (option string) (option string))
    (Some "done with it", Some "done with it")
    (B.dead b, B.dead v);
  equal ~msg:"its memory live" (option string) None (B.dead m);
  equal ~msg:"the reason C reads" (option string) (Some "done with it")
    (Support.Reader.why b);
  equal ~msg:"a claim from C" c_answer Support.Reader.Dead
    (Support.Reader.claim b Rig.Read);
  raises_match ~msg:"an ended borrow" Exn.invalid_arg (fun () -> B.unborrow b);
  let collected = borrow () in
  ignore (Sys.opaque_identity (B.keep [ collected ]));
  equal ~msg:"unreachable, before a call that moves memory" bool false
    (B.unborrow collected);
  collect d;
  equal ~msg:"once a keep through it is collected" bool true
    (B.unborrow collected);
  let other = borrow () in
  Support.Reader.read m;
  let owner = B.keep [ m ] in
  equal ~msg:"under a claim and a keep of its memory" bool true
    (B.unborrow other);
  B.release owner;
  Support.Reader.release m

(* Two domains: unborrows of one borrow with different reasons, beside reads of
   its reason. One unborrow ends the borrow and the others raise, and every
   reader then gives the reason of the one that ended it, as [dead] and as C
   reads it. *)
type lease = { mutable ended : string option }

let lease =
  abstract
    ~pp:(fun ppf r ->
      match r.ended with
      | None -> Format.pp_print_string ppf "live"
      | Some why -> Format.fprintf ppf "ended %S" why)
    "b"

(* The devices of every program's borrows, opened once. *)
let lenders =
  lazy (memory "buffer:unborrow-race-of", memory "buffer:unborrow-race-on")

let lend () =
  let owner, d = Lazy.force lenders in
  require_some (B.borrow d (B.create owner 64))

let live () = { ended = None }
let ended r = r.ended
let reason = Gen.of_list ~pp:Format.pp_print_string [ "a"; "b"; "c" ]
let unborrow why b = B.unborrow ~why b
let still_live r = equal ~msg:"ended" (option string) None r.ended
let already_ended r = not_equal ~msg:"ended" (option string) None r.ended

let judge_unborrow why r = function
  | Ok true ->
      still_live r;
      r.ended <- Some why
  | Ok false -> fail "refused, though nothing claims the borrow"
  | Error (Invalid_argument _) -> already_ended r
  | Error e -> raise e

let unborrow_commands =
  [
    command "borrow" (Gen.unit @-> makes lease) live lend;
    command "unborrow"
      (reason @-> lease ^-> judges bool)
      judge_unborrow unborrow;
    command "dead" (lease ^-> returns (option string)) ended B.dead;
    command "why from C"
      (lease ^-> returns (option string))
      ended Support.Reader.why;
  ]

(* Two domains: claims through a borrow, from OCaml and from C, beside its
   unborrows. A claim finds the borrow live before its end and dead after it,
   and [dead] then gives the end's reason. An unborrow may find a claim beside
   it and answer [false] while the borrow lives, which no order of whole calls
   shows: a claim spans its call. *)
let judge_unborrow_claimed why r = function
  | Ok false -> still_live r
  | outcome -> judge_unborrow why r outcome

let judge_read r = function
  | Ok () -> still_live r
  | Error (Invalid_argument _) -> already_ended r
  | Error e -> raise e

let c_claimed r =
  if r.ended = None then Support.Reader.Claimed else Support.Reader.Dead

let claim_from_c b =
  let a = Support.Reader.claim b Rig.Read in
  if a = Support.Reader.Claimed then Support.Reader.release b;
  a

let claim_commands =
  [
    command "borrow" (Gen.unit @-> makes lease) live lend;
    command "unborrow"
      (reason @-> lease ^-> judges bool)
      judge_unborrow_claimed unborrow;
    command "read"
      (lease ^-> judges unit)
      judge_read
      (fun b -> B.read b Bigarray.char ignore);
    command "claim from C" (lease ^-> returns c_answer) c_claimed claim_from_c;
    command "dead" (lease ^-> returns (option string)) ended B.dead;
  ]

(* Two domains: frees and takes of a borrow beside its unborrows. Whichever
   comes first kills the borrow with its reason, and the others raise; a take
   gives a live buffer. An unborrow may find a free or a take under way and
   answer [false] while the borrow lives, as beside a claim. *)
let judge_take why r = function
  | Ok (Some deaths) ->
      equal ~msg:"the taken buffer" (list (option string)) [ None ] deaths;
      judge_unborrow why r (Ok true)
  | Ok None -> judge_unborrow why r (Ok false)
  | Error e -> judge_unborrow why r (Error e)

let take why b = Option.map (List.map B.dead) (B.take ~why [ b ])

let end_commands =
  [
    command "borrow" (Gen.unit @-> makes lease) live lend;
    command "unborrow"
      (reason @-> lease ^-> judges bool)
      judge_unborrow_claimed unborrow;
    command "free"
      (reason @-> lease ^-> judges bool)
      judge_unborrow
      (fun why b -> B.free ~why b);
    command "take"
      (reason @-> lease ^-> judges (option (list (option string))))
      judge_take take;
    command "dead" (lease ^-> returns (option string)) ended B.dead;
  ]

(* Properties *)

(* Buffers each constructor makes, and the device it states. *)
let made_on =
  let polled () = fst (P.open_ "buffer:made-on") in
  [
    ( "create",
      fun () ->
        let d = polled () in
        (B.create d 64, d) );
    ("of_string", fun () -> (B.of_string "abc", Rig.host));
    ( "of_bigarray",
      fun () ->
        ( B.of_bigarray
            (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4),
          Rig.host ) );
    ( "view",
      fun () ->
        let d = polled () in
        (B.view (B.create d 64) ~first:8 ~length:8, d) );
    ( "borrow",
      fun () ->
        let d = polled () in
        (require_some (B.borrow d (B.create Rig.host B.paged_from)), d) );
    ( "take",
      fun () ->
        let d = polled () in
        (require_some (Support.donated (B.create d 64)), d) );
  ]

let test_device (_, make) =
  let b, d = make () in
  equal device d (B.device b)

(* The properties of [b], as one value. *)
let properties b v =
  ( (Rig.name (B.device b), B.length b, B.is_borrowed b, B.spans b),
    (B.overlaps b v, B.share_byte b v, B.access b, B.offset v) )

let properties_t =
  Testable.structural ~pp:(fun ppf ((d, n, borrowed, spans), (o, s, a, off)) ->
      Format.fprintf ppf
        "device %s, %d bytes, borrowed %b, spans %b, overlaps %b, shares a \
         byte %b, %s, view at %d"
        d n borrowed spans o s
        (match a with Rig.Read -> "Read" | Read_write -> "Read_write")
        off)

(* A dead buffer's properties answer as they did while it lived, and raise
   nothing: of a buffer a take killed, a view of it, and a borrow an unborrow
   ended. *)
let test_dead_properties () =
  let d, _ = P.open_ "buffer:dead-properties" in
  let b = B.create d 64 in
  let v = B.view b ~first:8 ~length:8 in
  let before = (properties b v, properties v b) in
  ignore (require_some (Support.donated b));
  equal ~msg:"a taken buffer and its view"
    (pair properties_t properties_t)
    before
    (properties b v, properties v b);
  let h = B.create Rig.host B.paged_from in
  let borrowed = require_some (B.borrow d h) in
  let before = properties borrowed h in
  equal ~msg:"unborrowed" bool true (B.unborrow borrowed);
  equal ~msg:"an ended borrow" properties_t before (properties borrowed h)

(* Whether each kind of buffer lies over memory something other than its device
   holds. *)
let borrowed_cases =
  [
    ("a host buffer", (fun () -> B.create Rig.host 40), false);
    ("a string's buffer", (fun () -> B.of_string "abc"), false);
    ( "a bigarray's buffer",
      (fun () ->
        B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4)),
      true );
    ( "an empty bigarray's buffer",
      (fun () ->
        B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 0)),
      true );
    ( "a view of a bigarray's buffer",
      (fun () ->
        B.view
          (B.of_bigarray
             (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4))
          ~first:1 ~length:2),
      true );
    ( "a device's borrow of host memory",
      (fun () ->
        require_some
          (B.borrow (memory "buffer:borrowed-on") (B.create Rig.host 32))),
      true );
    ( "a borrow on the device whose memory it is",
      (fun () ->
        let d, _ = P.open_ "buffer:borrowed-own" in
        require_some (B.borrow d (B.create d 64))),
      false );
  ]

let test_borrowed (_, make, want) = equal bool want (B.is_borrowed (make ()))

(* Which buffers span their memory: those the constructors make, a borrow of one
   and a view of all of one, and no view or borrow of part. *)
let spans_cases =
  let polled () = fst (P.open_ "buffer:spans") in
  let h () = B.create Rig.host (2 * B.paged_from) in
  [
    ("a host buffer", (fun () -> B.create Rig.host 40), true);
    ("a string's buffer", (fun () -> B.of_string "abc"), true);
    ( "a bigarray's buffer",
      (fun () ->
        B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4)),
      true );
    ( "a view of all of it",
      (fun () ->
        let b = B.create Rig.host 40 in
        B.view b ~first:0 ~length:40),
      true );
    ( "a view of part of it",
      (fun () -> B.view (B.create Rig.host 40) ~first:0 ~length:39),
      false );
    ( "a borrow of all of it",
      (fun () -> require_some (B.borrow (polled ()) (h ()))),
      true );
    ( "a borrow of part of it",
      (fun () ->
        require_some
          (B.borrow (polled ()) (B.view (h ()) ~first:0 ~length:B.paged_from))),
      false );
  ]

let test_spans (_, make, want) = equal bool want (B.spans (make ()))

(* A free changes no buffer's bytes or memory, so a buffer spans its memory once
   freed as it did before, the host's memory and a device's alike. *)
let test_spans_freed () =
  let d, _ = P.open_ "buffer:spans-freed" in
  List.iter
    (fun dev ->
      let msg = Rig.name dev in
      let b = B.create dev 64 in
      let part = B.view b ~first:0 ~length:32 in
      equal ~msg:(msg ^ ": freed") bool true (B.free b);
      equal
        ~msg:(msg ^ ": the buffer and a view of part span")
        (pair bool bool) (true, false)
        (B.spans b, B.spans part))
    [ Rig.host; d ]

(* Two views of one buffer of [n] bytes, or of two buffers, on the host or a
   software device: their ranges, the second often right after the first. *)
let pair_size = 24

type views = {
  on_host : bool;
  same : bool;
  first : int * int;
  second : int * int;
}

let views =
  let open Gen in
  let range =
    let* f = int_range 0 pair_size in
    let+ l = int_range 0 (pair_size - f) in
    (f, l)
  in
  let after (f, l) =
    let f' = f + l in
    let+ l' = int_range 0 (pair_size - f') in
    (f', l')
  in
  let* on_host = bool in
  let* first = range in
  let* kind = frequency [ (3, constant 0); (2, constant 1); (1, constant 2) ] in
  let+ second = if kind = 1 then after first else range in
  { on_host; same = kind <> 2; first; second }

let views =
  Gen.with_pp
    (fun ppf c ->
      let f, l = c.first and f', l' = c.second in
      Format.fprintf ppf "[%d,+%d] and [%d,+%d] of %s, on %s" f l f' l'
        (if c.same then "one buffer" else "two buffers")
        (if c.on_host then "the host" else "a software device"))
    views

let words_device = lazy (memory "buffer:words")

(* The two views of [c]. *)
let made_views c =
  let d = if c.on_host then Rig.host else Lazy.force words_device in
  let b = B.create d pair_size in
  let b' = if c.same then b else B.create d pair_size in
  let view b (first, length) = B.view b ~first ~length in
  (view b c.first, view b' c.second)

(* Whether the views of one allocation share a byte, and a word: an empty view
   shares neither. A buffer starts on a multiple of [Rig.word]. *)
let share_bytes c =
  let (f, l), (f', l') = (c.first, c.second) in
  c.same && l > 0 && l' > 0 && f < f' + l' && f' < f + l

let share_words c =
  let (f, l), (f', l') = (c.first, c.second) and w = Rig.word in
  c.same && l > 0 && l' > 0
  && f / w <= (f' + l' - 1) / w
  && f' / w <= (f + l - 1) / w

let cover_views c =
  cover "on the host" c.on_host;
  cover "on a software device" (not c.on_host);
  cover "sharing a byte" (share_bytes c);
  cover "sharing a word and no byte" (share_words c && not (share_bytes c));
  cover "of one buffer, sharing no word" (c.same && not (share_words c));
  cover "of two buffers" (not c.same)

(* [overlaps] is whether two buffers hold bytes of one word. *)
let overlaps_law c =
  cover_views c;
  let a, b = made_views c in
  equal ~msg:"overlaps" bool (share_words c) (B.overlaps a b)

(* [share_byte] is whether two buffers share a byte, which a copy between them
   refuses. *)
let share_byte_law c =
  cover_views c;
  let a, b = made_views c in
  equal ~msg:"share_byte" bool (share_bytes c) (B.share_byte a b);
  let copy =
    match B.submit_copy ~src:a ~dst:b [| B.row 0 |] with
    | pt ->
        Rig.Point.wait pt;
        "copied"
    | exception Invalid_argument _ -> "refused"
  in
  equal ~msg:"a copy of no bytes" string
    (if share_bytes c then "refused" else "copied")
    copy

(* The pairs of the table this law replaced, on both memories. *)
let views_examples =
  List.concat_map
    (fun on_host ->
      List.map
        (fun (first, second) -> { on_host; same = true; first; second })
        [
          ((0, 5), (5, 3));
          ((0, 4), (4, 4));
          ((1, 2), (3, 1));
          ((0, 3), (8, 8));
          ((6, 0), (4, 4));
          ((7, 2), (9, 7));
        ])
    [ true; false ]

(* A borrow overlaps the next bytes of its memory's word and not the next word,
   and bigarrays over one word overlap, though each is an allocation of its
   own. *)
let test_overlaps_across () =
  let d = memory "buffer:words-borrow" in
  let h = B.create Rig.host B.paged_from in
  let back = require_some (B.borrow d (B.view h ~first:0 ~length:5)) in
  equal ~msg:"a borrow and the next bytes of its word" bool true
    (B.overlaps back (B.view h ~first:5 ~length:3));
  equal ~msg:"a borrow and the next word" bool false
    (B.overlaps back (B.view h ~first:8 ~length:4));
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  let of_sub first length =
    B.of_bigarray (Bigarray.Array1.sub ba first length)
  in
  equal ~msg:"bigarrays over one word" bool true
    (B.overlaps (of_sub 0 5) (of_sub 5 3));
  equal ~msg:"a bigarray and one over its bytes" bool true
    (B.overlaps (B.of_bigarray ba) (of_sub 8 8));
  equal ~msg:"a bigarray and an empty view of it" bool false
    (B.overlaps (B.of_bigarray ba)
       (B.view (B.of_bigarray ba) ~first:4 ~length:0))

(* What the memory of each constructor admits. *)
let test_access () =
  let d, _ = P.open_ "buffer:access" in
  let access =
    Testable.structural ~pp:(fun ppf a ->
        Format.pp_print_string ppf
          (match a with Rig.Read -> "Read" | Read_write -> "Read_write"))
  in
  equal (list access)
    [ Read_write; Read_write; Read_write; Read_write ]
    (List.map B.access
       [
         B.create d 64;
         B.of_string "abc";
         B.of_bigarray
           (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4);
         require_some (Support.donated (B.create d 64));
       ])

(* Each call that reaches a dead buffer's bytes or claims its memory, on a host
   buffer and a device's that a take killed. *)
let dead_calls =
  [
    ("copy from", fun h _ other -> B.copy ~src:h ~dst:other);
    ("copy to", fun h _ other -> B.copy ~src:other ~dst:h);
    ( "submit_copy",
      fun h _ other ->
        ignore (B.submit_copy ~src:h ~dst:other [| B.row 8 |] : Rig.Point.t) );
    ("submit_fill", fun h _ _ -> ignore (B.submit_fill h 'f' : Rig.Point.t));
    ("aligned", fun h _ _ -> ignore (B.aligned h ~at:0 8));
    ("handle", fun _ m _ -> ignore (B.handle m));
    ("view", fun h _ _ -> ignore (B.view h ~first:0 ~length:8));
    ("wait", fun h _ _ -> B.wait h Rig.Read);
    ("reached", fun h _ _ -> ignore (B.reached h Rig.Read));
    ("read", fun h _ _ -> B.read h Bigarray.char ignore);
    ("write", fun h _ _ -> B.write h Bigarray.char ignore);
    ("export", fun h _ _ -> ignore (B.export h Bigarray.char));
    ("blit_from_string", fun h _ _ -> B.blit_from_string "s" 0 h 0 1);
    ("blit_to_bytes", fun h _ _ -> B.blit_to_bytes h 0 (Bytes.create 1) 0 1);
    ("fill", fun h _ _ -> B.fill h 'f');
    ("borrow", fun h m _ -> ignore (B.borrow (B.device m) h));
    ("free", fun h _ _ -> ignore (B.free h));
    ("take", fun h _ _ -> ignore (B.take ~why:"again" [ h ]));
    ("keep", fun h _ _ -> ignore (B.keep [ h ]));
    ("share", fun h _ _ -> B.share h);
    ( "dma",
      fun _ m _ ->
        ignore (B.dma m ~fn:"0000:41:00.0" : (Rig_edge.dma, string) result) );
  ]

(* Every call that reaches a dead buffer's bytes or claims its memory refuses it
   with the take's reason. *)
let test_dead_refused (_, call) =
  let d, _ = P.open_ "buffer:dead" in
  let h = B.create Rig.host B.paged_from and m = B.create d 64 in
  ignore (require_some (Support.donated h));
  ignore (require_some (Support.donated m));
  let other = B.create Rig.host B.paged_from in
  raises_match (Exn.invalid_arg ~sub:"donated") (fun () -> call h m other)

(* wait *)

let test_wait () =
  let d, p = P.open_ "buffer:wait" in
  let n = B.paged_from in
  let h = filled n 'a' and on = B.create d n in
  let part =
    let src = require_some (B.borrow d h) in
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer on))
  in
  ignore (submit (Sub.make d [| part |]));
  equal ~msg:"queued before" int 1 (P.queued p);
  B.wait on Rig.Read;
  equal ~msg:"queued after" int 0 (P.queued p)

(* Memory a lost device wrote raises Lost for a read and a wait. *)
let test_wait_lost () =
  let d, p = P.open_ "buffer:lost" in
  let on = B.create d 8 in
  P.fail p;
  raises_match ~msg:"the submit" lost (fun () ->
      submit ~args:one (writes d) ~buffers:[| on |]);
  raises_match ~msg:"a read" lost (fun () -> B.read on Bigarray.char ignore);
  raises_match ~msg:"a wait" lost (fun () -> B.wait on Rig.Read)

(* A host access that waits keeps its buffer until it returns: in another
   domain, [use] gets the only reference to memory of [d] that queued work
   writes, and sleeps on that work at the gate. The work then runs and the
   memory is collected and its device's cache freed, while the use sleeps:
   memory the access still walks the stamps of, or blits through, does not
   return. *)
let host_accesses =
  [
    ("a wait for every use", fun b -> B.wait b Rig.Read_write);
    ("a blit to bytes", fun b -> B.blit_to_bytes b 0 (Bytes.create 8) 0 8);
    ("a blit from a string", fun b -> B.blit_from_string "abcdefgh" 0 b 0 8);
  ]

let test_wait_keeps (name, use) =
  let d, p = P.open_ ("buffer:wait-keeps-" ^ name) in
  P.gate p;
  let user =
    Domain.spawn (fun () ->
        let b = B.create d 64 in
        ignore (submit ~args:one (writes d) ~buffers:[| b |]);
        use b)
  in
  Support.await "the use asleep" (fun () -> P.sleepers p = 1);
  ignore (P.run p);
  collect d;
  Rig.free_cache d;
  equal ~msg:"frees while the use sleeps" int 0 (count "free" p);
  P.open_gate p;
  Domain.join user

(* Once every point is reached, a wait allocates nothing. *)
let test_wait_words () =
  let d, _ = P.open_ "buffer:wait-words" in
  let b = B.create d 64 in
  ignore (submit ~args:one (writes d) ~buffers:[| b |]);
  B.wait b Rig.Read_write;
  equal ~msg:"bytes of 100 waits" int 0
    (Support.allocated_bytes (fun () ->
         for _ = 1 to 100 do
           B.wait b Rig.Read_write
         done))

(* Behind a transport, where the host reads a device's word through its driver,
   a wait for reached work allocates nothing: a buffer's wait, and a submit that
   waits for the point. *)
let transport_waits =
  [ ("a buffer's wait", `Buffer); ("a submit's wait", `Submit) ]

let test_wait_words_transport (_, which) =
  let d, _ = P.open_ ~transport:true "buffer:wait-words-transport" in
  let b = B.create d 64 in
  let p = submit ~args:one (writes d) ~buffers:[| b |] in
  B.wait b Rig.Read_write;
  let waiter = Sub.make (memory "buffer:wait-words-waiter") [||] in
  (* Passed as an option made once: a supplied optional argument may allocate
     its [Some] at each call. *)
  let waits = Some [| p |] in
  let call =
    match which with
    | `Buffer -> fun () -> B.wait b Rig.Read_write
    | `Submit ->
        fun () ->
          ignore
            (Rig.submit ?waits waiter ~buffers:[||] ~args:[||] : Rig.Point.t)
  in
  call ();
  equal ~msg:"bytes of 100 calls" int 0
    (Support.allocated_bytes (fun () ->
         for _ = 1 to 100 do
           call ()
         done))

(* reached *)

(* [reached b access] is whether [wait b access] would return at once: false
   while the work the access follows is queued, and true once it ran, with the
   work left to run. A read follows only the writes of the memory, so work that
   reads [b] holds back a write and no read. *)
let test_reached () =
  let d, p = P.open_ "buffer:reached" in
  let b = B.create d 64 in
  let reached () = (B.reached b Rig.Read, B.reached b Rig.Read_write) in
  equal ~msg:"no work" (pair bool bool) (true, true) (reached ());
  ignore (submit ~args:one (reads d) ~buffers:[| b |]);
  equal ~msg:"a read queued" (pair bool bool) (true, false) (reached ());
  ignore (P.run p);
  equal ~msg:"the read ran" (pair bool bool) (true, true) (reached ());
  ignore (submit ~args:one (writes d) ~buffers:[| b |]);
  equal ~msg:"a write queued" (pair bool bool) (false, false) (reached ());
  equal ~msg:"the work left queued" int 1 (P.queued p);
  ignore (P.run p);
  equal ~msg:"the write ran" (pair bool bool) (true, true) (reached ())

(* A loop on [reached] ends once the work runs: it commits the work, which a
   device that runs itself, and commits only every second value on its own, then
   runs. *)
let test_reached_commits () =
  let d, _ = P.open_ ~runs:`Itself ~lag:2 "buffer:reached-commits" in
  let b = B.create d 64 in
  ignore (submit ~args:one (writes d) ~buffers:[| b |]);
  while not (B.reached b Rig.Read_write) do
    Domain.cpu_relax ()
  done;
  equal ~msg:"a wait after it blocks for nothing" bool true
    (B.reached b Rig.Read)

(* Host memory that a lost device wrote through a borrow and never reached
   raises Lost for every access, as a wait does. Memory it only read raises for
   a write, which depends on every point, and is reached for a read. *)
let test_reached_lost () =
  let d, p = P.open_ "buffer:reached-lost" in
  let n = B.paged_from in
  let written = B.create Rig.host n and read = B.create Rig.host n in
  let w = require_some (B.borrow d written)
  and r = require_some (B.borrow d read) in
  let s = Sub.make d [| Support.touching [| Sub.write; Sub.read |] |] in
  ignore (submit ~args:one s ~buffers:[| w; r |]);
  P.fault p "the engine hung";
  raises_match ~msg:"the loss" lost (fun () -> B.create d 4096);
  raises_match ~msg:"a read of the written" lost (fun () ->
      B.reached written Rig.Read);
  raises_match ~msg:"a write of the written" lost (fun () ->
      B.reached written Rig.Read_write);
  raises_match ~msg:"the written's wait" lost (fun () ->
      B.wait written Rig.Read);
  equal ~msg:"a read of the read" bool true (B.reached read Rig.Read);
  raises_match ~msg:"a write of the read" lost (fun () ->
      B.reached read Rig.Read_write);
  raises_match ~msg:"a borrow on the lost device" lost (fun () ->
      B.reached w Rig.Read)

(* [reached] allocates nothing, whether the work is done or still queued once
   committed. *)
let test_reached_words () =
  let d, p = P.open_ "buffer:reached-words" in
  let b = B.create d 64 in
  let calls () =
    ignore (B.reached b Rig.Read_write);
    Support.allocated_bytes (fun () ->
        for _ = 1 to 100 do
          ignore (Sys.opaque_identity (B.reached b Rig.Read_write))
        done)
  in
  ignore (submit ~args:one (writes d) ~buffers:[| b |]);
  equal ~msg:"bytes of 100 calls, queued" int 0 (calls ());
  ignore (P.run p);
  equal ~msg:"bytes of 100 calls, done" int 0 (calls ())

(* read *)

(* A read follows the queued copy that writes the memory, with no wait by
   hand. *)
let test_read_follows () =
  let d, p = P.open_ "buffer:read-follows" in
  let n = B.paged_from in
  let h = filled n 'a' and src = B.create d n in
  B.write src Bigarray.char (fun a -> Bigarray.Array1.fill a 'x');
  let on_d = require_some (B.borrow d h) in
  ignore
    (submit (Sub.make d [| Sub.copy ~src:(Buffer src) ~dst:(Buffer on_d) |]));
  equal ~msg:"the write is queued" int 1 (P.queued p);
  equal ~msg:"a read sees it" string (String.make n 'x') (bytes h)

(* The bytes of each view that a read of an element kind refuses or reads, on a
   buffer of 32 bytes. *)
let element_views =
  [
    ("6 bytes as float32", 0, 6, `Refused);
    ("4 bytes from 2 as float32", 2, 4, `Refused);
    ("16 bytes from 4 as complex64", 4, 16, `Refused);
    ("16 bytes from 8 as complex64, one component's size", 8, 16, `Elements 1);
  ]

let test_read_elements (_, first, length, want) =
  let v = B.view (B.create Rig.host 32) ~first ~length in
  let read () =
    if length = 16 then B.read v Bigarray.complex64 Bigarray.Array1.dim
    else B.read v Bigarray.float32 Bigarray.Array1.dim
  in
  match want with
  | `Refused -> raises_match Exn.invalid_arg read
  | `Elements n -> equal ~msg:"elements" int n (read ())

(* The host reads and writes the memory of a device that it addresses as the
   memory itself, without a borrow: what a borrow on the host reads, byte for
   byte, and what it writes. *)
let test_host_addressed () =
  let d, _ = P.open_ "buffer:host-addressed" in
  let m = B.create d 64 in
  let on_host = require_some (B.borrow Rig.host m) in
  B.blit_from_string (String.make 64 'b') 0 m 0 64;
  equal ~msg:"a blit, through a borrow" string (String.make 64 'b')
    (bytes on_host);
  B.write m Bigarray.char (fun a -> Bigarray.Array1.fill a 'a');
  equal ~msg:"a bigarray, through a borrow" string (String.make 64 'a')
    (bytes on_host);
  B.write on_host Bigarray.char (fun a -> Bigarray.Array1.fill a 'h');
  equal ~msg:"a borrow's write, by a blit" string (String.make 64 'h')
    (contents m);
  equal ~msg:"a borrow's write, by a bigarray" string (String.make 64 'h')
    (bytes m)

(* Memory the host does not address: a device's that it reaches only by a
   mapping, an io device's, whose pages a borrow asks for, and another
   machine's, though its driver gives it a host address. *)
let unaddressed_memory name =
  let d, _ = P.open_ ~host_visible:false ("buffer:" ^ name) in
  let machine = Support.machine name in
  let far =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine
         {
           name = "buffer:" ^ name ^ "-far";
           driver = (module P);
           make = (fun ~word -> Ok (P.bind (P.make ()) ~word));
         })
  in
  [
    ("a device's", B.create d 8);
    ( "an io device's",
      Support.io_buffer (Support.io ("buffer:" ^ name ^ "-io")) 8 );
    ("another machine's", B.create far 8);
  ]

(* The host refuses to read or write in place memory it does not address. *)
let test_host_unaddressed () =
  List.iter
    (fun (whose, m) ->
      let refused what f =
        raises_match ~msg:(what ^ " " ^ whose) Exn.invalid_arg f
      in
      refused "a read of" (fun () -> B.read m Bigarray.char ignore);
      refused "a write of" (fun () -> B.write m Bigarray.char ignore))
    (unaddressed_memory "unaddressed")

(* The host reads and writes an empty buffer as an empty array, whatever its
   memory: there is nothing to address. *)
let test_host_empty () =
  let d, _ = P.open_ "buffer:host-empty" in
  let m = memory "buffer:host-empty-memory" in
  let empty =
    ("a software device's", B.create m 0)
    :: ("a driver's device's", B.create d 0)
    :: List.map
         (fun (whose, b) -> (whose, B.view b ~first:4 ~length:0))
         (unaddressed_memory "host-empty-unaddressed")
  in
  List.iter
    (fun (whose, b) ->
      let dim a = Bigarray.Array1.dim a in
      equal ~msg:("a read of " ^ whose) int 0 (B.read b Bigarray.float32 dim);
      equal ~msg:("a write of " ^ whose) int 0 (B.write b Bigarray.float32 dim))
    empty

(* write *)

(* A write follows the queued copy that reads the memory, with no wait by hand:
   the copy reads the bytes from before the write. *)
let test_write_follows () =
  let d, p = P.open_ "buffer:write-follows" in
  let n = B.paged_from in
  let h = filled n 'x' and dst = B.create d n in
  let on_d = require_some (B.borrow d h) in
  ignore
    (submit (Sub.make d [| Sub.copy ~src:(Buffer on_d) ~dst:(Buffer dst) |]));
  equal ~msg:"the read is queued" int 1 (P.queued p);
  B.write h Bigarray.char (fun a -> Bigarray.Array1.fill a 'w');
  equal ~msg:"the copy read the bytes before the write" string
    (String.make n 'x') (bytes dst);
  equal ~msg:"the host's write" string (String.make n 'w') (bytes h)

(* A close from the function of a write of Mapped memory, whose device's stop
   would wait for the write, raises Invalid_argument and changes nothing: the
   device stays open and the write flushes. *)
let test_mapped_close () =
  let d, p = P.open_ ~host_visible:false "buffer:mapped-close" in
  let m = B.create ~memory:Mapped d 64 in
  raises_match ~msg:"a close inside the write"
    (Exn.invalid_arg ~sub:"Rig.close") (fun () ->
      B.write m Bigarray.char (fun _ -> Rig.close d));
  is_none ~msg:"the device's loss" (Rig.lost d);
  equal ~msg:"flushes" int 1 (flushes p);
  Rig.close d;
  equal ~msg:"the loss of a close after the write" (option string)
    (Some "closed") (Rig.lost d)

(* A close from a callback that lands in the flush after the function of a write
   of Mapped memory, as a finaliser may, raises Invalid_argument whatever the
   device and changes nothing: a close of the written memory's device would wait
   for the flush. A close after the write ends each. *)
let test_flush_close () =
  let d, p = P.open_ ~host_visible:false "buffer:flush-close" in
  let other = memory "buffer:flush-close-other" in
  let m = B.create ~memory:Mapped d 64 in
  let close e =
    match Rig.close e with
    | () -> "returned"
    | exception Invalid_argument why
      when String.starts_with ~prefix:"Rig.close: " why ->
        "refused"
  in
  let answers = ref [] in
  P.before p "flush" (fun () -> answers := [ close other; close d ]);
  B.write m Bigarray.char (fun a -> a.{0} <- 'w');
  equal ~msg:"closes in the flush" (list string) [ "refused"; "refused" ]
    !answers;
  let losses () = [ Rig.lost other; Rig.lost d ] in
  equal ~msg:"the losses after the write"
    (list (option string))
    [ None; None ] (losses ());
  Rig.close other;
  Rig.close d;
  equal ~msg:"the losses of closes after the write"
    (list (option string))
    [ Some "closed"; Some "closed" ]
    (losses ())

(* Where the function of a write of Mapped memory and the flush after it both
   raise, the function's exception passes. *)
let test_both_raise () =
  let d, p = P.open_ ~host_visible:false "buffer:both-raise" in
  let m = B.create ~memory:Mapped d 64 in
  let flushed = ref "not called" in
  P.before p "flush" (fun () ->
      flushed := "called";
      failwith "the flush");
  raises ~msg:"the write" (Failure "the function") (fun () ->
      B.write m Bigarray.char (fun a ->
          a.{0} <- 'w';
          failwith "the function"));
  equal ~msg:"the flush" string "called" !flushed

(* A signal pending as the function of a write of Mapped memory returns or
   raises lands once the write's flush ran and its count and claim ended. [f]'s
   last act raises SIGINT, which the runtime records; its handler raises [Exit]
   at the next poll point, which the allocation after the call places if the
   write has none left. *)
exception Raised

let interrupted_window name f =
  if Sys.win32 then skip ~reason:"Windows has no signal masks" ();
  let d, p = P.open_ ~host_visible:false ("buffer:signal-" ^ name) in
  let m = B.create ~memory:Mapped d 8 in
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Exit))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () ->
      raises Exit (fun () ->
          (match B.write m Bigarray.char f with
          | () -> ()
          | exception Raised -> ());
          ignore (Sys.opaque_identity (ref ()))));
  equal ~msg:"counted calls" int 0 (Support.counted ());
  ignore (require_some ~msg:"taken" (Support.donated m));
  equal ~msg:"flushes" int 1 (flushes p)

let test_signal_returned () =
  interrupted_window "window-returned" (fun a ->
      a.{0} <- 'w';
      Support.interrupt ())

let test_signal_raised () =
  interrupted_window "window-raised" (fun a ->
      a.{0} <- 'w';
      Support.interrupt ();
      raise_notrace Raised)

(* A buffer on [d], whose memory the host addresses, filled with [c], seen
   through a borrow on the host. *)
let viewed d n c =
  let m = B.create d n in
  let view =
    B.write (require_some (B.borrow Rig.host m)) Bigarray.char Fun.id
  in
  Bigarray.Array1.fill view c;
  (Support.at m, view)

(* An array a write over a borrow on the host gave keeps the device memory it
   reads, as does an array made from it: no buffer reuses that memory while one
   is reachable. Collected, it returns to its device's cache. *)
let test_array_keeps_borrowed () =
  let d, _ = P.open_ "buffer:viewed" in
  let n = 4096 in
  let at, view = viewed d n 'v' in
  let view = ref (Some view) in
  let sub = ref (Some (Bigarray.Array1.sub (Option.get !view) 0 8)) in
  view := None;
  collect d;
  let other = B.create d n in
  B.write
    (require_some (B.borrow Rig.host other))
    Bigarray.char
    (fun a -> Bigarray.Array1.fill a 'w');
  not_equal ~msg:"another buffer's memory" int at (Support.at other);
  equal ~msg:"read through the array" char 'v' (Option.get !sub).{7};
  sub := None;
  collect d;
  equal ~msg:"once no array reads it" int at (Support.at (B.create d n));
  ignore (Sys.opaque_identity other)

(* rig holds a share of the storage of every array over a borrow: once
   collected, the arrays dropped while its memory lives leave that share and
   free nothing. *)
let test_array_shares () =
  let d, _ = P.open_ "buffer:shared" in
  let m = B.create d 4096 in
  let on_host = require_some (B.borrow Rig.host m) in
  let collected_shares () =
    Gc.full_major ();
    Gc.full_major ();
    let view = B.read on_host Bigarray.char Fun.id in
    Gc.full_major ();
    (view, Support.shares view)
  in
  let first () =
    let view, alone = collected_shares () in
    Bigarray.Array1.fill view 's';
    let sub = Bigarray.Array1.sub view 0 8 in
    [ alone; Support.shares sub ]
  in
  equal ~msg:"rig and a view, then its sub" (list int) [ 2; 3 ] (first ());
  let view, alone = collected_shares () in
  equal ~msg:"rig and a new view" int 2 alone;
  equal ~msg:"the bytes" char 's' view.{4095};
  ignore (Sys.opaque_identity m)

(* export *)

(* An export waits for the memory's work: the copy that writes it ran, and its
   bytes are the array's. *)
let test_export () =
  let d, p = P.open_ "buffer:export" in
  let n = B.paged_from in
  let h = filled n 'a' and src = B.create d n in
  B.write src Bigarray.char (fun a -> Bigarray.Array1.fill a 'x');
  let dst = require_some (B.borrow d h) in
  ignore
    (submit (Sub.make d [| Sub.copy ~src:(Buffer src) ~dst:(Buffer dst) |]));
  let a = B.export h Bigarray.char in
  equal ~msg:"work queued" int 0 (P.queued p);
  equal ~msg:"its last byte" char 'x' a.{n - 1}

(* An exported memory is outside the claims for good: no take succeeds. *)
let test_export_shares () =
  let h = B.create Rig.host 64 in
  ignore (B.export h Bigarray.char);
  is_none ~msg:"a take" ~pp:(Testable.pp buffer) (Support.donated h);
  is_none ~msg:"a take of a view" ~pp:(Testable.pp buffer)
    (Support.donated (B.view h ~first:0 ~length:64))

(* An export refuses a device's memory, even where the host addresses it. *)
let test_export_refused () =
  let d, _ = P.open_ "buffer:export-refused" in
  let m = B.create d 64 in
  raises_match ~msg:"a device's memory" (Exn.invalid_arg ~sub:"not the host's")
    (fun () -> ignore (B.export m Bigarray.char));
  raises_match ~msg:"its borrow on the host"
    (Exn.invalid_arg ~sub:"not the host's") (fun () ->
      ignore (B.export (require_some (B.borrow Rig.host m)) Bigarray.char))

(* An exported array keeps its memory once its buffer is collected: a buffer
   made after the collection does not take it. *)
let test_export_keeps () =
  List.iter
    (fun n ->
      let view =
        (fun () ->
          let ba = B.export (B.create Rig.host n) Bigarray.char in
          Bigarray.Array1.fill ba 'v';
          ba)
          ()
      in
      collect Rig.host;
      B.write (B.create Rig.host n) Bigarray.char (fun a ->
          Bigarray.Array1.fill a 'w');
      equal ~msg:(Printf.sprintf "%d bytes" n) char 'v' view.{n - 1})
    [ 16; 1 lsl 17 ]

(* aligned *)

(* An address [at] bytes past a view, inside it or outside it, of host memory or
   of memory named by handle alone, for [n = 2^k]. *)
type aligned_case = { handle_only : bool; first : int; at : int; k : int }

let aligned_case =
  let open Gen in
  let+ handle_only = bool
  and+ first = int_range 0 4096
  and+ at = int_range (-4096) 12288
  and+ k = frequency [ (3, int_range 0 12); (1, int_range 13 20) ] in
  { handle_only; first; at; k }

let aligned_case =
  Gen.with_pp
    (fun ppf c ->
      Format.fprintf ppf "%s, view from %d, byte %d, 2^%d"
        (if c.handle_only then "memory named by handle" else "host memory")
        c.first c.at c.k)
    aligned_case

let handle_only =
  lazy
    (fst (P.open_ ~host_visible:false ~addresses:false "buffer:aligned-handle"))

(* [aligned] is arithmetic on the address: host memory's own, and for memory
   named by handle alone its offset in an allocation that starts on its device's
   alignment, past which it raises. *)
let aligned_law c =
  let n = 1 lsl c.k in
  let v, address, limit =
    if c.handle_only then
      let d = Lazy.force handle_only in
      let v = B.view (B.create d (4096 + 8192)) ~first:c.first ~length:8192 in
      (v, B.offset v, Rig.alignment d)
    else
      let v =
        B.view (B.create Rig.host B.paged_from) ~first:c.first ~length:8192
      in
      (v, Support.at v, max_int)
  in
  let want = if n > limit then None else Some ((address + c.at) mod n = 0) in
  cover "host memory" (not c.handle_only);
  cover "memory named by handle" c.handle_only;
  cover "aligned" (want = Some true);
  cover "not aligned" (want = Some false);
  cover "past the alignment of memory named by handle" (want = None);
  cover "outside the view" (c.at < 0 || c.at >= 8192);
  let got =
    match B.aligned v ~at:c.at n with
    | a -> Some a
    | exception Invalid_argument _ -> None
  in
  equal ~msg:"aligned, or refused" (option bool) want got

let test_aligned_refused () =
  let b = B.create Rig.host 8 in
  raises_match ~msg:"0" Exn.invalid_arg (fun () -> B.aligned b ~at:0 0);
  raises_match ~msg:"3" Exn.invalid_arg (fun () -> B.aligned b ~at:0 3);
  raises_match ~msg:"-4" Exn.invalid_arg (fun () -> B.aligned b ~at:0 (-4));
  raises_match ~msg:"io memory" Exn.invalid_arg (fun () ->
      B.aligned (Support.io_buffer (Support.io "buffer:aligned-io") 8) ~at:0 16)

(* offset *)

(* A view starts its offset into its buffer's memory, at its address plus that
   offset. *)
let test_offset () =
  let d, _ = P.open_ "buffer:offset" in
  let b = B.create d 64 in
  let v = B.view b ~first:16 ~length:8 in
  equal ~msg:"its offset" int (B.offset b + 16) (B.offset v);
  let s = Sub.make ~fixed:[ (b, Rig.Read) ] d [||] in
  equal ~msg:"its address" int (Sub.address s b + 16) (Sub.address s v)

(* dma *)

let fn = "0000:41:00.0"

let pp_dma ppf = function
  | Rig_edge.Bus runs ->
      Format.fprintf ppf "Bus [%s]"
        (String.concat "; "
           (List.map (fun (a, n) -> Printf.sprintf "(0x%x, %d)" a n) runs))
  | Dmabuf { fd; offset } ->
      Format.fprintf ppf "Dmabuf { fd = %d; offset = %d }" fd offset

let dma_answer = result (Testable.make ~pp:pp_dma ~equal:( = )) string

(* A view's answer describes the allocation it lies in, as its driver does:
   Polled describes a region as one run of its own address and bytes. *)
let test_dma () =
  let d, p = P.open_ "buffer:dma" in
  let b = B.create d 4096 in
  equal ~msg:"the whole allocation" dma_answer
    (Ok (Rig_edge.Bus [ (Support.at b, 4096) ]))
    (B.dma (B.view b ~first:16 ~length:32) ~fn);
  equal ~msg:"asked of the driver" int 1 (count "dma" p)

let test_dma_refusal () =
  let d, p = P.open_ "buffer:dma-refusal" in
  let b = B.create d 64 in
  P.fail_at p 1 (`Refuse 1);
  equal ~msg:"the driver's refusal" dma_answer (Error "Polled: dma refused")
    (B.dma b ~fn)

(* The memory dma reached is outside the claims for good, also where the driver
   refused: no take succeeds. *)
let test_dma_shares () =
  let d, p = P.open_ "buffer:dma-shares" in
  let described = B.create d 64 and refused = B.create d 64 in
  ignore (B.dma described ~fn : (Rig_edge.dma, string) result);
  P.fail_at p 1 (`Refuse 1);
  ignore (B.dma refused ~fn : (Rig_edge.dma, string) result);
  is_none ~msg:"a take of described memory" ~pp:(Testable.pp buffer)
    (Support.donated described);
  is_none ~msg:"a take of refused memory" ~pp:(Testable.pp buffer)
    (Support.donated refused)

(* A borrow's answer is its memory's owner's: the owner's driver describes its
   own region, and the borrower's is not asked. *)
let test_dma_borrow () =
  let owner, p = P.open_ "buffer:dma-owner" in
  let borrower, q = P.open_ "buffer:dma-borrower" in
  let b = B.create owner 4096 in
  let borrowed = require_some (B.borrow borrower b) in
  equal ~msg:"the owner's allocation" dma_answer
    (Ok (Rig_edge.Bus [ (Support.at b, 4096) ]))
    (B.dma borrowed ~fn);
  equal ~msg:"asked of the owner and of the borrower" (pair int int) (1, 0)
    (count "dma" p, count "dma" q)

(* Collected memory a function reaches goes back to its driver, where memory no
   function reaches returns to the cache and is reused. *)
let test_dma_uncached () =
  let d, p = P.open_ "buffer:dma-uncached" in
  let n = 4096 in
  let frees () = List.map fst (P.frees p) in
  let first = ref (Some (B.create d n)) in
  let at = Support.at (Option.get !first) in
  first := None;
  collect d;
  satisfies ~msg:"cached, not freed" ~claim:"holds no free of the memory"
    (list int)
    (fun l -> not (List.mem at l))
    (frees ());
  let second = ref (Some (B.create d n)) in
  equal ~msg:"reused from the cache" int at (Support.at (Option.get !second));
  ignore (B.dma (Option.get !second) ~fn : (Rig_edge.dma, string) result);
  second := None;
  collect d;
  mem ~msg:"given back to the driver" int at (frees ())

let dma b () = ignore (B.dma b ~fn : (Rig_edge.dma, string) result)

let dma_refusals =
  [
    ("the host's memory", fun () -> B.create Rig.host 64);
    ("a software device's", fun () -> B.create (memory "buffer:dma-memory") 64);
    ( "an io device's",
      fun () -> Support.io_buffer (Support.io "buffer:dma-io") 64 );
    ("an empty buffer", fun () -> B.create (fst (P.open_ "buffer:dma-empty")) 0);
  ]

let test_dma_refused (_, make) = raises_match Exn.invalid_arg (dma (make ()))

let test_dma_lost () =
  let d, p = P.open_ "buffer:dma-lost" in
  let b = B.create d 64 in
  P.fault p "gone";
  raises_match lost (dma b)

(* handle *)

let test_handle_view () =
  let d, _ = P.open_ "buffer:handle-view" in
  let b = B.create d 64 in
  equal nativeint (B.handle b) (B.handle (B.view b ~first:16 ~length:8))

let test_handle_empty () =
  let d, _ = P.open_ "buffer:handle-empty" in
  equal nativeint 0n (B.handle (B.create d 0))

let handle_refusals =
  [
    ("the host's memory", fun () -> B.create Rig.host 8);
    ( "an io device's",
      fun () -> Support.io_buffer (Support.io "buffer:handle-io") 8 );
  ]

let test_handle_refused (_, make) =
  raises_match Exn.invalid_arg (fun () -> B.handle (make ()))

let () =
  exit
  @@ run "rig.buffer"
       [
         group "mappings"
           [
             test "a host buffer a device borrows is mapped once" test_maps_once;
             test
               "borrows of one allocation share one mapping until the last goes"
               test_borrows_share;
             test "a borrow collected and made again maps again" test_remade;
             test "a mapping is released once its memory died and its work ran"
               test_released;
             test
               "borrowed bytes return to their keeper once every mapping of \
                them is released"
               test_bytes_after_unmap;
             cases
               "empty memory borrows on every device of its machine, mapping \
                nothing"
               ~id:fst empty_borrows test_empty_borrow;
           ];
         group "t"
           [
             cases "compare is an order that terminates"
               ~id:(fun (n, _, _) -> n)
               distinct test_compare;
             cases
               "(=) refuses two buffers of one device, a buffer and itself \
                included, and finds two devices' unequal"
               ~id:(fun (n, _, _) -> n)
               distinct test_equality;
           ];
         group "kind"
           [
             test
               "a device whose memory the host addresses has no pinned memory"
               test_pinned_own;
             test
               "mapped memory where the host addresses a device's memory is \
                its own"
               test_mapped_own;
             cases
               "each of rig's writes of Mapped memory flushes once before it \
                returns, and later work reads it"
               ~id:(fun (n, _, _) -> n)
               window_writes test_window_write;
             test
               "a copy out of Mapped memory runs on its device, one in on the \
                host"
               test_window_copies;
             test
               "the host neither reads, borrows nor addresses Mapped memory of \
                a hidden device"
               test_window_unaddressed;
             test "mapped memory the window cannot hold raises once reclaimed"
               test_window_room;
           ];
         group "paged_from"
           [ test "is 64 KiB, or four of the host's pages" test_paged_from ];
         group "create"
           [
             prop "a host buffer of 128 bytes to paged_from starts on 128 bytes"
               line_case line_law;
             prop "a host buffer of paged_from bytes or more starts on a page"
               page_case page_law;
             test "an empty buffer allocates nothing" test_empty;
             test "a buffer's memory runs on to whole words, no other memory's"
               test_whole_words;
             test "a negative size raises" test_negative;
           ];
         group "of_string"
           [
             prop ~examples:[ "" ] "a string's buffer holds its bytes"
               Gen.string of_string_law;
           ];
         group "of_bigarray"
           [
             cases "a bigarray's buffer holds its bytes, without a copy"
               ~id:(fun (Kind (n, _, _)) -> n)
               kinds test_kind;
           ];
         group "view"
           [
             test "a view's bytes are its buffer's" test_view_bytes;
             prop ~examples:view_examples
               "a view is the bytes it names, or raises outside its buffer"
               view_case view_law;
           ];
         group "borrow"
           [
             test "a borrow lies over the memory it maps" test_borrow;
             test "a borrow on its own device is the buffer" test_borrow_own;
             test "host memory off a page does not borrow on a driver's device"
               test_off_page;
             test "a driver maps host memory only from the start of a page"
               test_mapped_from_page;
             test "a device that maps no host memory borrows none"
               test_maps_no_host;
             test "a device that maps no host memory reaches no software device"
               test_mapless_software;
             test
               "a device that maps no host memory maps a peer's visible memory"
               test_peer_no_host;
             test "a device borrows another device's memory of its driver"
               test_borrow_peer;
             test "a borrow of a borrow maps the memory under it"
               test_borrow_of_borrow;
             cases
               "a borrow back on its memory's device lies where that memory \
                does"
               ~id:fst owners test_borrow_back;
             test "a borrow of a lost device's buffer raises its loss"
               test_borrow_lost;
             test
               "a mapping refused room returns memory, the machine's collected \
                mappings included, and maps at the second ask"
               test_room;
             test "a mapping refused room twice raises Out_of_memory"
               test_no_room;
             test "a mapping the driver never makes is None, asked once"
               test_never_mapped;
             test "a room's return unmaps staging memory before it asks again"
               test_room_staging;
           ];
         group "free"
           [
             test "a freed buffer's memory returns once its work is done"
               test_free;
             test "a free given a reason kills with it" test_free_why;
             test "freed host memory returns once a device's work on it ran"
               test_free_host;
             test "a free where a take would be refused changes nothing"
               test_free_refused;
             test "a free of a dead buffer raises" test_free_dead;
             test "a free of a borrow returns its memory, also once taken"
               test_free_borrow;
             test "a wait beside a free walks the stamps it began on"
               test_free_beside_wait;
             test "a free beside a borrow that maps changes nothing"
               test_free_beside_borrow;
           ];
         group "unborrow"
           [
             test "unborrow ends the borrow, whose mapping goes" test_unborrow;
             test "a borrow of memory outside the claims ends"
               test_unborrow_outside;
             test "only a claim through a borrow keeps it from ending"
               test_unborrow_claims;
             test "a keep through a borrow keeps it from ending"
               test_unborrow_keeps;
             cases
               "a borrow that holds no mapping ends as one that holds one does"
               ~id:fst unmapped_borrows test_unborrow_unmapped;
             stateful ~domains:2
               "unborrows of one borrow on two domains end it once, and every \
                reader gives that one's reason"
               unborrow_commands;
             stateful ~domains:2
               "claims through a borrow beside its unborrow find it live, or \
                dead with the end's reason"
               claim_commands;
             stateful ~domains:2
               "frees and takes of a borrow beside its unborrow end it once, \
                with the first one's reason"
               end_commands;
           ];
         group "properties"
           [
             test
               "a dead buffer's properties answer as they did, raising nothing"
               test_dead_properties;
             cases "device is the device each constructor names" ~id:fst made_on
               test_device;
             cases
               "is_borrowed is whether something other than its device holds \
                its memory"
               ~id:(fun (n, _, _) -> n)
               borrowed_cases test_borrowed;
             cases "spans is whether a buffer's bytes are all of its memory"
               ~id:(fun (n, _, _) -> n)
               spans_cases test_spans;
             test "a freed buffer spans its memory as it did" test_spans_freed;
             prop ~count:300 ~examples:views_examples
               "overlaps is whether two buffers hold bytes of one word" views
               overlaps_law;
             test
               "overlaps holds across a borrow and its memory, and bigarrays \
                over one word"
               test_overlaps_across;
             prop ~count:300 ~examples:views_examples
               "share_byte is whether two buffers share a byte, which a copy \
                refuses"
               views share_byte_law;
             test "access is Read_write for what the constructors and take make"
               test_access;
           ];
         group "dead buffers"
           [
             cases
               "every call reaching a dead buffer's bytes refuses it with its \
                reason"
               ~id:fst dead_calls test_dead_refused;
           ];
         group "wait"
           [
             test "a wait returns once the work that wrote the memory ran"
               test_wait;
             test "memory a lost device wrote raises Lost" test_wait_lost;
             cases "a host access keeps its buffer through its wait" ~id:fst
               host_accesses test_wait_keeps;
             test "a wait whose points are reached allocates nothing"
               test_wait_words;
             cases
               "behind a transport a wait for reached work allocates nothing"
               ~id:fst transport_waits test_wait_words_transport;
           ];
         group "reached"
           [
             test "reached says whether a wait would return at once"
               test_reached;
             test "a loop on reached ends once the work runs"
               test_reached_commits;
             test "reached raises Lost where a wait does" test_reached_lost;
             test "reached allocates nothing" test_reached_words;
           ];
         group "read"
           [
             test "a read follows the memory's queued writes" test_read_follows;
             cases
               "a read refuses views that are no whole elements or start off \
                one"
               ~id:(fun (n, _, _, _) -> n)
               element_views test_read_elements;
             test "the host accesses the device memory it addresses directly"
               test_host_addressed;
             test
               "the host refuses in place access to memory it does not address"
               test_host_unaddressed;
             test "the host accesses an empty buffer as an empty array"
               test_host_empty;
           ];
         group "write"
           [
             test "a write follows the memory's queued uses" test_write_follows;
             test "a close inside a write of mapped memory raises"
               test_mapped_close;
             test "a close inside the flush of a write raises" test_flush_close;
             test
               "where the function and the flush both raise, the function's \
                exception passes"
               test_both_raise;
             test
               "a signal pending as a Mapped write's function returns lands \
                after its flush"
               test_signal_returned;
             test
               "a signal pending as a Mapped write's function raises lands \
                after its flush"
               test_signal_raised;
             test "an array over a borrow keeps the device memory it reads"
               test_array_keeps_borrowed;
             test "an array's storage keeps rig's share" test_array_shares;
           ];
         group "export"
           [
             test "an export waits for the memory's work" test_export;
             test "an exported memory is never taken" test_export_shares;
             test
               "an export refuses a device's memory, even one the host \
                addresses"
               test_export_refused;
             test
               "an exported array keeps its memory once its buffer is collected"
               test_export_keeps;
           ];
         group "aligned"
           [
             prop ~count:200
               "an address is aligned to a power of two, in or out of the \
                buffer"
               aligned_case aligned_law;
             test
               "aligned refuses what is no positive power of two, and io memory"
               test_aligned_refused;
           ];
         group "offset"
           [
             test "a view lies its offset into its buffer's memory" test_offset;
           ];
         group "dma"
           [
             test "dma describes the whole allocation as its driver does"
               test_dma;
             test "dma answers the driver's refusal" test_dma_refusal;
             test "memory dma reached is outside the claims, also where refused"
               test_dma_shares;
             test "a borrow's answer is its memory's owner's" test_dma_borrow;
             test
               "memory dma described goes back to its driver, never the cache"
               test_dma_uncached;
             cases "dma refuses memory no driver gives" ~id:fst dma_refusals
               test_dma_refused;
             test "dma of a lost device's memory raises Lost" test_dma_lost;
           ];
         group "handle"
           [
             test "a view's handle is its buffer's" test_handle_view;
             test "is 0n for an empty buffer of a driver's device"
               test_handle_empty;
             cases "handle refuses memory no driver object names" ~id:fst
               handle_refusals test_handle_refused;
           ];
       ]
