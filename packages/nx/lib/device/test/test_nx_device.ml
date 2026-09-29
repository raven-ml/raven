(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let host_bytes n : bytes_ba =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

let bytes_of_list l =
  let ba = host_bytes (List.length l) in
  List.iteri (fun i x -> ba.{i} <- x) l;
  ba

let bytes_of_buffer b : bytes_ba = B.bigarray Bigarray.int8_unsigned b

let list_of_bytes (ba : bytes_ba) =
  List.init (Bigarray.Array1.dim ba) (fun i -> ba.{i})

let read b =
  let ba = host_bytes (B.nbytes b) in
  B.copy ~src:b ~dst:(B.of_bigarray ba);
  list_of_bytes ba

let write b l = B.copy ~src:(B.of_bigarray (bytes_of_list l)) ~dst:b
let le64 v = List.init 8 (fun i -> (v lsr (8 * i)) land 0xff)
let bytes = list int

(* A device over host memory whose driver counts its calls. *)
type driver = {
  memory : (nativeint, bytes_ba) Hashtbl.t;
  mutable allocs : int;
  mutable frees : int;
  mutable refuse : bool;
}

let driver () =
  { memory = Hashtbl.create 16; allocs = 0; frees = 0; refuse = false }

let device ?(name = "TEST") ?(budget = max_int) ?load ?signal ?timeout_ms
    ?synchronized ?mapping drv =
  let alloc n =
    if drv.refuse then None
    else begin
      let ba = host_bytes n in
      let a = B.host_address (B.of_bigarray ba) in
      Hashtbl.add drv.memory a ba;
      drv.allocs <- drv.allocs + 1;
      Some { Nx_device.host = Some a; device = a; handle = a }
    end
  in
  let free (m : Nx_device.memory) =
    Hashtbl.remove drv.memory m.device;
    drv.frees <- drv.frees + 1
  in
  let signal = Option.map (fun s _ -> s) signal in
  Nx_device.make ~name ~arch:"test" ~budget ~memory:{ alloc; free } ?load
    ?signal ?timeout_ms ?synchronized ?mapping ()

(* A mapping of host memory, as a device that shares it gives, which counts its
   live mappings. *)
let map_host () =
  let live = ref 0 in
  let map a _ =
    incr live;
    Ok { Nx_device.host = Some a; device = a; handle = 1n }
  in
  ({ Nx_device.map; unmap = (fun _ -> decr live) }, live)

external store_signal : nativeint -> int -> unit = "test_nx_device_signal"

external memmove : nativeint -> nativeint -> int -> unit
  = "test_nx_device_memmove"

(* A device whose own memory the host does not address, as a GPU's. Its copy
   queue holds copies until the host waits for them, then runs them in order up
   to the value waited for, as a GPU runs behind the host, and counts them. Its
   host memory is the host's, and it maps any host memory, recording the ranges
   and counting the live mappings. It moves bytes into the memory of the devices
   of [peers]. [hang] stops its queue, and [refuse] makes its driver refuse to
   enqueue. *)
type far = {
  dev : Nx_device.t;
  copies : int ref;
  peers : Nx_device.t list ref;
  live : int ref;
  ranges : (nativeint * int) list ref;
  hang : bool ref;
  refuse : bool ref;
}

let far name =
  let own = driver () and copies = ref 0 and peers = ref [] in
  let live = ref 0 and ranges = ref [] in
  let hang = ref false and refuse = ref false in
  let queue = Queue.create () and signaled = ref 0 in
  let copy ~dst ~src n v =
    if !refuse then failwith "enqueue refused";
    Queue.push
      ( v,
        fun () ->
          memmove dst src n;
          incr copies )
      queue
  in
  let rec run_to v =
    if (not !hang) && !signaled < v && not (Queue.is_empty queue) then begin
      let v', f = Queue.pop queue in
      f ();
      signaled := v';
      run_to v
    end
  in
  let signal _ =
    {
      Nx_device.signaled = (fun () -> !signaled);
      wait =
        (fun v ~timeout_ms:_ ->
          run_to v;
          !signaled >= v);
    }
  in
  let map a n =
    incr live;
    ranges := (a, n) :: !ranges;
    Ok { Nx_device.host = Some a; device = a; handle = 1n }
  in
  let alloc ~host n =
    let ba = host_bytes n in
    let a = B.host_address (B.of_bigarray ba) in
    Hashtbl.add own.memory a ba;
    Some
      {
        Nx_device.host = (if host then Some a else None);
        device = a;
        handle = a;
      }
  in
  let free (m : Nx_device.memory) = Hashtbl.remove own.memory m.device in
  let dev =
    Nx_device.make ~name ~arch:"test" ~budget:max_int
      ~memory:{ alloc = alloc ~host:false; free }
      ~host_memory:{ alloc = alloc ~host:true; free }
      ~mapping:{ map; unmap = (fun _ -> decr live) }
      ~copy_queue:(fun _ ->
        {
          copy;
          transfer = (fun d -> if List.memq d !peers then Some copy else None);
        })
      ~signal ()
  in
  { dev; copies; peers; live; ranges; hang; refuse }

(* Runs [f] and collects what it allocated. *)
let dropped f =
  ignore (Sys.opaque_identity (f ()));
  Gc.full_major ()

let allocated d = Nx_device.Stats.allocated (Nx_device.stats d)
let cached d = Nx_device.Stats.cached (Nx_device.stats d)
let retained d = Nx_device.Stats.retained (Nx_device.stats d)

(* Signals [v] on [d]'s timeline after [delay] seconds from another domain, as
   the device would, and records that it did. *)
let signal_later d v delay =
  let signaled = Atomic.make false in
  let word = B.host_address (Nx_device.timeline d) in
  let domain =
    Domain.spawn (fun () ->
        Unix.sleepf delay;
        Atomic.set signaled true;
        store_signal word v)
  in
  (signaled, domain)

(* Host *)

let test_host_identity () =
  let h = Nx_device.host in
  equal ~msg:"name" string "CPU" (Nx_device.name h);
  is_true ~msg:"arch" (Nx_device.arch h <> "" && Nx_device.arch h <> "amd64");
  equal ~msg:"budget" int max_int (Nx_device.budget h);
  is_true ~msg:"equal" (Nx_device.equal h Nx_device.host);
  is_false ~msg:"distinct" (Nx_device.equal h (device (driver ())))

let test_create () =
  let b = B.create Nx_device.host S.Float32 3 in
  equal ~msg:"length" int 3 (B.length b);
  equal ~msg:"nbytes" int 12 (B.nbytes b);
  is_true ~msg:"dtype" (S.equal (B.dtype b) S.Float32);
  is_false ~msg:"owned" (B.is_borrowed b);
  equal ~msg:"no handle" nativeint 0n (B.handle b);
  equal ~msg:"int4 packs two per byte" int 3
    (B.nbytes (B.create Nx_device.host S.Int4 5));
  equal ~msg:"bool takes a byte" int 2
    (B.nbytes (B.create Nx_device.host S.Bool 2));
  let e = B.create Nx_device.host S.Float64 0 in
  equal ~msg:"empty" int 0 (B.nbytes e);
  raises_match (Exn.invalid_arg ~substring:"elements") (fun () ->
      B.create Nx_device.host S.Float32 (-1))

(* An element count whose bytes pass [max_int] is refused, not wrapped. *)
let test_overflow () =
  let overflow = Exn.invalid_arg ~substring:"overflow" in
  raises_match overflow (fun () ->
      B.create Nx_device.host S.Float64 ((1 lsl 58) + 1));
  raises_match overflow (fun () -> B.create Nx_device.host S.Float64 (1 lsl 59));
  let b = B.create Nx_device.host S.Float64 1 in
  raises_match overflow (fun () ->
      B.view b ~offset:0 S.Float64 ((1 lsl 58) + 1))

let test_copy_and_views () =
  let b = B.create Nx_device.host S.UInt8 8 in
  write b [ 0; 1; 2; 3; 4; 5; 6; 7 ];
  equal ~msg:"round trip" bytes [ 0; 1; 2; 3; 4; 5; 6; 7 ] (read b);
  let v = B.view b ~offset:4 S.UInt8 3 in
  equal ~msg:"view" bytes [ 4; 5; 6 ] (read v);
  write v [ 9; 9; 9 ];
  equal ~msg:"writes through a view" bytes [ 0; 1; 2; 3; 9; 9; 9; 7 ] (read b);
  let w = B.view b ~offset:4 S.UInt32 1 in
  equal ~msg:"format change" int 4 (B.nbytes w);
  equal ~msg:"view address" nativeint
    (Nativeint.add (B.host_address b) 4n)
    (B.host_address w);
  let inv = Exn.invalid_arg ~substring:"" in
  raises_match inv (fun () -> B.view b ~offset:(-1) S.UInt8 1);
  raises_match inv (fun () -> B.view b ~offset:6 S.UInt8 3);
  raises_match (Exn.invalid_arg ~substring:"aligned") (fun () ->
      B.view b ~offset:2 S.UInt32 1);
  raises_match (Exn.invalid_arg ~substring:"bytes into") (fun () ->
      B.copy ~src:b ~dst:v)

let test_of_bigarray () =
  let ba = bytes_of_list [ 1; 2; 3 ] in
  let before = Nx_device.stats Nx_device.host in
  let b = B.of_bigarray ba in
  is_true ~msg:"borrowed" (B.is_borrowed b);
  is_true ~msg:"on the host" (Nx_device.equal (B.device b) Nx_device.host);
  is_true ~msg:"bytes" (S.equal (B.dtype b) S.UInt8);
  ba.{1} <- 7;
  equal ~msg:"shares memory" bytes [ 1; 7; 3 ] (read b);
  let d = Nx_device.stats Nx_device.host in
  equal ~msg:"not counted" int 0
    (Nx_device.Stats.allocated (Nx_device.Stats.diff before d))

(* More bytes than a page on every platform. *)
let pages = 1 lsl 16

let test_of_bigarray_kinds () =
  let check (type a b) (k : (a, b) Bigarray.kind) s =
    let ba = Bigarray.Array1.create k Bigarray.c_layout 3 in
    let b = B.of_bigarray ba in
    is_true ~msg:(S.to_string s) (S.equal (B.dtype b) s);
    equal ~msg:"length" int 3 (B.length b);
    equal ~msg:"same memory" int 0
      (compare (B.host_address b)
         (B.host_address (B.of_bigarray (Bigarray.Array1.sub ba 0 3))))
  in
  check Bigarray.float16 S.Float16;
  check Bigarray.float32 S.Float32;
  check Bigarray.float64 S.Float64;
  check Bigarray.int8_signed S.Int8;
  check Bigarray.int8_unsigned S.UInt8;
  check Bigarray.char S.UInt8;
  check Bigarray.int16_signed S.Int16;
  check Bigarray.int16_unsigned S.UInt16;
  check Bigarray.int32 S.Int32;
  check Bigarray.int64 S.Int64;
  check Bigarray.complex32 S.Complex64;
  check Bigarray.complex64 S.Complex128;
  let f = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 2 in
  f.{1} <- 1.5;
  equal ~msg:"its bytes" bytes
    [ 0; 0; 0; 0; 0; 0; 0xc0; 0x3f ]
    (read (B.of_bigarray f));
  let inv = Exn.invalid_arg ~substring:"no storage format" in
  raises_match inv (fun () ->
      B.of_bigarray (Bigarray.Array1.create Bigarray.int Bigarray.c_layout 1));
  raises_match inv (fun () ->
      B.of_bigarray
        (Bigarray.Array1.create Bigarray.nativeint Bigarray.c_layout 1))

let test_file () =
  let file = { B.path = "/weights"; size = pages; mtime = 1.; inode = 2 } in
  let origin b =
    Option.map (fun ((f : B.file), off) -> (f.path, off)) (B.file b)
  in
  let origin_t = option (pair string int) in
  (* Page-aligned, as a mapping of a file is, so that a device can borrow it. *)
  let mapped =
    B.bigarray Bigarray.int8_unsigned (B.create Nx_device.host S.UInt8 pages)
  in
  let b = B.of_bigarray ~file mapped in
  equal ~msg:"the mapping" origin_t (Some ("/weights", 0)) (origin b);
  equal ~msg:"a view" origin_t
    (Some ("/weights", 4))
    (origin (B.view b ~offset:4 S.Float32 1));
  let m = device ~mapping:(fst (map_host ())) (driver ()) in
  equal ~msg:"a borrow of a view" origin_t
    (Some ("/weights", 2))
    (origin (B.borrow m (B.view b ~offset:2 S.UInt8 6)));
  equal ~msg:"other memory" origin_t None
    (origin (B.of_bigarray (host_bytes 8)));
  equal ~msg:"owned memory" origin_t None
    (origin (B.create Nx_device.host S.UInt8 8));
  raises_match (Exn.invalid_arg ~substring:"the mapping 7") (fun () ->
      B.of_bigarray ~file (host_bytes 7))

(* A bigarray whose elements do not lie at multiples of their size, as a file
   mapped from an unaligned position gives, is refused. *)
let test_of_bigarray_alignment () =
  let path = Filename.temp_file "nx_device_" ".bin" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      let fd = Unix.openfile path [ Unix.O_RDWR ] 0 in
      Fun.protect
        ~finally:(fun () -> Unix.close fd)
        (fun () ->
          ignore (Unix.write_substring fd (String.make 32 '\000') 0 32);
          let map pos k =
            Bigarray.array1_of_genarray
              (Unix.map_file fd ~pos k Bigarray.c_layout false [| 4 |])
          in
          let at_two = map 2L Bigarray.float32 in
          raises_match (Exn.invalid_arg ~substring:"aligned") (fun () ->
              B.of_bigarray at_two);
          raises_match (Exn.invalid_arg ~substring:"aligned") (fun () ->
              B.of_bigarray (map 4L Bigarray.complex64));
          is_true ~msg:"a complex at a component's multiple"
            (B.length (B.of_bigarray (map 8L Bigarray.complex64)) = 4);
          is_true ~msg:"bytes at any position"
            (B.length (B.of_bigarray (map 3L Bigarray.int8_unsigned)) = 4)))

external c_host : B.t -> nativeint = "test_nx_device_buffer_host"

let test_c_host () =
  let b = B.create Nx_device.host S.Float32 4 in
  let v = B.view b ~offset:8 S.UInt8 4 in
  equal ~msg:"a buffer" nativeint (B.host_address b) (c_host b);
  equal ~msg:"a view" nativeint (B.host_address v) (c_host v);
  let o = B.view (B.of_bigarray (host_bytes 16)) ~offset:3 S.UInt8 5 in
  equal ~msg:"a borrowed view" nativeint (B.host_address o) (c_host o)

let test_borrow () =
  let b = B.of_bigarray (bytes_of_list [ 1; 2 ]) in
  is_true ~msg:"host borrows itself" (B.borrow Nx_device.host b == b);
  let d = device (driver ()) in
  raises_match (Exn.invalid_arg ~substring:"cannot address") (fun () ->
      B.borrow d b);
  let on_d = B.create d S.UInt8 2 in
  raises_match (Exn.invalid_arg ~substring:"not CPU") (fun () ->
      B.borrow Nx_device.host on_d);
  let mapping, live = map_host () in
  let m = device ~mapping (driver ()) in
  let s0 = Nx_device.stats m in
  let hb = B.create Nx_device.host S.UInt8 pages in
  write (B.view hb ~offset:0 S.UInt8 2) [ 1; 2 ];
  let bm = B.borrow m hb in
  equal ~msg:"mapped" int 1 !live;
  is_true ~msg:"borrowed" (B.is_borrowed bm);
  is_true ~msg:"on the device" (Nx_device.equal (B.device bm) m);
  equal ~msg:"the same memory" nativeint (B.host_address hb) (B.host_address bm);
  equal ~msg:"shares it" bytes [ 1; 2 ] (read (B.view bm ~offset:0 S.UInt8 2));
  let e = B.borrow m (B.view hb ~offset:0 S.UInt8 0) in
  is_true ~msg:"a zero-byte borrow is borrowed" (B.is_borrowed e);
  equal ~msg:"not counted" int 0
    (Nx_device.Stats.allocated (Nx_device.Stats.diff s0 (Nx_device.stats m)));
  let off_page = B.bigarray Bigarray.int8_unsigned hb in
  raises_match (Exn.invalid_arg ~substring:"does not start on a page")
    (fun () -> B.borrow m (B.of_bigarray (Bigarray.Array1.sub off_page 1 2)))

(* The borrows of one host memory on a device share its one mapping, released
   once they are all collected. *)
let test_external_on_host () =
  raises_match (Exn.invalid_arg ~substring:"of_bigarray") (fun () ->
      Nx_device.external_buffer Nx_device.host
        { Nx_device.host = Some 0n; device = 0n; handle = 0n }
        S.UInt8 0)

let test_borrow_shared () =
  let mapping, live = map_host () in
  let m = device ~mapping (driver ()) in
  let hb = B.create Nx_device.host S.UInt8 (2 * pages) in
  (fun () ->
    let whole = B.borrow m hb in
    let tail = B.borrow m (B.view hb ~offset:pages S.UInt8 8) in
    equal ~msg:"one mapping" int 1 !live;
    equal ~msg:"a view at its offset" nativeint
      (Nativeint.add (B.address whole) (Nativeint.of_int pages))
      (B.address tail))
    ();
  Gc.full_major ();
  ignore (Nx_device.stats m);
  equal ~msg:"unmapped with its last borrow" int 0 !live;
  let again = B.borrow m hb in
  equal ~msg:"mapped again" int 1 !live;
  ignore (Sys.opaque_identity again)

(* The host memory under a borrow stays alive until the borrowing device's work
   is done, although the collector may run during that wait. *)
let test_borrow_lifetime () =
  let collected = ref false and during = ref true and armed = ref false in
  let signal =
    {
      Nx_device.signaled = (fun () -> max_int);
      wait =
        (fun _ ~timeout_ms:_ ->
          if !armed then begin
            armed := false;
            Gc.full_major ();
            Gc.full_major ();
            during := !collected
          end;
          true);
    }
  in
  let g = device ~mapping:(fst (map_host ())) ~signal (driver ()) in
  (fun () ->
    let hb = B.create Nx_device.host S.UInt8 (1 lsl 20) in
    Gc.finalise_last (fun () -> collected := true) hb;
    let bm = B.borrow g hb in
    ignore (Nx_device.submit g ~touches:[ Nx_device.host ] Fun.id);
    ignore (Sys.opaque_identity bm))
    ();
  Gc.full_major ();
  armed := true;
  ignore (Nx_device.stats g);
  is_false ~msg:"alive during the wait" !during;
  Gc.full_major ();
  Gc.full_major ();
  is_true ~msg:"collected after it" !collected

let test_bigarray_view () =
  let b = B.create Nx_device.host S.Float32 4 in
  let f = B.bigarray Bigarray.float32 b in
  equal ~msg:"length" int 4 (Bigarray.Array1.dim f);
  Bigarray.Array1.fill f 0.;
  f.{2} <- 1.5;
  let u = B.bigarray Bigarray.int8_unsigned b in
  equal ~msg:"one view per kind" int 16 (Bigarray.Array1.dim u);
  equal ~msg:"writes through" bytes (read b) (list_of_bytes u);
  write (B.view b ~offset:0 S.UInt8 4) [ 0; 0; 0x80; 0x3f ];
  equal ~msg:"sees buffer writes" float_exact 1. f.{0};
  let w =
    B.bigarray Bigarray.int16_unsigned (B.view b ~offset:8 S.BFloat16 2)
  in
  equal ~msg:"a view at an offset" int 0x3fc0 w.{1};
  let e = B.bigarray Bigarray.float64 (B.create Nx_device.host S.Float64 0) in
  equal ~msg:"empty" int 0 (Bigarray.Array1.dim e);
  let inv = Exn.invalid_arg ~substring:"" in
  raises_match inv (fun () ->
      B.bigarray Bigarray.int (B.create Nx_device.host S.Int64 1));
  raises_match inv (fun () ->
      B.bigarray Bigarray.float32 (B.create Nx_device.host S.UInt8 3));
  raises_match inv (fun () ->
      B.bigarray Bigarray.int16_signed
        (B.view (B.create Nx_device.host S.UInt8 4) ~offset:1 S.UInt8 2));
  raises_match (Exn.invalid_arg ~substring:"not CPU") (fun () ->
      B.bigarray Bigarray.char (B.create (device (driver ())) S.UInt8 1))

(* Copies of memory the host does not address run on a device's copy queue:
   directly between memory it addresses, through the host's staging memory from
   and to other host memory, and through the staging memory between devices that
   cannot reach each other. *)
let test_copy_queue () =
  let f = far "FAR" in
  let dev = B.create f.dev S.UInt8 5 in
  raises_match (Exn.invalid_arg ~substring:"does not address FAR memory")
    (fun () -> B.host_address dev);
  write dev [ 1; 2; 3; 4; 5 ];
  equal ~msg:"staged in and out" bytes [ 1; 2; 3; 4; 5 ] (read dev);
  equal ~msg:"one copy each way" int 2 !(f.copies);
  equal ~msg:"on the timeline" int 2 (Nx_device.submitted f.dev);
  let dev' = B.create f.dev S.UInt8 5 in
  B.copy ~src:dev ~dst:dev';
  equal ~msg:"device to device" int 3 !(f.copies);
  let pinned = B.create ~host:true f.dev S.UInt8 5 in
  B.copy ~src:dev' ~dst:pinned;
  equal ~msg:"into its host memory" bytes [ 1; 2; 3; 4; 5 ] (read pinned);
  let hb = B.create Nx_device.host S.UInt8 pages in
  let bm = B.borrow f.dev hb in
  let head = B.view hb ~offset:0 S.UInt8 5 in
  let c0 = !(f.copies) in
  B.copy ~src:dev ~dst:head;
  equal ~msg:"into mapped host memory, directly" int (c0 + 1) !(f.copies);
  equal ~msg:"mapped bytes" bytes [ 1; 2; 3; 4; 5 ] (read head);
  ignore (Sys.opaque_identity bm)

(* The host's staging memory is one, which every device maps once. *)
let test_staging_shared () =
  let a = far "SA" and b = far "SB" in
  let staged f =
    List.filter (fun (_, n) -> n = 128 lsl 20) !(f.ranges) |> List.map fst
  in
  write (B.create a.dev S.UInt8 3) [ 1; 2; 3 ];
  write (B.create a.dev S.UInt8 3) [ 1; 2; 3 ];
  write (B.create b.dev S.UInt8 3) [ 1; 2; 3 ];
  equal ~msg:"mapped once by each" (list int) [ 1; 1 ]
    [ List.length (staged a); List.length (staged b) ];
  equal ~msg:"the same memory" (list nativeint) (staged a) (staged b)

(* Host memory another device allocated is mapped for the copy alone, and copied
   directly. *)
let test_copy_other_host_memory () =
  let a = far "PA" and b = far "PB" in
  let pinned = B.create ~host:true a.dev S.UInt8 4 in
  write pinned [ 4; 3; 2; 1 ];
  let on_b = B.create b.dev S.UInt8 4 in
  let c0 = !(b.copies) and live = !(b.live) in
  B.copy ~src:pinned ~dst:on_b;
  equal ~msg:"one copy" int (c0 + 1) !(b.copies);
  equal ~msg:"unmapped after" int live !(b.live);
  equal ~msg:"bytes" bytes [ 4; 3; 2; 1 ] (read on_b)

let test_copy_between_queues () =
  let a = far "A" and b = far "B" and c = far "C" in
  a.peers := [ c.dev ];
  let src = B.create a.dev S.UInt8 4 in
  write src [ 9; 8; 7; 6 ];
  let on_b = B.create b.dev S.UInt8 4 and on_c = B.create c.dev S.UInt8 4 in
  let a0 = !(a.copies) and b0 = !(b.copies) and c0 = !(c.copies) in
  B.copy ~src ~dst:on_b;
  equal ~msg:"out of A" int (a0 + 1) !(a.copies);
  equal ~msg:"into B" int (b0 + 1) !(b.copies);
  equal ~msg:"through the staging memory" bytes [ 9; 8; 7; 6 ] (read on_b);
  B.copy ~src ~dst:on_c;
  equal ~msg:"moved by A" int (a0 + 2) !(a.copies);
  equal ~msg:"C copied nothing" int c0 !(c.copies);
  equal ~msg:"into C" bytes [ 9; 8; 7; 6 ] (read on_c)

(* Copies of several staging chunks through a queue that runs behind the host: a
   slot is refilled only after the copy that used it ran. *)
let test_pipeline () =
  let f = far "LAZY" and g = far "LAZY2" in
  let chunk = 64 lsl 20 in
  let n = (2 * chunk) + (chunk / 2) + 12345 in
  (* Differs between bytes one and two slots apart. *)
  let pattern i = (i * 7) + (i lsr 16) + (i / chunk * 13) in
  let src = B.create Nx_device.host S.UInt8 n in
  let s = bytes_of_buffer src in
  for i = 0 to n - 1 do
    s.{i} <- pattern i land 0xff
  done;
  let whole = B.create f.dev S.UInt8 (n + 1000) in
  let dev = B.view whole ~offset:1000 S.UInt8 n in
  B.copy ~src ~dst:dev;
  let back = B.create Nx_device.host S.UInt8 n in
  B.copy ~src:dev ~dst:back;
  let b = bytes_of_buffer back in
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if b.{i} <> s.{i} then incr bad
  done;
  equal ~msg:"round trip through an offset view" int 0 !bad;
  let on_g = B.create g.dev S.UInt8 n in
  B.copy ~src:dev ~dst:on_g;
  B.copy ~src:on_g ~dst:back;
  bad := 0;
  for i = 0 to n - 1 do
    if b.{i} <> s.{i} then incr bad
  done;
  equal ~msg:"bounced between devices" int 0 !bad;
  let tail = B.create f.dev S.UInt8 100 in
  B.copy ~src:(B.view dev ~offset:(2 * chunk) S.UInt8 100) ~dst:tail;
  equal ~msg:"device to device from an offset" bytes
    (List.init 100 (fun i -> pattern ((2 * chunk) + i) land 0xff))
    (read tail)

let test_copy_overlap () =
  let f = far "OVER" in
  let b = B.create f.dev S.UInt8 8 in
  raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
      B.copy
        ~src:(B.view b ~offset:0 S.UInt8 4)
        ~dst:(B.view b ~offset:2 S.UInt8 4));
  let h = B.create Nx_device.host S.UInt8 8 in
  raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
      B.copy ~src:h ~dst:h);
  write (B.view b ~offset:0 S.UInt8 4) [ 1; 2; 3; 4 ];
  B.copy ~src:(B.view b ~offset:0 S.UInt8 4) ~dst:(B.view b ~offset:4 S.UInt8 4);
  equal ~msg:"disjoint views copy" bytes [ 1; 2; 3; 4; 1; 2; 3; 4 ] (read b)

(* A driver error while enqueueing fails the device, like a fault. *)
let test_enqueue_error () =
  let f = far "REFUSING" in
  let b = B.create f.dev S.UInt8 4 in
  f.refuse := true;
  let refused = Exn.failure ~substring:"REFUSING: enqueue refused" in
  raises_match refused (fun () -> write b [ 1; 2; 3; 4 ]);
  raises_match refused (fun () -> B.create f.dev S.UInt8 1)

(* A transfer that cannot be waited for leaves its destination in the reach of
   the source, which failed: copies of it raise the source's error, and its
   memory is retained, never reused. *)
let test_hung_transfer () =
  let a = far "HUNG" and c = far "DEST" in
  a.peers := [ c.dev ];
  let src = B.create a.dev S.UInt8 4 in
  (fun () ->
    let dst = B.create c.dev S.UInt8 4 in
    a.hang := true;
    let hung = Exn.failure ~substring:"HUNG hang detected" in
    raises_match hung (fun () -> B.copy ~src ~dst);
    raises_match hung (fun () -> read dst);
    equal ~msg:"the destination device is healthy" int 3
      (List.length (read (B.create c.dev S.UInt8 3))))
    ();
  Gc.full_major ();
  equal ~msg:"retained" int 4 (retained c.dev);
  equal ~msg:"only the healthy buffer cached" int 3 (cached c.dev)

(* A view outlives the buffer it was taken from: the memory stays with it. *)
let test_bigarray_lifetime () =
  let v =
    let b = B.create Nx_device.host S.Int32 1000 in
    let v = B.bigarray Bigarray.int32 b in
    Bigarray.Array1.fill v 7l;
    v
  in
  Gc.full_major ();
  let reused =
    List.init 100 (fun _ ->
        let b = B.create Nx_device.host S.Int32 1000 in
        Bigarray.Array1.fill (B.bigarray Bigarray.int32 b) 0x55l;
        b)
  in
  Gc.full_major ();
  ignore (Sys.opaque_identity reused);
  is_true ~msg:"kept alive"
    (List.for_all
       (fun i -> v.{i} = 7l)
       (List.init (Bigarray.Array1.dim v) Fun.id));
  let tl = B.bigarray Bigarray.int64 (Nx_device.timeline Nx_device.host) in
  equal ~msg:"the timeline" int 2 (Bigarray.Array1.dim tl)

(* Two domains take views of the same fresh buffer at once, and one keeps its
   view: the view shares the buffer's storage, so it outlives the buffer
   whichever domain made the storage's proxy. *)
let test_concurrent_views () =
  let rounds = 2000 in
  let current = Atomic.make None
  and taken = Array.init 2 (fun _ -> Atomic.make 0) in
  let kept = ref [] in
  let worker k () =
    for r = 0 to rounds - 1 do
      let rec next () =
        match Atomic.get current with
        | Some b when Atomic.get taken.(k) = r -> b
        | _ ->
            Domain.cpu_relax ();
            next ()
      in
      let v = B.bigarray Bigarray.int32 (next ()) in
      if k = 0 then begin
        Bigarray.Array1.fill v 7l;
        kept := v :: !kept
      end;
      Atomic.incr taken.(k)
    done
  in
  let workers = List.init 2 (fun k -> Domain.spawn (worker k)) in
  for r = 0 to rounds - 1 do
    Atomic.set current (Some (B.create Nx_device.host S.Int32 16));
    while Atomic.get taken.(0) <= r || Atomic.get taken.(1) <= r do
      Domain.cpu_relax ()
    done
  done;
  Atomic.set current None;
  List.iter Domain.join workers;
  Gc.full_major ();
  Gc.full_major ();
  let filler =
    List.init 20000 (fun _ ->
        let a = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 16 in
        Bigarray.Array1.fill a 0x55l;
        a)
  in
  equal ~msg:"every kept view holds what was written" int 0
    (List.length (List.filter (fun v -> v.{5} <> 7l) !kept));
  ignore (Sys.opaque_identity filler)

(* Host memory is counted while its buffer lives, returned in the collection
   that finds it unreachable, and capped by the host's budget like any
   device's. *)
let test_host_memory () =
  let h = Nx_device.host in
  Gc.full_major ();
  let s0 = Nx_device.stats h in
  let b = B.create h S.UInt8 1000 in
  let grown = Nx_device.Stats.diff s0 (Nx_device.stats h) in
  equal ~msg:"counted" int 1000 (Nx_device.Stats.allocated grown);
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  let back = Nx_device.Stats.diff s0 (Nx_device.stats h) in
  equal ~msg:"returned when collected" int 0 (Nx_device.Stats.allocated back);
  let held = Nx_device.Stats.allocated (Nx_device.stats h) in
  Fun.protect ~finally:(fun () -> Nx_device.set_budget h max_int) @@ fun () ->
  Nx_device.set_budget h (held + 1000);
  let a = B.create h S.UInt8 600 in
  raises_match
    (function Nx_device.Out_of_memory (d, 600) -> d == h | _ -> false)
    (fun () -> B.create h S.UInt8 600);
  raises_match
    (function Nx_device.Out_of_memory (_, 2000) -> true | _ -> false)
    (fun () -> B.create h S.UInt8 2000);
  ignore (Sys.opaque_identity a);
  equal ~msg:"a collected buffer makes room" int 600
    (B.nbytes (B.create h S.UInt8 600))

(* Memory *)

let test_cache_reuse () =
  let drv = driver () in
  let d = device drv in
  dropped (fun () -> B.create d S.UInt8 1000);
  equal ~msg:"returned" int 0 (allocated d);
  equal ~msg:"cached" int 1000 (cached d);
  let b = B.create d S.UInt8 1000 in
  equal ~msg:"reused" int 1 drv.allocs;
  equal ~msg:"allocated" int 1000 (allocated d);
  equal ~msg:"taken from the cache" int 0 (cached d);
  ignore (Sys.opaque_identity b)

let test_budget () =
  let drv = driver () in
  let d = device ~budget:4096 drv in
  let a = B.create d S.UInt8 3000 in
  raises_match
    (function Nx_device.Out_of_memory (d', 3000) -> d' == d | _ -> false)
    (fun () -> B.create d S.UInt8 3000);
  ignore (Sys.opaque_identity a);
  let a = ref (Some (B.create d S.UInt8 3000)) in
  ignore (Sys.opaque_identity !a);
  equal ~msg:"the collected buffer's memory is reused" int 1 drv.allocs;
  equal ~msg:"taken from the cache" int 0 (cached d);
  (* An unreachable buffer is collected to make room. *)
  a := None;
  ignore (B.create d S.UInt8 2000);
  is_true ~msg:"collected, then freed to the system" (drv.frees >= 1);
  is_true ~msg:"within budget" (allocated d + cached d <= 4096)

let test_set_budget () =
  let drv = driver () in
  let d = device drv in
  dropped (fun () -> B.create d S.UInt8 2000);
  equal ~msg:"cached" int 2000 (cached d);
  Nx_device.set_budget d 1000;
  equal ~msg:"budget" int 1000 (Nx_device.budget d);
  equal ~msg:"cache released" int 0 (cached d);
  equal ~msg:"freed" int 1 drv.frees;
  raises_match (Exn.invalid_arg ~substring:"< 0") (fun () ->
      Nx_device.set_budget d (-1))

let test_free_cache () =
  let drv = driver () in
  let d = device drv in
  dropped (fun () -> (B.create d S.UInt8 100, B.create d S.UInt8 200));
  equal ~msg:"cached" int 300 (cached d);
  Nx_device.free_cache d;
  equal ~msg:"empty" int 0 (cached d);
  equal ~msg:"freed" int 2 drv.frees

(* A request larger than the budget raises without touching the cache. *)
let test_oversized () =
  let drv = driver () in
  let d = device ~budget:1000 drv in
  dropped (fun () -> B.create d S.UInt8 500);
  raises_match
    (function Nx_device.Out_of_memory (_, 2000) -> true | _ -> false)
    (fun () -> B.create d S.UInt8 2000);
  equal ~msg:"the cache is kept" int 500 (cached d);
  equal ~msg:"nothing freed" int 0 drv.frees

(* Memory a failed device's work may still use is retained: never freed and
   never reused. *)
let test_retained () =
  let signal =
    {
      Nx_device.signaled = (fun () -> 0);
      wait = (fun _ ~timeout_ms:_ -> false);
    }
  in
  let drv = driver () in
  let d = device ~name:"D" ~budget:1000 ~signal drv in
  dropped (fun () -> B.create d S.UInt8 600);
  ignore (Nx_device.submit d ~touches:[] Fun.id);
  raises_match (Exn.failure ~substring:"D hang detected") (fun () ->
      Nx_device.free_cache d);
  equal ~msg:"not freed" int 0 drv.frees;
  equal ~msg:"not reused" int 1 drv.allocs;
  equal ~msg:"stats report the retained bytes" int 600 (retained d);
  equal ~msg:"no longer cached" int 0 (cached d)

(* The allocator never holds more than the budget in live and cached memory,
   unless the budget fell below the live buffers, which are never released. *)
let test_budget_model () =
  let drv = driver () in
  let d = device ~budget:8192 drv in
  let live = ref [] in
  let check step =
    let s = Nx_device.stats d in
    let a = Nx_device.Stats.allocated s and c = Nx_device.Stats.cached s in
    if not (a + c <= Nx_device.budget d || c = 0) then
      failf "step %d: allocated %d + cached %d over budget %d" step a c
        (Nx_device.budget d)
  in
  for step = 1 to 600 do
    (match Random.int 10 with
    | 0 | 1 | 2 | 3 -> (
        let n = 1 + Random.int 3000 in
        match B.create d S.UInt8 n with
        | b -> live := b :: !live
        | exception Nx_device.Out_of_memory _ -> ())
    | 4 | 5 | 6 -> (
        match !live with
        | [] -> ()
        | l ->
            let k = Random.int (List.length l) in
            live := List.filteri (fun i _ -> i <> k) l)
    | 7 -> Nx_device.set_budget d (Random.int 20000)
    | 8 -> Nx_device.free_cache d
    | _ -> Gc.full_major ());
    check step;
    if step mod 100 = 0 then begin
      Gc.full_major ();
      equal ~msg:"allocated is the live buffers" int
        (List.fold_left (fun n b -> n + B.nbytes b) 0 !live)
        (allocated d)
    end
  done

let test_refused () =
  let drv = driver () in
  let d = device drv in
  dropped (fun () -> B.create d S.UInt8 100);
  drv.refuse <- true;
  raises_match
    (function Nx_device.Out_of_memory (_, 200) -> true | _ -> false)
    (fun () -> B.create d S.UInt8 200);
  equal ~msg:"cache released before failing" int 0 (cached d);
  equal ~msg:"freed" int 1 drv.frees

let test_transfer_stats () =
  let d = device (driver ()) in
  let h0 = Nx_device.stats Nx_device.host and d0 = Nx_device.stats d in
  let b = B.create d S.UInt8 4 in
  write b [ 1; 2; 3; 4 ];
  equal ~msg:"across" bytes [ 1; 2; 3; 4 ] (read b);
  B.copy ~src:b ~dst:(B.create d S.UInt8 4);
  let dh = Nx_device.Stats.diff h0 (Nx_device.stats Nx_device.host)
  and dd = Nx_device.Stats.diff d0 (Nx_device.stats d) in
  equal ~msg:"device in" int 4 (Nx_device.Stats.bytes_in dd);
  equal ~msg:"device out" int 4 (Nx_device.Stats.bytes_out dd);
  equal ~msg:"host out" int 4 (Nx_device.Stats.bytes_out dh);
  equal ~msg:"host in" int 4 (Nx_device.Stats.bytes_in dh);
  equal ~msg:"allocated" int 8 (Nx_device.Stats.allocated dd)

let test_domains () =
  let d = device (driver ()) in
  let work k () =
    for i = 1 to 200 do
      let n = 1 + (((i * 37) + k) mod 512) in
      let l = List.init n (fun j -> (i + j + k) land 0xff) in
      let b = B.create d S.UInt8 n in
      write b l;
      if read b <> l then failwith "corrupted copy"
    done;
    Gc.full_major ()
  in
  List.iter Domain.join (List.init 4 (fun k -> Domain.spawn (work k)));
  Gc.full_major ();
  equal ~msg:"all returned" int 0 (allocated d)

(* Copies between two devices in opposite directions take them in one order, so
   they cannot deadlock. *)
let test_opposite_copies () =
  let a = device (driver ()) and b = device (driver ()) in
  let ba = B.create a S.UInt8 64 and bb = B.create b S.UInt8 64 in
  let copies src dst () =
    for _ = 1 to 2000 do
      B.copy ~src ~dst
    done
  in
  let d1 = Domain.spawn (copies ba bb) and d2 = Domain.spawn (copies bb ba) in
  Domain.join d1;
  Domain.join d2;
  equal ~msg:"all counted" int (2000 * 64)
    (Nx_device.Stats.bytes_out (Nx_device.stats a))

(* Programs *)

let test_programs () =
  raises_match (Exn.invalid_arg ~substring:"loads no programs") (fun () ->
      Nx_device.Program.load Nx_device.host ~binary:"" ~name:"f");
  let loads = ref 0 in
  let load ~binary ~name =
    if binary = "bad" then failwith "rejected";
    incr loads;
    Nativeint.of_int (Hashtbl.hash (binary, name))
  in
  let d = device ~load (driver ()) in
  let p = Nx_device.Program.load d ~binary:"lib" ~name:"f" in
  let p' = Nx_device.Program.load d ~binary:"lib" ~name:"f" in
  is_true ~msg:"cached" (p == p');
  equal ~msg:"loaded once" int 1 !loads;
  equal ~msg:"name" string "f" (Nx_device.Program.name p);
  is_true ~msg:"device" (Nx_device.equal d (Nx_device.Program.device p));
  ignore (Nx_device.Program.load d ~binary:"lib" ~name:"g");
  equal ~msg:"per function" int 2 !loads;
  raises_match (Exn.failure ~substring:"rejected") (fun () ->
      Nx_device.Program.load d ~binary:"bad" ~name:"f")

(* Submitting work *)

(* A device with its own signal reports through [signaled] and leaves the
   timeline's signal word alone. *)
let test_timeline_signal_device () =
  let signal =
    { Nx_device.signaled = (fun () -> 5); wait = (fun _ ~timeout_ms:_ -> true) }
  in
  let d = device ~signal (driver ()) in
  ignore (Nx_device.submit d ~touches:[] Fun.id);
  Nx_device.synchronize d;
  equal ~msg:"signaled" int 5 (Nx_device.signaled d);
  equal ~msg:"the signal word stays 0" bytes
    (le64 0 @ le64 1)
    (read (Nx_device.timeline d))

let test_timeline () =
  let d = device (driver ()) in
  equal ~msg:"nothing submitted" int 0 (Nx_device.submitted d);
  equal ~msg:"nothing signaled" int 0 (Nx_device.signaled d);
  equal ~msg:"first value" int 1 (Nx_device.submit d ~touches:[] Fun.id);
  equal ~msg:"submitted" int 1 (Nx_device.submitted d);
  raises_match (Exn.failure ~substring:"encode") (fun () ->
      Nx_device.submit d ~touches:[] (fun _ -> failwith "encode"));
  equal ~msg:"a failed submission records nothing" int 1 (Nx_device.submitted d);
  equal ~msg:"timeline words" bytes
    (le64 0 @ le64 1)
    (read (Nx_device.timeline d));
  store_signal (B.host_address (Nx_device.timeline d)) 1;
  equal ~msg:"signaled" int 1 (Nx_device.signaled d)

let test_synchronize_waits () =
  let d = device (driver ()) in
  let signaled, domain =
    Nx_device.submit d ~touches:[] (fun v -> signal_later d v 0.05)
  in
  Nx_device.synchronize d;
  is_true ~msg:"waited for the signal" (Atomic.get signaled);
  equal ~msg:"signaled" int 1 (Nx_device.signaled d);
  Domain.join domain

let test_touched () =
  let d = device (driver ()) in
  let signaled, domain =
    Nx_device.submit d ~touches:[ Nx_device.host ] (fun v ->
        signal_later d v 0.05)
  in
  Nx_device.synchronize Nx_device.host;
  is_true ~msg:"the host waits for work that touched it" (Atomic.get signaled);
  Domain.join domain

let test_touched_devices () =
  let a = device ~name:"A" (driver ()) and b = device ~name:"B" (driver ()) in
  let signaled, domain =
    Nx_device.submit a ~touches:[ b ] (fun v -> signal_later a v 0.05)
  in
  Nx_device.synchronize b;
  is_true ~msg:"B waits for A's work" (Atomic.get signaled);
  Domain.join domain

let test_synchronized_hook () =
  let calls = ref 0 in
  let d = device ~synchronized:(fun () -> incr calls) (driver ()) in
  Nx_device.synchronize d;
  equal ~msg:"synchronize" int 1 !calls;
  write (B.create d S.UInt8 1) [ 0 ];
  equal ~msg:"a copy synchronizes" int 2 !calls

let test_copy_synchronizes () =
  let d = device (driver ()) in
  let b = B.create d S.UInt8 1 in
  let signaled, domain =
    Nx_device.submit d ~touches:[] (fun v -> signal_later d v 0.05)
  in
  write b [ 1 ];
  is_true ~msg:"copied after the work completed" (Atomic.get signaled);
  Domain.join domain

(* A device that never signals: the first wait times out and fails it, and from
   then on its own operations raise at once. *)
let test_hang () =
  let waits = ref 0 in
  let signal =
    {
      Nx_device.signaled = (fun () -> 0);
      wait =
        (fun _ ~timeout_ms:_ ->
          incr waits;
          false);
    }
  in
  let hung = device ~name:"HUNG" ~signal (driver ()) in
  ignore (Nx_device.submit hung ~touches:[] Fun.id);
  let hang = Exn.failure ~substring:"HUNG hang detected" in
  raises_match hang (fun () -> Nx_device.synchronize hung);
  raises_match hang (fun () -> Nx_device.synchronize hung);
  raises_match hang (fun () -> B.create hung S.UInt8 1);
  equal ~msg:"stats still answer" int 0
    (Nx_device.Stats.allocated (Nx_device.stats hung));
  raises_match hang (fun () -> Nx_device.submit hung ~touches:[] Fun.id);
  equal ~msg:"no second wait" int 1 !waits

(* A GPU hangs after work that touched the host through a borrow. The host stays
   healthy, collections included; only memory the GPU can reach raises its
   error. *)
let test_failure_scope () =
  let waits = ref 0 in
  let signal =
    {
      Nx_device.signaled = (fun () -> 0);
      wait =
        (fun _ ~timeout_ms:_ ->
          incr waits;
          false);
    }
  in
  let gpu =
    device ~name:"GPU" ~mapping:(fst (map_host ())) ~signal (driver ())
  in
  let shared =
    B.view (B.create Nx_device.host S.UInt8 pages) ~offset:0 S.UInt8 8
  in
  let mapped = B.borrow gpu shared in
  ignore (Nx_device.submit gpu ~touches:[ Nx_device.host ] Fun.id);
  let failed = Exn.failure ~substring:"GPU hang detected" in
  raises_match failed (fun () -> Nx_device.synchronize gpu);
  Nx_device.synchronize Nx_device.host;
  let a = B.create Nx_device.host S.UInt8 8 in
  write a [ 1; 2; 3; 4; 5; 6; 7; 8 ];
  equal ~msg:"host copies" bytes [ 1; 2; 3; 4; 5; 6; 7; 8 ] (read a);
  dropped (fun () -> B.create Nx_device.host S.UInt8 1000);
  equal ~msg:"host creates after a collection" int 8
    (B.nbytes (B.create Nx_device.host S.UInt8 8));
  Nx_device.synchronize Nx_device.host;
  raises_match failed (fun () -> B.copy ~src:shared ~dst:a);
  raises_match failed (fun () -> B.copy ~src:a ~dst:shared);
  raises_match failed (fun () ->
      B.copy
        ~src:(B.view shared ~offset:4 S.UInt8 4)
        ~dst:(B.view a ~offset:0 S.UInt8 4));
  raises_match failed (fun () -> B.bigarray Bigarray.char shared);
  equal ~msg:"a view touches no memory" int 4
    (B.length (B.view shared ~offset:0 S.UInt8 4));
  raises_match failed (fun () -> read mapped);
  equal ~msg:"no second wait" int 1 !waits

(* A borrow that was collected and unmapped no longer reaches the host memory:
   the device failing later leaves that memory usable. *)
let test_unmapped_before_failure () =
  let hung = ref false in
  let signal =
    {
      Nx_device.signaled = (fun () -> 0);
      wait = (fun _ ~timeout_ms:_ -> not !hung);
    }
  in
  let gpu =
    device ~name:"LATE" ~mapping:(fst (map_host ())) ~signal (driver ())
  in
  let shared =
    B.view (B.create Nx_device.host S.UInt8 pages) ~offset:0 S.UInt8 4
  in
  dropped (fun () -> B.borrow gpu shared);
  ignore (Nx_device.stats gpu);
  hung := true;
  ignore (Nx_device.submit gpu ~touches:[] Fun.id);
  raises_match (Exn.failure ~substring:"LATE hang detected") (fun () ->
      Nx_device.synchronize gpu);
  write shared [ 1; 2; 3; 4 ];
  equal ~msg:"still usable" bytes [ 1; 2; 3; 4 ] (read shared)

(* The wait bound is the device's, settable at any time. *)
let test_set_timeout () =
  let seen = ref 0 in
  let signal =
    {
      Nx_device.signaled = (fun () -> 0);
      wait =
        (fun _ ~timeout_ms ->
          seen := timeout_ms;
          true);
    }
  in
  let d = device ~signal (driver ()) in
  equal ~msg:"default" int 30_000 (Nx_device.timeout d);
  Nx_device.set_timeout d 5;
  Nx_device.submit d ~touches:[] ignore;
  Nx_device.synchronize d;
  equal ~msg:"waits use it" int 5 !seen;
  raises_match (Exn.invalid_arg ~substring:"0 ms") (fun () ->
      Nx_device.set_timeout d 0)

(* A borrow is collected, and its device hangs in the wait before it unmaps it:
   the host memory stays in the failed device's reach. *)
let test_cut_short () =
  let hung = ref false in
  let signal =
    {
      Nx_device.signaled = (fun () -> 0);
      wait = (fun _ ~timeout_ms:_ -> not !hung);
    }
  in
  let gpu =
    device ~name:"CUT" ~mapping:(fst (map_host ())) ~signal (driver ())
  in
  let shared = B.create Nx_device.host S.UInt8 pages in
  (fun () ->
    let b = B.borrow gpu shared in
    Nx_device.submit gpu ~touches:[ Nx_device.host ] (fun _ ->
        ignore (Sys.opaque_identity b)))
    ();
  Gc.full_major ();
  hung := true;
  let failed = Exn.failure ~substring:"CUT hang detected" in
  raises_match failed (fun () -> Nx_device.synchronize gpu);
  raises_match failed (fun () ->
      B.copy ~src:shared ~dst:(B.create Nx_device.host S.UInt8 pages))

(* A fault the driver reports fails the device with the driver's message. *)
let test_fault () =
  let signal =
    {
      Nx_device.signaled = (fun () -> 0);
      wait = (fun _ ~timeout_ms:_ -> failwith "page fault");
    }
  in
  let d = device ~name:"FAULTY" ~signal (driver ()) in
  ignore (Nx_device.submit d ~touches:[] Fun.id);
  let fault = Exn.failure ~substring:"FAULTY: page fault" in
  raises_match fault (fun () -> Nx_device.synchronize d);
  raises_match fault (fun () -> B.create d S.UInt8 1)

(* Without its own signal, a wait restarts its timeout whenever the signal word
   moves: slow progress is not a hang, and no progress is. *)
let test_timeout_restarts () =
  let d = device ~name:"SLOW" ~timeout_ms:200 (driver ()) in
  let word = B.host_address (Nx_device.timeline d) in
  ignore (Nx_device.submit d ~touches:[] Fun.id);
  ignore (Nx_device.submit d ~touches:[] Fun.id);
  let v = Nx_device.submit d ~touches:[] Fun.id in
  let progress =
    Domain.spawn (fun () ->
        for k = 1 to v do
          Unix.sleepf 0.12;
          store_signal word k
        done)
  in
  let t0 = Unix.gettimeofday () in
  Nx_device.synchronize d;
  is_true ~msg:"waited past one timeout" (Unix.gettimeofday () -. t0 > 0.3);
  Domain.join progress;
  let stuck = device ~name:"STUCK" ~timeout_ms:200 (driver ()) in
  ignore (Nx_device.submit stuck ~touches:[] Fun.id);
  raises_match (Exn.failure ~substring:"STUCK hang detected") (fun () ->
      Nx_device.synchronize stuck);
  store_signal (B.host_address (Nx_device.timeline stuck)) 1

let () =
  exit
    (run "nx.device"
       [
         group "host"
           [
             test "identity" test_host_identity;
             test "create" test_create;
             test "element counts do not overflow" test_overflow;
             test "copies and views" test_copy_and_views;
             test "of_bigarray" test_of_bigarray;
             test "host memory is counted and capped" test_host_memory;
             test "of_bigarray of every kind" test_of_bigarray_kinds;
             test "mapped files" test_file;
             test "of_bigarray refuses unaligned elements"
               test_of_bigarray_alignment;
             test "nx_device.h reads the host address" test_c_host;
             test "borrow" test_borrow;
             test "borrows share one mapping" test_borrow_shared;
             test "no external host memory" test_external_on_host;
             test "a borrow keeps its memory through the wait"
               test_borrow_lifetime;
             test "bigarray views" test_bigarray_view;
             test "a bigarray view keeps its memory" test_bigarray_lifetime;
             test "concurrent views share the storage" test_concurrent_views;
           ];
         group "memory"
           [
             test "the cache reuses memory" test_cache_reuse;
             test "the budget caps memory" test_budget;
             test "set_budget releases the cache" test_set_budget;
             test "free_cache empties the cache" test_free_cache;
             test "an oversized request raises at once" test_oversized;
             test "memory of a hung device is retained" test_retained;
             test "the budget holds under random use" test_budget_model;
             test "a refused allocation raises" test_refused;
             test "copies count across devices" test_transfer_stats;
             test "a copy queue" test_copy_queue;
             test "one staging memory" test_staging_shared;
             test "host memory of another device" test_copy_other_host_memory;
             test "copies between copy queues" test_copy_between_queues;
             test "a queue that runs behind the host" test_pipeline;
             test "overlapping copies are refused" test_copy_overlap;
             test "an enqueue error fails the device" test_enqueue_error;
             test "a hung transfer" test_hung_transfer;
             test "domains share a device" test_domains;
             test "opposite copies do not deadlock" test_opposite_copies;
           ];
         group "programs" [ test "load" test_programs ];
         group "submission"
           [
             test "the timeline" test_timeline;
             test "a signal device's timeline" test_timeline_signal_device;
             test "synchronize waits for the signal" test_synchronize_waits;
             test "work that touched a device" test_touched;
             test "work that touched another device" test_touched_devices;
             test "the synchronize hook" test_synchronized_hook;
             test "copies synchronize" test_copy_synchronizes;
             test "a hung device fails" test_hang;
             test "a failure is scoped to the memory it reaches"
               test_failure_scope;
             test "an unmapped borrow is out of reach"
               test_unmapped_before_failure;
             test "a borrow cut short by a failure" test_cut_short;
             test "set_timeout" test_set_timeout;
             test "a faulting device fails" test_fault;
             test "the timeout restarts on progress" test_timeout_restarts;
           ];
       ])
