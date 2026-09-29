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
    ?synchronized ?borrow drv =
  let alloc n =
    if drv.refuse then None
    else begin
      let ba = host_bytes n in
      let a = B.host_address (B.of_bigarray ba) in
      Hashtbl.add drv.memory a ba;
      drv.allocs <- drv.allocs + 1;
      Some { Nx_device.host = a; device = a; handle = a }
    end
  in
  let free (m : Nx_device.memory) =
    Hashtbl.remove drv.memory m.host;
    drv.frees <- drv.frees + 1
  in
  Nx_device.make ~name ~arch:"test" ~budget ~alloc ~free ?load ?signal
    ?timeout_ms ?synchronized ?borrow ()

(* A mapping of host memory, as a device that shares it gives. *)
let map_host a _ = Some { Nx_device.host = a; device = a; handle = 1n }

(* Runs [f] and collects what it allocated. *)
let dropped f =
  ignore (Sys.opaque_identity (f ()));
  Gc.full_major ()

let allocated d = Nx_device.Stats.allocated (Nx_device.stats d)
let cached d = Nx_device.Stats.cached (Nx_device.stats d)
let retained d = Nx_device.Stats.retained (Nx_device.stats d)

external store_signal : nativeint -> int -> unit = "test_nx_device_signal"

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

let test_borrow () =
  let b = B.of_bigarray (bytes_of_list [ 1; 2 ]) in
  is_true ~msg:"host borrows itself" (B.borrow Nx_device.host b == b);
  let d = device (driver ()) in
  raises_match (Exn.invalid_arg ~substring:"cannot address") (fun () ->
      B.borrow d b);
  let on_d = B.create d S.UInt8 2 in
  raises_match (Exn.invalid_arg ~substring:"not CPU") (fun () ->
      B.borrow Nx_device.host on_d);
  let m = device ~borrow:map_host (driver ()) in
  let s0 = Nx_device.stats m in
  let bm = B.borrow m b in
  is_true ~msg:"borrowed" (B.is_borrowed bm);
  is_true ~msg:"on the device" (Nx_device.equal (B.device bm) m);
  equal ~msg:"the same memory" nativeint (B.host_address b) (B.host_address bm);
  equal ~msg:"shares it" bytes [ 1; 2 ] (read bm);
  let e = B.borrow m (B.view b ~offset:0 S.UInt8 0) in
  is_true ~msg:"a zero-byte borrow is borrowed" (B.is_borrowed e);
  equal ~msg:"not counted" int 0
    (Nx_device.Stats.allocated (Nx_device.Stats.diff s0 (Nx_device.stats m)))

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
  let g = device ~borrow:map_host ~signal (driver ()) in
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
  let gpu = device ~name:"GPU" ~borrow:map_host ~signal (driver ()) in
  let shared = B.create Nx_device.host S.UInt8 8 in
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
  let gpu = device ~name:"LATE" ~borrow:map_host ~signal (driver ()) in
  let shared = B.create Nx_device.host S.UInt8 4 in
  dropped (fun () -> B.borrow gpu shared);
  ignore (Nx_device.stats gpu);
  hung := true;
  ignore (Nx_device.submit gpu ~touches:[] Fun.id);
  raises_match (Exn.failure ~substring:"LATE hang detected") (fun () ->
      Nx_device.synchronize gpu);
  write shared [ 1; 2; 3; 4 ];
  equal ~msg:"still usable" bytes [ 1; 2; 3; 4 ] (read shared)

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
             test "borrow" test_borrow;
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
             test "a faulting device fails" test_fault;
             test "the timeout restarts on progress" test_timeout_restarts;
           ];
       ])
