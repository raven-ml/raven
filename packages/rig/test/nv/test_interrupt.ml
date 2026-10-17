(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of a GPU through NVIDIA's kernel driver, driven by the
   driver alone. A call on an image of forty functions, each with local
   memory of its own size, is made once for each allocation it makes, with
   every allocation from that one on raising until the call returns, as an
   asynchronous exception does. The image's one unload then frees every
   launch the image made, the interrupted call's included, so the C heap
   holds no more than its allocator's rounding.

   A call is also made once for each allocation it makes with another call
   from a Gc.Memprof callback at that allocation, on the calling thread.
   Inside a lock of the driver's that the callback's call takes, an image's,
   the device's local memory lock or the path's lock on its ranges, the call
   raises Invalid_argument, having changed nothing; between locks it does
   its work. Through rig, a call from a callback inside any lock raises
   Invalid_argument as it starts, and the device is kept. *)

open Windtrap
module N = Rig_nv
module P = Rig_nv_nvidia

let strf = Printf.sprintf
let timeout = 60.
let functions = List.init 40 (fun k -> strf "local%d" k)
let made = List.filteri (fun k _ -> k < 32) functions
let last = List.nth functions 32

(* The C heap's bytes, once the collector freed what it holds there. *)
let heap () =
  Gc.full_major ();
  Gc.full_major ();
  Option.get (Rig_support.heap_bytes ())

(* The bytes the C heap may hold after a walk: its allocator's rounding. *)
let slack = 256

(* The entries of [fs] on the image [i]. *)
let entries i fs =
  List.iter
    (fun f ->
      match N.entry i f with
      | Entry _ -> ()
      | Absent -> failwith (f ^ " is absent")
      | No_memory n -> failwith (strf "no memory for %s's %d bytes" f n))
    fs

(* [loaded g bin f] is [f i] for a new image [i] of [bin] on [g], which it
   unloads and frees after. *)
let loaded g bin f =
  let i, r, _ = Rig_nv_support.image g bin in
  Fun.protect
    ~finally:(fun () ->
      N.unload g i;
      N.free g r)
    (fun () -> f i)

(* [opened f] is [f g bin] for a device [g] of the first GPU and the fixture
   [bin], once the local memory serves each of [bin]'s functions. *)
let opened f =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  let bin = Rig_nv_support.fixture "locals_sm89.cubin" in
  let g = Result.get_ok (Rig_gpu_support.opened (P.gpu 0)) in
  Fun.protect ~finally:(fun () -> N.stop g ~fault:None) @@ fun () ->
  loaded g bin (fun i -> entries i functions);
  f g bin

(* [walk attempt] checks [attempt n], the C heap's growth over an attempt
   interrupted from the [n]-th allocation and whether it was reached, from an
   uninterrupted attempt, which makes what the walk keeps for good, such as
   the memory profile's own, then from the first allocation on. *)
let walk attempt =
  ignore (attempt max_int);
  let rec from n =
    let kept, reached = attempt n in
    at_most
      ~msg:(strf "C heap kept, interrupted from allocation %d" n)
      int ~than:slack kept;
    if reached then from (n + 1)
  in
  from 1

(* An image's 33rd entry, where a table of 32 entries would grow. *)
let test_entries () =
  opened @@ fun g bin ->
  walk @@ fun n ->
  let before = heap () in
  let reached =
    loaded g bin (fun i ->
        entries i made;
        Rig_support.interrupted n (fun () -> entries i [ last ]))
  in
  (heap () - before, reached)

(* The unload of an image with an entry of each function. *)
let test_unload () =
  opened @@ fun g bin ->
  walk @@ fun n ->
  let before = heap () in
  let i, r, _ = Rig_nv_support.image g bin in
  entries i functions;
  let reached = Rig_support.interrupted n (fun () -> N.unload g i) in
  N.free g r;
  (heap () - before, reached)

(* [walk_calls attempt] runs [attempt n] from [n = 1] until an attempt no
   longer reaches its [n]-th allocation. *)
let walk_calls attempt =
  let rec from n = if attempt n then from (n + 1) in
  from 1

(* The start of the message of a take of a driver's lock inside one of its
   rank or a later one, such as by its holder. *)
let retaken = "Rig_nv: a lock taken inside one of its rank or a later one"

(* [reentry attempt] walks [during ()] with [call ()] from a Gc.Memprof
   callback at each of its allocations, on the calling thread, where
   [attempt ()] is [(during, call, after)], and runs [after ()] once
   [during] returned. Inside a lock [call] raises [Invalid_argument] for the
   lock it would take again, having changed nothing, and does its work
   outside every lock. *)
let reentry attempt =
  let seen = Rig_support.seen () in
  walk_calls (fun n ->
      let during, call, after = attempt () in
      let landing, landed = Rig_support.landing seen retaken call in
      let reached = Rig_support.interrupted ~landing n during in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      after ();
      reached);
  Rig_support.covered seen

(* A function asked for from a callback while another function of its image
   is: inside the image's lock the entry raises, and the function is made
   when asked for again; outside it, it is made. *)
let test_entry_in_entry () =
  opened @@ fun g bin ->
  reentry @@ fun () ->
  let i, r, _ = Rig_nv_support.image g bin in
  let during () = entries i [ List.nth functions 0 ]
  and call () = entries i [ List.nth functions 1 ]
  and after () =
    entries i functions;
    N.unload g i;
    N.free g r
  in
  (during, call, after)

(* The bytes of a thread's local memory in a growth: the least that grows a
   new device's, which keeps the growth's GPU memory small. *)
let local_bytes = 32

(* A stop from a callback in the first growth of a new device's local
   memory: inside the device's local memory lock the stop raises, and the
   growth completes; outside it, the device stops, after which a growth
   answers [Error]. The growth is the capability's, the one an entry makes;
   an entry's other allocations hold no local memory lock. *)
let test_stop_in_local () =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  let seen = Rig_support.seen () in
  walk_calls (fun n ->
      let g = Result.get_ok (Rig_gpu_support.opened (P.gpu 0)) in
      let local = (N.capability g).local in
      let stopped = ref false and grown = ref (Ok ()) in
      let stop () =
        N.stop g ~fault:None;
        stopped := true
      in
      let landing, landed = Rig_support.landing seen retaken stop in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            grown := local local_bytes)
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      if !stopped then
        is_error
          ~msg:(strf "from allocation %d, a growth after the stop" n)
          (local (2 * local_bytes))
      else begin
        equal (result unit string)
          ~msg:(strf "from allocation %d, the growth" n)
          (Ok ()) !grown;
        equal (result unit string)
          ~msg:(strf "from allocation %d, a growth after it" n)
          (Ok ())
          (local (2 * local_bytes));
        N.stop g ~fault:None
      end;
      reached);
  Rig_support.covered seen

(* The bytes of each region the memory walks make. *)
let bytes = 4096

let alloc g =
  match N.alloc g Rig_edge.Device bytes with
  | Some r -> r
  | None -> fail "no GPU memory"

(* A free from a callback in another free: inside the path's lock on the
   users of its memory the free raises and leaves the region for a later
   free; outside it, the region goes back. *)
let test_free_in_free () =
  opened @@ fun g _ ->
  reentry @@ fun () ->
  let r = alloc g and r' = alloc g in
  let freed = ref false in
  let during () = N.free g r
  and call () =
    N.free g r';
    freed := true
  and after () = if not !freed then N.free g r' in
  (during, call, after)

(* An allocation from a callback in another allocation: the path takes
   the GPU's addresses with no lock, so both are made, apart. *)
let test_alloc_in_alloc () =
  opened @@ fun g _ ->
  walk_calls @@ fun n ->
  let made = ref [] in
  let keep () =
    let r = alloc g in
    made := r :: !made
  in
  let reached = Rig_support.interrupted ~landing:(Call keep) n keep in
  equal ~msg:(strf "from allocation %d, the regions made" n) int
    (if reached then 2 else 1)
    (List.length !made);
  (match List.map Rig_nv_support.address !made with
  | [ a; b ] ->
      at_least
        ~msg:(strf "from allocation %d, the distance of %#x and %#x" n a b)
        int ~than:bytes
        (abs (a - b))
  | _ -> ());
  List.iter (N.free g) !made;
  reached

(* Through rig *)

(* A function of an image asked for through rig from a callback while
   another function of it is asked for: inside a lock, rig's or the
   driver's, the call raises as it starts; outside every lock, it answers.
   The function is made when asked for again, and the device is not lost. *)
let test_entry_in_entry_rig () =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  let t = Rig_nv_support.open_ () in
  Fun.protect ~finally:(fun () -> Rig_nv_support.close t) @@ fun () ->
  let d = t.d in
  let seen = Rig_support.seen () in
  walk_calls (fun n ->
      let i =
        match
          Rig.Image.load d (Rig_nv_support.fixture "locals_sm89.cubin")
        with
        | Ok i -> i
        | Error why -> fail why
      in
      let call () = ignore (Rig.Image.mem i (List.nth functions 1)) in
      let landing, landed = Rig_support.landing seen "Rig.Image.mem: " call in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore (Rig.Image.mem i (List.nth functions 0)))
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      is_none ~msg:(strf "from allocation %d, the loss" n) (Rig.lost d);
      equal ~msg:(strf "from allocation %d, the function" n) bool true
        (Rig.Image.mem i (List.nth functions 1));
      reached);
  Rig_support.covered seen

(* Inside a lock *)

(* The capability's growth of local memory, from inside a lock: it raises
   naming the call, and grows once the lock is given back. *)
let test_local_inside () =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  let g = Result.get_ok (Rig_gpu_support.opened (P.gpu 0)) in
  Fun.protect ~finally:(fun () -> N.stop g ~fault:None) @@ fun () ->
  let local = (N.capability g).local in
  raises_match
    (Exn.invalid_arg ~substring:"Rig_nv_abi.Gpu.local: called from")
    (fun () -> Rig_support.inside_lock (fun () -> ignore (local local_bytes)));
  equal (result unit string) ~msg:"the growth after" (Ok ())
    (local local_bytes)

let () =
  exit
    (run "rig_nv interrupt"
       [
         group ~timeout "an exception at any allocation"
           [
             test
               "of an image's entry leaves its unload to free every launch"
               test_entries;
             test "of an image's unload leaves no launch held" test_unload;
           ];
         group ~timeout
           "a call from a callback at any allocation raises inside the lock \
            it takes, and answers outside it"
           [
             test "an entry in an entry of its image" test_entry_in_entry;
             test "a stop in a growth of local memory" test_stop_in_local;
             test "a free in a free" test_free_in_free;
             test "an allocation in an allocation answers" test_alloc_in_alloc;
           ];
         group ~timeout
           "a call to rig from a callback at any allocation raises inside a \
            lock, answers outside it, and keeps the device"
           [ test "a function asked for while another is" test_entry_in_entry_rig ];
         group ~timeout "a call inside a lock of its thread's raises"
           [ test "the capability's growth of local memory" test_local_inside ];
       ])
