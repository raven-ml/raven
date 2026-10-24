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

let strf = Printf.sprintf
let timeout = 60.
let functions = List.init 40 (fun k -> strf "local%d" k)
let made = List.filteri (fun k _ -> k < 32) functions
let last = List.nth functions 32

(* What a failed walk left GPU 0 held by, which every later test needs. *)
let left_held : string option Atomic.t = Atomic.make None

(* Skips without an NVIDIA GPU either path opens, and fails at once where an
   earlier walk left the GPU held. *)
let needs_gpu () =
  if not (Rig_nv_support.present ()) then
    skip ~reason:"the machine has no NVIDIA GPU either path opens" ();
  Option.iter
    (fun why -> fail ("an earlier walk left GPU 0 held: " ^ why))
    (Atomic.get left_held)

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
  needs_gpu ();
  let bin = Rig_nv_support.fixture "locals_sm89.cubin" in
  let g = Result.get_ok (Rig_gpu_support.opened (Rig_nv_support.opener 0)) in
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

(* An image's 33rd entry, where a table of 32 entries would grow. The image
   and the device stay whole: the entry is made when asked for again. *)
let test_entries () =
  opened @@ fun g bin ->
  walk @@ fun n ->
  let before = heap () in
  let reached =
    loaded g bin (fun i ->
        entries i made;
        let reached =
          Rig_support.interrupted n (fun () -> entries i [ last ])
        in
        entries i [ last ];
        reached)
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

(* Over PCI an open boots the GSP, for seconds: a walk there lands at
   [spread] allocations or one more, apart over the call, so the suite keeps
   within its two minutes. *)
let spread = 3
let over_pci () = Rig_nv_support.path () = Some Rig_nv_support.Pci

(* The allocations [f ()] makes, on every thread: a walk counts those of the
   calling thread, the only one an open or a stop allocates on. *)
let allocations f =
  let n = ref 0 in
  let count _ =
    incr n;
    None
  in
  let tracker =
    { Gc.Memprof.null_tracker with alloc_minor = count; alloc_major = count }
  in
  let profile = Gc.Memprof.start ~sampling_rate:1. tracker in
  Fun.protect
    ~finally:(fun () ->
      Gc.Memprof.stop ();
      Gc.Memprof.discard profile)
    f;
  !n

(* The step that lands [spread] times over a call of [n] allocations. *)
let spread_over n = max 1 (n / spread)

(* [walk_by step attempt] runs [attempt n] from [n = 1], [step] apart, until
   an attempt no longer reaches its [n]-th allocation. *)
let walk_by step attempt =
  let rec from n = if attempt n then from (n + step) in
  from 1

(* A stop from a callback in the first growth of a new device's local
   memory: inside the device's local memory lock the stop raises, and the
   growth completes; outside it, the device stops, after which a growth
   answers [Error]. The growth is the capability's, the one an entry makes;
   an entry's other allocations hold no local memory lock. *)
let test_stop_in_local () =
  needs_gpu ();
  let seen = Rig_support.seen () in
  (* Over PCI each attempt opens a device for seconds: the walk lands
     [spread] times over a growth. *)
  let step =
    if not (over_pci ()) then 1
    else
      let g =
        Result.get_ok (Rig_gpu_support.opened (Rig_nv_support.opener 0))
      in
      let local = (N.capability g).local in
      let n = allocations (fun () -> ignore (local local_bytes)) in
      N.stop g ~fault:None;
      spread_over n
  in
  walk_by step (fun n ->
      let g =
        Result.get_ok (Rig_gpu_support.opened (Rig_nv_support.opener 0))
      in
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

(* An allocation from a callback in another allocation: on NVIDIA's kernel
   driver the path takes the GPU's addresses with no lock, so both are made,
   apart; over PCI it takes them under its lock of the GPU's memory, inside
   which the inner allocation raises, having changed nothing. *)
let test_alloc_in_alloc () =
  opened @@ fun g _ ->
  walk_calls @@ fun n ->
  let made = ref [] and refused = ref false in
  let keep () =
    match alloc g with
    | r -> made := r :: !made
    | exception Invalid_argument why when String.starts_with ~prefix:retaken why
      ->
        refused := true
  in
  let reached = Rig_support.interrupted ~landing:(Call keep) n keep in
  (* Over PCI the path allocates under its lock of the GPU's memory, inside
     which the inner allocation raises, having taken nothing. *)
  if !refused then
    equal ~msg:(strf "from allocation %d, the path" n) bool true (over_pci ());
  equal ~msg:(strf "from allocation %d, the regions made" n) int
    (if reached && not !refused then 2 else 1)
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

(* Opens and stops *)

(* [fresh what] is a new device of the first GPU, through the driver alone. *)
let fresh what =
  match Rig_gpu_support.opened (Rig_nv_support.opener 0) with
  | Ok g -> g
  | Error why -> fail (strf "%s, the GPU does not open: %s" what why)

(* [works g] allocates and frees a region of [g], then stops it. *)
let works g =
  N.free g (alloc g);
  N.stop g ~fault:None

(* Whether the allocation [a] is made outside the device's own code, [Rig_nv]'s:
   by the path as it opens the GPU, or by its caller. *)
let outside_device (a : Gc.Memprof.allocation) =
  let in_device s =
    match Printexc.Slot.location s with
    | Some l -> Filename.basename l.filename = "rig_nv.ml"
    | None -> false
  in
  match Printexc.backtrace_slots a.callstack with
  | Some slots -> not (Array.exists in_device slots)
  | None -> true

(* The allocations of an open apart in a walk of the device's make, which
   makes thousands: one in [stride] of them. *)
let stride = 64


(* [open_walk ~from ~step] walks an open, once the process's first open made
   what it keeps: the [(k step + 1)]-th allocation [from] holds for, with every
   one from it on raising until the open returns, for [k] from 0 on. The GPU
   opens again after each. It is the number of landings. *)
let open_walk ~from ~step =
  works (fresh "before the walk");
  let rec walk landed n =
    let opened = ref (Error "not reached") in
    let reached =
      Rig_support.interrupted ~from n (fun () ->
          opened := Rig_gpu_support.opened (Rig_nv_support.opener 0))
    in
    Result.iter (fun g -> N.stop g ~fault:None) !opened;
    works (fresh (strf "from allocation %d" n));
    if reached then walk (landed + 1) (n + step) else landed
  in
  walk 0 1

(* An open's every allocation outside the device's make, and one in [stride]
   of the make's, which gives back what it took on any exception; over PCI,
   [spread] allocations of the open. *)
let test_open () =
  needs_gpu ();
  if over_pci () then begin
    let opened = ref (Error "not opened") in
    let n =
      allocations (fun () ->
          opened := Rig_gpu_support.opened (Rig_nv_support.opener 0))
    in
    (match !opened with
    | Ok g -> N.stop g ~fault:None
    | Error why -> fail ("the GPU does not open: " ^ why));
    at_least ~msg:"the landings" int ~than:spread
      (open_walk ~from:(fun _ -> true) ~step:(spread_over n))
  end
  else begin
    ignore (open_walk ~from:outside_device ~step:1 : int);
    ignore (open_walk ~from:(fun _ -> true) ~step:stride : int)
  end

(* A stop with every allocation from the [(k stride + 1)]-th on raising until
   it returns, for [k] from 0 on: a region of the device is freed after it,
   and the GPU opens again. A landing before the RM freed the channels leaves
   them, as one the RM refuses does, for the life of the process, so the walk
   lands apart to stay within the RM's channels; over PCI, at [spread]
   allocations of the stop. *)
let test_stop () =
  needs_gpu ();
  let stride =
    if not (over_pci ()) then stride
    else
      let g = fresh "for the count" in
      spread_over (allocations (fun () -> N.stop g ~fault:None))
  in
  let rec walk landed n =
    let g = fresh (strf "for allocation %d" n) in
    let r = alloc g in
    let reached = Rig_support.interrupted n (fun () -> N.stop g ~fault:None) in
    N.free g r;
    works (fresh (strf "from allocation %d" n));
    if reached then walk (landed + 1) (n + stride) else landed
  in
  (* The count's stop holds no region, so it allocates no more than the
     walk's. *)
  at_least ~msg:"the landings" int
    ~than:(if over_pci () then spread else 1)
    (walk 0 1)

(* Whether the allocation [a] is made as rig makes a device of a handle its
   driver opened, before it publishes the device. *)
let in_rig_make (a : Gc.Memprof.allocation) =
  let make s =
    match Printexc.Slot.name s with
    | Some n -> String.ends_with ~suffix:"Dev.driver_device" n
    | None -> false
  in
  match Printexc.backtrace_slots a.callstack with
  | Some slots -> Array.exists make slots
  | None -> false

(* An open through rig with every allocation from the [n]-th on raising
   until it returns, counting those from the first rig makes as it makes the
   device of the driver's handle, its handlers' included, for [n] from 1 on,
   or [spread] of them over PCI: rig stops the handle, and the GPU opens
   again through rig. *)
let test_rig_open () =
  needs_gpu ();
  (* With backtraces recorded, as under OCAMLRUNPARAM=b, a handler's
     backtrace is an allocation the walk reaches. *)
  let recorded = Printexc.backtrace_status () in
  Printexc.record_backtrace true;
  Fun.protect ~finally:(fun () -> Printexc.record_backtrace recorded)
  @@ fun () ->
  let reopen what =
    match Rig.open_ (Rig_nv_support.opener 0) with
    | Ok d -> Rig.close d
    | Error why -> fail (strf "%s, a new device: %s" what why)
  in
  reopen "before the walk";
  let step =
    if not (over_pci ()) then 1
    else
      let n = ref 0 in
      let count a =
        if in_rig_make a then incr n;
        None
      in
      let tracker =
        { Gc.Memprof.null_tracker with alloc_minor = count; alloc_major = count }
      in
      let profile = Gc.Memprof.start ~sampling_rate:1. tracker in
      Fun.protect
        ~finally:(fun () ->
          Gc.Memprof.stop ();
          Gc.Memprof.discard profile)
        (fun () -> reopen "for the count");
      spread_over !n
  in
  let rec walk landed n =
    let opened = ref (Error "not reached") and made = ref false in
    let from a =
      if in_rig_make a then made := true;
      !made
    in
    let reached =
      Rig_support.interrupted ~from n (fun () ->
          opened := Rig.open_ (Rig_nv_support.opener 0))
    in
    Result.iter Rig.close !opened;
    reopen (strf "from allocation %d" n);
    if reached then walk (landed + 1) (n + step) else landed
  in
  at_least ~msg:"the landings" int ~than:(if over_pci () then spread else 2)
    (walk 0 1)

(* [guarded what f] is [f], run by a test that opens GPU 0 a device at a
   time. Where [f] fails and leaves the GPU held, so that no later test opens
   it, the later tests fail at once naming [what] and why ([needs_gpu]). *)
let guarded what f () =
  match f () with
  | () -> ()
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      (if Rig_nv_support.present () then
         match Rig_gpu_support.opened (Rig_nv_support.opener 0) with
         | Ok g -> N.stop g ~fault:None
         | Error why -> Atomic.set left_held (Some (what ^ ": " ^ why))
         | exception _ -> ());
      Printexc.raise_with_backtrace e bt

(* Through rig *)

(* A function of an image asked for through rig from a callback while
   another function of it is asked for: inside a lock, rig's or the
   driver's, the call raises as it starts; outside every lock, it answers.
   The function is made when asked for again, and the device is not lost. *)
let test_entry_in_entry_rig () =
  needs_gpu ();
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
      let call () = ignore (Rig.Image.resolve i (List.nth functions 1)) in
      let landing, landed =
        Rig_support.landing seen "Rig.Image.resolve: " call
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore (Rig.Image.resolve i (List.nth functions 0)))
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      is_none ~msg:(strf "from allocation %d, the loss" n) (Rig.lost d);
      equal ~msg:(strf "from allocation %d, the function" n) bool true
        (Option.is_some (Rig.Image.resolve i (List.nth functions 1)));
      reached);
  Rig_support.covered seen

(* [another bin n] checks that a new device of the first GPU opens through rig
   after the attempt from allocation [n], and resolves a function of [bin]. *)
let another bin n =
  match Rig.open_ (Rig_nv_support.opener 0) with
  | Error why -> fail (strf "from allocation %d, a new device: %s" n why)
  | Ok d ->
      Fun.protect ~finally:(fun () -> Rig.close d) @@ fun () ->
      let i =
        match Rig.Image.load d bin with Ok i -> i | Error why -> fail why
      in
      is_some
        ~msg:(strf "from allocation %d, a function of a new device" n)
        (Rig.Image.resolve i last)

(* A resolve through rig with every allocation from the [n]-th on raising
   until it returns, from the first allocation on until one loses the device.
   The lost device's stop runs at its close, after the call, so a new device
   of the GPU opens after each attempt. *)
let test_resolve_lost_rig () =
  needs_gpu ();
  let bin = Rig_nv_support.fixture "locals_sm89.cubin" in
  let rec from n =
    let d =
      match Rig.open_ (Rig_nv_support.opener 0) with
      | Ok d -> d
      | Error why -> fail (strf "from allocation %d, the open: %s" n why)
    in
    let reached, lost =
      Fun.protect ~finally:(fun () -> Rig.close d) @@ fun () ->
      let i =
        match Rig.Image.load d bin with Ok i -> i | Error why -> fail why
      in
      List.iter (fun f -> ignore (Rig.Image.resolve i f)) made;
      let reached =
        Rig_support.interrupted n (fun () -> ignore (Rig.Image.resolve i last))
      in
      (reached, Rig.lost d)
    in
    another bin n;
    match lost with
    | Some why ->
        equal ~msg:(strf "from allocation %d, the loss" n) string
          (Printexc.to_string Exit) why
    | None when reached -> from (n + 1)
    | None -> fail "no allocation of the resolve lost the device"
  in
  from 1

(* Inside a lock *)

(* The capability's growth of local memory, from inside a lock: it raises
   naming the call, and grows once the lock is given back. *)
let test_local_inside () =
  needs_gpu ();
  let g = Result.get_ok (Rig_gpu_support.opened (Rig_nv_support.opener 0)) in
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
             test "of an open leaves the GPU to open again"
               (guarded "the open walk" test_open);
             test "of a stop leaves the GPU to open again"
               (guarded "the stop walk" test_stop);
             test "of rig's open leaves the GPU to open again"
               (guarded "the walk of rig's open" test_rig_open);
             test "of a resolve through rig leaves the GPU to open again"
               (guarded "the resolve walk" test_resolve_lost_rig);
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
