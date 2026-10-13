(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Memory a device holds: its kinds, its budget, its cache and how collected
   buffers return. Polled logs what its driver allocates and frees. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers

let timeout = Rig_support.time 60.
let kib = 1024

let out_of_memory d n = function
  | Rig.Out_of_memory (d', n') -> Rig.equal d d' && n = n'
  | _ -> false

(* A buffer of [n] bytes on [d] that is unreachable once this returns: its
   address. *)
let[@inline never] dropped ?memory d n = Support.at (B.create ?memory d n)

(* Collects, then drains [d], as an allocation on it does first: of pinned
   memory, which no budget refuses. *)
let collect d =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create ~memory:Pinned d 8))

(* A buffer of [n] bytes on [d] that the submission [s], of one write, writes
   once, and that is unreachable once this returns: its address. Its return
   waits for that work. *)
let[@inline never] dropped_written ?memory s d n =
  let b = B.create ?memory d n in
  ignore (Rig.submit s ~run:(Sub.Run.make ()) ~buffers:[| b |]);
  Support.at b

let last n l = List.filteri (fun i _ -> i >= List.length l - n) l
let freed p at = List.exists (fun (a, _) -> a = at) (P.frees p)

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

(* Budget *)

(* An allocation over the budget raises at once: the cache stays. *)
(* The host's allocated bytes count its buffers: a buffer adds its bytes, and
   they leave once it is collected. *)
let test_host_allocated () =
  let n = 1 lsl 20 in
  Gc.full_major ();
  let before = Rig.allocated Rig.host in
  let b = ref (Some (B.create Rig.host n)) in
  at_least ~msg:"with the buffer" int ~than:(before + n)
    (Rig.allocated Rig.host);
  ignore (Sys.opaque_identity !b);
  b := None;
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  at_most ~msg:"once collected" int ~than:(before + kib)
    (Rig.allocated Rig.host)

let test_over_budget () =
  let d, p = P.open_ "memory:over-budget" in
  Rig.set_budget d (8 * kib);
  let at = dropped d (4 * kib) in
  collect d;
  raises_match
    (out_of_memory d ((8 * kib) + 1))
    (fun () -> B.create d ((8 * kib) + 1));
  equal ~msg:"the cached memory" bool false (freed p at);
  at_least int ~than:(4 * kib) (P.allocated p B.Device)

(* An allocation the driver refuses releases the cache, then raises. *)
let test_refused () =
  let d, p = P.open_ ~memory:(8 * kib) "memory:refused" in
  let live = B.create d (4 * kib) in
  let at = dropped d (4 * kib) in
  collect d;
  raises_match (out_of_memory d (8 * kib)) (fun () -> B.create d (8 * kib));
  equal ~msg:"the cached memory" bool true (freed p at);
  equal ~msg:"the live memory" bool false (freed p (Support.at live))

(* An allocation the driver refuses collects unreachable buffers and tries
   again: memory finalisers hold, two deep, returns. *)
let test_collects () =
  let d, _ = P.open_ ~memory:(4 * kib) "memory:collects" in
  (fun () ->
    let b = B.create d (4 * kib) in
    let inner = ref 0 and outer = ref 0 in
    Gc.finalise (fun _ -> ignore (Sys.opaque_identity b)) inner;
    Gc.finalise (fun _ -> ignore (Sys.opaque_identity inner)) outer)
    ();
  equal int (4 * kib) (B.length (B.create d (4 * kib)))

let test_set_budget () =
  let d, p = P.open_ ~host_visible:false "memory:set-budget" in
  let live = B.create d (8 * kib) in
  List.iter (fun _ -> ignore (dropped d (4 * kib))) [ 1; 2; 3 ];
  collect d;
  Rig.set_budget d (12 * kib);
  equal int (12 * kib) (Rig.budget d);
  collect d;
  at_most ~msg:"held after" int ~than:(12 * kib) (P.allocated p B.Device);
  Rig.set_budget d 0;
  collect d;
  equal ~msg:"live memory stays" bool false (freed p (Support.at live));
  raises_match Exn.invalid_arg (fun () -> Rig.set_budget d (-1))

(* A budget set while another domain loses the device raises nothing, and the
   cache it took returns once the device stopped, each memory once. The first
   free of the cache waits at Polled's free gate while the loss runs. *)
let test_budget_beside_loss () =
  let d, p = P.open_ "memory:budget-loss" in
  let ats = List.map (fun _ -> dropped d (4 * kib)) [ 1; 2; 3 ] in
  collect d;
  P.gate_frees p;
  let setter =
    Domain.spawn (fun () ->
        match Rig.set_budget d 0 with
        | () -> "returned"
        | exception e -> Printexc.to_string e)
  in
  Support.await "the cache's first free at the gate" (fun () ->
      P.freers p = 1);
  P.fail p;
  (try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ());
  P.open_frees p;
  equal ~msg:"the budget" string "returned" (Domain.join setter);
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  equal ~msg:"the cached memory, freed once"
    (list bool) [ true; true; true ] (List.map (freed p) ats);
  equal ~msg:"frees" int 3
    (List.length (List.filter (fun (a, _) -> List.mem a ats) (P.frees p)))

(* Collects, then frees the cache of each of [ds], which first drains it: a
   closed device's memory collected after its stop returns so. *)
let settle ds =
  Gc.full_major ();
  List.iter Rig.free_cache ds

(* A closed device's cache returns its memory to the driver once the device
   stopped. Buffers of 40 sizes fill 40 keys of the cache, which grows. *)
let test_closed_cache () =
  let d, p = P.open_ "memory:closed-cache" in
  let sizes = List.init 40 (fun k -> (k + 1) * 64) in
  let[@inline never] drop () =
    List.iter (fun n -> ignore (Sys.opaque_identity (B.create d n))) sizes
  in
  drop ();
  collect d;
  let cached = List.fold_left ( + ) 0 sizes in
  greater ~msg:"bytes in the cache" int ~than:(cached - 1) (Rig.allocated d);
  Rig.close d;
  settle [ d ];
  equal ~msg:"regions left with the driver" (list int) []
    (List.filter (fun n -> n = 8 || List.mem n sizes) (P.outstanding p))

(* Memory goes back to its driver once, when its last mapping goes after its
   device was lost and before its stop returned. The mapper's work keeps the
   mapping until the memory is freed; then, while a free of other memory holds
   the owner's stop at Polled's free gate in another domain, the owner is lost
   and the mapper's drain gives the mapping back. *)
let test_give_beside_stop () =
  let o, po = P.open_ "memory:give-owner" in
  let c, pc = P.open_ "memory:give-mapper" in
  let[@inline never] mapped () =
    let m = B.create o (4 * kib) in
    let s = Sub.make c [| Support.touching [| Sub.read |] |] in
    Rig.Point.wait (submit s ~buffers:[| Option.get (B.borrow c m) |]);
    (Support.at m, ref (Some m))
  in
  let at, held = mapped () in
  ignore (submit (Sub.make c [||]) : Rig.Point.t);
  held := None;
  collect o;
  Rig.set_budget o 0;
  equal ~msg:"the memory while it is mapped" bool false (freed po at);
  Rig.set_budget o (1 lsl 30);
  let other = dropped o (4 * kib) in
  collect o;
  P.gate_frees po;
  let freer = Domain.spawn (fun () -> Rig.set_budget o 0) in
  Support.await "the other memory's free at the gate" (fun () ->
      P.freers po = 1);
  P.fail po;
  (try ignore (submit (Sub.make o [||])) with Rig.Lost _ -> ());
  ignore (P.run pc : int);
  collect c;
  P.open_frees po;
  Domain.join freer;
  settle [ o; c ];
  let frees at = List.length (List.filter (fun (a, _) -> a = at) (P.frees po)) in
  equal ~msg:"the other memory's frees" int 1 (frees other);
  equal ~msg:"the mapped memory's frees" int 1 (frees at)

(* Two borrows on one device that race to map the same memory leave one
   mapping, and the memory goes back to its driver after the other is given
   back. The second borrow runs inside the first one's map, at Polled's probe
   of it. *)
let test_raced_maps () =
  let o, po = P.open_ "memory:raced-owner" in
  let c, pc = P.open_ "memory:raced-mapper" in
  let[@inline never] mapped () =
    let m = B.create o (4 * kib) in
    P.before pc "map_peer" (fun () -> ignore (Option.get (B.borrow c m)));
    ignore (Option.get (B.borrow c m));
    (Support.at m, ref (Some m))
  in
  let at, held = mapped () in
  equal ~msg:"the maps" int 2
    (List.length (List.filter (( = ) "map_peer") (P.log pc)));
  held := None;
  collect o;
  Rig.set_budget o 0;
  equal ~msg:"the memory" bool true (freed po at);
  equal ~msg:"the mapper's regions once it went back" (list int) []
    (P.outstanding pc)

(* A device closed while a borrow on another domain maps memory for it gives
   the mapping back, and the memory goes back to its driver once. The borrow's
   locate of the mapping waits at Polled's locate gate while the close
   runs. *)
let test_close_beside_borrow () =
  let o, po = P.open_ "memory:borrow-owner" in
  let d, pd = P.open_ "memory:borrow-mapper" in
  let[@inline never] made () =
    let m = B.create o (4 * kib) in
    (Support.at m, ref (Some m))
  in
  let at, held = made () in
  P.gate_locates pd;
  let borrower =
    Domain.spawn (fun () ->
        match B.borrow d (Option.get !held) with
        | Some _ -> "borrowed"
        | None -> "none"
        | exception Rig.Lost _ -> "lost")
  in
  Support.await "the borrow's locate at the gate" (fun () ->
      P.locaters pd = 1);
  Rig.close d;
  settle [ d ];
  P.open_locates pd;
  equal ~msg:"the borrow" string "lost" (Domain.join borrower);
  held := None;
  collect o;
  Rig.set_budget o 0;
  settle [ d; o ];
  equal ~msg:"the mapper's regions" (list int) [] (P.outstanding pd);
  equal ~msg:"the memory's frees" int 1
    (List.length (List.filter (fun (a, _) -> a = at) (P.frees po)))

(* Memory goes back to its driver after every other device's mapping of it,
   whichever ends first. The mapper's close gives back its mapping in another
   domain, held at Polled's free gate while the memory is collected and its
   owner's cache emptied. *)
let test_free_after_maps () =
  let o, po = P.open_ "memory:mapped-owner" in
  let d, pd = P.open_ "memory:mapper" in
  let[@inline never] mapped () =
    let m = B.create o (4 * kib) in
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    Rig.Point.wait (submit s ~buffers:[| Option.get (B.borrow d m) |]);
    (Support.at m, ref (Some m))
  in
  let at, held = mapped () in
  equal ~msg:"the mapper maps the memory" bool true
    (List.mem "map_peer" (P.log pd));
  P.gate_frees pd;
  let mapper = Domain.spawn (fun () -> Rig.close d) in
  Support.await "the mapper's unmap at the gate" (fun () -> P.freers pd = 1);
  held := None;
  collect o;
  Rig.set_budget o 0;
  equal ~msg:"the memory while it is mapped" bool false (freed po at);
  P.open_frees pd;
  Domain.join mapper;
  settle [ o; d ];
  equal ~msg:"the memory once unmapped" bool true (freed po at);
  equal ~msg:"the mapper's regions" (list int) [] (P.outstanding pd)

(* Memory two devices map is freed in another domain, whose give-back of one
   mapping waits at that mapper's free gate while it holds the other's
   mapping too. Both gates close first and the other's opens once one
   holds, so the test needs no order of the two. The other mapper meanwhile
   closes: nothing raises its loss, every region goes back, and the memory
   is freed once. *)
let test_mapper_beside_free () =
  let o, po = P.open_ "memory:outlived-owner" in
  let a, pa = P.open_ "memory:outlived-a" in
  let b, pb = P.open_ "memory:outlived-b" in
  let map d m =
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    Rig.Point.wait (submit s ~buffers:[| Option.get (B.borrow d m) |])
  in
  let[@inline never] mapped () =
    let m = B.create o (4 * kib) in
    map a m;
    map b m;
    (Support.at m, ref (Some m))
  in
  let at, held = mapped () in
  P.gate_frees pa;
  P.gate_frees pb;
  let freer =
    Domain.spawn (fun () ->
        held := None;
        match
          collect o;
          Rig.set_budget o 0
        with
        | () -> "freed"
        | exception Rig.Lost (d', _) -> Rig.name d' ^ " lost")
  in
  Support.await "an unmap at a gate" (fun () -> P.freers pa + P.freers pb = 1);
  let gated, other, po' =
    if P.freers pa = 1 then (pa, b, pb) else (pb, a, pa)
  in
  P.open_frees po';
  Rig.close other;
  settle [ other ];
  P.open_frees gated;
  equal ~msg:"the memory's free" string "freed" (Domain.join freer);
  settle [ o; a; b ];
  equal ~msg:"the closed mapper's regions" (list int) [] (P.outstanding po');
  equal ~msg:"the held mapping" bool true (List.mem "unmap" (P.log gated));
  equal ~msg:"the memory's frees" int 1
    (List.length (List.filter (fun (x, _) -> x = at) (P.frees po)))

(* A drain that returns collected memory to the driver of a device another
   domain loses meanwhile: its first free waits at Polled's free gate while the
   loss is recorded, and the device stops once that free returned. The drain's
   allocation raises the loss, and each memory is freed once. *)
let test_drain_beside_loss () =
  let d, p = P.open_ "memory:drain-loss" in
  let ats = List.map (fun _ -> dropped d (4 * kib)) [ 1; 2; 3 ] in
  Rig.set_budget d 8;
  Gc.full_major ();
  Gc.full_major ();
  P.gate_frees p;
  let drainer =
    Domain.spawn (fun () ->
        match B.create d 8 with
        | _ -> "returned"
        | exception Rig.Lost (_, why) -> "lost: " ^ why)
  in
  Support.await "the drain's first free at the gate" (fun () ->
      P.freers p = 1);
  P.fail p;
  (try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ());
  let stops () = List.length (List.filter (( = ) "stop") (P.log p)) in
  equal ~msg:"stops while the free waits" int 0 (stops ());
  P.open_frees p;
  equal ~msg:"the drain's allocation" string "lost: the submission failed"
    (Domain.join drainer);
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  equal ~msg:"stops" int 1 (stops ());
  equal ~msg:"the collected memory, freed once"
    (list bool) [ true; true; true ] (List.map (freed p) ats);
  equal ~msg:"frees" int 3
    (List.length (List.filter (fun (a, _) -> List.mem a ats) (P.frees p)))

(* The host's out-of-memory ladder waits for a device whose work holds back
   host memory, and a fault that wait finds is that device's loss, never the
   allocation's: the host allocation raises Out_of_memory for itself. *)
let test_ladder_beside_fault () =
  let d, p = P.open_ ~host_visible:false "memory:ladder-fault" in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  ignore (Sys.opaque_identity (dropped_written s ~memory:Pinned d (64 * kib)));
  collect Rig.host;
  P.fault p "the engine hung";
  let budget = Rig.budget Rig.host in
  Rig.set_budget Rig.host 0;
  let raised =
    Fun.protect
      ~finally:(fun () -> Rig.set_budget Rig.host budget)
      (fun () ->
        match B.create Rig.host (64 * kib) with
        | _ -> "allocated"
        | exception Rig.Out_of_memory (d', _) when Rig.equal d' Rig.host ->
            "out of memory"
        | exception e -> Printexc.to_string e)
  in
  equal ~msg:"the host allocation" string "out of memory" raised;
  equal ~msg:"the device" (option string) (Some "the engine hung") (Rig.lost d)

(* Memory collected while its device holds more than its budget returns to the
   driver: an allocation the budget refuses finds none of it to reuse, and the
   live buffers stay within the budget. *)
let test_over_budget_cache () =
  let d, _ = P.open_ "memory:over-budget-cache" in
  ignore (dropped d (4 * kib));
  let live = B.create d 1 in
  Rig.set_budget d (4 * kib);
  raises_match (out_of_memory d (4 * kib)) (fun () -> B.create d (4 * kib));
  ignore (Sys.opaque_identity live)

(* Memory queued work reads returns to the driver over the budget only once
   that work ran: an over-budget return skips the cache, never the wait. *)
let test_over_budget_queued () =
  let d, p = P.open_ "memory:over-budget-queued" in
  let reads = Sub.make d [| Support.touching [| Sub.read |] |] in
  let[@inline never] queued () =
    let b = B.create d 64 in
    ignore (submit reads ~buffers:[| b |]);
    Support.at b
  in
  let at = queued () in
  Rig.set_budget d 0;
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create d 0)
  in
  drain ();
  equal ~msg:"before the read ran" bool false (freed p at);
  ignore (P.run p);
  drain ();
  equal ~msg:"once it ran" bool true (freed p at)

(* Memory collected over the budget while queued work reads it, its return to
   the driver waiting for that work. *)
let queued_over_budget name =
  let d, p = P.open_ name in
  let reads = Sub.make d [| Support.touching [| Sub.read |] |] in
  let[@inline never] queued () =
    let b = B.create d (64 * kib) in
    ignore (submit reads ~buffers:[| b |]);
    Support.at b
  in
  let at = queued () in
  Rig.set_budget d (32 * kib);
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create d 0);
  (d, p, at)

(* An allocation the budget refuses waits for the work that holds back collected
   memory's return, then reuses its room. *)
let test_refused_waits () =
  let d, p, at = queued_over_budget "memory:refused-waits" in
  equal int (32 * kib) (B.length (B.create d (32 * kib)));
  equal ~msg:"the queued memory" bool true (freed p at)

(* A Ctrl-C in a reclamation round's wait for the device's work loses none of
   its cache: the cache the round would have freed returns at the next
   free_cache, and only the live buffer stays. *)
let test_break_in_round () =
  let d, p = P.open_ "memory:break-round" in
  Rig.set_budget d (8 * kib);
  let live = B.create d (4 * kib) in
  let at = dropped d (4 * kib) in
  Gc.full_major ();
  ignore (B.create d 0);
  ignore (submit (Sub.make d [||]));
  P.interrupt p;
  Sys.catch_break true;
  let broke =
    Fun.protect
      ~finally:(fun () -> Sys.catch_break false)
      (fun () ->
        match B.create d (2 * kib) with
        | _ -> false
        | exception Sys.Break -> true)
  in
  equal ~msg:"the round's wait broke" bool true broke;
  ignore (P.run p);
  Rig.free_cache d;
  equal ~msg:"the cached memory" bool true (freed p at);
  equal ~msg:"bytes held" int (4 * kib) (Rig.allocated d);
  ignore (Sys.opaque_identity live)

(* Two drains of one device judge a retiring buffer one at a time. The first
   judges a buffer over the budget whose work ran, which only a read of the
   word behind the transport shows; while it reads, a drain on another
   domain waits for it and frees nothing the first reads. The buffer goes
   back once. *)
let test_judged_once () =
  if not Support.sees_waits then skip ~reason:"no view of lock waits" ();
  let d, p = P.open_ ~transport:true "memory:judged-once" in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  let at = dropped_written s d (4 * kib) in
  collect d;
  Rig.set_budget d kib;
  ignore (P.run p);
  let judging = Atomic.make false and second = Atomic.make None in
  let drained = Atomic.make false and freed_inside = ref true in
  let other =
    Domain.spawn (fun () ->
        Support.await "the first drain's read" (fun () -> Atomic.get judging);
        Atomic.set second (Some (Support.self ()));
        ignore (Sys.opaque_identity (B.create ~memory:Pinned d 8));
        Atomic.set drained true)
  in
  P.before p "signaled" (fun () ->
      Atomic.set judging true;
      Support.await "the second drain" (fun () ->
          Atomic.get drained
          ||
          match Atomic.get second with
          | Some t -> Support.waits_for_lock t
          | None -> false);
      freed_inside := freed p at);
  collect d;
  Domain.join other;
  equal ~msg:"freed while the first drain read it" bool false !freed_inside;
  equal ~msg:"frees of the buffer" int 1
    (List.length (List.filter (fun (a, _) -> a = at) (P.frees p)));
  equal ~msg:"bytes held" int 0 (P.allocated p B.Device)

(* The wait finds the loss of a device that faulted with that work queued. *)
let test_refused_lost () =
  let d, p, _ = queued_over_budget "memory:refused-lost" in
  P.fault p "the engine hung";
  raises_match
    (function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false)
    (fun () -> B.create d (32 * kib))

let test_free_cache () =
  let d, p = P.open_ "memory:free-cache" in
  let at = dropped d (4 * kib) in
  collect d;
  equal ~msg:"cached" bool false (freed p at);
  Rig.free_cache d;
  equal ~msg:"after free_cache" bool true (freed p at)

(* A copy between devices that map none of each other's memory goes through the
   host's staging memory: a host whose budget cannot hold a half of a slot
   refuses at once, and the refused copy gives its slot back, so a copy after
   more refusals than slots still finds one. The staging memory is made at the
   first copy that needs it, and no earlier test of this suite copies through
   it. *)
let test_staging_refused () =
  let open_ name = P.open_ ~host_visible:false ~peers:false name in
  let d, _ = open_ "memory:staging-src" and e, _ = open_ "memory:staging-dst" in
  let src = B.create d 64 and dst = B.create e 64 in
  let budget = Rig.budget Rig.host in
  Rig.set_budget Rig.host (64 * kib);
  Fun.protect
    ~finally:(fun () -> Rig.set_budget Rig.host budget)
    (fun () ->
      for _ = 1 to 3 do
        raises_match
          (out_of_memory Rig.host (32 * kib * kib))
          (fun () -> B.copy ~src ~dst)
      done);
  B.copy ~src ~dst

let test_host_budget () = equal int max_int (Rig.budget Rig.host)

(* Kinds *)

(* Runs [f] with the host's budget [n], then restores it. *)
let with_host_budget n f =
  let before = Rig.budget Rig.host in
  Rig.set_budget Rig.host n;
  Fun.protect ~finally:(fun () -> Rig.set_budget Rig.host before) f

(* A host allocation the C library refuses raises Out_of_memory for the host,
   and gives its bytes back to the host's budget: a later buffer under a budget
   below the refused size, beyond what the host holds, is made. *)
let test_host_refused () =
  let n = 1 lsl 60 in
  raises_match (out_of_memory Rig.host n) (fun () -> B.create Rig.host n);
  with_host_budget
    (Support.host_held () + (64 * 1024 * kib))
    (fun () ->
      equal int
        (32 * 1024 * kib)
        (B.length (B.create Rig.host (32 * 1024 * kib))))

(* A host buffer holds its bytes in the host's budget until it is collected,
   small ones included, with up to 112 more where it starts on 128 bytes. *)
let test_host_held () =
  List.iter
    (fun n ->
      Gc.full_major ();
      let before = Support.host_held () in
      let held =
        (fun () ->
          let b = B.create Rig.host n in
          let held = Support.host_held () in
          ignore (Sys.opaque_identity b);
          held)
          ()
      in
      let msg = Printf.sprintf "%d bytes, live" n in
      at_least ~msg int ~than:(before + n) held;
      at_most ~msg int ~than:(before + n + 112) held;
      Gc.full_major ();
      Gc.full_major ();
      equal
        ~msg:(Printf.sprintf "%d bytes, collected" n)
        int before (Support.host_held ()))
    [ 16; 4 * kib; 128 * kib ]

(* On a device whose memory the host does not address, pinned memory is host
   memory: it counts in the host's budget, not the device's. *)
let test_pinned () =
  let d, p = P.open_ ~host_visible:false "memory:pinned" in
  Rig.set_budget d 0;
  equal int (4 * kib) (B.length (B.create ~memory:Pinned d (4 * kib)));
  equal allocs [ (B.Pinned, 4 * kib, true) ] (P.allocs p);
  raises_match (out_of_memory d 1) (fun () -> B.create d 1);
  let n = 1 lsl 40 in
  with_host_budget n (fun () ->
      raises_match
        (out_of_memory d (n + 1))
        (fun () -> B.create ~memory:Pinned d (n + 1)));
  equal ~msg:"asked of the driver" int 1 (List.length (P.allocs p))

(* Pinned memory charged to the host's budget that another device keeps in its
   cache returns when a pinned allocation the host's budget refuses reclaims:
   the allocation is made. *)
let test_pinned_reclaims () =
  let a, _ = P.open_ ~host_visible:false "memory:pinned-asker" in
  let b, _ = P.open_ ~host_visible:false "memory:pinned-keeper" in
  let n = 64 * kib in
  ignore (dropped ~memory:Pinned b n);
  collect b;
  with_host_budget (Support.host_held ()) (fun () ->
      equal int n (B.length (B.create ~memory:Pinned a n)))

(* A reclamation round waits for a device whose lock another thread holds:
   the device's cached memory charged to the host's budget returns to a host
   allocation the budget refused. The other thread holds the lock while its
   drain reads the device's word behind the transport, until the round waits
   for the lock. *)
let test_round_waits_for_lock () =
  if not Support.sees_waits then skip ~reason:"no view of lock waits" ();
  let d, p = P.open_ ~host_visible:false ~transport:true "memory:round-busy" in
  let n = 64 * kib in
  let at = dropped ~memory:Pinned d n in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  ignore (dropped_written s d kib);
  collect d;
  let held = Atomic.make false and round = Atomic.make None in
  let returned = Atomic.make false in
  P.before p "signaled" (fun () ->
      Atomic.set held true;
      Support.await "the round's wait for the lock" (fun () ->
          Atomic.get returned
          ||
          match Atomic.get round with
          | Some t -> Support.waits_for_lock t
          | None -> false));
  let holder = Thread.create (fun () -> ignore (B.create d 0)) () in
  Support.await "the device's lock held" (fun () -> Atomic.get held);
  let allocated =
    with_host_budget (Support.host_held ()) (fun () ->
        Domain.join
          (Domain.spawn (fun () ->
               Atomic.set round (Some (Support.self ()));
               let r =
                 match B.create Rig.host (n / 2) with
                 | b -> string_of_int (B.length b)
                 | exception e -> Printexc.to_string e
               in
               Atomic.set returned true;
               r)))
  in
  Thread.join holder;
  equal ~msg:"the host allocation" string (string_of_int (n / 2)) allocated;
  equal ~msg:"the cached memory" bool true (freed p at)

(* A host buffer a device borrowed, dropped, returns its bytes when a pinned
   allocation the host's budget refuses reclaims: the reclaim drains the host,
   whose list holds the buffer, and the allocation is made. *)
let test_borrowed_reclaimed () =
  let d, _ = P.open_ ~host_visible:false "memory:borrowed-reclaim" in
  let n = 64 * kib in
  let[@inline never] borrowed () =
    ignore (Sys.opaque_identity (B.borrow d (B.create Rig.host n)))
  in
  borrowed ();
  with_host_budget (Support.host_held ()) (fun () ->
      equal int n (B.length (B.create ~memory:Pinned d n)))

(* On a device whose memory the host addresses, pinned memory is the device's
   own: it counts in the device's budget. *)
let test_pinned_own () =
  let d, _ = P.open_ "memory:pinned-own" in
  Rig.set_budget d (8 * kib);
  raises_match
    (out_of_memory d (16 * kib))
    (fun () -> B.create ~memory:Pinned d (16 * kib));
  let b = B.create ~memory:Pinned d (4 * kib) in
  raises_match (out_of_memory d (8 * kib)) (fun () -> B.create d (8 * kib));
  ignore (Sys.opaque_identity b)

(* Mapped memory the window cannot hold is pinned memory, and the cache
   stays. *)
let test_mapped_window () =
  let d, p = P.open_ ~window:(4 * kib) "memory:window" in
  let at = dropped d (4 * kib) in
  collect d;
  let b = B.create ~memory:Mapped d (8 * kib) in
  equal ~msg:"the cached memory" bool false (freed p at);
  equal allocs
    [ (B.Mapped, 8 * kib, false); (B.Pinned, 8 * kib, true) ]
    (last 2 (P.allocs p));
  ignore (Sys.opaque_identity b)

let test_mapped_budget () =
  let d, p = P.open_ ~host_visible:false "memory:mapped-budget" in
  Rig.set_budget d (4 * kib);
  ignore (B.create ~memory:Mapped d (8 * kib));
  equal allocs [ (B.Pinned, 8 * kib, true) ] (P.allocs p)

let test_mapped () =
  let d, p = P.open_ "memory:mapped" in
  let b = B.create ~memory:Mapped d (4 * kib) in
  equal allocs [ (B.Mapped, 4 * kib, true) ] (P.allocs p);
  equal bool true (Rig.equal d (B.device b))

(* Reclamation *)

(* A buffer dropped by a domain that then blocks returns to its device. *)
let test_blocked_domain () =
  let d, p = P.open_ "memory:blocked" in
  let lock = Mutex.create () and at = Atomic.make 0 in
  Mutex.lock lock;
  let blocked =
    Domain.spawn (fun () ->
        Atomic.set at (dropped d (4 * kib));
        Mutex.lock lock;
        Mutex.unlock lock)
  in
  Support.await "a dropped buffer" (fun () -> Atomic.get at <> 0);
  collect d;
  Rig.free_cache d;
  collect d;
  let returned = freed p (Atomic.get at) in
  Mutex.unlock lock;
  Domain.join blocked;
  equal bool true returned

(* Dropped buffers of many budgets never force a collection: the collector is
   paced by the device's memory and finds them before the budget runs out. *)
let test_paced () =
  let budget = 4 * kib * kib in
  let d, _ = P.open_ "memory:paced" in
  Rig.set_budget d budget;
  let forced () = (Gc.quick_stat ()).forced_major_collections in
  let before = forced () in
  for _ = 1 to 20 * 16 do
    ignore (Sys.opaque_identity (B.create d (budget / 16)))
  done;
  equal int 0 (forced () - before)

(* Memory another device used enters its owner's cache once that use is
   reached. *)
let test_foreign_use () =
  let a, pa = P.open_ "memory:owner" in
  let b, pb = P.open_ "memory:reader" in
  let at =
    (fun () ->
      let m = B.create a (4 * kib) in
      let s = Sub.make b [| Support.touching [| Sub.read |] |] in
      ignore (submit s ~buffers:[| require_some (B.borrow b m) |]);
      Support.at m)
      ()
  in
  collect a;
  Rig.free_cache a;
  collect a;
  equal ~msg:"while the read is unreached" bool false (freed pa at);
  ignore (P.run pb);
  collect a;
  Rig.free_cache a;
  collect a;
  equal ~msg:"once it is reached" bool true (freed pa at)

(* Host memory a device borrowed returns only once that device's work on it is
   done: until then no host buffer reuses it. *)
let test_borrowed_host () =
  let d, p = P.open_ "memory:borrowed-host" in
  let n = 1 lsl 16 in
  let[@inline never] written () =
    let h = B.create Rig.host n in
    let src = B.create d n in
    let dst = require_some (B.borrow d h) in
    let part =
      Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
    in
    ignore (submit (Sub.make d [| part |]));
    Support.at h
  in
  let at = written () in
  Gc.full_major ();
  Gc.full_major ();
  let other = B.create Rig.host n in
  equal ~msg:"while the work is queued" bool false (Support.at other = at);
  ignore (P.run p);
  ignore (B.create Rig.host 0);
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"once it ran" int at (Support.at (B.create Rig.host n));
  ignore (Sys.opaque_identity other)

(* A bigarray a device borrowed through of_bigarray stays reachable until that
   device's queued work on it ran. *)
let test_borrowed_bigarray () =
  let d, p = P.open_ "memory:borrowed-bigarray" in
  let n = 1 lsl 16 in
  let collected = Atomic.make false in
  let[@inline never] written () =
    let ba = B.export (B.create Rig.host n) Bigarray.char in
    Gc.finalise (fun _ -> Atomic.set collected true) ba;
    let src = B.create d n in
    let dst = require_some (B.borrow d (B.of_bigarray ba)) in
    let part =
      Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
    in
    ignore (submit (Sub.make d [| part |]))
  in
  let settle () =
    Gc.full_major ();
    ignore (B.create Rig.host 0);
    Gc.full_major ();
    Gc.full_major ()
  in
  written ();
  settle ();
  equal ~msg:"while the work is queued" bool false (Atomic.get collected);
  ignore (P.run p);
  settle ();
  equal ~msg:"once it ran" bool true (Atomic.get collected)

(* Host memory an idle device borrowed returns to a host allocation that needs
   its room once the device's work ran, though nothing waited for it. *)
let test_idle_borrower () =
  let d, p = P.open_ "memory:idle-borrower" in
  let n = 1 lsl 20 in
  let[@inline never] written () =
    let h = B.create Rig.host n in
    let src = B.create d n in
    let dst = require_some (B.borrow d h) in
    let part =
      Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
    in
    ignore (submit (Sub.make d [| part |]))
  in
  Gc.full_major ();
  Rig.free_cache Rig.host;
  written ();
  ignore (P.run p);
  let room = Support.host_held () + (n / 2) in
  with_host_budget room (fun () -> equal int n (B.length (B.create Rig.host n)))

(* The C heap's allocated bytes, for a test that needs them: the C library
   says on macOS and glibc. *)
let heap_bytes () =
  match Support.heap_bytes () with
  | Some n -> n
  | None -> skip ~reason:"the C library does not count its heap" ()

(* free_cache on the host gives back the memory it keeps of collected buffers:
   the C heap shrinks by a dropped buffer's bytes. *)
let test_host_free_cache () =
  let n = 4 * 1024 * kib in
  Gc.full_major ();
  Rig.free_cache Rig.host;
  ignore (dropped Rig.host n);
  Gc.full_major ();
  Gc.full_major ();
  let kept = heap_bytes () in
  Rig.free_cache Rig.host;
  at_least ~msg:"bytes given back" int ~than:n (kept - heap_bytes ())

(* set_budget on the host gives back the memory it keeps until it holds at most
   the new budget: of four dropped buffers of 1 MiB, under a budget 2.5 MiB above
   its live bytes, it gives back at least 1.5 MiB and keeps the rest that fits. *)
let test_host_set_budget () =
  let mib = 1 lsl 20 in
  Gc.full_major ();
  Rig.free_cache Rig.host;
  let live = ref (List.init 4 (fun _ -> B.create Rig.host mib)) in
  ignore (Sys.opaque_identity !live);
  live := [];
  Gc.full_major ();
  Gc.full_major ();
  let kept = heap_bytes () in
  with_host_budget
    (Support.host_held () + (5 * mib / 2))
    (fun () ->
      let given = kept - heap_bytes () in
      at_least ~msg:"given back" int ~than:(3 * mib / 2) given;
      at_most ~msg:"what fits stays" int ~than:(3 * mib) given)

(* A collected buffer's memory serves the next buffer of the buffer's size,
   whichever array over it is collected last: a float32 view that outlives its
   buffer keeps the memory from reuse, and once it is collected too, the next
   buffer of the buffer's size takes it. *)
let test_host_keeps_size () =
  let n = 1 lsl 20 in
  Gc.full_major ();
  Rig.free_cache Rig.host;
  let at = ref 0 in
  let view =
    ref
      (Some
         ((fun () ->
            let b = B.create Rig.host n in
            at := Support.at b;
            B.export b Bigarray.float32)
            ()))
  in
  Gc.full_major ();
  Gc.full_major ();
  let other = B.create Rig.host n in
  not_equal ~msg:"while the view lives" int !at (Support.at other);
  ignore (Sys.opaque_identity !view);
  view := None;
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"once it is collected" int !at (Support.at (B.create Rig.host n));
  ignore (Sys.opaque_identity other)

(* The host keeps a collected buffer of 64 KiB or more for the next buffer of
   its size, unless a bigarray of it lives, which keeps its bytes. *)
let test_host_cache () =
  let n = (64 * kib) + 4093 in
  let first = dropped Rig.host n in
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"reused" int first (Support.at (B.create Rig.host n));
  let at, view =
    (fun () ->
      let b = B.create Rig.host n in
      let ba = B.export b Bigarray.char in
      Bigarray.Array1.fill ba 'a';
      (Support.at b, ba))
      ()
  in
  Gc.full_major ();
  Gc.full_major ();
  let b = B.create Rig.host n in
  not_equal ~msg:"the viewed memory" int at (Support.at b);
  B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a 'b');
  equal ~msg:"the view keeps its bytes" char 'a' view.{n - 1}

(* Two domains allocating on one device under a budget: the live buffers never
   exceed it. A program opens the device, then each domain allocates on it up
   to five times against a budget of nine buffers, so the last allocation
   races for the room. Each refusal runs the out-of-memory ladder, whose
   collections stop every domain, and a program runs 50 times: [~steps:1]
   keeps other calls out, so a program refuses at most once. A program that
   draws no open calls nothing, which [~count] allows for. *)
type budgeted = { d : Rig.t; live : B.t list ref; lock : Mutex.t }
type budgeted_model = { mutable held : int }

let per = 4 * kib
let slots = 9
let budgeted_opened = Atomic.make 0

let make_budgeted () =
  let n = Atomic.fetch_and_add budgeted_opened 1 in
  let d, _ = P.open_ (Printf.sprintf "memory:budgeted-%d" n) in
  Rig.set_budget d (slots * per);
  { d; live = ref []; lock = Mutex.create () }

let alloc_budgeted t =
  let b = B.create t.d per in
  Mutex.protect t.lock (fun () -> t.live := b :: !(t.live))

let judge_alloc r = function
  | Ok () ->
      at_most ~msg:"live buffers" int ~than:(slots - 1) r.held;
      cover "an allocation that fills the budget" (r.held = slots - 1);
      r.held <- r.held + 1
  | Error (Rig.Out_of_memory _) ->
      cover "an allocation the budget refuses" true;
      equal ~msg:"live buffers" int slots r.held
  | Error e -> raise e

let budgeted =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "held %d" r.held)
    ~release:(fun t -> Rig.close t.d)
    "b"

let budget_commands =
  [
    command "open"
      (Gen.unit @-> makes budgeted)
      (fun () -> { held = 0 })
      make_budgeted;
    command "alloc" (budgeted ^-> judges unit) judge_alloc alloc_budgeted;
  ]

let tests =
  [
    group ~timeout "budget"
      [
        test "the host's allocated bytes count its buffers"
          test_host_allocated;
        test "an allocation over the budget raises at once and keeps the cache"
          test_over_budget;
        test "a copy whose staging memory the host refuses gives its slot back"
          test_staging_refused;
        stateful ~count:16 ~steps:1 ~domains:2
          "two domains' allocations stay within the budget" budget_commands;
        test "an allocation the driver refuses releases the cache, then raises"
          test_refused;
        test "memory collected over the budget is not reused past it"
          test_over_budget_cache;
        test "memory queued work reads is freed over the budget once it ran"
          test_over_budget_queued;
        test
          "an allocation the budget refuses waits for the work collected \
           memory waits for"
          test_refused_waits;
        test "an allocation the budget refuses raises the loss of that work"
          test_refused_lost;
        test "an allocation the driver refuses collects unreachable buffers"
          test_collects;
        test "set_budget returns cached memory, never live memory"
          test_set_budget;
        test
          "set_budget while another domain loses the device raises nothing \
           and returns its cache once"
          test_budget_beside_loss;
        test "a drain beside a loss frees each memory once, then stops"
          test_drain_beside_loss;
        test "memory goes back after every mapping of it, a closed mapper's \
              included"
          test_free_after_maps;
        test "memory whose last mapping goes before its stop goes back once"
          test_give_beside_stop;
        test "memory two raced maps share goes back after both"
          test_raced_maps;
        test "a borrow beside its mapper's close goes back with the memory"
          test_close_beside_borrow;
        test
          "the host's out-of-memory ladder raises a device's fault it finds as \
           that device's loss, never the allocation's"
          test_ladder_beside_fault;
        test "free_cache with no work in flight returns the cache at once"
          test_free_cache;
        test "the host's budget is max_int" test_host_budget;
        test "a host buffer holds its bytes in the budget until collected"
          test_host_held;
        test "a host allocation the C library refuses raises for the host"
          test_host_refused;
      ];
    group ~timeout "kinds"
      [
        test "pinned memory of hidden device memory counts in the host's budget"
          test_pinned;
        test "pinned memory the host addresses counts in its device's budget"
          test_pinned_own;
        test "mapped memory the window cannot hold is pinned, the cache kept"
          test_mapped_window;
        test "mapped memory the budget cannot hold is pinned" test_mapped_budget;
        test "mapped memory is the device's" test_mapped;
        test "pinned memory another device keeps returns to a refused one"
          test_pinned_reclaims;
        test "a round waits for a device whose lock another thread holds"
          test_round_waits_for_lock;
        test
          "a dropped host buffer a device borrowed returns to a pinned \
           allocation"
          test_borrowed_reclaimed;
      ];
    group ~timeout "reclamation"
      [
        test "a Ctrl-C in a reclamation round's wait loses none of the cache"
          test_break_in_round;
        test "two drains of one device judge its retiring memory in turn"
          test_judged_once;
        test "a buffer dropped by a domain that then blocks returns"
          test_blocked_domain;
        test "dropped buffers of many budgets never force a collection"
          test_paced;
        test "memory another device used is cached once that use is reached"
          test_foreign_use;
        test "the host keeps a collected buffer's memory for its size"
          test_host_cache;
        test "free_cache on the host gives back what it keeps"
          test_host_free_cache;
        test "set_budget on the host gives back what it keeps beyond the budget"
          test_host_set_budget;
        test "a buffer's memory serves its size once a view outliving it goes"
          test_host_keeps_size;
        test "host memory a device borrowed returns once its work ran"
          test_borrowed_host;
        test "a bigarray a device borrowed lives until its work ran"
          test_borrowed_bigarray;
        test "host memory an idle device borrowed returns to an allocation"
          test_idle_borrower;
        test "a closed device's cache returns to its driver" test_closed_cache;
        test "a mapper closed beside the free of memory it maps raises nothing"
          test_mapper_beside_free;
      ];
  ]

let () = exit (run "rig.memory" tests)
