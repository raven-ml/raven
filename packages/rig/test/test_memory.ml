(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reclamation: how collected memory waits, enters the cache and returns, what a
   budget counts, how buffers pace the collector, and what a refused allocation
   does. Polled logs what its driver allocates and frees. *)

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
let mib = kib * kib

let out_of_memory d n = function
  | Rig.Out_of_memory (d', n') -> Rig.equal d d' && n = n'
  | _ -> false

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* A buffer of [n] bytes on [d] that is unreachable once this returns: its
   address. *)
let[@inline never] dropped ?memory d n = Support.at (B.create ?memory d n)

(* Collects, then drains [d], as an allocation on it does first: of no bytes,
   which no budget refuses. *)
let collect d =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create d 0))

(* A buffer of [n] bytes on [d] that the submission [s], of one write, writes
   once, and that is unreachable once this returns: its address. Its return
   waits for that work. *)
let[@inline never] dropped_written ?memory s d n =
  let b = B.create ?memory d n in
  ignore (Rig.submit s ~buffers:[| b |] ~args:one);
  Support.at b

(* How many times [p]'s driver freed the region at [at]. *)
let frees p at = List.length (List.filter (fun (a, _) -> a = at) (P.frees p))

(* Collects, then frees the cache of each of [ds], which first drains it: a
   closed device's memory collected after its stop returns so. *)
let settle ds =
  Gc.full_major ();
  List.iter Rig.free_cache ds

(* Runs [f] with the host's budget [n], then restores it. *)
let with_host_budget n f =
  let before = Rig.budget Rig.host in
  Rig.set_budget Rig.host n;
  Fun.protect ~finally:(fun () -> Rig.set_budget Rig.host before) f

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

(* The host's three kinds of buffer, by where their bytes start (Buffer.create):
   under a line of 128 bytes, on a line, and on a page from [paged_from], which
   the host keeps for reuse; and the bounds between them. *)
let line = 128
let paged = B.paged_from

let host_sizes =
  Gen.with_pp
    (fun ppf n -> Format.fprintf ppf "%d bytes" n)
    (Gen.one_of
       [
         Gen.of_list
           [ 0; 1; line - 1; line; line + 1; paged - 1; paged; paged + 1 ];
         Gen.int_range 0 (line - 1);
         Gen.int_range line (paged - 1);
         Gen.int_range paged (paged + (4 * line));
       ])

let cover_sizes n =
  cover ~share:0.1 "under a line" (n < line);
  cover ~share:0.1 "on a line" (line <= n && n < paged);
  cover ~share:0.1 "kept" (paged <= n)

(* Collection *)

(* An array {!B.export} made keeps its buffer's memory, which counts in the
   host's allocated bytes while the array is reachable, once its buffer is
   collected too, and leaves once the array is collected. *)
let test_host_exported () =
  let n = 512 * kib and k = 64 in
  let settle () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create Rig.host 0);
    Gc.full_major ();
    Rig.allocated Rig.host
  in
  let before = settle () in
  let arrays =
    ref (Array.init k (fun _ -> B.export (B.create Rig.host n) Bigarray.char))
  in
  at_least ~msg:"with the arrays, their buffers collected" int
    ~than:(before + (k * n))
    (settle ());
  ignore (Sys.opaque_identity !arrays);
  arrays := [||];
  equal ~msg:"once the arrays are collected" int before (settle ())

(* A collected buffer's memory serves the next buffer of the buffer's size,
   whichever array over it is collected last: a float32 view that outlives its
   buffer keeps the memory from reuse, and once it is collected too, the next
   buffer of the buffer's size takes it. *)
let test_host_keeps_size () =
  let n = mib in
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

(* An exported array keeps its memory alive while it is reachable
   ([Rig.Buffer.export]): the buffer collected, a buffer of its size made and
   written after it leaves the array's bytes as they were, and once the array is
   collected a buffer of the size holds what is written. *)
let fill c b = B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a c)
let chars a = String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a)

let export_law n =
  cover_sizes n;
  let array =
    ref
      (Some
         ((fun () ->
            let b = B.create Rig.host n in
            fill 'a' b;
            B.export b Bigarray.char)
            ()))
  in
  Gc.full_major ();
  Gc.full_major ();
  let other = B.create Rig.host n in
  fill 'b' other;
  equal ~msg:"the array's bytes" string (String.make n 'a')
    (chars (Option.get !array));
  array := None;
  Gc.full_major ();
  Gc.full_major ();
  let again = B.create Rig.host n in
  fill 'c' again;
  equal ~msg:"a buffer after the array" string (String.make n 'c')
    (B.read again Bigarray.char chars);
  ignore (Sys.opaque_identity other)

(* Waiting *)

(* Memory queued work reads returns to the driver over the budget only once that
   work ran: an over-budget return skips the cache, never the wait. *)
let test_over_budget_queued () =
  let d, p = P.open_ "memory:over-budget-queued" in
  let reads = Sub.make d [| Support.touching [| Sub.read |] |] in
  let[@inline never] [@local never] queued () =
    let b = B.create d 64 in
    ignore (submit ~args:one reads ~buffers:[| b |]);
    Support.at b
  in
  let at = queued () in
  Rig.set_budget d 0;
  collect d;
  equal ~msg:"frees before the read ran" int 0 (frees p at);
  ignore (P.run p);
  collect d;
  equal ~msg:"frees once it ran" int 1 (frees p at)

(* Memory another device used enters its owner's cache once that use is
   reached. *)
let test_foreign_use () =
  let a, pa = P.open_ "memory:owner" in
  let b, pb = P.open_ "memory:reader" in
  let at =
    (fun () ->
      let m = B.create a (4 * kib) in
      let s = Sub.make b [| Support.touching [| Sub.read |] |] in
      ignore (submit ~args:one s ~buffers:[| require_some (B.borrow b m) |]);
      Support.at m)
      ()
  in
  let cached () =
    collect a;
    Rig.free_cache a;
    collect a;
    frees pa at
  in
  equal ~msg:"frees while the read is unreached" int 0 (cached ());
  ignore (P.run pb);
  equal ~msg:"frees once it is reached" int 1 (cached ())

(* Host memory a device borrowed returns only once that device's work on it is
   done: until then no host buffer reuses it. *)
let test_borrowed_host () =
  let d, p = P.open_ "memory:borrowed-host" in
  let n = paged in
  let[@inline never] [@local never] written () =
    let h = B.create Rig.host n in
    let src = B.create d n in
    let dst = require_some (B.borrow d h) in
    ignore
      (submit (Sub.make d [| Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) |]));
    Support.at h
  in
  let at = written () in
  Gc.full_major ();
  Gc.full_major ();
  let other = B.create Rig.host n in
  not_equal ~msg:"while the work is queued" int at (Support.at other);
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
  let n = paged in
  let collected = Atomic.make 0 in
  let[@inline never] [@local never] written () =
    let ba = B.export (B.create Rig.host n) Bigarray.char in
    Gc.finalise (fun _ -> Atomic.incr collected) ba;
    let src = B.create d n in
    let dst = require_some (B.borrow d (B.of_bigarray ba)) in
    ignore
      (submit (Sub.make d [| Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) |]))
  in
  let settle () =
    Gc.full_major ();
    ignore (B.create Rig.host 0);
    Gc.full_major ();
    Gc.full_major ()
  in
  written ();
  settle ();
  equal ~msg:"collections while the work is queued" int 0 (Atomic.get collected);
  ignore (P.run p);
  settle ();
  equal ~msg:"collections once it ran" int 1 (Atomic.get collected)

(* Release *)

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

(* A closed device's cache returns its memory to the driver once the device
   stopped. Buffers of 40 sizes fill 40 keys of the cache, which grows. *)
let test_closed_cache () =
  let d, p = P.open_ "memory:closed-cache" in
  let sizes = List.init 40 (fun k -> (k + 1) * 64) in
  let[@inline never] [@local never] drop () =
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

(* A closed device's memory collected after its close returns to its driver at a
   call on any device: a program makes none on a device it closed. *)
let[@inline never] held_buffer d n held =
  let b = B.create d n in
  held := Some b;
  Support.at b

let test_closed_collected () =
  let d, p = P.open_ "memory:closed-collected" in
  let held = ref None in
  let at = held_buffer d (4 * kib) held in
  Rig.close d;
  equal ~msg:"frees at the close" int 0 (frees p at);
  held := None;
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create Rig.host 8));
  equal ~msg:"frees after a call on the host" int 1 (frees p at)

(* A drain of closed devices that raises in one device's free leaves the others'
   collected memory to the next call, which returns it. The first free either
   device starts raises. *)
let test_closed_drain_raises () =
  let d, p = P.open_ "memory:closed-raise-0" in
  let e, q = P.open_ "memory:closed-raise-1" in
  let held = ref None and held' = ref None in
  let at = held_buffer d (4 * kib) held in
  let at' = held_buffer e (4 * kib) held' in
  Rig.close d;
  Rig.close e;
  let raised = ref None in
  let raise_once name () =
    if Option.is_none !raised then begin
      raised := Some name;
      raise Exit
    end
  in
  P.before p "free" (raise_once "first");
  P.before q "free" (raise_once "second");
  held := None;
  held' := None;
  Gc.full_major ();
  raises ~msg:"the drain's free" Exit (fun () -> B.create Rig.host 8);
  ignore (Sys.opaque_identity (B.create Rig.host 8));
  let left = if !raised = Some "first" then frees q at' else frees p at in
  equal ~msg:"frees of the other device's memory" int 1 left

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
  let returned = frees p (Atomic.get at) in
  Mutex.unlock lock;
  Domain.join blocked;
  equal ~msg:"frees while its domain blocks" int 1 returned

(* Host memory borrowed on another domain returns through a host buffer's
   creation on this one, which first moves the host's collected memory. *)
let test_borrowed_host_elsewhere () =
  let d, p = P.open_ "memory:borrowed-host-elsewhere" in
  let n = paged in
  let written () =
    let h = B.create Rig.host n in
    let src = B.create d n in
    let dst = require_some (B.borrow d h) in
    ignore
      (submit (Sub.make d [| Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) |]));
    Support.at h
  in
  let at = Domain.join (Domain.spawn written) in
  ignore (P.run p);
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create Rig.host 0);
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"its memory, reused" int at (Support.at (B.create Rig.host n))

(* A keep collected after its domain ended lasts until a host buffer's creation
   on this one, which ends it: a take of its memory then succeeds. *)
let test_keep_elsewhere () =
  let a = B.create Rig.host 64 in
  let keep () = ignore (Sys.opaque_identity (B.keep [ a ])) in
  let taken () = Option.map List.length (B.take ~why:"taken" [ a ]) in
  Domain.join (Domain.spawn keep);
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"a take, collected, before a call that moves memory" (option int)
    None (taken ());
  ignore (Sys.opaque_identity (B.create Rig.host 8));
  equal ~msg:"a take after a host buffer's creation" (option int) (Some 1)
    (taken ())

(* rig.mli, Reclamation: "{!Buffer.create}, {!Buffer.of_string},
   {!Buffer.of_io}, {!Buffer.submit_copy} and {!Buffer.copy} first move the
   collected memory ... and let go of the values that collected submissions on
   those devices keep". A collected submission's kept value goes at the first
   such call on its device once its work is done, and no collection alone lets
   it go. An io device runs no submission, so [of_io] has no row. *)
type mover = Create | Of_string | Submit_copy | Copy

let mover_name = function
  | Create -> "create"
  | Of_string -> "of_string"
  | Submit_copy -> "submit_copy"
  | Copy -> "copy"

let test_mover mover =
  let d, p = P.open_ ("memory:mover-" ^ mover_name mover) in
  let on = if mover = Of_string then Rig.host else d in
  let src = B.create Rig.host 8 and dst = B.create d 8 in
  let runs = Atomic.make 0 in
  let[@inline never] [@local never] kept () =
    let v = ref () in
    Gc.finalise (fun _ -> Atomic.incr runs) v;
    ignore (submit (Sub.make ~keep:v on [||]))
  in
  kept ();
  ignore (P.run p);
  Gc.full_major ();
  equal ~msg:"collections of the value before the call" int 0 (Atomic.get runs);
  (match mover with
  | Create -> ignore (Sys.opaque_identity (B.create d 8))
  | Of_string -> ignore (Sys.opaque_identity (B.of_string "x"))
  | Submit_copy -> Rig.Point.wait (B.submit_copy ~src ~dst [| B.row 8 |])
  | Copy -> B.copy ~src ~dst);
  Gc.full_major ();
  equal ~msg:"collections of the value after the call" int 1 (Atomic.get runs)

(* A drain that returns collected memory to the driver of a device another
   domain loses meanwhile: its first free waits at Polled's free gate while the
   loss is recorded. The drain's allocation raises the loss, the device stops at
   its close, and each memory is freed once. *)
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
  Support.await "the drain's first free at the gate" (fun () -> P.freers p = 1);
  P.fail p;
  (try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ());
  equal ~msg:"stops while the free waits" int 0 (P.stops p);
  P.open_frees p;
  equal ~msg:"the drain's allocation" string "lost: the submission failed"
    (Domain.join drainer);
  equal ~msg:"stops before the close" int 0 (P.stops p);
  Rig.close d;
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  equal ~msg:"stops" int 1 (P.stops p);
  equal ~msg:"frees of the collected memory" (list int) [ 1; 1; 1 ]
    (List.map (frees p) ats)

(* Two drains of one device judge a retiring buffer one at a time. The first
   judges a buffer over the budget whose work ran, which only a read of the word
   behind the transport shows; while it reads, a drain on another domain waits
   for it and frees nothing the first reads. The buffer goes back once. *)
let test_judged_once () =
  if not Support.sees_waits then skip ~reason:"no view of lock waits" ();
  let d, p = P.open_ ~transport:true "memory:judged-once" in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  let at = dropped_written s d (4 * kib) in
  collect d;
  Rig.set_budget d kib;
  ignore (P.run p);
  let judging = Atomic.make false and second = Atomic.make None in
  let drained = Atomic.make false and freed_inside = ref (-1) in
  let other =
    Domain.spawn (fun () ->
        Support.await "the first drain's read" (fun () -> Atomic.get judging);
        Atomic.set second (Some (Support.self ()));
        ignore (Sys.opaque_identity (B.create d 0));
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
      freed_inside := frees p at);
  collect d;
  Domain.join other;
  equal ~msg:"frees while the first drain read it" int 0 !freed_inside;
  equal ~msg:"frees of the buffer" int 1 (frees p at);
  equal ~msg:"bytes held" int 0 (P.allocated p B.Device)

(* A drain that judges collected memory whose work is still queued allocates no
   more than one with nothing to judge: [B.create d 0] drains [d]. *)
let test_due_words () =
  let d, _ = P.open_ "memory:due-words" in
  let drains () =
    ignore (B.create d 0);
    Support.allocated_bytes (fun () ->
        for _ = 1 to 100 do
          ignore (B.create d 0)
        done)
  in
  let idle = drains () in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  let[@inline never] [@local never] queued () =
    ignore (submit ~args:one s ~buffers:[| B.create d 64 |])
  in
  queued ();
  Gc.full_major ();
  equal ~msg:"bytes with collected memory queued" int idle (drains ())

(* A drain that caches collected buffers whose work is done, into a cache that
   held as many of their size before their creates took them, allocates no more
   than a drain with nothing to cache: [B.create d 0] drains [d]. *)
let test_drain_words () =
  let d, _ = P.open_ "memory:drain-words" in
  let[@inline never] [@local never] dropped n =
    for _ = 1 to n do
      ignore (Sys.opaque_identity (B.create d 4096))
    done
  in
  let drain n =
    dropped n;
    Gc.full_major ();
    Support.allocated_bytes (fun () -> ignore (B.create d 0))
  in
  ignore (drain 100);
  let idle = drain 0 in
  equal ~msg:"bytes of a drain of 100 buffers" int idle (drain 100)

(* Memory goes back to its driver after every other device's mapping of it,
   whichever ends first. The mapper's close gives back its mapping in another
   domain, held at Polled's free gate while the memory is collected and its
   owner's cache emptied. *)
let test_free_after_maps () =
  let o, po = P.open_ "memory:mapped-owner" in
  let d, pd = P.open_ "memory:mapper" in
  let[@inline never] [@local never] mapped () =
    let m = B.create o (4 * kib) in
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    let buffers = [| Option.get (B.borrow d m) |] in
    Rig.Point.wait (submit ~args:one s ~buffers);
    (Support.at m, ref (Some m))
  in
  let at, held = mapped () in
  mem ~msg:"the mapper's calls" string "map_peer" (P.log pd);
  P.gate_frees pd;
  let mapper = Domain.spawn (fun () -> Rig.close d) in
  Support.await "the mapper's unmap at the gate" (fun () -> P.freers pd = 1);
  held := None;
  collect o;
  Rig.set_budget o 0;
  equal ~msg:"frees while it is mapped" int 0 (frees po at);
  P.open_frees pd;
  Domain.join mapper;
  settle [ o; d ];
  equal ~msg:"frees once unmapped" int 1 (frees po at);
  equal ~msg:"the mapper's regions" (list int) [] (P.outstanding pd)

(* Memory goes back to its driver once, when its last mapping goes after its
   device was lost and before its stop returned. The mapper's work keeps the
   mapping until the memory is freed; then, while a free of other memory waits
   at Polled's free gate in another domain, the owner is lost and the mapper's
   drain gives the mapping back. The owner's close then stops it. *)
let test_give_beside_stop () =
  let o, po = P.open_ "memory:give-owner" in
  let c, pc = P.open_ "memory:give-mapper" in
  let[@inline never] [@local never] mapped () =
    let m = B.create o (4 * kib) in
    let s = Sub.make c [| Support.touching [| Sub.read |] |] in
    let buffers = [| Option.get (B.borrow c m) |] in
    Rig.Point.wait (submit ~args:one s ~buffers);
    (Support.at m, ref (Some m))
  in
  let at, held = mapped () in
  ignore (submit (Sub.make c [||]) : Rig.Point.t);
  held := None;
  collect o;
  Rig.set_budget o 0;
  equal ~msg:"frees of the memory while it is mapped" int 0 (frees po at);
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
  Rig.close o;
  settle [ o; c ];
  equal ~msg:"the other memory's frees" int 1 (frees po other);
  equal ~msg:"the mapped memory's frees" int 1 (frees po at)

(* Two borrows on one device that race to map the same memory leave one mapping,
   and the memory goes back to its driver after the other is given back. The
   second borrow runs inside the first one's map, at Polled's probe of it. *)
let test_raced_maps () =
  let o, po = P.open_ "memory:raced-owner" in
  let c, pc = P.open_ "memory:raced-mapper" in
  let[@inline never] [@local never] mapped () =
    let m = B.create o (4 * kib) in
    P.before pc "map_peer" (fun () -> ignore (Option.get (B.borrow c m)));
    ignore (Option.get (B.borrow c m));
    (Support.at m, ref (Some m))
  in
  let at, held = mapped () in
  equal ~msg:"the maps, raced" int 2
    (List.length (List.filter (( = ) "map_peer") (P.log pc)));
  held := None;
  collect o;
  Rig.set_budget o 0;
  equal ~msg:"frees of the memory" int 1 (frees po at);
  equal ~msg:"the mapper's regions once it went back" (list int) []
    (P.outstanding pc)

(* A device closed while a borrow on another domain maps memory for it gives the
   mapping back, and the memory goes back to its driver once. The borrow's
   locate of the mapping waits at Polled's locate gate while the close runs. *)
let test_close_beside_borrow () =
  let o, po = P.open_ "memory:borrow-owner" in
  let d, pd = P.open_ "memory:borrow-mapper" in
  let[@inline never] [@local never] made () =
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
  Support.await "the borrow's locate at the gate" (fun () -> P.locaters pd = 1);
  Rig.close d;
  settle [ d ];
  P.open_locates pd;
  equal ~msg:"the borrow" string "lost" (Domain.join borrower);
  held := None;
  collect o;
  Rig.set_budget o 0;
  settle [ d; o ];
  equal ~msg:"the mapper's regions" (list int) [] (P.outstanding pd);
  equal ~msg:"the memory's frees" int 1 (frees po at)

(* Memory two devices map is freed in another domain, whose give-back of one
   mapping waits at that mapper's free gate while it holds the other's mapping
   too. Both gates close first and the other's opens once one holds, so the test
   needs no order of the two. The other mapper meanwhile closes: nothing raises
   its loss, every region goes back, and the memory is freed once. *)
let test_mapper_beside_free () =
  let o, po = P.open_ "memory:outlived-owner" in
  let a, pa = P.open_ "memory:outlived-a" in
  let b, pb = P.open_ "memory:outlived-b" in
  let map d m =
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    Rig.Point.wait (submit ~args:one s ~buffers:[| Option.get (B.borrow d m) |])
  in
  let[@inline never] [@local never] mapped () =
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
  mem ~msg:"the held mapping's calls" string "unmap" (P.log gated);
  equal ~msg:"the memory's frees" int 1 (frees po at)

(* Budgets *)

(* A host buffer counts its whole words of the C library's memory, at least 16
   bytes where it is not empty, and up to 112 more where it starts on 128 bytes;
   on Windows one that starts on a page, from [paged_from], counts a page of 4
   KiB less 16 more. Once collected it counts nothing: the host keeps a large
   one for reuse, which {!Rig.allocated} leaves out. *)
let host_slack n = if Sys.win32 && n >= paged then (4 * kib) - 16 else 112

let whole n =
  if n = 0 then 0 else Int.max 16 ((n + Rig.word - 1) / Rig.word * Rig.word)

let held_law n =
  cover_sizes n;
  Gc.full_major ();
  let before = Rig.allocated Rig.host in
  let held =
    (fun () ->
      let b = B.create Rig.host n in
      let held = Rig.allocated Rig.host in
      ignore (Sys.opaque_identity b);
      held)
      ()
  in
  at_least ~msg:"live" int ~than:(before + whole n) held;
  at_most ~msg:"live" int ~than:(before + whole n + host_slack n) held;
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"collected" int before (Rig.allocated Rig.host)

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

(* Mapped memory counts in its device's budget: a buffer of it adds its bytes to
   the device's allocated bytes, and one past the budget raises Out_of_memory,
   asking the driver nothing. *)
let test_mapped_budget () =
  let d, _ = P.open_ ~host_visible:false "memory:mapped-budget" in
  let before = Rig.allocated d in
  let b = B.create ~memory:Mapped d (4 * kib) in
  equal ~msg:"the device's allocated bytes" int
    (before + (4 * kib))
    (Rig.allocated d);
  ignore (Sys.opaque_identity b);
  let d', p' = P.open_ ~host_visible:false "memory:mapped-over" in
  Rig.set_budget d' (4 * kib);
  raises_match
    (out_of_memory d' (8 * kib))
    (fun () -> B.create ~memory:Mapped d' (8 * kib));
  equal ~msg:"asked of the driver" allocs [] (P.allocs p')

(* A budget counts each buffer by its whole words of 4 bytes ({!Rig.word}): a
   5-byte buffer holds 8 bytes, 5 bytes do not fit in 7, and 9 bytes of Mapped
   memory, 12 in words, do not fit in a budget of 10. *)
let test_budget_words () =
  let d, p = P.open_ ~host_visible:false "memory:budget-words" in
  let before = Rig.allocated d in
  let b = B.create d 5 in
  equal ~msg:"a 5-byte buffer" int (before + 8) (Rig.allocated d);
  equal allocs [ (B.Device, 8, true) ] (P.allocs p);
  ignore (Sys.opaque_identity b);
  let d', _ = P.open_ ~host_visible:false "memory:budget-words-7" in
  Rig.set_budget d' 7;
  raises_match (out_of_memory d' 5) (fun () -> B.create d' 5);
  let d'', p'' = P.open_ ~host_visible:false "memory:budget-words-10" in
  Rig.set_budget d'' 10;
  raises_match (out_of_memory d'' 9) (fun () -> B.create ~memory:Mapped d'' 9);
  equal ~msg:"asked of the driver" allocs [] (P.allocs p'')

(* Two allocations race for a budget's last room: one is held inside its
   driver's alloc, on another domain, while this domain asks for the same room.
   The held one took the room before it asked the driver, so this one is
   refused. A library that took the room only after the driver answered would
   let this one in: Polled's hook runs once, so it would not hang. A held
   allocation that returns without asking the driver fails the test with its
   outcome. *)
let per = 4 * kib

let test_last_room () =
  let d, p = P.open_ "memory:last-room" in
  Rig.set_budget d (2 * per);
  let first = B.create d per in
  let lock = Mutex.create () and moved = Condition.create () in
  let inside = ref false and go = ref false in
  P.before p "alloc" (fun () ->
      Mutex.protect lock (fun () ->
          inside := true;
          Condition.broadcast moved;
          while not !go do
            Condition.wait moved lock
          done));
  let returned = ref false in
  let held =
    Domain.spawn (fun () ->
        let outcome =
          match B.create d per with
          | b -> Ok (B.length b)
          | exception e -> Error (Printexc.to_string e)
        in
        Mutex.protect lock (fun () ->
            returned := true;
            Condition.broadcast moved);
        outcome)
  in
  let reached =
    Mutex.protect lock (fun () ->
        while not (!inside || !returned) do
          Condition.wait moved lock
        done;
        !inside)
  in
  if not reached then
    failf "the held allocation returned %s without asking the driver"
      (match Domain.join held with
      | Ok n -> Printf.sprintf "a buffer of %d bytes" n
      | Error e -> e);
  let release () =
    Mutex.protect lock (fun () ->
        go := true;
        Condition.broadcast moved)
  in
  Fun.protect ~finally:release (fun () ->
      raises_match ~msg:"the racing allocation" (out_of_memory d per) (fun () ->
          B.create d per));
  equal ~msg:"the held allocation" (result int string) (Ok per)
    (Domain.join held);
  ignore (Sys.opaque_identity first)

(* Pace *)

(* Dropped buffers of many budgets never force a collection: the collector is
   paced by the device's memory and finds them before the budget runs out.
   Beside a live heap of 64 MiB, the runtime's own pace, [custom_major_ratio] /
   150 of the heap, would bring a cycle once per about 19 MiB of buffers, more
   than four budgets: only a pace by the device's room collects in time. The
   pace holds once the collector runs. A fresh process's heap is small, and
   OCaml's runtime starts marking only after [small_heap_limit] words of work
   since its last sweep, while a buffer brings at most one cycle's work of that
   heap: the first budget may fill before the first cycle marks, and a ladder's
   round collects. One uncounted run takes that start. *)
let forced () = (Gc.quick_stat ()).forced_major_collections

let unforced ?heap d =
  let budget = 4 * mib in
  Rig.set_budget d budget;
  let run () =
    for _ = 1 to 20 * 16 do
      ignore (Sys.opaque_identity (B.create d (budget / 16)))
    done
  in
  run ();
  let before = forced () in
  run ();
  ignore (Sys.opaque_identity heap);
  equal ~msg:"forced collections" int 0 (forced () - before)

let test_paced () =
  let heap = Array.make (8 * mib) 0. in
  unforced ~heap (fst (P.open_ "memory:paced"))

(* A software device nearing its budget still collects in time: dropped buffers
   of 20 budgets of 4 MiB, a budget no larger than the floor of host memory's
   pace, never force a collection. One uncounted run takes a fresh heap's start,
   as in [test_paced]. *)
let test_paced_memory_budget () =
  unforced (Result.get_ok (Rig.software "memory:paced-budget"))

(* A device whose memory the host does not address paces at its room alone:
   dropped buffers of 20 budgets of 4 MiB never force a collection. *)
let test_paced_hidden () =
  unforced (fst (P.open_ ~host_visible:false "memory:paced-hidden"))

(* Major cycles over [k] dropped buffers of a MiB on [d], the host by default,
   each held across a minor collection that promotes it, as a buffer that
   outlives the minor heap is. *)
let promoted_cycles ?(d = Rig.host) k =
  let before = (Gc.quick_stat ()).major_collections in
  for _ = 1 to k do
    let b = B.create d mib in
    Gc.minor ();
    ignore (Sys.opaque_identity b)
  done;
  Float.of_int ((Gc.quick_stat ()).major_collections - before)

(* The least work, in words, a major cycle of OCaml 5.5's runtime does before it
   marks: [small_heap_limit]. *)
let small_heap_limit = 262_144

(* Major cycles per MiB over [k] dropped buffers of a MiB on [d], from a
   compacted heap beside [small_heap_limit] live words. The runtime estimates a
   cycle's work from the heap's size, as if [space_overhead] said its live
   share, and a cycle does the work of its live blocks only: a heap that earlier
   tests left mostly free ends cycles sooner. One of 2 million words, a 64th of
   them live, ended one per two buffers on the M1 Max. Compacted, the heap holds
   its live blocks alone. The live words keep the least work from dominating a
   cycle, which would hide a pace too fast. *)
let cycles_per_mib ?d k =
  let live = Array.make small_heap_limit 0 in
  Gc.compact ();
  let rate = promoted_cycles ?d k /. Float.of_int k in
  ignore (Sys.opaque_identity live);
  rate

(* The message of a count of [cycles_per_mib], with the heap it ran beside. *)
let rate_msg () =
  Printf.sprintf "cycles per MiB, beside a heap of %d words"
    (Gc.quick_stat ()).heap_words

(* The margin a cycle-per-bound assertion allows: a count over 256 buffers
   includes the cycles already under way at either end. *)
let margin = 1.5

(* Runs [f] with [custom_major_ratio] at 1, then restores it. At a ratio of 1
   the heap's own pace would bring a cycle per few buffers. *)
let at_ratio_one f =
  let was = (Gc.get ()).custom_major_ratio in
  Gc.set { (Gc.get ()) with custom_major_ratio = 1 };
  Fun.protect
    ~finally:(fun () -> Gc.set { (Gc.get ()) with custom_major_ratio = was })
    f

(* At a small heap, compacted, dropped host buffers bring a major cycle at most
   once per 4 MiB, the least that half the last-level cache is. *)
let test_paced_host () =
  let rate = cycles_per_mib 256 in
  at_most ~msg:(rate_msg ()) float_exact ~than:(margin /. 4.) rate

(* A software device's memory is the host's: with a budget that leaves it room,
   beside a compacted heap, its dropped buffers bring a major cycle at most once
   per half the host's last-level cache or 4 MiB, whichever is more, as host
   buffers do, whatever [custom_major_ratio] says of the heap. At a ratio of 1,
   the heap's and the device's share would bring a cycle per buffer. *)
let test_paced_software () =
  let d = Result.get_ok (Rig.software "memory:paced-floor") in
  let cache = Option.value (Rig.cache_bytes Rig.host) ~default:0 in
  let floor = Int.max (cache / 2) (4 * mib) in
  let rate = at_ratio_one (fun () -> cycles_per_mib ~d 256) in
  at_most ~msg:(rate_msg ()) float_exact
    ~than:(margin *. Float.of_int mib /. Float.of_int floor)
    rate

(* A device's cache is not live memory: dropped buffers of a software device
   bring major cycles beside 512 MiB of other sizes in its cache as often as
   beside none, give or take half again. Counted as live, the cache would raise
   the pace's bound by [custom_major_ratio] / 150 of its bytes, about 150 MiB at
   the default, and bring them several times less often, so that dropped results
   wait while new ones are allocated afresh. *)
let test_paced_beside_cache () =
  let d = Result.get_ok (Rig.software "memory:paced-cache") in
  let usual = cycles_per_mib ~d 256 in
  let held, beside =
    Fun.protect ~finally:(fun () -> Rig.free_cache d) @@ fun () ->
    for i = 1 to 64 do
      ignore (Sys.opaque_identity (B.create d ((8 * mib) + (i * 16 * kib))))
    done;
    Gc.full_major ();
    ignore (Sys.opaque_identity (B.create d 0));
    let beside = cycles_per_mib ~d 256 in
    (Rig.allocated d, beside)
  in
  at_least ~msg:"the bytes free_cache returned" int ~than:(512 * mib)
    (held - Rig.allocated d);
  at_least ~msg:"cycles per MiB beside the cache" float_exact
    ~than:(usual /. margin) beside

(* Small dropped host buffers that reach the major heap pace cycles as large
   ones do: at most once per 4 MiB, whatever [custom_major_ratio] says of the
   heap. 32 rounds of 64 buffers of 8 KiB, each round held across a minor
   collection that promotes it, drop 16 MiB, so at most 4 cycles and half again
   as margin. *)
let test_paced_host_small () =
  let cycles =
    at_ratio_one (fun () ->
        Gc.full_major ();
        let before = (Gc.quick_stat ()).major_collections in
        for _ = 1 to 32 do
          let held = Array.init 64 (fun _ -> B.create Rig.host (8 * kib)) in
          Gc.minor ();
          ignore (Sys.opaque_identity held)
        done;
        (Gc.quick_stat ()).major_collections - before)
  in
  at_most ~msg:"cycles over 16 MiB of 8 KiB buffers" float_exact
    ~than:(margin *. 4.) (Float.of_int cycles)

(* A larger minor heap leaves the major pace as it was: promoted dropped buffers
   of a MiB bring cycles at a minor heap of 64 MiB, past half the last-level
   cache, as often as at the default, give or take half again. A bound raised to
   the minor heap would bring them several times less often. *)
let test_paced_minor_heap () =
  let cycles () =
    Gc.full_major ();
    promoted_cycles 256
  in
  let usual = cycles () in
  let was = (Gc.get ()).minor_heap_size in
  Gc.set { (Gc.get ()) with minor_heap_size = 8 * mib };
  let raised =
    Fun.protect
      ~finally:(fun () -> Gc.set { (Gc.get ()) with minor_heap_size = was })
      cycles
  in
  at_least ~msg:"cycles over 256 MiB at a minor heap of 64 MiB" float_exact
    ~than:(usual /. margin) raised

(* A refused allocation leaves the pace as it was: its bytes were never a
   buffer's, so they count in no bound, and dropped buffers bring cycles after
   it as often as before it. *)
let test_paced_refused () =
  Gc.full_major ();
  let usual = promoted_cycles 256 in
  Gc.full_major ();
  let refused = 1 lsl 60 in
  raises_match (out_of_memory Rig.host refused) (fun () ->
      B.create Rig.host refused);
  at_least ~msg:"cycles over 256 MiB dropped after the refusal" float_exact
    ~than:(usual /. margin) (promoted_cycles 256)

(* The collector's share of the heap's bytes and [live] more:
   [custom_major_ratio] / 150 of them. *)
let share live =
  let words = (Gc.quick_stat ()).heap_words in
  let bytes = Float.of_int ((words * (Sys.word_size / 8)) + live) in
  bytes *. Float.of_int (Gc.get ()).custom_major_ratio /. 150.

(* Live host buffers raise the bound as heap bytes do: beside 256 MiB of them
   and a compacted heap, dropped buffers bring a cycle at most once per
   [custom_major_ratio] / 150 of the heap's and their bytes, about 75 MiB.
   Without the live term, cycles would come once per half the last-level cache,
   more often on any host whose last level is under about 100 MiB. *)
let test_paced_host_live () =
  let held = 256 * mib in
  let live = B.create Rig.host held in
  let rate = cycles_per_mib 256 in
  let share = share held in
  ignore (Sys.opaque_identity live);
  at_most ~msg:(rate_msg ()) float_exact
    ~than:(margin *. Float.of_int mib /. share)
    rate

(* At a large heap, compacted, cycles come at most once per the collector's
   share of the heap, [custom_major_ratio] / 150 of its bytes. *)
let test_paced_host_heap () =
  let heap = Array.make (16 * mib) 0. in
  let rate = cycles_per_mib 256 in
  let share = share 0 in
  ignore (Sys.opaque_identity heap);
  at_most ~msg:(rate_msg ()) float_exact
    ~than:(margin *. Float.of_int mib /. share)
    rate

(* The bound of the major pace at the current heap, with [live] bytes of
   buffers: the collector's share of both, at least the floor of half the
   last-level cache. *)
let pace_bound live =
  let cache = Option.value (Rig.cache_bytes Rig.host) ~default:0 in
  let floor = Int.max (cache / 2) (4 * mib) in
  Float.max (Float.of_int floor) (share live)

(* Buffers of a MiB whose bytes are 8 [bound]s, and at least 256: a test that
   asks for a cycle per bound counts the cycles that end, so its drops span
   several bounds. The floor of a host whose last-level cache is large is large:
   half of the 480 MiB that a CI runner's processor states. *)
let mibs_over bound =
  Int.max 256 (Float.to_int (Float.ceil (8. *. bound /. Float.of_int mib)))

(* A large dropped host buffer the minor heap still holds returns at a minor
   collection and brings an eighth of a major cycle's work: beside 64 MiB live,
   256 dropped buffers of a MiB bring cycles at about an eighth of their bytes
   over the bound, and the promotion of the buffer that crosses each bound; a
   buffer in the major heap from birth would bring eight times that. The bytes
   held of them stay within the bound. *)
let test_paced_host_young () =
  let heap = Array.make (8 * mib) 0. in
  Gc.full_major ();
  let bound = pace_bound 0 in
  let mib' = Float.of_int mib in
  let made = 256. *. mib' in
  let base = Rig.allocated Rig.host in
  let before = (Gc.quick_stat ()).major_collections in
  let peak = ref 0 in
  for _ = 1 to 256 do
    ignore (Sys.opaque_identity (B.create Rig.host mib));
    peak := Int.max !peak (Rig.allocated Rig.host - base)
  done;
  let cycles = (Gc.quick_stat ()).major_collections - before in
  ignore (Sys.opaque_identity heap);
  at_most ~msg:"major cycles over 256 MiB dropped young" float_exact
    ~than:((margin *. (made /. bound *. ((1. /. 8.) +. (mib' /. bound)))) +. 2.)
    (Float.of_int cycles);
  at_most ~msg:"bytes of dropped buffers held at once" float_exact
    ~than:((margin *. bound) +. (2. *. mib'))
    (Float.of_int !peak)

(* A [custom_minor_max_size] of 0 keeps every buffer out of the minor heap: a
   buffer of a MiB then brings its whole bytes to the major pace at birth, a
   cycle per bound's bytes at least, give or take a factor of two. *)
let test_paced_host_major () =
  let heap = Array.make (8 * mib) 0. in
  let was = (Gc.get ()).custom_minor_max_size in
  Gc.set { (Gc.get ()) with custom_minor_max_size = 0 };
  let cycles, k, bound =
    Fun.protect
      ~finally:(fun () ->
        Gc.set { (Gc.get ()) with custom_minor_max_size = was })
      (fun () ->
        Gc.full_major ();
        let bound = pace_bound 0 in
        let k = mibs_over bound in
        let before = (Gc.quick_stat ()).major_collections in
        for _ = 1 to k do
          ignore (Sys.opaque_identity (B.create Rig.host mib))
        done;
        (Float.of_int ((Gc.quick_stat ()).major_collections - before), k, bound))
  in
  ignore (Sys.opaque_identity heap);
  at_least
    ~msg:
      (Printf.sprintf "major cycles over %d MiB in the major heap from birth" k)
    float_exact
    ~than:(Float.of_int (k * mib) /. bound /. 2.)
    cycles

(* A large host buffer that the minor heap promotes still counts in full: beside
   64 MiB live, promoted dropped buffers of a MiB bring at least one cycle per
   bound's bytes, give or take a factor of two. Counting less of them than their
   bytes would bring cycles later or never, and hold more memory unreturned. *)
let test_paced_host_promoted () =
  let heap = Array.make (8 * mib) 0. in
  Gc.full_major ();
  let bound = pace_bound 0 in
  let k = mibs_over bound in
  let cycles = promoted_cycles k in
  ignore (Sys.opaque_identity heap);
  at_least
    ~msg:(Printf.sprintf "major cycles over %d MiB promoted and dropped" k)
    float_exact
    ~than:(Float.of_int (k * mib) /. bound /. 2.)
    cycles

(* A host buffer found live by a cycle and dropped afterwards returns although
   what follows dies young: nothing but a major cycle returns it, and each young
   buffer brings an eighth of its share of one, so churn past eight bounds,
   twice over, brings it back. *)
let test_paced_host_returns () =
  let n = 64 * mib in
  Gc.full_major ();
  let before = Rig.allocated Rig.host in
  let[@inline never] [@local never] hold () =
    let b = B.create Rig.host n in
    Gc.full_major ();
    ignore (Sys.opaque_identity b)
  in
  hold ();
  let churn = 4 * 8 * int_of_float (pace_bound n) / mib in
  for _ = 1 to churn do
    ignore (Sys.opaque_identity (B.create Rig.host mib))
  done;
  Gc.minor ();
  at_most ~msg:"host bytes held after churn past the dropped buffer" int
    ~than:(before + (n / 2))
    (Rig.allocated Rig.host)

(* The runtime's parameters ([Gc.control]) that the pace reads or that a library
   could tune, by name. *)
let parameters () =
  let c = Gc.get () in
  [
    ("minor_heap_size", c.minor_heap_size);
    ("space_overhead", c.space_overhead);
    ("max_overhead", c.max_overhead);
    ("stack_limit", c.stack_limit);
    ("allocation_policy", c.allocation_policy);
    ("window_size", c.window_size);
    ("custom_major_ratio", c.custom_major_ratio);
    ("custom_minor_ratio", c.custom_minor_ratio);
    ("custom_minor_max_size", c.custom_minor_max_size);
  ]

(* Opening devices, making buffers of each kind of host buffer and of a device,
   and returning them leave the runtime's parameters as they were. *)
let test_sets_none () =
  let before = parameters () in
  let d, _ = P.open_ ~host_visible:false "memory:sets-none" in
  let s = Result.get_ok (Rig.software "memory:sets-none-software") in
  List.iter
    (fun n -> ignore (Sys.opaque_identity (B.create Rig.host n)))
    [ 1; line; paged; 4 * mib ];
  ignore (Sys.opaque_identity (B.create ~memory:Pinned d (4 * kib)));
  ignore (Sys.opaque_identity (B.create s mib));
  collect Rig.host;
  List.iter Rig.free_cache [ Rig.host; d; s ];
  equal (list (pair string int)) before (parameters ())

(* Kept host buffers *)

(* The host keeps a collected buffer of [Rig.Buffer.paged_from] bytes or more
   for the next buffer of its size, unless a bigarray of it lives, which keeps
   its bytes. The test starts from an empty cache, with the buffers of earlier
   tests collected: one collected during it would enter the cache after its
   buffer, and at a cycle's end could push it out as the least recently kept. *)
let test_host_cache () =
  Gc.full_major ();
  Gc.full_major ();
  Rig.free_cache Rig.host;
  let n = paged + 4093 in
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

(* Refusals *)

(* An allocation over the budget raises at once: the cache stays. *)
let test_over_budget () =
  let d, p = P.open_ "memory:over-budget" in
  Rig.set_budget d (8 * kib);
  let at = dropped d (4 * kib) in
  collect d;
  raises_match
    (out_of_memory d ((8 * kib) + 1))
    (fun () -> B.create d ((8 * kib) + 1));
  equal ~msg:"frees of the cached memory" int 0 (frees p at);
  at_least int ~than:(4 * kib) (P.allocated p B.Device)

(* A host buffer its budget refuses raises Out_of_memory for its own bytes: the
   112 more it counts where it starts on 128 bytes are not the allocation's. *)
let test_host_budget_refused () =
  Gc.full_major ();
  Gc.full_major ();
  let n = 1000 in
  with_host_budget
    (Rig.allocated Rig.host + n)
    (fun () ->
      raises_match (out_of_memory Rig.host n) (fun () -> B.create Rig.host n))

(* An allocation the driver refuses releases the cache, then raises. *)
let test_refused () =
  let d, p = P.open_ ~memory:(8 * kib) "memory:refused" in
  let live = B.create d (4 * kib) in
  let at = dropped d (4 * kib) in
  collect d;
  raises_match (out_of_memory d (8 * kib)) (fun () -> B.create d (8 * kib));
  equal ~msg:"frees of the cached memory" int 1 (frees p at);
  equal ~msg:"frees of the live memory" int 0 (frees p (Support.at live))

(* Memory collected over the budget while queued work reads it, its return to
   the driver waiting for that work. *)
let queued_over_budget name =
  let d, p = P.open_ name in
  let reads = Sub.make d [| Support.touching [| Sub.read |] |] in
  let[@inline never] [@local never] queued () =
    let b = B.create d (64 * kib) in
    ignore (submit ~args:one reads ~buffers:[| b |]);
    Support.at b
  in
  let at = queued () in
  Rig.set_budget d (32 * kib);
  collect d;
  (d, p, at)

(* An allocation the budget refuses waits for the work that holds back collected
   memory's return, then reuses its room. *)
let test_refused_waits () =
  let d, p, at = queued_over_budget "memory:refused-waits" in
  equal int (32 * kib) (B.length (B.create d (32 * kib)));
  equal ~msg:"frees of the queued memory" int 1 (frees p at)

(* The wait finds the loss of a device that faulted with that work queued. *)
let test_refused_lost () =
  let d, p, _ = queued_over_budget "memory:refused-lost" in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> B.create d (32 * kib))

(* A reclamation round waits for a device whose lock another thread holds: the
   device's cached memory charged to the host's budget returns to a host
   allocation the budget refused. The other thread holds the lock while its
   drain reads the device's word behind the transport, until the round waits for
   the lock. *)
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
    with_host_budget (Rig.allocated Rig.host) (fun () ->
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
  equal ~msg:"frees of the cached memory" int 1 (frees p at)

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
        | _ -> "allocated"
        | exception Sys.Break -> "broke")
  in
  equal ~msg:"the round's wait" string "broke" broke;
  ignore (P.run p);
  Rig.free_cache d;
  equal ~msg:"frees of the cached memory" int 1 (frees p at);
  equal ~msg:"bytes held" int (4 * kib) (Rig.allocated d);
  ignore (Sys.opaque_identity live)

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

(* A buffer that only a collected submission's keep holds returns to an
   allocation its budget refuses: the second round's collection finds the
   submission, the drain after it lets go of the keep, and the third round
   collects again. *)
let test_refused_kept () =
  let d, p = P.open_ "memory:refused-kept" in
  Rig.set_budget d (4 * kib);
  let[@inline never] [@local never] kept () =
    let b = B.create d (4 * kib) in
    ignore (submit (Sub.make ~keep:b d [||]))
  in
  kept ();
  ignore (P.run p);
  equal int (4 * kib) (B.length (B.create d (4 * kib)))

(* A refusal that no collection can answer collects once when no finaliser runs
   and rig lets go of nothing meanwhile. *)
let test_refused_collects_once () =
  let d, _ = P.open_ "memory:refused-once" in
  Rig.set_budget d (4 * kib);
  let live = B.create d (4 * kib) in
  Gc.full_major ();
  let before = forced () in
  raises_match (out_of_memory d (4 * kib)) (fun () -> B.create d (4 * kib));
  equal ~msg:"collections" int 1 (forced () - before);
  ignore (Sys.opaque_identity live)

(* Pinned memory charged to the host's budget that another device keeps in its
   cache returns when a pinned allocation the host's budget refuses reclaims:
   the allocation is made. *)
let test_pinned_reclaims () =
  let a, _ = P.open_ ~host_visible:false "memory:pinned-asker" in
  let b, _ = P.open_ ~host_visible:false "memory:pinned-keeper" in
  let n = 64 * kib in
  ignore (dropped ~memory:Pinned b n);
  collect b;
  with_host_budget (Rig.allocated Rig.host) (fun () ->
      equal int n (B.length (B.create ~memory:Pinned a n)))

(* A host buffer a device borrowed, dropped, returns its bytes when a pinned
   allocation the host's budget refuses reclaims: the reclaim drains the host,
   whose list holds the buffer, and the allocation is made. *)
let test_borrowed_reclaimed () =
  let d, _ = P.open_ ~host_visible:false "memory:borrowed-reclaim" in
  let n = 64 * kib in
  let[@inline never] [@local never] borrowed () =
    ignore (Sys.opaque_identity (B.borrow d (B.create Rig.host n)))
  in
  borrowed ();
  with_host_budget (Rig.allocated Rig.host) (fun () ->
      equal int n (B.length (B.create ~memory:Pinned d n)))

(* Host memory an idle device borrowed returns to a host allocation that needs
   its room once the device's work ran, though nothing waited for it. *)
let test_idle_borrower () =
  let d, p = P.open_ "memory:idle-borrower" in
  let n = mib in
  let[@inline never] [@local never] written () =
    let h = B.create Rig.host n in
    let src = B.create d n in
    let dst = require_some (B.borrow d h) in
    ignore
      (submit (Sub.make d [| Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) |]))
  in
  Gc.full_major ();
  Rig.free_cache Rig.host;
  written ();
  ignore (P.run p);
  let room = Rig.allocated Rig.host + (n / 2) in
  with_host_budget room (fun () -> equal int n (B.length (B.create Rig.host n)))

(* A host allocation the C library refuses raises Out_of_memory for the host,
   and gives its bytes back to the host's budget: a later buffer under a budget
   below the refused size, beyond what the host holds, is made. *)
let test_host_refused () =
  let n = 1 lsl 60 in
  raises_match (out_of_memory Rig.host n) (fun () -> B.create Rig.host n);
  with_host_budget
    (Rig.allocated Rig.host + (64 * mib))
    (fun () -> equal int (32 * mib) (B.length (B.create Rig.host (32 * mib))))

(* A growth of a device's queues short of memory reclaims the host's budget too,
   which the device's pinned memory counts in: its cached pinned memory returns
   before the submit raises Out_of_memory. *)
let test_growth_reclaims_pinned () =
  let d, p =
    P.open_ ~host_visible:false ~capacity:4 ~grows:8 "memory:growth-pinned"
  in
  let at = dropped ~memory:Pinned d (64 * kib) in
  collect d;
  let s = Sub.make d (Array.make 9 (Support.touching [||])) in
  raises_match (out_of_memory d 4096) (fun () ->
      submit ~args:(Support.zeros 9) s);
  equal ~msg:"frees of its cached pinned memory" int 1 (frees p at)

(* The host's out-of-memory ladder waits for a device whose work holds back host
   memory, and a fault that wait finds is that device's loss, never the
   allocation's: the host allocation raises Out_of_memory for itself. *)
let test_ladder_beside_fault () =
  let d, p = P.open_ ~host_visible:false "memory:ladder-fault" in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  ignore (Sys.opaque_identity (dropped_written s ~memory:Pinned d (64 * kib)));
  collect Rig.host;
  P.fault p "the engine hung";
  let raised =
    with_host_budget 0 (fun () ->
        match B.create Rig.host (64 * kib) with
        | _ -> "allocated"
        | exception Rig.Out_of_memory (d', _) when Rig.equal d' Rig.host ->
            "out of memory"
        | exception e -> Printexc.to_string e)
  in
  equal ~msg:"the host allocation" string "out of memory" raised;
  equal ~msg:"the device" (option string) (Some "the engine hung") (Rig.lost d)

(* budget *)

(* rig.mli, budget: "It starts at [max_int] for {!host} and an io device". *)
let test_budget_max_int (_, d) = equal int max_int (Rig.budget (Lazy.force d))

(* A driver's device's budget starts at its driver's [budget]. *)
let test_budget_driver () =
  let d, _ = P.open_ ~budget:(64 * kib) "memory:budget-driver" in
  equal int (64 * kib) (Rig.budget d)

(* capacity *)

(* [d]'s capacity as (total, free). *)
let capacity d =
  Option.map (fun (c : Rig.capacity) -> (c.total, c.free)) (Rig.capacity d)

let reported = option (pair int int)

(* The host, a software device and an io device report none, their memory being
   the machine's, and so does a driver that states none. *)
let test_capacity_none (_, d) = equal reported None (capacity (Lazy.force d))

(* A driver's report at each call, which asks the driver: its total, and a free
   that its buffers and cache lower and that returns once rig returns the
   cache. *)
let test_capacity () =
  let d, p = P.open_ ~memory:(64 * kib) ~budget:(64 * kib) "memory:capacity" in
  equal ~msg:"at open" reported (Some (64 * kib, 64 * kib)) (capacity d);
  let b = B.create d (16 * kib) in
  equal ~msg:"a buffer held" reported (Some (64 * kib, 48 * kib)) (capacity d);
  satisfies ~msg:"its free" ~claim:"freed" bool Fun.id (B.free b);
  collect d;
  equal ~msg:"in the cache" reported (Some (64 * kib, 48 * kib)) (capacity d);
  Rig.free_cache d;
  equal ~msg:"returned" reported (Some (64 * kib, 64 * kib)) (capacity d);
  equal ~msg:"the driver's reports" int 4
    (List.length (List.filter (( = ) "capacity") (P.log p)))

(* A fault in the report loses the device, whose later reports raise its loss
   without asking the driver: rig_edge.mli has rig start no counted call on a
   lost device. *)
let test_capacity_fault () =
  let d, p = P.open_ ~memory:(64 * kib) "memory:capacity-fault" in
  P.fail_at p 1 (`Fault "no report");
  raises_match ~msg:"the faulting report" (lost d) (fun () -> Rig.capacity d);
  equal ~msg:"the device's loss" (option string) (Some "no report") (Rig.lost d);
  raises_match ~msg:"a report once lost" (lost d) (fun () -> Rig.capacity d);
  equal ~msg:"the driver's reports" int 1
    (List.length (List.filter (( = ) "capacity") (P.log p)))

(* charged *)

(* rig.mli, charged: "[d] for its [Device] and [Mapped] memory, and for [Pinned]
   memory where the host addresses [d]'s memory; {!host} for other [Pinned]
   memory". *)
type place = Host | Software | Addressed | Hidden

let charged_rows =
  [
    ("the host's Device memory", Host, B.Device, `Host);
    ("the host's Pinned memory", Host, B.Pinned, `Host);
    ("a software device's Device memory", Software, B.Device, `Itself);
    ("a software device's Pinned memory", Software, B.Pinned, `Itself);
    ("a software device's Mapped memory", Software, B.Mapped, `Itself);
    ("an addressed device's Pinned memory", Addressed, B.Pinned, `Itself);
    ("a hidden device's Device memory", Hidden, B.Device, `Itself);
    ("a hidden device's Mapped memory", Hidden, B.Mapped, `Itself);
    ("a hidden device's Pinned memory", Hidden, B.Pinned, `Host);
  ]

let test_charged (row, place, memory, expected) =
  let name =
    "memory:charged-" ^ String.map (function ' ' -> '-' | c -> c) row
  in
  let d =
    match place with
    | Host -> Rig.host
    | Software -> Result.get_ok (Rig.software name)
    | Addressed -> fst (P.open_ name)
    | Hidden -> fst (P.open_ ~host_visible:false name)
  in
  let expected = match expected with `Host -> Rig.host | `Itself -> d in
  equal string (Rig.name expected) (Rig.name (Rig.charged ~memory d))

(* allocated *)

(* Two domains allocate on one device under a budget the allocations of one
   program cannot pass: [allocated] counts each allocation once, as some order
   of the calls explains. A prefix and suffix make at most 10 calls and each
   branch 5, so at most 19 allocations. *)
type budgeted = {
  d : Rig.t;
  at_open : int;
  live : B.t list ref;
  lock : Mutex.t;
}

type budgeted_model = { mutable held : int }

let budgeted_opened = Atomic.make 0

let make_budgeted () =
  let n = Atomic.fetch_and_add budgeted_opened 1 in
  let d, _ = P.open_ (Printf.sprintf "memory:budgeted-%d" n) in
  Rig.set_budget d (20 * per);
  { d; at_open = Rig.allocated d; live = ref []; lock = Mutex.create () }

let alloc_budgeted t =
  let b = B.create t.d per in
  Mutex.protect t.lock (fun () -> t.live := b :: !(t.live))

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
    command "alloc"
      (budgeted ^-> returns unit)
      (fun r -> r.held <- r.held + 1)
      alloc_budgeted;
    command "allocated"
      (budgeted ^-> returns int)
      (fun r -> r.held * per)
      (fun t -> Rig.allocated t.d - t.at_open);
  ]

(* [d]'s allocated bytes once its collected memory returned to its driver. *)
let settled d =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (B.create d 0);
    Rig.free_cache d
  done;
  Rig.allocated d

let software =
  fixture (fun () -> Result.get_ok (Rig.software "memory:allocated"))

type held = On_host | Software_device | Software_mapped

let held_places =
  Gen.of_list
    ~pp:(fun ppf p ->
      Format.pp_print_string ppf
        (match p with
        | On_host -> "the host"
        | Software_device -> "a software device's Device memory"
        | Software_mapped -> "a software device's Mapped memory"))
    [ On_host; Software_device; Software_mapped ]

let allocated_sizes =
  Gen.with_pp
    (fun ppf n -> Format.fprintf ppf "%d bytes" n)
    (Gen.one_of
       [
         Gen.of_list
           [ 0; 1; 3; 4; 5; 127; 128; 129; 4095; 4096; paged - 1; paged; mib ];
         Gen.int_range 0 (2 * paged);
       ])

(* A buffer adds {!Rig.allocation}'s bytes to its device's allocated bytes. *)
let allocated_law (place, n) =
  cover ~share:0.1 "the host" (place = On_host);
  cover ~share:0.1 "Mapped memory" (place = Software_mapped);
  cover ~share:0.1 "under a line" (n < line);
  cover ~share:0.1 "a buffer the host keeps" (paged <= n);
  let d, memory =
    match place with
    | On_host -> (Rig.host, B.Device)
    | Software_device -> (software (), B.Device)
    | Software_mapped -> (software (), B.Mapped)
  in
  let before = settled d in
  let b = B.create ~memory d n in
  let added = settled d - before in
  ignore (Sys.opaque_identity b);
  equal int (Rig.allocation ~memory d n) added

let allocated_examples =
  List.concat_map
    (fun n -> [ (On_host, n); (Software_device, n) ])
    [ 0; 1; 3; 4; 5; 127; 128; 129; 256; 4095; 4096; 32768; mib ]
  @ [ (Software_mapped, 200) ]

(* allocation *)

(* A device whose driver takes its memory in units of 64 KiB below 2 MiB and 2
   MiB from there, as NVIDIA's RM takes video memory. *)
let granules = [ (0, 64 * kib); (2 * mib, 2 * mib) ]

let granular =
  fixture (fun () ->
      fst (P.open_ ~host_visible:false ~granules "memory:granules"))

let granule_rows =
  [
    (1, 64 * kib);
    (64 * kib, 64 * kib);
    ((64 * kib) + 1, 128 * kib);
    ((2 * mib) - 4, 2 * mib);
    (2 * mib, 2 * mib);
    ((2 * mib) + 1, 4 * mib);
    (3 * mib, 4 * mib);
  ]

let test_granules (n, taken) =
  let d = granular () in
  equal ~msg:"the device's granules"
    (list (pair int int))
    granules (Rig.granules d);
  equal ~msg:"allocation" int taken (Rig.allocation d n);
  let before = settled d in
  let[@inline never] [@local never] held () =
    let b = B.create d n in
    let added = settled d - before in
    ignore (Sys.opaque_identity b);
    added
  in
  equal ~msg:"allocated while it lives" int taken (held ());
  equal ~msg:"allocated once it returned" int before (settled d)

(* rig.mli, allocation: "on {!host}, ... at least 16 bytes where [n > 0]". *)
let test_allocation_host_byte () = equal int 16 (Rig.allocation Rig.host 1)

(* rig.mli, allocation: "in the host's budget, its whole words": 5 bytes of a
   hidden device's Pinned memory are two words. *)
let test_allocation_pinned () =
  let d, _ = P.open_ ~host_visible:false "memory:allocation-pinned" in
  equal int 8 (Rig.allocation ~memory:B.Pinned d 5)

let test_allocation_negative () =
  raises_match (Exn.invalid_arg ~sub:"negative") (fun () ->
      Rig.allocation Rig.host (-1))

(* set_budget *)

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
  equal ~msg:"frees of the live memory" int 0 (frees p (Support.at live));
  raises_match Exn.invalid_arg (fun () -> Rig.set_budget d (-1))

(* A budget set while another domain loses the device raises nothing, and the
   cache it took returns once the device's close stopped it, each memory once.
   The first free of the cache waits at Polled's free gate while the loss
   runs. *)
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
  Support.await "the cache's first free at the gate" (fun () -> P.freers p = 1);
  P.fail p;
  (try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ());
  P.open_frees p;
  equal ~msg:"the budget" string "returned" (Domain.join setter);
  Rig.close d;
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  equal ~msg:"frees of the cached memory" (list int) [ 1; 1; 1 ]
    (List.map (frees p) ats)

(* The C heap's allocated bytes, for a test that needs them: the C library says
   on macOS and glibc. *)
let heap_bytes () =
  match Support.heap_bytes () with
  | Some n -> n
  | None -> skip ~reason:"the C library does not count its heap" ()

(* set_budget on the host gives back the memory it keeps until it holds at most
   the new budget: of four dropped buffers of 1 MiB, under a budget 2.5 MiB
   above its live bytes, it gives back at least 1.5 MiB and keeps the rest that
   fits. *)
let test_host_set_budget () =
  Gc.full_major ();
  Rig.free_cache Rig.host;
  let live = ref (List.init 4 (fun _ -> B.create Rig.host mib)) in
  ignore (Sys.opaque_identity !live);
  live := [];
  Gc.full_major ();
  Gc.full_major ();
  let kept = heap_bytes () in
  with_host_budget
    (Rig.allocated Rig.host + (5 * mib / 2))
    (fun () ->
      let given = kept - heap_bytes () in
      at_least ~msg:"given back" int ~than:(3 * mib / 2) given;
      at_most ~msg:"what fits stays" int ~than:(3 * mib) given)

(* free_cache *)

let test_free_cache () =
  let d, p = P.open_ "memory:free-cache" in
  let at = dropped d (4 * kib) in
  collect d;
  equal ~msg:"frees while cached" int 0 (frees p at);
  Rig.free_cache d;
  equal ~msg:"frees after free_cache" int 1 (frees p at)

(* Bytes the C heap's count may grow by across a call and two full cycles, with
   what the runtime allocates for itself and does not free by the count: 144
   once in thirteen runs on Ubuntu 24.04, after the tests before this one. *)
let heap_slack = 4 * kib

(* free_cache on the host gives back the memory it keeps of collected buffers:
   the C heap shrinks by a dropped buffer's bytes, short of [heap_slack] at
   most, which no buffer kept would give. Both counts follow two full cycles,
   which free what the runtime allocated for the cycles before them. The cycles
   give back none of the buffer, which the host keeps across cycles in its cache
   of up to 32 MiB. *)
let test_host_free_cache () =
  let n = 4 * mib in
  let settled () =
    Gc.full_major ();
    Gc.full_major ();
    heap_bytes ()
  in
  Gc.full_major ();
  Rig.free_cache Rig.host;
  ignore (dropped Rig.host n);
  let kept = settled () in
  Rig.free_cache Rig.host;
  at_least ~msg:"bytes given back" int ~than:(n - heap_slack) (kept - settled ())

let tests =
  [
    group "collection"
      [
        test "an exported array's memory counts until it is collected"
          test_host_exported;
        test "a buffer's memory serves its size once a view outliving it goes"
          test_host_keeps_size;
        prop "an exported array keeps its bytes past its buffer's collection"
          ~count:60 host_sizes export_law;
      ];
    group "waiting"
      [
        test "memory queued work reads is freed over the budget once it ran"
          test_over_budget_queued;
        test "memory another device used is cached once that use is reached"
          test_foreign_use;
        test "host memory a device borrowed returns once its work ran"
          test_borrowed_host;
        test "a bigarray a device borrowed lives until its work ran"
          test_borrowed_bigarray;
      ];
    group "release"
      [
        test "memory collected over the budget is not reused past it"
          test_over_budget_cache;
        test "a closed device's cache returns to its driver" test_closed_cache;
        test "a closed device's memory collected later returns at any call"
          test_closed_collected;
        test
          "a drain of closed devices that raises leaves the others' memory to \
           the next call"
          test_closed_drain_raises;
        test "a buffer dropped by a domain that then blocks returns"
          test_blocked_domain;
        test "host memory borrowed on another domain returns at a host creation"
          test_borrowed_host_elsewhere;
        test "a keep collected from another domain ends at a host creation"
          test_keep_elsewhere;
        cases
          "each call that moves memory lets go of the values collected \
           submissions keep"
          ~id:mover_name
          [ Create; Of_string; Submit_copy; Copy ]
          test_mover;
        test "a drain beside a loss frees each memory once, then stops"
          test_drain_beside_loss;
        test "two drains of one device judge its retiring memory in turn"
          test_judged_once;
        test "a drain that judges queued memory allocates nothing more"
          test_due_words;
        test "a drain allocates nothing for each buffer it caches"
          test_drain_words;
        test
          "memory goes back after every mapping of it, a closed mapper's \
           included"
          test_free_after_maps;
        test "memory whose last mapping goes before its stop goes back once"
          test_give_beside_stop;
        test "memory two raced maps share goes back after both" test_raced_maps;
        test "a borrow beside its mapper's close goes back with the memory"
          test_close_beside_borrow;
        test "a mapper closed beside the free of memory it maps raises nothing"
          test_mapper_beside_free;
      ];
    group "budgets"
      [
        prop "a host buffer holds its bytes in the budget until collected"
          ~count:60
          ~examples:[ 16; 4 * kib; 128 * kib; mib ]
          host_sizes held_law;
        test "pinned memory of hidden device memory counts in the host's budget"
          test_pinned;
        test "mapped memory counts in its device's budget" test_mapped_budget;
        test "a budget counts each buffer by its whole words" test_budget_words;
        test
          "an allocation racing another's driver call for the last room is \
           refused"
          test_last_room;
      ];
    group "pace"
      [
        test "a large host buffer dropped young brings an eighth of its cycle"
          test_paced_host_young;
        test "a custom_minor_max_size of 0 keeps host buffers in the major heap"
          test_paced_host_major;
        test "a large buffer the minor heap promotes paces cycles by its bytes"
          test_paced_host_promoted;
        test "small dropped host buffers bring a cycle per 4 MiB at most"
          test_paced_host_small;
        test
          "at a small heap, dropped host buffers bring a cycle per 4 MiB at \
           most"
          test_paced_host;
        test "a large heap's cycles come once per its share at most"
          test_paced_host_heap;
        test "live host buffers raise the bound as heap bytes do"
          test_paced_host_live;
        test "a refused host allocation leaves the collector's pace as it was"
          test_paced_refused;
        test "a larger minor heap leaves the major pace as it was"
          test_paced_minor_heap;
        test "a buffer dropped after a cycle returns under young churn"
          test_paced_host_returns;
        test
          "opening devices and making and returning buffers leave the \
           runtime's parameters as they were"
          test_sets_none;
        test
          "dropped buffers of a host-addressed device's many budgets never \
           force a collection"
          test_paced;
        test
          "a software device's dropped buffers bring a cycle per half the \
           last-level cache at most"
          test_paced_software;
        test
          "a software device's dropped buffers bring cycles as often beside \
           its cache as beside none"
          test_paced_beside_cache;
        test
          "a software device's dropped buffers of many budgets never force a \
           collection"
          test_paced_memory_budget;
        test
          "a hidden device's dropped buffers of many budgets never force a \
           collection"
          test_paced_hidden;
      ];
    group "kept host buffers"
      [
        test "the host keeps a collected buffer's memory for its size"
          test_host_cache;
      ];
    group "refusals"
      [
        test "an allocation over the budget raises at once and keeps the cache"
          test_over_budget;
        test "a host allocation its budget refuses raises for its own bytes"
          test_host_budget_refused;
        test "an allocation the driver refuses releases the cache, then raises"
          test_refused;
        test
          "an allocation the budget refuses waits for the work collected \
           memory waits for"
          test_refused_waits;
        test "an allocation the budget refuses raises the loss of that work"
          test_refused_lost;
        test "a round waits for a device whose lock another thread holds"
          test_round_waits_for_lock;
        test "a Ctrl-C in a reclamation round's wait loses none of the cache"
          test_break_in_round;
        test "an allocation the driver refuses collects unreachable buffers"
          test_collects;
        test "a buffer a collected submission kept returns to a refusal"
          test_refused_kept;
        test "a refusal that lets go of nothing collects once"
          test_refused_collects_once;
        test "pinned memory another device keeps returns to a refused one"
          test_pinned_reclaims;
        test
          "a dropped host buffer a device borrowed returns to a pinned \
           allocation"
          test_borrowed_reclaimed;
        test "host memory an idle device borrowed returns to an allocation"
          test_idle_borrower;
        test "a host allocation the C library refuses raises for the host"
          test_host_refused;
        test "a growth short of memory reclaims the host's budget"
          test_growth_reclaims_pinned;
        test
          "the host's out-of-memory ladder raises a device's fault it finds as \
           that device's loss, never the allocation's"
          test_ladder_beside_fault;
      ];
    group "budget"
      [
        cases "the budget of the host and of an io device is max_int" ~id:fst
          [
            ("the host", lazy Rig.host);
            ( "an io device",
              lazy (Result.get_ok (Rig.open_io "memory:budget-io")) );
          ]
          test_budget_max_int;
        test "a driver's device's budget starts at its driver's budget"
          test_budget_driver;
      ];
    group "capacity"
      [
        cases "capacity is None where a device's memory is the machine's"
          ~id:fst
          [
            ("the host", lazy Rig.host);
            ( "a software device",
              lazy (Result.get_ok (Rig.software "memory:capacity-software")) );
            ( "an io device",
              lazy (Result.get_ok (Rig.open_io "memory:capacity-io")) );
            ( "a driver that reports none",
              lazy (fst (P.open_ "memory:capacity-unbounded")) );
          ]
          test_capacity_none;
        test "capacity is the driver's report at the call" test_capacity;
        test "a fault in a capacity report loses the device" test_capacity_fault;
      ];
    group "charged"
      [
        cases "charged is the device whose budget a kind counts in"
          ~id:(fun (row, _, _, _) -> row)
          charged_rows test_charged;
      ];
    group "allocated"
      [
        prop "allocated grows by a buffer's allocation" ~count:60
          ~examples:allocated_examples
          (Gen.pair held_places allocated_sizes)
          allocated_law;
        stateful ~count:40 ~steps:10 ~domains:2
          "allocated counts each of two domains' allocations once"
          budget_commands;
      ];
    group "allocation"
      [
        cases "allocation counts in whole units of the driver's granules"
          ~id:(fun (n, _) -> Printf.sprintf "%d bytes" n)
          granule_rows test_granules;
        test "a non-empty host buffer counts at least 16 bytes"
          test_allocation_host_byte;
        test "in the host's budget a buffer counts its whole words"
          test_allocation_pinned;
        test "allocation of a negative size raises" test_allocation_negative;
      ];
    group "set_budget"
      [
        test "set_budget returns cached memory, never live memory"
          test_set_budget;
        test
          "set_budget while another domain loses the device raises nothing and \
           returns its cache once"
          test_budget_beside_loss;
        test "set_budget on the host gives back what it keeps beyond the budget"
          test_host_set_budget;
      ];
    group "free_cache"
      [
        test "free_cache with no work in flight returns the cache at once"
          test_free_cache;
        test "free_cache on the host gives back what it keeps"
          test_host_free_cache;
      ];
  ]

let () = exit (run "rig.memory" tests)
