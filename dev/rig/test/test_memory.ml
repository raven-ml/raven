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

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s =
  Rig.submit s ~reads ~writes ~waits

let timeout = 60.
let kib = 1024

let out_of_memory d n = function
  | Rig.Out_of_memory (d', n') -> Rig.equal d d' && n = n'
  | _ -> false

(* A buffer of [n] bytes on [d] that is unreachable once this returns: its
   address. *)
let[@inline never] dropped ?memory d n = B.address (B.create ?memory d n)

(* Collects, then drains [d], as an allocation on it does first: of pinned
   memory, which no budget refuses. *)
let collect d =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create ~memory:Pinned d 8))

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
  equal ~msg:"the live memory" bool false (freed p (B.address live))

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
  let d, p = P.open_ "memory:set-budget" in
  let live = B.create d (8 * kib) in
  List.iter (fun _ -> ignore (dropped d (4 * kib))) [ 1; 2; 3 ];
  collect d;
  Rig.set_budget d (12 * kib);
  equal int (12 * kib) (Rig.budget d);
  collect d;
  at_most ~msg:"held after" int ~than:(12 * kib) (P.allocated p B.Device);
  Rig.set_budget d 0;
  collect d;
  equal ~msg:"live memory stays" bool false (freed p (B.address live));
  raises_match Exn.invalid_arg (fun () -> Rig.set_budget d (-1))

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
  let reads = Sub.make ~reads:1 ~writes:0 d [||] in
  let[@inline never] queued () =
    let b = B.create d 64 in
    ignore (submit reads ~reads:[| b |]);
    B.address b
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
  let reads = Sub.make ~reads:1 ~writes:0 d [||] in
  let[@inline never] queued () =
    let b = B.create d (64 * kib) in
    ignore (submit reads ~reads:[| b |]);
    B.address b
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
   small ones included. *)
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
      equal ~msg:(Printf.sprintf "%d bytes, live" n) int (before + n) held;
      Gc.full_major ();
      Gc.full_major ();
      equal
        ~msg:(Printf.sprintf "%d bytes, collected" n)
        int before (Support.host_held ()))
    [ 16; 4 * kib; 128 * kib ]

(* On a device with copies, pinned memory is host memory: it counts in the
   host's budget, not the device's. *)
let test_pinned () =
  let d, p = P.open_ "memory:pinned" in
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
  let a, _ = P.open_ "memory:pinned-asker" in
  let b, _ = P.open_ "memory:pinned-keeper" in
  let n = 64 * kib in
  ignore (dropped ~memory:Pinned b n);
  collect b;
  with_host_budget (Support.host_held ()) (fun () ->
      equal int n (B.length (B.create ~memory:Pinned a n)))

(* A host buffer a device borrowed, dropped, returns its bytes when a pinned
   allocation the host's budget refuses reclaims: the reclaim drains the host,
   whose list holds the buffer, and the allocation is made. *)
let test_borrowed_reclaimed () =
  let d, _ = P.open_ "memory:borrowed-reclaim" in
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
  let d, _ = P.open_ ~copies:false "memory:pinned-own" in
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
  let d, p = P.open_ "memory:mapped-budget" in
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
      let s = Sub.make ~reads:1 ~writes:0 b [||] in
      ignore (submit s ~reads:[| require_some (B.borrow b m) |]);
      B.address m)
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
      { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
    in
    ignore (submit (Sub.make ~reads:0 ~writes:0 d [| part |]));
    B.address h
  in
  let at = written () in
  Gc.full_major ();
  Gc.full_major ();
  let other = B.create Rig.host n in
  equal ~msg:"while the work is queued" bool false (B.address other = at);
  ignore (P.run p);
  ignore (B.create Rig.host 0);
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"once it ran" int at (B.address (B.create Rig.host n));
  ignore (Sys.opaque_identity other)

(* A bigarray a device borrowed through of_bigarray stays reachable until that
   device's queued work on it ran. *)
let test_borrowed_bigarray () =
  let d, p = P.open_ "memory:borrowed-bigarray" in
  let n = 1 lsl 16 in
  let collected = Atomic.make false in
  let[@inline never] written () =
    let ba = B.bigarray Bigarray.char (B.create Rig.host n) in
    Gc.finalise (fun _ -> Atomic.set collected true) ba;
    let src = B.create d n in
    let dst = require_some (B.borrow d (B.of_bigarray ba)) in
    let part =
      { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
    in
    ignore (submit (Sub.make ~reads:0 ~writes:0 d [| part |]))
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
      { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
    in
    ignore (submit (Sub.make ~reads:0 ~writes:0 d [| part |]))
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
            at := B.address b;
            B.bigarray Bigarray.float32 b)
            ()))
  in
  Gc.full_major ();
  Gc.full_major ();
  let other = B.create Rig.host n in
  not_equal ~msg:"while the view lives" int !at (B.address other);
  ignore (Sys.opaque_identity !view);
  view := None;
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"once it is collected" int !at (B.address (B.create Rig.host n));
  ignore (Sys.opaque_identity other)

(* The host keeps a collected buffer of 64 KiB or more for the next buffer of
   its size, unless a bigarray of it lives, which keeps its bytes. *)
let test_host_cache () =
  let n = (64 * kib) + 4093 in
  let first = dropped Rig.host n in
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"reused" int first (B.address (B.create Rig.host n));
  let at, view =
    (fun () ->
      let b = B.create Rig.host n in
      let ba = B.bigarray Bigarray.char b in
      Bigarray.Array1.fill ba 'a';
      (B.address b, ba))
      ()
  in
  Gc.full_major ();
  Gc.full_major ();
  let b = B.create Rig.host n in
  not_equal ~msg:"the viewed memory" int at (B.address b);
  Bigarray.Array1.fill (B.bigarray Bigarray.char b) 'b';
  equal ~msg:"the view keeps its bytes" char 'a' view.{n - 1}

(* Two domains allocating on one device under a budget of three buffers: the
   live buffers never exceed it. *)
type budgeted = { d : Rig.t; live : B.t list ref; lock : Mutex.t }
type budgeted_model = { mutable held : int }

let per = 4 * kib

(* Devices go back to a pool when their program ends, with their buffers
   dropped: a drain at the next program's first allocation takes them back. *)
let budgeted_pool = Mutex.create ()
let budgeted_free = ref []
let budgeted_opened = Atomic.make 0

let make_budgeted () =
  let d =
    match
      Mutex.protect budgeted_pool (fun () ->
          match !budgeted_free with
          | d :: rest ->
              budgeted_free := rest;
              Some d
          | [] -> None)
    with
    | Some d -> d
    | None ->
        let n = Atomic.fetch_and_add budgeted_opened 1 in
        let d, _ = P.open_ (Printf.sprintf "memory:budgeted-%d" n) in
        Rig.set_budget d (3 * per);
        d
  in
  { d; live = ref []; lock = Mutex.create () }

let release_budgeted t =
  t.live := [];
  Gc.full_major ();
  Rig.free_cache t.d;
  Mutex.protect budgeted_pool (fun () -> budgeted_free := t.d :: !budgeted_free)

let alloc_budgeted t =
  let b = B.create t.d per in
  Mutex.protect t.lock (fun () -> t.live := b :: !(t.live))

let judge_alloc r = function
  | Ok () ->
      at_most ~msg:"live buffers" int ~than:2 r.held;
      cover "an allocation that fills the budget" (r.held = 2);
      r.held <- r.held + 1
  | Error (Rig.Out_of_memory _) ->
      cover "an allocation the budget refuses" true;
      equal ~msg:"live buffers" int 3 r.held
  | Error e -> raise e

let budgeted =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "held %d" r.held)
    ~release:release_budgeted "b"

(* Allocations are drawn three times as often as opens, so a program that opens
   a device allocates on it. *)
let budget_commands =
  let alloc =
    command "alloc" (budgeted ^-> judges unit) judge_alloc alloc_budgeted
  in
  [
    command "open"
      (Gen.unit @-> makes budgeted)
      (fun () -> { held = 0 })
      make_budgeted;
    alloc;
    alloc;
    alloc;
  ]

let tests =
  [
    group ~timeout "budget"
      [
        test "an allocation over the budget raises at once and keeps the cache"
          test_over_budget;
        test "a copy whose staging memory the host refuses gives its slot back"
          test_staging_refused;
        stateful ~count:5 ~domains:2
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
        test "pinned memory on a device with copies counts in the host's budget"
          test_pinned;
        test "pinned memory the host addresses counts in its device's budget"
          test_pinned_own;
        test "mapped memory the window cannot hold is pinned, the cache kept"
          test_mapped_window;
        test "mapped memory the budget cannot hold is pinned" test_mapped_budget;
        test "mapped memory is the device's" test_mapped;
        test "pinned memory another device keeps returns to a refused one"
          test_pinned_reclaims;
        test
          "a dropped host buffer a device borrowed returns to a pinned \
           allocation"
          test_borrowed_reclaimed;
      ];
    group ~timeout "reclamation"
      [
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
      ];
  ]

let () = exit (run "rig.memory" tests)
