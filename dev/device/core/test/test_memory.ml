(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Memory a device holds: its kinds, its budget, its cache and how collected
   buffers return. Polled logs what its driver allocates and frees. *)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled
module Support = Device_core_support
module S = Device_dtype.Scalar

let timeout = 60.
let kib = 1024

let out_of_memory d n = function
  | C.Out_of_memory (d', n') -> C.equal d d' && n = n'
  | _ -> false

(* A buffer of [n] bytes on [d] that is unreachable once this returns: its
   address. *)
let[@inline never] dropped ?memory d n =
  B.address (B.create ?memory d S.UInt8 n)

(* Collects, then drains [d], as an allocation on it does first: of pinned
   memory, which no budget refuses. *)
let collect d =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create ~memory:Pinned d S.UInt8 8))

let last n l = List.filteri (fun i _ -> i >= List.length l - n) l
let freed p at = List.exists (fun (a, _) -> a = at) (P.frees p)

let allocs =
  Testable.make
    ~pp:
      (Format.pp_print_list (fun ppf (k, n, ok) ->
           Format.fprintf ppf "%s %d %b"
             (match k with
             | `Device -> "device"
             | `Pinned -> "pinned"
             | `Mapped -> "mapped")
             n ok))
    ~equal:( = )

(* Budget *)

(* An allocation over the budget raises at once: the cache stays. *)
let test_over_budget () =
  let d, p = P.open_ "memory:over-budget" in
  C.set_budget d (8 * kib);
  let at = dropped d (4 * kib) in
  collect d;
  raises_match
    (out_of_memory d ((8 * kib) + 1))
    (fun () -> B.create d S.UInt8 ((8 * kib) + 1));
  equal ~msg:"the cached memory" bool false (freed p at);
  at_least int ~than:(4 * kib) (P.allocated p `Device)

(* An allocation the driver refuses releases the cache, then raises. *)
let test_refused () =
  let d, p = P.open_ ~memory:(8 * kib) "memory:refused" in
  let live = B.create d S.UInt8 (4 * kib) in
  let at = dropped d (4 * kib) in
  collect d;
  raises_match
    (out_of_memory d (8 * kib))
    (fun () -> B.create d S.UInt8 (8 * kib));
  equal ~msg:"the cached memory" bool true (freed p at);
  equal ~msg:"the live memory" bool false (freed p (B.address live))

(* An allocation the driver refuses collects unreachable buffers and tries
   again: memory finalisers hold, two deep, returns. *)
let test_collects () =
  let d, _ = P.open_ ~memory:(4 * kib) "memory:collects" in
  (fun () ->
    let b = B.create d S.UInt8 (4 * kib) in
    let inner = ref 0 and outer = ref 0 in
    Gc.finalise (fun _ -> ignore (Sys.opaque_identity b)) inner;
    Gc.finalise (fun _ -> ignore (Sys.opaque_identity inner)) outer)
    ();
  equal int (4 * kib) (B.nbytes (B.create d S.UInt8 (4 * kib)))

let test_set_budget () =
  let d, p = P.open_ "memory:set-budget" in
  let live = B.create d S.UInt8 (8 * kib) in
  List.iter (fun _ -> ignore (dropped d (4 * kib))) [ 1; 2; 3 ];
  collect d;
  C.set_budget d (12 * kib);
  equal int (12 * kib) (C.budget d);
  collect d;
  at_most ~msg:"held after" int ~than:(12 * kib) (P.allocated p `Device);
  C.set_budget d 0;
  collect d;
  equal ~msg:"live memory stays" bool false (freed p (B.address live));
  raises_match Exn.invalid_arg (fun () -> C.set_budget d (-1))

let test_free_cache () =
  let d, p = P.open_ "memory:free-cache" in
  let at = dropped d (4 * kib) in
  collect d;
  equal ~msg:"cached" bool false (freed p at);
  C.free_cache d;
  equal ~msg:"after free_cache" bool true (freed p at)

let test_host_budget () = equal int max_int (C.budget C.host)

(* Kinds *)

let test_pinned () =
  let d, p = P.open_ "memory:pinned" in
  C.set_budget d 0;
  equal int (4 * kib) (B.nbytes (B.create ~memory:Pinned d S.UInt8 (4 * kib)));
  equal allocs [ (`Pinned, 4 * kib, true) ] (P.allocs p);
  raises_match (out_of_memory d 1) (fun () -> B.create d S.UInt8 1)

(* Mapped memory the window cannot hold is pinned memory, and the cache
   stays. *)
let test_mapped_window () =
  let d, p = P.open_ ~window:(4 * kib) "memory:window" in
  let at = dropped d (4 * kib) in
  collect d;
  let b = B.create ~memory:Mapped d S.UInt8 (8 * kib) in
  equal ~msg:"the cached memory" bool false (freed p at);
  equal allocs
    [ (`Mapped, 8 * kib, false); (`Pinned, 8 * kib, true) ]
    (last 2 (P.allocs p));
  ignore (Sys.opaque_identity b)

let test_mapped_budget () =
  let d, p = P.open_ "memory:mapped-budget" in
  C.set_budget d (4 * kib);
  ignore (B.create ~memory:Mapped d S.UInt8 (8 * kib));
  equal allocs [ (`Pinned, 8 * kib, true) ] (P.allocs p)

let test_mapped () =
  let d, p = P.open_ "memory:mapped" in
  let b = B.create ~memory:Mapped d S.UInt8 (4 * kib) in
  equal allocs [ (`Mapped, 4 * kib, true) ] (P.allocs p);
  equal bool true (C.equal d (B.device b))

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
  C.free_cache d;
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
  C.set_budget d budget;
  let forced () = (Gc.quick_stat ()).forced_major_collections in
  let before = forced () in
  for _ = 1 to 20 * 16 do
    ignore (Sys.opaque_identity (B.create d S.UInt8 (budget / 16)))
  done;
  equal int 0 (forced () - before)

(* Memory another device used enters its owner's cache once that use is
   reached. *)
let test_foreign_use () =
  let a, pa = P.open_ "memory:owner" in
  let b, pb = P.open_ "memory:reader" in
  let at =
    (fun () ->
      let m = B.create a S.UInt8 (4 * kib) in
      let s = Sub.make ~reads:1 ~writes:0 ~waits:0 b [||] in
      Sub.read s 0 m;
      ignore (C.submit s);
      B.address m)
      ()
  in
  collect a;
  C.free_cache a;
  collect a;
  equal ~msg:"while the read is unreached" bool false (freed pa at);
  ignore (P.run pb);
  collect a;
  C.free_cache a;
  collect a;
  equal ~msg:"once it is reached" bool true (freed pa at)

(* The host keeps a collected buffer of 64 KiB or more for the next buffer of
   its size, unless a bigarray of it lives, which keeps its bytes. *)
let test_host_cache () =
  let n = (64 * kib) + 4093 in
  let first = dropped C.host n in
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"reused" int first (B.address (B.create C.host S.UInt8 n));
  let at, view =
    (fun () ->
      let b = B.create C.host S.UInt8 n in
      let ba = B.bigarray Bigarray.char b in
      Bigarray.Array1.fill ba 'a';
      (B.address b, ba))
      ()
  in
  Gc.full_major ();
  Gc.full_major ();
  let b = B.create C.host S.UInt8 n in
  not_equal ~msg:"the viewed memory" int at (B.address b);
  Bigarray.Array1.fill (B.bigarray Bigarray.char b) 'b';
  equal ~msg:"the view keeps its bytes" char 'a' view.{n - 1}

let tests =
  [
    group ~timeout "budget"
      [
        test "an allocation over the budget raises at once and keeps the cache"
          test_over_budget;
        test "an allocation the driver refuses releases the cache, then raises"
          test_refused;
        test "an allocation the driver refuses collects unreachable buffers"
          test_collects;
        test "set_budget returns cached memory, never live memory"
          test_set_budget;
        xfail
          ~reason:
            "free_cache defers each free to the next drain, even with no work \
             in flight"
          (test "free_cache with no work in flight returns the cache at once"
             test_free_cache);
        test "the host's budget is max_int" test_host_budget;
      ];
    group ~timeout "kinds"
      [
        test "pinned memory counts in no budget" test_pinned;
        test "mapped memory the window cannot hold is pinned, the cache kept"
          test_mapped_window;
        test "mapped memory the budget cannot hold is pinned" test_mapped_budget;
        test "mapped memory is the device's" test_mapped;
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
      ];
  ]

let () = exit (run "device_core.memory" tests)
