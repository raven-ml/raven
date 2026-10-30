(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of an AMD GPU, driven by the driver alone. A call on an
   image of 128 functions, each with scratch, is made once for each
   allocation it makes, with every allocation from that one on raising until
   the call returns, as an asynchronous exception does. The image's one
   unload then frees every launch the image made, the interrupted call's
   included, so the C heap holds no more than its allocator's rounding.

   Views of another GPU's memory are walked the same way on a machine of two
   AMD GPUs the topology links: the 33rd view, where a table of 32 would
   grow, leaves every view's free and the memory's own free to succeed. A
   view's free is also walked with a signal at each of its allocations, as a
   Ctrl-C lands: a free it stopped leaves the view to a later free, after
   which the memory's own free gives its GPU memory back.

   A path's lock counts among the locks its thread holds: a callback at any
   allocation inside it that takes it again raises Invalid_argument, having
   taken nothing, and the capability's calls from inside a lock raise it as
   they start. *)

open Windtrap
module A = Rig_amd
module S = Rig_amd_support

let strf = Printf.sprintf
let timeout = 120.

(* many.cl's kernels, k1000 to k2333, numbered in base 4. *)
let functions =
  List.init 128 (fun k ->
      strf "k%d%d%d%d" ((k / 64) + 1) (k / 16 mod 4) (k / 4 mod 4) (k mod 4))

let made = List.filteri (fun k _ -> k < 32) functions
let last = List.nth functions 32

(* The C heap's bytes, once the collector freed what it holds there. *)
let heap () =
  Gc.full_major ();
  Gc.full_major ();
  match Rig_support.heap_bytes () with
  | Some n -> n
  | None -> skip ~reason:"the system counts no C heap" ()

(* The bytes the C heap may hold after a walk: its allocator's rounding. *)
let slack = 256

let many =
  lazy
    (In_channel.with_open_bin "abi/fixtures/many_gfx1201.hsaco"
       In_channel.input_all)

(* A new image of many.cl's kernels on [g], laid over new [Device] memory:
   the image and its memory. *)
let image g =
  match A.image g (Lazy.force many) with
  | Error why -> skip ~reason:why ()
  | Ok (Rig_edge.Loaded _) -> fail "an image the device placed itself"
  | Ok (Rig_edge.Place (n, lay)) -> (
      match A.alloc g Device n with
      | None -> fail "no GPU memory for the image"
      | Some r -> (fst (lay r), r))

(* The entries of [fs] on the image [i]. *)
let entries i fs =
  List.iter
    (fun f ->
      match A.entry i f with
      | Entry _ -> ()
      | Absent -> failwith (f ^ " is absent")
      | No_memory n -> failwith (strf "no memory for %s's %d bytes" f n))
    fs

(* [loaded g f] is [f i] for a new image [i] on [g], which it unloads and
   frees after. *)
let loaded g f =
  let i, r = image g in
  Fun.protect
    ~finally:(fun () ->
      A.unload g i;
      A.free g r)
    (fun () -> f i)

(* [opened f] is [f g] for a device [g] of GPU 0, once its scratch serves
   each of many.cl's functions. *)
let opened f =
  if not (S.present ()) then skip ~reason:"the machine has no AMD GPU" ();
  S.with_driver @@ fun g ->
  loaded g (fun i -> entries i functions);
  f g

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
  opened @@ fun g ->
  walk @@ fun n ->
  let before = heap () in
  let reached =
    loaded g (fun i ->
        entries i made;
        Rig_support.interrupted n (fun () -> entries i [ last ]))
  in
  (heap () - before, reached)

(* The unload of an image with an entry of each function. *)
let test_unload () =
  opened @@ fun g ->
  walk @@ fun n ->
  let before = heap () in
  let i, r = image g in
  entries i functions;
  let reached = Rig_support.interrupted n (fun () -> A.unload g i) in
  A.free g r;
  (heap () - before, reached)

(* Views *)

(* The bytes of each region the view walk maps. *)
let bytes = 4096

(* [linked f] is [f g g'] for devices of GPUs 0 and 1 through amdgpu, where
   [g] maps [g']'s memory. *)
let linked f =
  if S.driverless () then skip ~reason:"the walk is of amdgpu's views" ();
  if Result.value (Rig_amd_amdgpu.count ()) ~default:0 < 2 then
    skip ~reason:"the machine has one AMD GPU" ();
  let opened i =
    match Rig_gpu_support.opened (Rig_amd_amdgpu.gpu i) with
    | Ok g -> g
    | Error why -> fail why
  in
  let g = opened 0 in
  Fun.protect ~finally:(fun () -> A.stop g ~fault:None) @@ fun () ->
  let g' = opened 1 in
  Fun.protect ~finally:(fun () -> A.stop g' ~fault:None) @@ fun () ->
  if not (A.peer g g') then skip ~reason:"GPU 0 does not reach GPU 1" ();
  f g g'

(* [n] regions of [g']'s Device memory. *)
let regions g' n =
  List.init n (fun _ ->
      match A.alloc g' Device bytes with
      | Some r -> r
      | None -> fail "no GPU memory")

(* The 33rd view of another GPU's memory, where a table of 32 views would
   grow: each view and each region then goes back. A lost table raises in
   a view's free. The GPU memory is not compared: from the view's MAP to the
   region's return, the driver allocates, and an attempt interrupted there
   raises at every later allocation, so no cleanup can end the view. *)
let test_views () =
  linked @@ fun g g' ->
  let walk_calls attempt =
    let rec from n = if attempt n then from (n + 1) in
    from 1
  in
  walk_calls @@ fun n ->
  let rs = regions g' 33 in
  let first = List.filteri (fun k _ -> k < 32) rs and r = List.nth rs 32 in
  let view r =
    match A.map_peer g g' r with
    | Some v -> v
    | None -> fail "GPU 0 maps no view of GPU 1's memory"
  in
  let views = List.map view first in
  let interrupted = ref None in
  let reached =
    Rig_support.interrupted n (fun () -> interrupted := A.map_peer g g' r)
  in
  List.iter (A.free g) (Option.to_list !interrupted @ views);
  List.iter (A.free g') rs;
  reached

(* A view's free, interrupted by a signal at each of its allocations: the
   view goes back at once or at a later free, after which the memory's own
   free gives back what KFD counts. *)
let test_view_free () =
  linked @@ fun g g' ->
  let held () =
    match S.vram 1 with
    | Some n -> n
    | None -> skip ~reason:"KFD shows no count of this process's GPU memory" ()
  in
  let walk_calls attempt =
    let rec from n = if attempt n then from (n + 1) in
    from 1
  in
  walk_calls @@ fun n ->
  let before = held () in
  let r = List.hd (regions g' 1) in
  let v =
    match A.map_peer g g' r with
    | Some v -> v
    | None -> fail "GPU 0 maps no view of GPU 1's memory"
  in
  let freed = ref false in
  let reached =
    Rig_support.interrupted ~landing:Signal n (fun () ->
        A.free g v;
        freed := true)
  in
  if not !freed then A.free g v;
  A.free g' r;
  equal ~msg:(strf "from allocation %d, GPU 1's memory held" n) int before
    (held ());
  reached

(* Locks *)

(* A callback at any allocation of a section under a path's lock takes the
   lock, in one walk, and checks the thread's locks, in another: inside the
   section each raises [Invalid_argument], having taken nothing; outside it
   each runs. *)
let test_lock_reentered () =
  let l = A.Lock.create () in
  let section () =
    let xs = List.init 4 (fun i -> ref i) in
    A.Lock.protect l (fun () ->
        ignore (Sys.opaque_identity (List.map (fun r -> ref !r) xs)))
  in
  let walk prefix call =
    let seen = Rig_support.seen () in
    let rec from n =
      let landing, landed = Rig_support.landing seen prefix call in
      let reached = Rig_support.interrupted ~landing n section in
      ignore
        (Rig_support.answered (strf "%s, called at %d" prefix n) landed
          : bool option);
      if reached then from (n + 1)
    in
    from 1;
    Rig_support.covered seen
  in
  walk "Rig_amd: a lock taken again" (fun () -> A.Lock.protect l ignore);
  walk "Rig_amd.lock: called from" (fun () -> A.Lock.guarded "Rig_amd.lock" ignore)

(* An exception at any allocation of a section under a path's lock gives the
   lock back: the thread holds what it held before, and takes the lock again. *)
let test_lock_interrupted () =
  let before = Rig_support.holds () in
  let l = A.Lock.create () in
  let rec from n =
    let section () =
      let xs = List.init 4 (fun i -> ref i) in
      A.Lock.protect l (fun () ->
          ignore (Sys.opaque_identity (List.map (fun r -> ref !r) xs)))
    in
    let reached = Rig_support.interrupted n section in
    equal
      ~msg:(strf "locks held, interrupted at %d" n)
      int before (Rig_support.holds ());
    A.Lock.protect l ignore;
    if reached then from (n + 1)
  in
  from 1

(* The capability's trace and, on an AQL queue, its scratch, from inside a lock
   of the calling thread: each raises as it starts. A scratch outside the lock
   runs. *)
let test_capability_held () =
  opened @@ fun g ->
  let c = A.capability g in
  let refused what f =
    let r =
      Rig_support.inside_lock @@ fun () ->
      match f () with () -> None | exception Invalid_argument why -> Some why
    in
    match r with
    | None -> failf "%s ran inside a lock" what
    | Some why -> starts_with ~msg:what ~prefix:(what ^ ": called from") why
  in
  refused "Rig_amd_abi.Capability.trace" (fun () -> ignore (c.trace ()));
  match c.compute with
  | Aql { scratch } ->
      refused "Rig_amd_abi.Capability.scratch" (fun () -> ignore (scratch 16));
      equal ~msg:"a scratch outside a lock" (result unit string) (Ok ())
        (scratch 16)
  | Pm4 -> ()

let () =
  exit
    (run "rig_amd interrupt"
       [
         group ~timeout "a path's lock"
           [
             test
               "a callback's take inside it raises Invalid_argument, having \
                taken nothing, and one outside it runs"
               test_lock_reentered;
             test "an exception at any allocation inside it gives it back"
               test_lock_interrupted;
             test
               "the capability's calls from inside a lock raise \
                Invalid_argument as they start"
               test_capability_held;
           ];
         group ~timeout "an exception at any allocation"
           [
             test
               "of an image's entry leaves its unload to free every launch"
               test_entries;
             test "of an image's unload leaves no launch held" test_unload;
             test "of a view leaves every view and region to go back"
               test_views;
             test "of a view's free leaves the view to a later free"
               test_view_free;
           ];
       ])
