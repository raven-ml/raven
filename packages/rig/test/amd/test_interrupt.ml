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

(* Opens *)

module E = S.Edge
module H = Rig_gpu_support.Host

(* A PM4 NOP one word long, as amdgpu pads compute rings with. *)
let pm4_nop = 0xffff1000

(* An allocation's landing that raises [Exit] at it alone, as one
   asynchronous exception does: the handlers after it run. *)
let raise_once = Rig_support.Call (fun () -> raise Exit)

(* Whether the allocation [a] is made inside one of the functions [fs], named
   with their module, such as ["Rig_amd_amdgpu.acquire"]. A walk lands only
   there where landing at each of an open's thousands of allocations would
   take minutes. *)
let inside fs (a : Gc.Memprof.allocation) =
  let named s =
    match Printexc.Slot.name s with
    | Some n ->
        List.exists (fun f -> n = f || String.starts_with ~prefix:(f ^ ".") n) fs
    | None -> false
  in
  match Printexc.backtrace_slots a.callstack with
  | Some slots -> Array.exists named slots
  | None -> false

(* Skips but on a GPU amdgpu holds. *)
let needs_amdgpu () =
  if not (S.present ()) then skip ~reason:"the machine has no AMD GPU" ();
  if S.driverless () then skip ~reason:"the walk is of amdgpu's path" ()

(* GPU 0, opened through the path the suites select. *)
let open_gpu () = Rig_gpu_support.opened (S.gpu ())

let amdgpu fs = inside (List.map (fun f -> "Rig_amd_amdgpu." ^ f) fs)

(* [ran g] is [Ok ()] once [g] ran a NOP on its compute queue and a copy of
   64 bytes of [Pinned] memory on its copy queue within 10 s, or why not.
   It stops [g] either way. *)
let ran g =
  let host r = Option.get (A.locate r).host in
  let address r = Option.get (A.locate r).address in
  let src = A.alloc g Pinned 64 and dst = A.alloc g Pinned 64 in
  let outcome =
    match (src, dst) with
    | (None, _ | _, None) ->
        Option.iter (A.free g) src;
        Option.iter (A.free g) dst;
        Error "no Pinned memory"
    | Some src, Some dst -> (
        H.write (host src) (String.make 64 'f');
        H.write (host dst) (String.make 64 '\000');
        let v = A.signaled g + 1 in
        let parts =
          [|
            E.words ~queue:"COMPUTE:0" [| pm4_nop |];
            E.copy ~dst:(address dst) ~src:(address src) 64;
          |]
        in
        match E.submit g ~v parts with
        | `Failed why -> Error ("the submit failed: " ^ why)
        | `Ok ->
            let rec wait n =
              A.signaled g >= v
              || n > 0
                 && begin
                   A.sleep g ~seen:(A.signaled g) ~still_ms:500;
                   wait (n - 1)
                 end
            in
            if not (wait 20) then Error "the work did not end within 10 s"
            else begin
              let copied = H.read (host dst) 64 in
              A.free g src;
              A.free g dst;
              if copied = String.make 64 'f' then Ok ()
              else Error "the copy did not land"
            end)
  in
  A.stop g ~fault:None;
  outcome

(* [again what] checks that GPU 0 opens and runs work. *)
let again what =
  match open_gpu () with
  | Error why -> failf "%s, the GPU does not open: %s" what why
  | Ok g -> (
      match ran g with Ok () -> () | Error why -> failf "%s, %s" what why)

(* The exit codes of a walk's child: it landed, and the GPU opened again and
   ran work; it did not land, and the GPU ran work. Any other is a failure,
   whose reason the child printed. *)
let landed_code = 0
let ended_code = 1

(* The child of a walk of the process's first open: the open of GPU 0 with
   every allocation [from] holds for from the [n]-th on raising until it
   returns, then another open that runs work. *)
let first_open from n =
  match
    let opened = ref (Error "not reached") in
    let reached =
      Rig_support.interrupted ~from n (fun () -> opened := open_gpu ())
    in
    Result.iter (fun g -> A.stop g ~fault:None) !opened;
    match open_gpu () with
    | Error why -> Error ("the GPU does not open again: " ^ why)
    | Ok g -> Result.map (fun () -> reached) (ran g)
  with
  | Ok true -> landed_code
  | Ok false -> ended_code
  | Error why ->
      prerr_endline why;
      2
  | exception e ->
      prerr_endline (Printexc.to_string e);
      3

(* The process's first open, interrupted at each allocation of the steps
   that take KFD's state for the process: its file, the GPU's address space
   (ACQUIRE_VM, which KFD refuses through a second render node) and the event
   page (which KFD takes once). The topology's reads, which take no state
   and make most of those allocations, are passed over. A child forks for
   each, since KFD's state is the process's. The next open takes the GPU and
   runs work. *)
let test_first_open () =
  needs_amdgpu ();
  Rig_support.needs_fork ();
  let topology = inside [ "Rig_amd_amdgpu__Topology" ] in
  let walk fs =
    let taking a = amdgpu fs a && not (topology a) in
    let rec from n =
      match Unix.fork () with
      | 0 -> Unix._exit (first_open taking n)
      | pid -> (
          match Unix.waitpid [] pid with
          | _, WEXITED c when c = landed_code -> from (n + 1)
          | _, WEXITED c when c = ended_code -> n - 1
          | _, WEXITED c ->
              failf "interrupted from allocation %d of %s, the child exited %d"
                n (String.concat ", " fs) c
          | _, (WSIGNALED s | WSTOPPED s) ->
              failf
                "interrupted from allocation %d of %s, the child stopped on \
                 signal %d"
                n (String.concat ", " fs) s)
    in
    at_least ~msg:(String.concat ", " fs ^ ": the landings") int ~than:1
      (from 1)
  in
  walk [ "kfd_fd"; "acquire" ];
  walk [ "page_for" ]

(* A later open, with one exception at each allocation of the path's making
   of a queue: the path's stop, which the device calls on the exception,
   destroys the queue, so the process holds none, and the GPU opens again. *)
let test_queue_raised () =
  needs_amdgpu ();
  again "before the walk";
  let none = Some 0 in
  equal ~msg:"the KFD queues held before the walk" (option int) none
    (S.queues ());
  let rec walk n =
    let opened = ref (Error "not reached") in
    let reached =
      Rig_support.interrupted ~from:(amdgpu [ "queue" ]) ~landing:raise_once n
        (fun () -> opened := open_gpu ())
    in
    Result.iter (fun g -> A.stop g ~fault:None) !opened;
    equal
      ~msg:(strf "from allocation %d, the KFD queues held" n)
      (option int) none (S.queues ());
    again (strf "from allocation %d" n);
    if reached then walk (n + 1) else n - 1
  in
  at_least ~msg:"the landings" int ~than:1 (walk 1)

(* Over PCI *)

(* Skips but on a GPU taken over PCI. *)
let needs_pci () =
  if not (S.present ()) then skip ~reason:"the machine has no AMD GPU" ();
  if not (S.driverless ()) then skip ~reason:"the walk is of AMD over PCI" ()

(* The allocations of [f ()] that [from] holds for, on every thread: a walk
   counts those of the calling thread, the only one an open or a stop
   allocates on. *)
let allocations from f =
  let n = ref 0 in
  let count a =
    if from a then incr n;
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

(* The landings of a walk over a boot's or a stop's allocations, each of
   which costs a boot of the GPU. *)
let spread = 3

(* [walk_open ~step from] walks an open of GPU 0 with every allocation
   [from] holds for from the [(k step + 1)]-th on raising until the open
   returns, for [k] from 0 on: the GPU opens again after each and runs work.
   It is the number of landings. *)
let walk_open ~step from =
  let rec walk landed n =
    let opened = ref (Error "not reached") in
    let reached =
      Rig_support.interrupted ~from n (fun () -> opened := open_gpu ())
    in
    Result.iter (fun g -> A.stop g ~fault:None) !opened;
    again (strf "interrupted from allocation %d" n);
    if reached then walk (landed + 1) (n + step) else landed
  in
  walk 0 1

(* An open whose boot an exception ends from before its first write to a
   register to after its return: the path's steps around the boot, at each
   of their allocations, and the boot, at [spread] of its allocations. rig.pci
   resets the function at the next open of a GPU the boot wrote to, which
   then boots it and runs work. *)
let test_pci_open () =
  needs_pci ();
  again "before the walk";
  let boot = inside [ "Rig_amd_pci__Boot.start" ] in
  let around a =
    inside [ "Rig_amd_pci.boot"; "Rig_amd_pci.booted" ] a && not (boot a)
  in
  at_least ~msg:"the landings around the boot" int ~than:1
    (walk_open ~step:1 around);
  let opened = ref (Error "not opened") in
  let n = allocations boot (fun () -> opened := open_gpu ()) in
  (match !opened with
  | Ok g -> A.stop g ~fault:None
  | Error why -> fail ("the GPU does not open: " ^ why));
  at_least ~msg:"the landings in the boot" int ~than:spread
    (walk_open ~step:(max 1 (n / spread)) boot)

(* A stop that an exception ends at [spread] of its allocations: rig.pci
   holds the stop as unknown and resets the function at the next open, which
   boots the GPU and runs work. *)
let test_pci_stop () =
  needs_pci ();
  let from = inside [ "Rig_amd_pci__Boot.stop" ] in
  let fresh what =
    match open_gpu () with
    | Ok g -> g
    | Error why -> failf "%s, the GPU does not open: %s" what why
  in
  let g = fresh "for the count" in
  let n = allocations from (fun () -> A.stop g ~fault:None) in
  let step = max 1 (n / spread) in
  let rec walk landed k =
    let g = fresh (strf "for allocation %d" k) in
    let reached =
      Rig_support.interrupted ~from k (fun () -> A.stop g ~fault:None)
    in
    again (strf "interrupted from allocation %d" k);
    if reached then walk (landed + 1) (k + step) else landed
  in
  at_least ~msg:"the landings" int ~than:spread (walk 0 1)

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
               "of the process's first open leaves the next open to take the \
                GPU"
               test_first_open;
             test
               "of the path's making of a queue leaves the process no queue"
               test_queue_raised;
             test
               "of an open over PCI from its boot's first write on leaves the \
                next open to boot the GPU"
               test_pci_open;
             test "of a stop over PCI leaves the next open to boot the GPU"
               test_pci_stop;
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
