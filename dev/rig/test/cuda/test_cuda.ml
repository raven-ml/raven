(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let strf = Printf.sprintf
let watchdog = 17 (* CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT *)
let second = 1_000_000_000
let address r = Option.get (C.address r)
let host r = Option.get (C.host r)
let still = Rig_gpu_support.still
let word g = host (C.word g)

module B = Rig.Buffer

(* The reason of the loss [f ()] raises. *)
let lost f =
  match f () with _ -> failf "no loss" | exception Rig.Lost (_, why) -> why

(* Opening *)

let gpu_once () =
  let t = S.open_ () in
  let e = require_error (C.open_ 0) in
  contains ~sub:"open" e;
  S.close t;
  C.stop (require_ok (C.open_ 0))

let opening =
  group ~timeout:60. "opening"
    [
      test "count never raises" (fun () -> at_least int ~than:0 (C.count ()));
      test "names GPU 0 CUDA and GPU i CUDA:i" (fun () ->
          equal (list string)
            [ "CUDA"; "CUDA:1"; "CUDA:7" ]
            (List.map C.device_name [ 0; 1; 7 ]));
      test "a negative GPU raises" (fun () ->
          raises_match Exn.invalid_arg (fun () -> C.open_ (-1));
          raises_match Exn.invalid_arg (fun () -> C.device_name (-1)));
      test "a GPU past the count is an error naming the count" (fun () ->
          let n = C.count () in
          let e = require_error (C.open_ n) in
          contains ~sub:(if n = 0 then "CUDA" else strf "CUDA sees %d" n) e);
      test "a GPU has one device until it is closed" gpu_once;
    ]

(* Facts *)

let facts () =
  S.with_ @@ fun { g; _ } ->
  let arch = C.arch g in
  equal int ~msg:"length of the arch" 5 (String.length arch);
  starts_with ~affix:"sm_" arch;
  greater int ~than:0 (C.budget g);
  equal (list string) [ "COMPUTE:0"; "COPY:0" ] (C.queues g);
  equal bool ~msg:"completion is the store" true (C.completion g = `Store);
  equal (list bool) [ true; false; true ]
    (List.map (C.waits_on g) [ `Store; `Object; `Host ]);
  equal bool ~msg:"submit may block" true (C.blocks g = `May_block);
  equal nativeint ~msg:"the word's handle is its host address"
    (Nativeint.of_int (word g))
    (C.handle (C.word g));
  equal int ~msg:"the word starts at 0" 0 (C.signaled g)

let symbols () =
  S.with_ @@ fun { g; _ } ->
  let symbol = (C.capability g).symbol in
  let found n = Option.is_some (symbol n) in
  equal (list bool)
    [ true; true; true; true; false; false ]
    (List.map found
       [
         "cuLaunchKernel";
         "cuMemcpyAsync";
         "cuLaunchHostFunc";
         "cuGraphLaunch";
         "cuNoSuchFunction";
         "cuLaunchKernel\000";
       ]);
  equal bool ~msg:"the key is the ABI's" true
    (Option.is_some (Type.Id.provably_equal C.capability_key Rig_cuda_abi.key))

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "finds the functions a fill calls" symbols;
    ]

(* Memory *)

let kinds = [ `Device; `Pinned; `Mapped ]

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | `Device -> "`Device"
    | `Pinned -> "`Pinned"
    | `Mapped -> "`Mapped")

let kind = Gen.of_list ~pp:pp_kind kinds
let size = Gen.of_list ~pp:Format.pp_print_int [ 1; 7; 4096; (1 lsl 20) + 7 ]
let offset = Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ]

let pattern n seed =
  String.init n (fun i -> Char.chr (((i * 7) + seed) land 255))

let memory_of = function
  | `Device -> B.Device
  | `Pinned -> B.Pinned
  | `Mapped -> B.Mapped

(* host -> a -> b -> host through copies, on both queues. *)
let round_trip (ka, kb, n, (oa, ob)) =
  S.with_ @@ fun ({ d; g } as t) ->
  let r = require_some (C.alloc g ka 1) in
  equal bool ~msg:"host of a" (ka <> `Device) (Option.is_some (C.host r));
  C.free g r;
  let at k o =
    B.view (B.create ~memory:(memory_of k) d (n + o)) ~first:o ~length:n
  in
  let src = at `Pinned 0 and dst = at `Pinned 0 in
  let a = at ka oa and b = at kb ob in
  let data = pattern n (n + oa) in
  H.write (B.address src) data;
  let v =
    S.submit t
      [|
        S.copy ~queue:"COPY:0" ~dst:a src;
        S.copy ~queue:"COMPUTE:0" ~after:[| 0 |] ~dst:b a;
        S.copy ~queue:"COPY:0" ~after:[| 1 |] ~dst b;
      |]
  in
  S.wait t v;
  equal string data (H.read (B.address dst) n)

let past_memory () =
  S.with_ @@ fun { g; _ } -> is_none (C.alloc g `Device (2 * C.budget g))

let memory =
  group ~timeout:120. "memory"
    [
      prop ~count:30 "copies through any two kinds of memory are the identity"
        (Gen.quad kind kind size (Gen.pair offset offset))
        round_trip;
      test "an allocation past the GPU's memory is None" past_memory;
    ]

(* Work *)

let fills_in_a_fresh_domain () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let n = 1000 in
  let out = require_some (C.alloc g `Pinned (4 * n)) in
  let f = S.launch (kernel "double_index") ~grid:4 ~block:256 (address out) n in
  let r, current =
    Domain.join
      (Domain.spawn (fun () ->
           let v = S.submit t [| S.part ~queue:"COMPUTE:0" f |] in
           (v, S.current ())))
  in
  equal int ~msg:"the value" 1 r;
  equal nativeint ~msg:"the domain's thread has no context after" 0n current;
  not_equal nativeint ~msg:"the fill saw a context" 0n (S.seen f);
  S.wait t 1;
  for i = 0 to n - 1 do
    equal int ~msg:(strf "word %d" i) (2 * i) (H.get32 (host out + (4 * i)))
  done;
  C.unload g m;
  C.free g out

(* A kernel writes host memory through the address map_host gives for a range
   inside a registered one, away from its start. *)
let kernel_through_map_host () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let n = 1000 in
  let p = H.pages (2 * H.page) in
  let whole = require_some (C.map_host g p (2 * H.page)) in
  let at = p + H.page + 64 in
  let inside = require_some (C.map_host g at (4 * n)) in
  let f =
    S.launch (kernel "double_index") ~grid:4 ~block:256 (address inside) n
  in
  S.wait t (S.submit t [| S.part ~queue:"COMPUTE:0" f |]);
  equal (list int)
    (List.init n (fun i -> 2 * i))
    (List.init n (fun i -> H.get32 (at + (4 * i))));
  C.free g inside;
  C.free g whole;
  C.unload g m;
  H.free_pages p (2 * H.page)

(* Returns once [t]'s word reached [v], which a lost device's stop brings it
   to. *)
let drained (t : S.t) v =
  while Rig.signaled t.d < v do
    Domain.cpu_relax ()
  done

(* A copy on COPY:0, then a fill on COMPUTE:0 that fails: the failed value is
   still reached after the copy. *)
let failed_fill () =
  S.with_ @@ fun ({ d; _ } as t) ->
  let src = B.create d 64 and dst = B.create ~memory:Pinned d 64 in
  let data = pattern 64 3 in
  S.write_gpu (Nativeint.of_int (B.address src)) data;
  let parts =
    [|
      S.copy ~queue:"COPY:0" ~dst src; S.part ~queue:"COMPUTE:0" (S.failing 1);
    |]
  in
  let why = lost (fun () -> S.submit t parts) in
  starts_with ~affix:"running a fill: CUDA_ERROR_INVALID_VALUE: " why;
  S.close t;
  drained t 1;
  equal string ~msg:"copied before the word" data (H.read (B.address dst) 64)

(* A fill that fails behind a 100 ms kernel: the loss's stop finds the kernel
   running, and the word still reaches the failed value. *)
let failed_behind_work () =
  S.with_ @@ fun ({ g; _ } as t) ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g `Pinned 8) in
  H.set64 (host flag) 0;
  Fun.protect ~finally:(fun () -> H.set64 (host flag) 1) @@ fun () ->
  let _, kernel = S.kernels g in
  let spin =
    S.launch (kernel "spin") ~grid:1 ~block:1 (address flag) (second / 10)
  in
  let parts =
    [|
      S.part ~queue:"COMPUTE:0" spin; S.part ~queue:"COMPUTE:0" (S.failing 1);
    |]
  in
  ignore (lost (fun () -> S.submit t parts));
  H.set64 (host flag) 1;
  S.close t;
  drained t 1

let misuse () =
  S.with_ @@ fun { g; _ } ->
  let r = require_some (C.alloc g `Device 64) in
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  raises "alloc of 0 bytes" (fun () -> C.alloc g `Device 0);
  raises "map_host of 0 bytes" (fun () -> C.map_host g (word g) 0);
  raises "peer of one device" (fun () -> C.peer g g);
  raises "map_peer of one device" (fun () -> C.map_peer g g r);
  C.free g r;
  raises "free twice" (fun () -> C.free g r);
  let p = H.pages H.page in
  let m = require_some (C.map_host g p 64) in
  C.free g m;
  raises "free of a mapping twice" (fun () -> C.free g m);
  H.free_pages p H.page

let another_device () =
  let a = S.open_ () in
  let r = require_some (C.alloc a.g `Pinned 64) in
  S.close a;
  S.with_ @@ fun b ->
  raises_match Exn.invalid_arg (fun () -> C.free b.g r);
  C.free a.g r

(* Parts CUDA's queues do not run: ring words, ring units and segment bytes
   are refused, and no value is assigned. *)
let refused () =
  S.with_ @@ fun ({ d; _ } as t) ->
  let fill ~units ~bytes =
    match S.part ~queue:"COMPUTE:0" (S.failing 0) with
    | { work = Fill f; _ } as p ->
        {
          p with
          work = Fill { f with ring_units = units; segment_bytes = bytes };
        }
    | p -> p
  in
  let words =
    {
      Rig.Submission.queue = "COMPUTE:0";
      after = [||];
      work = Words (B.create Rig.host 8);
    }
  in
  let refuses msg p =
    raises_match ~msg
      (Exn.invalid_arg ~substring:"never fit")
      (fun () -> S.submit t [| p |])
  in
  refuses "ring words" words;
  refuses "a ring unit" (fill ~units:1 ~bytes:0);
  refuses "a segment byte" (fill ~units:0 ~bytes:1);
  equal int ~msg:"values assigned" 0 (Rig.submitted d)

let work =
  group ~timeout:60. "work"
    [
      test "a fill runs with the context current from a fresh domain"
        fills_in_a_fresh_domain;
      test "a kernel addresses host memory where map_host says"
        kernel_through_map_host;
      test "a failed fill loses the device, and its value still drains"
        failed_fill;
      test "a value failed behind running work drains" failed_behind_work;
      test "misuse raises" misuse;
      test "a region of another device raises" another_device;
      test "a submission of work the device does not run raises" refused;
    ]

(* Images *)

let images () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m = S.loaded (require_ok (C.image g (S.fixture "kernels.ptx"))) in
  equal bool ~msg:"double_index" true
    (Option.is_some (C.entry m "double_index"));
  equal (option int) ~msg:"a missing kernel" None (C.entry m "missing");
  let e = require_error (C.image g "not a module") in
  starts_with ~affix:"loading the image: CUDA_ERROR_" e;
  (match C.image g (S.fixture "kernels.cubin") with
  | Ok i ->
      equal string ~msg:"the cubin's GPU" "sm_89" (C.arch g);
      C.unload g (S.loaded i)
  | Error e ->
      not_equal string ~msg:"the cubin's GPU" "sm_89" (C.arch g);
      starts_with ~affix:"loading the image: CUDA_ERROR_" e);
  let f = S.launch (Option.get (C.entry m "empty")) ~grid:1 ~block:1 0 0 in
  S.wait t (S.submit t [| S.part ~queue:"COMPUTE:0" f |]);
  C.unload g m;
  raises_match Exn.invalid_arg (fun () -> C.entry m "empty");
  raises_match Exn.invalid_arg (fun () -> C.unload g m)

(* Loading places every function's code, whatever CUDA_MODULE_LOADING says:
   a function CUDA loads lazily, at its [entry], could fail there for lack of
   memory. One entry, which loads its own function, shows the others. *)
let loads_every_function () =
  S.with_ @@ fun { g; _ } ->
  let m = S.loaded (require_ok (C.image g (S.fixture "kernels.ptx"))) in
  equal (pair int int) (5, 5)
    (S.functions_loaded (Option.get (C.entry m "empty")));
  C.unload g m

(* An entry launches with as much dynamic shared memory as the GPU allows
   a block, beyond the 48 KiB CUDA allows by default. *)
let largest_shared_memory () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m = S.loaded (require_ok (C.image g (S.fixture "kernels.ptx"))) in
  (* cuda.h's CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN. *)
  let shared = S.attribute 97 in
  let f =
    S.launch ~shared (Option.get (C.entry m "empty")) ~grid:1 ~block:1 0 0
  in
  S.wait t (S.submit t [| S.part ~queue:"COMPUTE:0" f |]);
  C.unload g m

let images =
  group ~timeout:60. "images"
    [
      test "load, find their kernels and unload" images;
      test "loading places every function's code" loads_every_function;
      test "an entry takes the largest shared memory" largest_shared_memory;
    ]

(* Timeline and loss *)

(* A part that runs a kernel until [flag]'s first word is not 0, for at most 10
   seconds, and the image it loaded. *)
let spinning (t : S.t) flag =
  let m, kernel = S.kernels t.g in
  H.set64 (host flag) 0;
  let f =
    S.launch (kernel "spin") ~grid:1 ~block:1 (address flag) (10 * second)
  in
  (m, S.part ~queue:"COMPUTE:0" f)

(* [spin t flag] submits {!spinning} as value 1. *)
let spin t flag =
  let m, p = spinning t flag in
  equal int ~msg:"the value" 1 (S.submit t [| p |]);
  m

(* [lose t flag] submits {!spinning}, then a fill that fails: the loss stops
   [t] while the kernel runs. *)
let lose t flag =
  let _, p = spinning t flag in
  let fails = S.part ~queue:"COMPUTE:0" (S.failing 1) in
  ignore (lost (fun () -> S.submit t [| p; fails |]))

(* GPU 0, opened once the work a stop left running ended. *)
let rec reopened () =
  match C.open_ 0 with
  | Ok g -> g
  | Error _ ->
      Domain.cpu_relax ();
      reopened ()

let long_work () =
  S.with_ @@ fun ({ g; _ } as t) ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g `Pinned 8) in
  Fun.protect ~finally:(fun () -> H.set64 (host flag) 1) @@ fun () ->
  let m = spin t flag in
  equal int ~msg:"the work committed" 0 (Rig.signaled t.d);
  C.sleep g ~seen:0 ~still_ms:50;
  equal int ~msg:"the work still runs" 0 (C.signaled g);
  H.set64 (host flag) 1;
  S.wait t 1;
  C.sleep g ~seen:0 ~still_ms:60_000;
  C.unload g m

(* A collection needs every domain at a safe point, so it waits for a domain
   whose C call holds the runtime: here, an unload that CUDA holds until the
   GPU's running work ends. *)
let unload_aside () =
  S.with_ @@ fun ({ g; _ } as t) ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g `Pinned 8) in
  Fun.protect ~finally:(fun () -> H.set64 (host flag) 1) @@ fun () ->
  let other, _ = S.kernels g in
  let m = spin t flag in
  equal int ~msg:"the work committed" 0 (Rig.signaled t.d);
  let unloading = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        Atomic.set unloading true;
        C.unload g other)
  in
  while not (Atomic.get unloading) do
    Domain.cpu_relax ()
  done;
  still ~msg:"the work while unload waits" int 0
    (fun () -> C.signaled g)
    ~ms:50;
  Gc.full_major ();
  equal int ~msg:"the work after a collection" 0 (C.signaled g);
  H.set64 (host flag) 1;
  Domain.join d;
  S.wait t 1;
  C.unload g m

(* The 256 MiB global of fixtures/global.ptx, loaded on [g] and touched by a run
   of its kernel, as the value after [g]'s last. *)
let global = 256 * 1024 * 1024

let load_global t =
  let m = S.loaded (require_ok (C.image t.S.g (S.fixture "global.ptx"))) in
  let touch = S.launch (Option.get (C.entry m "touch")) ~grid:1 ~block:1 0 0 in
  S.wait t (S.submit t [| S.part ~queue:"COMPUTE:0" touch |]);
  m

(* A stop unloads no image: the global's memory returns at the unload that
   follows it. *)
let unload_after_stop () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let before = S.free_memory () in
  let m = load_global t in
  at_most int ~msg:"loaded" ~than:(before - (global / 2)) (S.free_memory ());
  S.close t;
  at_most int ~msg:"stopped" ~than:(before - (global / 2)) (S.free_memory ());
  C.unload g m;
  at_least int ~msg:"unloaded" ~than:(before - (global / 2)) (S.free_memory ())

(* A stop that finds work running leaves the GPU to that work; once it ended,
   the GPU opens again, and the old device's unload then returns its image's
   memory and leaves the new device working. *)
let unload_after_reopen () =
  S.with_ @@ fun ({ g; _ } as t) ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g `Pinned 8) in
  Fun.protect ~finally:(fun () -> H.set64 (host flag) 1) @@ fun () ->
  let before = S.free_memory () in
  let m = load_global t in
  lose t flag;
  S.close t;
  H.set64 (host flag) 1;
  C.stop (reopened ());
  S.with_ @@ fun t' ->
  at_most int ~msg:"reopened" ~than:(before - (global / 2)) (S.free_memory ());
  C.unload g m;
  at_least int ~msg:"unloaded" ~than:(before - (global / 2)) (S.free_memory ());
  S.wait t' (S.submit t' [||])

let stop_idle () =
  S.with_ @@ fun ({ d; g } as t) ->
  let p = H.pages H.page in
  let r = require_some (C.map_host g p 64) in
  equal bool ~msg:"locked" true (S.locked p);
  S.wait t (S.submit t [||]);
  S.close t;
  equal int ~msg:"the word" 1 (Rig.signaled d);
  C.free g r;
  equal bool ~msg:"locked after free" false (S.locked p);
  H.free_pages p H.page

let stop_running () =
  S.with_ @@ fun ({ g; _ } as t) ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g `Pinned 8) in
  Fun.protect ~finally:(fun () -> H.set64 (host flag) 1) @@ fun () ->
  lose t flag;
  S.close t;
  let e = require_error (C.open_ 0) in
  contains ~sub:"still runs" e;
  H.set64 (host flag) 1;
  C.stop (reopened ())

let registry_is_the_process () =
  let p = H.pages H.page in
  let a = S.open_ () in
  let ra = require_some (C.map_host a.g p 256) in
  S.close a;
  S.with_ @@ fun b ->
  let rb = require_some (C.map_host b.g (p + 64) 64) in
  C.free a.g ra;
  equal bool ~msg:"locked after the first free" true (S.locked p);
  C.free b.g rb;
  equal bool ~msg:"locked after the last free" false (S.locked p);
  H.free_pages p H.page

(* Commits *)

(* Encoded work starts with no other call: a kernel stores into pinned memory
   the host reads, while nothing waits for its value. *)
let runs_uncommitted () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let _, kernel = S.kernels g in
  let out = require_some (C.alloc g `Pinned 8) in
  H.set64 (host out) 0;
  let f = S.launch (kernel "step") ~grid:1 ~block:1 (address out) 0 in
  let v = S.submit t [| S.part ~queue:"COMPUTE:0" f |] in
  let t0 = Rig.Profile.now () in
  while H.get64 (host out) = 0 do
    if Rig.Profile.now () - t0 > 10 * second then
      fail "the kernel did not run within 10 s";
    Domain.cpu_relax ()
  done;
  S.wait t v;
  C.free g out

(* Submits once on [d] a submission naming a hold of [m] whose release sets
   [released], and drops both. *)
let[@inline never] submit_held d m released =
  let h = Rig.Hold.make ~release:(fun () -> Atomic.set released true) [ m ] in
  let s = Rig.Submission.make ~hold:h ~reads:0 ~writes:0 d [||] in
  ignore (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])

(* The device commits on its own every 64 values: a hold's release runs in a
   drain once its value is reached, with no wait for it. *)
let lag_bounded () =
  S.with_ @@ fun ({ d; _ } as t) ->
  let m = B.create d 64 and released = Atomic.make false in
  submit_held d m released;
  for _ = 1 to 64 do
    ignore (S.submit t [||])
  done;
  let t0 = Rig.Profile.now () in
  while
    Gc.full_major ();
    ignore (Sys.opaque_identity (B.create d 8));
    not (Atomic.get released)
  do
    if Rig.Profile.now () - t0 > 10 * second then
      fail "the hold was not released within 10 s";
    Domain.cpu_relax ()
  done

(* A value on COPY:0 starts after the values before it on COMPUTE:0, which ended
   on the other stream: it copies what a late compute copy wrote. *)
let copy_after_compute () =
  S.with_ @@ fun ({ d; g } as t) ->
  let _, kernel = S.kernels g in
  let flag = require_some (C.alloc g `Pinned 8) in
  H.set64 (host flag) 0;
  let data = String.init 256 (fun i -> Char.chr (i land 255)) in
  let src = B.create d 256 and mid = B.create d 256 in
  let out = B.create ~memory:Pinned d 256 in
  S.write_gpu (Nativeint.of_int (B.address src)) data;
  H.write (B.address out) (String.make 256 ' ');
  let late =
    S.delayed ~spin:(kernel "spin") ~flag:(address flag) ~ns:(second / 20)
      ~dst:(B.address mid) ~src:(B.address src) 256
  in
  ignore (S.submit t [| S.part ~queue:"COMPUTE:0" late |]);
  let v = S.submit t [| S.copy ~queue:"COPY:0" ~dst:out mid |] in
  S.wait t v;
  equal string ~msg:"the copied bytes" data (H.read (B.address out) 256);
  C.free g flag

let commits =
  group ~timeout:60. "commits"
    [
      test "encoded work runs while nothing waits for it" runs_uncommitted;
      test "the device commits on its own within 64 values" lag_bounded;
      test "a value on the copy stream follows the compute values before it"
        copy_after_compute;
    ]

let timeline =
  group ~timeout:60. "timeline"
    [
      test "long work is no fault, and a stale seen returns at once" long_work;
      test
        "unload lets other domains run while CUDA waits for the GPU (sampled)"
        unload_aside;
      test "a close of an idle device leaves the word at the last value"
        stop_idle;
      test "a loss that stops running work leaves the GPU closed until it ends"
        stop_running;
      test "page-locking is the process's" registry_is_the_process;
      test "an image a close left loaded is unloaded after it"
        unload_after_stop;
      test "an unload after the GPU opened again leaves the new device working"
        unload_after_reopen;
    ]

(* Two GPUs *)

let two_gpus () =
  if C.count () < 2 then skip ~reason:"CUDA sees fewer than two GPUs" ();
  S.with_ @@ fun { g = a; _ } ->
  let b = require_ok (C.open_ 1) in
  Fun.protect ~finally:(fun () -> C.stop b) @@ fun () ->
  let h = require_some (C.alloc b `Pinned 64) in
  let d = require_some (C.alloc b `Device 64) in
  let ph = require_some (C.map_peer a b h) in
  equal (option int) ~msg:"host memory maps" (C.host h) (C.host ph);
  let pd = C.map_peer a b d in
  equal bool ~msg:"peer is map_peer's answer" (C.peer a b) (Option.is_some pd);
  (match pd with Some pd -> C.free a pd | None -> ());
  C.free a ph;
  C.free b h;
  C.free b d

let two =
  group ~timeout:60. "two GPUs" [ test "map each other's memory" two_gpus ]

(* Graphs *)

let graph g ks = (C.capability g).graph ks
let stopped_graph = "making the graph: the device was stopped"

(* A 64-bit word of pinned memory, zeroed. *)
let counter g =
  let r = require_some (C.alloc g `Pinned 8) in
  H.set64 (host r) 0;
  r

(* [launch t gr us] submits a fill that updates [gr]'s nodes [us] and launches
   it, and is its value. *)
let launch t gr us =
  S.submit t [| S.part ~queue:"COMPUTE:0" (S.graph_launch gr us) |]

(* The steps [first], [first + 1], ... of [n] nodes on the word [out]. *)
let steps step out ~first n =
  Array.init n (fun j -> S.kernel step (address out) (first + j))

let chain n =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let out = counter g in
  let gr = require_ok (graph g (steps (kernel "step") out ~first:0 n)) in
  equal int ~msg:"its nodes" n (Array.length gr.nodes);
  S.wait t (launch t gr [||]);
  equal int ~msg:"the word" n (H.get64 (host out));
  gr.release ();
  C.unload g m;
  C.free g out

let arguments () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let out = require_some (C.alloc g `Pinned 256) in
  H.write (host out) (String.make 256 '\000');
  let k = S.kernel (kernel "double_index") ~block:64 (address out) 37 in
  let gr = require_ok (graph g [| k |]) in
  S.wait t (launch t gr [||]);
  equal (list int) ~msg:"words 0 to 39"
    (List.init 40 (fun i -> if i < 37 then 2 * i else 0))
    (List.init 40 (fun i -> H.get32 (host out + (4 * i))));
  gr.release ();
  C.unload g m;
  C.free g out

(* Runs in flight, each on one of two words and updating every node first: each
   word counts its runs' steps only if every launch ran the parameters of its
   own fill, and none of a later fill. *)
let width = 8

let updates schedule =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let step = kernel "step" in
  let outs = [| counter g; counter g |] in
  let gr = require_ok (graph g (steps step outs.(0) ~first:0 width)) in
  let runs = [| 0; 0 |] in
  let last = ref 0 in
  List.iteri
    (fun r b ->
      cover "a run on the other word than the run before"
        (r > 0 && b <> List.nth schedule (r - 1));
      cover "a run on the same word as the run before"
        (r > 0 && b = List.nth schedule (r - 1));
      let ks = steps step outs.(b) ~first:(runs.(b) * width) width in
      runs.(b) <- runs.(b) + 1;
      last := launch t gr (Array.mapi (fun j k -> (j, k)) ks))
    schedule;
  S.wait t !last;
  equal (list int) ~msg:"each word's steps"
    (List.map (fun n -> n * width) (Array.to_list runs))
    (List.map (fun r -> H.get64 (host r)) (Array.to_list outs));
  gr.release ();
  C.unload g m;
  Array.iter (C.free g) outs

let refusals () =
  S.with_ @@ fun { g; _ } ->
  let m, kernel = S.kernels g in
  let step = kernel "step" in
  let k = S.kernel step 0 0 in
  let invalid ks = raises_match Exn.invalid_arg (fun () -> graph g ks) in
  invalid [| { k with grid = (0, 1, 1) } |];
  invalid [| { k with grid = (1, 1, 1 lsl 32) } |];
  invalid [| { k with block = (1, 0, 1) } |];
  invalid [| { k with shared = -1 } |];
  invalid [| { k with shared = 1 lsl 32 } |];
  invalid [| k; { k with func = step + 1 } |];
  let e = require_error (graph g [| { k with shared = (1 lsl 32) - 1 } |]) in
  starts_with ~affix:"making the graph: CUDA_ERROR_" e;
  let empty = require_ok (graph g [||]) in
  equal int ~msg:"an empty graph's nodes" 0 (Array.length empty.nodes);
  empty.release ();
  C.unload g m;
  invalid [| k |]

let released_twice () =
  S.with_ @@ fun { g; _ } ->
  let m, kernel = S.kernels g in
  let gr = require_ok (graph g [| S.kernel (kernel "empty") 0 0 |]) in
  gr.release ();
  raises_match Exn.invalid_arg gr.release;
  C.unload g m

(* A graph's release follows its device's stop. *)
let release_after_stop () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let _, kernel = S.kernels g in
  let gr = require_ok (graph g [| S.kernel (kernel "empty") 0 0 |]) in
  S.wait t (launch t gr [||]);
  S.close t;
  gr.release ();
  raises_match Exn.invalid_arg gr.release

let after_stop () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let _, kernel = S.kernels g in
  let k = S.kernel (kernel "empty") 0 0 in
  S.close t;
  equal (result pass string) ~msg:"after the stop" (Error stopped_graph)
    (Result.map ignore (graph g [| k |]))

(* Graph calls on another domain while the device stops: each answers as made
   before the stop or after it. The GPU has one device at a time, so the calls
   are checked against the only orders a stop allows: graphs made, then the
   stop's answer, and nothing else, such as CUDA's error for a kernel unloaded
   under the call. *)
let beside_stop () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let _, kernel = S.kernels g in
  let ks = Array.init 64 (fun i -> S.kernel (kernel "step") 0 i) in
  let made = Atomic.make 0 in
  let rec make () =
    match graph g ks with
    | Ok gr ->
        gr.release ();
        Atomic.incr made;
        make ()
    | Error e -> e
  in
  let maker = Domain.spawn make in
  while Atomic.get made = 0 do
    Domain.cpu_relax ()
  done;
  S.close t;
  let after = Result.map ignore (graph g ks) in
  equal string ~msg:"the other domain's last answer" stopped_graph
    (Domain.join maker);
  equal (result pass string) ~msg:"after the stop" (Error stopped_graph) after

let graphs =
  group ~timeout:60. "graphs"
    [
      cases
        ~name:(strf "a chain of %d kernels runs each after the one before")
        "chain" [ 0; 1; 17; 64 ] chain;
      test "a kernel reads its argument block" arguments;
      prop ~count:30
        "a launch runs the updates of its own fill, none of a later one"
        (Gen.list ~size:(Gen.int_range 1 40) (Gen.int_range 0 1))
        updates;
      test "refuses what it cannot record" refusals;
      test "release raises when called twice" released_twice;
      test "a graph's release follows its device's stop"
        release_after_stop;
      test "a graph call after the stop answers the stop" after_stop;
      test "graph calls beside a stop answer as before or after it" beside_stop;
    ]

(* The shared device: the stateful tests' programs use one device, opened by the
   first and closed when the run ends. *)

let shared = fixture ~teardown:S.close S.open_

(* map_host: the registry *)

module Registry = struct
  type area = Arena | Foreign | Split | Read_only
  type entry = { start : int; bytes : int; mutable maps : int }
  type region = Counted of entry | Uncounted
  type t = { mutable entries : entry list; mutable regions : region list }

  let arena = 4
  let pages a n = (a / H.page, (a + n - 1) / H.page)

  let shares (a, n) e =
    let lo, hi = pages a n and lo', hi' = pages e.start e.bytes in
    lo <= hi' && lo' <= hi

  let make () = { entries = []; regions = [] }

  let map ?(cover = cover) m (area, a, n) =
    match area with
    | Split ->
        let inside lo = lo <= a && a + n <= lo + H.page in
        let one = inside 0 || inside (2 * H.page) in
        cover "a range inside one of another owner's two allocations" one;
        cover "a range across another owner's two allocations"
          (a < H.page && a + n > 2 * H.page);
        if one then m.regions <- m.regions @ [ Uncounted ];
        one
    | Read_only ->
        cover "read-only memory" true;
        false
    | Foreign ->
        cover "memory CUDA locked for another owner" true;
        m.regions <- m.regions @ [ Uncounted ];
        true
    | Arena -> (
        let inside e = e.start <= a && a + n <= e.start + e.bytes in
        match List.find_opt inside m.entries with
        | Some e ->
            cover "a range inside a registered one, elsewhere"
              (a <> e.start || n <> e.bytes);
            cover "a range equal to a registered one"
              (a = e.start && n = e.bytes);
            e.maps <- e.maps + 1;
            m.regions <- m.regions @ [ Counted e ];
            true
        | None when List.exists (shares (a, n)) m.entries ->
            let overlaps =
              List.exists
                (fun e -> a < e.start + e.bytes && e.start < a + n)
                m.entries
            in
            cover "a range overlapping a registered one" overlaps;
            cover "a range sharing only a page" (not overlaps);
            false
        | None ->
            let e = { start = a; bytes = n; maps = 1 } in
            m.entries <- e :: m.entries;
            m.regions <- m.regions @ [ Counted e ];
            true)

  let free m i =
    let r = List.nth m.regions i in
    m.regions <- List.filteri (fun j _ -> j <> i) m.regions;
    match r with
    | Uncounted -> ()
    | Counted e ->
        e.maps <- e.maps - 1;
        if e.maps = 0 then begin
          cover "the last free of a registration" true;
          m.entries <- List.filter (fun e' -> e' != e) m.entries
        end

  (* The system *)

  type sys = {
    g : C.t;
    base : int;
    foreign : C.region;
    split : int; (* pages 0 and 2 page-locked by another owner *)
    read_only : int;
    lock : Mutex.t;
    mutable live : C.region list;
  }

  let at s a = s.base + a

  let start () =
    let g = (shared ()).g in
    let foreign = Option.get (C.alloc g `Pinned (2 * H.page)) in
    let split = H.pages (3 * H.page) in
    S.register split H.page;
    S.register (split + (2 * H.page)) H.page;
    {
      g;
      base = H.pages (arena * H.page);
      foreign;
      split;
      read_only = H.pages ~read_only:true H.page;
      lock = Mutex.create ();
      live = [];
    }

  let release s =
    List.iter (C.free s.g) s.live;
    C.free s.g s.foreign;
    S.unregister s.split;
    S.unregister (s.split + (2 * H.page));
    H.free_pages s.split (3 * H.page);
    H.free_pages s.base (arena * H.page);
    H.free_pages s.read_only H.page

  let map_sys s (area, a, n) =
    let a =
      match area with
      | Arena -> at s a
      | Foreign -> host s.foreign + a
      | Split -> s.split + a
      | Read_only -> s.read_only + a
    in
    match C.map_host s.g a n with
    | Some r ->
        Mutex.protect s.lock (fun () -> s.live <- s.live @ [ r ]);
        true
    | None -> false

  let free_sys s i =
    C.free s.g (List.nth s.live i);
    Mutex.protect s.lock (fun () ->
        s.live <- List.filteri (fun j _ -> j <> i) s.live)

  (* CUDA holds every registered range locked, and no arena page that no
     registration shares. *)
  let invariant m s =
    List.iter
      (fun e ->
        equal bool
          ~msg:(strf "first byte of [%d, %d)" e.start (e.start + e.bytes))
          true
          (S.locked (at s e.start));
        equal bool
          ~msg:(strf "last byte of [%d, %d)" e.start (e.start + e.bytes))
          true
          (S.locked (at s (e.start + e.bytes - 1))))
      m.entries;
    for p = 0 to arena - 1 do
      if not (List.exists (shares (p * H.page, H.page)) m.entries) then
        equal bool ~msg:(strf "page %d" p) false (S.locked (at s (p * H.page)))
    done

  let range =
    let point = [ 0; 8; 2048; 4088; 4096; 4104; 8192; 12280 ] in
    let length = [ 8; 64; 2048; 4096; 4104; 8192 ] in
    let pp ppf (area, a, n) =
      Format.fprintf ppf "(%s, %d, %d)"
        (match area with
        | Arena -> "arena"
        | Foreign -> "foreign"
        | Split -> "split"
        | Read_only -> "read-only")
        a n
    in
    let scale x = x * H.page / 4096 in
    Gen.with_pp pp
      (Gen.frequency
         [
           ( 8,
             Gen.such_that
               (fun (_, a, n) -> a + n <= arena * H.page)
               (Gen.map
                  (fun (a, n) -> (Arena, scale a, scale n))
                  (Gen.pair (Gen.of_list point) (Gen.of_list length))) );
           (* One range drawn often, so that a range equal to a registered one
              is drawn too. *)
           (2, Gen.constant (Arena, 0, H.page));
           ( 1,
             Gen.map
               (fun a -> (Foreign, scale a, 64))
               (Gen.of_list [ 0; 8; 4096 ]) );
           (* A split range starts in another owner's page: map_host would
              page-lock one in page 1 itself. *)
           ( 2,
             Gen.such_that
               (fun (_, a, n) -> a + n <= 3 * H.page)
               (Gen.map
                  (fun (a, n) -> (Split, scale a, scale n))
                  (Gen.pair
                     (Gen.of_list [ 0; 8; 4088; 8192; 8200 ])
                     (Gen.of_list [ 8; 64; 4096; 8200; 12288 ]))) );
           (1, Gen.constant (Read_only, 0, 64));
         ])
end

let registry =
  abstract "r" ~invariant:Registry.invariant ~release:Registry.release

let mapped =
  among int registry (fun m ->
      List.init (List.length m.Registry.regions) Fun.id)

(* On two domains which ranges a call finds registered depends on the order the
   calls ran in, so the model labels nothing there. *)
let registry_commands ~cover =
  [
    command "start" (Gen.unit @-> makes registry) Registry.make Registry.start;
    command "map_host"
      (registry ^-> Registry.range @-> returns bool)
      (Registry.map ~cover) Registry.map_sys;
    command "free"
      (registry ^-> mapped ^-> returns unit)
      Registry.free Registry.free_sys;
  ]

(* Submissions: the order of values *)

module Order = struct
  (* A part copies [n] bytes of buffer [src] at [so] to buffer [dst] at [do_],
     on COPY:0 iff [copy], after the earlier parts [after] lists. A part with a
     [delay] is a fill that starts its copy that many nanoseconds late, so that
     work run out of order shows in the buffers. *)
  type part = {
    delay : int;
    copy : bool;
    src : int;
    so : int;
    dst : int;
    do_ : int;
    n : int;
    after : int list;
  }

  let buffers = 3
  let size = 65536

  let pp_part ppf p =
    Format.fprintf ppf "{%s%s %d@%d -> %d@%d n=%d after=[%s]}"
      (if p.copy then "COPY" else "COMPUTE")
      (if p.delay > 0 then strf " +%dns" p.delay else "")
      p.src p.so p.dst p.do_ p.n
      (String.concat ";" (List.map string_of_int p.after))

  let pp_one ppf ps =
    Format.fprintf ppf "[%a]"
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_part)
      ps

  let pp ppf subs =
    Format.pp_print_list ~pp_sep:Format.pp_print_space pp_one ppf subs

  let initial b =
    Bytes.init size (fun i -> Char.chr (((b * 61) + (i * 7)) land 255))

  type t = { bufs : Bytes.t array; mutable last : bool option }

  let make () = { bufs = Array.init buffers initial; last = None }

  (* Part [i] runs before part [j], [i < j]: same queue, or [j] runs after a
     part that runs after [i]. *)
  let rec before ps i j =
    let pj = List.nth ps j in
    (List.nth ps i).copy = pj.copy
    || List.exists (fun k -> k = i || (k > i && before ps i k)) pj.after

  let overlap (b, o) (b', o') n n' = b = b' && o < o' + n' && o' < o + n

  let conflict p q =
    overlap (p.dst, p.do_) (q.dst, q.do_) p.n q.n
    || overlap (p.src, p.so) (q.dst, q.do_) p.n q.n
    || overlap (p.dst, p.do_) (q.src, q.so) p.n q.n

  (* The result is one: every pair of parts that conflict is ordered, and no
     part copies over its own source. *)
  let determined_one ps =
    List.for_all
      (fun p -> not (overlap (p.src, p.so) (p.dst, p.do_) p.n p.n))
      ps
    &&
    let n = List.length ps in
    List.for_all
      (fun j ->
        List.for_all
          (fun i ->
            before ps i j || not (conflict (List.nth ps i) (List.nth ps j)))
          (List.init j Fun.id))
      (List.init n Fun.id)

  let determined _ subs = List.for_all determined_one subs

  let run_one m ps =
    let queues = List.sort_uniq compare (List.map (fun p -> p.copy) ps) in
    cover "a submission of no parts after one released on COPY"
      (ps = [] && m.last = Some true);
    cover "a submission on the queue that released the last"
      (queues <> [] && m.last = Some (List.hd (List.rev ps)).copy);
    cover "a switch of queue"
      (queues <> [] && m.last = Some (not (List.hd (List.rev ps)).copy));
    cover "two queues in one submission" (List.length queues = 2);
    cover "an after across queues"
      (List.exists
         (fun p ->
           List.exists (fun k -> (List.nth ps k).copy <> p.copy) p.after)
         ps);
    List.iter
      (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n)
      ps;
    m.last <- Some (match List.rev ps with p :: _ -> p.copy | [] -> false)

  let run m subs =
    cover "several values in flight" (List.length subs > 1);
    List.iter (run_one m) subs

  (* The system *)

  type sys = {
    t : S.t;
    buffers : B.t array;
    flag : C.region;
    image : C.image;
    spin : int;
  }

  let start () =
    let t = shared () in
    let buffers =
      Array.init buffers (fun b ->
          let r = B.create t.d size in
          S.write_gpu
            (Nativeint.of_int (B.address r))
            (Bytes.to_string (initial b));
          r)
    in
    let flag = Option.get (C.alloc t.g `Pinned 8) in
    H.set64 (host flag) 0;
    let image, kernel = S.kernels t.g in
    { t; buffers; flag; image; spin = kernel "spin" }

  let release s =
    C.free s.t.g s.flag;
    C.unload s.t.g s.image

  (* Submits [subs] back to back, then reads the word until it holds the last
     value: each read is at least the one before. *)
  let run_sys s subs =
    let part p =
      let queue = if p.copy then "COPY:0" else "COMPUTE:0" in
      let after = Array.of_list p.after in
      let dst = B.view s.buffers.(p.dst) ~first:p.do_ ~length:p.n
      and src = B.view s.buffers.(p.src) ~first:p.so ~length:p.n in
      if p.delay = 0 then S.copy ~queue ~after ~dst src
      else
        S.part ~queue ~after
          (S.delayed ~spin:s.spin ~flag:(address s.flag) ~ns:p.delay
             ~dst:(B.address dst) ~src:(B.address src) p.n)
    in
    let first = Rig.submitted s.t.d + 1 in
    let hand ps = S.submit s.t (Array.of_list (List.map part ps)) in
    let last = List.fold_left (fun _ ps -> hand ps) (first - 1) subs in
    let rec watch seen =
      let w = Rig.signaled s.t.d in
      at_least int ~msg:"the word" ~than:seen w;
      at_most int ~msg:"the word" ~than:last w;
      if w < last then watch w
    in
    watch (first - 1);
    S.wait s.t last;
    still ~msg:"the word" int last (fun () -> C.signaled s.t.g) ~ms:1

  let invariant m s =
    Array.iteri
      (fun b r ->
        equal string ~msg:(strf "buffer %d" b)
          (Bytes.to_string m.bufs.(b))
          (S.read_gpu (Nativeint.of_int (B.address r)) size))
      s.buffers

  let parts =
    let open Gen in
    let part =
      let+ delay = frequency [ (2, constant 0); (1, constant 50_000) ]
      and+ copy = bool
      and+ src = int_range 0 (buffers - 1)
      and+ dst = int_range 0 (buffers - 1)
      and+ n = one_of [ int_range 1 64; int_range 1 16384 ]
      and+ so = int_range 0 (size - 16384)
      and+ do_ = int_range 0 (size - 16384)
      and+ after = list ~size:(int_range 0 2) (int_range 0 2) in
      { delay; copy; src; so; dst; do_; n; after }
    in
    let submission =
      let+ ps = list ~size:(int_range 0 3) part in
      List.mapi
        (fun i p ->
          let after = List.filter (fun k -> k < i) p.after in
          { p with after = List.sort_uniq compare after })
        ps
    in
    list ~size:(int_range 1 4) submission
end

let order = abstract "o" ~invariant:Order.invariant ~release:Order.release

let order_commands =
  [
    command "start" (Gen.unit @-> makes order) Order.make Order.start;
    command "submit" ~pre:Order.determined
      (order ^-> Gen.with_pp Order.pp Order.parts @-> returns unit)
      Order.run Order.run_sys;
  ]

(* Ending from two domains: whatever the order, an allocation's or a mapping's
   first [free] and an image's first [unload] return, and every later one
   raises. *)

type ended = { mutable live : bool }

let end_model m =
  if not m.live then invalid_arg "ended";
  m.live <- false

let ends_once ~make ~finish ~release name =
  let v =
    abstract name ~release:(fun x ->
        try release x with Invalid_argument _ -> ())
  in
  [
    command "make" (Gen.unit @-> makes v) (fun () -> { live = true }) make;
    command "end" (v ^-> returns unit) end_model finish;
  ]

(* Each value carries the shared device: the fixture is read on the test's
   domain only. *)
let allocation_commands =
  ends_once "a"
    ~make:(fun () ->
      let g = (shared ()).g in
      (g, Option.get (C.alloc g `Device 64)))
    ~finish:(fun (g, r) -> C.free g r)
    ~release:(fun (g, r) -> C.free g r)

(* A mapping of its own page, freed once the run ends. *)
let mapping_commands =
  ends_once "m"
    ~make:(fun () ->
      let g = (shared ()).g and p = H.pages H.page in
      (g, p, Option.get (C.map_host g p 64)))
    ~finish:(fun (g, _, r) -> C.free g r)
    ~release:(fun (g, p, r) ->
      Fun.protect
        ~finally:(fun () -> H.free_pages p H.page)
        (fun () -> C.free g r))

let image_commands =
  ends_once "i"
    ~make:(fun () ->
      let g = (shared ()).g in
      (g, fst (S.kernels g)))
    ~finish:(fun (g, m) -> C.unload g m)
    ~release:(fun (g, m) -> C.unload g m)

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        (registry_commands ~cover);
      stateful ~count:20 ~domains:2
        "map_host from two domains shares a host range both ways"
        (registry_commands ~cover:(fun _ _ -> ()));
      stateful ~count:100 ~steps:20
        "values complete in order and the word never moves backwards (sampled)"
        order_commands;
      stateful ~count:30 ~domains:2
        "an allocation freed from two domains is freed once" allocation_commands;
      stateful ~count:30 ~domains:2
        "a mapping freed from two domains is freed once" mapping_commands;
      stateful ~count:30 ~domains:2
        "an image unloaded from two domains is unloaded once" image_commands;
    ]

let () =
  S.hold ();
  exit
    (run "rig_cuda"
       [
         opening;
         facts;
         memory;
         work;
         images;
         graphs;
         commits;
         timeline;
         two;
         stateful;
       ])
