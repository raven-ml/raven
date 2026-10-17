(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let strf = Printf.sprintf

(* The function an entry answered, which the test needs. *)
let found : Rig_edge.entry_answer -> Rig_edge.entry = function
  | Entry e -> e
  | Absent -> fail "no such function"
  | No_memory n -> failf "no memory for %d bytes" n

(* The code of the function an entry answered, if it answered one. *)
let code_of : Rig_edge.entry_answer -> int option = function
  | Entry e -> Some e.code
  | Absent | No_memory _ -> None

let watchdog = 17 (* CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT *)
let second = 1_000_000_000
let address r = Option.get (C.locate r).address
let host r = Option.get (C.locate r).host
let still = Rig_gpu_support.still
let word g = host (C.facts g).word

module B = Rig.Buffer

(* The reason of the loss [f ()] raises. *)
let lost f =
  match f () with _ -> failf "no loss" | exception Rig.Lost (_, why) -> why

(* Opening *)

let gpu_once () =
  let t = S.open_ () in
  let e = require_error (Rig_gpu_support.opened (C.gpu 0)) in
  contains ~sub:"open" e;
  S.close t;
  C.stop (require_ok (Rig_gpu_support.opened (C.gpu 0))) ~fault:None

let opening =
  group ~timeout:60. "opening"
    [
      test "count never raises" (fun () -> at_least int ~than:0 (C.count ()));
      test "names GPU 0 CUDA and GPU i CUDA:i" (fun () ->
          equal (list string)
            [ "CUDA"; "CUDA:1"; "CUDA:7" ]
            (List.map (fun i -> (C.gpu i).name) [ 0; 1; 7 ]));
      test "a negative GPU raises" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Rig_gpu_support.opened (C.gpu (-1)));
          raises_match Exn.invalid_arg (fun () -> (C.gpu (-1)).name));
      test "a GPU past the count is an error naming the count" (fun () ->
          let n = C.count () in
          let e = require_error (Rig_gpu_support.opened (C.gpu n)) in
          contains ~sub:(if n = 0 then "CUDA" else strf "CUDA sees %d" n) e);
      test "a GPU has one device until it is closed" gpu_once;
      cases ~name:fst
        "a call inside a lock of its thread's raises naming it, changing \
         nothing"
        [
          ("count", fun () () -> ignore (C.count () : int));
          ( "gpu",
            fun () ->
              let o = C.gpu (C.count ()) in
              fun () -> ignore (Rig_gpu_support.opened o : _ result) );
        ]
        (fun (fn, prepare) ->
          let call = prepare () in
          let prefix = strf "Rig_cuda.%s: called from" fn in
          let inside =
            Rig_support.inside_lock @@ fun () ->
            match call () with
            | () -> "ran"
            | exception Invalid_argument why when String.starts_with ~prefix why
              ->
                "refused"
          in
          equal ~msg:"inside the lock" string "refused" inside;
          call ());
    ]

(* Facts *)

let facts () =
  S.with_ @@ fun { g; _ } ->
  let f = C.facts g in
  equal int ~msg:"length of the arch" 5 (String.length f.arch);
  starts_with ~affix:"sm_" f.arch;
  greater int ~than:0 f.budget;
  let runs (q : Rig_edge.queue) =
    let kind : Rig_edge.kind -> string = function
      | Words -> "Words"
      | Fill -> "Fill"
      | Copy -> "Copy"
      | Launch -> "Launch"
    in
    (q.name, List.map kind q.runs)
  in
  equal
    (list (pair string (list string)))
    ~msg:"queues"
    [
      ("COMPUTE:0", [ "Fill"; "Copy"; "Launch" ]); ("COPY:0", [ "Fill"; "Copy" ]);
    ]
    (List.map runs f.queues);
  equal (option int) ~msg:"the copy engine" (Some 1) f.copy_engine;
  equal int ~msg:"alignment" 256 f.alignment;
  equal bool ~msg:"completion is the store" true (f.completion = Store);
  equal (list bool) ~msg:"waits on stores, hosts" [ true; true ]
    [ f.waits.stores; f.waits.hosts ];
  equal int ~msg:"most waits" max_int f.waits.most;
  equal bool ~msg:"submit may block" true f.may_block;
  equal (option int) ~msg:"the hang bound, where no display's watchdog bounds"
    (if S.attribute watchdog = 0 then Some 30_000 else None)
    f.hang_ms;
  equal nativeint ~msg:"the word's handle is its host address"
    (Nativeint.of_int (word g))
    (C.locate f.word).handle;
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
       ])

(* rig finds the record the device's facts declare, and none on the host. *)
let declared () =
  S.with_ @@ fun t ->
  (match Rig.capability t.d Rig_cuda_abi.key with
  | Some c -> equal bool ~msg:"the record" true (c == C.capability t.g)
  | None -> fail "no record under the key");
  is_none ~msg:"the host" (Rig.capability Rig.host Rig_cuda_abi.key)

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "finds the functions a fill calls" symbols;
      test "rig finds its record" declared;
    ]

(* Memory *)

let pattern n seed =
  String.init n (fun i -> Char.chr (((i * 7) + seed) land 255))

let past_memory () =
  S.with_ @@ fun { g; _ } -> is_none (C.alloc g Device (2 * (C.facts g).budget))

(* [write_gpu]'s bytes are in the GPU's memory once it returns, whatever runs on
   CUDA's default stream: a copy on one of rig's streams, which wait for no
   other stream, reads them right after, behind a 50 ms kernel there. *)
let written_on_return () =
  S.with_ @@ fun ({ d; g } as t) ->
  let _, kernel = S.kernels g in
  let flag = require_some (C.alloc g Pinned 8) in
  H.set64 (host flag) 0;
  let n = 64 in
  let src = B.create d n and dst = B.create ~memory:Pinned d n in
  let data = pattern n 11 in
  S.stall (kernel "spin") ~flag:(address flag) ~ns:(second / 20);
  let addresses = Rig.Submission.make ~fixed:[ (src, Read_write) ] d [||] in
  S.write_gpu (Nativeint.of_int (Rig.Submission.address addresses src)) data;
  Rig.Point.wait
    (S.submit t Rig.Submission.[| copy ~src:(Buffer src) ~dst:(Buffer dst) |]);
  equal string ~msg:"the bytes copied" data
    (H.read (H.at dst) n);
  C.free g flag

let memory =
  group ~timeout:120. "memory"
    [
      test "an allocation past the GPU's memory is None" past_memory;
      test "write_gpu's bytes are on the GPU when it returns" written_on_return;
    ]

(* Work *)

let fills_in_a_fresh_domain () =
  S.with_ @@ fun t ->
  let f = S.context () in
  let r, current =
    Domain.join
      (Domain.spawn (fun () ->
           let v = S.submit_work t [ S.part t.d f ] in
           (v, S.current ())))
  in
  equal int ~msg:"the value" 1 (Rig.Point.value r);
  equal nativeint ~msg:"the domain's thread has no context after" 0n current;
  Rig.Point.wait r;
  not_equal nativeint ~msg:"the fill saw a context" 0n (S.seen f)

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
  let addresses = Rig.Submission.make ~fixed:[ (src, Read_write) ] d [||] in
  S.write_gpu (Nativeint.of_int (Rig.Submission.address addresses src)) data;
  let parts =
    [|
      Rig_gpu_support.work
        Rig.Submission.(copy ~src:(Buffer src) ~dst:(Buffer dst));
      S.part t.d (S.failing 1);
    |]
  in
  let why = lost (fun () -> S.submit_work t (Array.to_list parts)) in
  starts_with ~affix:"running a fill: CUDA_ERROR_INVALID_VALUE: " why;
  S.close t;
  drained t 1;
  equal string ~msg:"copied before the word" data
    (H.read (H.at dst) 64)

let no_bytes () =
  S.with_ @@ fun { g; _ } ->
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  raises "alloc of 0 bytes" (fun () -> C.alloc g Device 0);
  raises "map_host of 0 bytes" (fun () -> C.map_host g (word g) 0)

let work =
  group ~timeout:60. "work"
    [
      test "a fill runs with the context current from a fresh domain"
        fills_in_a_fresh_domain;
      test "a failed fill loses the device, and its value still drains"
        failed_fill;
      test "an allocation or a mapping of no bytes raises" no_bytes;
    ]

(* Images *)

let images () =
  S.with_ @@ fun { g; _ } ->
  let m = S.loaded (require_ok (C.image g (S.fixture "kernels.ptx"))) in
  is_some ~msg:"double_index" (code_of (C.entry m "double_index"));
  is_none ~msg:"a missing kernel" (code_of (C.entry m "missing"));
  let entry () =
    let e = found (C.entry m "empty") in
    (e.code, e.launch)
  in
  let first = entry () in
  equal (pair int nativeint) ~msg:"a second entry" first (entry ());
  let e = require_error (C.image g "not a module") in
  starts_with ~affix:"loading the image: CUDA_ERROR_" e;
  let arch = (C.facts g).arch in
  (match C.image g (S.fixture "kernels.cubin") with
  | Ok i ->
      equal string ~msg:"the cubin's GPU" "sm_89" arch;
      C.unload g (S.loaded i)
  | Error e ->
      not_equal string ~msg:"the cubin's GPU" "sm_89" arch;
      starts_with ~affix:"loading the image: CUDA_ERROR_" e);
  C.unload g m

(* Loading places every function's code, whatever CUDA_MODULE_LOADING says: a
   function CUDA loads lazily, at its [entry], could fail there for lack of
   memory. One entry, which loads its own function, shows the others. *)
let loads_every_function () =
  S.with_ @@ fun { g; _ } ->
  let m = S.loaded (require_ok (C.image g (S.fixture "kernels.ptx"))) in
  equal (pair int int) (5, 5)
    (S.functions_loaded (found (C.entry m "empty")).code);
  C.unload g m

let images =
  group ~timeout:60. "images"
    [
      test "load, find their kernels and unload" images;
      test "loading places every function's code" loads_every_function;
    ]

(* Launches, through rig. The laws every driver's launches keep are the
   conformance suite's; these are CUDA's limits. *)

module Sub = Rig.Submission
module Run = Rig.Submission.Run

let launch_image = Rig_gpu_support.loader (fun () -> S.fixture "launch.ptx")

(* A launch of [kernel] of fixtures/launch.ptx on [d], of [params] bytes of
   parameters whose first 8 point into the run's one buffer, a write, over
   [groups] of [threads] with [shared] bytes of dynamic shared memory: a run
   of its submission holding its block. *)
let launching ?(kernel = "ids") ?(params = 24) d ~groups:(gx, gy, gz)
    ~threads:(tx, ty, tz) ~shared =
  let refs = [| Sub.write 0 ~at:0 |] in
  let fn = require_some (Rig.Image.resolve (launch_image d) kernel) in
  let s = Sub.make d [| Sub.launch ~refs fn ~params |] in
  let run = Run.make s in
  let b = Sub.block s 0 in
  Run.groups run b gx gy gz;
  Run.threads run b tx ty tz;
  Run.shared run b shared;
  run

let submitted run buffers = Rig.submit run ~buffers

(* The 32-bit words [b] holds. *)
let words b n =
  Array.init n (fun i ->
      H.get32 (H.at b + (4 * i)))

(* cuda.h's CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN. *)
let largest_shared = 97

(* Launches the GPU or the function cannot run are refused before any value, and
   the device stays live: an sm_89 has at most 65535 groups along z, 1024
   threads along x and 1024 threads in a group. *)
let refused_launches () =
  S.with_ @@ fun { d; _ } ->
  let out = B.create ~memory:Pinned d 4096 in
  let refuses msg ?kernel ?params ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1))
      ?(shared = 0) () =
    let run = launching ?kernel ?params d ~groups ~threads ~shared in
    raises_match ~msg (Exn.invalid_arg ~substring:"never fit") (fun () ->
        submitted run [| out |])
  in
  (* More threads than the function has, which the setter refuses. *)
  List.iter
    (fun threads ->
      raises_match
        (Exn.invalid_arg ~substring:"threads per group")
        (fun () -> launching d ~groups:(1, 1, 1) ~threads ~shared:0))
    [ (1025, 1, 1); (1024, 2, 1) ];
  refuses "no groups along x" ~groups:(0, 1, 1) ();
  refuses "no threads along z" ~threads:(1, 1, 0) ();
  refuses "65536 groups along z" ~groups:(1, 1, 65536) ();
  refuses "a byte of shared memory past the largest" ~kernel:"rotate" ~params:12
    ~shared:(S.attribute largest_shared + 1)
    ();
  equal int ~msg:"values assigned" 0 (Rig.submitted d);
  equal (option string) ~msg:"the device's loss" None (Rig.lost d);
  let n = 65535 * 1024 in
  let run =
    launching d ~groups:(1, 1, 65535) ~threads:(1024, 1, 1) ~shared:0
  in
  let big = B.create d (4 * n) in
  Rig.Point.wait (submitted run [| big |])

(* A launch takes as much dynamic shared memory as the GPU gives a group, beyond
   the 48 KiB CUDA allows by default. [rotate] stores into each word the value
   of the next thread of its group, through shared memory. *)
let largest_shared_memory () =
  S.with_ @@ fun { d; _ } ->
  let shared = S.attribute largest_shared in
  let run =
    launching ~kernel:"rotate" ~params:12 d ~groups:(1, 1, 1)
      ~threads:(256, 1, 1) ~shared
  in
  Run.int32 run (Sub.block (Run.submission run) 0) 8 5;
  let out = B.create ~memory:Pinned d (4 * 256) in
  Rig.Point.wait (submitted run [| out |]);
  equal (array int) ~msg:"the words"
    (Array.init 256 (fun t -> 5 + (3 * ((t + 1) mod 256))))
    (words out 256)

(* A cached launch submitted with a warm run, its block stored anew each time
   through every setter, allocates nothing. *)
let no_allocation () =
  S.with_ @@ fun { d; _ } ->
  let run = launching d ~groups:(1, 1, 1) ~threads:(32, 1, 1) ~shared:0 in
  Run.release run;
  let buffers = [| B.create d 4096 |] in
  let b = Sub.block (Run.submission run) 0 in
  (* [float64] stores first, where [int64] then stores [a]. *)
  let once i =
    if not (Run.take run) then fail "the run is taken";
    Run.groups run b 1 1 1;
    Run.threads run b 32 1 1;
    Run.shared run b 0;
    Run.int64 run b 0 0;
    Run.float64 run b 8 2.5;
    Run.int64 run b 8 i;
    Run.int32 run b 16 3;
    Run.float32 run b 20 1.5;
    Sys.opaque_identity (submitted run buffers)
  in
  Rig.Point.wait (once 0);
  let before = Gc.minor_words () in
  let last = ref (once 1) in
  for i = 2 to 100 do
    last := once i
  done;
  let words = int_of_float (Gc.minor_words () -. before) / 100 in
  Rig.Point.wait !last;
  equal int ~msg:"words per launch" 0 words

let launches =
  group ~timeout:120. "launches"
    [
      test "launches the GPU cannot run are refused" refused_launches;
      test "a launch takes the largest shared memory" largest_shared_memory;
      test "a cached launch allocates nothing" no_allocation;
    ]

(* Timeline and loss *)

(* A launch that runs a kernel until [flag]'s first word is not 0, for at most
   10 seconds. *)
let spinning (t : S.t) flag =
  H.set64 (host flag) 0;
  S.launch t "spin" (address flag) (10 * second)

(* [spin t flag] submits {!spinning} as [t]'s first value, and is its
   point. *)
let spin (t : S.t) flag =
  let base = Rig.submitted t.d in
  let p = S.submit_work t [ spinning t flag ] in
  equal int ~msg:"the value" (base + 1) (Rig.Point.value p);
  p

(* [lose t flag] submits {!spinning}, then a fill that fails: the loss stops [t]
   while the kernel runs. *)
let lose (t : S.t) flag =
  let fails = S.part t.d (S.failing 1) in
  ignore (lost (fun () -> S.submit_work t [ spinning t flag; fails ]))

(* GPU 0, opened once the work a stop left running ended, within 10 s of the
   host's monotonic clock: the bound of that work's kernel. *)
let reopened () =
  let t0 = Rig.Profile.now () in
  let rec go () =
    match Rig_gpu_support.opened (C.gpu 0) with
    | Ok g -> g
    | Error e when Rig.Profile.now () - t0 > 10 * second ->
        failf "GPU 0 does not open again within 10 s: %s" e
    | Error _ ->
        Domain.cpu_relax ();
        go ()
  in
  go ()

(* A collection needs every domain at a safe point, so it waits for a domain
   whose C call holds the runtime: here, an unload that CUDA holds until the
   GPU's running work ends. *)
let unload_aside () =
  S.with_ @@ fun ({ g; _ } as t) ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g Pinned 8) in
  Fun.protect ~finally:(fun () -> H.set64 (host flag) 1) @@ fun () ->
  let other, _ = S.kernels g in
  let p = spin t flag in
  let before = Rig.Point.value p - 1 in
  equal int ~msg:"the word" before (Rig.signaled t.d);
  let unloading = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        Atomic.set unloading true;
        C.unload g other)
  in
  while not (Atomic.get unloading) do
    Domain.cpu_relax ()
  done;
  still ~msg:"the work while unload waits" int before
    (fun () -> C.signaled g)
    ~ms:50;
  Gc.full_major ();
  equal int ~msg:"the work after a collection" before (C.signaled g);
  H.set64 (host flag) 1;
  Domain.join d;
  Rig.Point.wait p

(* The 256 MiB global of fixtures/global.ptx, loaded on [g]: the load places it
   with every function's code. *)
let global = 256 * 1024 * 1024

let load_global t =
  S.loaded (require_ok (C.image t.S.g (S.fixture "global.ptx")))

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
  let flag = require_some (C.alloc g Pinned 8) in
  Fun.protect ~finally:(fun () -> H.set64 (host flag) 1) @@ fun () ->
  let before = S.free_memory () in
  let m = load_global t in
  lose t flag;
  S.close t;
  H.set64 (host flag) 1;
  C.stop (reopened ()) ~fault:None;
  S.with_ @@ fun t' ->
  at_most int ~msg:"reopened" ~than:(before - (global / 2)) (S.free_memory ());
  C.unload g m;
  at_least int ~msg:"unloaded" ~than:(before - (global / 2)) (S.free_memory ());
  Rig.Point.wait (S.submit t' [||])

let stop_idle () =
  S.with_ @@ fun ({ d; g } as t) ->
  let p = H.pages H.page in
  let r = require_some (C.map_host g p 64) in
  equal bool ~msg:"locked" true (S.locked p);
  Rig.Point.wait (S.submit t [||]);
  S.close t;
  equal int ~msg:"the word" 1 (Rig.signaled d);
  C.free g r;
  equal bool ~msg:"locked after free" false (S.locked p);
  H.free_pages p H.page

(* A stop after the work ended, its last value not yet committed: value 1, of
   no parts, is short of the 64 values that commit on their own. *)
let stop_ended () =
  if not (S.present ()) then skip ~reason:"the machine has no CUDA GPU" ();
  S.release ();
  let g = require_ok (Rig_gpu_support.opened (C.gpu 0)) in
  is_none ~msg:"value 1" (S.submit_edge g ~v:1 []);
  equal int ~msg:"the word before the stop" 0 (C.signaled g);
  C.stop g ~fault:None;
  equal int ~msg:"the word as the stop returns" 1 (C.signaled g);
  let g =
    require_ok ~msg:"the GPU opened again" (Rig_gpu_support.opened (C.gpu 0))
  in
  C.stop g ~fault:None

let stop_running () =
  if not (S.present ()) then skip ~reason:"the machine has no CUDA GPU" ();
  S.release ();
  let opener = C.gpu 0 in
  let close g = C.stop g ~fault:None in
  let checked = ref false and flag_at = ref 0 and base = ref 0 in
  (* Checks while the kernel's gate is still held. *)
  let module Driver = struct
    include C

    let stop g ~fault =
      C.stop g ~fault;
      Fun.protect ~finally:(fun () -> H.set64 !flag_at 1) @@ fun () ->
      equal int ~msg:"the work still runs after stop" !base (C.signaled g);
      let opened = Rig_gpu_support.opened opener in
      Result.iter close opened;
      let e = require_error opened in
      contains ~sub:"still runs" e;
      checked := true
  end in
  let made = ref None in
  let make ~word =
    let r = opener.make ~word in
    Result.iter (fun g -> made := Some g) r;
    r
  in
  let d = require_ok (Rig.open_ { opener with driver = (module Driver); make }) in
  let g = Option.get !made in
  base := Rig.submitted d;
  S.bind g;
  if S.attribute watchdog <> 0 then begin
    Rig.close d;
    skip ~reason:"a display watchdog ends long kernels" ()
  end;
  let flag = require_some (C.alloc g Pinned 8) in
  flag_at := host flag;
  let t = { S.d; g } in
  Fun.protect
    ~finally:(fun () ->
      H.set64 (host flag) 1;
      Rig.close d;
      close (reopened ());
      C.free g flag)
  @@ fun () ->
  lose t flag;
  equal bool ~msg:"the stop checked reopening while work ran" true !checked;
  H.set64 (host flag) 1;
  close (reopened ());
  equal int ~msg:"the word once the work ended" (Rig.submitted d)
    (C.signaled g)

(* After a failure, a submit's write of its value still follows every earlier
   value: here value 1, a kernel that runs on COPY:0 until [flag] is set. Value
   2 fails on COPY:0, and value 3 comes after the failure. *)
let write_after_failure () =
  if not (S.present ()) then skip ~reason:"the machine has no CUDA GPU" ();
  S.release ();
  let g = require_ok (Rig_gpu_support.opened (C.gpu 0)) in
  S.bind g;
  let flag = require_some (C.alloc g Pinned 8) in
  let _, kernel = S.kernels g in
  H.set64 (host flag) 0;
  let spinning = S.spinning (kernel "spin") ~flag:(address flag) ~ns:second in
  (* Value 2's write follows value 1's kernel, whatever value 3's does. *)
  let ended () =
    H.set64 (host flag) 1;
    while C.signaled g < 2 do
      Domain.cpu_relax ()
    done
  in
  Fun.protect ~finally:(fun () ->
      ended ();
      C.stop g ~fault:None;
      C.free g flag)
  @@ fun () ->
  is_none ~msg:"value 1" (S.submit_edge g ~v:1 [ (1, spinning) ]);
  is_some ~msg:"value 2" (S.submit_edge g ~v:2 [ (1, S.failing 1) ]);
  is_some ~msg:"value 3" (S.submit_edge g ~v:3 []);
  still ~msg:"the word while value 1 runs" int 0 (fun () -> C.signaled g) ~ms:50;
  ended ();
  while C.signaled g < 3 do
    Domain.cpu_relax ()
  done

(* Two fills on one queue in one C submission, the first with 64 bytes of
   argument: each runs on its own argument, wherever the first placed its
   own, and the second's failure answers the submit. *)
let fills_own_arguments () =
  if not (S.present ()) then skip ~reason:"the machine has no CUDA GPU" ();
  S.release ();
  let g = require_ok (Rig_gpu_support.opened (C.gpu 0)) in
  Fun.protect ~finally:(fun () -> C.stop g ~fault:None) @@ fun () ->
  is_some ~msg:"the second fill's failure"
    (S.submit_edge g ~v:1 [ (1, S.failing ~bytes:64 0); (1, S.failing 3) ])

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

(* A device idle past the age of its clock's measure, 8 s, measures again at
   its next timed submit. That submission's span, and one before the idle,
   lie within the host's times around their submit and their wait, up to the
   error rig_cuda.mli states (S.clock_error). *)
let spans_after_idle () =
  let opened = Rig.Profile.now () in
  S.with_ @@ fun t ->
  let within what =
    let t0 = Rig.Profile.now () in
    let (), events =
      Rig.Profile.take (fun () ->
          Rig.Point.wait (S.submit_work t [ S.spin t ~ns:100_000 ]))
    in
    let t1 = Rig.Profile.now () in
    let error = S.clock_error ~since:(t1 - opened) in
    match
      List.filter_map
        (function
          | Rig.Profile.Work { device; start; stop; _ } when device == t.d ->
              Some (start, stop)
          | _ -> None)
        events
    with
    | [ (start, stop) ] ->
        at_least int ~msg:(what ^ ": the span's start") ~than:(t0 - error)
          start;
        at_most int ~msg:(what ^ ": the span's stop") ~than:(t1 + error) stop
    | spans ->
        equal int ~msg:(what ^ ": the device's spans") 1 (List.length spans)
  in
  within "before the idle";
  Unix.sleepf 8.5;
  within "after the idle"

(* The hang bound of the devices of {!own_word_after_running}. *)
let own_hang_ms = 300

(* A device on a word of its own, lost past its hang bound while a kernel
   runs, gives its word back once that work ended, which writes the last
   value, and a drain found it stopped. The GPU's next open asks whether the
   stopped device's work ended, and reads the value its edge kept, never the
   freed word. *)
let own_word_after_running () =
  let module W = Rig_gpu_support.Watch (C) in
  let module Capped = struct
    include W

    let facts h = { (W.facts h) with hang_ms = Some own_hang_ms }
  end in
  (* The process's first failure keeps its slot, and so its word: a Polled
     device fails first. *)
  if Rig.failure () = None then begin
    let d, p = Rig_support.Polled.open_ "CUDA:first-failure" in
    Rig_support.Polled.fault p "the first failure";
    ignore (lost (fun () -> Rig.Buffer.create d 8));
    Rig.close d
  end;
  let rounds = 3 in
  for i = 1 to rounds do
    let t = S.open_own (module Capped) in
    if S.attribute watchdog <> 0 then begin
      S.close t;
      skip ~reason:"a display watchdog ends long kernels" ()
    end;
    let flag = require_some (C.alloc t.g Pinned 8) in
    H.set64 (host flag) 0;
    (* The kernel runs from a module the driver loaded, which rig does not
       hold: nothing of the device stays reachable once it closed. *)
    let m, kernel = S.kernels t.g in
    let spinning =
      S.spinning (kernel "spin") ~flag:(address flag) ~ns:(10 * second)
    in
    let p = S.submit_work t [ S.part t.d spinning ] in
    raises_match ~msg:"the wait past the hang bound"
      (function Rig.Lost _ -> true | _ -> false)
      (fun () -> Rig.Point.wait p);
    S.close t;
    H.set64 (host flag) 1;
    (* The submission's record holds the slot until it is collected. *)
    let t0 = Rig.Profile.now () in
    while
      Gc.full_major ();
      Rig.free_cache t.d;
      W.word_frees t.g = 0
    do
      if Rig.Profile.now () - t0 > 10 * second then
        failf "round %d: the word did not go back within 10 s" i;
      Unix.sleepf 0.01
    done;
    equal int ~msg:(strf "round %d: the word's frees" i) 1 (W.word_frees t.g);
    equal int ~msg:"the reads of the word after its free" 0 (W.late_reads t.g);
    C.unload t.g m;
    C.free t.g flag
  done;
  S.with_ @@ fun t -> Rig.Point.wait (S.submit t [||])

let timeline =
  group ~timeout:60. "timeline"
    [
      test
        "unload lets other domains run while CUDA waits for the GPU (sampled)"
        unload_aside;
      test "a close of an idle device leaves the word at the last value"
        stop_idle;
      test "a span after an idle past the clock's age lies within its window"
        spans_after_idle;
      test "a stop after the work ended leaves the word at the last value"
        stop_ended;
      test "a loss that stops running work leaves the GPU closed until it ends"
        stop_running;
      test "a write after a failure follows the work before it"
        write_after_failure;
      test "each fill of a submission runs on its own argument"
        fills_own_arguments;
      test "page-locking is the process's" registry_is_the_process;
      test "an image a close left loaded is unloaded after it" unload_after_stop;
      test "an unload after the GPU opened again leaves the new device working"
        unload_after_reopen;
      test "a word of its own goes back once the running work ended"
        own_word_after_running;
    ]

(* Graphs *)

let graph g ks = (C.capability g).graph ks
let stopped_graph = "making the graph: the device was stopped"

(* A 64-bit word of pinned memory, zeroed. *)
let counter g =
  let r = require_some (C.alloc g Pinned 8) in
  H.set64 (host r) 0;
  r

(* [launch t gr us] submits a fill that updates [gr]'s nodes [us] and launches
   it, and is its value. *)
let launch (t : S.t) gr us =
  S.submit_work t [ S.part t.d (S.graph_launch gr us) ]

(* The steps [first], [first + 1], ... of [n] nodes on the word [out]. *)
let steps step out ~first n =
  Array.init n (fun j -> S.kernel step (address out) (first + j))

let chain n =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let out = counter g in
  let gr = require_ok (graph g (steps (kernel "step") out ~first:0 n)) in
  equal int ~msg:"its nodes" n (Array.length gr.nodes);
  Rig.Point.wait (launch t gr [||]);
  equal int ~msg:"the word" n (H.get64 (host out));
  C.unload g m;
  C.free g out

let prepared_graph () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let out = counter g in
  let weak = Weak.create 1 in
  let[@inline never] made () =
    let gr = require_ok (graph g (steps (kernel "step") out ~first:0 2)) in
    Weak.set weak 0 (Some gr);
    Rig_gpu_support.prepare t.d
      [ S.part t.d (S.graph_launch gr [||]) ]
  in
  let run, buffers = made () in
  (* Collection and draining also release an unreachable fill's hold. *)
  for round = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create t.d 0);
    equal bool
      ~msg:(strf "the submission keeps its graph after collection %d" round)
      true (Weak.check weak 0)
  done;
  Rig.Point.wait (Rig.submit run ~buffers);
  equal int ~msg:"both kernels ran" 2 (H.get64 (host out));
  C.unload g m;
  C.free g out

let collected_image () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let out = counter g in
  let weak = Weak.create 1 in
  let[@inline never] make () =
    let image = require_ok (Rig.Image.load t.d (S.fixture "kernels.ptx")) in
    Weak.set weak 0 (Some image);
    let fn = require_some (Rig.Image.resolve image "step") in
    let step = Rig.Image.entry fn in
    require_ok (graph g [| S.kernel step (address out) 0 |])
  in
  let gr = make () in
  for round = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create t.d 0);
    equal bool
      ~msg:(strf "image collected after round %d" round)
      false (Weak.check weak 0)
  done;
  Rig.Point.wait (launch t gr [||]);
  equal int ~msg:"the graph kept its kernel loaded" 1 (H.get64 (host out));
  C.free g out

let collected_together () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let images = Weak.create 1 and graphs = Weak.create 1 in
  let[@inline never] make () =
    let image = require_ok (Rig.Image.load t.d (S.fixture "kernels.ptx")) in
    let fn = require_some (Rig.Image.resolve image "empty") in
    let empty = Rig.Image.entry fn in
    let gr = require_ok (graph g [| S.kernel empty 0 0 |]) in
    Weak.set images 0 (Some image);
    Weak.set graphs 0 (Some gr)
  in
  make ();
  for round = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create t.d 0);
    ignore (require_ok (graph g [||]));
    equal bool
      ~msg:(strf "image collected after round %d" round)
      false (Weak.check images 0);
    equal bool
      ~msg:(strf "graph collected after round %d" round)
      false (Weak.check graphs 0)
  done

let collection_does_not_wait () =
  S.with_ @@ fun ({ g; _ } as t) ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g Pinned 8) in
  let[@inline never] make () =
    let m, kernel = S.kernels g in
    let gr = require_ok (graph g [| S.kernel (kernel "empty") 0 0 |]) in
    C.unload g m;
    ref (Some gr)
  in
  let gr = make () in
  Fun.protect
    ~finally:(fun () -> H.set64 (host flag) 1)
    (fun () ->
      let p = spin t flag in
      gr := None;
      Gc.full_major ();
      ignore (Sys.opaque_identity gr);
      equal int ~msg:"collection returned while the GPU waits"
        (Rig.Point.value p - 1)
        (C.signaled g);
      H.set64 (host flag) 1;
      Rig.Point.wait p;
      ignore (require_ok (graph g [||])));
  C.free g flag

let arguments () =
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let out = require_some (C.alloc g Pinned 256) in
  H.write (host out) (String.make 256 '\000');
  let k = S.kernel (kernel "double_index") ~block:64 (address out) 37 in
  let gr = require_ok (graph g [| k |]) in
  Rig.Point.wait (launch t gr [||]);
  equal (list int) ~msg:"words 0 to 39"
    (List.init 40 (fun i -> if i < 37 then 2 * i else 0))
    (List.init 40 (fun i -> H.get32 (host out + (4 * i))));
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
  let last = ref None in
  List.iteri
    (fun r b ->
      cover "a run on the other word than the run before"
        (r > 0 && b <> List.nth schedule (r - 1));
      cover "a run on the same word as the run before"
        (r > 0 && b = List.nth schedule (r - 1));
      let ks = steps step outs.(b) ~first:(runs.(b) * width) width in
      runs.(b) <- runs.(b) + 1;
      last := Some (launch t gr (Array.mapi (fun j k -> (j, k)) ks)))
    schedule;
  Option.iter Rig.Point.wait !last;
  equal (list int) ~msg:"each word's steps"
    (List.map (fun n -> n * width) (Array.to_list runs))
    (List.map (fun r -> H.get64 (host r)) (Array.to_list outs));
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
  C.unload g m;
  invalid [| k |]

(* Collection after close queues destruction for the next open of the GPU. *)
let collected_after_stop () =
  let weak = Weak.create 1 in
  let[@inline never] make () =
    S.with_ @@ fun ({ g; _ } as t) ->
    let m = load_global t in
    let touch = (found (C.entry m "touch")).code in
    let gr = require_ok (graph g [| S.kernel touch 0 0 |]) in
    Rig.Point.wait (launch t gr [||]);
    C.unload g m;
    S.close t;
    Weak.set weak 0 (Some gr)
  in
  let collect () =
    for round = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create Rig.host 0);
      if round = 4 then
        equal bool ~msg:"the stopped graph was collected" false
          (Weak.check weak 0)
    done
  in
  let reopen () =
    S.with_ (fun { g; _ } -> ignore (require_ok (graph g [||])))
  in
  make ();
  collect ();
  reopen ();
  let before = S.free_memory () in
  make ();
  collect ();
  at_most int ~msg:"collected module awaits reopening"
    ~than:(before - (global / 2))
    (S.free_memory ());
  reopen ();
  at_least int ~msg:"reopening reclaimed the module"
    ~than:(before - (global / 2))
    (S.free_memory ())

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
    | Ok _ ->
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

(* A graph call inside a lock of its thread's, as from a callback inside one,
   raises naming it, having made nothing; one after makes the graph. *)
let graph_inside_lock () =
  S.with_ @@ fun { g; _ } ->
  let made =
    Rig_support.inside_lock @@ fun () ->
    match graph g [||] with
    | _ -> "made"
    | exception Invalid_argument why
      when String.starts_with ~prefix:"Rig_cuda_abi.graph: called from" why ->
        "refused"
  in
  equal ~msg:"inside the lock" string "refused" made;
  ignore (require_ok (graph g [||]))

let graphs =
  group ~timeout:60. "graphs"
    [
      cases
        ~name:(strf "a chain of %d kernels runs each after the one before")
        "chain" [ 0; 1; 17; 64 ] chain;
      test "a prepared fill keeps its graph" prepared_graph;
      test "a graph keeps a collected image's code" collected_image;
      test "a graph and its image can be collected together" collected_together;
      test "graph collection never waits for unrelated work"
        collection_does_not_wait;
      test "a kernel reads its argument block" arguments;
      prop ~count:30
        "a launch runs the updates of its own fill, none of a later one"
        (Gen.list ~size:(Gen.int_range 1 40) (Gen.int_range 0 1))
        updates;
      test "refuses what it cannot record" refusals;
      test "a collected graph is reclaimed after its device stopped"
        collected_after_stop;
      test "a graph call after the stop answers the stop" after_stop;
      test "graph calls beside a stop answer as before or after it" beside_stop;
      test "a graph call inside a lock of its thread's raises naming it"
        graph_inside_lock;
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
        | None
          when List.exists
                 (fun e -> a < e.start + e.bytes && e.start < a + n)
                 m.entries ->
            cover "a range overlapping a registered one" true;
            false
        | None ->
            cover "a range sharing only a page with a registered one"
              (List.exists (shares (a, n)) m.entries);
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
    let foreign = Option.get (C.alloc g Pinned (2 * H.page)) in
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

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        (registry_commands ~cover);
      stateful ~count:20 ~domains:2
        "map_host from two domains shares a host range both ways"
        (registry_commands ~cover:(fun _ _ -> ()));
    ]

let () =
  exit
    (run "rig_cuda"
       [
         opening;
         facts;
         memory;
         work;
         images;
         launches;
         graphs;
         timeline;
         stateful;
       ])
