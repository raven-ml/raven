(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let maps = Rig_gpu_support.mapped

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
let l2_cache_size = 38 (* CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE *)
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

(* The scheduling of a context, the three low bits of its flags, and CUDA's
   values of it (cuDevicePrimaryCtxSetFlags). *)
let scheduling_bits = 0x07

let schedulings =
  [
    ("CU_CTX_SCHED_AUTO", 0x00);
    ("CU_CTX_SCHED_SPIN", 0x01);
    ("CU_CTX_SCHED_YIELD", 0x02);
    ("CU_CTX_SCHED_BLOCKING_SYNC", 0x04);
  ]

(* An open finds the primary context's flags with the scheduling [s], as
   another library set them, and leaves them. The flags are the process's,
   so the case sets back those it found, whatever happens. *)
let keeps_scheduling (_, s) =
  let found = S.with_driver (fun _ -> S.context_flags ()) in
  Fun.protect ~finally:(fun () -> S.set_context_flags found) @@ fun () ->
  S.set_context_flags ((found land lnot scheduling_bits) lor s);
  let before = S.context_flags () in
  equal ~msg:"the scheduling set" int s (before land scheduling_bits);
  S.with_driver ignore;
  equal ~msg:"the flags after the open" int before (S.context_flags ())

(* The test's executable, run with [unloadable], prints what count and GPU
   0's make answer. *)
let unloadable = "--unloadable"

let print_unloadable () =
  (match C.count () with
  | Ok n -> Printf.printf "%d\n%!" n
  | Error why -> Printf.printf "Error %s\n%!" why);
  match Rig_gpu_support.opened (C.gpu 0) with
  | Ok _ -> print_string "opened"
  | Error why -> print_string why

(* The platform's names of the CUDA library, and the variable that puts a
   directory first on the library search path: an empty file of each name
   there is found first and does not load. Linux's loader ends a search at a
   file it finds that is no ELF object; macOS's goes on, and finds no other
   CUDA library. *)
let search () =
  if Sys.win32 then None
  else
    let ic = Unix.open_process_args_in "uname" [| "uname"; "-s" |] in
    let os = input_line ic in
    ignore (Unix.close_process_in ic);
    match os with
    | "Linux" -> Some ([ "libcuda.so.1"; "libcuda.so" ], "LD_LIBRARY_PATH")
    | "Darwin" -> Some ([ "libcuda.dylib" ], "DYLD_LIBRARY_PATH")
    | _ -> None

(* The process's environment, [var] set to [v]. *)
let with_var var v =
  Unix.environment ()
  |> Array.to_list
  |> List.filter (fun e -> not (String.starts_with ~prefix:(var ^ "=") e))
  |> List.cons (var ^ "=" ^ v)
  |> Array.of_list

(* The test's executable run with the argument [arg] in the environment
   [env]: its exit code and what it printed. *)
let child arg env =
  let exe = Sys.executable_name in
  let r, w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process_env exe [| exe; arg |] env Unix.stdin w Unix.stderr
  in
  Unix.close w;
  let out = In_channel.input_all (Unix.in_channel_of_descr r) in
  Unix.close r;
  let _, status = Unix.waitpid [] pid in
  ((match status with WEXITED c -> c | _ -> -1), out)

(* count and make in a process whose search path finds no CUDA library that
   loads, whatever the machine installed: each is an error naming the
   library. *)
let test_unloadable () =
  let names, var =
    match search () with
    | Some s -> s
    | None ->
        skip
          ~reason:
            "Windows searches its system directory, which holds nvcuda.dll, \
             before PATH"
          ()
  in
  let dir = Filename.temp_dir "rig-cuda" "" in
  List.iter
    (fun n -> Out_channel.with_open_bin (Filename.concat dir n) ignore)
    names;
  let path =
    match Sys.getenv_opt var with Some p -> dir ^ ":" ^ p | None -> dir
  in
  let code, out = child unloadable (with_var var path) in
  List.iter (fun n -> Sys.remove (Filename.concat dir n)) names;
  Sys.rmdir dir;
  equal ~msg:"the child's exit code" int 0 code;
  match String.split_on_char '\n' out with
  | [ count; why ] ->
      starts_with ~msg:"count" ~prefix:"Error " count;
      contains ~msg:"count's error" ~sub:(List.hd names) count;
      contains ~msg:"make's error" ~sub:(List.hd names) why
  | _ -> failf "the child printed %S" out

(* Where CUDA lists its GPUs, buses is one address per GPU, in the form
   /sys/bus/pci names them, and on Linux each is there; where it cannot, both
   answer the same error. *)
let buses () =
  match (C.count (), C.buses ()) with
  | Error why, b ->
      equal (result (list string) string) (Error why) b
  | Ok n, Error why -> failf "count is %d, buses is Error %S" n why
  | Ok n, Ok buses ->
      equal ~msg:"one per GPU" int n (List.length buses);
      let hex c = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') in
      let address b =
        match String.split_on_char ':' b with
        | [ domain; bus; slot ] ->
            String.length domain >= 4
            && String.length bus = 2
            && String.length slot = 4
            && slot.[2] = '.'
            && String.for_all hex (domain ^ bus ^ String.sub slot 0 2)
            && slot.[3] >= '0' && slot.[3] <= '7'
        | _ -> false
      in
      let listed b = Sys.file_exists ("/sys/bus/pci/devices/" ^ b) in
      List.iter
        (fun b ->
          satisfies ~claim:"a bus address" string address b;
          if Sys.file_exists "/sys/bus/pci/devices" then
            satisfies ~claim:"a function of /sys/bus/pci" string listed b)
        buses

let opening =
  group ~timeout:60. "opening"
    [
      test "count never raises" (fun () ->
          ignore (C.count () : (int, string) result));
      test
        "with no CUDA library that loads, count and make are errors naming \
         the library"
        test_unloadable;
      test "buses has a sysfs address per GPU CUDA sees" buses;
      test "names GPU 0 CUDA and GPU i CUDA:i" (fun () ->
          equal (list string)
            [ "CUDA"; "CUDA:1"; "CUDA:7" ]
            (List.map (fun i -> (C.gpu i).name) [ 0; 1; 7 ]));
      test "a negative GPU raises" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Rig_gpu_support.opened (C.gpu (-1)));
          raises_match Exn.invalid_arg (fun () -> (C.gpu (-1)).name));
      test "a GPU past the count is an error naming the count" (fun () ->
          let n = Result.value (C.count ()) ~default:0 in
          let e = require_error (Rig_gpu_support.opened (C.gpu n)) in
          contains ~sub:(if n = 0 then "CUDA" else strf "CUDA sees %d" n) e);
      test "a GPU has one device until it is closed" gpu_once;
      cases "an open leaves the context's scheduling as it finds it" ~id:fst
        schedulings keeps_scheduling;
      cases
        "a call inside a lock of its thread's raises naming it, changing \
         nothing"
        ~id:fst
        [
          ("count", fun () () -> ignore (C.count () : (int, string) result));
          ( "buses",
            fun () () -> ignore (C.buses () : (string list, string) result) );
          ( "gpu",
            fun () ->
              let o = C.gpu (Result.value (C.count ()) ~default:0) in
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
  S.with_ @@ fun { g; d; _ } ->
  let f = C.facts g in
  equal int ~msg:"length of the arch" 5 (String.length f.arch);
  starts_with ~prefix:"sm_" f.arch;
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
  equal (option int) ~msg:"the L2's bytes, as CUDA states them"
    (Some (S.attribute l2_cache_size))
    (Rig.cache_bytes d);
  equal bool ~msg:"completion is the store" true (f.completion = Store);
  equal (list bool) ~msg:"waits on stores, hosts" [ true; true ]
    [ f.waits.stores; f.waits.hosts ];
  equal bool ~msg:"submit may block" true f.may_block;
  equal (option int) ~msg:"the hang bound" None f.hang_ms;
  equal nativeint ~msg:"the word's handle is its host address"
    (Nativeint.of_int (word g))
    (C.locate f.word).handle;
  equal int ~msg:"the word starts at 0" 0 (C.signaled g)

let symbols () =
  S.with_ @@ fun { g; _ } ->
  let symbol = (C.capability g).symbol in
  let found n = Option.is_some (symbol n) in
  equal (list bool)
    [ true; true; true; false; false ]
    (List.map found
       [
         "cuLaunchKernel";
         "cuMemcpyAsync";
         "cuLaunchHostFunc";
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

(* The capacity is cuMemGetInfo's pair, whose total starts the budget: its
   free lies between two reads of CUDA's around it. *)
let capacity () =
  S.with_ @@ fun t ->
  let before = S.free_memory () in
  let c = require_some (Rig.capacity t.d) in
  let after = S.free_memory () in
  equal int ~msg:"the budget" (Rig.budget t.d) c.total;
  at_least int ~msg:"free" ~than:(Int.min before after) c.free;
  at_most int ~msg:"free" ~than:(Int.max before after) c.free

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "finds the functions a fill calls" symbols;
      test "rig finds its record" declared;
      test "reports cuMemGetInfo's capacity" capacity;
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

(* Another function's DMA. The GPU's own function stands for the other one,
   as every machine has it. CUDA exports a dma-buf of an allocation on the
   host's page, where its kernel driver can: an export is [Ok] iff the
   buffer spans an allocation that starts on the page and is a whole number
   of pages, and then lies at offset 0. A buffer that spans no allocation,
   as one rig rounded up, says nothing of it and is skipped. *)
let dma_sizes = [ 1; 4095; 4096; 4097; 8192; 1 lsl 22; (1 lsl 22) + 1 ]

let gpu_fn d =
  let fn = require_some ~msg:"the device's bus" (Rig.bus d) in
  equal ~msg:"GPU 0's function" (option string)
    (List.nth_opt (require_ok (C.buses ())) 0)
    (Some fn);
  fn

let export_law d ~fn memory n =
  let module Dmabuf = Rig_gpu_support.Dmabuf in
  let b = B.create ~memory d n in
  let on_pages = B.aligned b ~at:0 H.page && n mod H.page = 0 in
  match B.dma b ~fn with
  | Error why when Rig_gpu_support.has ~sub:"_DMA_BUF_SUPPORTED is false" why ->
      skip ~reason:why ()
  | _ when not (B.spans b) -> ()
  | Error why ->
      equal ~msg:(strf "%d bytes refused: %s" n why) bool false on_pages;
      contains ~msg:"its cause" ~sub:"off the host's page" why
  | Ok (Bus _) -> fail "CUDA gives a dma-buf"
  | Ok (Dmabuf { fd; offset }) -> (
      Fun.protect ~finally:(fun () -> Dmabuf.close fd) @@ fun () ->
      equal ~msg:(strf "%d bytes exported" n) bool true on_pages;
      equal ~msg:"close-on-exec" bool true (Dmabuf.cloexec fd);
      equal ~msg:"its offset" int 0 offset;
      match Dmabuf.size fd with
      | Some bytes -> at_least ~msg:"its bytes" int ~than:n bytes
      | None -> fail "the descriptor is no dma-buf")

let exports () =
  S.with_ @@ fun { d; _ } ->
  let fn = gpu_fn d in
  List.iter (export_law d ~fn Device) dma_sizes;
  List.iter
    (fun fn ->
      match B.dma (B.create d 64) ~fn with
      | Ok _ -> failf "%s reaches the memory" fn
      | Error why -> contains ~msg:fn ~sub:fn why)
    [ "41:00.0"; "ffff:ff:1f.7" ]

(* Page-locked host memory exports only where CUDA says it can, which it
   says apart from GPU memory: a case of its own, so a GPU that refuses it
   still reports the export of its own memory. *)
let exports_pinned () =
  S.with_ @@ fun { d; _ } -> export_law d ~fn:(gpu_fn d) Pinned (1 lsl 22)

let memory =
  group ~timeout:120. "memory"
    [
      test "an allocation past the GPU's memory is None" past_memory;
      test "write_gpu's bytes are on the GPU when it returns" written_on_return;
      test
        "dma exports an allocation on the host's page as a dma-buf, and \
         refuses another"
        exports;
      test "dma exports page-locked host memory where CUDA says it can"
        exports_pinned;
    ]

(* Host memory *)

(* A copy between a host buffer and the GPU passes through rig's staging
   memory and locks none of the buffer. A borrow of it on the GPU locks it, a
   copy from the borrow runs on the GPU's copy engine from it, and the borrow's
   end unlocks it once the GPU ran that copy, while the buffer lives on: a
   borrow made again locks it again, until its collection. *)
let host_copies () =
  S.with_ @@ fun { d; _ } ->
  let n = 1 lsl 20 in
  let h = B.create Rig.host n and g = B.create d n in
  B.write h Bigarray.char (fun a -> Bigarray.Array1.fill a 'h');
  let locked () = S.locked (H.at h) && S.locked (H.at h + n - 1) in
  B.copy ~src:h ~dst:g;
  B.copy ~src:g ~dst:h;
  equal ~msg:"copies to and from the buffer" bool false (locked ());
  let on_g = require_some (B.borrow d h) in
  equal ~msg:"the borrow" bool true (locked ());
  B.copy ~src:on_g ~dst:g;
  equal ~msg:"the borrow ended" bool true (B.unborrow on_g);
  ignore (B.create d 0);
  equal ~msg:"once the copy ran" bool false (locked ());
  equal ~msg:"the buffer lives" (option string) None (B.dead h);
  (fun () ->
    let again = require_some (B.borrow d h) in
    equal ~msg:"a borrow made again" bool true (locked ());
    B.copy ~src:g ~dst:again)
    ();
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create d 0);
  equal ~msg:"once collected" bool false (locked ());
  equal ~msg:"the bytes" char 'h'
    (B.read h Bigarray.char (fun a -> Bigarray.Array1.get a (n - 1)))

(* A copy from the GPU into a host buffer returns once its bytes are there,
   as the host moves them out of staging memory last: its point is reached at
   the return. Into a borrow of the buffer the GPU copies directly, and the
   point is the GPU's. *)
let into_host_waits () =
  S.with_ @@ fun { d; _ } ->
  let n = 1 lsl 20 in
  let src = B.create d n and h = B.create Rig.host n in
  B.copy ~src:(B.create Rig.host n) ~dst:src;
  let p = B.submit_copy ~src ~dst:h [| B.row n |] in
  equal ~msg:"a host buffer: reached at the return" (pair bool bool)
    (true, true)
    (Rig.Point.reached p, B.reached h Read_write);
  let on_d = require_some (B.borrow d h) in
  let p = B.submit_copy ~src ~dst:on_d [| B.row n |] in
  equal ~msg:"a borrow: the GPU's point" bool true
    (Option.fold ~none:false ~some:(Rig.equal d) (Rig.Point.device p));
  Rig.Point.wait p

(* Dropped Pinned memory of the GPU, which counts in the host's budget, paces
   the collector as host buffers do, by the host's memory: GPU memory held
   live changes nothing. Over 400 drops of 4 MiB the host holds about as much
   of it at most as of dropped host buffers of that size, never most of
   what was dropped. *)
let pinned_paced () =
  S.with_ @@ fun { d; _ } ->
  let n = 4 lsl 20 and drops = 400 in
  let peak make =
    Gc.full_major ();
    ignore (B.create d 0);
    ignore (B.create Rig.host 0);
    let start = Rig.allocated Rig.host and most = ref 0 in
    for _ = 1 to drops do
      ignore (Sys.opaque_identity (make ()));
      most := Int.max !most (Rig.allocated Rig.host - start)
    done;
    !most
  in
  let host () = B.create Rig.host n
  and pinned () = B.create ~memory:Pinned d n in
  ignore (peak pinned);
  let held = peak host in
  let live = B.create d (4 lsl 30) in
  let pinned_held = peak pinned in
  less ~msg:"pinned memory held, beside 4 GiB of the GPU's" int
    ~than:((4 * held) + (64 lsl 20)) pinned_held;
  less ~msg:"of what was dropped" int ~than:(drops * n / 4) pinned_held;
  ignore (Sys.opaque_identity live)

let host_memory =
  group ~timeout:120. "host memory"
    [
      test
        "a copy locks no host buffer it does not name; a borrow locks it until \
         it goes"
        host_copies;
      test "a copy into a host buffer returns once its bytes are there"
        into_host_waits;
      test "dropped Pinned memory paces the collector by the host's memory"
        pinned_paced;
      test "map_host answers Never for read-only pages" (fun () ->
          S.with_ @@ fun { g; _ } ->
          let p = H.pages ~read_only:true H.page in
          equal string "Never" (Rig_gpu_support.refusal (C.map_host g p 64));
          H.free_pages p H.page);
    ]

(* Work *)

let fills_in_a_fresh_domain () =
  S.with_ @@ fun t ->
  let f = S.context () in
  let before = Rig.submitted t.d in
  let r, current =
    Domain.join
      (Domain.spawn (fun () ->
           let v = S.submit_work t [ S.part t.d f ] in
           (v, S.current ())))
  in
  equal int ~msg:"the value" (before + 1) (Rig.Point.value r);
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
  starts_with ~prefix:"running a fill: CUDA_ERROR_INVALID_VALUE: " why;
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
  starts_with ~prefix:"loading the image: CUDA_ERROR_" e;
  let arch = (C.facts g).arch in
  (match C.image g (S.fixture "kernels.cubin") with
  | Ok i ->
      equal string ~msg:"the cubin's GPU" "sm_89" arch;
      C.unload g (S.loaded i)
  | Error e ->
      not_equal string ~msg:"the cubin's GPU" "sm_89" arch;
      starts_with ~prefix:"loading the image: CUDA_ERROR_" e);
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

(* [count] launches (defaults to 1) of [kernel] of fixtures/launch.ptx on
   [d], each of [params] bytes of parameters whose first 8 point into the
   run's one buffer, a write, over [groups] of [threads] with [shared] bytes
   of dynamic shared memory: a run of their submission holding their
   blocks. *)
let launching ?(kernel = "ids") ?(params = 24) ?(count = 1) d
    ~groups:(gx, gy, gz) ~threads:(tx, ty, tz) ~shared =
  let refs = [| Sub.write 0 ~at:0 |] in
  let fn = require_some (Rig.Image.resolve (launch_image d) kernel) in
  let part _ = Sub.launch ~refs fn ~params in
  let s = Sub.make d (Array.init count part) in
  let run = Sub.take s in
  for i = 0 to count - 1 do
    let b = Sub.block s i in
    Run.groups run b gx gy gz;
    Run.threads run b tx ty tz;
    Run.shared run b shared
  done;
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
  (* A device's values go on past those of its slot's earlier devices. *)
  let opened = Rig.submitted d in
  let out = B.create ~memory:Pinned d 4096 in
  let refuses msg ?kernel ?params ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1))
      ?(shared = 0) () =
    let run = launching ?kernel ?params d ~groups ~threads ~shared in
    raises_match ~msg (Exn.invalid_arg ~sub:"never fit") (fun () ->
        submitted run [| out |])
  in
  (* More threads than the function has, which the setter refuses. *)
  List.iter
    (fun threads ->
      raises_match (Exn.invalid_arg ~sub:"threads per group") (fun () ->
          launching d ~groups:(1, 1, 1) ~threads ~shared:0))
    [ (1025, 1, 1); (1024, 2, 1) ];
  refuses "no groups along x" ~groups:(0, 1, 1) ();
  refuses "no threads along z" ~threads:(1, 1, 0) ();
  refuses "65536 groups along z" ~groups:(1, 1, 65536) ();
  refuses "a byte of shared memory past the largest" ~kernel:"rotate" ~params:16
    ~shared:(S.attribute largest_shared + 1)
    ();
  equal int ~msg:"values assigned" opened (Rig.submitted d);
  equal (option string) ~msg:"the device's loss" None (Rig.lost d);
  let n = 65535 * 1024 in
  let run =
    launching d ~groups:(1, 1, 65535) ~threads:(1024, 1, 1) ~shared:0
  in
  let big = B.create d (4 * n) in
  Rig.Point.wait (submitted run [| big |])

(* A launch takes as much dynamic shared memory as the GPU gives a group, past
   the 48 KiB CUDA allows by default: one word past it, 64 KiB and the largest,
   99 KiB on sm_89. [rotate] stores into each word the value of the next thread
   of its group, through the last 1 KiB of its shared memory, and at the
   largest through the first 1 KiB too. *)
let largest_shared_memory () =
  S.with_ @@ fun { d; _ } ->
  let largest = S.attribute largest_shared in
  let last shared = (shared, (shared / 4) - 256) in
  List.iter
    (fun (shared, first) ->
      let run =
        launching ~kernel:"rotate" ~params:16 d ~groups:(1, 1, 1)
          ~threads:(256, 1, 1) ~shared
      in
      let b = Sub.block (Run.submission run) 0 in
      Run.int32 run b 8 5;
      Run.int32 run b 12 first;
      let out = B.create ~memory:Pinned d (4 * 256) in
      Rig.Point.wait (submitted run [| out |]);
      equal (array int)
        ~msg:(Printf.sprintf "the words of %d bytes from word %d" shared first)
        (Array.init 256 (fun t -> 5 + (3 * ((t + 1) mod 256))))
        (words out 256))
    [ last ((48 * 1024) + 4); last (64 * 1024); last largest; (largest, 0) ]

(* A cached submission of [count] launches submitted with a warm run, their
   blocks stored anew each time through every setter, allocates nothing: one
   launch runs plain, and two play through a chain. *)
let no_allocation count () =
  S.with_ @@ fun { d; g } ->
  let run =
    launching ~count d ~groups:(1, 1, 1) ~threads:(32, 1, 1) ~shared:0
  in
  Run.release run;
  let buffers = [| B.create d 4096 |] in
  let s = Run.submission run in
  let blocks = Array.init count (Sub.block s) in
  (* [float64] stores first, where [int64] then stores [a]. *)
  let once i =
    if Sub.take s != run then fail "the run is taken";
    for j = 0 to count - 1 do
      let b = blocks.(j) in
      Run.groups run b 1 1 1;
      Run.threads run b 32 1 1;
      Run.shared run b 0;
      Run.int64 run b 0 0L;
      Run.float64 run b 8 2.5;
      Run.int64 run b 8 (Int64.of_int i);
      Run.int32 run b 16 3;
      Run.float32 run b 20 1.5
    done;
    Sys.opaque_identity (submitted run buffers)
  in
  Rig.Point.wait (once 0);
  let last = ref (once 1) in
  let pieces, plain = C.replays g in
  let bytes =
    Rig_support.allocated_bytes (fun () ->
        for i = 2 to 100 do
          last := once i
        done)
  in
  Rig.Point.wait !last;
  equal int ~msg:(strf "bytes of 99 submits of %d launches" count) 0 bytes;
  let played = if count = 1 then (0, 99) else (99, 0) in
  equal (pair int int) ~msg:"pieces and plain launches" played
    (let p, l = C.replays g in
     (p - pieces, l - plain))

let launches =
  group ~timeout:120. "launches"
    [
      test "launches the GPU cannot run are refused" refused_launches;
      test "a launch takes shared memory past 48 KiB, up to the largest"
        largest_shared_memory;
      test "a cached launch allocates nothing" (no_allocation 1);
      test "a cached run of two launches allocates nothing" (no_allocation 2);
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

(* [lose t flag] submits {!spinning}, then a fill that fails, and closes the
   lost [t]: its stop runs while the kernel runs. *)
let lose (t : S.t) flag =
  let fails = S.part t.d (S.failing 1) in
  ignore (lost (fun () -> S.submit_work t [ spinning t flag; fails ]));
  Rig.close t.d

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
  let r = require_some (maps (C.map_host g p 64)) in
  equal bool ~msg:"locked" true (S.locked p);
  let last = S.submit t [||] in
  Rig.Point.wait last;
  S.close t;
  equal int ~msg:"the word" (Rig.Point.value last) (Rig.signaled d);
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
  let ra = require_some (maps (C.map_host a.g p 256)) in
  S.close a;
  S.with_ @@ fun b ->
  let rb = require_some (maps (C.map_host b.g (p + 64) 64)) in
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

(* How long {!own_word_after_running} waits for its kernel before it gives
   the device up. *)
let give_up_ns = 300_000_000

(* A device on a word of its own, given up while a kernel runs, gives its
   word back once that work ended, which writes the last value, and a drain
   found it stopped. The GPU's next open asks whether the stopped device's
   work ended, and reads the value its edge kept, never the freed word. *)
let own_word_after_running () =
  let module W = Rig_gpu_support.Watch (C) in
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
    let t = S.open_own (module W) in
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
    equal ~msg:"the bounded wait for the spinning kernel" bool false
      (Rig.Point.wait_for ~ns:give_up_ns p);
    Rig.lose t.d "given up";
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

(* Replay *)

let kernels_image = Rig_gpu_support.loader (fun () -> S.fixture "kernels.ptx")

(* [f ()], and the pieces and plain launches of [g]'s submits meanwhile. *)
let replaying g f =
  let pieces, plain = C.replays g in
  let r = f () in
  let pieces', plain' = C.replays g in
  (r, (pieces' - pieces, plain' - plain))

(* The pieces and plain launches of a run of [n] launches, as rig_cuda.mli's
   Replay section cuts it, summed over [runs]. *)
let cut runs =
  let count n = if n = 1 then (0, 1) else ((n + 31) / 32, 0) in
  List.fold_left
    (fun (p, l) n ->
      let p', l' = count n in
      (p + p', l + l'))
    (0, 0)
    (List.filter (fun n -> n > 0) runs)

(* A launch of [fn] of [params] bytes over [groups] of [threads] with
   [shared] bytes of dynamic shared memory, its first 8 bytes a write of
   the buffer [slot] unless [slot] is negative, its other bytes as [store]
   stores them. *)
type launch = {
  fn : Rig.Image.func;
  params : int;
  slot : int;
  groups : int * int * int;
  threads : int * int * int;
  shared : int;
  store : Run.t -> Sub.block -> unit;
}

let launch ?(slot = 0) ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1))
    ?(shared = 0) ?(store = fun _ _ -> ()) fn ~params =
  { fn; params; slot; groups; threads; shared; store }

(* A run of a submission of [ls] on [d], their blocks stored. *)
let run_of d ls =
  let part l =
    let refs = if l.slot < 0 then [||] else [| Sub.write l.slot ~at:0 |] in
    Sub.launch ~refs l.fn ~params:l.params
  in
  let s = Sub.make d (Array.of_list (List.map part ls)) in
  let run = Sub.take s in
  List.iteri
    (fun i l ->
      let b = Sub.block s i in
      let gx, gy, gz = l.groups and tx, ty, tz = l.threads in
      Run.groups run b gx gy gz;
      Run.threads run b tx ty tz;
      Run.shared run b l.shared;
      l.store run b)
    ls;
  run

let func image name = require_some (Rig.Image.resolve image name)

(* [ids] over one thread, storing [v] into word [j] of its buffer, the
   offset its ref holds. *)
let store_id d j v =
  launch (func (launch_image d) "ids") ~params:24 ~store:(fun run b ->
      Run.int64 run b 0 (Int64.of_int (4 * j));
      Run.int64 run b 8 (Int64.of_int v);
      Run.int32 run b 16 0;
      Run.float32 run b 20 0.)

(* A chain's node runs the bounds a plain launch runs, updated from the
   placeholder's launch of one thread with no parameters or shared memory:
   each case plays a piece of a length no earlier case played, so that
   every update starts from the placeholder. A grid of 2{^31} - 1 groups
   along x is left out, as it takes 2 s. *)
let placeholder_bounds () =
  S.with_ @@ fun ({ d; g } as t) ->
  let fn = func (launch_image d) in
  let pinned n = B.create ~memory:Pinned d n in
  let played msg ls buffers =
    let run = run_of d ls in
    let (), counts =
      replaying g (fun () -> Rig.Point.wait (submitted run buffers))
    in
    equal (pair int int) ~msg:(msg ^ ": pieces and plain launches") (1, 0)
      counts
  in
  (* [mark] takes no parameters. *)
  let marks = Array.init 2 (fun _ -> pinned 4) in
  let mark = launch (fn "mark") ~params:0 ~slot:(-1) in
  let count slot = launch (fn "marks") ~params:8 ~slot in
  played "no parameters" [ count 0; mark; mark; mark; count 1 ] marks;
  equal int ~msg:"marks" 3 (H.get32 (H.at marks.(1)));
  (* Each launch [j] stores its words [10j + 1] and on into its own buffer. *)
  let wide msg ~name ~params ats ls =
    let words_of = List.length ats in
    let outs = Array.init (List.length ls) (fun _ -> pinned (4 * words_of)) in
    let each j f =
      launch f ~params ~slot:j ~store:(fun run b ->
          List.iteri (fun k at -> Run.int32 run b at ((10 * j) + k + 1)) ats)
    in
    played msg (List.mapi each ls) outs;
    Array.iteri
      (fun j out ->
        equal (array int)
          ~msg:(strf "%s %d's words" name j)
          (Array.init (List.length ats) (fun k -> (10 * j) + k + 1))
          (words out (List.length ats)))
      outs
  in
  wide "4,096 bytes of parameters" ~name:"page" ~params:4096 [ 8; 4092 ]
    [ fn "page"; fn "page" ];
  (match S.wide_binary t with
  | None -> ()
  | Some binary ->
      let widest = func (require_ok (Rig.Image.load d binary)) "widest" in
      wide "32,764 bytes of parameters" ~name:"widest" ~params:32764
        [ 8; 16384; 32760 ] [ widest; widest; widest ]);
  (* [rotate] at 1 KiB, the largest, 48 KiB and a word, and the largest. *)
  let largest = S.attribute largest_shared in
  let shared = [ 1024; largest; (48 * 1024) + 4; largest ] in
  let outs = Array.init 4 (fun _ -> pinned (4 * 256)) in
  let rotate j shared =
    launch (fn "rotate") ~params:16 ~slot:j ~threads:(256, 1, 1) ~shared
      ~store:(fun run b ->
        Run.int32 run b 8 (5 + j);
        Run.int32 run b 12 0)
  in
  played "dynamic shared memory" (List.mapi rotate shared) outs;
  Array.iteri
    (fun j out ->
      equal (array int)
        ~msg:(strf "rotate %d's words" j)
        (Array.init 256 (fun i -> 5 + j + (3 * ((i + 1) mod 256))))
        (words out 256))
    outs;
  (* [ids] over 65535 groups along each axis, and 1024 threads along x and
     y, each storing [j + (j + 1)k] into word [k] of one buffer. *)
  let n = 65535 * 1024 in
  let big = B.create d (4 * n) in
  let shapes =
    [
      ((1, 1, 65535), (1024, 1, 1));
      ((65535, 1, 1), (1, 1024, 1));
      ((1, 65535, 1), (1024, 1, 1));
      ((65535, 1, 1), (1024, 1, 1));
      ((1, 1, 65535), (16, 64, 1));
      ((1, 65535, 1), (1, 1024, 1));
    ]
  in
  let ids j (groups, threads) =
    launch (fn "ids") ~params:24 ~groups ~threads ~store:(fun run b ->
        Run.int64 run b 8 (Int64.of_int j);
        Run.int32 run b 16 (j + 1);
        Run.float32 run b 20 0.)
  in
  played "the largest grids" (List.mapi ids shapes) [| big |];
  let s = Sub.make ~fixed:[ (big, Read) ] d [||] in
  let at k = Nativeint.of_int (Sub.address s big + (4 * k)) in
  let word k = String.get_int32_le (S.read_gpu (at k) 4) 0 in
  List.iter
    (fun k ->
      equal int32 ~msg:(strf "word %d" k)
        (Int32.of_int (5 + (6 * k)))
        (word k))
    [ 0; 1; n / 2; n - 1 ]

(* An update CUDA refuses runs its piece plain. No image of a function whose
   launches CUDA runs and whose updates it refuses, such as one that
   launches kernels from the GPU, loads without CUDA's linker, so the
   refusal comes from a bound: [rotate]'s largest shared memory, lowered
   behind rig's back below the 1 KiB a run of two takes, which CUDA then
   refuses at the first update and at the plain launch that follows it. *)
let refused_update () =
  S.with_ @@ fun { d; g } ->
  let fn = func (launch_image d) "rotate" in
  let rotate =
    launch fn ~params:16 ~threads:(256, 1, 1) ~shared:1024 ~store:(fun run b ->
        Run.int32 run b 8 5;
        Run.int32 run b 12 0)
  in
  let out = B.create ~memory:Pinned d (4 * 256) in
  let run = run_of d [ rotate; rotate ] in
  let (), counts =
    replaying g (fun () -> Rig.Point.wait (submitted run [| out |]))
  in
  equal (pair int int) ~msg:"pieces and plain launches before" (1, 0) counts;
  S.limit_shared (Rig.Image.entry fn) 512;
  let run = Sub.take (Run.submission run) in
  let why, counts =
    replaying g (fun () -> lost (fun () -> submitted run [| out |]))
  in
  starts_with ~prefix:"launching a kernel: CUDA_ERROR_INVALID_VALUE" why;
  equal (pair int int) ~msg:"pieces and plain launches after" (0, 1) counts

(* The order law. A copy that follows nothing runs on COPY:0, then come [a]
   steps of a counter, then, if [across], [twice] of the copy's last words,
   which follows the copy, then [b] steps: the counter counts every step and
   [twice] reads the words the copy wrote. Unless [named], the steps name no
   buffer and step a counter the driver allocated, out of rig's sight: they
   follow nothing, run on COMPUTE:0 in their order, and play as rig_cuda.mli
   cuts them, [a] then [1 + b] launches with [twice] and [a + b] without.
   [named] steps name a buffer, which may share memory with the copy, so
   each follows it; the first launch's wait covers the others, which play
   with it as one run. The copy is large, so that a [twice] that did not
   wait reads words of the case before. *)
let copied = 1 lsl 20
let tail = 256
let twice_c = 7
let order_buffers = ref None
let order_cases = ref 0

let order_law (named, a, across, b) =
  S.with_shared @@ fun ({ d; g } as t) ->
  let src, dst, out, stepped =
    match !order_buffers with
    | Some (d', buffers) when d' == d -> buffers
    | Some _ | None ->
        let pinned n = B.create ~memory:Pinned d n in
        let buffers =
          ( pinned (4 * copied),
            pinned (4 * copied),
            pinned (4 * tail),
            pinned 8 )
        in
        order_buffers := Some (d, buffers);
        buffers
  in
  let counter = require_some (C.alloc g Pinned 8) in
  incr order_cases;
  let first = 1000 * !order_cases in
  for j = 0 to tail - 1 do
    H.set32 (H.at src + (4 * (copied - tail + j))) (first + j)
  done;
  H.set64 (host counter) 0;
  H.set64 (H.at stepped) 0;
  let step i =
    if not named then S.launch t "step" (address counter) i
    else
      Rig_gpu_support.launching (kernels_image d) "step" ~params:16
        [| (stepped, Rig.Read_write, 0) |]
        (fun run b -> Run.int64 run b 8 (Int64.of_int i))
  in
  let twice =
    Rig_gpu_support.launching (launch_image d) "twice" ~params:20 ~threads:tail
      [| (out, Rig.Read_write, 0); (dst, Rig.Read, 8) |]
      (fun run b ->
        Run.int64 run b 8 (Int64.of_int (4 * (copied - tail)));
        Run.int32 run b 16 twice_c)
  in
  let copy =
    Rig_gpu_support.work Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let works =
    (copy :: List.init a step)
    @ (if across then [ twice ] else [])
    @ List.init b (fun i -> step (a + i))
  in
  let twices = if across then 1 else 0 in
  let runs =
    if named then [ a + twices + b ]
    else if across then [ a; 1 + b ]
    else [ a + b ]
  in
  cover "a cut before a launch after COPY:0" ((not named) && across && a > 0);
  cover "a launch after COPY:0 an earlier launch's wait covers"
    (named && across && a > 0);
  cover "a run of more than 32 launches" (List.exists (fun n -> n > 32) runs);
  cover "a run of 33 launches" (List.mem 33 runs);
  cover "a run of one launch" (List.mem 1 runs);
  let (), counts =
    replaying g (fun () -> Rig.Point.wait (S.submit_work t works))
  in
  let steps = H.get64 (if named then H.at stepped else host counter) in
  C.free g counter;
  equal int ~msg:"steps" (a + b) steps;
  if across then
    equal (array int) ~msg:"twice's words"
      (Array.init tail (fun j -> (2 * (first + j)) + twice_c))
      (words out tail);
  equal (pair int int) ~msg:"pieces and plain launches" (cut runs) counts

let run_length =
  Gen.frequency
    [
      (2, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 2; 31; 32; 33; 64; 65 ]);
      (1, Gen.int_range 0 70);
    ]

let order_case =
  Gen.with_pp
    (fun ppf (named, a, across, b) ->
      Format.fprintf ppf "(%b, %d, %b, %d)" named a across b)
    (Gen.quad Gen.bool run_length Gen.bool run_length)

(* An image unloads while a chain's nodes hold its function: the memory of
   its 256 MiB global returns, and the chain then plays another image's
   launches. *)
let unload_chained () =
  S.with_ @@ fun { d; g } ->
  let before = S.free_memory () in
  let[@inline never] play () =
    let image = require_ok (Rig.Image.load d (S.fixture "global.ptx")) in
    let touch = launch (func image "touch") ~params:16 ~slot:(-1) in
    let run = run_of d [ touch; touch ] in
    let (), counts =
      replaying g (fun () -> Rig.Point.wait (submitted run [||]))
    in
    equal (pair int int) ~msg:"touch's pieces and plain launches" (1, 0) counts;
    at_most int ~msg:"loaded" ~than:(before - (global / 2)) (S.free_memory ())
  in
  play ();
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (B.create d 0)
  done;
  at_least int ~msg:"unloaded" ~than:(before - (global / 2)) (S.free_memory ());
  let out = B.create ~memory:Pinned d 8 in
  let run = run_of d [ store_id d 0 1; store_id d 1 2 ] in
  let (), counts =
    replaying g (fun () -> Rig.Point.wait (submitted run [| out |]))
  in
  equal (pair int int) ~msg:"ids' pieces and plain launches" (1, 0) counts;
  equal (array int) ~msg:"ids' words" [| 1; 2 |] (words out 2)

(* Spins until [flag] is set, failing if [what] has not come within 10 s. *)
let await flag what =
  let t0 = Rig.Profile.now () in
  while not (Atomic.get flag) do
    if Rig.Profile.now () - t0 > 10 * second then
      failf "no %s within 10 s" what;
    Domain.cpu_relax ()
  done

(* Two submits that need one new length make its chain once. [a]'s grow
   waits until [b]'s has handed the chain, and [b]'s grow returns only once
   [a]'s submit has: [a]'s grow makes nothing, its room installs the chain
   [b]'s grow made, and its piece plays through it before [b]'s room finds
   the chain installed. Each submit plays one piece of its own launches. *)
let one_growth () =
  let grows = Atomic.make 0 in
  let a_grows = Atomic.make false and handed = Atomic.make false in
  let a_returned = Atomic.make false in
  let module Driver = struct
    include C

    let grow g =
      match Atomic.fetch_and_add grows 1 with
      | 0 ->
          Atomic.set a_grows true;
          await handed "chain handed by b's grow";
          C.grow g
      | 1 ->
          let grown = C.grow g in
          Atomic.set handed true;
          await a_returned "return of a's submit";
          grown
      | _ -> C.grow g
  end in
  let ({ S.d; g } as t) = S.open_as (module Driver) in
  Fun.protect ~finally:(fun () -> S.close t) @@ fun () ->
  let n = 5 in
  ignore (launch_image d);
  let base = C.replays g and made = C.made g in
  let since (p, l) = (p - fst base, l - snd base) in
  let play first out =
    let run = run_of d (List.init n (fun j -> store_id d j (first + j))) in
    let p = submitted run [| out |] in
    Rig.Point.wait p
  in
  let out_a = B.create ~memory:Pinned d (4 * n) in
  let out_b = B.create ~memory:Pinned d (4 * n) in
  let a =
    Domain.spawn (fun () ->
        Fun.protect ~finally:(fun () -> Atomic.set a_returned true)
        @@ fun () ->
        play 1 out_a;
        since (C.replays g))
  in
  await a_grows "grow of a's submit";
  play 101 out_b;
  let after_a = Domain.join a in
  equal int ~msg:"grows" 2 (Atomic.get grows);
  equal int ~msg:"chains made" 1 (C.made g - made);
  equal (pair int int) ~msg:"pieces and plain launches as a's submit returned"
    (1, 0) after_a;
  equal (pair int int) ~msg:"pieces and plain launches of both" (2, 0)
    (since (C.replays g));
  equal (array int) ~msg:"a's words" (Array.init n (fun j -> 1 + j))
    (words out_a n);
  equal (array int) ~msg:"b's words" (Array.init n (fun j -> 101 + j))
    (words out_b n)

(* The test's executable, run with [jitless] where CUDA compiles no PTX,
   prints what loading fixtures/kernels.ptx on GPU 0 answers, then the
   counter of two [step]s of fixtures/kernels.cubin in one submit, and the
   device's pieces and plain launches. *)
let jitless = "--jitless"

let print_jitless () =
  let ({ S.d; g } as t) = S.open_ () in
  Fun.protect ~finally:(fun () -> S.close t) @@ fun () ->
  (match Rig.Image.load d (S.fixture "kernels.ptx") with
  | Ok _ -> print_endline "loaded"
  | Error why -> print_endline why);
  match Rig.Image.load d (S.fixture "kernels.cubin") with
  | Error why -> print_string why
  | Ok image ->
      let counter = B.create ~memory:Pinned d 8 in
      H.set64 (H.at counter) 0;
      let step i =
        Rig_gpu_support.launching image "step" ~params:16
          [| (counter, Rig.Read_write, 0) |]
          (fun run b -> Run.int64 run b 8 (Int64.of_int i))
      in
      Rig.Point.wait (S.submit_work t [ step 0; step 1 ]);
      let pieces, plain = C.replays g in
      Printf.printf "%d %d %d\n%!" (H.get64 (H.at counter)) pieces plain

(* A GPU whose driver compiles no PTX, as under CUDA_DISABLE_PTX_JIT=1 in a
   process of its own, has no placeholder: a cubin loads all the same, and
   its run of two launches plays plain. *)
let without_placeholder () =
  let arch = S.with_ (fun { g; _ } -> (C.facts g).arch) in
  if arch <> "sm_89" then
    skip ~reason:"fixtures/kernels.cubin is compiled for sm_89" ();
  let code, out = child jitless (with_var "CUDA_DISABLE_PTX_JIT" "1") in
  equal ~msg:"the child's exit code" int 0 code;
  match String.split_on_char '\n' out with
  | [ ptx; ran; "" ] ->
      contains ~msg:"kernels.ptx's load"
        ~sub:"loading the image: CUDA_ERROR_JIT_COMPILATION_DISABLED" ptx;
      equal string ~msg:"the counter, pieces and plain launches" "2 0 2" ran
  | _ -> failf "the child printed %S" out

let replay =
  group ~timeout:120. "replay"
    [
      test "a chain's node runs the bounds of a launch from the placeholder"
        placeholder_bounds;
      test "an update CUDA refuses runs its piece plain" refused_update;
      prop ~count:200
        "runs play in order, cut at 32 and at a launch's first wait for COPY:0"
        order_case order_law;
      test "an image unloads while a chain holds its function" unload_chained;
      test "two submits that need one new length make its chain once"
        one_growth;
      test "a GPU whose driver compiles no PTX loads a cubin and plays plain"
        without_placeholder;
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

  let map ?(cover = fun label cond -> cover label cond) m (area, a, n) =
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
    | Maps r ->
        Mutex.protect s.lock (fun () -> s.live <- s.live @ [ r ]);
        true
    | Never | No_room -> false

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

(* Chains with launches queued *)

module Chains = struct
  type kind = Step | Set

  (* A launch advances one of two counters: [Step] through kernels.ptx's
     [step], which stores the next value only if the counter holds the
     launch's own, [Set] through launch.ptx's [ids], which stores the
     launch's next value. A launch that ran another launch's parameters, or
     before the launches it follows, leaves its counter off the model's. *)
  type launch = kind * int

  let launches : launch list Gen.t =
    let name = function Step -> "step" | Set -> "set" in
    let pp ppf ls =
      Format.fprintf ppf "[%s]"
        (String.concat " "
           (List.map (fun (k, w) -> strf "%s%d" (name k) w) ls))
    in
    let length =
      Gen.frequency
        [
          (2, Gen.of_list ~pp:Format.pp_print_int [ 1; 2; 31; 32; 33 ]);
          (1, Gen.int_range 1 40);
        ]
    in
    let launch =
      Gen.pair
        (Gen.of_list ~pp:(fun ppf k -> Format.pp_print_string ppf (name k))
           [ Step; Set ])
        (Gen.int_range 0 1)
    in
    Gen.with_pp pp (Gen.list ~size:length launch)

  (* The reference: each counter's value once the work submitted ends. *)
  let make () = [| 0; 0 |]

  let rec changes = function
    | (k, _) :: ((k', _) :: _ as rest) -> k <> k' || changes rest
    | [] | [ _ ] -> false

  let submit ?(cover = fun label cond -> cover label cond) m ls =
    let n = List.length ls in
    cover "a run of one launch" (n = 1);
    cover "a run of more than 32 launches" (n > 32);
    cover "a run whose function changes" (changes ls);
    List.iter (fun (_, w) -> m.(w) <- m.(w) + 1) ls

  let read m = (m.(0), m.(1))

  (* The system: submits never wait, so a submit updates chains whose
     earlier launches are still queued. A lock orders each value's calls,
     and two values' calls run concurrently. *)
  type sys = {
    t : S.t;
    counters : B.t array;
    values : int array;
    lock : Mutex.t;
    mutable last : Rig.Point.t option;
  }

  let start () =
    let t = shared () in
    let counter _ =
      let b = B.create ~memory:Pinned t.d 8 in
      H.set64 (H.at b) 0;
      b
    in
    {
      t;
      counters = Array.init 2 counter;
      values = [| 0; 0 |];
      lock = Mutex.create ();
      last = None;
    }

  let work s (kind, w) =
    let v = s.values.(w) in
    s.values.(w) <- v + 1;
    let uses = [| (s.counters.(w), Rig.Read_write, 0) |] in
    match kind with
    | Step ->
        Rig_gpu_support.launching (kernels_image s.t.d) "step" ~params:16 uses
          (fun run b -> Run.int64 run b 8 (Int64.of_int v))
    | Set ->
        Rig_gpu_support.launching (launch_image s.t.d) "ids" ~params:24 uses
          (fun run b ->
            Run.int64 run b 8 (Int64.of_int (v + 1));
            Run.int32 run b 16 0;
            Run.float32 run b 20 0.)

  let submit_sys s ls =
    Mutex.protect s.lock (fun () ->
        s.last <- Some (S.submit_work s.t (List.map (work s) ls)))

  let wait s = Option.iter Rig.Point.wait s.last

  let read_sys s =
    Mutex.protect s.lock (fun () ->
        wait s;
        let value w = H.get64 (H.at s.counters.(w)) in
        (value 0, value 1))

  let release s = Mutex.protect s.lock (fun () -> wait s)
end

let chains = abstract "c" ~release:Chains.release

(* On two domains a submit's covers count once per order, so the model
   labels nothing there. *)
let chains_commands ~cover =
  [
    command "start" (Gen.unit @-> makes chains) Chains.make Chains.start;
    command "submit"
      (chains ^-> Chains.launches @-> returns unit)
      (Chains.submit ~cover) Chains.submit_sys;
    command "read"
      (chains ^-> returns (pair int int))
      Chains.read Chains.read_sys;
  ]

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        (registry_commands ~cover);
      stateful ~count:20 ~domains:2
        "map_host from two domains shares a host range both ways"
        (registry_commands ~cover:(fun _ _ -> ()));
      stateful ~count:30 ~steps:20
        "chains play each launch's own values with launches queued"
        (chains_commands ~cover);
      stateful ~count:5 ~domains:2
        "chains from two domains play each launch's own values"
        (chains_commands ~cover:(fun _ _ -> ()));
    ]

let () =
  match Sys.argv with
  | [| _; arg |] when arg = unloadable -> print_unloadable ()
  | [| _; arg |] when arg = jitless -> print_jitless ()
  | _ ->
      exit
        (run "rig_cuda"
           [
             opening;
             facts;
             memory;
             host_memory;
             work;
             images;
             launches;
             timeline;
             replay;
             stateful;
           ])
