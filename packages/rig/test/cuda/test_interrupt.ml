(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of CUDA's device. A call is made once for each allocation
   it makes, with another call of the driver from a Gc.Memprof callback at
   that allocation, on the calling thread. Inside a lock of the driver, the
   device's graph guard, an image's lock or the registry of page-locked
   memory, the callback's call raises Invalid_argument, having changed
   nothing: an entry point of the driver's checks the thread's locks as it
   starts, and a function of the driver's signature that takes the held lock
   again is refused at the take. Between locks the call runs as any call
   does. Through rig, a call from such a callback raises Invalid_argument
   inside a lock and runs between locks, and the device is not lost. *)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let strf = Printf.sprintf
let timeout = 120.
let stopped_graph = "making the graph: the device was stopped"

(* The text of the driver's refusal of a take by a lock's holder. *)
let retaken = "Rig_cuda: a lock taken again by the thread that holds it"

let found : Rig_edge.entry_answer -> Rig_edge.entry = function
  | Entry e -> e
  | Absent -> fail "no such function"
  | No_memory n -> failf "no memory for %d bytes" n

let image g = S.loaded (require_ok (C.image g (S.fixture "kernels.ptx")))

(* [walk attempt] runs [attempt g n] on a device [g] the driver opened alone,
   from [n = 1] until an attempt no longer reaches its [n]-th allocation.
   [attempt] stops [g] with {!S.stop_driver}, and a failed attempt leaves
   it to {!S.release}. *)
let walk attempt =
  let rec from n =
    let reached =
      Fun.protect ~finally:S.release (fun () -> attempt (S.driver ()) n)
    in
    if reached then from (n + 1)
  in
  from 1

(* A stop from a callback in a graph call: inside the guard it is refused
   and the device works on; between locks it stops the device, and the
   graph call answers so. *)
let stop_in_graph () =
  let seen = Rig_support.seen () in
  walk (fun g n ->
      let m = image g in
      let k = S.kernel (found (C.entry m "empty")).code 0 0 in
      let landing, landed =
        Rig_support.landing seen retaken (fun () -> S.stop_driver g)
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore ((C.capability g).graph [| k |]))
      in
      let what = strf "from allocation %d" n in
      let stopped = Rig_support.answered what landed = Some false in
      let after = what ^ ", a graph after the call" in
      (match (C.capability g).graph [| k |] with
      | Error why when stopped -> equal ~msg:after string stopped_graph why
      | Error why -> failf "%s: %s" after why
      | Ok _ when stopped -> failf "%s: made on a stopped device" after
      | Ok _ -> ());
      if not stopped then S.stop_driver g;
      C.unload g m;
      reached);
  Rig_support.covered seen

(* An open of the GPU from a callback in a graph call of its device: inside
   the guard it is refused; between locks it answers that the GPU has a
   device open. Neither claims the GPU, which opens once the device
   stopped. *)
let make_in_graph () =
  let seen = Rig_support.seen () in
  walk (fun g n ->
      let m = image g in
      let k = S.kernel (found (C.entry m "empty")).code 0 0 in
      let opened = ref 0 in
      let make () =
        match (C.gpu 0).make ~word:(H.word ()) with
        | Ok h ->
            incr opened;
            C.stop h ~fault:None
        | Error _ -> ()
      in
      let landing, landed = Rig_support.landing seen "Rig_cuda.gpu: " make in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore ((C.capability g).graph [| k |]))
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      equal ~msg:(strf "from allocation %d, the opens in the call" n) int 0
        !opened;
      C.unload g m;
      S.stop_driver g;
      (match (C.gpu 0).make ~word:(H.word ()) with
      | Ok h -> C.stop h ~fault:None
      | Error why ->
          failf "from allocation %d, the open after the stop: %s" n why);
      reached);
  Rig_support.covered seen

(* A function asked for from a callback while another function of its image
   is: inside the image's lock the entry is refused, and between locks it
   makes the function. Either way the function is made when asked for
   again. *)
let entry_in_entry () =
  let seen = Rig_support.seen () in
  walk (fun g n ->
      let m = image g in
      let landing, landed =
        Rig_support.landing seen retaken (fun () -> ignore (C.entry m "step"))
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore (C.entry m "empty"))
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      let step = found (C.entry m "step") and empty = found (C.entry m "empty") in
      not_equal ~msg:(strf "from allocation %d, the functions" n) int empty.code
        step.code;
      C.unload g m;
      S.stop_driver g;
      reached);
  Rig_support.covered seen

(* A free of page-locked memory from a callback in a mapping of other host
   memory: inside the registry's lock it is refused and leaves the memory
   locked for a later free; between locks it unlocks the memory. *)
let free_in_map_host () =
  let seen = Rig_support.seen () in
  walk (fun g n ->
      S.bind g;
      let p = H.pages H.page and q = H.pages H.page in
      let r = require_some (C.map_host g p 64) in
      let landing, landed =
        Rig_support.landing seen retaken (fun () -> C.free g r)
      in
      let mapped = ref None in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            mapped := C.map_host g q 64)
      in
      let what = strf "from allocation %d" n in
      let freed = Rig_support.answered what landed = Some false in
      equal ~msg:(what ^ ", the first memory locked") bool (not freed)
        (S.locked p);
      if not freed then C.free g r;
      Option.iter (C.free g) !mapped;
      equal ~msg:(what ^ ", locked after the frees") bool false
        (S.locked p || S.locked q);
      S.stop_driver g;
      H.free_pages p H.page;
      H.free_pages q H.page;
      reached);
  Rig_support.covered seen

(* Through rig *)

(* [walk_rig attempt] is {!walk} on a device that rig opened, which it
   closes after each attempt. *)
let walk_rig attempt =
  let rec from n =
    let t = S.open_ () in
    let reached =
      Fun.protect ~finally:(fun () -> S.close t) (fun () -> attempt t.S.d n)
    in
    if reached then from (n + 1)
  in
  from 1

let loaded d =
  match Rig.Image.load d (S.fixture "kernels.ptx") with
  | Ok i -> i
  | Error why -> fail why

(* A function of an image asked for from a callback while another function
   of it is asked for: inside a lock the call is refused, and between locks
   it answers. The device is not lost, and the function is made when asked
   for again. *)
let entry_in_entry_rig () =
  let seen = Rig_support.seen () in
  walk_rig (fun d n ->
      let i = loaded d in
      let mem () =
        if Option.is_none (Rig.Image.resolve i "step") then
          fail "no function step"
      in
      let landing, landed =
        Rig_support.landing seen "Rig.Image.resolve: " mem
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore (Rig.Image.resolve i "empty"))
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      is_none ~msg:(strf "from allocation %d, the loss" n) (Rig.lost d);
      equal ~msg:(strf "from allocation %d, the function" n) bool true
        (Option.is_some (Rig.Image.resolve i "step"));
      reached);
  Rig_support.covered seen

(* An open of the device's GPU under another name from a callback in a
   graph call: inside the guard it is refused, and between locks it answers
   that the GPU has a device open. Neither opens a device, and the device
   is not lost. *)
let open_in_graph_rig () =
  let seen = Rig_support.seen () in
  walk_rig (fun d n ->
      let i = loaded d in
      let empty = Rig.Image.entry (Option.get (Rig.Image.resolve i "empty")) in
      let cap = Option.get (Rig.capability d Rig_cuda_abi.key) in
      let k = S.kernel empty 0 0 in
      let opened = ref 0 in
      let call () =
        match Rig.open_ { (C.gpu 0) with name = "CUDA:refused" } with
        | Ok d' ->
            incr opened;
            Rig.close d'
        | Error _ -> ()
      in
      let landing, landed = Rig_support.landing seen "Rig.open_: " call in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore (cap.graph [| k |]))
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      equal ~msg:(strf "from allocation %d, the opens in the call" n) int 0
        !opened;
      is_none ~msg:(strf "from allocation %d, the loss" n) (Rig.lost d);
      ignore (Sys.opaque_identity i);
      reached);
  Rig_support.covered seen

(* A close from a callback in a graph call: inside the guard it is refused,
   and between locks it stops the device. After a close, a graph call
   answers that the device was stopped. *)
let close_in_graph () =
  let seen = Rig_support.seen () in
  walk_rig (fun d n ->
      let i = loaded d in
      let empty = Rig.Image.entry (Option.get (Rig.Image.resolve i "empty")) in
      let cap = Option.get (Rig.capability d Rig_cuda_abi.key) in
      let k = S.kernel empty 0 0 in
      let landing, landed =
        Rig_support.landing seen "Rig.close: " (fun () -> Rig.close d)
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            ignore (cap.graph [| k |]))
      in
      ignore (Rig_support.answered (strf "from allocation %d" n) landed);
      Rig.close d;
      (match cap.graph [| k |] with
      | Error why ->
          equal ~msg:(strf "from allocation %d, the graph's refusal" n) string
            stopped_graph why
      | Ok _ -> failf "from allocation %d, a graph made after the close" n);
      ignore (Sys.opaque_identity i);
      reached);
  Rig_support.covered seen

(* The stops run last: a driver that cannot stop the device from a graph
   call leaves it open for the rest of the process. *)
let () =
  exit
    (run "rig_cuda interrupt"
       [
         group ~timeout "a call from a callback at any allocation"
           [
             test "an entry in an entry of its image, refused inside its lock"
               entry_in_entry;
             test "a free in a mapping of host memory, refused inside its lock"
               free_in_map_host;
             test "an open in a graph call claims nothing" make_in_graph;
           ];
         group ~timeout
           "a call to rig from a callback at any allocation keeps the device"
           [
             test "a function asked for while another is" entry_in_entry_rig;
             test "an open in a graph call" open_in_graph_rig;
           ];
         group ~timeout "a stop from a callback at any allocation of a graph call"
           [
             test "through rig, refused inside the guard" close_in_graph;
             test "refused inside the guard" stop_in_graph;
           ];
       ])
