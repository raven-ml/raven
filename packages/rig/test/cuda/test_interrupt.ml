(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of CUDA's device. A call is made once for each allocation
   it makes, with another call of the driver from a Gc.Memprof callback at
   that allocation, on the calling thread. A call that holds a lock of the
   driver there, the device's graph guard, an image's lock or the registry
   of page-locked memory, makes the callback's call raise Rig_cuda.Refused,
   having changed nothing, where it would otherwise take that lock again.
   Through rig, a call from such a callback keeps the device: it works, or
   is refused and leaves its work to a later call; an open refused so
   raises Invalid_argument. *)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let strf = Printf.sprintf
let timeout = 120.
let stopped_graph = "making the graph: the device was stopped"

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

(* [refusing call] is [call], counting its refusals in [n]. *)
let refusing n call () = try call () with C.Refused _ -> incr n

(* A stop from a callback in a graph call: it stops the device, or raises
   Refused and leaves it working. *)
let stop_in_graph () =
  let refusals = ref 0 in
  walk (fun g n ->
      let m = image g in
      let k = S.kernel (found (C.entry m "empty")).code 0 0 in
      let stopped = ref false in
      let stop () =
        S.stop_driver g;
        stopped := true
      in
      let reached =
        Rig_support.interrupted ~landing:(Call (refusing refusals stop)) n
          (fun () -> ignore ((C.capability g).graph [| k |]))
      in
      let what = strf "from allocation %d, a graph after the call" n in
      (match (C.capability g).graph [| k |] with
      | Error why when !stopped -> equal ~msg:what string stopped_graph why
      | Error why -> failf "%s: %s" what why
      | Ok _ when !stopped -> failf "%s: made on a stopped device" what
      | Ok _ -> ());
      if not !stopped then S.stop_driver g;
      C.unload g m;
      reached);
  greater ~msg:"stops refused" int ~than:0 !refusals

(* An open of the GPU from a callback in a graph call of its device: it
   raises Refused, or answers that the GPU has a device open, and claims
   nothing, so the GPU opens once the device stopped. *)
let make_in_graph () =
  let refusals = ref 0 in
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
      let reached =
        Rig_support.interrupted ~landing:(Call (refusing refusals make)) n
          (fun () -> ignore ((C.capability g).graph [| k |]))
      in
      equal ~msg:(strf "from allocation %d, the opens in the call" n) int 0
        !opened;
      C.unload g m;
      S.stop_driver g;
      (match (C.gpu 0).make ~word:(H.word ()) with
      | Ok h -> C.stop h ~fault:None
      | Error why ->
          failf "from allocation %d, the open after the stop: %s" n why);
      reached);
  greater ~msg:"opens refused" int ~than:0 !refusals

(* A function asked for from a callback while another function of its image
   is: it is made, or the entry raises Refused and the function is made when
   asked for again. *)
let entry_in_entry () =
  let refusals = ref 0 in
  walk (fun g n ->
      let m = image g in
      let reached =
        Rig_support.interrupted
          ~landing:(Call (refusing refusals (fun () -> ignore (C.entry m "step"))))
          n
          (fun () -> ignore (C.entry m "empty"))
      in
      let step = found (C.entry m "step") and empty = found (C.entry m "empty") in
      not_equal ~msg:(strf "from allocation %d, the functions" n) int empty.code
        step.code;
      C.unload g m;
      S.stop_driver g;
      reached);
  greater ~msg:"entries refused" int ~than:0 !refusals

(* A free of page-locked memory from a callback in a mapping of other host
   memory: the memory is unlocked, or the free raises Refused and leaves it
   locked for a later free. *)
let free_in_map_host () =
  let refusals = ref 0 in
  walk (fun g n ->
      S.bind g;
      let p = H.pages H.page and q = H.pages H.page in
      let r = require_some (C.map_host g p 64) in
      let freed = ref false in
      let free () =
        C.free g r;
        freed := true
      in
      let mapped = ref None in
      let reached =
        Rig_support.interrupted ~landing:(Call (refusing refusals free)) n
          (fun () -> mapped := C.map_host g q 64)
      in
      equal ~msg:(strf "from allocation %d, the first memory locked" n) bool
        (not !freed) (S.locked p);
      if not !freed then C.free g r;
      Option.iter (C.free g) !mapped;
      equal ~msg:(strf "from allocation %d, locked after the frees" n) bool
        false
        (S.locked p || S.locked q);
      S.stop_driver g;
      H.free_pages p H.page;
      H.free_pages q H.page;
      reached);
  greater ~msg:"frees refused" int ~than:0 !refusals

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
   of it is asked for: it is made, at once or when asked for again, and the
   device is not lost. *)
let entry_in_entry_rig () =
  let refusals = ref 0 in
  walk_rig (fun d n ->
      let i = loaded d in
      let call () =
        try ignore (Rig.Image.mem i "step")
        with Invalid_argument _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            ignore (Rig.Image.mem i "empty"))
      in
      is_none ~msg:(strf "from allocation %d, the loss" n) (Rig.lost d);
      equal ~msg:(strf "from allocation %d, the function" n) bool true
        (Rig.Image.mem i "step");
      reached);
  greater ~msg:"entries refused" int ~than:0 !refusals

(* An open of the device's GPU under another name from a callback in a
   graph call: it raises Invalid_argument for the driver's refusal, or
   answers that the GPU has a device open, and the device is not lost. *)
let open_in_graph_rig () =
  let refusals = ref 0 in
  walk_rig (fun d n ->
      let i = loaded d in
      let empty = Option.get (Rig.Image.entry i "empty") in
      let cap = Option.get (Rig.capability d Rig_cuda_abi.key) in
      let k = S.kernel empty 0 0 in
      let opened = ref 0 in
      let call () =
        match Rig.open_ { (C.gpu 0) with name = "CUDA:refused" } with
        | Ok d' ->
            incr opened;
            Rig.close d'
        | Error _ -> ()
        | exception Invalid_argument _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            ignore (cap.graph [| k |]))
      in
      equal ~msg:(strf "from allocation %d, the opens in the call" n) int 0
        !opened;
      is_none ~msg:(strf "from allocation %d, the loss" n) (Rig.lost d);
      ignore (Sys.opaque_identity i);
      reached);
  greater ~msg:"opens refused" int ~than:0 !refusals

(* A close from a callback in a graph call: the device's stop runs, at once
   or at a later close, after which a graph call answers that the device was
   stopped. *)
let close_in_graph () =
  let refusals = ref 0 in
  walk_rig (fun d n ->
      let i = loaded d in
      let empty = Option.get (Rig.Image.entry i "empty") in
      let cap = Option.get (Rig.capability d Rig_cuda_abi.key) in
      let k = S.kernel empty 0 0 in
      let call () = try Rig.close d with Invalid_argument _ -> incr refusals in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            ignore (cap.graph [| k |]))
      in
      Rig.close d;
      (match cap.graph [| k |] with
      | Error why ->
          equal ~msg:(strf "from allocation %d, the graph's refusal" n) string
            stopped_graph why
      | Ok _ -> failf "from allocation %d, a graph made after the close" n);
      ignore (Sys.opaque_identity i);
      reached);
  greater ~msg:"closes refused" int ~than:0 !refusals

(* The stops run last: a driver that cannot stop the device from a graph
   call leaves it open for the rest of the process. *)
let () =
  exit
    (run "rig_cuda interrupt"
       [
         group ~timeout "a call from a callback at any allocation"
           [
             test "an entry in an entry of its image answers, or raises Refused"
               entry_in_entry;
             test "a free in a mapping of host memory frees, or raises Refused"
               free_in_map_host;
             test "an open in a graph call claims nothing, or raises Refused"
               make_in_graph;
           ];
         group ~timeout
           "a call to rig from a callback at any allocation keeps the device"
           [
             test "a function asked for while another is" entry_in_entry_rig;
             test "an open in a graph call, refused with Invalid_argument"
               open_in_graph_rig;
           ];
         group ~timeout "a stop from a callback at any allocation of a graph call"
           [
             test "through rig runs, at once or at a later close"
               close_in_graph;
             test "stops the device, or raises Refused" stop_in_graph;
           ];
       ])
