(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of CUDA's device. A call is made once for each allocation
   it makes, with another call of the driver from a Gc.Memprof callback at
   that allocation, on the calling thread. Inside a lock of the driver or an
   image's lock, the callback's call raises Invalid_argument, having changed
   nothing: an entry point of the driver's
   checks the thread's locks as it starts, and a function of the driver's
   signature that takes the held lock again is refused at the take. A free
   of page-locked memory waits for no lock, and returns inside the lock of
   the registry of page-locked memory too. Between locks the call runs as
   any call does. Through rig, a call from such a callback raises Invalid_argument
   inside a lock and runs between locks, and the device is not lost. Other
   walks end the call with an exception at each allocation, as an
   asynchronous exception does, and check that the driver's state outlasts
   it whole. *)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let maps = Rig_gpu_support.mapped

let strf = Printf.sprintf
let timeout = 120.

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
   memory returns, inside the registry's lock and between locks, and the
   memory is unlocked once the mapping returns: inside the lock, the free
   leaves the unlocking to the mapping. *)
let free_in_map_host () =
  let seen = Rig_support.seen () in
  walk (fun g n ->
      S.bind g;
      let p = H.pages H.page and q = H.pages H.page in
      let r = require_some (maps (C.map_host g p 64)) in
      let landing, landed =
        Rig_support.landing seen retaken (fun () -> C.free g r)
      in
      (* The answer as the driver gave it: nothing allocates between its
         return and the store. *)
      let mapped = ref Rig_edge.Never in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            mapped := C.map_host g q 64)
      in
      let what = strf "from allocation %d" n in
      let freed = Option.is_some (Rig_support.returned what landed) in
      equal ~msg:(what ^ ", the first memory locked") bool (not freed)
        (S.locked p);
      if not freed then C.free g r;
      Option.iter (C.free g) (maps !mapped);
      equal ~msg:(what ^ ", locked after the frees") bool false
        (S.locked p || S.locked q);
      S.stop_driver g;
      H.free_pages p H.page;
      H.free_pages q H.page;
      reached);
  Rig_support.covered seen

(* An exception at any allocation *)

(* [cut attempt] runs [attempt n] from [n = 1] until it no longer reaches
   its [n]-th allocation. *)
let rec cut ?(n = 1) attempt = if attempt n then cut ~n:(n + 1) attempt

(* An open that an exception ends at any allocation leaves the GPU to the
   next open. *)
let open_cut () =
  S.with_driver ignore;
  let opener = C.gpu 0 and word = H.word () in
  cut (fun n ->
      let made = ref (Error "not made") in
      let reached =
        Rig_support.interrupted n (fun () -> made := opener.make ~word)
      in
      Result.iter (fun g -> C.stop g ~fault:None) !made;
      (match opener.make ~word with
      | Ok g -> C.stop g ~fault:None
      | Error why -> failf "from allocation %d, the open after: %s" n why);
      reached)

(* A mapping of host memory that an exception ends at any allocation leaves
   the memory as it was: it maps again, and it is unlocked once the regions
   over it are freed. Where [over] holds, the memory was mapped before, so
   the mapping counts against that one's lock. *)
let map_host_cut ~over () =
  S.with_driver @@ fun g ->
  S.bind g;
  cut (fun n ->
      let p = H.pages H.page in
      let before =
        if over then maps (C.map_host g p H.page) else None
      in
      if over then is_some ~msg:"the mapping before" before;
      let mapped = ref Rig_edge.Never in
      let reached =
        Rig_support.interrupted n (fun () -> mapped := C.map_host g p 64)
      in
      let what = strf "from allocation %d" n in
      let again = maps (C.map_host g p 64) in
      is_some ~msg:(what ^ ", a mapping after") again;
      List.iter (Option.iter (C.free g)) [ again; maps !mapped; before ];
      equal ~msg:(what ^ ", locked after the frees") bool false (S.locked p);
      H.free_pages p H.page;
      reached)

(* A free of page-locked memory reaches no allocation, so no exception
   lands in it: rig never frees a region again after its free raised. Its
   memory is unlocked after the last free over it. *)
let free_uncut () =
  S.with_driver @@ fun g ->
  S.bind g;
  let p = H.pages H.page and q = H.pages H.page in
  let first =
    require_some ~msg:"a mapping" (maps (C.map_host g p H.page))
  in
  let inner =
    require_some ~msg:"a mapping inside it" (maps (C.map_host g p 64))
  in
  S.register q H.page;
  let foreign =
    require_some ~msg:"a mapping of another owner's memory"
      (maps (C.map_host g q 64))
  in
  let free what r =
    let reached = Rig_support.interrupted 1 (fun () -> C.free g r) in
    equal ~msg:(what ^ ", an allocation reached") bool false reached
  in
  free "a region that counts" inner;
  equal ~msg:"locked after a free that counts" bool true (S.locked p);
  free "the last region" first;
  equal ~msg:"locked after the last free" bool false (S.locked p);
  free "another owner's region" foreign;
  S.unregister q;
  H.free_pages p H.page;
  H.free_pages q H.page

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

let () =
  exit
    (run "rig_cuda interrupt"
       [
         group ~timeout "an exception at any allocation"
           [
             test "an open leaves the GPU to the next open" open_cut;
             test "a mapping of host memory leaves it unlocked"
               (map_host_cut ~over:false);
             test "a mapping inside a locked range leaves it unlocked"
               (map_host_cut ~over:true);
             test "a free of page-locked memory reaches no allocation"
               free_uncut;
           ];
         group ~timeout "a call from a callback at any allocation"
           [
             test "an entry in an entry of its image, refused inside its lock"
               entry_in_entry;
             test "a free in a mapping of host memory returns" free_in_map_host;
           ];
         group ~timeout
           "a call to rig from a callback at any allocation keeps the device"
           [
             test "a function asked for while another is" entry_in_entry_rig;
           ];
       ])
