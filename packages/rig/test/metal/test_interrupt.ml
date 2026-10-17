(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of Metal's device, driven by the driver, then through rig.
   A call that changes what an icb checks, an entry's pipeline or a region,
   is made once for each allocation it makes, with every allocation from
   that one on raising until the call returns, as an asynchronous exception
   does, each time on a device of its own. An entry or a region is made on
   a device that already holds many, so the call adds to a deep table.
   Every pipeline and region made before the call is still one an icb
   records, the call's pipeline or buffer is held only if the call kept it,
   and the device stops. A free from a callback at each allocation of a
   call raises Invalid_argument, having changed nothing, inside the
   device's guard, where it would take the guard again, and frees between
   locks. Through rig, a call from a callback inside an icb or an entry
   raises Invalid_argument inside a lock, having changed nothing, and works
   between locks; the device is not lost. *)

open Windtrap
module S = Rig_metal_support
module M = Rig_metal

let strf = Printf.sprintf
let timeout = 60.

(* The text of the driver's refusal of a take by a lock's holder. *)
let retaken = "Rig_metal: a lock taken again by the thread that holds it"

let found : Rig_edge.entry_answer -> Rig_edge.entry = function
  | Entry e -> e
  | Absent -> fail "no such function"
  | No_memory n -> failf "no memory for %d bytes" n

let image g =
  match M.image g (S.fixture ~dir:"fixtures" "fill") with
  | Ok (Rig_edge.Loaded i) -> i
  | Ok (Place (n, _)) -> failf "the device asked to place %d bytes of code" n
  | Error why -> fail why

let alloc g n =
  match M.alloc g Rig_edge.Device n with
  | Some r -> r
  | None -> failf "no memory for %d bytes" n

(* The bytes of the fill kernels' arguments. *)
let args_bytes = 16

let dispatch p =
  {
    Rig_metal_abi.pipeline = p;
    offset = 0;
    groups = (1, 1, 1);
    threads = (1, 1, 1);
  }

(* Fails unless an icb of [g] records a dispatch of the pipeline [p] with the
   argument buffer [r]. *)
let records g what r p =
  match (M.capability g).icb (M.locate r).handle [| dispatch p |] with
  | Ok _ -> ()
  | Error why -> failf "%s: %s" what why
  | exception Invalid_argument why -> failf "%s: %s" what why

(* [walk attempt] runs [attempt n], interrupted from the [n]-th allocation, on
   a device of its own, from [n = 1] until an attempt no longer reaches its
   [n]-th. *)
let walk attempt =
  if not (S.present ()) then
    skip ~reason:"the machine has no METAL GPU rig drives" ();
  let rec from n =
    let g = S.driver () in
    let reached =
      Fun.protect ~finally:(fun () -> S.stop_driver g) (fun () -> attempt g n)
    in
    if reached then from (n + 1)
  in
  from 1

(* The pipelines on the device before the walked entry. A table that grew in
   place, as a stdlib Hashtbl of its first size does, would grow at the
   33rd. *)
let pipelines = 32

(* The 33rd entry's pipeline, each of the first 32 from an image of its
   own. Once every image is unloaded, the device holds none of their
   pipelines, the 33rd's included, whether the entry returned or raised. *)
let test_entries () =
  walk @@ fun g n ->
  let held = S.pipelines () in
  let args = alloc g args_bytes in
  let made = List.init pipelines (fun _ -> image g) and last = image g in
  let codes = List.map (fun i -> (found (M.entry i "step")).code) made in
  let reached =
    Rig_support.interrupted n (fun () -> ignore (M.entry last "step"))
  in
  let what k = strf "from allocation %d, pipeline %d" n k in
  List.iteri (fun k p -> records g (what k) args p) codes;
  List.iter (M.unload g) (last :: made);
  equal ~msg:(strf "from allocation %d, pipelines held" n) int held
    (S.pipelines ());
  M.free g args;
  reached

(* The regions on the device before the walked allocation. A table that grew
   in place, as a stdlib Hashtbl of its first size does, would grow at the
   129th. *)
let regions = 128

(* The 129th region. An allocation that raised holds no buffer. *)
let test_regions () =
  walk @@ fun g n ->
  let i = image g in
  let step = (found (M.entry i "step")).code in
  let made = List.init regions (fun _ -> alloc g args_bytes) in
  let last = ref None in
  let held = S.buffers () in
  let reached =
    Rig_support.interrupted n (fun () ->
        last := M.alloc g Rig_edge.Device args_bytes)
  in
  if Option.is_none !last then
    equal ~msg:(strf "from allocation %d, buffers held" n) int held
      (S.buffers ());
  let what k = strf "from allocation %d, region %d" n k in
  List.iteri (fun k r -> records g (what k) r step) made;
  Option.iter (fun r -> records g (what regions) r step) !last;
  List.iter (M.free g) made;
  Option.iter (M.free g) !last;
  M.unload g i;
  reached

(* [reentry call] runs [call g args step], on a device [g] with the region
   [args] and the pipeline [step], with a free of another region of [g] at
   its [n]-th allocation, from a [Gc.Memprof] callback on the calling
   thread. Inside the device's guard the free is refused and leaves the
   region to a later free; between locks it frees the region. *)
let reentry call () =
  let seen = Rig_support.seen () in
  walk (fun g n ->
      let i = image g in
      let step = (found (M.entry i "step")).code in
      let args = alloc g args_bytes and other = alloc g args_bytes in
      let held = S.buffers () in
      let landing, landed =
        Rig_support.landing seen retaken (fun () -> M.free g other)
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () -> call g args step)
      in
      let what = strf "from allocation %d" n in
      let freed = Rig_support.answered what landed = Some false in
      let gone = if freed then 1 else 0 in
      equal ~msg:(what ^ ", buffers held") int (held - gone) (S.buffers ());
      if not freed then M.free g other;
      M.free g args;
      M.unload g i;
      reached);
  Rig_support.covered seen

let icb g args step =
  match (M.capability g).icb (M.locate args).handle [| dispatch step |] with
  | Ok _ | Error _ -> ()

let region g _ _ = M.free g (alloc g args_bytes)

(* Through rig *)

let fill () = S.fixture ~dir:"fixtures" "fill"

(* [walk_rig attempt] is {!walk} on a device that rig opened. *)
let walk_rig attempt =
  if not (S.present ()) then
    skip ~reason:"the machine has no METAL GPU rig drives" ();
  let rec from n =
    let t = S.open_ () in
    let reached =
      Fun.protect ~finally:(fun () -> S.close t) (fun () -> attempt t.S.d n)
    in
    if reached then from (n + 1)
  in
  from 1

(* [reentered ~ready ~during ~refusal ~call ~did ~after ()] walks
   [during d x] on a device [d] that rig opened, [x] what [ready d] made,
   with [call d x] at its [n]-th allocation from a [Gc.Memprof] callback on
   the calling thread. Inside a lock of the thread's, [call] raises
   [Invalid_argument] whose text starts with [refusal], and [did d x] is
   then [false]; between locks it returns, and [did d x] is [true]. Either
   way the device is not lost, and [after d x n] does [call]'s work if it
   was refused and checks it. *)
let reentered ~ready ~during ~refusal ~call ~did ~after () =
  let seen = Rig_support.seen () in
  walk_rig (fun d n ->
      let x = ready d in
      let landing, landed =
        Rig_support.landing seen refusal (fun () -> call d x)
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () -> during d x)
      in
      let what = strf "from allocation %d" n in
      (match Rig_support.answered what landed with
      | None -> ()
      | Some held ->
          equal ~msg:(what ^ ", the call's work done") bool (not held)
            (did d x));
      is_none ~msg:(what ^ ", the loss") (Rig.lost d);
      after d x n;
      reached);
  Rig_support.covered seen

let loaded d =
  match Rig.Image.load d (fill ()) with Ok i -> i | Error why -> fail why

(* An icb of a device: [cap] its record, [args] the handle of its argument
   buffer [buffer], [step] a pipeline of [image]. *)
type icb = {
  cap : Rig_metal_abi.t;
  args : nativeint;
  buffer : Rig.Buffer.t;
  step : int;
  image : Rig.Image.t;
}

let ready_icb d =
  let image = loaded d in
  let step = Option.get (Rig.Image.entry image "step") in
  let buffer = Rig.Buffer.create d args_bytes in
  let cap = Option.get (Rig.capability d Rig_metal_abi.key) in
  { cap; args = Rig.Buffer.handle buffer; buffer; step; image }

(* An icb of a dispatch of [x.step], whose answer is dropped. *)
let rig_icb x =
  (match x.cap.icb x.args [| dispatch x.step |] with Ok _ | Error _ -> ());
  ignore (Sys.opaque_identity (x.buffer, x.image))

let collect () =
  Gc.full_major ();
  Gc.full_major ()

(* The bytes of the memory a test drops into the cache. *)
let cached_bytes = 4096

(* A buffer collected and moved into the device's cache: the allocation of
   the icb's arguments drains the device. The device then holds [held]
   bytes. *)
let ready_cached d =
  ignore (Rig.Buffer.create d cached_bytes);
  collect ();
  let x = ready_icb d in
  (x, Rig.allocated d)

(* A free of the cache from a callback in an icb: the cached buffer goes
   back to Metal, at once or at a later free. *)
let free_in_icb =
  reentered ~ready:ready_cached
    ~during:(fun _ (x, _) -> rig_icb x)
    ~refusal:"Rig.free_cache: "
    ~call:(fun d _ -> Rig.free_cache d)
    ~did:(fun d (_, held) -> Rig.allocated d = held - cached_bytes)
    ~after:(fun d (_, held) n ->
      Rig.free_cache d;
      equal ~msg:(strf "from allocation %d, bytes held" n) int
        (held - cached_bytes) (Rig.allocated d))

(* A weak reference to the pipeline [fill] of an image that is dropped,
   which lives while the image is loaded. *)
let[@inline never] dropped_pipeline d =
  let i = loaded d in
  S.weak (Nativeint.of_int (Option.get (Rig.Image.entry i "fill")))

(* An image collected, and an icb's arguments. *)
let ready_collected d =
  let x = ready_icb d in
  let w = dropped_pipeline d in
  collect ();
  (x, w)

(* A free of the cache from a callback in an icb, after an image was
   collected: the image unloads, at once or at a later free. *)
let unload_in_icb =
  reentered ~ready:ready_collected
    ~during:(fun _ (x, _) -> rig_icb x)
    ~refusal:"Rig.free_cache: "
    ~call:(fun d _ -> Rig.free_cache d)
    ~did:(fun _ (_, w) -> not (S.alive w))
    ~after:(fun d (_, w) n ->
      Rig.free_cache d;
      equal ~msg:(strf "from allocation %d, the pipeline lives" n) bool false
        (S.alive w))

(* A function of an image asked for from a callback while another function
   of it is asked for: it is made, at once or when asked for again. The
   device's pipelines are [fill]'s and, once made, [step]'s. *)
let entry_in_entry =
  reentered
    ~ready:(fun d -> (loaded d, S.pipelines ()))
    ~during:(fun _ (i, _) -> ignore (Rig.Image.mem i "fill"))
    ~refusal:"Rig.Image.mem: "
    ~call:(fun _ (i, _) ->
      if not (Rig.Image.mem i "step") then fail "no function step")
    ~did:(fun _ (_, before) -> S.pipelines () = before + 2)
    ~after:(fun _ (i, _) n ->
      equal ~msg:(strf "from allocation %d, the function" n) bool true
        (Rig.Image.mem i "step"))

(* A close from a callback in an icb: inside a lock it is refused and an icb
   still records; between locks it stops the device. After a close, an icb
   answers that the device was stopped. *)
let close_in_icb () =
  let seen = Rig_support.seen () in
  walk_rig (fun d n ->
      let x = ready_icb d in
      let landing, landed =
        Rig_support.landing seen "Rig.close: " (fun () -> Rig.close d)
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () -> rig_icb x)
      in
      let what = strf "from allocation %d" n in
      let stopped = "the device was stopped" in
      let answered = Rig_support.answered what landed in
      (match (answered, x.cap.icb x.args [| dispatch x.step |]) with
      | Some false, Error why ->
          equal ~msg:(what ^ ", the icb's refusal") string stopped why
      | Some false, Ok _ -> failf "%s, an icb recorded after the close" what
      | (Some true | None), Ok _ -> ()
      | (Some true | None), Error why -> failf "%s, an icb: %s" what why);
      Rig.close d;
      (match x.cap.icb x.args [| dispatch x.step |] with
      | Error why ->
          equal ~msg:(what ^ ", the icb's refusal after the close") string
            stopped why
      | Ok _ -> failf "%s, an icb recorded after the close" what);
      reached);
  Rig_support.covered seen

let () =
  exit
    (run "rig_metal interrupt"
       [
         group ~timeout "an exception at any allocation"
           [
             test
               "of a pipeline leaves every earlier one recordable, and the \
                pipeline held only by its image"
               test_entries;
             test
               "of a region leaves every earlier one recordable, and its \
                buffer freed if it raised"
               test_regions;
           ];
         group ~timeout "a free from a callback at any allocation"
           [
             test "of an icb, refused inside the device's guard" (reentry icb);
             test "of an allocation, refused inside the device's guard"
               (reentry region);
           ];
         group ~timeout
           "a call to rig from a callback at any allocation keeps the device"
           [
             test "a free of the cache in an icb" free_in_icb;
             test "a free of the cache in an icb, unloading an image"
               unload_in_icb;
             test "a function asked for while another is" entry_in_entry;
             test "a close in an icb" close_in_icb;
           ];
       ])
