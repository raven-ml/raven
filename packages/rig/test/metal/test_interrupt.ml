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
   call that holds the device's guard raises rather than waits. Through
   rig, a call from a callback inside an icb or an entry keeps the device:
   it works, or is refused and leaves its work to a later call. *)

open Windtrap
module S = Rig_metal_support
module M = Rig_metal

let strf = Printf.sprintf
let timeout = 60.

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
   thread. The free frees the region, or raises [Rig_metal.Refused] where
   [call] holds the device's guard and leaves the region to a later free.
   Some free raises. *)
let reentry call () =
  let refusals = ref 0 in
  walk (fun g n ->
      let i = image g in
      let step = (found (M.entry i "step")).code in
      let args = alloc g args_bytes and other = alloc g args_bytes in
      let held = S.buffers () in
      let freed = ref false in
      let free () =
        match M.free g other with
        | () -> freed := true
        | exception M.Refused _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call free) n (fun () ->
            call g args step)
      in
      let gone = if !freed then 1 else 0 in
      equal ~msg:(strf "from allocation %d, buffers held" n) int (held - gone)
        (S.buffers ());
      if not !freed then M.free g other;
      M.free g args;
      M.unload g i;
      reached);
  greater ~msg:"frees refused" int ~than:0 !refusals

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

(* [reentered ~ready ~during ~call ~after ()] walks [during d x] on a device
   [d] that rig opened, [x] what [ready d] made, with [call d x] at its
   [n]-th allocation from a [Gc.Memprof] callback on the calling thread.
   [call] works, or the driver refuses it where [during] holds a guard of
   the driver's: [call] is then [true], or raises [Invalid_argument]. Either
   way the device is not lost, and [after d x n] finds [call]'s work done or
   does it. Some call is refused. *)
let reentered ~ready ~during ~call ~after () =
  let refusals = ref 0 in
  walk_rig (fun d n ->
      let x = ready d in
      let call () =
        match call d x with
        | true -> incr refusals
        | false -> ()
        | exception Invalid_argument _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () -> during d x)
      in
      is_none ~msg:(strf "from allocation %d, the loss" n) (Rig.lost d);
      after d x n;
      reached);
  greater ~msg:"calls refused" int ~than:0 !refusals

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
    ~call:(fun d (_, held) ->
      Rig.free_cache d;
      Rig.allocated d > held - cached_bytes)
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
    ~call:(fun d (_, w) ->
      Rig.free_cache d;
      S.alive w)
    ~after:(fun d (_, w) n ->
      Rig.free_cache d;
      equal ~msg:(strf "from allocation %d, the pipeline lives" n) bool false
        (S.alive w))

(* A function of an image asked for from a callback while another function
   of it is asked for: it is made, at once or when asked for again. *)
let entry_in_entry =
  reentered ~ready:loaded
    ~during:(fun _ i -> ignore (Rig.Image.mem i "fill"))
    ~call:(fun _ i -> not (Rig.Image.mem i "step"))
    ~after:(fun _ i n ->
      equal ~msg:(strf "from allocation %d, the function" n) bool true
        (Rig.Image.mem i "step"))

(* A close from a callback in an icb: the device's stop runs, at once or at
   a later close, after which an icb answers that the device was stopped. *)
let close_in_icb () =
  let refusals = ref 0 in
  walk_rig (fun d n ->
      let x = ready_icb d in
      let call () = try Rig.close d with Invalid_argument _ -> incr refusals in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () -> rig_icb x)
      in
      Rig.close d;
      (match x.cap.icb x.args [| dispatch x.step |] with
      | Error why ->
          equal ~msg:(strf "from allocation %d, the icb's refusal" n) string
            "the device was stopped" why
      | Ok _ -> failf "from allocation %d, an icb recorded after the close" n);
      reached);
  greater ~msg:"closes refused" int ~than:0 !refusals

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
             test "of an icb frees, or raises Refused" (reentry icb);
             test "of an allocation frees, or raises Refused"
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
