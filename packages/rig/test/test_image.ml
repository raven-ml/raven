(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Polled loads "code:N": N bytes of code in its memory, its one function "main"
   at their start. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Image = Rig.Image
module Prof = Rig.Profile
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers

let timeout = 60.
let count call p = List.length (List.filter (( = ) call) (P.log p))

let load d binary =
  require_ok ~pp:Format.pp_print_string (Image.load d binary)

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

let placement_name (copies, host_visible) =
  Printf.sprintf "copies=%b host_visible=%b" copies host_visible

let test_load (copies, host_visible) =
  let name = "image:load-" ^ placement_name (copies, host_visible) in
  let d, pd = P.open_ ~copies ~host_visible name in
  let previous = submit (Sub.make d [||]) in
  if copies || host_visible then begin
    let p = load d "code:64" in
    at_least ~msg:"previous work completed" int ~than:(Rig.Point.value previous)
      (Rig.signaled d);
    equal ~msg:"no work left queued" int 0 (P.queued pd);
    equal bool true (Rig.equal d (Image.device p));
    equal bool false (Image.mem p "other");
    let main = require_some (Image.entry p "main") in
    equal ~msg:"the code placed" int 0x6363636363636363 (Support.load main)
  end
  else begin
    raises_match Exn.invalid_arg (fun () -> Image.load d "code:64");
    let later = submit (Sub.make d [||]) in
    Rig.Point.wait later;
    equal ~msg:"later work completed" int (Rig.Point.value later)
      (Rig.signaled d);
    equal ~msg:"no work left queued" int 0 (P.queued pd)
  end

let test_refused () =
  let d, _ = P.open_ "image:refused" in
  match Image.load d "not code" with
  | Ok _ -> failf "a refused binary loaded"
  | Error why ->
      equal bool true (String.starts_with ~prefix:"image:refused" why);
      equal int 1 (Rig.Point.value (submit (Sub.make d [||])))

let test_no_code () =
  raises_match Exn.invalid_arg (fun () ->
      Image.load (Rig_support.io "image:no-code") "code:64")

(* An unreachable image is unloaded once the work its device was handed until
   then is done, and its code's memory then returns to its device. *)
let test_unload () =
  let d, pd = P.open_ "image:unload" in
  let code =
    (fun () ->
      let p = load d "code:64" in
      ignore (submit (Sub.make d [||]));
      require_some (Image.entry p "main"))
      ()
  in
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create ~memory:Pinned d 8)
  in
  drain ();
  equal ~msg:"while its work is unrun" int 0 (count "unload" pd);
  ignore (P.run pd);
  drain ();
  equal ~msg:"once it ran" int 1 (count "unload" pd);
  Rig.free_cache d;
  equal ~msg:"its code's memory" bool true
    (List.exists (fun (a, _) -> a = code) (P.frees pd))

(* An entry keeps its image loaded until it returns: a collection and a drain at
   the start of the driver's entry call, the image's last use, unload
   nothing. *)
let test_entry_keeps () =
  let d, pd = P.open_ "image:entry-keeps" in
  let unloads = ref (-1) in
  P.before pd "entry" (fun () ->
      Gc.full_major ();
      Gc.full_major ();
      ignore (B.create ~memory:Pinned d 8);
      unloads := count "unload" pd);
  ignore (Image.mem (load d "code:64") "main");
  equal ~msg:"unloads during the entry" int 0 !unloads

let test_entry_lost () =
  let d, p = P.open_ "image:entry-lost" in
  let i = load d "code:64" in
  P.fail p;
  raises_match (lost d) (fun () -> submit (Sub.make d [||]));
  raises_match (lost d) (fun () -> Image.mem i "main")

(* A function the device has no memory for is asked for again after each round
   of reclamation, and raises Out_of_memory for the bytes its launches need
   after the last. *)
let test_entry_no_memory () =
  let d, pd = P.open_ "image:entry-no-memory" in
  let i = load d "code:64" in
  P.fail_at pd 1 (`Refuse max_int);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = P.entry_scratch
      | _ -> false)
    (fun () -> Image.mem i "main");
  P.fail_at pd 1 (`Refuse 0);
  at_least ~msg:"times asked" int ~than:2 (count "entry" pd)

(* A function the device had no memory for at first is found once a round of
   reclamation is done. *)
let test_entry_reclaims () =
  let d, pd = P.open_ "image:entry-reclaims" in
  let i = load d "code:64" in
  P.before pd "entry" (fun () -> P.fail_at pd 1 (`Refuse 1));
  equal bool true (Image.mem i "main");
  equal ~msg:"times asked" int 2 (count "entry" pd)

let test_budget () =
  let d, _ = P.open_ "image:budget" in
  Rig.set_budget d 4096;
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n >= 8192 | _ -> false)
    (fun () -> Image.load d "code:8192")

(* A load whose upload fails loses the device. *)
let test_upload_fails () =
  let d, p = P.open_ "image:upload" in
  P.fail p;
  raises_match (lost d) (fun () -> Image.load d "code:64")

(* A load whose upload loses the device gives its code memory back only once the
   device counts as stopped: its work may still run until then. *)
let test_upload_frees_after_stop () =
  let d, p = P.open_ ~answer:`Unknown "image:upload-unknown" in
  P.fail p;
  raises_match (lost d) (fun () -> Image.load d "code:64");
  let frees () = List.length (P.frees p) in
  ignore (B.create Rig.host 8);
  equal ~msg:"before the word drained" int 0 (frees ());
  P.set_word p (Rig.submitted d);
  ignore (B.create Rig.host 8);
  equal ~msg:"once it did" int 1 (frees ())

(* Collects, then drains as an allocation does. *)
let collect () =
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create Rig.host 8)

let rec after_stop = function
  | "stop" :: rest -> rest
  | _ :: rest -> after_stop rest
  | [] -> []

(* An image still loaded when its device stops is unloaded after the stop, once
   unreachable: the stop releases nothing this library made. *)
let test_unload_after_stop () =
  let d, p = P.open_ "image:unload-after-stop" in
  let i = ref (Some (load d "code:64")) in
  let code = require_some (Image.entry (Option.get !i) "main") in
  P.fail p;
  raises_match (lost d) (fun () -> submit (Sub.make d [||]));
  equal ~msg:"by the stop" int 0 (count "unload" p);
  raises_match (lost d) (fun () -> Image.mem (Option.get !i) "main");
  i := None;
  collect ();
  equal ~msg:"after the stop" (list string) [ "unload" ]
    (List.filter (( = ) "unload") (after_stop (P.log p)));
  equal ~msg:"its code's memory" bool true
    (List.exists (fun (a, _) -> a = code) (P.frees p))

(* A lost device whose work may still run unloads nothing until its word reads
   its last value. *)
let test_unload_after_unknown () =
  let d, p = P.open_ ~answer:`Unknown "image:unload-unknown" in
  let i = ref (Some (load d "code:64")) in
  ignore (submit (Sub.make d [||]));
  P.fail p;
  raises_match (lost d) (fun () -> submit (Sub.make d [||]));
  i := None;
  collect ();
  equal ~msg:"with the word short" int 0 (count "unload" p);
  P.set_word p (Rig.submitted d);
  collect ();
  equal ~msg:"once the word drained" int 1 (count "unload" p)

let drain d = ignore (Sys.opaque_identity (B.create d 0))

let collect_on d =
  for _ = 1 to 3 do
    Gc.full_major ();
    drain d
  done

(* Code can follow an image with no code in the release queue. Pressure must
   find those bytes and finish the work that holds them before allocating. *)
let test_pending_code_pressure () =
  let d, p = P.open_ ~budget:4096 ~copies:false "image:pending-code-pressure" in
  let first = ref (Some (load d "functions")) in
  let second = ref (Some (load d "code:4096")) in
  ignore (submit (Sub.make d [||]));
  first := None;
  collect_on d;
  second := None;
  collect_on d;
  equal ~msg:"both images wait" int 0 (count "unload" p);
  let b = B.create d 4096 in
  equal ~msg:"the replacement fits" int 4096 (B.length b);
  equal ~msg:"the work completed" int 0 (P.queued p);
  equal ~msg:"both images unloaded" int 2 (count "unload" p)

(* A native unload holds no release lock. A concurrent collection can append
   another image for a later value while that unload is still running. *)
let test_append_during_unload () =
  let d, p = P.open_ "image:append-during-unload" in
  let first = ref (Some (load d "functions")) in
  let second = ref (Some (load d "functions")) in
  ignore (submit (Sub.make d [||]));
  first := None;
  collect_on d;
  ignore (P.run p);
  let entered = Atomic.make false and finish = Atomic.make false in
  P.before p "unload" (fun () ->
      Atomic.set entered true;
      Support.await "the concurrent append" (fun () -> Atomic.get finish));
  let worker = Domain.spawn (fun () -> drain d) in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set finish true;
      Domain.join worker)
    (fun () ->
      Support.await "the first unload" (fun () -> Atomic.get entered);
      ignore (submit (Sub.make d [||]));
      second := None;
      collect_on d;
      equal ~msg:"only the first image unloads" int 1 (count "unload" p);
      equal ~msg:"later work stays queued" int 1 (P.queued p));
  ignore (P.run p);
  drain d;
  equal ~msg:"the later image unloads once" int 2 (count "unload" p)

(* The first unload discovers a fault after newer work was handed over. Other
   releases already taken from the queue must wait again for that work. *)
let test_unloads_beside_loss () =
  let d, p = P.open_ ~answer:`Unknown "image:unloads-beside-loss" in
  let images = Array.init 2 (fun _ -> ref (Some (load d "functions"))) in
  ignore (submit (Sub.make d [||]));
  Array.iter (fun image -> image := None) images;
  collect_on d;
  ignore (P.run p);
  P.before p "unload" (fun () ->
      ignore (submit (Sub.make d [||]));
      P.fault p "the engine hung");
  drain d;
  equal ~msg:"the fault is recorded" (option string) (Some "the engine hung")
    (Rig.lost d);
  equal ~msg:"later unloads wait" int 1 (count "unload" p);
  collect ();
  equal ~msg:"the unknown stop keeps them" int 1 (count "unload" p);
  P.set_word p (Rig.submitted d);
  collect ();
  equal ~msg:"the final release retries once stopped" int 2 (count "unload" p)

(* Two domains drop images and drain their device while its work waits to run:
   no drain unloads an image.

   Every collection stops each domain, and on a loaded machine waits for each
   to be scheduled, and each program runs 50 times, so the test collects
   little. A program's images stay on its own device, which no later program
   uses, so a release collects nothing. A drop of a live image ends one major
   cycle, which may finalise it while the other domain drains. *)
type images = {
  d : Rig.t;
  p : P.t;
  cells : Image.t option Atomic.t array;
  unloaded : int;
}

let[@inline never] drop_image t i =
  Option.is_some (Atomic.exchange t.cells.(i) None)

let image_opened = ref 0
let image_count = 2

(* Loads [image_count] images on a fresh device whose submitted work waits to
   run, then drops the first: every later drain is one after a drop, whichever
   calls a program draws. *)
let make_images () =
  incr image_opened;
  let d, p =
    P.open_ (Printf.sprintf "image:parallel-releases-%d" !image_opened)
  in
  let cells =
    Array.init image_count (fun _ -> Atomic.make (Some (load d "functions")))
  in
  ignore (submit (Sub.make d [||]));
  let t = { d; p; cells; unloaded = count "unload" p } in
  ignore (drop_image t 0 : bool);
  t

(* The images unload once a later collection finalises them; none twice. *)
let release_images t =
  Array.iter (fun cell -> Atomic.set cell None) t.cells;
  ignore (P.run t.p);
  at_most ~msg:"each image unloads at most once" int ~than:image_count
    (count "unload" t.p - t.unloaded)

let images = abstract ~release:release_images "images"

let image_commands =
  [
    command "load"
      (Gen.unit @-> makes images)
      (fun () -> Array.init image_count (fun i -> i > 0))
      make_images;
    command "drop"
      (images ^-> Gen.int_range 0 (image_count - 1) @-> returns bool)
      (fun cells i ->
        let was_live = cells.(i) in
        cover "a live image dropped" was_live;
        cells.(i) <- false;
        was_live)
      (fun t i ->
        let was_live = drop_image t i in
        if was_live then Gc.major ();
        drain t.d;
        was_live);
    command "drain"
      (images ^-> returns int)
      (fun cells ->
        cover "a drain after a drop" (Array.exists not cells);
        0)
      (fun t ->
        drain t.d;
        count "unload" t.p - t.unloaded);
  ]

let test_profiled () =
  let d, _ = P.open_ "image:profiled" in
  let p, events = Prof.take (fun () -> load d "code:64") in
  let loads =
    List.filter_map
      (function Prof.Load l -> Some (l.image == p, l.binary) | _ -> None)
      events
  in
  equal (list (pair bool string)) [ (true, "code:64") ] loads

let tests =
  [
    group ~timeout "images"
      [
        cases ~name:placement_name "code placement"
          [ (true, true); (true, false); (false, true); (false, false) ]
          test_load;
        test "a lost device's image names no function" test_entry_lost;
        test "an entry keeps its image loaded until it returns" test_entry_keeps;
        test "an entry with no memory raises Out_of_memory after reclaiming"
          test_entry_no_memory;
        test "an entry with no memory is found once reclaiming frees some"
          test_entry_reclaims;
        test "a binary its driver refuses is an error naming the device"
          test_refused;
        test "an io device refuses a binary" test_no_code;
        test "an unreachable image unloads once its device's work is done"
          test_unload;
        test "code counts in its device's budget" test_budget;
        test "a load whose upload fails loses the device" test_upload_fails;
        test "a failed upload frees its code once the device counts as stopped"
          test_upload_frees_after_stop;
        test "a lost device's image is unloaded after its stop"
          test_unload_after_stop;
        test "a lost device's image waits for its word to drain"
          test_unload_after_unknown;
        test "allocation pressure waits for queued code memory"
          test_pending_code_pressure;
        test "a release can be appended while an image unloads"
          test_append_during_unload;
        test "releases taken before loss wait again for an unknown stop"
          test_unloads_beside_loss;
        stateful ~count:10 ~domains:2
          "parallel drops and drains keep images until work completes"
          image_commands;
        test "a load is an event of the profiles taken" test_profiled;
      ];
  ]

let () = exit (run "rig.image" tests)
