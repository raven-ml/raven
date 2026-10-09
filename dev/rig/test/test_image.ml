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

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s =
  Rig.submit s ~run:(Sub.Run.make ()) ~reads ~writes ~waits

let timeout = 60.
let count call p = List.length (List.filter (( = ) call) (P.log p))

let load d binary =
  require_ok ~pp:Format.pp_print_string (Image.load d binary)

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

let test_load () =
  let d, _ = P.open_ "image:load" in
  let p = load d "code:64" in
  equal bool true (Rig.equal d (Image.device p));
  equal (option int) None (Image.entry p "other");
  let main = require_some (Image.entry p "main") in
  equal ~msg:"the code placed" int 0x6363636363636363 (Support.load main)

let test_refused () =
  let d, _ = P.open_ "image:refused" in
  match Image.load d "not code" with
  | Ok _ -> failf "a refused binary loaded"
  | Error why ->
      equal bool true (String.starts_with ~prefix:"image:refused" why);
      equal int 1
        (Rig.Point.value (submit (Sub.make ~reads:0 ~writes:0 d [||])))

let test_no_code () =
  raises_match Exn.invalid_arg (fun () -> Image.load Rig.host "code:64")

(* An unreachable image is unloaded once the work its device was handed until
   then is done, and its code's memory then returns to its device. *)
let test_unload () =
  let d, pd = P.open_ "image:unload" in
  let code =
    (fun () ->
      let p = load d "code:64" in
      ignore (submit (Sub.make ~reads:0 ~writes:0 d [||]));
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

let test_entry_lost () =
  let d, p = P.open_ "image:entry-lost" in
  let i = load d "code:64" in
  P.fail p;
  raises_match (lost d) (fun () -> submit (Sub.make ~reads:0 ~writes:0 d [||]));
  raises_match (lost d) (fun () -> Image.entry i "main")

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
  raises_match (lost d) (fun () -> submit (Sub.make ~reads:0 ~writes:0 d [||]));
  equal ~msg:"by the stop" int 0 (count "unload" p);
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
  ignore (submit (Sub.make ~reads:0 ~writes:0 d [||]));
  P.fail p;
  raises_match (lost d) (fun () -> submit (Sub.make ~reads:0 ~writes:0 d [||]));
  i := None;
  collect ();
  equal ~msg:"with the word short" int 0 (count "unload" p);
  P.set_word p (Rig.submitted d);
  collect ();
  equal ~msg:"once the word drained" int 1 (count "unload" p)

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
        test "an image is loaded on its device with its functions" test_load;
        test "a lost device's image names no function" test_entry_lost;
        test "a binary its driver refuses is an error naming the device"
          test_refused;
        test "a device that loads no code refuses a binary" test_no_code;
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
        test "a load is an event of the profiles taken" test_profiled;
      ];
  ]

let () = exit (run "rig.image" tests)
