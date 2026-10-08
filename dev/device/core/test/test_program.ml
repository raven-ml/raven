(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Polled loads "code:N": N bytes of code in its memory, its one function "main"
   at their start. *)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module Program = Device_core.Program
module Prof = Device_core.Profile
module P = Device_core_support.Polled
module Support = Device_core_support
module S = Device_dtype.Scalar

let timeout = 60.
let count call p = List.length (List.filter (( = ) call) (P.log p))

let load d binary =
  require_ok ~pp:Format.pp_print_string (Program.load d binary)

let lost d = function C.Lost (d', _) -> C.equal d d' | _ -> false

let test_load () =
  let d, _ = P.open_ "program:load" in
  let p = load d "code:64" in
  equal bool true (C.equal d (Program.device p));
  equal (option int) None (Program.entry p "other");
  let main = require_some (Program.entry p "main") in
  equal ~msg:"the code placed" int 0x6363636363636363 (Support.load main)

let test_refused () =
  let d, _ = P.open_ "program:refused" in
  match Program.load d "not code" with
  | Ok _ -> failf "a refused binary loaded"
  | Error why ->
      equal bool true (String.starts_with ~prefix:"program:refused" why);
      equal int 1
        (C.Point.value (C.submit (Sub.make ~reads:0 ~writes:0 ~waits:0 d [||])))

let test_no_code () =
  raises_match Exn.invalid_arg (fun () -> Program.load C.host "code:64")

(* An unreachable program is unloaded once the work its device was handed until
   then is done, and its code's memory then returns to its device. *)
let test_unload () =
  let d, pd = P.open_ "program:unload" in
  let code =
    (fun () ->
      let p = load d "code:64" in
      ignore (C.submit (Sub.make ~reads:0 ~writes:0 ~waits:0 d [||]));
      require_some (Program.entry p "main"))
      ()
  in
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create ~memory:Pinned d S.UInt8 8)
  in
  drain ();
  equal ~msg:"while its work is unrun" int 0 (count "unload" pd);
  ignore (P.run pd);
  drain ();
  equal ~msg:"once it ran" int 1 (count "unload" pd);
  C.free_cache d;
  equal ~msg:"its code's memory" bool true
    (List.exists (fun (a, _) -> a = code) (P.frees pd))

let test_budget () =
  let d, _ = P.open_ "program:budget" in
  C.set_budget d 4096;
  raises_match
    (function
      | C.Out_of_memory (d', n) -> C.equal d d' && n >= 8192 | _ -> false)
    (fun () -> Program.load d "code:8192")

(* A load whose upload fails loses the device. *)
let test_upload_fails () =
  let d, p = P.open_ "program:upload" in
  P.fail p;
  raises_match (lost d) (fun () -> Program.load d "code:64")

let test_profiled () =
  let d, _ = P.open_ "program:profiled" in
  let p, events = Prof.take (fun () -> load d "code:64") in
  let loads =
    List.filter_map
      (function Prof.Load l -> Some (l.program == p, l.binary) | _ -> None)
      events
  in
  equal (list (pair bool string)) [ (true, "code:64") ] loads

let tests =
  [
    group ~timeout "programs"
      [
        test "a program is loaded on its device with its functions" test_load;
        test "a binary its driver refuses is an error naming the device"
          test_refused;
        test "a device that loads no code refuses a binary" test_no_code;
        test "an unreachable program unloads once its device's work is done"
          test_unload;
        test "code counts in its device's budget" test_budget;
        test "a load whose upload fails loses the device" test_upload_fails;
        test "a load is an event of the profiles taken" test_profiled;
      ];
  ]

let () = exit (run "device_core.program" tests)
