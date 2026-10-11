(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The limit on opens counts every call of an opener in the process, so this
   suite opens nothing else. *)

open Windtrap

(* The opener calls a process may make: device indices run from 1 to
   65,535. *)
let most = 65_535

(* Closing a device gives back no index: opens past [most] fail however few
   devices are open. *)
let test_limit () =
  for i = 1 to most do
    Rig.close (Rig.memory_device (Printf.sprintf "limit:%d" i))
  done;
  raises_match
    (Exn.failure ~substring:"opened 65535 devices")
    (fun () -> Rig.memory_device "limit:over")

(* An open and a close cost the same however many devices the process
   made, so the limit is reached well within the timeout. *)
let tests =
  [
    test ~timeout:(Rig_support.time 10.)
      "the 65,536th open fails, past closed devices"
      test_limit;
  ]

let () = exit (run "rig.limit" tests)
