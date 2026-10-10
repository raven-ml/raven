(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Points of devices from index 32768 on, whose points set the top bit of an
   OCaml int: every device this suite tests opens after that many others. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled

let timeout = 60.
let empty d = Sub.make d [||]

let submit ?(waits = [||]) s =
  Rig.submit s ~run:(Sub.Run.make ()) ~buffers:[||] ~waits

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* The first device index whose points are negative ints. *)
let high = 32_768

(* Device indices are never reused, so opening this many devices puts every
   later one at [high] or above. *)
let () =
  for i = 1 to high do
    ignore (P.open_ (Printf.sprintf "point:filler-%d" i))
  done

(* A lost device's point its word did not reach is not done: a submit that
   follows it raises. *)
let test_lost_unreached () =
  let d, p = P.open_ ~answer:`Unknown "point:lost" in
  let c, _ = P.open_ "point:lost-consumer" in
  let a = submit (empty d) in
  P.fail p;
  raises_match ~msg:"the failed submit" (lost d) (fun () -> submit (empty d));
  raises_match ~msg:"a submit after the point" (lost d) (fun () ->
      submit (empty c) ~waits:[| a |])

(* A host read of a buffer waits for its last write. *)
let test_read_waits_write () =
  let d, p = P.open_ "point:write" in
  let src = B.create d 8 and dst = B.create d 8 in
  let copy =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  ignore (submit (Sub.make d [| copy |]));
  B.wait dst Read;
  equal int ~msg:"queued once the read returned" 0 (P.queued p)

(* A point of the submitting device itself is no foreign point: the submit
   waits for nothing. *)
let test_own_wait () =
  let d, p = P.open_ "point:own" in
  let a = submit (empty d) in
  ignore (submit (empty d) ~waits:[| a |]);
  equal int ~msg:"queued" 2 (P.queued p)

let tests =
  [
    group ~timeout "high device indices"
      [
        test "a lost device's unreached point is not done" test_lost_unreached;
        test "a host read waits for the last write" test_read_waits_write;
        test "a wait on the submitting device's own point waits for nothing"
          test_own_wait;
      ];
  ]

let () = exit (run "rig.point" tests)
