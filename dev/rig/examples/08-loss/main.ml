(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Loss.

   A device whose work fails is lost, once and for good. Every later use of it,
   and of memory whose stamps name it, raises [Lost]; its facts still answer;
   other devices go on. Opening its name again makes a new device.

   Here a fill that answers a failure ([fail.c]) loses a memory device, as a GPU
   is lost when its driver reports a fault. *)

open Rig

external fail : unit -> nativeint = "caml_rig_example_fail"

let copy src dst =
  { Submission.queue = "COPY:0"; after = [||]; work = Copy { src; dst } }

let lost f =
  match f () with
  | () -> print_endline "no loss"
  | exception (Lost _ as e) -> print_endline (Printexc.to_string e)

let () =
  let a = Result.get_ok (memory_device "A") in
  let b = Result.get_ok (memory_device "B") in

  (* A writes [x]: [x]'s stamps name A. *)
  let src = Buffer.create a 16 and x = Buffer.create a 16 in
  ignore
    (submit (Submission.make ~reads:0 ~writes:0 ~waits:0 a [| copy src x |]));

  (* A's next work fails, and A is lost. *)
  let arg = Buffer.create a 8 in
  let bad =
    Submission.Fill { fill = fail (); arg; ring_units = 0; segment_bytes = 0 }
  in
  let part = { Submission.queue = "COMPUTE:0"; after = [||]; work = bad } in
  let s = Submission.make ~reads:0 ~writes:0 ~waits:0 a [| part |] in
  lost (fun () -> ignore (submit s));

  (* Its facts answer: the reason, its name, its values. *)
  Printf.printf "lost: %s\n" (Option.value (Rig.lost a) ~default:"no");
  Printf.printf "%s submitted %d, signaled %d\n" (name a) (submitted a)
    (signaled a);

  (* Every use of it raises, and so does a use of memory its stamps name. *)
  lost (fun () -> ignore (Buffer.create a 16));
  lost (fun () -> Buffer.copy ~src:x ~dst:(Buffer.create host 16));

  (* B goes on. *)
  let y = Buffer.create b 16 and z = Buffer.create b 16 in
  let p =
    submit (Submission.make ~reads:0 ~writes:0 ~waits:0 b [| copy y z |])
  in
  Format.printf "B goes on: %a@." Point.pp p;

  (* The name opens again as a new device, with a timeline of its own. *)
  let a' = Result.get_ok (memory_device "A") in
  Printf.printf "%s again: the same device: %b, submitted %d, lost: %b\n"
    (name a') (equal a a') (submitted a')
    (Option.is_some (Rig.lost a'))
