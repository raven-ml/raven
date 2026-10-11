(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Loss.

   A device whose work fails is lost, once and for good. Every later use of
   it and of its memory raises [Lost], and so does a use of other memory
   that depends on a point it never reached. Its properties keep their
   values, and other devices go on. Opening its name again makes a new
   device.

   Here a fill that returns a failure ([fail.c]) loses a memory device, as a
   GPU is lost when its driver reports a fault. *)

open Rig

external fail : unit -> nativeint = "caml_rig_example_fail"

let lost f =
  match f () with
  | () -> print_endline "no loss"
  | exception (Lost _ as e) -> print_endline (Printexc.to_string e)

let () =
  let a = memory_device "A" in
  let b = memory_device "B" in

  (* A writes [x], so [x]'s stamps name A. *)
  let src = Buffer.create a 16 and x = Buffer.create a 16 in
  let run = Submission.Run.make () in
  let s = Submission.(make a [| copy ~src:(Buffer src) ~dst:(Buffer x) |]) in
  ignore (submit s ~run ~buffers:[||]);

  (* A's next work fails, and A is lost. *)
  let s = Submission.make a [| Submission.fill (fail ()) ~params:0 |] in
  lost (fun () -> ignore (submit s ~run ~buffers:[||]));

  (* Its properties and timeline still answer, and [lost] gives the reason.
     Value 2, the failed fill's, is never reached. *)
  Printf.printf "lost: %s\n" (Option.value (Rig.lost a) ~default:"no");
  Printf.printf "%s submitted %d, signaled %d\n" (name a) (submitted a)
    (signaled a);

  (* Every use of it raises, and so does a use of its memory. *)
  lost (fun () -> ignore (Buffer.create a 16));
  lost (fun () -> Buffer.copy ~src:x ~dst:(Buffer.create host 16));

  (* B goes on. *)
  let y = Buffer.create b 16 and z = Buffer.create b 16 in
  let s = Submission.(make b [| copy ~src:(Buffer y) ~dst:(Buffer z) |]) in
  let p = submit s ~run ~buffers:[||] in
  Format.printf "B goes on: %a@." Point.pp p;

  (* The name opens again as a new device, with a timeline of its own. *)
  let a' = memory_device "A" in
  Printf.printf "%s again: the same device: %b, submitted %d, lost: %b\n"
    (name a') (equal a a') (submitted a')
    (Option.is_some (Rig.lost a'))
