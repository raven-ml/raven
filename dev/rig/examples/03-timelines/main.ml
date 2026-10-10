(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Timelines.

   A device's work is one sequence of submissions, numbered 1, 2, ...: the
   values of its timeline. A point names a device and a value; the value is
   reached once every submission up to it completed.

   A memory device is a device with a timeline of its own whose memory is the
   host's and whose work runs before [submit] returns. A GPU's work runs after
   [submit] returns; the calls are the same. *)

open Rig

let int32s b = Buffer.bigarray Bigarray.int32 b

(* The host reads a memory device's buffer, whose memory it addresses, once the
   work that wrote it is done. *)
let show name b =
  Buffer.wait b Read;
  let a = int32s b in
  let xs = List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}) in
  Printf.printf "%-4s [%s]\n" name (String.concat "; " xs)

let copy ?(after = [||]) src dst =
  { Submission.queue = "COPY:0"; after; work = Copy { src; dst } }

let timeline d =
  Printf.printf "%s submitted %d, signaled %d\n" (name d) (submitted d)
    (signaled d)

let () =
  let d = Result.get_ok (memory_device "M") in
  timeline d;

  (* Three buffers of the device; the host fills the first by a copy. *)
  let src = Buffer.create d 16 and mid = Buffer.create d 16 in
  let dst = Buffer.create d 16 in
  let init = Buffer.create host 16 in
  List.iteri (fun i x -> (int32s init).{i} <- x) [ 1l; 2l; 3l; 4l ];
  Buffer.copy ~src:init ~dst:src;

  (* A submission is made once and submitted many times, each submit with a
     run, the caller's storage for one submit at a time. Each submit is the
     point of the value it assigns. *)
  let s = Submission.make d [| copy src mid |] in
  let run = Submission.Run.make () in
  let p = submit s ~run ~buffers:[||] ~waits:[||] in
  Format.printf "first submit: %a@." Point.pp p;
  Format.printf "again:        %a@." Point.pp
    (submit s ~run ~buffers:[||] ~waits:[||]);
  timeline d;

  (* Parts of one submission run in order: the second copy runs after the first,
     as its [after] says. *)
  let chain =
    Submission.make d [| copy src mid; copy ~after:[| 0 |] mid dst |]
  in
  let p = submit chain ~run ~buffers:[||] ~waits:[||] in
  Point.wait p;
  Format.printf "chain:        %a@." Point.pp p;
  show "dst" dst;

  (* An empty submission takes a value too: a point after all earlier work. *)
  let empty = Submission.make d [||] in
  Format.printf "empty:        %a@." Point.pp
    (submit empty ~run ~buffers:[||] ~waits:[||]);
  timeline d;

  (* A value not yet submitted cannot be waited for. *)
  match wait d (submitted d + 1) with
  | () -> ()
  | exception Invalid_argument msg -> print_endline msg
