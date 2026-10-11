(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Timelines.

   A device numbers the work submitted to it 1, 2, 3, ... in the order it
   accepts it. These numbers are the values of its timeline. A point is a
   device and one of its values, and it is reached once the work of every
   value up to it has completed.

   A memory device has a timeline of its own and its memory lies in the
   host's memory. Its work runs before [submit] returns. A GPU's work runs
   after [submit] returns, and the calls are the same. *)

open Rig

(* [a]'s elements, joined. *)
let ints a =
  String.concat "; "
    (List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}))

(* A bigarray of [l]. *)
let int32s l = Bigarray.(Array1.of_array int32 c_layout (Array.of_list l))

(* The host reads a memory device's buffer, whose memory it addresses, once
   the work that wrote it is done. *)
let show name b =
  Buffer.read b Bigarray.int32 (fun a ->
      Printf.printf "%-4s [%s]\n" name (ints a))

let timeline d =
  Printf.printf "%s submitted %d, signaled %d\n" (name d) (submitted d)
    (signaled d)

let () =
  let d = memory_device "M" in
  timeline d;

  (* Three buffers of the device. The host fills the first by a copy. *)
  let src = Buffer.create d 16 and mid = Buffer.create d 16 in
  let dst = Buffer.create d 16 in
  Buffer.copy ~src:(Buffer.of_bigarray (int32s [ 1l; 2l; 3l; 4l ])) ~dst:src;

  (* A submission is made once and submitted many times, each submit with a
     run, the caller's storage for one submit at a time. A submit returns the
     point of the value it assigns. *)
  let s = Submission.(make d [| copy ~src:(Buffer src) ~dst:(Buffer mid) |]) in
  let run = Submission.Run.make () in
  Format.printf "first submit: %a@." Point.pp (submit s ~run ~buffers:[||]);
  Format.printf "again:        %a@." Point.pp (submit s ~run ~buffers:[||]);
  timeline d;

  (* The parts of one submission run in the order their memory requires. The
     second copy reads what the first writes, so it runs after it. *)
  let chain =
    Submission.(
      make d
        [|
          copy ~src:(Buffer src) ~dst:(Buffer mid);
          copy ~src:(Buffer mid) ~dst:(Buffer dst);
        |])
  in
  let p = submit chain ~run ~buffers:[||] in
  Format.printf "chain:        %a, reached: %b@." Point.pp p (Point.reached p);
  show "dst" dst;

  (* An empty submission takes a value too, which is reached once all earlier
     work has completed. *)
  let empty = Submission.make d [||] in
  Format.printf "empty:        %a@." Point.pp (submit empty ~run ~buffers:[||]);
  timeline d
