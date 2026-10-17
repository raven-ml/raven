(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A compiled step, made once and run many times.

   Here the step's work is a fill, a C function the device calls with the
   parameters of each run ([scale.c]). A step names its fixed memory, here
   constants, when it is made, and passes the buffers that change from run
   to run to each submit. Every submit orders both. Fixed memory has an
   address known at [make], so compiled code writes it once into parameters
   or recorded commands, which every run then replays as they are. Once the
   step is unreachable and its work is done, its memory returns and it no
   longer keeps its value. *)

open Rig

external scale : unit -> nativeint = "caml_rig_example_scale"

let n = 4

(* [a]'s elements, joined. *)
let ints a =
  String.concat "; "
    (List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}))

(* A bigarray of [l]. *)
let int32s l = Bigarray.(Array1.of_array int32 c_layout (Array.of_list l))

(* A buffer of [d] that holds [l]. *)
let holding d l =
  let b = Buffer.create d (4 * List.length l) in
  Buffer.copy ~src:(Buffer.of_bigarray (int32s l)) ~dst:b;
  b

let show name b =
  Buffer.read b Bigarray.int32 (fun a ->
      Printf.printf "%-4s [%s]\n" name (ints a))

(* The step: its constants [k] as fixed memory, a value it keeps that says
   when it is collected, one fill that reads one buffer and writes another,
   and the run each submit uses. The constants' address goes into the run
   once. The function it returns submits the step with the call's
   buffers. *)
let step d =
  let k = holding d [ 1l; 10l; 100l; 1000l ] in
  let held = ref () in
  Gc.finalise (fun _ -> print_endline "step released") held;
  let refs = Submission.[| write 1 ~at:0; read 0 ~at:8 |] in
  let part = Submission.fill (scale ()) ~params:32 ~refs in
  let s = Submission.make ~keep:held ~fixed:[ (k, Read) ] d [| part |] in
  let run = Submission.Run.make s in
  let block = Submission.block s 0 in
  Submission.Run.int64 run block 16 (Int64.of_int (Submission.address s k));
  Submission.Run.int64 run block 24 (Int64.of_int n);
  fun ~src ~dst -> submit run ~buffers:[| src; dst |]

(* Two runs of one step, each on buffers of its own. *)
let runs d =
  let run = step d in
  let x = holding d [ 1l; 2l; 3l; 4l ] and y = Buffer.create d (4 * n) in
  Format.printf "run 1: %a@." Point.pp (run ~src:x ~dst:y);
  show "y" y;
  let z = Buffer.create d (4 * n) in
  Format.printf "run 2: %a@." Point.pp (run ~src:y ~dst:z);
  show "z" z

let () =
  let d = Result.get_ok (memory_device "M") in
  runs d;

  (* The step is unreachable. Once the collector has collected it, the next
     [Buffer.create] on the device returns its memory and releases its hold,
     whose value the next collection finds. *)
  Gc.full_major ();
  ignore (Buffer.create d 0);
  Gc.full_major ();
  print_endline "done"
