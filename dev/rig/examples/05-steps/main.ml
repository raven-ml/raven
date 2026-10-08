(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A compiled step, prepared once and run many times.

   Compiled work reaches a device as a fill, a C function the device calls with
   an argument buffer ([scale.c]). A step keeps its fixed memory, here constants
   and the argument, in a hold; it names the buffers that change from run to run
   in the submission's slots, which a submit orders and clears. Once the step is
   unreachable and its work done, its memory returns and the hold's release
   runs. *)

open Rig

external scale : unit -> nativeint = "caml_rig_example_scale"

let n = 4

let ints d xs =
  let b = Buffer.create d (4 * n) in
  let h = Buffer.create host (4 * n) in
  List.iteri (fun i x -> (Buffer.bigarray Bigarray.int32 h).{i} <- x) xs;
  Buffer.copy ~src:h ~dst:b;
  b

let show name b =
  Buffer.wait b Read;
  let a = Buffer.bigarray Bigarray.int32 (Option.get (Buffer.borrow host b)) in
  let xs = List.init n (fun i -> Int32.to_string a.{i}) in
  Printf.printf "%-4s [%s]\n" name (String.concat "; " xs)

(* The step: its constants [k] and its argument in one hold, and one fill that
   reads one slot and writes another. [run] sets the slots and the argument,
   then submits. *)
let step d =
  let k = ints d [ 1l; 10l; 100l; 1000l ] in
  let arg = Buffer.create d 32 in
  let hold =
    Hold.make ~release:(fun () -> print_endline "step released") [ k; arg ]
  in
  let fill =
    Submission.Fill { fill = scale (); arg; ring_units = 0; segment_bytes = 0 }
  in
  let part = { Submission.queue = "COMPUTE:0"; after = [||]; work = fill } in
  let s = Submission.make ~hold ~reads:1 ~writes:1 ~waits:0 d [| part |] in
  let words =
    Buffer.bigarray Bigarray.int64 (Option.get (Buffer.borrow host arg))
  in
  fun ~src ~dst ->
    (* The host rewrites the argument once the work that read it is done. *)
    Buffer.wait arg Read_write;
    List.iteri
      (fun i a -> words.{i} <- Int64.of_int a)
      [ Buffer.address dst; Buffer.address src; Buffer.address k; n ];
    Submission.read s 0 src;
    Submission.write s 0 dst;
    submit s

(* Two runs of one step, each on buffers of its own. *)
let runs d =
  let run = step d in
  let x = ints d [ 1l; 2l; 3l; 4l ] and y = Buffer.create d (4 * n) in
  Format.printf "run 1: %a@." Point.pp (run ~src:x ~dst:y);
  show "y" y;
  let z = Buffer.create d (4 * n) in
  Format.printf "run 2: %a@." Point.pp (run ~src:y ~dst:z);
  show "z" z

let () =
  let d = Result.get_ok (memory_device "M") in
  runs d;

  (* The step is unreachable: once collected, the next drain of the device, such
     as a [Buffer.create], returns its memory and runs its release. *)
  Gc.full_major ();
  ignore (Buffer.create d 0);
  print_endline "done"
