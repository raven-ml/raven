(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kernel on an AMD GPU.

   Needs an AMD GPU of processor gfx1201, such as a Radeon AI PRO R9700, that
   Linux's amdgpu driver holds.

   The code object ([add.cl], compiled for gfx1201) is loaded on the device,
   which places its image in the GPU's memory. A submission launches its kernel
   [add] once: its parameters are the addresses of three arrays, each a ref to
   a buffer the submit passes, which the device writes into the parameters as
   it hands the launch over. The run holds what changes from one submit to the
   next: the launch's grid, its groups and the offsets into the arrays. *)

open Rig

let n = 1 lsl 20
let group = 64

let floats g f =
  let h = Buffer.create host (4 * n) in
  let a = Buffer.bigarray Bigarray.float32 h in
  for i = 0 to n - 1 do
    a.{i} <- f i
  done;
  let b = Buffer.create g (4 * n) in
  Buffer.copy ~src:h ~dst:b;
  b

let run g =
  let bin = In_channel.with_open_bin "add_gfx1201.hsaco" In_channel.input_all in
  let image = Result.get_ok (Image.load g bin) in

  (* One launch of [add], whose 24 bytes of parameters are three addresses:
     the run's buffers 0 and 1, which it reads, and 2, which it writes. *)
  let refs =
    Submission.
      [| { at = 0; slot = 0 }; { at = 8; slot = 1 }; { at = 16; slot = 2 } |]
  in
  let part =
    {
      Submission.queue = "COMPUTE:0";
      after = [||];
      work = Launch { image; kernel = "add"; params = 24; refs };
    }
  in
  let s = Submission.make ~reads:2 ~writes:1 g [| part |] in

  (* The run: n / 64 groups of 64 work-items, each array from its first
     byte. *)
  let run = Submission.Run.make () in
  let block = Submission.block s 0 in
  Submission.Run.groups run block (n / group) 1 1;
  Submission.Run.threads run block group 1 1;
  List.iter (fun at -> Submission.Run.int64 run block at 0) [ 0; 8; 16 ];

  let a = floats g float_of_int and b = floats g (fun _ -> 0.5) in
  let out = Buffer.create g (4 * n) in
  let pt = submit s ~run ~reads:[| a; b |] ~writes:[| out |] ~waits:[||] in
  Format.printf "%s (%s) ran add on %d floats at %a@." (name g) (arch g) n
    Point.pp pt;

  (* The copy back waits for the kernel's write. *)
  let back = Buffer.create host (4 * n) in
  Buffer.copy ~src:out ~dst:back;
  let r = Buffer.bigarray Bigarray.float32 back in
  let wrong = ref 0 in
  for i = 0 to n - 1 do
    if r.{i} <> float_of_int i +. 0.5 then incr wrong
  done;
  Printf.printf "out[0] = %g, out[%d] = %g, %d wrong\n" r.{0} (n - 1)
    r.{n - 1}
    !wrong

let () =
  if Rig_amd_amdgpu.count () = 0 then print_endline "no AMD GPU on this machine"
  else
    let g =
      open_
        (module Rig_amd)
        ~name:(Rig_amd_amdgpu.device_name 0)
        (fun () -> Rig_amd_amdgpu.open_ 0)
      |> Result.get_ok
    in
    if arch g <> "gfx1201" then
      Printf.printf "the code object is for gfx1201; this GPU is %s\n" (arch g)
    else if not (List.exists (fun (q : queue) -> List.mem Launch q.runs) (queues g))
    then print_endline "this GPU's compute queue runs no launch"
    else run g
