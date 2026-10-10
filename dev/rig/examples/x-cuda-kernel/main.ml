(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kernel on a CUDA GPU.

   Needs an NVIDIA GPU and the CUDA library ([libcuda]), which the NVIDIA driver
   installs.

   The kernel runs as a launch: a part that names the image's function and the
   buffers its parameters point into. Each run's grid and parameters are stored
   in a run, and rig adds each buffer's address to its parameter. The code is
   PTX text ([add.ptx]), which CUDA compiles for the GPU it loads on. *)

open Rig

let n = 1 lsl 20
let block = 256

let floats g f =
  let h = Buffer.create host (4 * n) in
  let a = Buffer.bigarray Bigarray.float32 h in
  for i = 0 to n - 1 do
    a.{i} <- f i
  done;
  let b = Buffer.create g (4 * n) in
  Buffer.copy ~src:h ~dst:b;
  b

(* [add]'s parameters: the addresses of [a], [b] and [out], each 8 bytes, then
   [n], 4 bytes. *)
let params = 28

let () =
  if Rig_cuda.count () = 0 then print_endline "no CUDA GPU on this machine"
  else
    let g =
      open_
        (module Rig_cuda)
        ~name:(Rig_cuda.device_name 0)
        (fun () -> Rig_cuda.open_ 0)
      |> Result.get_ok
    in

    (* The image, compiled by CUDA as it loads. *)
    let ptx = In_channel.with_open_bin "add.ptx" In_channel.input_all in
    let image = Result.get_ok (Image.load g ptx) in

    (* The arrays, in the GPU's memory. *)
    let a = floats g float_of_int and b = floats g (fun _ -> 0.5) in
    let out = Buffer.create g (4 * n) in

    (* The step: [add] reading the run's buffers 0 and 1 and writing its buffer
       2, each named by a parameter. *)
    let into at slot = { Submission.at; slot } in
    let refs = [| into 0 0; into 8 1; into 16 2 |] in
    let launch = Submission.Launch { image; kernel = "add"; params; refs } in
    let part =
      { Submission.queue = "COMPUTE:0"; after = [||]; work = launch }
    in
    let s = Submission.make ~access:[| Read; Read; Read_write |] g [| part |] in

    (* The run: the grid of n threads, and [n]. The refs' offsets stay 0, the
       start of each buffer. *)
    let run = Submission.Run.make () in
    let k = Submission.block s 0 in
    Submission.Run.groups run k (n / block) 1 1;
    Submission.Run.threads run k block 1 1;
    Submission.Run.int32 run k 24 n;
    let pt = submit s ~run ~buffers:[| a; b; out |] ~waits:[||] in
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
