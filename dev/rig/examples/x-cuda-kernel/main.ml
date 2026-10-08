(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kernel on a CUDA GPU.

   Needs an NVIDIA GPU and the CUDA library ([libcuda]), which the NVIDIA driver
   installs.

   Compiled code runs a kernel through a fill ([run.c]) that calls
   [cuLaunchKernel] on the stream the device hands it. It finds that function
   through the device's capability, so it links no CUDA library. The program is
   PTX text ([add.ptx]), which CUDA compiles for the GPU it loads on. *)

open Rig

external run : unit -> nativeint = "caml_rig_example_run"

let n = 1 lsl 20
let block = 256

let words b xs =
  let a = Buffer.bigarray Bigarray.int64 (Option.get (Buffer.borrow host b)) in
  List.iteri (fun i x -> a.{i} <- Int64.of_int x) xs

let floats g f =
  let h = Buffer.create host (4 * n) in
  let a = Buffer.bigarray Bigarray.float32 h in
  for i = 0 to n - 1 do
    a.{i} <- f i
  done;
  let b = Buffer.create g (4 * n) in
  Buffer.copy ~src:h ~dst:b;
  b

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
    let cap = Option.get (capability g Rig_cuda_abi.key) in
    let launch = Option.get (cap.symbol "cuLaunchKernel") in

    (* The program, compiled by CUDA as it loads, and its kernel. *)
    let ptx = In_channel.with_open_bin "add.ptx" In_channel.input_all in
    let p = Result.get_ok (Program.load g ptx) in
    let add = Option.get (Program.entry p "add") in

    (* The arrays, in the GPU's memory. *)
    let a = floats g float_of_int and b = floats g (fun _ -> 0.5) in
    let out = Buffer.create g (4 * n) in

    (* The fill's argument, in pinned host memory that the fill reads on the
       host. It is fixed memory of the step: a hold keeps it, and the program
       until the work is done. *)
    let arg = Buffer.create ~memory:Pinned g 64 in
    words arg
      [
        Nativeint.to_int launch;
        add;
        n / block;
        block;
        Buffer.address a;
        Buffer.address b;
        Buffer.address out;
        n;
      ];
    let hold =
      Hold.make ~release:(fun () -> ignore (Sys.opaque_identity p)) [ arg ]
    in
    let fill =
      Submission.Fill { fill = run (); arg; ring_units = 0; segment_bytes = 0 }
    in
    let part = { Submission.queue = "COMPUTE:0"; after = [||]; work = fill } in
    let s = Submission.make ~hold ~reads:2 ~writes:1 g [| part |] in
    let pt = submit s ~reads:[| a; b |] ~writes:[| out |] ~waits:[||] in
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
