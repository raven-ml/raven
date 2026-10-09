(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kernel on the Mac's GPU.

   Needs a Mac whose GPU supports Metal, on macOS 15 or later.

   Compiled code runs a kernel through a fill ([run.c]). The image is loaded
   on the device, its function named by its pipeline, and the dispatch recorded
   once in an indirect command buffer, which the device's capability makes. Each
   submission's fill executes that buffer. The kernel's arguments, the GPU
   addresses of its arrays, sit in an argument buffer it reads as buffer 0. *)

open Rig

external run : unit -> nativeint = "caml_rig_example_run"

let n = 1 lsl 20
let group = 256

let floats g f =
  let b = Buffer.create g (4 * n) in
  let a =
    Buffer.bigarray Bigarray.float32 (Option.get (Buffer.borrow host b))
  in
  for i = 0 to n - 1 do
    a.{i} <- f i
  done;
  b

let words b xs =
  let a = Buffer.bigarray Bigarray.int64 (Option.get (Buffer.borrow host b)) in
  List.iteri (fun i x -> a.{i} <- Int64.of_int x) xs

let () =
  if Rig_metal.count () = 0 then print_endline "no Metal GPU on this machine"
  else
    let g =
      open_
        (module Rig_metal)
        ~name:(Rig_metal.device_name 0)
        (fun () -> Rig_metal.open_ 0)
      |> Result.get_ok
    in
    let cap = Option.get (capability g Rig_metal_abi.key) in

    (* The image and its kernel's pipeline. *)
    let lib = In_channel.with_open_bin "add.metallib" In_channel.input_all in
    let p = Result.get_ok (Image.load g lib) in
    let pipeline = Option.get (Image.entry p "add") in

    (* The arrays, and the argument buffer that names them by GPU address. *)
    let a = floats g float_of_int and b = floats g (fun _ -> 0.5) in
    let out = Buffer.create g (4 * n) in
    let args = Buffer.create g 24 in
    words args [ Buffer.address a; Buffer.address b; Buffer.address out ];

    (* One dispatch of n threads, recorded once. *)
    let dispatch =
      {
        Rig_metal_abi.pipeline;
        offset = Buffer.offset args;
        groups = (n / group, 1, 1);
        threads = (group, 1, 1);
      }
    in
    let icb = Result.get_ok (cap.icb (Buffer.handle args) [| dispatch |]) in

    (* The step: its fixed memory in a hold whose release ends the indirect
       command buffer and keeps the image until then; its arrays passed to
       each submit. *)
    let fill_arg = Buffer.create g 16 in
    words fill_arg [ Nativeint.to_int icb.handle; 1 ];
    let hold =
      Hold.make
        ~release:(fun () ->
          icb.release ();
          ignore (Sys.opaque_identity p))
        [ args; fill_arg ]
    in
    let fill =
      Submission.Fill
        { fill = run (); arg = fill_arg; ring_units = 0; segment_bytes = 0 }
    in
    let part = { Submission.queue = "COMPUTE:0"; after = [||]; work = fill } in
    let s = Submission.make ~hold ~reads:2 ~writes:1 g [| part |] in
    let run = Submission.Run.make () in
    let pt = submit s ~run ~reads:[| a; b |] ~writes:[| out |] ~waits:[||] in
    Format.printf "%s ran add on %d floats at %a@." (name g) n Point.pp pt;

    (* The host reads the result once the GPU wrote it. *)
    Buffer.wait out Read;
    let r =
      Buffer.bigarray Bigarray.float32 (Option.get (Buffer.borrow host out))
    in
    let wrong = ref 0 in
    for i = 0 to n - 1 do
      if r.{i} <> float_of_int i +. 0.5 then incr wrong
    done;
    Printf.printf "out[0] = %g, out[%d] = %g, %d wrong\n" r.{0} (n - 1)
      r.{n - 1}
      !wrong
