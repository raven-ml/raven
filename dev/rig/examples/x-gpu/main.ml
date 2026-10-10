(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Opening a GPU.

   Needs a GPU: the Mac's through Metal (macOS 15 or later), or on Linux an
   NVIDIA GPU (through CUDA or NVIDIA's kernel driver) or an AMD GPU (through
   amdgpu).

   A GPU is opened by a driver, the library that runs it, and a path, the
   library that reached it. [Rig.open_] takes both under one name, so that one
   name has one live device. After that it is a device like the others: its
   buffers, copies and submissions are the ones of the examples before, with its
   work running after [submit] returns.

   With an argument, metal, cuda, nv or amd, it opens GPU 0 of that path;
   without, GPU 0 of the first path that sees one. *)

open Rig

let mib = 1024 * 1024

(* Each path: whether it sees a GPU, and how the driver and the path open GPU
   0. *)
let paths =
  [
    ( "metal",
      (fun () -> Rig_metal.count ()),
      fun () ->
        open_
          (module Rig_metal)
          ~name:(Rig_metal.device_name 0)
          (fun () -> Rig_metal.open_ 0) );
    ( "cuda",
      (fun () -> Rig_cuda.count ()),
      fun () ->
        open_
          (module Rig_cuda)
          ~name:(Rig_cuda.device_name 0)
          (fun () -> Rig_cuda.open_ 0) );
    ( "nv",
      (fun () -> Rig_nv_nvidia.count ()),
      fun () ->
        open_
          (module Rig_nv)
          ~name:(Rig_nv_nvidia.device_name 0)
          (fun () -> Rig_nv_nvidia.open_ 0) );
    ( "amd",
      (fun () -> Rig_amd_amdgpu.count ()),
      fun () ->
        open_
          (module Rig_amd)
          ~name:(Rig_amd_amdgpu.device_name 0)
          (fun () -> Rig_amd_amdgpu.open_ 0) );
  ]

let choose () =
  let seen = List.filter (fun (_, count, _) -> count () > 0) paths in
  match Sys.argv with
  | [| _ |] -> List.nth_opt seen 0
  | [| _; path |] -> List.find_opt (fun (p, _, _) -> p = path) seen
  | _ -> invalid_arg "usage: main.exe [metal|cuda|nv|amd]"

(* Copies timed by the profile's copy events. *)
let report events =
  List.iter
    (function
      | Profile.Copy { src; dst; bytes; start; stop } ->
          Printf.printf "  %-6s -> %-6s %4d MiB at %5.1f GB/s\n" (name src)
            (name dst) (bytes / mib)
            (float bytes /. float (stop - start))
      | _ -> ())
    events

let () =
  match choose () with
  | None -> print_endline "no GPU on this machine for the path asked"
  | Some (path, _, open_gpu) ->
      let g = Result.get_ok (open_gpu ()) in
      Printf.printf "%s through %s: %s, budget %d MiB\n" (name g) path (arch g)
        (budget g / mib);
      Printf.printf "shares the host's memory: %b; reaches the host's: %b\n\n"
        (shares_host_memory g) (reaches g host);

      (* A round trip: host to GPU to host, ordered by the copies alone. *)
      let n = 256 * mib in
      let src = Buffer.create host n and back = Buffer.create host n in
      let ba = Buffer.bigarray Bigarray.char src in
      Bigarray.Array1.fill ba 'r';
      let on_gpu = Buffer.create g n in
      let (), events =
        Profile.take (fun () ->
            Buffer.copy ~src ~dst:on_gpu;
            Buffer.copy ~src:on_gpu ~dst:back)
      in
      report events;
      let same = Buffer.bigarray Bigarray.char back = ba in
      Printf.printf "the bytes came back: %b\n\n" same;

      (* Work runs after [submit] returns: right after it, the GPU has not
         always reached the value it assigned. [Point.wait] returns once it
         has. *)
      let s = Submission.make g [||] in
      let run = Submission.Run.make () in
      let p = submit s ~run ~buffers:[||] ~waits:[||] in
      Format.printf "submitted %a; signaled %d on return@." Point.pp p
        (signaled g);
      Point.wait p;
      Printf.printf "after wait: signaled %d\n" (signaled g)
