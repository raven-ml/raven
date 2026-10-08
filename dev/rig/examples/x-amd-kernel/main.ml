(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kernel on an AMD GPU.

   Needs an AMD GPU of processor gfx1201, such as a Radeon AI PRO R9700, that
   Linux's amdgpu driver holds, with a compute queue that reads PM4 packets.

   An AMD GPU's queue reads packets that compiled code writes. The code object
   ([add.cl], compiled for gfx1201) is loaded on the device, which places its
   image in the GPU's memory. The dispatch of its kernel is written once as PM4
   words from the kernel's descriptor, and each submission places those words on
   the compute queue. The kernel reads its arguments, the addresses of its
   arrays, from memory the GPU addresses. *)

open Rig
module Abi = Rig_amd_abi

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

(* A host buffer holding [s]'s bytes. *)
let of_string s =
  let b = Buffer.create host (String.length s) in
  let a = Buffer.bigarray Bigarray.char b in
  String.iteri (fun i c -> a.{i} <- c) s;
  b

let le64 xs =
  String.concat ""
    (List.map
       (fun x ->
         let b = Bytes.create 8 in
         Bytes.set_int64_le b 0 (Int64.of_int x);
         Bytes.to_string b)
       xs)

let run g (cap : Abi.Capability.t) =
  let bin = In_channel.with_open_bin "add_gfx1201.hsaco" In_channel.input_all in
  let co = Result.get_ok (Abi.Code_object.of_string bin) in
  let k = Option.get (Abi.Code_object.kernel co "add") in

  (* The device places the image; the kernel's descriptor names where its code
     starts. *)
  let p = Result.get_ok (Program.load g bin) in
  let base = Option.get (Program.entry p "add") - k.descriptor in

  (* The arrays, and the kernel's arguments in pinned memory. *)
  let a = floats g float_of_int and b = floats g (fun _ -> 0.5) in
  let out = Buffer.create g (4 * n) in
  let args = Buffer.create ~memory:Pinned g 24 in
  Buffer.copy
    ~src:
      (of_string
         (le64 [ Buffer.address a; Buffer.address b; Buffer.address out ]))
    ~dst:args;

  (* The dispatch, as words: [Pm4.run] makes the kernel read what earlier work
     wrote and the packets after it wait for its waves. *)
  let dispatch =
    Abi.Pm4.dispatch cap.gpu k ~program:(base + k.entry) ~scratch:0
      ~args:(Buffer.address args) ~packet:0 ~threads:(group, 1, 1)
      ~groups:(n / group, 1, 1)
      ()
  in
  let words =
    of_string (Abi.Packet.encode Int64.of_int (Abi.Pm4.run cap.gpu dispatch))
  in
  Printf.printf "the dispatch is %d words of PM4\n" (Buffer.length words / 4);

  (* The step: the arguments held with the program; the arrays in slots. *)
  let hold =
    Hold.make ~release:(fun () -> ignore (Sys.opaque_identity p)) [ args ]
  in
  let part =
    { Submission.queue = "COMPUTE:0"; after = [||]; work = Words words }
  in
  let s = Submission.make ~hold ~reads:2 ~writes:1 ~waits:0 g [| part |] in
  Submission.read s 0 a;
  Submission.read s 1 b;
  Submission.write s 0 out;
  let pt = submit s in
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
    let cap = Option.get (capability g Abi.Capability.key) in
    match cap.compute with
    | Aql _ -> print_endline "this GPU's compute queue reads AQL packets"
    | Pm4 when Abi.Gpu.processor cap.gpu <> "gfx1201" ->
        Printf.printf "the code object is for gfx1201; this GPU is %s\n"
          (Abi.Gpu.processor cap.gpu)
    | Pm4 -> run g cap
