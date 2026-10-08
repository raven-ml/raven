(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kernel on an NVIDIA GPU, through its resource manager.

   Needs an NVIDIA GPU of architecture sm_89, such as an RTX 5000 Ada, that
   NVIDIA's Linux kernel driver holds.

   An NVIDIA GPU's channel runs words that compiled code writes. The cubin
   ([simple_add.cu], compiled for sm_89) is loaded on the device, which places
   its image in the GPU's memory. A launch is a descriptor, built from the
   kernel and the GPU, and a constant bank holding the launch's sizes and the
   kernel's parameters; a segment of channel words schedules the descriptor, and
   each submission places one ring entry, naming that segment, on the compute
   channel. *)

open Rig
module Abi = Rig_nv_abi

let n = 1 lsl 20
let block = 256

(* The launch's memory: its descriptor at 0, its constant bank 0 at [bank_at],
   its segment at [segment_at], in one page the host writes. *)
let bank_at = 512
let segment_at = 3584

let ints g f =
  let h = Buffer.create host (4 * n) in
  let a = Buffer.bigarray Bigarray.int32 h in
  for i = 0 to n - 1 do
    a.{i} <- Int32.of_int (f i)
  done;
  let b = Buffer.create g (4 * n) in
  Buffer.copy ~src:h ~dst:b;
  b

let write b ~at s =
  let a = Buffer.bigarray Bigarray.char (Option.get (Buffer.borrow host b)) in
  String.iteri (fun i c -> a.{at + i} <- c) s

let le64 x =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int x);
  Bytes.to_string b

let encode p = Abi.Packet.encode Int64.of_int p

let run g (gpu : Abi.Gpu.t) =
  let bin =
    In_channel.with_open_bin "simple_add_sm89.cubin" In_channel.input_all
  in
  let cubin = Result.get_ok (Abi.Cubin.of_string bin) in
  let k = Option.get (Abi.Cubin.kernel cubin "simple_add") in
  let launch = Result.get_ok (Abi.Launch.make gpu k) in

  (* The device places the image; its entry is the kernel's first
     instruction. *)
  let p = Result.get_ok (Program.load g bin) in
  let entry = Option.get (Program.entry p "simple_add") in
  let base = entry - k.code in

  (* The channels' local memory serves the kernel's threads. *)
  let local = Abi.Launch.local_bytes launch in
  Result.get_ok (gpu.local local);
  let per_thread = (Abi.Local_memory.make gpu local).per_thread in

  let a = ints g Fun.id and b = ints g (fun i -> 2 * i) in
  let out = Buffer.create g (4 * n) in

  (* The descriptor: the grid, the block, the program, its constant banks. *)
  let mem = Buffer.create ~memory:Mapped g 4096 in
  let at = Buffer.address mem in
  let bank q (c : Abi.Cubin.bank) =
    let addr = if c.index = 0 then at + bank_at else base + c.offset in
    Abi.Qmd.set_bank c.index addr q
  in
  let q =
    Abi.Qmd.make launch
    |> Abi.Qmd.set_dim (Grid X) (n / block)
    |> Abi.Qmd.set_dim (Grid Y) 1 |> Abi.Qmd.set_dim (Grid Z) 1
    |> Abi.Qmd.set_dim (Block X) block
    |> Abi.Qmd.set_dim (Block Y) 1
    |> Abi.Qmd.set_dim (Block Z) 1
    |> Abi.Qmd.set_program entry
    |> Abi.Qmd.set_local_memory per_thread
  in
  let q = List.fold_left bank q (Abi.Launch.banks launch) in

  (* Bank 0: the driver's parameters, then the kernel's, out, a, b and n, each
     in a 64-bit slot, n read as its low 32 bits. *)
  write mem ~at:bank_at
    (Abi.Structure.encode Int64.of_int (Abi.Qmd.parameters q));
  write mem
    ~at:(bank_at + k.params_offset)
    (String.concat ""
       (List.map le64
          [ Buffer.address out; Buffer.address a; Buffer.address b; n ]));
  write mem ~at:0 (Abi.Structure.encode Int64.of_int (Abi.Qmd.structure q));

  (* The segment that schedules the descriptor, and the ring entry that names
     the segment. *)
  let segment = encode (Abi.Method.schedule at) in
  write mem ~at:segment_at segment;
  let entry_words =
    encode
      (Abi.Gpfifo.entry (at + segment_at) ~offset:0
         ~words:(String.length segment / 4))
  in
  let words = Buffer.create host (String.length entry_words) in
  write words ~at:0 entry_words;

  (* The step: the launch's memory held with the program; the arrays passed to
     each submit. *)
  let hold =
    Hold.make ~release:(fun () -> ignore (Sys.opaque_identity p)) [ mem ]
  in
  let part =
    { Submission.queue = "COMPUTE:0"; after = [||]; work = Words words }
  in
  let s = Submission.make ~hold ~reads:2 ~writes:1 g [| part |] in
  let pt = submit s ~reads:[| a; b |] ~writes:[| out |] ~waits:[||] in
  Format.printf "%s (%s) ran simple_add on %d ints at %a@." (name g) (arch g) n
    Point.pp pt;

  (* The copy back waits for the kernel's write. *)
  let back = Buffer.create host (4 * n) in
  Buffer.copy ~src:out ~dst:back;
  let r = Buffer.bigarray Bigarray.int32 back in
  let wrong = ref 0 in
  for i = 0 to n - 1 do
    if r.{i} <> Int32.of_int (3 * i) then incr wrong
  done;
  Printf.printf "out[0] = %ld, out[%d] = %ld, %d wrong\n" r.{0} (n - 1)
    r.{n - 1}
    !wrong

let () =
  if Rig_nv_nvidia.count () = 0 then
    print_endline "no NVIDIA GPU on this machine"
  else
    let g =
      open_
        (module Rig_nv)
        ~name:(Rig_nv_nvidia.device_name 0)
        (fun () -> Rig_nv_nvidia.open_ 0)
      |> Result.get_ok
    in
    let gpu = Option.get (capability g Abi.Gpu.key) in
    if gpu.sass_version <> 0x89 then
      Printf.printf "the cubin is for sm_89; this GPU is %s\n" (arch g)
    else run g gpu
