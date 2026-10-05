(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AMD GPU code objects: the image a device runs, and where each kernel's
   descriptor lies in it. *)

module D = Amd_defs
module Elf = Nx_device_elf

type kernel = { descriptor : int; (* an image offset *) private_segment : int }

let r_amdgpu_rel64 = 5

(* The image of [binary], relocated, padded to whole dwords. *)
let image binary =
  let o = Elf.load binary in
  let b = Bytes.of_string o.image in
  List.iter
    (fun (r : Elf.relocation) ->
      if r.kind <> r_amdgpu_rel64 then
        failwith (Printf.sprintf "an unknown AMD GPU relocation %d" r.kind);
      match r.target with
      | Offset target ->
          Bytes.set_int64_le b r.at (Int64.of_int (target - r.at + r.addend))
      | External s ->
          failwith
            (Printf.sprintf "the code object refers to an undefined symbol %s" s))
    o.relocations;
  let n = Bytes.length b in
  let padded = Bytes.make ((n + 3) / 4 * 4) '\000' in
  Bytes.blit b 0 padded 0 n;
  (o, Bytes.to_string padded)

(* The kernel [name] of the loaded object [o] with image [img], for a GPU with
   [lds_kib] of LDS per work-group. *)
let kernel (o : Elf.t) img ~name ~lds_kib =
  let kd =
    match Elf.symbol o (name ^ ".kd") with
    | Some kd -> kd
    | None -> failwith (Printf.sprintf "the code object has no kernel %s" name)
  in
  let module K = D.Kernel_descriptor in
  if kd + K.sizeof > String.length img then
    failwith "a truncated kernel descriptor";
  let f field = Amdev.get img kd field in
  let group = f K.group_segment_fixed_size in
  if (group + 511) / 512 land 0x1FF > lds_kib * 1024 / 512 then
    failwith
      (Printf.sprintf "kernel %s needs %d bytes of LDS; the GPU has %d KiB" name
         group lds_kib);
  { descriptor = kd; private_segment = f K.private_segment_fixed_size }
