(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AMD GPU code objects: the image a device runs, and the dispatch parameters of
   its kernels. *)

module D = Amd_defs
module Elf = Nx_device_support.Elf

type kernel = {
  descriptor : int; (* image offsets *)
  entry : int;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave32 : bool;
  private_segment : int;
  group_segment : int;
  kernarg_segment : int;
  dispatch_ptr : bool;
  private_segment_buffer : bool;
}

let r_amdgpu_rel64 = 5

(* The image of [binary], relocated, padded to whole dwords. *)
let image binary =
  let o = Elf.load binary in
  let b = Bytes.of_string o.image in
  List.iter
    (fun (r : Elf.relocation) ->
      if r.kind <> r_amdgpu_rel64 then
        failwith (Printf.sprintf "an unknown AMD GPU relocation %d" r.kind);
      Bytes.set_int64_le b r.at (Int64.of_int (r.target - r.at + r.addend)))
    o.relocations;
  let n = Bytes.length b in
  let padded = Bytes.make ((n + 3) / 4 * 4) '\000' in
  Bytes.blit b 0 padded 0 n;
  (o, Bytes.to_string padded)

(* The kernel [name] of the loaded object [o] with image [img], for a GPU of
   graphics target [major] with [lds_kib] of LDS per work-group. *)
let kernel (o : Elf.t) img ~name ~major ~lds_kib =
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
  let lds = (group + 511) / 512 land 0x1FF in
  if lds > lds_kib * 1024 / 512 then
    failwith
      (Printf.sprintf "kernel %s needs %d bytes of LDS; the GPU has %d KiB" name
         group lds_kib);
  let props = f K.kernel_code_properties in
  {
    descriptor = kd;
    entry = kd + f K.kernel_code_entry_byte_offset;
    (* gfx11 runs kernels privileged, for their context save and restore. *)
    rsrc1 = (f K.compute_pgm_rsrc1 lor if major = 11 then 1 lsl 20 else 0);
    rsrc2 = f K.compute_pgm_rsrc2 lor (lds lsl 15);
    rsrc3 = f K.compute_pgm_rsrc3;
    wave32 =
      props land D.amd_kernel_code_properties_enable_wavefront_size32 <> 0;
    private_segment = f K.private_segment_fixed_size;
    group_segment = group;
    kernarg_segment = f K.kernarg_size;
    dispatch_ptr =
      props land D.amd_kernel_code_properties_enable_sgpr_dispatch_ptr <> 0;
    private_segment_buffer =
      props land D.amd_kernel_code_properties_enable_sgpr_private_segment_buffer
      <> 0;
  }
