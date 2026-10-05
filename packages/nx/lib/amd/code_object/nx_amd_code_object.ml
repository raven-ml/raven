(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Elf = Nx_device_elf
module D = Code_object_defs
module K = D.Kernel_descriptor

type t = { elf : Elf.t; image : string; target : string }

(* LLVM's AMDGPU relocations, ELFRelocs/AMDGPU.def. *)
let r_amdgpu_rel64 = 5
let ( let* ) = Result.bind

let relocate b (r : Elf.relocation) =
  match r.target with
  | _ when r.kind <> r_amdgpu_rel64 ->
      Error (Printf.sprintf "an unknown AMD GPU relocation %d" r.kind)
  | External s ->
      Error
        (Printf.sprintf "the code object refers to an undefined symbol %s" s)
  | Offset target ->
      Bytes.set_int64_le b r.at (Int64.of_int (target - r.at + r.addend));
      Ok ()

(* The processor of [elf]'s flags. A generic processor's code object carries the
   version of the processor's code, which LLVM numbers from 1, in a code object
   of version 6 or later. *)
let processor (elf : Elf.t) =
  let mach = elf.flags land D.ef_amdgpu_mach in
  let version =
    (elf.flags land D.ef_amdgpu_generic_version)
    lsr D.ef_amdgpu_generic_version_offset
  in
  match List.assoc_opt mach D.processors with
  | None ->
      Error
        (Printf.sprintf "a code object for no AMD GPU (EF_AMDGPU_MACH 0x%x)"
           mach)
  | Some name when not (List.mem_assoc name D.generic) -> Ok name
  | Some name when elf.abi_version < D.elfabiversion_amdgpu_hsa_v6 ->
      Error
        (Printf.sprintf "a %s code object before code object version 6" name)
  | Some name when version = 0 ->
      Error (Printf.sprintf "a %s code object of generic version 0" name)
  | Some name -> Ok name

let of_string obj =
  let* elf =
    match Elf.load obj with e -> Ok e | exception Failure why -> Error why
  in
  let* () =
    if elf.machine = D.em_amdgpu then Ok ()
    else
      Error
        (Printf.sprintf "not an AMD GPU code object (e_machine %d)" elf.machine)
  in
  let* target = processor elf in
  let n = String.length elf.image in
  let b = Bytes.make ((n + 3) / 4 * 4) '\000' in
  Bytes.blit_string elf.image 0 b 0 n;
  let* () =
    List.fold_left
      (fun acc r -> Result.bind acc (fun () -> relocate b r))
      (Ok ()) elf.relocations
  in
  Ok { elf; image = Bytes.to_string b; target }

let image co = co.image
let target co = co.target

let runs_on co gpu =
  co.target = gpu
  || List.exists
       (fun (g, members) -> g = co.target && List.mem gpu members)
       D.generic

let kd_suffix = ".kd"

let kernels co =
  Array.to_list co.elf.symbols
  |> List.filter_map (fun (s : Elf.symbol) ->
      match s.place with
      | Defined _ when String.ends_with ~suffix:kd_suffix s.name ->
          Some
            (String.sub s.name 0
               (String.length s.name - String.length kd_suffix))
      | Defined _ | Undefined -> None)
  |> List.sort_uniq String.compare

type kernel = {
  descriptor : int;
  entry : int;
  group_segment : int;
  private_segment : int;
  kernarg_size : int;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave32 : bool;
  dispatch_ptr : bool;
  private_segment_buffer : bool;
}

let get s base (off, width) =
  let at = base + off in
  match width with
  | 2 -> String.get_uint16_le s at
  | 4 -> Int32.to_int (String.get_int32_le s at) land 0xffff_ffff
  | _ -> Int64.to_int (String.get_int64_le s at)

let kernel co name =
  let img = co.image in
  let* kd =
    Option.to_result
      (Elf.symbol co.elf (name ^ kd_suffix))
      ~none:(Printf.sprintf "the code object has no kernel %s" name)
  in
  let f = get img kd in
  if kd < 0 || kd + K.sizeof > String.length img then
    Error (Printf.sprintf "kernel %s's descriptor is truncated" name)
  else
    let entry = kd + f K.kernel_code_entry_byte_offset in
    if entry < 0 || entry >= String.length img then
      Error (Printf.sprintf "kernel %s's code lies outside the image" name)
    else
      let props = f K.kernel_code_properties in
      let has flag = props land flag <> 0 in
      Ok
        {
          descriptor = kd;
          entry;
          group_segment = f K.group_segment_fixed_size;
          private_segment = f K.private_segment_fixed_size;
          kernarg_size = f K.kernarg_size;
          rsrc1 = f K.compute_pgm_rsrc1;
          rsrc2 = f K.compute_pgm_rsrc2;
          rsrc3 = f K.compute_pgm_rsrc3;
          wave32 = has D.amd_kernel_code_properties_enable_wavefront_size32;
          dispatch_ptr =
            has D.amd_kernel_code_properties_enable_sgpr_dispatch_ptr;
          private_segment_buffer =
            has D.amd_kernel_code_properties_enable_sgpr_private_segment_buffer;
        }
