(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module K = Defs.Kernel_descriptor

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

type t = {
  elf : Device_elf.t;
  target : string;
  size : int;
  patches : (int * string) list;
  kernels : (string * kernel) list; (* by name, in increasing order *)
}

let ( let* ) = Result.bind

(* LLVM's AMDGPU relocations, ELFRelocs/AMDGPU.def. *)
let r_amdgpu_rel64 = 5

(* The bytes of an R_AMDGPU_REL64 field. *)
let rel64_bytes = 8

(* The longest image: what a GPU's 48-bit virtual addresses reach. *)
let max_size = 1 lsl 48

(* The processor of [o]'s flags. A generic processor's code object carries the
   version of the processor's code, which LLVM numbers from 1, in a code object
   of version 6 or later. *)
let processor (o : Device_elf.t) =
  let mach = o.flags land Defs.ef_amdgpu_mach in
  let version =
    (o.flags land Defs.ef_amdgpu_generic_version)
    lsr Defs.ef_amdgpu_generic_version_offset
  in
  match List.assoc_opt mach Defs.processors with
  | None -> Error (Printf.sprintf "EF_AMDGPU_MACH 0x%x names no processor" mach)
  | Some name when not (List.mem_assoc name Defs.generic) -> Ok name
  | Some name when o.abi_version < Defs.elfabiversion_amdgpu_hsa_v6 ->
      Error
        (Printf.sprintf "a %s code object before code object version 6" name)
  | Some name when version = 0 ->
      Error (Printf.sprintf "a %s code object of generic version 0" name)
  | Some name -> Ok name

(* The patch of [r]: its target's offset from the field, which REL64 writes. *)
let patch ~size i (r : Device_elf.relocation) =
  let* target =
    match r.symbol.place with
    | _ when r.kind <> r_amdgpu_rel64 ->
        Error
          (Printf.sprintf "relocation %d is of kind %d, expected R_AMDGPU_REL64"
             i r.kind)
    | Image { offset; _ } -> Ok offset
    | Undefined | Absolute _ | Outside _ ->
        Error
          (Printf.sprintf "relocation %d uses %S, whose bytes the image lacks" i
             r.symbol.name)
  in
  if r.offset + rel64_bytes > size then
    Error
      (Printf.sprintf "relocation %d patches bytes past the image's end at %d" i
         r.offset)
  else
    let b = Bytes.create rel64_bytes in
    Bytes.set_int64_le b 0 (Int64.of_int (target + r.addend - r.offset));
    Ok (r.offset, Bytes.unsafe_to_string b)

let rec patches ~size i acc = function
  | [] -> Ok (List.rev acc)
  | r :: rs ->
      let* p = patch ~size i r in
      patches ~size (i + 1) (p :: acc) rs

(* The [n] bytes of the image at [at], which lie in it: the sections' bytes
   there, then the patches over them. *)
let read (o : Device_elf.t) ps at n =
  let b = Bytes.make n '\000' in
  let blit src src_at dst_at len =
    let lo = Int.max at dst_at and hi = Int.min (at + n) (dst_at + len) in
    if lo < hi then
      Bytes.blit_string src (src_at + lo - dst_at) b (lo - at) (hi - lo)
  in
  let section (s : Device_elf.section) =
    match s.offset with Some off -> blit o.file s.at off s.length | None -> ()
  in
  Iarray.iter section o.sections;
  List.iter (fun (off, p) -> blit p 0 off (String.length p)) ps;
  Bytes.unsafe_to_string b

let field s (off, width) =
  match width with
  | 2 -> String.get_uint16_le s off
  | 4 -> Int32.to_int (String.get_int32_le s off) land 0xffff_ffff
  | _ -> Int64.to_int (String.get_int64_le s off)

let kd_suffix = ".kd"

let kernel_of o ~size ps name kd =
  if kd + K.sizeof > size then
    Error
      (Printf.sprintf "kernel %s's descriptor at %d lies past the image's end"
         name kd)
  else
    let d = read o ps kd K.sizeof in
    let entry = kd + field d K.kernel_code_entry_byte_offset in
    if entry < 0 || entry >= size then
      Error
        (Printf.sprintf "kernel %s's code at %d lies outside the image" name
           entry)
    else
      let has flag = field d K.kernel_code_properties land flag <> 0 in
      Ok
        {
          descriptor = kd;
          entry;
          group_segment = field d K.group_segment_fixed_size;
          private_segment = field d K.private_segment_fixed_size;
          kernarg_size = field d K.kernarg_size;
          rsrc1 = field d K.compute_pgm_rsrc1;
          rsrc2 = field d K.compute_pgm_rsrc2;
          rsrc3 = field d K.compute_pgm_rsrc3;
          wave32 = has Defs.amd_kernel_code_properties_enable_wavefront_size32;
          dispatch_ptr =
            has Defs.amd_kernel_code_properties_enable_sgpr_dispatch_ptr;
          private_segment_buffer =
            has
              Defs.amd_kernel_code_properties_enable_sgpr_private_segment_buffer;
        }

(* The kernels of [o], by name in increasing order, each at the first symbol
   [name ^ ".kd"] in its image. *)
let descriptors (o : Device_elf.t) =
  let descriptor (s : Device_elf.symbol) =
    match s.place with
    | Image { offset; _ } when String.ends_with ~suffix:kd_suffix s.name ->
        let n = String.length s.name - String.length kd_suffix in
        Some (String.sub s.name 0 n, offset)
    | Image _ | Undefined | Absolute _ | Outside _ -> None
  in
  let rec first = function
    | ((a, _) as k) :: (b, _) :: rest when String.equal a b -> first (k :: rest)
    | k :: rest -> k :: first rest
    | [] -> []
  in
  List.filter_map descriptor (Iarray.to_list o.symbols)
  |> List.stable_sort (fun (a, _) (b, _) -> String.compare a b)
  |> first

let rec kernels o ~size ps acc = function
  | [] -> Ok (List.rev acc)
  | (name, kd) :: rest ->
      let* k = kernel_of o ~size ps name kd in
      kernels o ~size ps ((name, k) :: acc) rest

let of_string obj =
  let* o = Device_elf.of_string obj in
  let* () =
    if o.machine = Defs.em_amdgpu then Ok ()
    else
      Error
        (Printf.sprintf "e_machine %d, expected EM_AMDGPU (%d)" o.machine
           Defs.em_amdgpu)
  in
  let* target = processor o in
  let* () =
    if o.size <= max_size then Ok ()
    else Error (Printf.sprintf "the image is %d bytes, longer than 2^48" o.size)
  in
  let size = (o.size + 3) / 4 * 4 in
  let* ps = patches ~size 0 [] o.relocations in
  let* ks = kernels o ~size ps [] (descriptors o) in
  Ok { elf = o; target; size; patches = ps; kernels = ks }

let target co = co.target
let size co = co.size
let elf co = co.elf
let patches co = co.patches
let kernels co = List.map fst co.kernels
let kernel co name = List.assoc_opt name co.kernels

let runs_on co gpu =
  String.equal co.target gpu
  || List.exists
       (fun (g, members) -> String.equal g co.target && List.mem gpu members)
       Defs.generic
