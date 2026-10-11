(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Boot_defs

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Windows *)

(* The memory controller's locations and the fabric's segments are in units of
   16 MiB. *)
let aperture_unit = 24
let aperture_mask = 0xff_ffff

(* The memory controller's window on the GPU's memory: its first byte, and its
   last, the end of its top unit. *)
let first base = (base land aperture_mask) lsl aperture_unit
let last top = (((top land aperture_mask) + 1) lsl aperture_unit) - 1

let window ~base ~top ~fabric ~memory =
  let lo = first base and hi = last top in
  let at = lo + fabric in
  if hi >= lo && at + memory - 1 <= hi then Ok ()
  else
    Error
      (strf
         "the memory controller's window [0x%x, 0x%x] does not hold the GPU's \
          %d MiB at 0x%x"
         lo hi (memory lsr 20) at)

(* Entries *)

let ( |: ) = Int64.logor
let flag c f = if c then f else 0L
let entry_address = 0x0000_ffff_ffff_f000

let mtype_uc = function
  | 9, _, _ -> D.soc15_mtype_uc
  | (10 | 11), _, _ -> D.soc21_mtype_uc
  | _ -> D.soc24_mtype_uc

let entry ~gc ~level ~pa target ~uncached ~snooped ~fragment =
  if pa land lnot entry_address <> 0 then
    invalid_argf "Gmc.entry: address 0x%x is not a 4 KiB page below 2^48" pa;
  let table = target = `Table in
  let system = target = `Page `System in
  let shift v s = Int64.shift_left (Int64.of_int v) s in
  (* GFX12 reaches system memory MTYPE_UC whatever the mapping asks, as the
     kernel's GART maps the queues and words of VMID 0, which every table of
     this library serves (gmc_v12_0_gart_init). Its MTYPE_NC is for other VMIDs'
     system pages (gmc_v12_0_get_vm_pte). *)
  let mtype =
    match gc with
    | (12 | 13), _, _ when system -> D.soc24_mtype_uc
    | _ -> if uncached then mtype_uc gc else 0
  in
  let base =
    D.amdgpu_pte_valid
    |: flag system D.amdgpu_pte_system
    |: flag snooped D.amdgpu_pte_snooped
    |: shift (fragment land 0x1f) D.amdgpu_pte_frag_shift
    |: flag (not table)
         D.(
           amdgpu_pte_writeable |: amdgpu_pte_readable |: amdgpu_pte_executable)
    |: Int64.of_int pa
  in
  (* A page above the leaf level is a page directory entry marked as a page. *)
  let leaf_of_pd = (not table) && level <> D.amdgpu_vm_ptb in
  match gc with
  | (12 | 13), _, _ ->
      base
      |: shift mtype D.amdgpu_pte_mtype_gfx12_shift
      |:
      if leaf_of_pd then D.amdgpu_pde_pte_gfx12
      else flag (not table) D.amdgpu_pte_is_pte
  | (10 | 11), _, _ ->
      base
      |: shift mtype D.amdgpu_pte_mtype_nv10_shift
      |: flag leaf_of_pd D.amdgpu_pde_pte
  | _ ->
      (* GFX9 walks PDB0's entries as pages unless marked to translate further,
         and PDB1's as covering 2^9 PDB0 entries. *)
      base
      |: shift mtype D.amdgpu_pte_mtype_vg10_shift
      |: flag
           (table && level = D.amdgpu_vm_pdb1)
           (shift 0x9 D.amdgpu_pde_bfs_shift)
      |: flag (table && level = D.amdgpu_vm_pdb0) D.amdgpu_pte_tf
      |: flag (leaf_of_pd && level <> D.amdgpu_vm_pdb0) D.amdgpu_pde_pte
