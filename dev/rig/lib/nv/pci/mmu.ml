(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_pci

type version = V2 | V3

let version = function Chip.Ampere | Ada -> V2 | Blackwell -> V3

let levels = function
  | V2 -> [ 12; 21; 29; 38; 47 ]
  | V3 -> [ 12; 21; 29; 38; 47; 56 ]

let bits = function V2 -> 49 | V3 -> 57

(* Entries *)

(* [put (lo, n) x e] is [e] with its [n]-bit field from bit [lo] set to [x]; a
   field of a dual entry's high half lies at bits 64 and up. *)
let put (lo, n) x e =
  let lo = if lo >= 64 then lo - 64 else lo in
  let mask = Int64.(pred (shift_left 1L n)) in
  Int64.(logor e (shift_left (logand (of_int x) mask) lo))

let page = 12

let pte v ~pa (target : Page_table.target) ~uncached ~snooped =
  let a = pa lsr page in
  let system =
    if snooped then Defs.nv_mmu_ver2_pte_aperture_system_coherent_memory
    else Defs.nv_mmu_ver2_pte_aperture_system_non_coherent_memory
  in
  match v with
  | V2 ->
      let where e =
        match target with
        | Gpu ->
            e
            |> put Defs.nv_mmu_ver2_pte_aperture
                 Defs.nv_mmu_ver2_pte_aperture_video_memory
            |> put Defs.nv_mmu_ver2_pte_address_vid a
        | Peer i ->
            e
            |> put Defs.nv_mmu_ver2_pte_aperture
                 Defs.nv_mmu_ver2_pte_aperture_peer_memory
            |> put Defs.nv_mmu_ver2_pte_address_vid a
            |> put Defs.nv_mmu_ver2_pte_address_vid_peer i
        | System ->
            e
            |> put Defs.nv_mmu_ver2_pte_aperture system
            |> put Defs.nv_mmu_ver2_pte_address_sys a
      in
      0L
      |> put Defs.nv_mmu_ver2_pte_valid 1
      |> where
      |> put Defs.nv_mmu_ver2_pte_vol (Bool.to_int uncached)
      |> put Defs.nv_mmu_ver2_pte_kind Defs.nv_mmu_pte_kind_generic_memory
  | V3 ->
      let pcf =
        if uncached then Defs.nv_mmu_ver3_pte_pcf_regular_rw_atomic_uncached_ace
        else Defs.nv_mmu_ver3_pte_pcf_regular_rw_atomic_cached_ace
      in
      let where e =
        match target with
        | Gpu ->
            e
            |> put Defs.nv_mmu_ver3_pte_aperture
                 Defs.nv_mmu_ver3_pte_aperture_video_memory
            |> put Defs.nv_mmu_ver3_pte_address_vid a
        | Peer i ->
            e
            |> put Defs.nv_mmu_ver3_pte_aperture
                 Defs.nv_mmu_ver3_pte_aperture_peer_memory
            |> put Defs.nv_mmu_ver3_pte_address_peer a
            |> put Defs.nv_mmu_ver3_pte_peer_id i
        | System ->
            e
            |> put Defs.nv_mmu_ver3_pte_aperture system
            |> put Defs.nv_mmu_ver3_pte_address_sys a
      in
      0L
      |> put Defs.nv_mmu_ver3_pte_valid 1
      |> where
      |> put Defs.nv_mmu_ver3_pte_pcf pcf
      |> put Defs.nv_mmu_ver3_pte_kind Defs.nv_mmu_pte_kind_generic_memory

(* Tables live in the GPU's memory. *)
let pde v ~child =
  let a = child lsr page in
  match v with
  | V2 ->
      0L
      |> put Defs.nv_mmu_ver2_pde_aperture
           Defs.nv_mmu_ver2_pde_aperture_video_memory
      |> put Defs.nv_mmu_ver2_pde_no_ats 1
      |> put Defs.nv_mmu_ver2_pde_address_vid a
  | V3 ->
      0L
      |> put Defs.nv_mmu_ver3_pde_aperture
           Defs.nv_mmu_ver3_pde_aperture_video_memory
      |> put Defs.nv_mmu_ver3_pde_pcf
           Defs.nv_mmu_ver3_pde_pcf_valid_cached_ats_not_allowed
      |> put Defs.nv_mmu_ver3_pde_address a

let dual v = function
  | `None -> (0L, 0L)
  | `Page e -> (e, 0L)
  | `Table child -> (
      let a = child lsr page in
      match v with
      | V2 ->
          ( put Defs.nv_mmu_ver2_dual_pde_no_ats 1 0L,
            0L
            |> put Defs.nv_mmu_ver2_dual_pde_aperture_small
                 Defs.nv_mmu_ver2_dual_pde_aperture_small_video_memory
            |> put Defs.nv_mmu_ver2_dual_pde_address_small_vid a )
      | V3 ->
          ( 0L,
            0L
            |> put Defs.nv_mmu_ver3_dual_pde_aperture_small
                 Defs.nv_mmu_ver3_dual_pde_aperture_small_video_memory
            |> put Defs.nv_mmu_ver3_dual_pde_pcf_small
                 Defs
                 .nv_mmu_ver3_dual_pde_pcf_small_valid_cached_ats_not_allowed
            |> put Defs.nv_mmu_ver3_dual_pde_address_small a ))

(* The format *)

(* How long the GPU may take to invalidate its TLBs: nouveau's tu102_vmm_flush
   waits 2 seconds for the trigger to clear. *)
let invalidate_ms = 2_000

let format (c : Chip.t) bar : Page_table.format =
  let v = version c.family in
  let levels = levels v in
  let n = List.length levels in
  (* The dual level is the one above the leaf; pages map at the leaf, the dual
     level and the one above it (4 KiB, 2 MiB, 512 MiB). *)
  let dual_level = n - 2 in
  let set64 table i e = Window.set64 bar (table + (8 * i)) e in
  let set_dual table i (lo, hi) =
    Window.set64 bar (table + (16 * i)) lo;
    Window.set64 bar (table + (16 * i) + 8) hi
  in
  let set_table ~level ~table i ~child =
    if level = dual_level then set_dual table i (dual v (`Table child))
    else set64 table i (pde v ~child)
  in
  let set_page ~level ~table i ~pa target ~uncached ~snooped ~fragment:_ =
    let e = pte v ~pa target ~uncached ~snooped in
    if level = dual_level then set_dual table i (dual v (`Page e))
    else set64 table i e
  in
  let clear ~level ~table i =
    if level = dual_level then set_dual table i (dual v `None)
    else set64 table i 0L
  in
  let invalidate =
    List.fold_left
      (fun x (lo, _) -> x lor (1 lsl lo))
      0
      Defs.
        [
          nv_virtual_function_priv_mmu_invalidate_all_va;
          nv_virtual_function_priv_mmu_invalidate_all_pdb;
          nv_virtual_function_priv_mmu_invalidate_sys_membar;
          nv_virtual_function_priv_mmu_invalidate_trigger;
        ]
  in
  let flush () =
    (* The entries' stores reach the GPU's memory before the invalidation makes
       it walk them. *)
    Window.flush bar;
    let r = Defs.nv_virtual_function_priv_mmu_invalidate in
    Chip.set c r invalidate;
    let cleared () =
      Chip.field Defs.nv_virtual_function_priv_mmu_invalidate_trigger
        (Chip.get c r)
      = 0
    in
    match Chip.wait c "the TLB invalidation" ~ms:invalidate_ms cleared with
    | Ok () -> ()
    | Error why -> raise (Rig_nv.Fault why)
  in
  {
    levels;
    bits = bits v;
    first = 0;
    set_table;
    set_page;
    clear;
    large = (fun ~level -> level >= n - 3);
    zero = (fun pa len -> Window.fill bar pa len '\000');
    flush;
  }
