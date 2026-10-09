(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type version = V2 | V3

let levels = function
  | V2 -> [ 12; 21; 29; 38; 47 ]
  | V3 -> [ 12; 21; 29; 38; 47; 56 ]

let bits = function V2 -> 49 | V3 -> 57

(* Pages map at the leaf, the dual level and the one above it. *)
let pages v =
  List.filteri (fun i _ -> i < 3) (levels v)
  |> List.rev_map (fun b -> (1 lsl b, 1 lsl b))

(* [put (lo, n) x e] is [e] with its [n]-bit field from bit [lo] set to [x]; a
   field of a dual entry's high half lies at bits 64 and up. *)
let put (lo, n) x e =
  let lo = if lo >= 64 then lo - 64 else lo in
  let mask = Int64.(pred (shift_left 1L n)) in
  Int64.(logor e (shift_left (logand (of_int x) mask) lo))

let page = 12

(* An entry holds a system page's frame number: 46 bits on V2, 40 on V3. *)
let pa_bits v =
  page
  + snd
      (match v with
      | V2 -> Defs.nv_mmu_ver2_pte_address_sys
      | V3 -> Defs.nv_mmu_ver3_pte_address_sys)

type target = Gpu | Peer of int | System of { snooped : bool }

let pte v ~pa target ~uncached =
  let a = pa lsr page in
  let system snooped =
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
        | System { snooped } ->
            e
            |> put Defs.nv_mmu_ver2_pte_aperture (system snooped)
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
        | System { snooped } ->
            e
            |> put Defs.nv_mmu_ver3_pte_aperture (system snooped)
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
