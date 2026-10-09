(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_pci

let version : Chip.family -> Page_entry.version = function
  | Ampere | Ada -> V2
  | Blackwell -> V3

(* The format *)

(* How long the GPU may take to invalidate its TLBs: nouveau's tu102_vmm_flush
   waits 2 seconds for the trigger to clear. *)
let invalidate_ms = 2_000

let format (c : Chip.t) bar ~failed : Page_table.format =
  let v = version c.family in
  let levels = Page_entry.levels v in
  let n = List.length levels in
  (* The dual level is the one above the leaf. *)
  let dual_level = n - 2 in
  let set64 table i e = Window.set64 bar (table + (8 * i)) e in
  let set_dual table i (lo, hi) =
    Window.set64 bar (table + (16 * i)) lo;
    Window.set64 bar (table + (16 * i) + 8) hi
  in
  let set_table ~level ~table i ~child =
    if level = dual_level then
      set_dual table i (Page_entry.dual v (`Table child))
    else set64 table i (Page_entry.pde v ~child)
  in
  let set_page ~level ~table i ~pa target ~uncached ~snooped ~fragment:_ =
    let target : Page_entry.target =
      match (target : Page_table.target) with
      | Gpu -> Gpu
      | Peer i -> Peer i
      | System -> System { snooped }
    in
    let e = Page_entry.pte v ~pa target ~uncached in
    if level = dual_level then set_dual table i (Page_entry.dual v (`Page e))
    else set64 table i e
  in
  let clear ~level ~table i =
    if level = dual_level then set_dual table i (Page_entry.dual v `None)
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
    match
      Rig_pci.Function.wait c.fn ~us:(invalidate_ms * 1000)
        "the TLB invalidation" cleared
    with
    | Ok () -> true
    | Error why ->
        failed why;
        false
  in
  {
    levels;
    bits = Page_entry.bits v;
    pa_bits = Page_entry.pa_bits v;
    first = 0;
    set_table;
    set_page;
    clear;
    large = (fun ~level -> level >= n - 3);
    zero = (fun pa len -> Window.fill bar pa len '\000');
    flush;
  }
