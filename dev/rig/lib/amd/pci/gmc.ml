(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Window = Rig_pci.Window
module Page_table = Rig_pci.Page_table

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* A hub's invalidation engine 17 on one instance, its registers resolved: the
   request, the acknowledgement, and on the MM hub the semaphore and, from GC
   11, the L2's private cache invalidation bit. *)
type engine = {
  req : int;
  ack : int;
  sem : int option;
  cid2 : (int * int) option; (* the register and its bit *)
}

type t = {
  r : Regs.t;
  vram : Window.t;
  gc : Discovery.version;
  xccs : int list; (* the GC hub's instances *)
  mm : int list; (* the MM hub's instances *)
  node : int; (* where the GPU's memory starts in the fabric's window *)
  fabric_base : int; (* where its page tables name it *)
  mc_base : int; (* where it starts for the memory controller *)
  base : int; (* the FB location registers *)
  top : int;
  hive : bool;
  address_mask : int; (* the physical addresses the GPU's entries hold *)
  mutable gc_started : bool; (* whether the GC hub walks tables yet *)
  engines : ([ `Gc | `Mm ] * (int * engine list)) list;
      (* each hub's request value and engines *)
}

let hub_prefix = function `Gc -> "GC" | `Mm -> "MM"

(* Apertures *)

(* The memory controller's locations and the fabric's segments are in units of
   16 MiB. *)
let aperture_unit = 24
let aperture_mask = 0xff_ffff

(* The memory controller's window on the GPU's memory: its first byte, and its
   last, the end of its top unit. *)
let first base = (base land aperture_mask) lsl aperture_unit
let last top = (((top land aperture_mask) + 1) lsl aperture_unit) - 1

(* GC 9.4 and 9.5 address 48 bits of physical memory; the others 44. *)
let address_bits = function 9, (4 | 5), _ -> 48 | _ -> 44

let make r vram =
  let l = Regs.layout_of r in
  let gc = Regs.version l D.gc_hwid in
  let lfb = Regs.has l "regGCMC_VM_XGMI_LFB_CNTL" in
  let lfb_field reg f = if lfb then Regs.field r reg f else 0 in
  let segment =
    lfb_field "regGCMC_VM_XGMI_LFB_SIZE" "pf_lfb_size" lsl aperture_unit
  in
  let region = lfb_field "regGCMC_VM_XGMI_LFB_CNTL" "pf_lfb_region" in
  let regions = lfb_field "regGCMC_VM_XGMI_LFB_CNTL" "pf_max_region" in
  let d = Regs.discovery l in
  let nbio79 =
    match Discovery.version d D.nbif_hwid with
    | Some (7, 9, (0 | 1)) -> true
    | _ -> false
  in
  (* The GPU's memory starts at its node's segment in the fabric's window, and
     its page tables name it past the memory controller's offset, which a
     virtual function's tables leave out (the kernel's vram_base_offset). *)
  let node = region * segment in
  let offset =
    let name =
      if gc < (10, 0, 0) then "regMC_VM_FB_OFFSET" else "regMMMC_VM_FB_OFFSET"
    in
    if Regs.vf r then 0 else Regs.read r name lsl aperture_unit
  in
  let base = Regs.read r "regMMMC_VM_FB_LOCATION_BASE" in
  let xccs = List.init (Regs.gpu l).xccs Fun.id in
  let mm =
    if nbio79 then Discovery.aids d
    else List.map fst (Discovery.live d D.mmhub_hwid)
  in
  let engines hub insts =
    let name s = strf "reg%sVM_INVALIDATE_ENG17_%s" (hub_prefix hub) s in
    let req =
      Rig_amd_abi.Register.encode
        (Regs.register l (name "REQ"))
        [
          ("flush_type", 0);
          ("per_vmid_invalidate_req", 1);
          ("invalidate_l2_ptes", 1);
          ("invalidate_l2_pde0", 1);
          ("invalidate_l2_pde1", 1);
          ("invalidate_l2_pde2", 1);
          ("invalidate_l1_ptes", 1);
          ("clear_protection_fault_status_addr", 0);
        ]
    in
    (* From GC 11, the L2's private cache is invalidated through a bit of its
       own. *)
    let cid2 inst =
      let name = "regMMVM_L2_BANK_SELECT_RESERVED_CID2" in
      let bit =
        Rig_amd_abi.Register.encode (Regs.register l name)
          [ ("reserved_cache_private_invalidation", 1) ]
      in
      (Regs.address ~inst l name, bit)
    in
    let engine inst =
      let at = Regs.address ~inst l in
      let mm = hub = `Mm in
      {
        req = at (name "REQ");
        ack = at (name "ACK");
        sem = (if mm then Some (at (name "SEM")) else None);
        cid2 = (if mm && gc >= (11, 0, 0) then Some (cid2 inst) else None);
      }
    in
    (hub, (req, List.map engine insts))
  in
  {
    r;
    vram;
    gc;
    xccs;
    mm;
    node;
    fabric_base = offset + node;
    mc_base = first base + node;
    base;
    top = Regs.read r "regMMMC_VM_FB_LOCATION_TOP";
    (* A segment size alone describes an address window; a fabric of GPUs also
       has regions. *)
    hive = segment > 0 && regions > 0;
    address_mask = (1 lsl address_bits gc) - 1;
    gc_started = false;
    engines = [ engines `Gc xccs; engines `Mm mm ];
  }

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

let covers g ~memory = window ~base:g.base ~top:g.top ~fabric:g.node ~memory
let mc g pa = g.mc_base + pa
let fabric g pa = g.fabric_base + pa
let hive g = g.hive
let instances g = function `Gc -> g.xccs | `Mm -> g.mm

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
     this library serves (gmc_v12_0_gart_init). Its MTYPE_NC is for other
     VMIDs' system pages (gmc_v12_0_get_vm_pte). *)
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

(* The tables of 512 entries index bits 12, 21, 30 and 39 of a 48-bit address,
   from PDB2, the root, to PTB. *)
let levels = [ 12; 21; 30; 39 ]
let bits = 48

let format g ~flush =
  let set ~table i e = Window.set64 g.vram (table + (8 * i)) e in
  let at pa =
    if pa land g.address_mask <> pa then
      invalid_argf "Gmc.format: address 0x%x is beyond the GPU's %d bits" pa
        (address_bits g.gc);
    pa
  in
  {
    Page_table.levels;
    bits;
    first = D.amdgpu_vm_pdb2;
    set_table =
      (fun ~level ~table i ~child ->
        set ~table i
          (entry ~gc:g.gc ~level
             ~pa:(at (fabric g child))
             `Table ~uncached:false ~snooped:false ~fragment:0));
    set_page =
      (fun ~level ~table i ~pa target ~uncached ~snooped ~fragment ->
        let pa, kind =
          match target with
          | Page_table.Gpu -> (fabric g pa, `Gpu)
          | Peer _ -> (pa, `Gpu)
          | System -> (pa, `System)
        in
        set ~table i
          (entry ~gc:g.gc ~level ~pa:(at pa) (`Page kind) ~uncached ~snooped
             ~fragment));
    clear = (fun ~level:_ ~table i -> set ~table i 0L);
    (* Pages map at PDB1 (1 GiB), PDB0 (2 MiB) and PTB (4 KiB). *)
    large = (fun ~level -> level >= D.amdgpu_vm_pdb1);
    zero = (fun pa n -> Window.fill g.vram pa n '\000');
    flush;
  }

(* Hubs *)

(* The hubs translate up to the canonical end of a 48-bit address space. *)
let vm_last = 0x7fff_ffff_ffff

(* GFX9 walks one level more ("translate further"), with tables of 2^9 2-MiB
   blocks. *)
let further g = g.gc < (10, 0, 0)

(* The faults every context reports: each raises an interrupt and is redirected
   to the default page. *)
let faults =
  [ "pde0"; "dummy_page"; "range"; "valid"; "read"; "write"; "execute" ]

let start_hub g hub tables ~scratch =
  let r = g.r and ip = hub_prefix hub in
  let reg s = strf "reg%s%s" ip s in
  let ctx s = strf "reg%sVM_CONTEXT0_%s" ip s in
  let base = Page_table.base tables in
  List.iter
    (fun inst ->
      let w ?value name fs = Regs.write ~inst ?value r name fs in
      let w64 name ~lo ~hi v = Regs.write64 ~inst r name ~lo ~hi v in
      w ~value:0 (reg "MC_VM_AGP_BASE") [];
      w ~value:(0xffff_ffff_ffff lsr aperture_unit) (reg "MC_VM_AGP_BOT") [];
      w ~value:0 (reg "MC_VM_AGP_TOP") [];
      w ~value:(first g.base lsr 18) (reg "MC_VM_SYSTEM_APERTURE_LOW_ADDR") [];
      w ~value:(last g.top lsr 18) (reg "MC_VM_SYSTEM_APERTURE_HIGH_ADDR") [];
      w64
        (reg "MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR")
        ~lo:"_LSB" ~hi:"_MSB" (scratch lsr 12);
      Regs.update ~inst r
        (reg "VM_L2_PROTECTION_FAULT_CNTL2")
        [ ("active_page_migration_pte_read_retry", 1) ];
      Regs.update ~inst r
        (reg "MC_VM_MX_L1_TLB_CNTL")
        [
          ("enable_l1_tlb", 1);
          ("system_access_mode", 3);
          ("enable_advanced_driver_model", 1);
          ("system_aperture_unmapped_access", 0);
          ("mtype", mtype_uc g.gc);
        ];
      Regs.update ~inst r (reg "VM_L2_CNTL")
        [
          ("enable_l2_cache", 1);
          ("enable_default_page_out_to_system_memory", 1);
          ("l2_pde0_cache_tag_generation_mode", 0);
          ("pde_fault_classification", 0);
          ("context1_identity_access_mode", 1);
          ("identity_mode_fragment_size", 0);
          ("enable_l2_fragment_processing", Bool.to_int (further g));
        ];
      Regs.update ~inst r (reg "VM_L2_CNTL2")
        [ ("invalidate_all_l1_tlbs", 1); ("invalidate_l2_cache", 1) ];
      w (reg "VM_L2_CNTL3")
        [
          ("l2_cache_4k_associativity", 1);
          ("l2_cache_bigk_associativity", 1);
          ("bank_select", if further g then 12 else 9);
          ("l2_cache_bigk_fragment_size", if further g then 9 else 6);
        ];
      w (reg "VM_L2_CNTL4") [ ("l2_cache_4k_partition_count", 1) ];
      if g.gc >= (10, 0, 0) then
        w (reg "VM_L2_CNTL5") [ ("walker_priority_client_id", 0x1ff) ];
      w64 (ctx "PAGE_TABLE_START_ADDR") ~lo:"_LO32" ~hi:"_HI32" (base lsr 12);
      w64
        (ctx "PAGE_TABLE_END_ADDR")
        ~lo:"_LO32" ~hi:"_HI32"
        (min (base + Page_table.span tables - 1) vm_last lsr 12);
      w64
        (ctx "PAGE_TABLE_BASE_ADDR")
        ~lo:"_LO32" ~hi:"_HI32"
        (fabric g (Page_table.root tables) lor 1);
      w (ctx "CNTL")
        (List.map
           (fun f -> (f ^ "_protection_fault_enable_interrupt", 1))
           faults
        @ List.map (fun f -> (f ^ "_protection_fault_enable_default", 1)) faults
        @ [
            ("enable_context", 1);
            ("page_table_depth", (if further g then 2 else 3) - D.amdgpu_vm_pdb2);
            ("page_table_block_size", if further g then 9 else 0);
          ]);
      w64
        (reg "VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR")
        ~lo:"_LO32" ~hi:"_HI32" 0xf_ffff_ffff;
      w64
        (reg "VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR")
        ~lo:"_LO32" ~hi:"_HI32" 0;
      w64
        (reg "VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET")
        ~lo:"_LO32" ~hi:"_HI32" 0;
      for eng = 0 to 17 do
        w64
          (reg (strf "VM_INVALIDATE_ENG%d_ADDR_RANGE" eng))
          ~lo:"_LO32" ~hi:"_HI32" 0x1f_ffff_ffff
      done)
    (instances g hub);
  if hub = `Gc then g.gc_started <- true

(* A hub translates once its L2 cache and context 0 are on, over the tables'
   root: a block's reset clears them, as the RLC's autoload does the GC's, and
   the hub then passes addresses through untranslated. The reset also clears
   the faults' page, which then sends them to host address 0, and the system
   aperture and its access mode. *)
let translates g hub tables ~fault =
  let r = g.r and ip = hub_prefix hub in
  let reg s = strf "reg%s%s" ip s in
  let read64 ~inst name =
    Regs.read ~inst r (reg (name ^ "_LO32"))
    lor (Regs.read ~inst r (reg (name ^ "_HI32")) lsl 32)
  in
  let wrong inst =
    let field name f = Regs.field ~inst r (reg name) f in
    let checks =
      [
        ("its L2 cache", field "VM_L2_CNTL" "enable_l2_cache", 1);
        ("its context 0", field "VM_CONTEXT0_CNTL" "enable_context", 1);
        ( "its table base",
          read64 ~inst "VM_CONTEXT0_PAGE_TABLE_BASE_ADDR",
          fabric g (Page_table.root tables) lor 1 );
        ( "its faults' page",
          read64 ~inst "VM_L2_PROTECTION_FAULT_DEFAULT_ADDR",
          fault lsr 12 );
        ( "its system access mode",
          field "MC_VM_MX_L1_TLB_CNTL" "system_access_mode",
          3 );
        ( "its system aperture's low end",
          Regs.read ~inst r (reg "MC_VM_SYSTEM_APERTURE_LOW_ADDR"),
          first g.base lsr 18 );
        ( "its system aperture's high end",
          Regs.read ~inst r (reg "MC_VM_SYSTEM_APERTURE_HIGH_ADDR"),
          last g.top lsr 18 );
      ]
    in
    List.find_map
      (fun (what, got, want) ->
        if got = want then None
        else
          Some
            (strf
               "the %s hub (instance %d) does not translate: %s reads 0x%x \
                where 0x%x was written"
               ip inst what got want))
      checks
  in
  match List.find_map wrong (instances g hub) with
  | None -> Ok ()
  | Some why -> Error why

let fault_page g hub a =
  let name = strf "reg%sVM_L2_PROTECTION_FAULT_DEFAULT_ADDR" (hub_prefix hub) in
  List.iter
    (fun inst -> Regs.write64 ~inst g.r name ~lo:"_LO32" ~hi:"_HI32" (a lsr 12))
    (instances g hub)

(* Flushes *)

let hdp g =
  if Regs.vf g.r then
    4
    * Regs.address (Regs.layout_of g.r)
        "regBIF_BX_DEV0_EPF0_VF0_HDP_MEM_COHERENCY_FLUSH_CNTL"
  else Soc.hdp_flush g.r

(* A store to the HDP flush register flushes, and its read back returns once the
   flush is done, as amdgpu_hdp_generic_flush does. *)
let flush_hdp g =
  let a = hdp g / 4 in
  Regs.set g.r a 0;
  ignore (Regs.get g.r a)

let invalidate g =
  let r = g.r in
  let invalidate hub =
    let req, engines = List.assoc hub g.engines in
    List.iter
      (fun e ->
        Option.iter
          (fun sem ->
            Regs.wait r "the MM hub's invalidation semaphore" (fun () ->
                Regs.get r sem land 1 = 1))
          e.sem;
        Regs.set r e.req req;
        Regs.wait r
          (strf "the %s hub's invalidation" (hub_prefix hub))
          (fun () -> Regs.get r e.ack land 1 = 1);
        Option.iter (fun sem -> Regs.set r sem 0) e.sem;
        Option.iter
          (fun (cid2, bit) ->
            Regs.set r cid2 (Regs.get r cid2 lor bit);
            ignore (Regs.get r cid2))
          e.cid2)
      engines
  in
  if g.gc_started then invalidate `Gc;
  invalidate `Mm

let fault g =
  let r = g.r in
  let status =
    if Regs.has (Regs.layout_of r) "regGCVM_L2_PROTECTION_FAULT_STATUS_LO32"
    then "regGCVM_L2_PROTECTION_FAULT_STATUS_LO32"
    else "regGCVM_L2_PROTECTION_FAULT_STATUS"
  in
  let fields = Regs.fields r status in
  let page =
    (Regs.read r "regGCVM_L2_PROTECTION_FAULT_ADDR_HI32" lsl 32)
    lor Regs.read r "regGCVM_L2_PROTECTION_FAULT_ADDR_LO32"
  in
  Regs.update r "regGCVM_L2_PROTECTION_FAULT_CNTL"
    [ ("clear_protection_fault_status_addr", 1) ];
  strf "page fault at 0x%x (%s)" (page lsl 12)
    (String.concat ", "
       (List.filter_map
          (fun (f, v) -> if v <> 0 then Some (strf "%s=%d" f v) else None)
          fields))
