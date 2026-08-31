(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver-less NVIDIA device core, on pieces that run without
   hardware: register resolution over the included families, named
   bitfield access against hand-computed words, chip detection and the
   armed-firmware-region recovery over scripted register read-backs,
   VRAM sizing, page-table entry encoding for both format generations,
   the memory manager's regions and mappings, and boot-memory
   allocation over an anonymous mapping standing in for the VRAM
   BAR. *)

open Windtrap
module Nvdev = Tolk_nv.Nvdev
module Nv_reg = Nvdev.Nv_reg
module Nv_page_table = Nvdev.Nv_page_table
module Memory = Tolk.Memory
module Mmio = Tolk_hcq.Hcq.Mmio
module File_io = Tolk_hcq.Hcq.File_io

let with_fake_vram size f =
  let addr =
    File_io.mmap ~addr:0n ~size
      ~prot:(File_io.prot_read lor File_io.prot_write)
      ~flags:(File_io.map_private lor File_io.map_anonymous)
      ~fd:(-1) ~offset:0L
  in
  Fun.protect
    ~finally:(fun () -> File_io.munmap addr ~size)
    (fun () -> f (Mmio.make ~addr ~size))

(* Register byte addresses the scripts key on. *)
let boot_0_addr = 0x0
let boot_42_addr = 0xa00
let wpr2_hi_addr = 0x1fa828
let scratch_42_addr = 0x1183a4

(* A scripted device: a register file over a hashtable with per-address
   read hooks and a write log, an anonymous mapping standing in for the
   VRAM BAR, counters for the PCI actions, and a clock that advances one
   millisecond per reading so the reset settle delay consumes no wall
   time. The default 80MB of VRAM leaves the memory manager 16MB past
   its 64MB tail reservation: a 2MB boot region and 14MB of main
   memory, whose first allocation lands at 0x200000. *)

type fake = {
  dev : Nvdev.t;
  fvram : Mmio.t;
  store : (int, int) Hashtbl.t;
  log : (int * int) list ref;
  resets : int ref;
  cfg_writes : (int * int * int) list ref;
  clock : int ref;
  sys_reqs : (bool * int) list ref;
}

let with_fake_dev ?(arch = 0x17) ?(impl = 2) ?(boot0 = 0x174000a1)
    ?(scratch_mb = 80) ?(vram_bytes = 0x5000000) ?(bar1_base = 0x40000000)
    ?(cfg = 0x7) ?(pre = fun ~reads:_ ~resets:_ -> ()) f =
  with_fake_vram vram_bytes (fun fvram ->
      let store = Hashtbl.create 16 in
      let reads = Hashtbl.create 16 in
      let log = ref [] in
      let resets = ref 0 in
      let cfg_writes = ref [] in
      let cfg_val = ref cfg in
      let clock = ref 0 in
      let sys_reqs = ref [] in
      Hashtbl.replace store boot_0_addr boot0;
      Hashtbl.replace store boot_42_addr ((arch lsl 24) lor (impl lsl 20));
      Hashtbl.replace store scratch_42_addr scratch_mb;
      pre ~reads ~resets;
      let rreg addr =
        match Hashtbl.find_opt reads addr with
        | Some hook -> hook ()
        | None -> Option.value ~default:0 (Hashtbl.find_opt store addr)
      in
      let wreg addr v =
        log := (addr, v) :: !log;
        Hashtbl.replace store addr v
      in
      let dev =
        Nvdev.make
          ~read_config:(fun ~offset:_ ~size:_ -> !cfg_val)
          ~write_config_flush:(fun ~offset ~value ~size ->
            cfg_writes := (offset, value, size) :: !cfg_writes;
            cfg_val := value)
          ~reset:(fun () -> incr resets)
          ~now_ms:(fun () ->
            incr clock;
            !clock)
          ~alloc_sysmem:(fun ~contiguous size ->
            sys_reqs := (contiguous, size) :: !sys_reqs;
            (Mmio.view fvram ~off:0x100000 ~size:0x1000 (), [ 0xaa000; 0xab000 ]))
          ~bar1_base ~rreg ~wreg
          ~mmio:(Mmio.view fvram ~off:0 ~size:0x1000 ())
          ~vram:fvram ~devfmt:"test" ()
      in
      f { dev; fvram; store; log; resets; cfg_writes; clock; sys_reqs })

(* The chronological write log. *)
let writes fd = List.rev !(fd.log)
let cfg_writes fd = List.rev !(fd.cfg_writes)

(* Page-table operations resolved against the device's own entry
   descriptors, building tables in a scratch corner of the fake VRAM
   clear of the manager's regions. *)
let pt_ops fd =
  let ver = Printf.sprintf "NV_MMU_VER%d" (Nvdev.mmu_ver fd.dev) in
  Nv_page_table.ops ~vram:fd.fvram ~mmu_ver:(Nvdev.mmu_ver fd.dev)
    ~pte:(Nvdev.reg fd.dev (ver ^ "_PTE"))
    ~pde:(Nvdev.reg fd.dev (ver ^ "_PDE"))
    ~dual_pde:(Nvdev.reg fd.dev (ver ^ "_DUAL_PDE"))
    ()

(* The MMU-invalidate poke issued after every mapping: all_va, all_pdb,
   sys_membar and trigger, at dev_vm's 0xb80000 + 0x30b0. *)
let invalidate_write = (0xb830b0, 0x80000043)

let () =
  run "Nvdev"
    [
      group "register resolution"
        [
          test "resolves one register per included family" (fun () ->
              with_fake_dev (fun fd ->
                  let a name = Nv_reg.addr (Nvdev.reg fd.dev name) in
                  equal int 0x0 (a "NV_PMC_BOOT_0");
                  equal int wpr2_hi_addr (a "NV_PFB_PRI_MMU_WPR2_ADDR_HI");
                  equal int scratch_42_addr
                    (a "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42");
                  (* dev_vm registers sit in the virtual-function block. *)
                  equal int 0xb830b0
                    (a "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE")));
          test "indexed registers resolve affine addresses" (fun () ->
              with_fake_dev (fun fd ->
                  let r =
                    Nvdev.reg fd.dev "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05"
                  in
                  equal int 0x118234 (Nv_reg.addr (Nv_reg.with_idx r 0));
                  equal int 0x118240 (Nv_reg.addr (Nv_reg.with_idx r 3));
                  raises_match
                    (Exn.invalid_arg ~substring:"apply with_idx")
                    (fun () -> Nv_reg.addr r);
                  raises_match
                    (Exn.invalid_arg ~substring:"is not indexed")
                    (fun () ->
                      Nv_reg.with_idx (Nvdev.reg fd.dev "NV_PMC_BOOT_0") 1);
                  let leaf =
                    Nvdev.reg fd.dev "NV_VIRTUAL_FUNCTION_PRIV_CPU_INTR_LEAF"
                  in
                  equal int 0xb81008 (Nv_reg.addr (Nv_reg.with_idx leaf 2))));
          test "with_base rebases block-relative registers" (fun () ->
              with_fake_dev (fun fd ->
                  let r = Nvdev.reg fd.dev "NV_PMC_BOOT_0" in
                  equal int 0x110000 (Nv_reg.addr (Nv_reg.with_base r 0x110000))));
          test "later families can be included and override" (fun () ->
              with_fake_dev (fun fd ->
                  raises_match
                    (Exn.invalid_arg ~substring:"has no register")
                    (fun () -> Nvdev.reg fd.dev "NV_PGSP_QUEUE_HEAD");
                  Nvdev.include_regs fd.dev ~family:"dev_gsp" ~arch:"ga102";
                  let head = Nvdev.reg fd.dev "NV_PGSP_QUEUE_HEAD" in
                  equal int 0x110c08 (Nv_reg.addr (Nv_reg.with_idx head 1));
                  equal int 4 (Nvdev.const fd.dev "NV_PGSP_MAILBOX__SIZE_1");
                  raises_match
                    (Exn.invalid_arg ~substring:"has no gh999 table")
                    (fun () ->
                      Nvdev.include_regs fd.dev ~family:"dev_gsp"
                        ~arch:"gh999")));
          test "constants and registers are distinct namespaces" (fun () ->
              with_fake_dev (fun fd ->
                  equal int 0x17
                    (Nvdev.const fd.dev "NV_PMC_BOOT_0_ARCHITECTURE_GA100");
                  equal int 0x170
                    (Nvdev.const fd.dev "NV_PMC_BOOT_42_CHIP_ID_GA100");
                  raises_match
                    (Exn.invalid_arg ~substring:"is a constant, not a register")
                    (fun () ->
                      Nvdev.reg fd.dev "NV_PMC_BOOT_0_ARCHITECTURE_GA100");
                  raises_match
                    (Exn.invalid_arg ~substring:"is a register, not a constant")
                    (fun () -> Nvdev.const fd.dev "NV_PMC_BOOT_0");
                  raises_match
                    (Exn.invalid_arg ~substring:"has no register")
                    (fun () -> Nvdev.const fd.dev "NV_NOPE")));
          test "bitfield descriptors resolve without an address" (fun () ->
              with_fake_dev (fun fd ->
                  let pte = Nvdev.reg fd.dev "NV_MMU_VER2_PTE" in
                  equal
                    (pair int int)
                    (56, 63)
                    (List.assoc "kind" (Nv_reg.fields pte));
                  raises_match
                    (Exn.invalid_arg ~substring:"no address")
                    (fun () -> Nv_reg.addr pte);
                  raises_match
                    (Exn.invalid_arg ~substring:"no address")
                    (fun () -> Nv_reg.with_base pte 0x1000);
                  (* The version-3 tables come with version-3 chips only. *)
                  raises_match
                    (Exn.invalid_arg ~substring:"has no register")
                    (fun () -> Nvdev.reg fd.dev "NV_MMU_VER3_PTE")));
        ];
      group "bitfields"
        [
          test "encode, decode and mask against hand-computed words"
            (fun () ->
              with_fake_dev (fun fd ->
                  let r = Nvdev.reg fd.dev "NV_PMC_BOOT_42" in
                  equal int 0x19400000
                    (Nv_reg.encode r
                       [ ("architecture", 0x19); ("implementation", 4) ]);
                  let d = Nv_reg.decode r 0x19400000 in
                  equal int 0x19 (List.assoc "architecture" d);
                  equal int 4 (List.assoc "implementation" d);
                  equal int 0x194 (List.assoc "chip_id" d);
                  equal int 0x3f000000 (Nv_reg.mask r [ "architecture" ]);
                  raises_match
                    (Exn.invalid_arg ~substring:"has no field")
                    (fun () -> Nv_reg.encode r [ ("nope", 1) ])));
          test "field writes and read-modify-write updates" (fun () ->
              with_fake_dev (fun fd ->
                  let r =
                    Nvdev.reg fd.dev "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE"
                  in
                  Nv_reg.write r [ ("trigger", 1); ("ack", 2) ];
                  equal
                    (list (pair int int))
                    [ (0xb830b0, 0x80000100) ]
                    (writes fd);
                  Hashtbl.replace fd.store 0xb830b0 0xffffffff;
                  Nv_reg.update r [ ("ack", 1) ];
                  equal (pair int int) (0xb830b0, 0xfffffeff)
                    (List.nth (writes fd) 1);
                  equal int 1
                    (List.assoc "ack" (Nv_reg.read_bitfields r))));
          test "a raw value passes through unchanged" (fun () ->
              with_fake_dev (fun fd ->
                  Nvdev.include_regs fd.dev ~family:"dev_gsp" ~arch:"ga102";
                  let r = Nvdev.reg fd.dev "NV_PGSP_FALCON_MAILBOX0" in
                  Nv_reg.write r ~value:0xdeadbeef [];
                  equal (pair int int) (0x110040, 0xdeadbeef)
                    (List.hd (writes fd));
                  Nvdev.wreg fd.dev 0x123 7;
                  equal int 7 (Nvdev.rreg fd.dev 0x123)));
          test "fields past the native int raise and stay readable"
            (fun () ->
              with_fake_dev (fun fd ->
                  let pte = Nvdev.reg fd.dev "NV_MMU_VER2_PTE" in
                  equal int 1 (Nv_reg.encode pte [ ("valid", 1) ]);
                  raises_match
                    (Exn.invalid_arg ~substring:"past the native int")
                    (fun () -> Nv_reg.encode pte [ ("kind", 6) ]);
                  raises_match
                    (Exn.invalid_arg ~substring:"past the native int")
                    (fun () -> Nv_reg.decode pte 5)));
        ];
      group "chip detection"
        [
          test "GA102" (fun () ->
              with_fake_dev ~arch:0x17 ~impl:2 (fun fd ->
                  equal int 0x174000a1 (Nvdev.chip_id fd.dev);
                  equal string "GA102" (Nvdev.chip_name fd.dev);
                  equal string "ga102" (Nvdev.fw_name fd.dev);
                  equal int 2 (Nvdev.mmu_ver fd.dev);
                  equal bool false (Nvdev.fmc_boot fd.dev)));
          test "AD103" (fun () ->
              with_fake_dev ~arch:0x19 ~impl:3 (fun fd ->
                  equal string "AD103" (Nvdev.chip_name fd.dev);
                  equal string "ad102" (Nvdev.fw_name fd.dev);
                  equal int 2 (Nvdev.mmu_ver fd.dev);
                  equal bool false (Nvdev.fmc_boot fd.dev)));
          test "GB202 selects the version-3 page-table format" (fun () ->
              with_fake_dev ~arch:0x1b ~impl:2 (fun fd ->
                  equal string "GB202" (Nvdev.chip_name fd.dev);
                  equal string "gb202" (Nvdev.fw_name fd.dev);
                  equal int 3 (Nvdev.mmu_ver fd.dev);
                  equal bool true (Nvdev.fmc_boot fd.dev);
                  (* The dual-entry descriptor spans two 64-bit words;
                     its raw ranges stay exposed. *)
                  let dual = Nvdev.reg fd.dev "NV_MMU_VER3_DUAL_PDE" in
                  equal
                    (pair int int)
                    (76, 115)
                    (List.assoc "address_small" (Nv_reg.fields dual))));
          test "unknown chips fail naming the architecture" (fun () ->
              raises_match
                (Exn.failure ~substring:"unsupported chip, architecture 0x18")
                (fun () -> with_fake_dev ~arch:0x18 (fun _ -> ())));
        ];
      group "vram"
        [
          test "sizes vram from the scratch register" (fun () ->
              with_fake_dev ~scratch_mb:80 ~vram_bytes:0x5000000 (fun fd ->
                  equal int 0x5000000 (Nvdev.vram_size fd.dev);
                  equal bool true (Nvdev.large_bar fd.dev)));
          test "a bar smaller than vram is not large" (fun () ->
              with_fake_dev ~scratch_mb:128 ~vram_bytes:0x1000000 (fun fd ->
                  equal int 0x8000000 (Nvdev.vram_size fd.dev);
                  equal bool false (Nvdev.large_bar fd.dev)));
        ];
      (* Expected entry words are hand-computed from the descriptor
         field positions in the pinned clone's generated register maps
         (tinygrad/runtime/autogen/nv_regs/dev_mmu.py):
         tu102 NV_MMU_VER2_PTE:468 valid(0) aperture(1,2) vol(3)
         address_sys(8,53) kind(56,63); NV_MMU_VER2_PDE:424
         aperture(1,2) no_ats(5) address_sys(8,53);
         NV_MMU_VER2_DUAL_PDE:442 no_ats(5) aperture_small(65,66)
         address_small_sys(72,117); gh100 NV_MMU_VER3_PTE:834 valid(0)
         aperture(1,2) pcf(3,7) kind(8,11) address_sys(12,51);
         NV_MMU_VER3_PDE:771 aperture(1,2) pcf(3,5) address(12,51);
         NV_MMU_VER3_DUAL_PDE:794 aperture_small(65,66) pcf_small(67,69)
         address_small(76,115). *)
      group "page-table entries"
        [
          test "version-2 page entries" (fun () ->
              with_fake_dev (fun fd ->
                  let ops = pt_ops fd in
                  (* Leaf level of the five-level tree. *)
                  let pt = ops.Memory.make ~paddr:0x10000 ~lv:4 in
                  ops.Memory.set_entry pt ~idx:3 ~paddr:0xabc000 ~valid:true ();
                  equal int64 0x06000000000abc01L
                    (Mmio.read64 fd.fvram (0x10000 + (3 * 8)));
                  equal bool true (ops.Memory.valid pt 3);
                  equal bool true (ops.Memory.is_page pt 3);
                  equal int 0xabc000 (ops.Memory.address pt 3);
                  ops.Memory.set_entry pt ~idx:4 ~paddr:0xabc000
                    ~aspace:Memory.Sys ~uncached:true ~valid:true ();
                  equal int64 0x06000000000abc0dL
                    (Mmio.read64 fd.fvram (0x10000 + (4 * 8)))));
          test "version-2 directory entries" (fun () ->
              with_fake_dev (fun fd ->
                  let ops = pt_ops fd in
                  let pt = ops.Memory.make ~paddr:0x10000 ~lv:1 in
                  ops.Memory.set_entry pt ~idx:0 ~paddr:0x5000 ~table:true
                    ~valid:true ();
                  equal int64 0x522L (Mmio.read64 fd.fvram 0x10000);
                  equal bool true (ops.Memory.valid pt 0);
                  equal bool false (ops.Memory.is_page pt 0);
                  equal int 0x5000 (ops.Memory.address pt 0);
                  (* An invalidated directory entry keeps its address
                     bits; the zero aperture alone makes it invalid. *)
                  ops.Memory.set_entry pt ~idx:1 ~paddr:0x5000 ~table:true
                    ~valid:false ();
                  equal int64 0x520L (Mmio.read64 fd.fvram (0x10000 + 8));
                  equal bool false (ops.Memory.valid pt 1)));
          test "version-2 paired entries at the next-to-leaf level"
            (fun () ->
              with_fake_dev (fun fd ->
                  let ops = pt_ops fd in
                  let pt = ops.Memory.make ~paddr:0x11000 ~lv:3 in
                  ops.Memory.set_entry pt ~idx:1 ~paddr:0x6000 ~table:true
                    ~valid:true ();
                  equal int64 0x20L (Mmio.read64 fd.fvram (0x11000 + 16));
                  equal int64 0x602L (Mmio.read64 fd.fvram (0x11000 + 24));
                  equal bool true (ops.Memory.valid pt 1);
                  equal bool false (ops.Memory.is_page pt 1);
                  equal int 0x6000 (ops.Memory.address pt 1);
                  (* A page at this level is a whole 2MB mapping: the
                     entry is a plain page word in the first slot. *)
                  ops.Memory.set_entry pt ~idx:2 ~paddr:0x200000 ~valid:true ();
                  equal int64 0x0600000000020001L
                    (Mmio.read64 fd.fvram (0x11000 + 32));
                  equal int64 0L (Mmio.read64 fd.fvram (0x11000 + 40));
                  equal bool true (ops.Memory.is_page pt 2);
                  equal bool true (ops.Memory.valid pt 2)));
          test "version-3 entries" (fun () ->
              with_fake_dev ~arch:0x1b (fun fd ->
                  let ops = pt_ops fd in
                  let leaf = ops.Memory.make ~paddr:0x10000 ~lv:5 in
                  ops.Memory.set_entry leaf ~idx:0 ~paddr:0xabc000 ~valid:true
                    ();
                  equal int64 0xabc601L (Mmio.read64 fd.fvram 0x10000);
                  equal int 0xabc000 (ops.Memory.address leaf 0);
                  ops.Memory.set_entry leaf ~idx:1 ~paddr:0xabc000
                    ~aspace:Memory.Sys ~uncached:true ~valid:true ();
                  equal int64 0xabc60dL (Mmio.read64 fd.fvram (0x10000 + 8));
                  let pde = ops.Memory.make ~paddr:0x11000 ~lv:1 in
                  ops.Memory.set_entry pde ~idx:0 ~paddr:0x5000 ~table:true
                    ~valid:true ();
                  equal int64 0x5012L (Mmio.read64 fd.fvram 0x11000);
                  equal int 0x5000 (ops.Memory.address pde 0);
                  (* The paired level puts the whole directory entry in
                     the second word. *)
                  let dual = ops.Memory.make ~paddr:0x12000 ~lv:4 in
                  ops.Memory.set_entry dual ~idx:0 ~paddr:0x6000 ~table:true
                    ~valid:true ();
                  equal int64 0L (Mmio.read64 fd.fvram 0x12000);
                  equal int64 0x6012L (Mmio.read64 fd.fvram (0x12000 + 8));
                  equal bool true (ops.Memory.valid dual 0);
                  equal bool false (ops.Memory.is_page dual 0);
                  equal int 0x6000 (ops.Memory.address dual 0)));
          test "huge pages need a deep-enough level and alignment"
            (fun () ->
              with_fake_dev ~arch:0x1b (fun fd ->
                  let ops = pt_ops fd in
                  let at lv = ops.Memory.make ~paddr:0x10000 ~lv in
                  equal bool false
                    (ops.Memory.supports_huge_page (at 2) ~paddr:0);
                  equal bool true
                    (ops.Memory.supports_huge_page (at 3) ~paddr:(1 lsl 29));
                  equal bool true
                    (ops.Memory.supports_huge_page (at 4) ~paddr:0x200000);
                  equal bool false
                    (ops.Memory.supports_huge_page (at 4) ~paddr:0x1000);
                  equal bool true
                    (ops.Memory.supports_huge_page (at 5) ~paddr:0x1000)));
        ];
      group "memory manager"
        [
          test "the tree geometry follows the format generation" (fun () ->
              with_fake_dev (fun fd ->
                  let mm = Nvdev.mm fd.dev in
                  equal int 5 (Memory.level_cnt mm);
                  equal int 48 (Memory.va_bits mm);
                  equal int 4 (Memory.pte_cnt mm 0);
                  equal int 256 (Memory.pte_cnt mm 3);
                  equal int 0x200000 (Memory.pte_covers mm 3);
                  let root = Memory.root_page_table mm in
                  equal int 0 (Nv_page_table.paddr root);
                  equal int 0 (Nv_page_table.lv root));
              with_fake_dev ~arch:0x1b (fun fd ->
                  let mm = Nvdev.mm fd.dev in
                  equal int 6 (Memory.level_cnt mm);
                  equal int 56 (Memory.va_bits mm);
                  equal int 2 (Memory.pte_cnt mm 0);
                  equal int 256 (Memory.pte_cnt mm 4);
                  equal int 0x200000 (Memory.pte_covers mm 4)));
          test "regions: tail reservation, boot flip, zeroed pallocs"
            (fun () ->
              with_fake_dev (fun fd ->
                  let mm = Nvdev.mm fd.dev in
                  (* 80MB of VRAM less the 64MB tail reservation. *)
                  equal int 0x1000000 (Memory.vram_size mm);
                  equal bool false (Nvdev.is_booting fd.dev);
                  raises_match
                    (Exn.invalid_arg ~substring:"only boot memory")
                    (fun () -> Memory.palloc mm 0x1000 ~boot:true ());
                  (* The main region starts past the 2MB boot region and
                     hands out zeroed pages. *)
                  Mmio.write64 fd.fvram 0x200100 0xdeadbeefL;
                  equal int 0x200000 (Memory.palloc mm 0x1000 ());
                  equal int64 0L (Mmio.read64 fd.fvram 0x200100)));
          test "the virtual window is shared and starts at 64GB" (fun () ->
              equal int 0x1000000000 Nvdev.va_base;
              equal int (1 lsl 44) Nvdev.va_size;
              with_fake_dev (fun fd ->
                  let va = Memory.alloc_vaddr (Nvdev.mm fd.dev) 0x1000 () in
                  equal bool true
                    (va >= Nvdev.va_base && va < Nvdev.va_base + Nvdev.va_size)));
          test "mapping writes exact table images and invalidates the TLBs"
            (fun () ->
              with_fake_dev (fun fd ->
                  let mm = Nvdev.mm fd.dev in
                  let vaddr = 0x1000000000 in
                  let vm =
                    Memory.map_range mm ~vaddr ~size:0x2000
                      [ (0x400000, 0x2000) ]
                      Memory.Phys ()
                  in
                  equal int vaddr vm.Memory.va_addr;
                  equal (list int)
                    [ 0x0; 0x200000; 0x201000; 0x202000; 0x203000 ]
                    (List.map Nv_page_table.paddr
                       (Memory.page_tables mm ~vaddr ~size:0x2000));
                  equal int64 0x20022L (Mmio.read64 fd.fvram 0x0);
                  equal int64 0x20122L (Mmio.read64 fd.fvram 0x200000);
                  equal int64 0x20222L
                    (Mmio.read64 fd.fvram (0x201000 + (128 * 8)));
                  equal int64 0x20L (Mmio.read64 fd.fvram 0x202000);
                  equal int64 0x20302L (Mmio.read64 fd.fvram (0x202000 + 8));
                  equal int64 0x0600000000040001L
                    (Mmio.read64 fd.fvram 0x203000);
                  equal int64 0x0600000000040101L
                    (Mmio.read64 fd.fvram (0x203000 + 8));
                  equal (list (pair int int)) [ invalidate_write ] (writes fd)));
          test "a huge page maps as one paired page entry" (fun () ->
              with_fake_dev ~arch:0x1b (fun fd ->
                  let mm = Nvdev.mm fd.dev in
                  let vaddr = 0x1000000000 in
                  let (_ : Memory.virt_mapping) =
                    Memory.map_range mm ~vaddr ~size:0x200000
                      [ (0x400000, 0x200000) ]
                      Memory.Phys ()
                  in
                  equal int64 0x200012L (Mmio.read64 fd.fvram 0x0);
                  equal int64 0x201012L (Mmio.read64 fd.fvram 0x200000);
                  equal int64 0x202012L (Mmio.read64 fd.fvram 0x201000);
                  equal int64 0x203012L
                    (Mmio.read64 fd.fvram (0x202000 + (128 * 8)));
                  equal int64 0x400601L (Mmio.read64 fd.fvram 0x203000);
                  equal int64 0L (Mmio.read64 fd.fvram (0x203000 + 8));
                  equal (list (pair int int)) [ invalidate_write ] (writes fd);
                  (* Unmapping clears the page, frees the emptied chain
                     back to the root and does not invalidate. *)
                  Memory.unmap_range mm ~vaddr ~size:0x200000;
                  equal int64 0x600L (Mmio.read64 fd.fvram 0x0);
                  equal (list (pair int int)) [ invalidate_write ] (writes fd)));
          test "valloc walks the allocation tiers" (fun () ->
              with_fake_dev (fun fd ->
                  let mm = Nvdev.mm fd.dev in
                  let vm = Memory.valloc mm 0x201000 () in
                  (* One 2MB range, then a 4KB one for the tail. *)
                  equal (list (pair int int))
                    [ (0x200000, 0x200000); (0x400000, 0x1000) ]
                    vm.Memory.paddrs;
                  equal bool true
                    (vm.Memory.va_addr >= Nvdev.va_base
                    && vm.Memory.va_addr < Nvdev.va_base + Nvdev.va_size);
                  equal int 0 (vm.Memory.va_addr land 0x1fffff);
                  Memory.vfree mm vm));
          test "a small bar reserves a page-table region" (fun () ->
              with_fake_dev ~scratch_mb:128 (fun fd ->
                  (* 128MB of VRAM behind an 80MB BAR: page tables get a
                     1MB carve-out behind the boot region and the main
                     region starts after it. *)
                  let mm = Nvdev.mm fd.dev in
                  equal int 0x4000000 (Memory.vram_size mm);
                  equal int 0x200000 (Memory.palloc mm 0x1000 ~ptable:true ());
                  equal int 0x300000 (Memory.palloc mm 0x1000 ()));
              with_fake_dev (fun fd ->
                  (* A large bar keeps page tables in the main region. *)
                  let mm = Nvdev.mm fd.dev in
                  equal int 0x200000 (Memory.palloc mm 0x1000 ~ptable:true ());
                  equal int 0x201000 (Memory.palloc mm 0x1000 ())));
        ];
      group "armed-region recovery"
        [
          test "a clean device only enables bus mastering" (fun () ->
              with_fake_dev (fun fd ->
                  equal int 0 !(fd.resets);
                  equal
                    (list (triple int int int))
                    [ (0x4, 0x7, 2) ]
                    (cfg_writes fd)));
          test "an armed region drops mastering, resets, re-enables"
            (fun () ->
              with_fake_dev
                ~pre:(fun ~reads ~resets ->
                  Hashtbl.replace reads wpr2_hi_addr (fun () ->
                      if !resets > 0 then 0 else 0x00100000))
                (fun fd ->
                  equal int 1 !(fd.resets);
                  equal
                    (list (triple int int int))
                    [ (0x4, 0x3, 2); (0x4, 0x7, 2) ]
                    (cfg_writes fd);
                  (* The settle delay consumed the scripted clock. *)
                  equal bool true (!(fd.clock) >= 100);
                  equal int 0
                    (Nv_reg.read
                       (Nvdev.reg fd.dev "NV_PFB_PRI_MMU_WPR2_ADDR_HI"))));
        ];
      group "boot memory"
        [
          test "device memory rounds to pages and maps bus addresses"
            (fun () ->
              with_fake_dev (fun fd ->
                  let view, paddr, sysaddr =
                    Nvdev.alloc_boot_mem fd.dev
                      ~data:(Bytes.of_string "BOOTMEM!") 0x1800
                  in
                  (* The manager's main region starts past the 2MB boot
                     region. *)
                  equal (option int) (Some 0x200000) paddr;
                  equal int 0x2000 (Mmio.size view);
                  equal (list int) [ 0x40200000; 0x40201000 ] sysaddr;
                  equal string "BOOTMEM!"
                    (Bytes.to_string
                       (Mmio.read_bytes fd.fvram ~off:0x200000 ~len:8))));
          test "system memory keeps the raw size and has no paddr"
            (fun () ->
              with_fake_dev (fun fd ->
                  let view, paddr, sysaddr =
                    Nvdev.alloc_boot_mem fd.dev ~sysmem:true ~contiguous:true
                      ~data:(Bytes.of_string "SYS!") 0x1800
                  in
                  equal (option int) None paddr;
                  equal (list (pair bool int)) [ (true, 0x1800) ] !(fd.sys_reqs);
                  equal (list int) [ 0xaa000; 0xab000 ] sysaddr;
                  ignore view;
                  equal string "SYS!"
                    (Bytes.to_string
                       (Mmio.read_bytes fd.fvram ~off:0x100000 ~len:4))));
          test "a small bar defaults to system memory" (fun () ->
              with_fake_dev ~scratch_mb:128 (fun fd ->
                  let _, paddr, _ = Nvdev.alloc_boot_mem fd.dev 0x1000 in
                  equal (option int) None paddr;
                  equal int 1 (List.length !(fd.sys_reqs));
                  (* Forcing device memory overrides the default; the
                     main region sits past the boot region and the 1MB
                     page-table carve-out. *)
                  let _, paddr, _ =
                    Nvdev.alloc_boot_mem fd.dev ~sysmem:false 0x1000
                  in
                  equal (option int) (Some 0x300000) paddr));
        ];
    ]
