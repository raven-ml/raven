(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver-less NVIDIA device core, on pieces that run without
   hardware: register resolution over the included families, named
   bitfield access against hand-computed words, chip detection and the
   armed-firmware-region recovery over scripted register read-backs,
   VRAM sizing, and boot-memory allocation over an anonymous mapping
   standing in for the VRAM BAR. *)

open Windtrap
module Nvdev = Tolk_nv.Nvdev
module Nv_reg = Nvdev.Nv_reg
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
   time. *)

type fake = {
  dev : Nvdev.t;
  fvram : Mmio.t;
  store : (int, int) Hashtbl.t;
  log : (int * int) list ref;
  resets : int ref;
  cfg_writes : (int * int * int) list ref;
  clock : int ref;
  pallocs : int list ref;
  sys_reqs : (bool * int) list ref;
}

let with_fake_dev ?(arch = 0x17) ?(impl = 2) ?(boot0 = 0x174000a1)
    ?(scratch_mb = 2) ?(vram_bytes = 0x400000) ?(bar1_base = 0x40000000)
    ?(cfg = 0x7) ?(pre = fun ~reads:_ ~resets:_ -> ()) f =
  with_fake_vram vram_bytes (fun fvram ->
      let store = Hashtbl.create 16 in
      let reads = Hashtbl.create 16 in
      let log = ref [] in
      let resets = ref 0 in
      let cfg_writes = ref [] in
      let cfg_val = ref cfg in
      let clock = ref 0 in
      let pallocs = ref [] in
      let next_paddr = ref 0x1000 in
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
          ~bar1_base
          ~palloc:(fun sz ->
            pallocs := sz :: !pallocs;
            let p = !next_paddr in
            next_paddr := p + sz;
            p)
          ~rreg ~wreg
          ~mmio:(Mmio.view fvram ~off:0 ~size:0x1000 ())
          ~vram:fvram ~devfmt:"test" ()
      in
      f { dev; fvram; store; log; resets; cfg_writes; clock; pallocs; sys_reqs })

(* The chronological write log. *)
let writes fd = List.rev !(fd.log)
let cfg_writes fd = List.rev !(fd.cfg_writes)

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
              with_fake_dev ~scratch_mb:2 ~vram_bytes:0x400000 (fun fd ->
                  equal int 0x200000 (Nvdev.vram_size fd.dev);
                  equal bool true (Nvdev.large_bar fd.dev)));
          test "a bar smaller than vram is not large" (fun () ->
              with_fake_dev ~scratch_mb:8 ~vram_bytes:0x400000 (fun fd ->
                  equal int 0x800000 (Nvdev.vram_size fd.dev);
                  equal bool false (Nvdev.large_bar fd.dev)));
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
                  equal (option int) (Some 0x1000) paddr;
                  equal (list int) [ 0x2000 ] !(fd.pallocs);
                  equal int 0x2000 (Mmio.size view);
                  equal (list int) [ 0x40001000; 0x40002000 ] sysaddr;
                  equal string "BOOTMEM!"
                    (Bytes.to_string
                       (Mmio.read_bytes fd.fvram ~off:0x1000 ~len:8))));
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
              with_fake_dev ~scratch_mb:8 (fun fd ->
                  let _, paddr, _ = Nvdev.alloc_boot_mem fd.dev 0x1000 in
                  equal (option int) None paddr;
                  equal int 1 (List.length !(fd.sys_reqs));
                  (* Forcing device memory overrides the default. *)
                  let _, paddr, _ =
                    Nvdev.alloc_boot_mem fd.dev ~sysmem:false 0x1000
                  in
                  equal (option int) (Some 0x1000) paddr));
        ];
    ]
