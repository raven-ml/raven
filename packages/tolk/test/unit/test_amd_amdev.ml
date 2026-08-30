(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver-less AMD device core, on pieces that run without hardware:
   the IP-discovery parser on synthesized tables, the page-table entry
   encoding against hand-computed golden words, page-table walks over an
   anonymous mapping standing in for VRAM, named-bitfield register
   access over a fake register file, and firmware loading and image
   splitting over synthesized firmware blobs. *)

open Windtrap
module Amdev = Tolk_amd.Amdev
module Firmware = Tolk_amd.Amdev.Firmware
module Am = Tolk_amd.Amd_tables.Am_defs
module Fw_defs = Tolk_amd.Amd_tables.Fw_defs
module Reg = Tolk_amd.Amd_tables.Reg
module Mmio = Tolk_amd.Hcq.Mmio
module File_io = Tolk_amd.Hcq.File_io
module Memory = Tolk.Memory
module Tlsf = Tolk.Tlsf

(* IP discovery fixtures: tables synthesized from the same struct
   layouts the parser reads, with the ip-discovery header at 0x100, the
   die at 0x200 and the gc table at 0x400. *)

let s8 = Bytes.set_uint8
let s16 = Bytes.set_uint16_le
let s32 b p v = Bytes.set_int32_le b p (Int32.of_int v)

let add_ip ~base64 b pos ~hw_id ~inst ~ver:(ma, mi, rv) bases =
  s16 b pos hw_id;
  s8 b (pos + 2) inst;
  s8 b (pos + 3) (List.length bases);
  s8 b (pos + 4) ma;
  s8 b (pos + 5) mi;
  s8 b (pos + 6) rv;
  List.iteri
    (fun i v ->
      if base64 then Bytes.set_int64_le b (pos + 8 + (i * 8)) (Int64.of_int v)
      else s32 b (pos + 8 + (i * 4)) v)
    bases;
  pos + 8 + ((if base64 then 8 else 4) * List.length bases)

let discovery_blob ?(base64 = false) ips =
  let b = Bytes.make 0x800 '\x00' in
  s32 b 0 Am.binary_signature;
  s16 b
    (Am.Binary_header.table_list_offset
    + (Am.table_ip_discovery * Am.Table_info.sizeof))
    0x100;
  s16 b
    (Am.Binary_header.table_list_offset + (Am.table_gc * Am.Table_info.sizeof))
    0x400;
  s32 b 0x100 Am.discovery_table_signature;
  s16 b (0x100 + 0xc) 1;
  s16 b (0x100 + Am.Ip_discovery_header.die_info_offset + 2) 0x200;
  s8 b (0x100 + 0x4e) (if base64 then 1 else 0);
  s16 b 0x200 0;
  s16 b 0x202 (List.length ips);
  let pos = ref (0x200 + Am.Die_header.sizeof) in
  List.iter
    (fun (hw_id, inst, ver, bases) ->
      pos := add_ip ~base64 b !pos ~hw_id ~inst ~ver bases)
    ips;
  (* gc geometry table, major version 2 *)
  s16 b 0x404 2;
  s16 b 0x406 0;
  s32 b (0x400 + 0xc) 2;
  s32 b (0x400 + 0x10) 8;
  s32 b (0x400 + 0x14) 1;
  s32 b (0x400 + 0x3c) 16;
  s32 b (0x400 + 0x40) 32;
  s32 b (0x400 + 0x44) 64;
  b

let sample_ips =
  [
    (0xb, 0, (11, 0, 2), [ 0x1c000; 0x2400 ]);
    (0xb, 1, (11, 0, 2), [ 0x5000 ]);
    (0x6c, 0, (4, 3, 0), [ 0xd20 ]);
    (0xff, 0, (13, 0, 10), [ 0x1000 ]);
  ]

(* Page-table fixtures over an anonymous mapping standing in for the
   VRAM BAR. *)

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

let make_mm ~gc_ver ?paddr_base vram =
  let ops = Amdev.Am_page_table.ops ~vram ~gc_ver ?paddr_base () in
  let booting = ref true in
  let mm =
    Memory.create ~pt_ops:ops ~vram_size:(Mmio.size vram) ~boot_size:0x100000
      ~va_bits:48
      ~va_shifts:[ 12; 21; 30; 39 ]
      ~va_base:0
      ~palloc_ranges:[ (0x200000, 0x200000); (0x1000, 0x1000) ]
      ~va_allocator:(Tlsf.create ~size:0x40000000 ~base:0 ())
      ~is_booting:(fun () -> !booting)
      ~zero_vram:(fun ~paddr ~size ->
        Mmio.blit_bytes vram ~off:paddr (Bytes.make size '\000'))
      ~first_lv:Am.amdgpu_vm_pdb2 ()
  in
  booting := false;
  (mm, ops)

let flags ~gc_ver ?(lv = Am.amdgpu_vm_ptb) ?(table = false) ?(frag = 0)
    ?(uncached = false) ?(system = false) ?(snooped = false) ?(valid = true) ()
    =
  Amdev.Am_page_table.pte_flags ~gc_ver ~lv ~table ~frag ~uncached ~system
    ~snooped ~valid

(* Register fixture: a register file backed by a hash table. *)

let fake_register () =
  let store = Hashtbl.create 4 in
  let r =
    {
      Reg.name = "regTEST";
      offset = 0x10;
      segment = 0;
      fields = [| ("lo", (0, 7)); ("mid", (8, 19)); ("hi", (20, 31)) |];
      addr = 0x123;
    }
  in
  let amr =
    Amdev.Am_register.make ~reg:r
      ~rreg:(fun a -> Option.value ~default:0 (Hashtbl.find_opt store a))
      ~wreg:(fun a v -> Hashtbl.replace store a v)
  in
  (amr, store)

(* Firmware fixtures: blobs synthesized at the struct offsets the
   parser reads, with distinctive payload tags so descriptor slices can
   be asserted whole. *)

let put b off s = Bytes.blit_string s 0 b off (String.length s)

let common_header b ~ver:(ma, mi) ~ucode_off ~ucode_size =
  s16 b 8 ma;
  s16 b 0xa mi;
  s32 b 0x14 ucode_size;
  s32 b 0x18 ucode_off

(* A v2_0 sOS container with two components. *)
let sos_blob () =
  let b = Bytes.make 0x140 '\x00' in
  common_header b ~ver:(2, 0) ~ucode_off:0x100 ~ucode_size:0;
  s32 b 0x20 2;
  (* component descriptors at the v2_0 bin offset 0x24 *)
  s32 b 0x24 2;
  s32 b 0x2c 0;
  s32 b 0x30 4;
  s32 b 0x34 3;
  s32 b 0x3c 4;
  s32 b 0x40 8;
  put b 0x100 "SYS!";
  put b 0x104 "KDB-DATA";
  b

let smu_blob_gfx11 () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:8;
  put b 0x40 "SMUCODE!";
  b

(* A v2_1 SMU header whose soft-pptable list holds one P2S table. *)
let smu_blob_gfx9 () =
  let b = Bytes.make 0x100 '\x00' in
  common_header b ~ver:(2, 1) ~ucode_off:0 ~ucode_size:0;
  s32 b 0x24 2;
  s32 b 0x28 0x40;
  s32 b 0x40 0x11223344;
  s32 b 0x44 0x98;
  s32 b 0x48 4;
  s32 b 0x4c 0x50325358;
  s32 b 0x50 0x90;
  s32 b 0x54 6;
  put b 0x90 "P2STAB";
  b

let sdma_blob_v1 () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:12;
  put b 0x40 "SDMA-CODE-12";
  b

let sdma_blob_v2 () =
  let b = Bytes.make 0x80 '\x00' in
  common_header b ~ver:(2, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x24 8;
  s32 b 0x30 0x60;
  s32 b 0x34 6;
  put b 0x40 "CTXCODE!";
  put b 0x60 "CTLCOD";
  b

let sdma_blob_v3 () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(3, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x28 8;
  put b 0x40 "SDMA3TH0";
  b

(* A v1_0 GFX header: code with a trailing jump table. *)
let gfx_blob_v1 () =
  let b = Bytes.make 0x80 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:0x20;
  s32 b 0x24 6;
  s32 b 0x28 2;
  put b 0x40 "MEC-V1-CODE-24-BYTES-OK!";
  put b 0x58 "MECJTAB!";
  b

(* A v2_0 GFX header: code, stack data, and a start address. *)
let gfx_blob_v2 ~code ~stack ~start_lo ~start_hi =
  let b = Bytes.make 0x80 '\x00' in
  common_header b ~ver:(2, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x24 8;
  s32 b 0x2c 8;
  s32 b 0x30 0x60;
  s32 b 0x34 start_lo;
  s32 b 0x38 start_hi;
  put b 0x40 code;
  put b 0x60 stack;
  b

let imu_blob () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x20 8;
  s32 b 0x28 4;
  put b 0x40 "IMUIRAM!";
  put b 0x48 "IMUD";
  b

let rlc_blob_v2_1 () =
  let b = Bytes.make 0x140 '\x00' in
  common_header b ~ver:(2, 1) ~ucode_off:0x100 ~ucode_size:8;
  s32 b 0x74 4;
  s32 b 0x78 0x110;
  s32 b 0x84 8;
  s32 b 0x88 0x118;
  s32 b 0x94 4;
  s32 b 0x98 0x120;
  put b 0x100 "RLCGCODE";
  put b 0x110 "CNTL";
  put b 0x118 "GPMLIST!";
  put b 0x120 "SRM!";
  b

let rlc_blob_v2_3 () =
  let b = Bytes.make 0x140 '\x00' in
  common_header b ~ver:(2, 3) ~ucode_off:0x100 ~ucode_size:8;
  s32 b 0x9c 8;
  s32 b 0xa0 0x110;
  s32 b 0xa4 4;
  s32 b 0xa8 0x118;
  s32 b 0xb4 8;
  s32 b 0xb8 0x120;
  s32 b 0xc4 4;
  s32 b 0xc8 0x128;
  put b 0x100 "RLCGCODE";
  put b 0x110 "RLCIRAM!";
  put b 0x118 "RLCD";
  put b 0x120 "RLCPCODE";
  put b 0x128 "RLCV";
  b

(* A loader over an in-memory file set that records what was asked. *)
let fw_loader files =
  let requested = ref [] in
  let load name =
    requested := name :: !requested;
    match List.assoc_opt name files with
    | Some b -> b
    | None -> fail ("unexpected firmware request " ^ name)
  in
  (load, fun () -> List.rev !requested)

let desc_strings descs =
  List.map (fun (types, data) -> (types, Bytes.to_string data)) descs

(* A firmware directory in a temp dir, handed to the loader's [dir]. *)
let with_fw_dir f =
  let dir = Filename.temp_file "tolk_fw_test" "" in
  Sys.remove dir;
  Sys.mkdir dir 0o700;
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun e -> Sys.remove (Filename.concat dir e))
        (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () -> f dir)

let write_file path content =
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc content)

let sha_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

let () =
  run "Amdev"
    [
      group "ip discovery"
        [
          test "parses versions and per-instance bases" (fun () ->
              let d = Amdev.parse_discovery (discovery_blob sample_ips) in
              equal
                (triple int int int)
                (11, 0, 2)
                (List.assoc Am.gc_hwip d.Amdev.ip_ver);
              equal
                (triple int int int)
                (13, 0, 10)
                (List.assoc Am.mp0_hwip d.Amdev.ip_ver);
              equal
                (list (pair int (array int)))
                [ (0, [| 0x1c000; 0x2400 |]); (1, [| 0x5000 |]) ]
                (List.assoc Am.gc_hwip d.Amdev.regs_offset));
          test "one bus-interface entry feeds both nbio and nbif" (fun () ->
              let d = Amdev.parse_discovery (discovery_blob sample_ips) in
              equal
                (triple int int int)
                (4, 3, 0)
                (List.assoc Am.nbio_hwip d.Amdev.ip_ver);
              equal
                (triple int int int)
                (4, 3, 0)
                (List.assoc Am.nbif_hwip d.Amdev.ip_ver);
              equal
                (list (pair int (array int)))
                [ (0, [| 0xd20 |]) ]
                (List.assoc Am.nbif_hwip d.Amdev.regs_offset));
          test "reads the gc geometry table" (fun () ->
              let d = Amdev.parse_discovery (discovery_blob sample_ips) in
              match d.Amdev.gc_info with
              | Amdev.Gc_info_v2 g ->
                  equal int 2 g.num_se;
                  equal int 8 g.num_cu_per_sh;
                  equal int 1 g.num_sh_per_se;
                  equal int 16 g.max_waves_per_simd;
                  equal int 32 g.max_scratch_slots_per_cu;
                  equal int 64 g.lds_size
              | Amdev.Gc_info_v1 _ -> fail "expected a v2 gc table");
          test "decodes 64-bit base addresses" (fun () ->
              let d =
                Amdev.parse_discovery
                  (discovery_blob ~base64:true
                     [ (0xb, 0, (12, 0, 1), [ 0x100000040 ]) ])
              in
              equal
                (list (pair int (array int)))
                [ (0, [| 0x100000040 |]) ]
                (List.assoc Am.gc_hwip d.Amdev.regs_offset));
          test "rejects corrupted signatures" (fun () ->
              let b = discovery_blob sample_ips in
              s32 b 0 0xdead;
              raises_match
                (Exn.failure ~substring:"discovery signatures mismatch")
                (fun () -> Amdev.parse_discovery b);
              let b = discovery_blob sample_ips in
              s32 b 0x100 0xdead;
              raises_match
                (Exn.failure ~substring:"discovery signatures mismatch")
                (fun () -> Amdev.parse_discovery b));
        ];
      group "pte flags"
        [
          test "gfx11 words" (fun () ->
              equal int64 0x71L (flags ~gc_ver:(11, 0, 0) ());
              equal int64 0x0003000000000071L
                (flags ~gc_ver:(11, 0, 0) ~uncached:true ());
              (* A leaf above the leaf level is a huge page. *)
              equal int64 0x400000000004F1L
                (flags ~gc_ver:(11, 0, 0) ~lv:Am.amdgpu_vm_pdb0 ~frag:9 ());
              equal int64 0x1L (flags ~gc_ver:(11, 0, 0) ~table:true ()));
          test "gfx12 words keep bit 63" (fun () ->
              let leaf = flags ~gc_ver:(12, 0, 0) () in
              equal int64 0x8000000000000071L leaf;
              equal bool true (Int64.logand leaf Am.amdgpu_pte_is_pte <> 0L);
              equal int64 0x80C0000000000071L
                (flags ~gc_ver:(12, 0, 0) ~lv:Am.amdgpu_vm_pdb0 ~uncached:true
                   ());
              equal int64 0x1L (flags ~gc_ver:(12, 0, 0) ~table:true ()));
          test "gfx9 words" (fun () ->
              equal int64 0x4800000000000001L
                (flags ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb1 ~table:true ());
              equal int64 0x0100000000000001L
                (flags ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb0 ~table:true ());
              equal int64 0x0040000000000071L
                (flags ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb1 ());
              equal int64 0x77L
                (flags ~gc_ver:(9, 4, 3) ~system:true ~snooped:true ()));
          test "huge-page detection per generation" (fun () ->
              let huge = Amdev.Am_page_table.is_pte_huge_page in
              equal bool true
                (huge ~gc_ver:(12, 0, 0) ~lv:Am.amdgpu_vm_pdb0
                   0x8000000000000000L);
              equal bool false (huge ~gc_ver:(12, 0, 0) ~lv:Am.amdgpu_vm_pdb0 0x71L);
              equal bool true
                (huge ~gc_ver:(11, 0, 0) ~lv:Am.amdgpu_vm_pdb0
                   Am.amdgpu_pde_pte);
              (* On gfx9 a pdb0 entry is a table iff it carries the
                 translate-further flag. *)
              equal bool false
                (huge ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb0 Am.amdgpu_pte_tf);
              equal bool true (huge ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb0 0L);
              equal bool true
                (huge ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb1
                   Am.amdgpu_pde_pte));
        ];
      group "page tables"
        [
          test "walk writes small pages at every level" (fun () ->
              with_fake_vram 0x800000 (fun vram ->
                  let mm, ops = make_mm ~gc_ver:(11, 0, 0) vram in
                  let (_ : Memory.virt_mapping) =
                    Memory.map_range mm ~vaddr:0x3000 ~size:0x2000
                      [ (0x300000, 0x2000) ]
                      Memory.Phys ()
                  in
                  equal int64 0x100001L (Mmio.read64 vram 0x0);
                  equal int64 0x101001L (Mmio.read64 vram 0x100000);
                  equal int64 0x102001L (Mmio.read64 vram 0x101000);
                  equal int64 0x300071L (Mmio.read64 vram (0x102000 + (3 * 8)));
                  equal int64 0x301071L (Mmio.read64 vram (0x102000 + (4 * 8)));
                  let pts = Memory.page_tables mm ~vaddr:0x3000 ~size:0x1000 in
                  equal (list int) [ 0; 1; 2; 3 ]
                    (List.map ops.Memory.lv pts);
                  equal (list int)
                    [ 0x0; 0x100000; 0x101000; 0x102000 ]
                    (List.map ops.Memory.paddr pts);
                  let ptb = List.nth pts 3 in
                  equal bool true (ops.Memory.valid ptb 3);
                  equal bool false (ops.Memory.valid ptb 5);
                  equal bool true (ops.Memory.is_page ptb 3);
                  equal int 0x300000 (ops.Memory.address ptb 3);
                  equal int64 0x301071L (ops.Memory.entry ptb 4)));
          test "walk maps a gfx12 huge page and unmaps it" (fun () ->
              with_fake_vram 0x800000 (fun vram ->
                  let mm, ops = make_mm ~gc_ver:(12, 0, 0) vram in
                  let (_ : Memory.virt_mapping) =
                    Memory.map_range mm ~vaddr:0x200000 ~size:0x200000
                      [ (0x400000, 0x200000) ]
                      Memory.Phys ()
                  in
                  equal int64 0x100001L (Mmio.read64 vram 0x0);
                  equal int64 0x101001L (Mmio.read64 vram 0x100000);
                  let huge = Mmio.read64 vram (0x101000 + 8) in
                  equal int64 0x80000000004004F1L huge;
                  equal bool true
                    (Int64.logand huge Am.amdgpu_pde_pte_gfx12 <> 0L);
                  let pts =
                    Memory.page_tables mm ~vaddr:0x200000 ~size:0x200000
                  in
                  equal (list int) [ 0; 1; 2 ] (List.map ops.Memory.lv pts);
                  Memory.unmap_range mm ~vaddr:0x200000 ~size:0x200000;
                  (* A cleared entry keeps its flag bits with valid off. *)
                  equal int64 0x8000000000000070L (Mmio.read64 vram 0x0);
                  equal bool false (ops.Memory.valid (List.nth pts 0) 0)));
          test "rebases device-local addresses" (fun () ->
              with_fake_vram 0x4000 (fun vram ->
                  let ops =
                    Amdev.Am_page_table.ops ~vram ~gc_ver:(9, 4, 3)
                      ~paddr_base:(fun () -> 0x10000000)
                      ()
                  in
                  let pt = ops.Memory.make ~paddr:0x1000 ~lv:Am.amdgpu_vm_ptb in
                  ops.Memory.set_entry pt ~idx:0 ~paddr:0x2000 ~valid:true ();
                  equal int64 0x10002071L (Mmio.read64 vram 0x1000);
                  equal int 0x2000 (ops.Memory.address pt 0)));
          test "system entries skip the rebase and refuse address" (fun () ->
              with_fake_vram 0x4000 (fun vram ->
                  let ops =
                    Amdev.Am_page_table.ops ~vram ~gc_ver:(9, 4, 3)
                      ~paddr_base:(fun () -> 0x10000000)
                      ()
                  in
                  let pt = ops.Memory.make ~paddr:0x1000 ~lv:Am.amdgpu_vm_ptb in
                  ops.Memory.set_entry pt ~idx:1 ~paddr:0x3000
                    ~aspace:Memory.Sys ~uncached:true ~snooped:true ~valid:true
                    ();
                  equal int64 0x0600000000003077L
                    (Mmio.read64 vram (0x1000 + 8));
                  raises_match
                    (Exn.invalid_arg ~substring:"system address")
                    (fun () -> ops.Memory.address pt 1)));
          test "rejects addresses beyond the address-space mask" (fun () ->
              with_fake_vram 0x4000 (fun vram ->
                  let ops =
                    Amdev.Am_page_table.ops ~vram ~gc_ver:(11, 0, 0) ()
                  in
                  let pt = ops.Memory.make ~paddr:0x1000 ~lv:Am.amdgpu_vm_ptb in
                  raises_match
                    (Exn.invalid_arg ~substring:"Invalid physical address")
                    (fun () ->
                      ops.Memory.set_entry pt ~idx:0 ~paddr:(1 lsl 45)
                        ~valid:true ())));
        ];
      group "registers"
        [
          test "field writes encode and decode back" (fun () ->
              let amr, store = fake_register () in
              Amdev.Am_register.write amr [ ("lo", 0xab); ("hi", 0xcd) ];
              equal int 0xcd000ab (Hashtbl.find store 0x123);
              equal
                (list (pair string int))
                [ ("lo", 0xab); ("mid", 0); ("hi", 0xcd) ]
                (Amdev.Am_register.read_bitfields amr));
          test "update touches only the named fields" (fun () ->
              let amr, _ = fake_register () in
              Amdev.Am_register.write amr [ ("lo", 0xab); ("hi", 0xcd) ];
              Amdev.Am_register.update amr [ ("mid", 5) ];
              equal int 0xcd005ab (Amdev.Am_register.read amr));
          test "a raw value passes through unchanged" (fun () ->
              let amr, store = fake_register () in
              Amdev.Am_register.write amr ~value:0xdeadbeef [];
              equal int 0xdeadbeef (Hashtbl.find store 0x123));
          test "unknown fields raise" (fun () ->
              let amr, _ = fake_register () in
              raises_match
                (Exn.invalid_arg ~substring:"has no field")
                (fun () -> Amdev.Am_register.write amr [ ("nope", 1) ]);
              raises_match
                (Exn.invalid_arg ~substring:"has no field")
                (fun () -> Amdev.Am_register.update amr [ ("nope", 1) ]));
        ];
      group "firmware"
        [
          test "loads and splits a gfx11 firmware set" (fun () ->
              let files =
                [
                  ("psp_13_0_10_sos.bin", sos_blob ());
                  ("smu_13_0_10.bin", smu_blob_gfx11 ());
                  ("sdma_6_0_2.bin", sdma_blob_v2 ());
                  ( "gc_11_0_2_mec.bin",
                    gfx_blob_v2 ~code:"MECCODE1" ~stack:"MECSTAK1"
                      ~start_lo:0x1000 ~start_hi:2 );
                  ("gc_11_0_2_imu.bin", imu_blob ());
                  ("gc_11_0_2_rlc.bin", rlc_blob_v2_3 ());
                ]
              in
              let load, requested = fw_loader files in
              let fw =
                Firmware.create ~load
                  [
                    (Am.gc_hwip, (11, 0, 2));
                    (Am.sdma0_hwip, (6, 0, 2));
                    (Am.mp0_hwip, (13, 0, 10));
                    (Am.mp1_hwip, (13, 0, 10));
                  ]
              in
              equal (list string) (List.map fst files) (requested ());
              equal
                (list (pair int string))
                [ (2, "SYS!"); (3, "KDB-DATA") ]
                (List.map
                   (fun (t, d) -> (t, Bytes.to_string d))
                   fw.Firmware.sos_fw);
              (match fw.Firmware.smu_psp_desc with
              | Some (types, data) ->
                  equal (list int) [ Am.gfx_fw_type_smu ] types;
                  equal string "SMUCODE!" (Bytes.to_string data)
              | None -> fail "expected an smu image");
              equal
                (list (pair string int))
                [ ("MEC", 0x200001000) ]
                fw.Firmware.ucode_start;
              equal
                (list (pair (list int) string))
                [
                  ([ Am.gfx_fw_type_sdma_ucode_th1 ], "CTLCOD");
                  ([ Am.gfx_fw_type_sdma_ucode_th0 ], "CTXCODE!");
                  ([ Am.gfx_fw_type_rs64_mec ], "MECCODE1");
                  ([ Am.gfx_fw_type_rs64_mec_p0_stack ], "MECSTAK1");
                  ([ Am.gfx_fw_type_imu_i ], "IMUIRAM!");
                  ([ Am.gfx_fw_type_imu_d ], "IMUD");
                  ([ Am.gfx_fw_type_rlc_iram ], "RLCIRAM!");
                  ([ Am.gfx_fw_type_rlc_dram_boot ], "RLCD");
                  ([ Am.gfx_fw_type_rlc_p ], "RLCPCODE");
                  ([ Am.gfx_fw_type_rlc_v ], "RLCV");
                  ([ Am.gfx_fw_type_rlc_g ], "RLCGCODE");
                ]
                (desc_strings fw.Firmware.descs));
          test "gfx9: pptable scan, jump table, save-restore lists"
            (fun () ->
              let files =
                [
                  ("psp_13_0_6_sos.bin", sos_blob ());
                  ("smu_13_0_6.bin", smu_blob_gfx9 ());
                  ("sdma_4_4_2.bin", sdma_blob_v1 ());
                  ("gc_9_4_3_mec.bin", gfx_blob_v1 ());
                  ("gc_9_4_3_rlc.bin", rlc_blob_v2_1 ());
                ]
              in
              let load, requested = fw_loader files in
              let fw =
                Firmware.create ~load
                  [
                    (Am.gc_hwip, (9, 4, 3));
                    (Am.sdma0_hwip, (4, 4, 2));
                    (Am.mp0_hwip, (13, 0, 6));
                    (Am.mp1_hwip, (13, 0, 6));
                  ]
              in
              equal (list string) (List.map fst files) (requested ());
              equal bool true (fw.Firmware.smu_psp_desc = None);
              equal (list (pair string int)) [] fw.Firmware.ucode_start;
              equal
                (list (pair (list int) string))
                [
                  ([ Am.gfx_fw_type_p2s_table ], "P2STAB");
                  ( [
                      Am.gfx_fw_type_sdma0; Am.gfx_fw_type_sdma1;
                      Am.gfx_fw_type_sdma2; Am.gfx_fw_type_sdma3;
                    ],
                    "SDMA-CODE-12" );
                  ([ Am.gfx_fw_type_cp_mec ], "MEC-V1-CODE-24-BYTES-OK!");
                  ([ Am.gfx_fw_type_cp_mec_me1 ], "MECJTAB!");
                  ([ Am.gfx_fw_type_rlc_restore_list_srm_cntl ], "CNTL");
                  ([ Am.gfx_fw_type_rlc_restore_list_gpm_mem ], "GPMLIST!");
                  ([ Am.gfx_fw_type_rlc_restore_list_srm_mem ], "SRM!");
                  ([ Am.gfx_fw_type_rlc_g ], "RLCGCODE");
                ]
                (desc_strings fw.Firmware.descs));
          test "gfx12: pfp and me images, smu 13.0.12 skipped" (fun () ->
              let files =
                [
                  ("psp_14_0_3_sos.bin", sos_blob ());
                  ("sdma_7_0_0.bin", sdma_blob_v3 ());
                  ( "gc_12_0_1_pfp.bin",
                    gfx_blob_v2 ~code:"PFPCODE1" ~stack:"PFPSTAK1"
                      ~start_lo:0x100 ~start_hi:0 );
                  ( "gc_12_0_1_me.bin",
                    gfx_blob_v2 ~code:"ME-CODE1" ~stack:"ME-STAK1"
                      ~start_lo:0x200 ~start_hi:0 );
                  ( "gc_12_0_1_mec.bin",
                    gfx_blob_v2 ~code:"MECCODE1" ~stack:"MECSTAK1"
                      ~start_lo:0x1000 ~start_hi:2 );
                  ("gc_12_0_1_imu.bin", imu_blob ());
                  ("gc_12_0_1_rlc.bin", rlc_blob_v2_3 ());
                ]
              in
              let load, requested = fw_loader files in
              let fw =
                Firmware.create ~load
                  [
                    (Am.gc_hwip, (12, 0, 1));
                    (Am.sdma0_hwip, (7, 0, 0));
                    (Am.mp0_hwip, (14, 0, 3));
                    (Am.mp1_hwip, (13, 0, 12));
                  ]
              in
              equal (list string) (List.map fst files) (requested ());
              equal bool true (fw.Firmware.smu_psp_desc = None);
              equal
                (list (pair string int))
                [ ("PFP", 0x100); ("ME", 0x200); ("MEC", 0x200001000) ]
                fw.Firmware.ucode_start;
              equal
                (list (list int))
                [
                  [ Am.gfx_fw_type_sdma_ucode_th0 ];
                  [ Am.gfx_fw_type_rs64_pfp ];
                  [ Am.gfx_fw_type_rs64_pfp_p0_stack ];
                  [ Am.gfx_fw_type_rs64_me ];
                  [ Am.gfx_fw_type_rs64_me_p0_stack ];
                  [ Am.gfx_fw_type_rs64_mec ];
                  [ Am.gfx_fw_type_rs64_mec_p0_stack ];
                  [ Am.gfx_fw_type_imu_i ];
                  [ Am.gfx_fw_type_imu_d ];
                  [ Am.gfx_fw_type_rlc_iram ];
                  [ Am.gfx_fw_type_rlc_dram_boot ];
                  [ Am.gfx_fw_type_rlc_p ];
                  [ Am.gfx_fw_type_rlc_v ];
                  [ Am.gfx_fw_type_rlc_g ];
                ]
                (List.map fst fw.Firmware.descs));
          test "rejects unknown image header versions" (fun () ->
              let b = Bytes.make 0x40 '\x00' in
              common_header b ~ver:(9, 9) ~ucode_off:0 ~ucode_size:0;
              raises_match
                (Exn.failure ~substring:"unhandled psp firmware header v9_9")
                (fun () ->
                  Firmware.create
                    ~load:(fun _ -> b)
                    [
                      (Am.gc_hwip, (11, 0, 2));
                      (Am.sdma0_hwip, (6, 0, 2));
                      (Am.mp0_hwip, (13, 0, 10));
                      (Am.mp1_hwip, (13, 0, 10));
                    ]));
          test "fetches files whose digests match" (fun () ->
              with_fw_dir (fun dir ->
                  write_file (Filename.concat dir "fw.bin") "abc";
                  equal string "abc"
                    (Bytes.to_string
                       (Firmware.fetch_fw ~dir "fw.bin" ~sha256:sha_abc));
                  (* a message padding into a second sha256 block *)
                  let msg =
                    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
                  in
                  write_file (Filename.concat dir "fw2.bin") msg;
                  equal string msg
                    (Bytes.to_string
                       (Firmware.fetch_fw ~dir "fw2.bin"
                          ~sha256:
                            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"))));
          test "corrupted files fail naming both digests" (fun () ->
              with_fw_dir (fun dir ->
                  write_file (Filename.concat dir "smu_13_0_0.bin") "garbage";
                  let pinned = List.assoc "smu_13_0_0.bin" Fw_defs.hashes in
                  raises_match
                    (Exn.failure
                       ~substring:("fetch sha mismatch, expected " ^ pinned))
                    (fun () -> Firmware.load_fw ~dir "smu_13_0_0.bin");
                  (* the sha256 of "garbage" *)
                  raises_match
                    (Exn.failure
                       ~substring:
                         "795b6904e54f82411df4b0e27a373a55eea3f9d66dac5a9bce1dd92f7b401da5")
                    (fun () -> Firmware.load_fw ~dir "smu_13_0_0.bin")));
          test "missing files name the searched paths" (fun () ->
              with_fw_dir (fun dir ->
                  raises_match
                    (Exn.failure
                       ~substring:(Filename.concat dir "psp_13_0_0_sos.bin"))
                    (fun () -> Firmware.load_fw ~dir "psp_13_0_0_sos.bin");
                  raises_match
                    (Exn.failure
                       ~substring:
                         (Filename.concat dir "psp_13_0_0_sos.bin.zst"))
                    (fun () -> Firmware.load_fw ~dir "psp_13_0_0_sos.bin");
                  raises_match
                    (Exn.failure
                       ~substring:"gitlab.com/kernel-firmware/linux-firmware")
                    (fun () -> Firmware.load_fw ~dir "psp_13_0_0_sos.bin");
                  raises_match
                    (Exn.failure ~substring:"has no pinned sha256")
                    (fun () -> Firmware.load_fw ~dir "unknown_fw.bin")));
          test "decompresses the zst variant" (fun () ->
              if Sys.command "command -v zstd >/dev/null 2>&1" <> 0 then
                skip ~reason:"zstd not on PATH" ()
              else
                with_fw_dir (fun dir ->
                    let raw = Filename.concat dir "fw.raw" in
                    write_file raw "abc";
                    equal int 0
                      (Sys.command
                         (Printf.sprintf "zstd -q %s -o %s"
                            (Filename.quote raw)
                            (Filename.quote
                               (Filename.concat dir "fw.bin.zst"))));
                    equal string "abc"
                      (Bytes.to_string
                         (Firmware.fetch_fw ~dir "fw.bin" ~sha256:sha_abc))));
        ];
    ]
