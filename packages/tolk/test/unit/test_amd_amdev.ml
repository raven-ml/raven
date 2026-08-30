(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver-less AMD device core, on pieces that run without hardware:
   the IP-discovery parser on synthesized tables, the page-table entry
   encoding against hand-computed golden words, page-table walks over an
   anonymous mapping standing in for VRAM, and named-bitfield register
   access over a fake register file. *)

open Windtrap
module Amdev = Tolk_amd.Amdev
module Am = Tolk_amd.Amd_tables.Am_defs
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
    ]
