(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk

(* Fake page tables over an int64 array standing in for device memory.
   Synthetic PTE encoding: bit 0 valid, bit 1 table, bit 2 uncached,
   bit 3 snooped, bits 4-5 aspace (0 phys, 1 sys, 2 peer), bits 6-11
   frag, bits 12+ physical address. Huge pages are allowed at every
   level below the root, provided the physical address is aligned to
   the level's coverage. *)

type fake_pt = { paddr : int; lv : int }

type fixture = {
  mm : fake_pt Memory.t;
  vram : int64 array;
  zeroed : (int * int) list ref;
  booting : bool ref;
  flushes : int ref;
}

let encode ~paddr ~table ~uncached ~aspace ~snooped ~frag ~valid =
  let bit b pos = if b then Int64.shift_left 1L pos else 0L in
  let aspace_bits =
    match aspace with Memory.Phys -> 0 | Memory.Sys -> 1 | Memory.Peer -> 2
  in
  List.fold_left Int64.logor (Int64.of_int paddr)
    [
      bit valid 0;
      bit table 1;
      bit uncached 2;
      bit snooped 3;
      Int64.of_int (aspace_bits lsl 4);
      Int64.of_int (frag lsl 6);
    ]

let make_fixture ?(va_base = 0) ?(vram_size = 0x100000) ?(boot_size = 0x10000)
    ?(reserve_ptable = false)
    ?(palloc_ranges = [ (0x8000, 0x8000); (0x1000, 0x1000) ])
    ?(va_size = 0x200000) ?(smi_dev = false) () =
  let vram = Array.make (vram_size / 8) 0L in
  let zeroed = ref [] in
  let booting = ref true in
  let flushes = ref 0 in
  let leaf_lv = 2 in
  let covers = [| 0x40000; 0x8000; 0x1000 |] in
  let word pt idx = vram.((pt.paddr / 8) + idx) in
  let pt_ops =
    {
      Memory.make = (fun ~paddr ~lv -> { paddr; lv });
      set_entry =
        (fun pt ~idx ~paddr ?(table = false) ?(uncached = false)
             ?(aspace = Memory.Phys) ?(snooped = false) ?(frag = 0) ~valid () ->
          vram.((pt.paddr / 8) + idx) <-
            encode ~paddr ~table ~uncached ~aspace ~snooped ~frag ~valid);
      entry = word;
      valid = (fun pt idx -> Int64.logand (word pt idx) 1L <> 0L);
      address =
        (fun pt idx -> Int64.to_int (Int64.logand (word pt idx) (-0x1000L)));
      is_page =
        (fun pt idx -> pt.lv = leaf_lv || Int64.logand (word pt idx) 3L = 1L);
      supports_huge_page =
        (fun pt ~paddr -> pt.lv >= 1 && paddr mod covers.(pt.lv) = 0);
      paddr = (fun pt -> pt.paddr);
      lv = (fun pt -> pt.lv);
    }
  in
  let mm =
    Memory.create ~pt_ops ~vram_size ~boot_size ~va_bits:21
      ~va_shifts:[ 12; 15; 18 ] ~va_base ~palloc_ranges
      ~va_allocator:(Tlsf.create ~size:va_size ~base:va_base ())
      ~is_booting:(fun () -> !booting)
      ~zero_vram:(fun ~paddr ~size ->
        zeroed := (paddr, size) :: !zeroed;
        Array.fill vram (paddr / 8) (size / 8) 0L)
      ~reserve_ptable ~smi_dev
      ~on_range_mapped:(fun () -> incr flushes)
      ()
  in
  booting := false;
  { mm; vram; zeroed; booting; flushes }

(* A second fake with six levels and a 57-bit virtual address space,
   whose level-4 tables pack two 64-bit words per logical entry: a
   child-table entry keeps the flag bits of the encoding above in the
   low word and the child's physical address in the high word, while a
   page entry keeps the whole encoding in the low word and zeroes the
   high one. The manager must consume such entries only through the
   injected operations, so the packing stays internal to them. Huge
   pages are allowed from level 3 down, provided the physical address
   is aligned to the level's coverage. *)

let make_six_level_fixture () =
  let vram_size = 0x100000 and boot_size = 0x10000 in
  let vram = Array.make (vram_size / 8) 0L in
  let zeroed = ref [] in
  let booting = ref true in
  let flushes = ref 0 in
  let dual_lv = 4 and leaf_lv = 5 in
  let covers = [| 1 lsl 56; 1 lsl 47; 1 lsl 38; 1 lsl 29; 1 lsl 21; 1 lsl 12 |] in
  let low pt idx =
    vram.((pt.paddr / 8) + if pt.lv = dual_lv then 2 * idx else idx)
  in
  let pt_ops =
    {
      Memory.make = (fun ~paddr ~lv -> { paddr; lv });
      set_entry =
        (fun pt ~idx ~paddr ?(table = false) ?(uncached = false)
             ?(aspace = Memory.Phys) ?(snooped = false) ?(frag = 0) ~valid () ->
          let base = pt.paddr / 8 in
          if pt.lv = dual_lv then begin
            vram.(base + (2 * idx)) <-
              encode
                ~paddr:(if table then 0 else paddr)
                ~table ~uncached ~aspace ~snooped ~frag ~valid;
            vram.(base + (2 * idx) + 1) <-
              (if table then Int64.of_int paddr else 0L)
          end
          else
            vram.(base + idx) <-
              encode ~paddr ~table ~uncached ~aspace ~snooped ~frag ~valid);
      entry = low;
      valid = (fun pt idx -> Int64.logand (low pt idx) 1L <> 0L);
      address =
        (fun pt idx ->
          if pt.lv = dual_lv then
            Int64.to_int vram.((pt.paddr / 8) + (2 * idx) + 1)
          else Int64.to_int (Int64.logand (low pt idx) (-0x1000L)));
      is_page =
        (fun pt idx -> pt.lv = leaf_lv || Int64.logand (low pt idx) 3L = 1L);
      supports_huge_page =
        (fun pt ~paddr -> pt.lv >= 3 && paddr mod covers.(pt.lv) = 0);
      paddr = (fun pt -> pt.paddr);
      lv = (fun pt -> pt.lv);
    }
  in
  let mm =
    Memory.create ~pt_ops ~vram_size ~boot_size ~va_bits:56
      ~va_shifts:[ 12; 21; 29; 38; 47; 56 ] ~va_base:0
      ~palloc_ranges:[ (0x1000, 0x1000) ]
      ~va_allocator:(Tlsf.create ~size:0x200000 ())
      ~is_booting:(fun () -> !booting)
      ~zero_vram:(fun ~paddr ~size ->
        zeroed := (paddr, size) :: !zeroed;
        Array.fill vram (paddr / 8) (size / 8) 0L)
      ~on_range_mapped:(fun () -> incr flushes)
      ()
  in
  booting := false;
  { mm; vram; zeroed; booting; flushes }

let slice fx paddr n = Array.sub fx.vram (paddr / 8) n
let sparse n entries = Array.init n (fun i -> Option.value ~default:0L (List.assoc_opt i entries))
let oom = function Tlsf.Out_of_memory _ -> true | _ -> false
let vm_paddrs vm = vm.Memory.paddrs

let () =
  run "Memory"
    [
      group "Map_range"
        [
          test "writes small-page entries at every level" (fun () ->
              let fx = make_fixture () in
              let vm =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              equal (array int64) (sparse 16 [ (0, 0x10003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (0, 0x11003L) ])
                (slice fx 0x10000 8);
              equal (array int64)
                (sparse 8 [ (3, 0x20001L); (4, 0x21001L) ])
                (slice fx 0x11000 8);
              equal int 0x3000 vm.Memory.va_addr;
              equal int 0x2000 vm.Memory.size;
              equal (list (pair int int)) [ (0x20000, 0x2000) ] (vm_paddrs vm);
              equal int 1 !(fx.flushes));
          test "uses a huge page when address and size allow it" (fun () ->
              let fx = make_fixture () in
              let vm =
                Memory.map_range fx.mm ~vaddr:0x10000 ~size:0x8000
                  [ (0x30000, 0x8000) ]
                  Memory.Phys ()
              in
              equal (array int64) (sparse 16 [ (0, 0x10003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (2, 0x300C1L) ])
                (slice fx 0x10000 8);
              equal int 0x8000 vm.Memory.size);
          test "splits a misaligned range into small pages" (fun () ->
              let fx = make_fixture () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x7000 ~size:0x9000
                  [ (0x40000, 0x9000) ]
                  Memory.Phys ()
              in
              equal (array int64) (sparse 16 [ (0, 0x10003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (0, 0x11003L); (1, 0x12003L) ])
                (slice fx 0x10000 8);
              equal (array int64)
                (sparse 8 [ (7, 0x40001L) ])
                (slice fx 0x11000 8);
              equal (array int64)
                (Array.init 8 (fun i -> Int64.of_int (0x41001 + (i * 0x1000))))
                (slice fx 0x12000 8));
          test "records uncached, snooped and address space bits" (fun () ->
              let fx = make_fixture () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x1000
                  [ (0x20000, 0x1000) ]
                  Memory.Sys ~uncached:true ~snooped:true ()
              in
              equal (array int64)
                (sparse 8 [ (3, 0x2001DL) ])
                (slice fx 0x11000 8));
          test "rebases virtual addresses" (fun () ->
              let fx = make_fixture ~va_base:0x200000 () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x203000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              equal (array int64)
                (sparse 8 [ (3, 0x20001L); (4, 0x21001L) ])
                (slice fx 0x11000 8));
          test "rejects mismatched physical sizes" (fun () ->
              let fx = make_fixture () in
              raises_match (Exn.invalid_arg ~substring:"Size mismatch")
                (fun () ->
                  Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                    [ (0x20000, 0x1000) ]
                    Memory.Phys ()));
          test "rejects mapping an already mapped page" (fun () ->
              let fx = make_fixture () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              raises_match (Exn.invalid_arg ~substring:"PTE already mapped")
                (fun () ->
                  Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x1000
                    [ (0x50000, 0x1000) ]
                    Memory.Phys ()));
        ];
      group "Unmap_range"
        [
          test "clears entries and reclaims empty page tables" (fun () ->
              let fx = make_fixture () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              Memory.unmap_range fx.mm ~vaddr:0x3000 ~size:0x2000;
              equal (array int64) (Array.make 16 0L) (slice fx 0 16);
              equal (array int64) (Array.make 8 0L) (slice fx 0x10000 8);
              equal (array int64) (Array.make 8 0L) (slice fx 0x11000 8);
              (* The freed tables go back to the allocator: remapping
                 rebuilds the identical image at the same addresses. *)
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              equal (array int64) (sparse 16 [ (0, 0x10003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (0, 0x11003L) ])
                (slice fx 0x10000 8);
              equal (array int64)
                (sparse 8 [ (3, 0x20001L); (4, 0x21001L) ])
                (slice fx 0x11000 8));
          test "keeps page tables that still hold mappings" (fun () ->
              let fx = make_fixture () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              Memory.unmap_range fx.mm ~vaddr:0x3000 ~size:0x1000;
              equal (array int64) (sparse 16 [ (0, 0x10003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (4, 0x21001L) ])
                (slice fx 0x11000 8));
          test "rejects unmapping an unmapped range" (fun () ->
              let fx = make_fixture () in
              raises_match
                (Exn.invalid_arg ~substring:"Not allowed to create")
                (fun () -> Memory.unmap_range fx.mm ~vaddr:0x3000 ~size:0x1000));
        ];
      group "Page_tables"
        [
          test "creates and returns the chain covering a range" (fun () ->
              let fx = make_fixture () in
              let pts = Memory.page_tables fx.mm ~vaddr:0x3000 ~size:0x1000 in
              equal (list int) [ 0; 1; 2 ] (List.map (fun pt -> pt.lv) pts);
              equal (list int) [ 0; 0x10000; 0x11000 ]
                (List.map (fun pt -> pt.paddr) pts));
        ];
      group "Alloc_vaddr"
        [
          test "aligns naturally to the request size" (fun () ->
              let fx = make_fixture ~va_base:0x200000 () in
              equal int 0x200000 (Memory.alloc_vaddr fx.mm 0x1000 ());
              equal int 0x202000 (Memory.alloc_vaddr fx.mm 0x3000 ()));
          test "honours a larger explicit alignment" (fun () ->
              let fx = make_fixture ~va_base:0x200000 () in
              equal int 0x200000 (Memory.alloc_vaddr fx.mm 0x1000 ());
              equal int 0x208000
                (Memory.alloc_vaddr fx.mm 0x1000 ~align:0x8000 ()));
        ];
      group "Palloc"
        [
          test "enforces the boot allocation discipline" (fun () ->
              let fx = make_fixture () in
              fx.booting := true;
              raises_match (Exn.invalid_arg ~substring:"During booting")
                (fun () -> Memory.palloc fx.mm 0x1000 ());
              equal int 0x1000 (Memory.palloc fx.mm 0x1000 ~boot:true ());
              fx.booting := false;
              raises_match (Exn.invalid_arg ~substring:"During booting")
                (fun () -> Memory.palloc fx.mm 0x1000 ~boot:true ());
              equal int 0x10000 (Memory.palloc fx.mm 0x1000 ()));
          test "zeroes allocations unless told otherwise" (fun () ->
              let fx = make_fixture () in
              equal (list (pair int int)) [ (0, 0x1000) ] !(fx.zeroed);
              let p = Memory.palloc fx.mm 0x2000 () in
              equal (list (pair int int)) [ (p, 0x2000); (0, 0x1000) ]
                !(fx.zeroed);
              let (_ : int) = Memory.palloc fx.mm 0x1000 ~zero:false () in
              equal int 2 (List.length !(fx.zeroed)));
          test "skips zeroing the root page table for smi devices" (fun () ->
              let fx = make_fixture ~smi_dev:true () in
              equal (list (pair int int)) [] !(fx.zeroed));
          test "reserves a dedicated page-table region" (fun () ->
              let fx = make_fixture ~vram_size:0x300000 ~reserve_ptable:true ()
              in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x1000
                  [ (0x200000, 0x1000) ]
                  Memory.Phys ()
              in
              (* Page tables come from the reserved region right after
                 the boot region; plain allocations from the main
                 region after it. *)
              equal (array int64) (sparse 16 [ (0, 0x10003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (0, 0x11003L) ])
                (slice fx 0x10000 8);
              equal int 0x110000 (Memory.palloc fx.mm 0x1000 ()));
        ];
      group "Valloc"
        [
          test "maps small requests from the smallest range" (fun () ->
              let fx = make_fixture () in
              let vm = Memory.valloc fx.mm 0x2000 () in
              equal int 0 vm.Memory.va_addr;
              equal (list (pair int int))
                [ (0x10000, 0x1000); (0x11000, 0x1000) ]
                (vm_paddrs vm);
              equal (array int64) (sparse 16 [ (0, 0x12003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (0, 0x13003L) ])
                (slice fx 0x12000 8);
              equal (array int64)
                (sparse 8 [ (0, 0x10001L); (1, 0x11001L) ])
                (slice fx 0x13000 8));
          test "prefers large ranges and falls back for the tail" (fun () ->
              let fx = make_fixture () in
              let vm = Memory.valloc fx.mm 0x9000 () in
              equal (list (pair int int))
                [ (0x10000, 0x8000); (0x18000, 0x1000) ]
                (vm_paddrs vm);
              equal (array int64) (sparse 16 [ (0, 0x19003L) ]) (slice fx 0 16);
              equal (array int64)
                (sparse 8 [ (0, 0x100C1L); (1, 0x1A003L) ])
                (slice fx 0x19000 8);
              equal (array int64)
                (sparse 8 [ (0, 0x18001L) ])
                (slice fx 0x1A000 8));
          test "allocates one zeroed range when contiguous" (fun () ->
              let fx = make_fixture () in
              let vm = Memory.valloc fx.mm 0x2000 ~contiguous:true () in
              equal (list (pair int int)) [ (0x10000, 0x2000) ] (vm_paddrs vm);
              equal bool true (List.mem (0x10000, 0x2000) !(fx.zeroed)));
          test "retries smaller ranges when a larger one is exhausted"
            (fun () ->
              let fx =
                make_fixture ~vram_size:0x13000 ~boot_size:0x8000
                  ~palloc_ranges:[ (0x6000, 0x1000); (0x1000, 0x1000) ]
                  ()
              in
              (* Fragment the main region so no block fits the 0x6000
                 range but 4KB pieces still do. *)
              let p1 = Memory.palloc fx.mm 0x4000 () in
              let (_ : int) = Memory.palloc fx.mm 0x1000 () in
              Memory.pfree fx.mm p1 ();
              let vm = Memory.valloc fx.mm 0x6000 () in
              equal (list (pair int int))
                [
                  (0x8000, 0x1000);
                  (0x9000, 0x1000);
                  (0xA000, 0x1000);
                  (0xD000, 0x1000);
                  (0xE000, 0x1000);
                  (0xF000, 0x1000);
                ]
                (vm_paddrs vm));
          test "releases partial allocations when memory runs out" (fun () ->
              let fx =
                make_fixture ~vram_size:0xA000 ~boot_size:0x8000
                  ~palloc_ranges:[ (0x1000, 0x1000) ]
                  ()
              in
              raises_match oom (fun () -> Memory.valloc fx.mm 0x3000 ());
              equal int 0x8000 (Memory.palloc fx.mm 0x1000 ()));
        ];
      group "Vfree"
        [
          test "returns virtual and physical space for reuse" (fun () ->
              let fx = make_fixture () in
              let vm = Memory.valloc fx.mm 0x2000 () in
              Memory.vfree fx.mm vm;
              equal (array int64) (Array.make 16 0L) (slice fx 0 16);
              let vm2 = Memory.valloc fx.mm 0x2000 () in
              equal int vm.Memory.va_addr vm2.Memory.va_addr;
              equal (list (pair int int)) (vm_paddrs vm) (vm_paddrs vm2));
        ];
      group "Identity_map"
        [
          test "maps requests through one shared identity mapping" (fun () ->
              Unix.putenv "GMMU" "0";
              Fun.protect
                ~finally:(fun () -> Unix.putenv "GMMU" "1")
                (fun () ->
                  let fx = make_fixture () in
                  let vm = Memory.valloc fx.mm 0x2000 () in
                  (* The request's memory is allocated first, then the
                     whole physical memory is identity mapped once with
                     huge pages; the request itself maps nothing new. *)
                  equal (array int64)
                    (sparse 16
                       [
                         (0, 0x12003L);
                         (1, 0x13003L);
                         (2, 0x14003L);
                         (3, 0x15003L);
                       ])
                    (slice fx 0 16);
                  equal (array int64)
                    (Array.init 8 (fun i ->
                         Int64.of_int (0x181 + (i * 0x8000))))
                    (slice fx 0x12000 8);
                  equal int 0x10000 vm.Memory.va_addr;
                  equal (list (pair int int))
                    [ (0x10000, 0x2000) ]
                    (vm_paddrs vm);
                  let vm2 = Memory.valloc fx.mm 0x1000 () in
                  equal int 0x16000 vm2.Memory.va_addr;
                  Memory.vfree fx.mm vm;
                  equal int 0x10000 (Memory.palloc fx.mm 0x1000 ~zero:false ())));
        ];
      group "Six_level_dual"
        [
          test "derives the tree geometry from the shifts" (fun () ->
              let fx = make_six_level_fixture () in
              equal int 6 (Memory.level_cnt fx.mm);
              equal int (1 lsl 56) (Memory.pte_covers fx.mm 0);
              equal int 2 (Memory.pte_cnt fx.mm 0);
              equal int 0x200000 (Memory.pte_covers fx.mm 4);
              equal int 256 (Memory.pte_cnt fx.mm 4);
              equal int 0x1000 (Memory.pte_covers fx.mm 5);
              equal int 512 (Memory.pte_cnt fx.mm 5));
          test "writes both words of a dual entry on the way to a leaf"
            (fun () ->
              let fx = make_six_level_fixture () in
              let vm =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              equal (array int64) (sparse 8 [ (0, 0x10003L) ]) (slice fx 0 8);
              equal (array int64)
                (sparse 8 [ (0, 0x11003L) ])
                (slice fx 0x10000 8);
              equal (array int64)
                (sparse 8 [ (0, 0x12003L) ])
                (slice fx 0x11000 8);
              equal (array int64)
                (sparse 8 [ (0, 0x13003L) ])
                (slice fx 0x12000 8);
              equal (array int64)
                (sparse 8 [ (0, 3L); (1, 0x14000L) ])
                (slice fx 0x13000 8);
              equal (array int64)
                (sparse 8 [ (3, 0x20001L); (4, 0x21001L) ])
                (slice fx 0x14000 8);
              equal int 0x2000 vm.Memory.size;
              equal int 1 !(fx.flushes));
          test "round-trips a mapping through all six levels" (fun () ->
              let fx = make_six_level_fixture () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              Memory.unmap_range fx.mm ~vaddr:0x3000 ~size:0x2000;
              List.iter
                (fun paddr ->
                  equal (array int64) (Array.make 8 0L) (slice fx paddr 8))
                [ 0; 0x10000; 0x11000; 0x12000; 0x13000; 0x14000 ];
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x3000 ~size:0x2000
                  [ (0x20000, 0x2000) ]
                  Memory.Phys ()
              in
              equal (array int64)
                (sparse 8 [ (0, 3L); (1, 0x14000L) ])
                (slice fx 0x13000 8);
              equal (array int64)
                (sparse 8 [ (3, 0x20001L); (4, 0x21001L) ])
                (slice fx 0x14000 8));
          test "maps a huge page as one dual page entry" (fun () ->
              let fx = make_six_level_fixture () in
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr:0x200000 ~size:0x200000
                  [ (0x40000000, 0x200000) ]
                  Memory.Phys ()
              in
              equal (array int64)
                (sparse 8 [ (0, 0x13003L) ])
                (slice fx 0x12000 8);
              (* One 2MB page at logical entry 1 of the dual table:
                 flags, frag 9 and address in the low word, zero in
                 the high word. *)
              equal (array int64)
                (sparse 8 [ (2, 0x40000241L) ])
                (slice fx 0x13000 8);
              Memory.unmap_range fx.mm ~vaddr:0x200000 ~size:0x200000;
              List.iter
                (fun paddr ->
                  equal (array int64) (Array.make 8 0L) (slice fx paddr 8))
                [ 0; 0x10000; 0x11000; 0x12000; 0x13000 ]);
          test "indexes the root by the fifty-seventh address bit" (fun () ->
              let fx = make_six_level_fixture () in
              let vaddr = (1 lsl 56) + 0x5000 in
              (* Build the chain first: an unmapped walk that stops at
                 the root's final entry would step past the root. *)
              let pts = Memory.page_tables fx.mm ~vaddr ~size:0x1000 in
              equal (list int) [ 0; 1; 2; 3; 4; 5 ]
                (List.map (fun pt -> pt.lv) pts);
              equal (list int)
                [ 0; 0x10000; 0x11000; 0x12000; 0x13000; 0x14000 ]
                (List.map (fun pt -> pt.paddr) pts);
              let (_ : Memory.virt_mapping) =
                Memory.map_range fx.mm ~vaddr ~size:0x1000
                  [ (0x20000, 0x1000) ]
                  Memory.Phys ()
              in
              equal (array int64) (sparse 8 [ (1, 0x10003L) ]) (slice fx 0 8);
              equal (array int64)
                (sparse 8 [ (0, 3L); (1, 0x14000L) ])
                (slice fx 0x13000 8);
              equal (array int64)
                (sparse 8 [ (5, 0x20001L) ])
                (slice fx 0x14000 8);
              Memory.unmap_range fx.mm ~vaddr ~size:0x1000;
              equal (array int64) (Array.make 8 0L) (slice fx 0 8));
        ];
    ]
