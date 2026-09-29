(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The vendor-independent support of GPU runtimes: address allocators, page
   tables over a GPU memory held in a table, locked system memory, firmware
   lookup and digests, and mapped memory over memory the test allocates. *)

open Windtrap
open Nx_device_support

(* Mmio *)

external alloc : int -> nativeint = "test_support_alloc"

let test_mmio () =
  let m = Mmio.v (alloc 64) 64 in
  Mmio.fill m 0 64 '\000';
  Mmio.set32 m 4 0xdead_beef;
  equal ~msg:"32-bit round trip" int 0xdead_beef (Mmio.get32 m 4);
  Mmio.set64 m 8 0x0123_4567_89ab_cdefL;
  equal ~msg:"64-bit round trip" int64 0x0123_4567_89ab_cdefL (Mmio.get64 m 8);
  equal ~msg:"little-endian" int 0xef (Mmio.get8 m 8);
  Mmio.set8 m 1 0x1ff;
  equal ~msg:"a byte keeps its low bits" int 0xff (Mmio.get8 m 1);
  Mmio.write m 17 "hello";
  equal ~msg:"unaligned bulk" string "hello" (Mmio.read m 17 5);
  let s = Mmio.sub m 16 16 in
  equal ~msg:"sub" string "hello" (Mmio.read s 1 5);
  let inv = Exn.invalid_arg ~substring:"" in
  raises_match inv (fun () -> Mmio.get32 m 2);
  raises_match inv (fun () -> Mmio.get64 m 60);
  raises_match inv (fun () -> Mmio.write m 62 "abc");
  raises_match inv (fun () -> Mmio.sub m 60 8)

(* Tlsf *)

type op = Alloc of int * int | Free of int

let op_gen =
  Gen.frequency
    [
      ( 3,
        Gen.map
          (fun (n, a) -> Alloc (n, 1 lsl a))
          (Gen.pair (Gen.int_range 1 5000) (Gen.int_range 0 8)) );
      (2, Gen.map (fun i -> Free i) Gen.nat);
    ]

(* Random allocations and frees against the list of live blocks: blocks stay
   inside the range, aligned and disjoint, and once all are freed the whole
   range is one block again. *)
let test_tlsf =
  prop "blocks stay disjoint and coalesce" (Gen.list op_gen) (fun ops ->
      let base = 0x10000 and size = 1 lsl 16 in
      let t = Tlsf.create ~base size in
      let live = ref [] in
      List.iter
        (function
          | Alloc (n, align) -> (
              match Tlsf.alloc ~align t n with
              | None -> ()
              | Some a ->
                  is_true ~msg:"aligned" (a mod align = 0);
                  is_true ~msg:"inside" (a >= base && a + n <= base + size);
                  List.iter
                    (fun (b, m) ->
                      is_true ~msg:"disjoint" (a + n <= b || b + m <= a))
                    !live;
                  live := (a, n) :: !live)
          | Free i -> (
              match !live with
              | [] -> ()
              | l ->
                  let a, _ = List.nth l (i mod List.length l) in
                  Tlsf.free t a;
                  live := List.filter (fun (b, _) -> b <> a) l))
        ops;
      List.iter (fun (a, _) -> Tlsf.free t a) !live;
      is_some ~msg:"coalesced" (Tlsf.alloc t size))

let test_tlsf_refusals () =
  let t = Tlsf.create ~base:0 4096 in
  is_none ~msg:"too large" (Tlsf.alloc t 8192);
  let a = require_some (Tlsf.alloc t 16) in
  Tlsf.free t a;
  raises_match (Exn.invalid_arg ~substring:"no block") (fun () -> Tlsf.free t a)

(* Page tables *)

(* A test format: bit 0 valid, bit 1 a page, bits 7-11 the fragment, bits 12-47
   the address; the leaf level always maps pages. *)
let format mem ~flushes =
  let open Page_table in
  {
    levels = [ 12; 21; 30; 39 ];
    bits = 48;
    first = 0;
    get =
      (fun ~level:_ ~table i ->
        Option.value ~default:0L (Hashtbl.find_opt mem (table + (8 * i))));
    set = (fun ~level:_ ~table i e -> Hashtbl.replace mem (table + (8 * i)) e);
    encode =
      (fun ~level:_ ~table _ ~uncached:_ ~snooped:_ ~fragment ~valid pa ->
        if not valid then 0L
        else
          Int64.(
            logor (of_int pa)
              (logor 1L
                 (logor
                    (if table then 0L else 2L)
                    (shift_left (of_int fragment) 7)))));
    valid = (fun e -> Int64.logand e 1L <> 0L);
    leaf = (fun ~level e -> level = 3 || Int64.logand e 2L <> 0L);
    address = (fun e -> Int64.to_int (Int64.logand e 0xFFFF_FFFF_F000L));
    large = (fun ~level:_ -> true);
    zero =
      (fun pa n ->
        for i = 0 to (n / 8) - 1 do
          Hashtbl.remove mem (pa + (8 * i))
        done);
    flush = (fun () -> incr flushes);
  }

let space_base = 0x1_0000_0000

(* The main pool is a power of two, so that one request can take all of it:
   [memory] less the 1 MiB boot pool and the 1 MiB table pool, if any. *)
let tables ?(table_pool = false) ?(memory = 65 lsl 20) () =
  let mem = Hashtbl.create 64 and flushes = ref 0 in
  let space = Page_table.Space.create ~base:space_base (1 lsl 40) in
  let pages =
    List.init 10 (fun k ->
        let i = 9 - k in
        (1 lsl (i + 12), if i >= 9 then 2 lsl 20 else 0x1000))
  in
  let t =
    Page_table.create (format mem ~flushes) space ~memory ~boot:(1 lsl 20)
      ~tables:table_pool ~pages
  in
  Page_table.booted t;
  (t, mem, flushes)

(* The entry that maps [va], and its level and the offset of [va] in its page,
   in tables that translate from [base] (defaults to the space's). *)
let leaf ?(base = space_base) t mem va =
  let off = va - base in
  let rec go table level =
    let shift = List.nth [ 39; 30; 21; 12 ] level in
    let count = if level = 0 then 1024 else 512 in
    let e =
      Option.value ~default:0L
        (Hashtbl.find_opt mem (table + (8 * ((off lsr shift) mod count))))
    in
    if Int64.logand e 1L = 0L then None
    else if level = 3 || Int64.logand e 2L <> 0L then
      Some (e, level, off land ((1 lsl shift) - 1))
    else go (Int64.to_int (Int64.logand e 0xFFFF_FFFF_F000L)) (level + 1)
  in
  go (Page_table.root t) 0

(* The physical address [va] maps to, and the level of its entry. *)
let translate ?base t mem va =
  Option.map
    (fun (e, level, off) ->
      (Int64.to_int (Int64.logand e 0xFFFF_FFFF_F000L) + off, level))
    (leaf ?base t mem va)

let fragment t mem va =
  Option.map
    (fun (e, _, _) -> Int64.to_int (Int64.shift_right_logical e 7) land 0x1f)
    (leaf t mem va)

let test_map_pages () =
  let t, mem, flushes = tables () in
  let va = space_base + 0x5000 in
  let m = Page_table.map t ~va Page_table.Phys [ (0x40_0000, 0x3000) ] in
  equal ~msg:"size" int 0x3000 m.size;
  for p = 0 to 2 do
    equal ~msg:"translated"
      (option (pair int int))
      (Some (0x40_0000 + (p * 0x1000) + 8, 3))
      (translate t mem (va + (p * 0x1000) + 8))
  done;
  is_true ~msg:"flushed" (!flushes > 0);
  raises_match (Exn.invalid_arg ~substring:"mapped already") (fun () ->
      Page_table.map t ~va:(va + 0x1000) Page_table.Phys [ (0x50_0000, 0x1000) ]);
  let before = !flushes in
  Page_table.unmap t ~va 0x3000;
  is_none ~msg:"unmapped" (translate t mem va);
  is_true ~msg:"unmapping flushes" (!flushes > before);
  raises_match (Exn.invalid_arg ~substring:"not mapped") (fun () ->
      Page_table.unmap t ~va 0x1000)

let test_large_pages () =
  let t, mem, _ = tables () in
  let va = space_base + (4 lsl 20) in
  ignore (Page_table.map t ~va Page_table.Phys [ (2 lsl 20, 2 lsl 20) ]);
  equal ~msg:"one 2 MiB page"
    (option (pair int int))
    (Some ((2 lsl 20) + 0x1234, 2))
    (translate t mem (va + 0x1234));
  (* 2 MiB and 4 KiB entries mix when a range ends off a 2 MiB boundary. *)
  let va = space_base + (16 lsl 20) in
  ignore
    (Page_table.map t ~va Page_table.Phys [ (8 lsl 20, (2 lsl 20) + 0x2000) ]);
  equal ~msg:"the tail in 4 KiB pages"
    (option (pair int int))
    (Some ((10 lsl 20) + 0x1000, 3))
    (translate t mem (va + (2 lsl 20) + 0x1000));
  (* Physical memory aligned to 4 KiB maps with 4 KiB pages, whatever the
     virtual alignment. *)
  let va = space_base + (32 lsl 20) and pa = 0x30_1000 in
  ignore (Page_table.map t ~va Page_table.Phys [ (pa, 4 lsl 20) ]);
  List.iter
    (fun off ->
      equal ~msg:"4 KiB pages over unaligned memory"
        (option (pair int int))
        (Some (pa + off + 0x234, 3))
        (translate t mem (va + off + 0x234)))
    [ 0; 2 lsl 20; (4 lsl 20) - 0x1000 ]

(* A run's fragment is the largest block aligned in both address spaces that its
   size allows, from the space's first address on. *)
let test_fragments () =
  let t, mem, _ = tables () in
  ignore
    (Page_table.map t ~va:space_base Page_table.Phys [ (0x40_0000, 0x1000) ]);
  equal ~msg:"a page at the space's first address" (option int) (Some 0)
    (fragment t mem space_base);
  let va = space_base + (1 lsl 20) in
  ignore (Page_table.map t ~va Page_table.Phys [ (0x80_0000, 0x1_0000) ]);
  equal ~msg:"bounded by the size" (option int) (Some 4) (fragment t mem va);
  let va = space_base + (2 lsl 20) + 0x8000 in
  ignore (Page_table.map t ~va Page_table.Phys [ (0x100_0000, 0x1_0000) ]);
  equal ~msg:"bounded by the virtual address" (option int) (Some 3)
    (fragment t mem va);
  let va = space_base + (3 lsl 20) in
  ignore (Page_table.map t ~va Page_table.Phys [ (0x140_4000, 0x1_0000) ]);
  equal ~msg:"bounded by the physical address" (option int) (Some 2)
    (fragment t mem va)

(* A mapping that runs out of memory for its tables leaves nothing behind. *)
let test_out_of_tables () =
  let t, mem, _ = tables () in
  let va = space_base + (2 lsl 20) - 0x2000 in
  ignore (Page_table.map t ~va Page_table.Phys [ (0x40_0000, 0x1000) ]);
  let rec drain () =
    match Page_table.palloc ~zero:false t 0x1000 with
    | Some _ -> drain ()
    | None -> ()
  in
  drain ();
  raises_match (Exn.failure ~substring:"no memory for a page table") (fun () ->
      Page_table.map t ~va:(va + 0x1000) Page_table.Phys [ (0x50_0000, 0x2000) ]);
  is_none ~msg:"the part mapped is unmapped" (translate t mem (va + 0x1000));
  is_some ~msg:"the mapping before stays" (translate t mem va);
  (* Far mappings take the table pool, and then an allocation has main memory
     but no table. *)
  let t, _, _ = tables ~table_pool:true ~memory:(66 lsl 20) () in
  let rec fill k =
    match
      Page_table.map t
        ~va:(space_base + (1 lsl 39) + (k * (2 lsl 20)))
        Page_table.Sys
        [ (0x1000, 0x1000) ]
    with
    | _ -> fill (k + 1)
    | exception Failure _ -> ()
  in
  fill 0;
  raises_match (Exn.failure ~substring:"no memory for a page table") (fun () ->
      Page_table.alloc t 0x1000);
  is_some ~msg:"alloc freed its blocks"
    (Page_table.palloc ~align:1 ~zero:false t (Page_table.memory t));
  equal ~msg:"and its addresses" (option int) (Some space_base)
    (Page_table.Space.alloc (Page_table.space t) 0x1000)

(* Mapping and unmapping at random leaves no table behind: the whole pool is
   allocatable again. *)
let test_tables_freed =
  prop "unmapping frees the tables" ~count:30
    (Gen.list (Gen.pair (Gen.int_range 0 200) (Gen.int_range 1 40)))
    (fun ranges ->
      let t, mem, _ = tables () in
      let used = Hashtbl.create 16 in
      let mapped =
        List.filter_map
          (fun (page, n) ->
            let va = space_base + (page * 0x10_0000) in
            if Hashtbl.mem used page then None
            else begin
              Hashtbl.add used page ();
              Some
                (Page_table.map t ~va Page_table.Sys [ (0x1000, n * 0x1000) ])
            end)
          ranges
      in
      List.iter
        (fun (m : Page_table.mapping) -> Page_table.unmap t ~va:m.va m.size)
        mapped;
      is_true ~msg:"no entry is valid"
        (Hashtbl.fold (fun _ e ok -> ok && Int64.logand e 1L = 0L) mem true);
      is_some ~msg:"the pool is whole"
        (Page_table.palloc ~align:1 ~zero:false t (Page_table.memory t)))

let test_alloc () =
  let t, mem, _ = tables ~memory:(17 lsl 20) () in
  let m = require_some (Page_table.alloc t (3 lsl 20)) in
  equal ~msg:"pages cover the size" int (3 lsl 20)
    (List.fold_left (fun n (_, s) -> n + s) 0 m.pages);
  is_some ~msg:"mapped" (translate t mem (m.va + (1 lsl 20)));
  is_none ~msg:"more than the pool" (Page_table.alloc t (64 lsl 20));
  Page_table.free t m;
  is_none ~msg:"freed" (translate t mem m.va);
  is_some ~msg:"nothing leaked"
    (Page_table.palloc ~align:1 ~zero:false t (Page_table.memory t))

let test_boot_pool () =
  let mem = Hashtbl.create 8 and flushes = ref 0 in
  let space = Page_table.Space.create ~base:space_base (1 lsl 40) in
  let t =
    Page_table.create (format mem ~flushes) space ~memory:(8 lsl 20)
      ~boot:(1 lsl 20) ~tables:false
      ~pages:[ (0x1000, 0x1000) ]
  in
  let a = require_some (Page_table.palloc t 0x1000) in
  is_true ~msg:"booting allocates boot memory" (a < 1 lsl 20);
  Page_table.booted t;
  let b = require_some (Page_table.palloc t 0x1000) in
  is_true ~msg:"then main memory" (b >= 1 lsl 20);
  let c = require_some (Page_table.palloc ~boot:true t 0x1000) in
  is_true ~msg:"boot memory on request" (c < 1 lsl 20)

(* Entries are read and written with the level of their table: [set] sees each
   level from the root to the leaf once when one page is mapped. *)
let test_entry_levels () =
  let mem = Hashtbl.create 16 and flushes = ref 0 and levels = ref [] in
  let f = format mem ~flushes in
  let f =
    {
      f with
      set =
        (fun ~level ~table i e ->
          levels := level :: !levels;
          f.set ~level ~table i e);
    }
  in
  let space = Page_table.Space.create ~base:space_base (1 lsl 40) in
  let t =
    Page_table.create f space ~memory:(8 lsl 20) ~boot:(1 lsl 20) ~tables:false
      ~pages:[ (0x1000, 0x1000) ]
  in
  Page_table.booted t;
  ignore
    (Page_table.map t ~va:(space_base + 0x7000) Page_table.Phys
       [ (0x40_0000, 0x1000) ]);
  equal ~msg:"root to leaf" (list int) [ 0; 1; 2; 3 ] (List.rev !levels)

let test_path () =
  let t, mem, _ = tables () in
  let va = space_base + (6 lsl 20) in
  let path = Page_table.tables t ~va (2 lsl 20) in
  equal ~msg:"three tables down to the 2 MiB entries" int 3 (List.length path);
  equal ~msg:"root first" int (Page_table.root t) (List.hd path);
  equal ~msg:"the same tables again" (list int) path
    (Page_table.tables t ~va (2 lsl 20));
  ignore (Page_table.map t ~va Page_table.Phys [ (2 lsl 20, 2 lsl 20) ]);
  equal ~msg:"which map uses"
    (option (pair int int))
    (Some ((2 lsl 20) + 8, 2))
    (translate t mem (va + 8))

(* Tables may translate from a base below their space's: an address is then
   indexed from that base. *)
let test_base () =
  let mem = Hashtbl.create 16 and flushes = ref 0 in
  let space = Page_table.Space.create ~base:space_base (1 lsl 40) in
  let t =
    Page_table.create ~base:0 (format mem ~flushes) space ~memory:(8 lsl 20)
      ~boot:(1 lsl 20) ~tables:false
      ~pages:[ (0x1000, 0x1000) ]
  in
  Page_table.booted t;
  equal ~msg:"its base" int 0 (Page_table.base t);
  let va = space_base + 0x5000 in
  ignore (Page_table.map t ~va Page_table.Phys [ (0x40_0000, 0x1000) ]);
  equal ~msg:"indexed from 0"
    (option (pair int int))
    (Some (0x40_0008, 3))
    (translate ~base:0 t mem (va + 8));
  is_none ~msg:"not from the space's base" (translate t mem (va + 8))

(* Sysmem *)

(* Addresses the tests reserve for locked memory. *)
let sysmem_base = 0x30_0000_0000

(* The kilobytes of the process's memory Linux keeps locked. *)
let locked_kib () =
  In_channel.with_open_text "/proc/self/status" In_channel.input_lines
  |> List.find_map (fun l ->
      match String.split_on_char ':' l with
      | [ "VmLck"; v ] ->
          int_of_string_opt
            (String.trim (Filename.chop_suffix (String.trim v) "kB"))
      | _ -> None)
  |> Option.get

(* Memory [alloc] locked stays locked through a pin and an unpin of it, which
   another device's mapping of it makes. *)
let test_sysmem_pins () =
  match
    Sysmem.reserve ~base:sysmem_base (8 lsl 20);
    Sysmem.alloc ~va:sysmem_base (1 lsl 20)
  with
  | exception Failure why -> skip ~reason:why ()
  | m, pages ->
      equal ~msg:"a page each" int
        ((1 lsl 20) / Sysmem.page)
        (List.length pages);
      let locked = locked_kib () in
      is_true ~msg:"locked" (locked >= 1024);
      ignore (Sysmem.pin (Mmio.address m) (Mmio.length m) : int list);
      Sysmem.unpin (Mmio.address m) (Mmio.length m);
      equal ~msg:"still locked after a pin and an unpin" int locked
        (locked_kib ());
      Sysmem.free m;
      equal ~msg:"unlocked once freed" int (locked - 1024) (locked_kib ())

(* Pins are counted: of two pins of one range, one unpin leaves its pages
   locked, and the second unlocks them. *)
let test_sysmem_counted_pins () =
  let page = Sysmem.page in
  let raw = alloc (3 * page) in
  let a =
    Nativeint.(
      logand (add raw (of_int (page - 1))) (lognot (of_int (page - 1))))
  in
  match Sysmem.pin a page with
  | exception Failure why -> skip ~reason:why ()
  | _ ->
      let pinned = locked_kib () in
      ignore (Sysmem.pin a page : int list);
      Sysmem.unpin a page;
      equal ~msg:"still locked after one of two unpins" int pinned
        (locked_kib ());
      Sysmem.unpin a page;
      equal ~msg:"unlocked after the second" int
        (pinned - (page / 1024))
        (locked_kib ())

(* Contiguous memory is one huge page at an address on 2 MiB. *)
let test_sysmem_contiguous () =
  raises_match (Exn.invalid_arg ~substring:"not on 2 MiB") (fun () ->
      Sysmem.alloc ~contiguous:true
        ~va:(sysmem_base + Sysmem.page)
        (4 * Sysmem.page));
  match
    Sysmem.reserve ~base:sysmem_base (8 lsl 20);
    Sysmem.alloc ~contiguous:true ~va:(sysmem_base + (2 lsl 20)) (300 * 1024)
  with
  | exception Failure why -> skip ~reason:why ()
  | m, pages ->
      equal ~msg:"one address" int 1 (List.length pages);
      equal ~msg:"a huge page" int (2 lsl 20) (Mmio.length m);
      Sysmem.free m

(* Firmware *)

let test_sha256 () =
  equal ~msg:"empty" string
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    (Firmware.sha256 "");
  equal ~msg:"abc" string
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    (Firmware.sha256 "abc");
  equal ~msg:"two blocks" string
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    (Firmware.sha256 "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
  equal ~msg:"a million" string
    "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    (Firmware.sha256 (String.make 1_000_000 'a'))

let image = "raven firmware image\n"
let digest = "f4bd2d0ec861f4d7ce351f995c824a6b18368a3d9b0b053366c50336923ed291"

let of_hex h =
  String.init
    (String.length h / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))

let xz =
  of_hex
    "fd377a585a000004e6d6b44604c0191521011600000000000000000009e390b5010014726176656e206669726d7761726520696d6167650a00000000f2d3364f7087ea610001351576936aef1fb6f37d010000000004595a"

let zst =
  of_hex "28b52ffd2415a90000726176656e206669726d7761726520696d6167650a4a856cd1"

let has s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

let temp_dir () =
  let d = Filename.temp_dir "nx-firmware" "" in
  Sys.mkdir (Filename.concat d "amdgpu") 0o755;
  d

let write dir name s =
  Out_channel.with_open_bin (Filename.concat dir name) (fun oc ->
      output_string oc s)

(* The firmware cache's location is part of [Firmware]'s interface: the
   environment names it. *)
let with_cache f =
  let cache = Filename.temp_dir "nx-cache" "" in
  Unix.putenv "RAVEN_CACHE_ROOT" cache;
  Fun.protect
    ~finally:(fun () -> Unix.putenv "RAVEN_CACHE_ROOT" "")
    (fun () -> f cache)

let nowhere = "file:///nonexistent-raven-firmware/"

let test_firmware_dir () =
  with_cache @@ fun _ ->
  let dir = temp_dir () in
  write dir "amdgpu/plain.bin" image;
  equal ~msg:"plain" (result string string) (Ok image)
    (Firmware.get ~dir ~url:nowhere "amdgpu/plain.bin" ~sha256:digest);
  write dir "amdgpu/other.bin" "another image";
  match Firmware.get ~dir ~url:nowhere "amdgpu/other.bin" ~sha256:digest with
  | Error why -> contains ~msg:"names the file" ~sub:"amdgpu/other.bin" why
  | Ok _ -> fail "a file with another digest was loaded"

let test_firmware_compressed () =
  with_cache @@ fun _ ->
  let dir = temp_dir () in
  write dir "amdgpu/x.bin.xz" xz;
  write dir "amdgpu/z.bin.zst" zst;
  List.iter
    (fun name ->
      match Firmware.get ~dir ~url:nowhere name ~sha256:digest with
      | Ok s -> equal ~msg:name string image s
      | Error why ->
          if has why "downloading" then
            skip ~reason:"the system has no decompression library" ()
          else fail why)
    [ "amdgpu/x.bin"; "amdgpu/z.bin" ]

let test_firmware_download () =
  with_cache @@ fun cache ->
  let origin = temp_dir () in
  write origin "amdgpu/remote.bin" image;
  let url = "file://" ^ origin ^ "/" in
  (match Firmware.get ~url "amdgpu/remote.bin" ~sha256:digest with
  | Ok s -> equal ~msg:"downloaded" string image s
  | Error why when has why "libcurl" -> skip ~reason:why ()
  | Error why -> fail why);
  equal ~msg:"kept in the cache" string image
    (In_channel.with_open_bin
       (Filename.concat cache "firmware/amdgpu/remote.bin")
       In_channel.input_all);
  Sys.remove (Filename.concat origin "amdgpu/remote.bin");
  equal ~msg:"read from the cache" (result string string) (Ok image)
    (Firmware.get ~url "amdgpu/remote.bin" ~sha256:digest);
  write origin "amdgpu/bad.bin" "tampered";
  (match Firmware.get ~url "amdgpu/bad.bin" ~sha256:digest with
  | Error why -> contains ~msg:"refused, naming it" ~sub:"bad.bin" why
  | Ok _ -> fail "a download with another digest was loaded");
  match Firmware.get ~url "amdgpu/missing.bin" ~sha256:digest with
  | Error why -> contains ~msg:"a failed download" ~sub:"downloading" why
  | Ok _ -> fail "nothing to download"

(* Remote: a server in this process, on the loopback. *)

let key = "a key of the test, long enough"
let loopback = Unix.ADDR_INET (Unix.inet_addr_loopback, 0)

let port s =
  match Remote_server.address s with
  | Unix.ADDR_INET (_, p) -> p
  | Unix.ADDR_UNIX _ -> assert false

let with_server ?(key = key) ?programs f =
  let s = Remote_server.listen ~key ?programs loopback in
  Fun.protect ~finally:(fun () -> Remote_server.stop s) (fun () -> f s)

let connect ?(key = key) ?timeout_ms s =
  Remote.connect ?timeout_ms ~key "127.0.0.1" (port s)

let test_handshake () =
  with_server @@ fun s ->
  let r = connect s in
  equal ~msg:"the name as given" string
    (Printf.sprintf "127.0.0.1:%d" (port s))
    (Remote.name r);
  is_true ~msg:"a page" (Remote.page r > 0);
  is_true ~msg:"an architecture" (Remote.arch r <> "");
  Remote.ping r;
  Remote.close r;
  raises_match (Exn.failure ~substring:"closed") (fun () -> Remote.ping r);
  raises_match (Exn.failure ~substring:"does not know the key") (fun () ->
      connect ~key:"another key of sixteen bytes" s);
  let r = connect s in
  Remote.ping r;
  Remote.close r;
  raises_match (Exn.invalid_arg ~substring:"16") (fun () ->
      connect ~key:"short" s)

let test_server_proves () =
  (* A server with another key: the client refuses it, whatever it says. *)
  with_server ~key:"the server's own key, not ours" @@ fun s ->
  raises_match (Exn.failure ~substring:"key") (fun () -> connect s)

let test_busy () =
  with_server @@ fun s ->
  let r = connect s in
  raises_match (Exn.failure ~substring:"busy") (fun () -> connect s);
  Remote.ping r;
  Remote.close r;
  (* The server frees the session once the client has gone. *)
  let rec retry k =
    match connect s with
    | r -> r
    | exception Failure _ when k > 0 ->
        Unix.sleepf 0.01;
        retry (k - 1)
  in
  Remote.close (retry 500)

let test_memory () =
  with_server @@ fun s ->
  let r = connect s in
  let n = 3 * 4096 in
  let a = Option.get (Remote.alloc r n) in
  let local = alloc n and back = alloc n in
  let m = Mmio.v local n in
  for i = 0 to n - 1 do
    Mmio.set8 m i (i * 7)
  done;
  Remote.write r ~dst:a ~src:local n;
  Remote.copy r ~dst:(Nativeint.add a 16n) ~src:a 64;
  Remote.read r ~src:a ~dst:back n;
  let b = Mmio.v back n in
  equal ~msg:"the copy overlapped forwards" int (Mmio.get8 m 0) (Mmio.get8 b 16);
  equal ~msg:"the tail came back" string
    (Mmio.read m 80 (n - 80))
    (Mmio.read b 80 (n - 80));
  raises_match (Exn.failure ~substring:"no memory of this connection")
    (fun () ->
      Remote.read r ~src:(Nativeint.add a (Nativeint.of_int n)) ~dst:back 1);
  Remote.ping r;
  let remote = Mmio.remote (Remote.access r) a n in
  Mmio.set32 remote 4 0xcafe_f00d;
  equal ~msg:"a remote word" int 0xcafe_f00d (Mmio.get32 remote 4);
  Mmio.set64 remote 8 0x1122_3344_5566_7788L;
  equal ~msg:"a remote double word" int64 0x1122_3344_5566_7788L
    (Mmio.get64 remote 8);
  Mmio.fill remote 100 5000 'z';
  equal ~msg:"a remote fill" string (String.make 5000 'z')
    (Mmio.read remote 100 5000);
  raises_match (Exn.invalid_arg ~substring:"another machine") (fun () ->
      Mmio.bigarray remote);
  Remote.free r a;
  raises_match (Exn.failure ~substring:"no memory") (fun () ->
      Remote.read r ~src:a ~dst:back 1);
  Remote.close r

let test_posted_failure () =
  with_server @@ fun s ->
  let r = connect s in
  let local = alloc 16 in
  (* A posted write outside the connection's memory has no answer: the next
     command reads the error, and the connection is gone for good. *)
  Remote.write r ~dst:0x1000n ~src:local 16;
  raises_match (Exn.failure ~substring:"no memory of this connection")
    (fun () -> Remote.ping r);
  is_some ~msg:"failed" (Remote.failed r);
  raises_match (Exn.failure ~substring:"no memory of this connection")
    (fun () -> Remote.ping r)

let test_cleanup () =
  with_server @@ fun s ->
  let r = connect s in
  let a = Option.get (Remote.alloc r 4096) in
  Remote.close r;
  let rec retry k =
    match connect s with
    | r -> r
    | exception Failure _ when k > 0 ->
        Unix.sleepf 0.01;
        retry (k - 1)
  in
  let r = retry 500 in
  raises_match (Exn.failure ~substring:"no memory of this connection")
    (fun () -> Remote.read r ~src:a ~dst:(alloc 8) 8);
  Remote.close r

let test_functions () =
  with_server @@ fun s ->
  let r = connect s in
  let buses = Remote.scan r ~vendor:0xffff [ (0xffff, [ 0xffff ]) ] in
  equal ~msg:"no such function" (list string) [] buses;
  raises_match (Exn.failure ~substring:"") (fun () ->
      Pci.take ~remote:r ~lock:"test" "0000:ff:1f.7");
  raises_match (Exn.failure ~substring:"no lock name") (fun () ->
      Pci.take ~remote:r ~lock:"../test" "0000:ff:1f.7");
  raises_match (Exn.failure ~substring:"no PCI address") (fun () ->
      Pci.take ~remote:r ~lock:"test" "../../0000:ff:1f.7");
  raises_match (Exn.failure ~substring:"no function") (fun () ->
      Remote.read_config r 3 0 4);
  Remote.ping r;
  Remote.close r

let test_timeout () =
  (* A listener that never speaks. *)
  let socket = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind socket loopback;
  Unix.listen socket 1;
  let p =
    match Unix.getsockname socket with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  Fun.protect
    ~finally:(fun () -> Unix.close socket)
    (fun () ->
      raises_match (Exn.failure ~substring:"no answer within 200 ms") (fun () ->
          Remote.connect ~timeout_ms:200 ~key "127.0.0.1" p))

(* The machine's system memory, as a GPU there gets it: pinned, anywhere or at
   an address of a reserved range, and its host memory pinned for a GPU that
   maps it. Skips where the machine cannot lock memory or read its pages. *)
let test_remote_sysmem () =
  with_server @@ fun s ->
  let r = connect s in
  let base = 0x7e00_0000_0000 in
  (* An address outside the connection's reservations is refused before anything
     is mapped there. *)
  raises_match (Exn.failure ~substring:"outside this connection's reservations")
    (fun () -> Remote.alloc_sysmem r ~va:base 4096);
  (match Remote.alloc_sysmem r (3 * Remote.page r) with
  | exception Failure why -> skip ~reason:why ()
  | m, pages ->
      equal ~msg:"a page each" int 3 (List.length pages);
      is_true ~msg:"another machine's range" (Mmio.is_remote m);
      Mmio.set64 m 8 0x5a5aL;
      equal ~msg:"its memory" int64 0x5a5aL (Mmio.get64 m 8);
      Remote.free_sysmem r m);
  Remote.reserve r ~base (4 lsl 20);
  let m, _ = Remote.alloc_sysmem r ~va:base (1 lsl 20) in
  equal ~msg:"at the address asked" nativeint (Nativeint.of_int base)
    (Mmio.address m);
  raises_match (Exn.failure ~substring:"overlaps memory of this connection")
    (fun () -> Remote.alloc_sysmem r ~va:(base + 4096) 4096);
  raises_match (Exn.failure ~substring:"outside") (fun () ->
      Remote.alloc_sysmem r ~va:(base + (4 lsl 20) - 4096) 8192);
  Remote.free_sysmem r m;
  let n = 4 * Remote.page r in
  let a = Option.get (Remote.alloc r n) in
  equal ~msg:"pinned pages" int 4 (List.length (Remote.pin r a n));
  raises_match (Exn.failure ~substring:"is pinned") (fun () -> Remote.free r a);
  Remote.unpin r a n;
  Remote.free r a;
  raises_match (Exn.failure ~substring:"is not pinned") (fun () ->
      Remote.unpin r a n);
  Remote.close r;
  (* The reservation went with its client: another range over it can be
     reserved. *)
  let rec retry k =
    match connect s with
    | r -> r
    | exception Failure _ when k > 0 ->
        Unix.sleepf 0.01;
        retry (k - 1)
  in
  let r = retry 300 in
  Remote.reserve r ~base:(base + (2 lsl 20)) (4 lsl 20);
  Remote.close r

(* Lock files: a planted link is not followed, and nothing but a regular file
   serves. *)
let test_lock_files () =
  if Sys.win32 then skip ~reason:"PCI functions need a POSIX system" ();
  let dir = Filename.temp_dir "nx_lock" "" in
  let previous = Filename.get_temp_dir_name () in
  Filename.set_temp_dir_name dir;
  Fun.protect
    ~finally:(fun () -> Filename.set_temp_dir_name previous)
    (fun () ->
      let bus = "0000:ff:1f.7" in
      let target = Filename.concat dir "target" in
      Unix.symlink target (Filename.concat dir ("nx_" ^ bus ^ ".lock"));
      raises_match (Exn.failure ~substring:"") (fun () ->
          Pci.take ~lock:"t" bus);
      is_false ~msg:"the link's target is not created" (Sys.file_exists target);
      Sys.remove (Filename.concat dir ("nx_" ^ bus ^ ".lock"));
      Unix.mkdir (Filename.concat dir ("nx_" ^ bus ^ ".lock")) 0o700;
      raises_match (Exn.failure ~substring:"") (fun () ->
          Pci.take ~lock:"t" bus))

(* Key files: read once, of this user alone, and refused with the reason. *)
let test_key_files () =
  if not Sys.unix then skip ~reason:"file owners and modes are POSIX" ();
  let dir = Filename.temp_dir "nx_key" "" in
  let file name contents perm =
    let f = Filename.concat dir name in
    Out_channel.with_open_bin f (fun oc -> output_string oc contents);
    Unix.chmod f perm;
    f
  in
  let good = file "good" (String.make 32 'k') 0o600 in
  equal ~msg:"its bytes" string (String.make 32 'k') (Remote.read_key good);
  let refused ~sub f =
    raises_match (Exn.failure ~substring:sub) (fun () -> Remote.read_key f)
  in
  refused ~sub:"chmod 600" (file "open" (String.make 32 'k') 0o644);
  refused ~sub:"16 to 4096" (file "short" "hunter2" 0o600);
  refused ~sub:"16 to 4096" (file "long" (String.make 5000 'k') 0o600);
  refused ~sub:"No such file" (Filename.concat dir "missing");
  let fifo = Filename.concat dir "fifo" in
  Unix.mkfifo fifo 0o600;
  refused ~sub:"not a regular file" fifo;
  if Unix.geteuid () <> 0 then
    refused ~sub:"Permission denied"
      (file "unreadable" (String.make 32 'k') 0o000)

(* Peers that do not know the key: raw sockets to the server. *)

let raw s =
  let fd = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, port s));
  fd

(* The status byte of the server's first message: 0 when it sends its nonce. *)
let hello_status fd =
  let b = Bytes.create 13 in
  let rec go off =
    if off < 13 then
      match Unix.read fd b off (13 - off) with 0 -> () | k -> go (off + k)
  in
  go 0;
  Bytes.get_uint8 b 12

(* Connections reset before the server even sets them up. *)
let test_resets () =
  with_server @@ fun s ->
  for _ = 1 to 20 do
    let fd = raw s in
    Unix.setsockopt_optint fd Unix.SO_LINGER (Some 0);
    Unix.close fd
  done;
  let r = connect s in
  Remote.ping r;
  Remote.close r

(* A peer that says nothing, and one that sends a wrong proof, hold no slot: a
   client with the key is served meanwhile. *)
let test_silent_peers () =
  with_server @@ fun s ->
  let silent = raw s and wrong = raw s in
  equal ~msg:"a nonce for everyone" int 0 (hello_status silent);
  ignore (hello_status wrong);
  ignore (Unix.write_substring wrong (String.make 64 'x') 0 64);
  let r = connect s in
  Remote.ping r;
  Remote.close r;
  Unix.close silent;
  Unix.close wrong

(* Beyond the handshakes the server runs at once, a connection is told so and
   closed; once they end, a client with the key is served. *)
let test_handshake_cap () =
  with_server @@ fun s ->
  let idle = List.init 64 (fun _ -> raw s) in
  List.iter (fun fd -> ignore (hello_status fd)) idle;
  let extra = raw s in
  equal ~msg:"too many" int 1 (hello_status extra);
  Unix.close extra;
  List.iter Unix.close idle;
  let rec retry k =
    match connect s with
    | r -> r
    | exception Failure _ when k > 0 ->
        Unix.sleepf 0.01;
        retry (k - 1)
  in
  Remote.close (retry 300)

(* A scan naming many ids answers, and the connection stays usable. *)
let test_large_probe () =
  with_server @@ fun s ->
  let r = connect s in
  equal ~msg:"no such function" (list string) []
    (Remote.scan r ~vendor:0xffff [ (0xffff, List.init 1_000_000 Fun.id) ]);
  Remote.ping r;
  Remote.close r

(* A program that raises what the server does not expect ends that session
   through its cleanup, and the next client is served. *)
let test_unexpected_exception () =
  let programs =
    {
      Remote_server.load = (fun ~binary:_ ~name:_ -> 0);
      call = (fun _ _ _ -> raise Exit);
      unload = ignore;
    }
  in
  with_server ~programs @@ fun s ->
  let r = connect s in
  let p = Remote.load r ~binary:"" ~name:"f" in
  Remote.call r p [||] [||];
  raises_match (Exn.failure ~substring:"Exit") (fun () -> Remote.ping r);
  let rec retry k =
    match connect s with
    | r -> r
    | exception Failure _ when k > 0 ->
        Unix.sleepf 0.01;
        retry (k - 1)
  in
  let r = retry 300 in
  Remote.ping r;
  Remote.close r

let test_stop () =
  let s = Remote_server.listen ~key loopback in
  let r = connect s in
  Remote_server.stop s;
  raises_match (Exn.failure ~substring:"") (fun () -> Remote.ping r);
  is_some ~msg:"the connection failed" (Remote.failed r)

let () =
  exit
    (run "nx.device.support"
       [
         group "mmio" [ test "access" test_mmio ];
         group "tlsf" [ test_tlsf; test "refusals" test_tlsf_refusals ];
         group "page tables"
           [
             test "4 KiB pages" test_map_pages;
             test "large pages" test_large_pages;
             test "fragments" test_fragments;
             test "out of table memory" test_out_of_tables;
             test_tables_freed;
             test "alloc and free" test_alloc;
             test "the boot pool" test_boot_pool;
             test "entries by level" test_entry_levels;
             test "the tables of a range" test_path;
             test "a base of their own" test_base;
           ];
         group "sysmem"
           [
             test "pins are counted" test_sysmem_counted_pins;
             test "allocated memory stays locked" test_sysmem_pins;
             test "contiguous memory" test_sysmem_contiguous;
           ];
         group "remote"
           [
             test "handshake" test_handshake;
             test "the server proves the key" test_server_proves;
             test "one client at a time" test_busy;
             test "memory" test_memory;
             test "a posted command fails the connection" test_posted_failure;
             test "cleanup at disconnect" test_cleanup;
             test "functions" test_functions;
             test "timeout" test_timeout;
             test "system memory" test_remote_sysmem;
             test "stop" test_stop;
             test "reset connections" test_resets;
             test "peers without the key hold no slot" test_silent_peers;
             test "handshakes at once" test_handshake_cap;
             test "a large scan" test_large_probe;
             test "an unexpected exception" test_unexpected_exception;
             test "lock files" test_lock_files;
             test "key files" test_key_files;
           ];
         group "firmware"
           [
             test "sha256" test_sha256;
             test "a directory" test_firmware_dir;
             test "compressed files" test_firmware_compressed;
             test "downloads and the cache" test_firmware_download;
           ];
       ])
