(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module File_io = Tolk_hcq.Hcq.File_io
module Mmio = Tolk_hcq.Hcq.Mmio
module Buffer = Tolk_hcq.Hcq.Buffer
module Q = Tolk_hcq.Hcq.Q
module Signal = Tolk_hcq.Hcq.Signal
module Timeline = Tolk_hcq.Hcq.Timeline
module Kernargs = Tolk_hcq.Hcq.Kernargs
module Tables = Tolk_nv.Nv_tables
module Defs = Tolk_nv.Nv_tables.Defs
module Qmd = Tolk_nv.Qmd
module Compute_queue = Tolk_nv.Compute_queue
module Copy_queue = Tolk_nv.Copy_queue
module Nv_iface = Tolk_nv.Nv_iface
module Nvk_iface = Tolk_nv.Nvk_iface
module Program = Tolk_nv.Program

let is_invalid_arg = function Invalid_argument _ -> true | _ -> false

let with_map size f =
  let addr =
    File_io.mmap ~addr:0n ~size
      ~prot:(File_io.prot_read lor File_io.prot_write)
      ~flags:(File_io.map_private lor File_io.map_anonymous)
      ~fd:(-1) ~offset:0L
  in
  Fun.protect
    ~finally:(fun () -> File_io.munmap addr ~size)
    (fun () -> f (Mmio.make ~addr ~size))

(* One 0x8000-byte anonymous mapping backs everything a test touches:
   the command staging page at 0, the usermode register region at
   0x1000, ring and put pointer at 0x2000, a signal slot at 0x3000, two
   kernel-argument areas at 0x4000 and 0x5000, and template descriptor
   storage at 0x6000. Device addresses are made up and distinct from
   the CPU mapping. *)
let with_fixture f = with_map 0x8000 f

let nv_dev ?(compute_class = Defs.ada_compute_a) ?(cmdq_size = 0x1000)
    ?(sass_version = 0x59) ?slm_per_thread m =
  Tolk_nv.device ~compute_class ~dma_class:Defs.ampere_dma_copy_b
    ~gpfifo_class:Defs.ampere_channel_gpfifo_a ~sass_version ?slm_per_thread
    ~shared_mem_window:0x729400000000n ~local_mem_window:0x729300000000n
    ~cmdq_page:
      (Buffer.make ~va:0x400000n ~size:cmdq_size
         ~view:(Mmio.view m ~off:0 ~size:cmdq_size ())
         ~meta:() ())
    ~gpu_mmio:(Mmio.view m ~off:0x1000 ~size:0x1000 ())
    ()

let queue_desc ?(entries = 8) m =
  {
    Tolk_nv.Queue_desc.ring = Mmio.view m ~off:0x2000 ~size:(entries * 8) ();
    gpput = Mmio.view m ~off:(0x2000 + (entries * 8)) ~size:4 ();
    token = 0x1abcd;
    put_value = 0;
  }

(* A signal whose value address has non-zero top bits, so encodings that
   split it are visible. *)
let signal m =
  Signal.make
    (Buffer.make ~va:0x200000010n ~size:16
       ~view:(Mmio.view m ~off:0x3000 ~size:16 ())
       ~meta:() ())

let kernarg1 m =
  Buffer.make ~va:0x30000000n ~size:0x1000
    ~view:(Mmio.view m ~off:0x4000 ~size:0x1000 ())
    ~meta:() ()

let kernarg2 m =
  Buffer.make ~va:0x40000000n ~size:0x1000
    ~view:(Mmio.view m ~off:0x5000 ~size:0x1000 ())
    ~meta:() ()

let template_qmd ~compute_class m =
  Qmd.create
    ~view:
      (Mmio.view m ~off:0x6000 ~size:(Qmd.sizeof ~compute_class) ())
    ~compute_class

let nv_prog ?(cbuf0_size = 0x160) (dev : unit Tolk_nv.device) m =
  {
    Tolk_nv.dev;
    qmd = template_qmd ~compute_class:dev.Tolk_nv.compute_class m;
    cbuf0_size;
  }

let dwords cq = Q.dwords (Compute_queue.q cq)
let copy_dwords cq = Q.dwords (Copy_queue.q cq)

(* Wire-format tests compare structures as hex strings so a mismatch
   shows the whole layout. *)
let blob_hex (b : Tables.blob) =
  String.concat ""
    (List.init (Bigarray.Array1.dim b) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bigarray.Array1.get b i))))

let bytes_hex b =
  String.concat ""
    (List.init (Bytes.length b) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bytes.get b i))))

let version_blob s =
  let b =
    Tables.create_blob
      Defs.Nv0000_ctrl_system_get_build_version_v2_params.sizeof
  in
  String.iteri (fun i c -> Bigarray.Array1.set b i c) s;
  b

(* Opening the kernel driver needs Linux, the driver and a device; the
   first failure to do so skips every device test. *)
let nvk_iface =
  let cached = ref None in
  fun () ->
    match !cached with
    | Some i -> i
    | None -> (
        try
          let i = Nvk_iface.iface ~device_id:0 in
          cached := Some i;
          i
        with Failure msg -> skip ~reason:msg ())

(* The kernel-argument area's descriptor copy: cbuf0_size 0x160 rounds
   up to 0x200. *)
let exec_qmd ~compute_class m ~kernarg_off =
  Qmd.create
    ~view:
      (Mmio.view m ~off:(kernarg_off + 0x200) ~size:(Qmd.sizeof ~compute_class)
         ())
    ~compute_class

let set16 b off v = Bytes.set_uint16_le b off v
let set32 b off v = Bytes.set_int32_le b off (Int32.of_int v)
let set64 b off v = Bytes.set_int64_le b off (Int64.of_int v)

(* Fresh anonymous mappings back the buffers Program.load writes through;
   they live for the rest of the test process. *)
let anon_mmio size =
  let addr =
    File_io.mmap ~addr:0n ~size
      ~prot:(File_io.prot_read lor File_io.prot_write)
      ~flags:(File_io.map_private lor File_io.map_anonymous)
      ~fd:(-1) ~offset:0L
  in
  Mmio.make ~addr ~size

(* Hand-crafted 64-bit little-endian shared object shaped like a cubin
   for a kernel named "k": [.text.k] at image address 0x2000, constant
   banks 0 and 3 at 0x12000 and 0x1a000, a shared-memory carrier of 0x80
   bytes, the global and per-kernel [.nv.info] descriptor sections, and
   three relocations patching [.text.k] against the constant banks.
   Field values are spelled as literals so the loader's parsing is
   checked against independently written numbers. *)
let cubin_fixture ?(reloc0 = 2) ?(undefined_sym = false) ?(bad_info = false)
    ?(regcount = 32) () =
  let module Buf = Stdlib.Buffer in
  let entry b typ param sz =
    Buf.add_char b (Char.chr typ);
    Buf.add_char b (Char.chr param);
    Buf.add_char b (Char.chr (sz land 0xff));
    Buf.add_char b (Char.chr (sz lsr 8))
  in
  let word b v =
    for i = 0 to 3 do
      Buf.add_char b (Char.chr ((v lsr (8 * i)) land 0xff))
    done
  in
  let info = Buf.create 64 in
  if bad_info then entry info 7 0x99 0;
  (* a short-form value entry, then MIN_STACK_SIZE 0x140, REGCOUNT, and
     an ignored payload entry (EXIT_INSTR_OFFSETS) *)
  entry info 3 0x25 0x10;
  entry info 4 0x12 8;
  word info 4;
  word info 0x140;
  entry info 4 0x2f 8;
  word info 4;
  word info regcount;
  entry info 4 0x1c 4;
  word info 0x120;
  let info_k = Buf.create 32 in
  (* an ignored payload entry (KPARAM_INFO), then PARAM_CBANK: a 32-bit
     bank ordinal followed by the 16-bit bank size *)
  entry info_k 4 0x17 12;
  word info_k 0;
  word info_k 0;
  word info_k 0;
  entry info_k 4 0xa 8;
  word info_k 0x18;
  word info_k 0x160;
  let info_b = Buf.to_bytes info and info_k_b = Buf.to_bytes info_k in
  let symtab = Bytes.make 72 '\000' in
  set32 symtab 24 1 (* st_name: "c0" *);
  set16 symtab 30 (if undefined_sym then 0 else 3) (* st_shndx *);
  set64 symtab 32 0x10 (* st_value *);
  set32 symtab 48 4 (* st_name: "c3" *);
  set16 symtab 54 4;
  set64 symtab 56 0x20;
  let strtab = Bytes.of_string "\000c0\000c3\000" in
  let rela = Bytes.make 72 '\000' in
  set64 rela 0 0x100 (* r_offset, within .text.k *);
  set64 rela 8 ((1 lsl 32) lor reloc0);
  set64 rela 24 0x200;
  set64 rela 32 ((1 lsl 32) lor 0x38);
  set64 rela 48 0x300;
  set64 rela 56 ((2 lsl 32) lor 0x39);
  let buf = Buf.create 16384 in
  Buf.add_bytes buf (Bytes.make 64 '\000');
  let pad_to align =
    while Buf.length buf mod align <> 0 do
      Buf.add_char buf '\000'
    done
  in
  let add align content =
    pad_to align;
    let off = Buf.length buf in
    Buf.add_bytes buf content;
    off
  in
  let off_text = add 8 (Bytes.make 0x1800 '\xcc') in
  let off_const0 = add 8 (Bytes.make 0x160 '\xaa') in
  let off_const3 = add 8 (Bytes.make 0x200 '\xbb') in
  let off_info = add 4 info_b in
  let off_info_k = add 4 info_k_b in
  let off_symtab = add 8 symtab in
  let off_strtab = add 1 strtab in
  let off_rela = add 8 rela in
  let shstr = Buf.create 128 in
  Buf.add_char shstr '\000';
  let name s =
    let off = Buf.length shstr in
    Buf.add_string shstr s;
    Buf.add_char shstr '\000';
    off
  in
  let n_text = name ".text.k" in
  let n_shared = name ".nv.shared.k" in
  let n_const0 = name ".nv.constant0.k" in
  let n_const3 = name ".nv.constant3.k" in
  let n_info = name ".nv.info" in
  let n_info_k = name ".nv.info.k" in
  let n_symtab = name ".symtab" in
  let n_strtab = name ".strtab" in
  let n_rela = name ".rela.text.k" in
  let n_shstrtab = name ".shstrtab" in
  let shstr_b = Buf.to_bytes shstr in
  let off_shstr = add 1 shstr_b in
  pad_to 8;
  let e_shoff = Buf.length buf in
  let shdr ~nm ~ty ~flags ~addr ~off ~size ~link ~info ~salign ~entsize =
    let b = Bytes.make 64 '\000' in
    set32 b 0 nm;
    set32 b 4 ty;
    set64 b 8 flags;
    set64 b 16 addr;
    set64 b 24 off;
    set64 b 32 size;
    set32 b 40 link;
    set32 b 44 info;
    set64 b 48 salign;
    set64 b 56 entsize;
    Buf.add_bytes buf b
  in
  shdr ~nm:0 ~ty:0 ~flags:0 ~addr:0 ~off:0 ~size:0 ~link:0 ~info:0 ~salign:0
    ~entsize:0;
  shdr ~nm:n_text ~ty:1 ~flags:0x6 ~addr:0x2000 ~off:off_text ~size:0x1800
    ~link:0 ~info:0 ~salign:128 ~entsize:0;
  shdr ~nm:n_shared ~ty:8 ~flags:0 ~addr:0 ~off:0 ~size:0x80 ~link:0 ~info:0
    ~salign:16 ~entsize:0;
  shdr ~nm:n_const0 ~ty:1 ~flags:0x2 ~addr:0x12000 ~off:off_const0 ~size:0x160
    ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_const3 ~ty:1 ~flags:0x2 ~addr:0x1a000 ~off:off_const3 ~size:0x200
    ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_info ~ty:1 ~flags:0 ~addr:0 ~off:off_info
    ~size:(Bytes.length info_b) ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_info_k ~ty:1 ~flags:0 ~addr:0 ~off:off_info_k
    ~size:(Bytes.length info_k_b) ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_symtab ~ty:2 ~flags:0 ~addr:0 ~off:off_symtab ~size:72 ~link:8
    ~info:1 ~salign:8 ~entsize:24;
  shdr ~nm:n_strtab ~ty:3 ~flags:0 ~addr:0 ~off:off_strtab
    ~size:(Bytes.length strtab) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  shdr ~nm:n_rela ~ty:4 ~flags:0 ~addr:0 ~off:off_rela ~size:72 ~link:7 ~info:1
    ~salign:8 ~entsize:24;
  shdr ~nm:n_shstrtab ~ty:3 ~flags:0 ~addr:0 ~off:off_shstr
    ~size:(Bytes.length shstr_b) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  let obj = Buf.to_bytes buf in
  Bytes.blit_string "\x7fELF\x02\x01\x01" 0 obj 0 7;
  set16 obj 16 3 (* e_type: ET_DYN *);
  set16 obj 18 190 (* e_machine: CUDA *);
  set32 obj 20 1 (* e_version *);
  set64 obj 40 e_shoff;
  set16 obj 52 64 (* e_ehsize *);
  set16 obj 58 64 (* e_shentsize *);
  set16 obj 60 11 (* e_shnum *);
  set16 obj 62 10 (* e_shstrndx *);
  obj

(* A [Program.load]-ready allocator recording the sizes it served; the
   device address defaults to putting [.text.k] at 0x100000, the address
   the qmd_init goldens pin. *)
let lib_alloc ?(va = 0xfe000n) () =
  let sizes = ref [] in
  let alloc size =
    sizes := size :: !sizes;
    Buffer.make ~va ~size ~view:(anon_mmio size) ~meta:() ()
  in
  (alloc, sizes)

let load_fixture ?va ?lib dev =
  let alloc, _ = lib_alloc ?va () in
  let lib = match lib with Some l -> l | None -> cubin_fixture () in
  Program.load dev ~alloc ~ensure_local_memory:(fun _ -> ()) ~name:"k" lib

let qmd_template_dwords prg =
  let b = Qmd.to_bytes prg.Program.params.Tolk_nv.qmd in
  Array.init (Bytes.length b / 4) (fun i ->
      Int32.to_int (Bytes.get_int32_le b (4 * i)) land 0xffffffff)

let staged_dwords m ~off n =
  Array.init n (fun i -> Int32.to_int (Mmio.read32 m (off + (4 * i))) land 0xffffffff)

(* The nonzero dwords of the qmd_init goldens
   (test/golden/nvqueue/qmd_init_{ada,blackwell}.expected), which pin the
   reference template for the same program descriptor. *)
let qmd_expected words entries =
  let a = Array.make words 0 in
  List.iter (fun (i, v) -> a.(i) <- v) entries;
  a

let qmd_expected_ada =
  qmd_expected 64
    [
      (4, 0x0000007f); (5, 0x3c000000); (8, 0x00001000); (11, 0x44010000);
      (17, 0x34240480); (18, 0x00000030); (20, 0x00122009); (23, 0x08000000);
      (32, 0x00110000); (33, 0x0b040000); (38, 0x00118000); (39, 0x10000000);
      (48, 0x00100000); (50, 0x00000240); (51, 0x89003000);
    ]

let qmd_expected_blackwell =
  qmd_expected 96
    [
      (4, 0x013f0000); (14, 0x0f5003a4); (19, 0x00010000); (32, 0x00010000);
      (33, 0x03000000); (35, 0x00022000); (36, 0x04b44809); (37, 0x00240000);
      (42, 0x00004400); (43, 0x0b000000); (48, 0x00004600); (49, 0x10000000);
      (58, 0x00001009); (59, 0x00001000);
    ]

(* A ready timeline over two mapped slots; the counter starts at 1 as on
   a fresh device. *)
let timeline m =
  let sig_at off va =
    Signal.make ~is_timeline:true
      (Buffer.make ~va ~size:16 ~view:(Mmio.view m ~off ~size:16 ()) ~meta:() ())
  in
  {
    Timeline.timeline = sig_at 0x3000 0x200000010n;
    shadow_timeline = sig_at 0x3010 0x200000020n;
    timeline_value = 1;
    error_state = None;
    bounce = [||];
    bounce_timeline = [||];
    bounce_next = 0;
    on_hang = (fun () -> ());
  }

let failure_with prefix = function
  | Failure msg -> String.starts_with ~prefix msg
  | _ -> false

let () =
  run "Nv_runtime"
    [
      group "qmd"
        [
          test "layout follows the compute class" (fun () ->
              equal int 0x100 (Qmd.sizeof ~compute_class:Defs.ada_compute_a);
              equal int 0x180
                (Qmd.sizeof ~compute_class:Defs.blackwell_compute_b);
              with_fixture (fun m ->
                  let v3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  let v5 =
                    template_qmd ~compute_class:Defs.blackwell_compute_b m
                  in
                  equal int 3 (Qmd.version v3);
                  equal int 5 (Qmd.version v5);
                  equal int 0x100 (Bytes.length (Qmd.to_bytes v3));
                  equal int 0x180 (Bytes.length (Qmd.to_bytes v5))));
          test "a view smaller than the descriptor is rejected" (fun () ->
              with_fixture (fun m ->
                  raises_match is_invalid_arg (fun () ->
                      Qmd.create
                        ~view:(Mmio.view m ~off:0 ~size:0xff ())
                        ~compute_class:Defs.ada_compute_a)));
          test "field writes round-trip and land at the table offsets"
            (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  (* CTA_RASTER_WIDTH is bits 415..384: bytes 48-51. *)
                  Qmd.write q3 [ ("cta_raster_width", 0xdeadbeef) ];
                  equal int 0xdeadbeef (Qmd.read q3 "cta_raster_width");
                  equal int 48 (Qmd.field_offset q3 "cta_raster_width");
                  let b = Qmd.to_bytes q3 in
                  equal int 0xef (Char.code (Bytes.get b 48));
                  equal int 0xbe (Char.code (Bytes.get b 49));
                  equal int 0xad (Char.code (Bytes.get b 50));
                  equal int 0xde (Char.code (Bytes.get b 51))));
          test "names are case-insensitive" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  Qmd.write q3 [ ("CTA_Raster_Width", 7) ];
                  equal int 7 (Qmd.read q3 "cta_raster_width")));
          test "an unaligned field leaves its byte-sharing neighbours"
            (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  (* CONSTANT_BUFFER_ADDR_UPPER_0 (bits 1072..1056) and
                     CONSTANT_BUFFER_SIZE_SHIFTED4_0 (bits 1087..1075)
                     share byte 134. *)
                  Qmd.write q3 [ ("constant_buffer_addr_upper_0", 0x1ffff) ];
                  Qmd.write q3 [ ("constant_buffer_size_shifted4_0", 0x1abc) ];
                  equal int 0x1ffff (Qmd.read q3 "constant_buffer_addr_upper_0");
                  equal int 0x1abc
                    (Qmd.read q3 "constant_buffer_size_shifted4_0")));
          test "per-slot fields address distinct slots" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  Qmd.write q3
                    [
                      ("constant_buffer_addr_lower_0", 0x11111111);
                      ("constant_buffer_addr_lower_3", 0x33333333);
                    ];
                  equal int 0x11111111
                    (Qmd.read q3 "constant_buffer_addr_lower_0");
                  equal int 0x33333333
                    (Qmd.read q3 "constant_buffer_addr_lower_3");
                  equal int 128 (Qmd.field_offset q3 "constant_buffer_addr_lower_0");
                  equal int 152 (Qmd.field_offset q3 "constant_buffer_addr_lower_3")));
          test "unknown names and oversized values are rejected" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  raises_match is_invalid_arg (fun () ->
                      Qmd.read q3 "not_a_field");
                  raises_match is_invalid_arg (fun () ->
                      Qmd.write q3 [ ("not_a_field", 1) ]);
                  raises_match is_invalid_arg (fun () ->
                      Qmd.write q3 [ ("release0_enable", 2) ])));
          test "constant buffer addresses split per version" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  Qmd.set_constant_buf_addr q3 0 0x234500000n;
                  equal int 0x34500000
                    (Qmd.read q3 "constant_buffer_addr_lower_0");
                  equal int 2 (Qmd.read q3 "constant_buffer_addr_upper_0");
                  let q5 =
                    template_qmd ~compute_class:Defs.blackwell_compute_b m
                  in
                  Qmd.set_constant_buf_addr q5 0 0x234500000n;
                  equal int 0x08d14000
                    (Qmd.read q5 "constant_buffer_addr_lower_shifted6_0");
                  equal int 0
                    (Qmd.read q5 "constant_buffer_addr_upper_shifted6_0")));
        ];
      group "compute stream"
        [
          test "wait" (fun () ->
              with_fixture (fun m ->
                  let cq = Compute_queue.create (nv_dev m) in
                  Compute_queue.wait cq ~value:5 (signal m);
                  equal (array int)
                    [| 0x20050017; 0x10; 2; 5; 0; 0x01000003 |]
                    (dwords cq)));
          test "signal without a pending launch" (fun () ->
              with_fixture (fun m ->
                  let cq = Compute_queue.create (nv_dev m) in
                  Compute_queue.signal cq ~value:7 (signal m);
                  equal (array int)
                    [|
                      0x20050017; 0x10; 2; 7; 0; 0x03100001; 0x20010008; 0;
                    |]
                    (dwords cq)));
          test "timestamp is a zero-valued signal" (fun () ->
              with_fixture (fun m ->
                  let cq = Compute_queue.create (nv_dev m) in
                  Compute_queue.timestamp cq (signal m);
                  equal (array int)
                    [|
                      0x20050017; 0x10; 2; 0; 0; 0x03100001; 0x20010008; 0;
                    |]
                    (dwords cq)));
          test "memory_barrier" (fun () ->
              with_fixture (fun m ->
                  let cq = Compute_queue.create (nv_dev m) in
                  Compute_queue.memory_barrier cq;
                  equal (array int) [| 0x200125a6; 0x1011 |] (dwords cq)));
          test "write selects the payload size" (fun () ->
              with_fixture (fun m ->
                  let buf = Buffer.make ~va:0x50000000n ~size:16 ~meta:() () in
                  let cq = Compute_queue.create (nv_dev m) in
                  Compute_queue.write cq ~b64:true buf 0x100000002L;
                  Compute_queue.write cq buf 5L;
                  equal (array int)
                    [|
                      0x20050017; 0x50000000; 0; 2; 1; 0x01100001;
                      0x20050017; 0x50000000; 0; 5; 0; 0x00100001;
                    |]
                    (dwords cq)));
          test "poll_bit waits for set or clear bits" (fun () ->
              with_fixture (fun m ->
                  let buf = Buffer.make ~va:0x50000000n ~size:16 ~meta:() () in
                  let cq = Compute_queue.create (nv_dev m) in
                  Compute_queue.poll_bit cq buf ~value:0x8 ~mask:0x8;
                  Compute_queue.poll_bit cq buf ~value:0 ~mask:0x8;
                  equal (array int)
                    [|
                      0x20050017; 0x50000000; 0; 0x8; 0; 0x4;
                      0x20050017; 0x50000000; 0; 0xfffffff7; 0; 0x5;
                    |]
                    (dwords cq)));
          test "setup emits one method per argument" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let cq = Compute_queue.create dev in
                  Compute_queue.setup cq ~compute_class:dev.Tolk_nv.compute_class
                    ~local_mem_window:dev.Tolk_nv.local_mem_window
                    ~shared_mem_window:dev.Tolk_nv.shared_mem_window
                    ~local_mem:0x123400000n ~local_mem_tpc_bytes:0x8000 ();
                  equal (array int)
                    [|
                      0x20012000; 0xc9c0;
                      0x200221ec; 0x7293; 0;
                      0x200220a8; 0x7294; 0;
                      0x200221e4; 0x1; 0x23400000;
                      0x200320b9; 0; 0x8000; 0xff;
                    |]
                    (dwords cq)));
        ];
      group "copy stream"
        [
          test "setup binds the copy class" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let cq = Copy_queue.create dev in
                  Copy_queue.setup cq ~copy_class:dev.Tolk_nv.dma_class ();
                  equal (array int) [| 0x20018000; 0xc7b5 |] (copy_dwords cq)));
          test "copy" (fun () ->
              with_fixture (fun m ->
                  let src = Buffer.make ~va:0x50000000n ~size:0x1000 ~meta:() ()
                  and dest =
                    Buffer.make ~va:0x60000000n ~size:0x1000 ~meta:() ()
                  in
                  let cq = Copy_queue.create (nv_dev m) in
                  Copy_queue.copy cq ~dest ~src 0x1000;
                  equal (array int)
                    [|
                      0x20048100; 0; 0x50000000; 0; 0x60000000;
                      0x20018106; 0x1000;
                      0x200180c0; 0x182;
                    |]
                    (copy_dwords cq)));
          test "a copy beyond 2 GiB chunks" (fun () ->
              with_fixture (fun m ->
                  let src = Buffer.make ~va:0x50000000n ~size:0 ~meta:() ()
                  and dest = Buffer.make ~va:0x160000000n ~size:0 ~meta:() () in
                  let cq = Copy_queue.create (nv_dev m) in
                  Copy_queue.copy cq ~dest ~src ((1 lsl 31) + 0x100);
                  equal (array int)
                    [|
                      0x20048100; 0; 0x50000000; 0x1; 0x60000000;
                      0x20018106; 0x80000000;
                      0x200180c0; 0x182;
                      0x20048100; 0; 0xd0000000; 0x1; 0xe0000000;
                      0x20018106; 0x100;
                      0x200180c0; 0x182;
                    |]
                    (copy_dwords cq)));
          test "signal flushes through a semaphore release" (fun () ->
              with_fixture (fun m ->
                  let cq = Copy_queue.create (nv_dev m) in
                  Copy_queue.signal cq ~value:3 (signal m);
                  equal (array int)
                    [| 0x20038090; 2; 0x10; 3; 0x200180c0; 0x14 |]
                    (copy_dwords cq)));
          test "wait matches the compute encoding" (fun () ->
              with_fixture (fun m ->
                  let cq = Copy_queue.create (nv_dev m) in
                  Copy_queue.wait cq ~value:5 (signal m);
                  equal (array int)
                    [| 0x20050017; 0x10; 2; 5; 0; 0x01000003 |]
                    (copy_dwords cq)));
        ];
      group "launch chaining"
        [
          test "exec copies the template and patches the geometry" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let prg = nv_prog dev m in
                  Qmd.write prg.qmd [ ("barrier_count", 3) ];
                  let cq = Compute_queue.create dev in
                  Compute_queue.exec cq prg ~kernargs:(kernarg1 m)
                    ~global_size:(0x30003, 5, 3) ~local_size:(33, 7, 2);
                  (* descriptor address 0x30000200 shifted right by 8 *)
                  equal (array int)
                    [| 0x200120ad; 0x300002; 0x200120b0; 9 |]
                    (dwords cq);
                  let q =
                    exec_qmd ~compute_class:dev.Tolk_nv.compute_class m
                      ~kernarg_off:0x4000
                  in
                  equal int 3 (Qmd.read q "barrier_count");
                  equal int 0x30003 (Qmd.read q "cta_raster_width");
                  equal int 5 (Qmd.read q "cta_raster_height");
                  equal int 3 (Qmd.read q "cta_raster_depth");
                  equal int 33 (Qmd.read q "cta_thread_dimension0");
                  equal int 7 (Qmd.read q "cta_thread_dimension1");
                  equal int 2 (Qmd.read q "cta_thread_dimension2");
                  equal int 0x30000000
                    (Qmd.read q "constant_buffer_addr_lower_0");
                  equal int 0 (Qmd.read q "constant_buffer_addr_upper_0")));
          test "exec uses the version-5 layout on Blackwell" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev ~compute_class:Defs.blackwell_compute_b m in
                  let prg = nv_prog dev m in
                  Qmd.write prg.qmd [ ("register_count", 42) ];
                  let cq = Compute_queue.create dev in
                  Compute_queue.exec cq prg ~kernargs:(kernarg1 m)
                    ~global_size:(0x30003, 5, 3) ~local_size:(33, 7, 2);
                  let q =
                    exec_qmd ~compute_class:dev.Tolk_nv.compute_class m
                      ~kernarg_off:0x4000
                  in
                  equal int 0x30003 (Qmd.read q "grid_width");
                  equal int 5 (Qmd.read q "grid_height");
                  equal int 3 (Qmd.read q "grid_depth");
                  equal int 33 (Qmd.read q "cta_thread_dimension0");
                  equal int 2 (Qmd.read q "cta_thread_dimension2");
                  (* the byte after the third dimension holds the register
                     count: the single-byte store must not clobber it *)
                  equal int 42 (Qmd.read q "register_count");
                  equal int (0x30000000 lsr 6)
                    (Qmd.read q "constant_buffer_addr_lower_shifted6_0")));
          test "a descriptor address above 40 bits is rejected" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let prg = nv_prog dev m in
                  let kernargs =
                    Buffer.make ~va:0x10000000000n ~size:0x1000
                      ~view:(Mmio.view m ~off:0x4000 ~size:0x1000 ())
                      ~meta:() ()
                  in
                  let cq = Compute_queue.create dev in
                  raises_match is_invalid_arg (fun () ->
                      Compute_queue.exec cq prg ~kernargs
                        ~global_size:(1, 1, 1) ~local_size:(1, 1, 1))));
          test "a second exec chains instead of launching" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let prg = nv_prog dev m in
                  let cq = Compute_queue.create dev in
                  Compute_queue.exec cq prg ~kernargs:(kernarg1 m)
                    ~global_size:(1, 1, 1) ~local_size:(1, 1, 1);
                  Compute_queue.exec cq prg ~kernargs:(kernarg2 m)
                    ~global_size:(1, 1, 1) ~local_size:(1, 1, 1);
                  (* no new stream methods for the chained launch *)
                  equal int 4 (Q.length (Compute_queue.q cq));
                  let q1 =
                    exec_qmd ~compute_class:dev.Tolk_nv.compute_class m
                      ~kernarg_off:0x4000
                  in
                  equal int 1 (Qmd.read q1 "dependent_qmd0_enable");
                  equal int 1 (Qmd.read q1 "dependent_qmd0_action");
                  equal int 1 (Qmd.read q1 "dependent_qmd0_prefetch");
                  (* second descriptor at 0x40000200, shifted right by 8 *)
                  equal int 0x400002 (Qmd.read q1 "dependent_qmd0_pointer")));
          test "wait, write, poll_bit and memory_barrier end the launch"
            (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let prg = nv_prog dev m in
                  let buf = Buffer.make ~va:0x50000000n ~size:16 ~meta:() () in
                  let break_with name f =
                    let cq = Compute_queue.create dev in
                    Compute_queue.exec cq prg ~kernargs:(kernarg1 m)
                      ~global_size:(1, 1, 1) ~local_size:(1, 1, 1);
                    let before = Q.length (Compute_queue.q cq) in
                    f cq;
                    let between = Q.length (Compute_queue.q cq) in
                    Compute_queue.exec cq prg ~kernargs:(kernarg2 m)
                      ~global_size:(1, 1, 1) ~local_size:(1, 1, 1);
                    equal ~msg:name int (between + 4)
                      (Q.length (Compute_queue.q cq));
                    let q1 =
                      exec_qmd ~compute_class:dev.Tolk_nv.compute_class m
                        ~kernarg_off:0x4000
                    in
                    equal ~msg:name int 0 (Qmd.read q1 "dependent_qmd0_enable");
                    is_true ~msg:name (between > before)
                  in
                  break_with "wait" (fun cq ->
                      Compute_queue.wait cq (signal m));
                  break_with "write" (fun cq -> Compute_queue.write cq buf 1L);
                  break_with "poll_bit" (fun cq ->
                      Compute_queue.poll_bit cq buf ~value:0 ~mask:1);
                  break_with "memory_barrier" (fun cq ->
                      Compute_queue.memory_barrier cq)));
          test "signal rides the pending descriptor's release slots"
            (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let prg = nv_prog dev m in
                  let cq = Compute_queue.create dev in
                  Compute_queue.exec cq prg ~kernargs:(kernarg1 m)
                    ~global_size:(1, 1, 1) ~local_size:(1, 1, 1);
                  Compute_queue.signal cq ~value:9 (signal m);
                  (* nothing appended: the release is in the descriptor *)
                  equal int 4 (Q.length (Compute_queue.q cq));
                  let q1 =
                    exec_qmd ~compute_class:dev.Tolk_nv.compute_class m
                      ~kernarg_off:0x4000
                  in
                  equal int 1 (Qmd.read q1 "release0_enable");
                  let qview = Mmio.view m ~off:0x4200 ~size:0x100 () in
                  (* address 0x200000010: low word at RELEASE0_ADDRESS_LOWER
                     (byte 96); its top nibble lands in the next word
                     without clobbering the enable bit set just before *)
                  equal int32 0x10l (Mmio.read32 qview 96);
                  equal int32 0x800002l (Mmio.read32 qview 100);
                  equal int32 9l (Mmio.read32 qview 104);
                  equal int32 0l (Mmio.read32 qview 108);
                  (* a second signal takes the second slot, a third falls
                     back to the stream and ends the launch *)
                  Compute_queue.signal cq ~value:10 (signal m);
                  equal int 4 (Q.length (Compute_queue.q cq));
                  equal int 1 (Qmd.read q1 "release1_enable");
                  Compute_queue.signal cq ~value:11 (signal m);
                  equal int 12 (Q.length (Compute_queue.q cq));
                  Compute_queue.exec cq prg ~kernargs:(kernarg2 m)
                    ~global_size:(1, 1, 1) ~local_size:(1, 1, 1);
                  equal int 16 (Q.length (Compute_queue.q cq));
                  equal int 0 (Qmd.read q1 "dependent_qmd0_enable")));
        ];
      group "submit"
        [
          test "stages the stream and rings the doorbell" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let qd = queue_desc m in
                  let cq = Compute_queue.create dev in
                  Compute_queue.wait cq ~value:5 (signal m);
                  Compute_queue.submit cq qd;
                  (* the stream lands at the staging page's start *)
                  equal int32 0x20050017l (Mmio.read32 m 0);
                  equal int32 5l (Mmio.read32 m 12);
                  (* ring entry: address 0x400000, length 6 dwords at bit
                     42, fetch flag at bit 41 *)
                  equal int64 0x1a0000400000L
                    (Mmio.read64 qd.Tolk_nv.Queue_desc.ring 0);
                  equal int32 1l (Mmio.read32 qd.Tolk_nv.Queue_desc.gpput 0);
                  equal int32 0x1abcdl (Mmio.read32 m 0x1090);
                  equal int 1 qd.Tolk_nv.Queue_desc.put_value));
          test "resubmission stages a fresh copy and advances the ring"
            (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let qd = queue_desc m in
                  let cq = Compute_queue.create dev in
                  Compute_queue.wait cq ~value:5 (signal m);
                  Compute_queue.submit cq qd;
                  Compute_queue.submit cq qd;
                  (* 24 bytes round up to the next 16-byte slot at 0x20 *)
                  equal int32 0x20050017l (Mmio.read32 m 0x20);
                  equal int64 0x1a0000400020L
                    (Mmio.read64 qd.Tolk_nv.Queue_desc.ring 8);
                  equal int32 2l (Mmio.read32 qd.Tolk_nv.Queue_desc.gpput 0);
                  equal int 2 qd.Tolk_nv.Queue_desc.put_value));
          test "the staging page and the ring wrap" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev ~cmdq_size:0x40 m in
                  let qd = queue_desc ~entries:2 m in
                  let cq1 = Compute_queue.create dev in
                  Compute_queue.wait cq1 ~value:5 (signal m);
                  let cq2 = Compute_queue.create dev in
                  Compute_queue.wait cq2 ~value:7 (signal m);
                  Compute_queue.submit cq1 qd;
                  Compute_queue.submit cq1 qd;
                  equal int32 0l (Mmio.read32 qd.Tolk_nv.Queue_desc.gpput 0);
                  (* a third stream no longer fits behind 0x38: the
                     allocator restarts at the page base and the entry
                     reuses ring slot 0 *)
                  Compute_queue.submit cq2 qd;
                  equal int32 7l (Mmio.read32 m 12);
                  equal int64 0x1a0000400000L
                    (Mmio.read64 qd.Tolk_nv.Queue_desc.ring 0);
                  equal int32 1l (Mmio.read32 qd.Tolk_nv.Queue_desc.gpput 0);
                  equal int 3 qd.Tolk_nv.Queue_desc.put_value));
        ];
      group "iface wire formats"
        [
          test "the allocation envelope wires the nested parameter pointer"
            (fun () ->
              let inner = Tables.create_blob 0x38 in
              let b =
                Nvk_iface.nvos21_params ~root:0xc1d00001 ~parent:0xbeef
                  ~cls:0x80 ~params:inner ()
              in
              let expected = Bytes.make 0x20 '\000' in
              Bytes.set_int32_le expected 0x00 0xc1d00001l;
              Bytes.set_int32_le expected 0x04 0xbeefl;
              Bytes.set_int32_le expected 0x0c 0x80l;
              Bytes.set_int64_le expected 0x10
                (Int64.of_nativeint (Tables.blob_addr inner));
              equal string (bytes_hex expected) (blob_hex b);
              (* without a parameter structure the pointer stays null *)
              let bare = Nvk_iface.nvos21_params ~root:1 ~parent:2 ~cls:3 () in
              equal int 0
                (Tables.get_field bare Defs.Nvos21_parameters.pallocparms));
          test "memory allocation parameters compose the attribute words"
            (fun () ->
              (* cached, contiguous device memory in 2 MiB pages *)
              let cls, p =
                Nvk_iface.memory_allocation_params ~root:0xc1d00001
                  ~size:0x200000 ~page_size:0x200000 ~uncached:false
                  ~contiguous:true ~read_only:false
              in
              equal int Defs.nv1_memory_user cls;
              let expected = Bytes.make 0x80 '\000' in
              Bytes.set_int32_le expected 0x00 0xc1d00001l;
              (* map-not-required, handle-provided, forced alignment,
                 ignored bank placement, persistent *)
              Bytes.set_int32_le expected 0x08 0x1c101l;
              (* contiguous at 27, huge pages at 23 *)
              Bytes.set_int32_le expected 0x18 0x11800000l;
              (* cacheable at 2, huge 2 MiB at 20, no zbc *)
              Bytes.set_int32_le expected 0x1c 0x100005l;
              Bytes.set_int32_le expected 0x20 6l;
              Bytes.set_int64_le expected 0x40 0x200000L;
              Bytes.set_int64_le expected 0x48 0x200000L;
              Bytes.set_int64_le expected 0x58 0x1fffffL;
              equal string (bytes_hex expected) (blob_hex p);
              (* uncached, read-only system pages *)
              let cls, p =
                Nvk_iface.memory_allocation_params ~root:0xc1d00001 ~size:0x1000
                  ~page_size:0x1000 ~uncached:true ~contiguous:false
                  ~read_only:true
              in
              equal int Defs.nv1_memory_system cls;
              let expected = Bytes.make 0x80 '\000' in
              Bytes.set_int32_le expected 0x00 0xc1d00001l;
              (* notifier type *)
              Bytes.set_int32_le expected 0x04 0xdl;
              (* no persistent-vidmem flag for system pages *)
              Bytes.set_int32_le expected 0x08 0xc101l;
              (* noncontiguous at 27, system location at 25 *)
              Bytes.set_int32_le expected 0x18 0x1a000000l;
              (* uncacheable at 2, no zbc, read-only protection at 22 *)
              Bytes.set_int32_le expected 0x1c 0x400009l;
              Bytes.set_int32_le expected 0x20 6l;
              Bytes.set_int64_le expected 0x40 0x1000L;
              Bytes.set_int64_le expected 0x48 0x1000L;
              Bytes.set_int64_le expected 0x58 0xfffL;
              equal string (bytes_hex expected) (blob_hex p));
          test "the mapping request carries one gpu attribute" (fun () ->
              let uuid = Bytes.init 16 (fun i -> Char.chr (0xa0 + i)) in
              let b =
                Nvk_iface.map_external_params ~rm_ctrl_fd:7 ~root:0xc1d00001
                  ~va:0x1234500000n ~size:0x10000 ~mem_handle:0x5abc1234
                  ~gpu_uuid:uuid
              in
              let expected = Bytes.make 0x2430 '\000' in
              Bytes.set_int64_le expected 0x00 0x1234500000L;
              Bytes.set_int64_le expected 0x08 0x10000L;
              Bytes.blit uuid 0 expected 0x18 16;
              (* mapping type of the single attribute entry *)
              Bytes.set_int32_le expected 0x28 1l;
              Bytes.set_int64_le expected 0x2418 1L;
              Bytes.set_int32_le expected 0x2420 7l;
              Bytes.set_int32_le expected 0x2424 0xc1d00001l;
              Bytes.set_int32_le expected 0x2428 0x5abc1234l;
              equal string (bytes_hex expected) (blob_hex b);
              raises_match is_invalid_arg (fun () ->
                  Nvk_iface.map_external_params ~rm_ctrl_fd:0 ~root:0 ~va:0n
                    ~size:0 ~mem_handle:0 ~gpu_uuid:(Bytes.create 8)));
          test "escape request codes embed the parameter sizes" (fun () ->
              equal int 0xc00446c9
                (Tables.escape_code ~nr:Defs.nv_esc_register_fd
                   ~size:Defs.Nv_ioctl_register_fd.sizeof);
              (* card enumeration passes an array of 64 entries *)
              equal int 0xd20046c8
                (Tables.escape_code ~nr:Defs.nv_esc_card_info
                   ~size:(64 * Defs.Nv_ioctl_card_info.sizeof));
              equal int 0xc020462a
                (Tables.escape_code ~nr:Defs.nv_esc_rm_control
                   ~size:Defs.Nvos54_parameters.sizeof);
              equal int 0xc0104629
                (Tables.escape_code ~nr:Defs.nv_esc_rm_free
                   ~size:Defs.Nvos00_parameters.sizeof);
              (* the dma-mapping parameters grew at 580, moving the code *)
              equal int 0xc0384657
                (Tables.escape_code ~nr:Defs.nv_esc_rm_map_memory_dma
                   ~size:0x38);
              equal int 0xc0404657
                (Tables.escape_code ~nr:Defs.nv_esc_rm_map_memory_dma
                   ~size:0x40));
        ];
      group "driver version"
        [
          test "the reported version selects the layout generation" (fun () ->
              equal int 570
                (Nvk_iface.driver_version_major (version_blob "570.144.03"));
              let sel s =
                Tables.defs_for_driver
                  ~major:(Nvk_iface.driver_version_major (version_blob s))
              in
              is_true ~msg:"570" (sel "570.144.03" == Tables.Versions.v570);
              is_true ~msg:"575" (sel "575.51.02" == Tables.Versions.v570);
              is_true ~msg:"580" (sel "580.65.06" == Tables.Versions.v580);
              is_true ~msg:"609" (sel "609.1" == Tables.Versions.v580);
              is_true ~msg:"615" (sel "615.29" == Tables.Versions.v610);
              raises_match
                (function Failure _ -> true | _ -> false)
                (fun () ->
                  Nvk_iface.driver_version_major (version_blob "unknown")));
        ];
      group "va allocator"
        [
          test "cpu-visible ranges come from the low window" (fun () ->
              let a =
                Nativeint.to_int
                  (Nvk_iface.alloc_gpu_vaddr ~force_low:true 0x4000)
              in
              let b =
                Nativeint.to_int
                  (Nvk_iface.alloc_gpu_vaddr ~force_low:true 0x4000)
              in
              is_true ~msg:"low base" (a >= 0x1000000000);
              is_true ~msg:"below the split" (b + 0x4000 <= 0x2000000000);
              is_true ~msg:"disjoint" (b >= a + 0x4000);
              equal int 0 (a land 0xfff));
          test "device-only ranges come from above the split" (fun () ->
              let a = Nativeint.to_int (Nvk_iface.alloc_gpu_vaddr 0x1000) in
              is_true ~msg:"high base" (a >= 0x2000000000);
              let b =
                Nativeint.to_int
                  (Nvk_iface.alloc_gpu_vaddr ~alignment:0x200000 0x1000)
              in
              equal int 0 (b land 0x1fffff);
              is_true ~msg:"disjoint" (b >= a + 0x1000));
        ];
      group "program load"
        [
          test "load parses a hand-built kernel object" (fun () ->
              with_fixture (fun m ->
                  let dev =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x240 m
                  in
                  let alloc, sizes = lib_alloc () in
                  let ensured = ref [] in
                  let prg =
                    Program.load dev ~alloc
                      ~ensure_local_memory:(fun n -> ensured := n :: !ensured)
                      ~name:"k" (cubin_fixture ())
                  in
                  (* image 0x1a29c rounds to 0x1b000 plus the 4 KiB guard *)
                  equal (list int) [ 0x1c000 ] !sizes;
                  equal (list int) [ 0x380 ] !ensured;
                  equal string "k" prg.Program.name;
                  equal int 32 prg.Program.regs_usage;
                  equal int 0x480 prg.Program.shmem_usage;
                  equal int 0x380 prg.Program.lcmem_usage;
                  equal int 0x160 prg.Program.params.Tolk_nv.cbuf0_size;
                  equal int 0xa00 prg.Program.kernargs_alloc_size;
                  equal int 2048 prg.Program.max_threads;
                  equal int 2 (List.length prg.Program.constbufs);
                  let a0, s0 = List.assoc 0 prg.Program.constbufs in
                  equal nativeint 0x110000n a0;
                  equal int 0x160 s0;
                  let a3, s3 = List.assoc 3 prg.Program.constbufs in
                  equal nativeint 0x118000n a3;
                  equal int 0x200 s3;
                  (* the driver-parameter words: the two windows and the
                     window-configuration constant at entries 6-11 *)
                  equal int 88 (Array.length prg.Program.cbuf_0);
                  equal (array int)
                    [| 0; 0x7294; 0; 0x7293; 0xfffdc0; 0 |]
                    (Array.sub prg.Program.cbuf_0 6 6);
                  (* section contents landed at their fixed addresses,
                     and the relocations patched the staged image *)
                  let v = Buffer.cpu_view prg.Program.lib_gpu in
                  let u32 off =
                    Int32.to_int (Mmio.read32 v off) land 0xffffffff
                  in
                  equal int 0xcccccccc (u32 0x2000);
                  equal int 0xaaaaaaaa (u32 0x12000);
                  equal int 0xbbbbbbbb (u32 0x1a000);
                  equal int64 0x110010L (Mmio.read64 v 0x2100);
                  equal int 0x110010 (u32 0x2204);
                  equal int 0 (u32 0x2304)));
          test "the descriptor template matches the qmd_init goldens"
            (fun () ->
              with_fixture (fun m ->
                  let ada =
                    load_fixture
                      (nv_dev ~sass_version:0x89 ~slm_per_thread:0x240 m)
                  in
                  equal (array int) qmd_expected_ada
                    (qmd_template_dwords ada);
                  let bw =
                    load_fixture
                      (nv_dev ~compute_class:Defs.blackwell_compute_b
                         ~sass_version:0xa4 ~slm_per_thread:0x240 m)
                  in
                  equal (array int) qmd_expected_blackwell
                    (qmd_template_dwords bw)));
          test "relocation targets carry the image's high address bits"
            (fun () ->
              with_fixture (fun m ->
                  let prg =
                    load_fixture ~va:0x8000fe000n
                      (nv_dev ~sass_version:0x89 ~slm_per_thread:0x240 m)
                  in
                  let v = Buffer.cpu_view prg.Program.lib_gpu in
                  equal int64 0x800110010L (Mmio.read64 v 0x2100);
                  equal int 0x110010
                    (Int32.to_int (Mmio.read32 v 0x2204) land 0xffffffff);
                  equal int 0x8
                    (Int32.to_int (Mmio.read32 v 0x2304) land 0xffffffff)));
          test "unsupported objects fail loudly" (fun () ->
              with_fixture (fun m ->
                  let dev =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x240 m
                  in
                  raises_match (failure_with "unknown NV reloc 55") (fun () ->
                      load_fixture ~lib:(cubin_fixture ~reloc0:0x37 ()) dev);
                  raises_match
                    (failure_with
                       "Attempting to relocate against an undefined symbol c0")
                    (fun () ->
                      load_fixture
                        ~lib:(cubin_fixture ~undefined_sym:true ())
                        dev);
                  raises_match (failure_with "unknown EIATTR format 7")
                    (fun () ->
                      load_fixture ~lib:(cubin_fixture ~bad_info:true ()) dev)));
          test "free releases the image" (fun () ->
              with_fixture (fun m ->
                  let prg =
                    load_fixture
                      (nv_dev ~sass_version:0x89 ~slm_per_thread:0x240 m)
                  in
                  let freed = ref [] in
                  Program.free
                    ~free:(fun b -> freed := Buffer.size b :: !freed)
                    prg;
                  equal (list int) [ 0x1c000 ] !freed));
        ];
      group "local memory"
        [
          test "growing sizes the store from the topology" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let qd = queue_desc m in
                  let tl = timeline m in
                  let allocs = ref [] and frees = ref [] in
                  let alloc size =
                    allocs := size :: !allocs;
                    Buffer.make ~va:0x60000000n ~size ~meta:() ()
                  in
                  let free buf = frees := Buffer.size buf :: !frees in
                  Tolk_nv.ensure_has_local_memory dev ~alloc ~free ~num_gpcs:2
                    ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2 ~max_warps_per_sm:48
                    ~tl ~queue:qd 0x100;
                  (* 0x100 * 32 rounds to 0x2000 per warp slot; times 48
                     warps and 2 SMs is 0xc0000 per TPC; times 6 TPCs is
                     0x480000 *)
                  equal int 0x100 dev.Tolk_nv.slm_per_thread;
                  equal (list int) [ 0x480000 ] !allocs;
                  equal (list int) [] !frees;
                  (match dev.Tolk_nv.shader_local_mem with
                  | Some b -> equal int 0x480000 (Buffer.size b)
                  | None -> fail "expected a backing store");
                  equal int 2 tl.Timeline.timeline_value;
                  equal int 1 qd.Tolk_nv.Queue_desc.put_value;
                  let expected =
                    let cq = Compute_queue.create dev in
                    Compute_queue.wait cq ~value:0 tl.Timeline.timeline;
                    Compute_queue.setup cq ~local_mem:0x60000000n
                      ~local_mem_tpc_bytes:0xc0000 ();
                    Compute_queue.signal cq ~value:1 tl.Timeline.timeline;
                    Q.dwords (Compute_queue.q cq)
                  in
                  equal (array int) expected
                    (staged_dwords m ~off:0 (Array.length expected));
                  (* a covered request changes nothing *)
                  Tolk_nv.ensure_has_local_memory dev ~alloc ~free ~num_gpcs:2
                    ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2 ~max_warps_per_sm:48
                    ~tl ~queue:qd 0x80;
                  equal int 1 (List.length !allocs);
                  equal int 2 tl.Timeline.timeline_value));
          test "out of memory reallocates the old size and restores the state"
            (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let qd = queue_desc m in
                  let tl = timeline m in
                  let allocs = ref [] and frees = ref [] in
                  let fail_next = ref false in
                  let alloc size =
                    allocs := size :: !allocs;
                    if !fail_next then begin
                      fail_next := false;
                      raise (Nv_iface.Out_of_memory "scripted")
                    end;
                    Buffer.make ~va:0x60000000n ~size ~meta:() ()
                  in
                  let free buf = frees := Buffer.size buf :: !frees in
                  let ensure required =
                    Tolk_nv.ensure_has_local_memory dev ~alloc ~free
                      ~num_gpcs:2 ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2
                      ~max_warps_per_sm:48 ~tl ~queue:qd required
                  in
                  ensure 0x100;
                  fail_next := true;
                  ensure 0x200;
                  (* the grow to 0x900000 failed: the old 0x480000 store
                     is reallocated and the sizing state restored *)
                  equal (list int)
                    [ 0x480000; 0x900000; 0x480000 ]
                    (List.rev !allocs);
                  equal (list int) [ 0x480000 ] !frees;
                  equal int 0x100 dev.Tolk_nv.slm_per_thread;
                  (match dev.Tolk_nv.shader_local_mem with
                  | Some b -> equal int 0x480000 (Buffer.size b)
                  | None -> fail "expected a backing store");
                  (* the engine is still repointed, with the attempted
                     per-TPC size *)
                  equal int 3 tl.Timeline.timeline_value;
                  equal int 2 qd.Tolk_nv.Queue_desc.put_value;
                  let expected =
                    let cq = Compute_queue.create dev in
                    Compute_queue.wait cq ~value:1 tl.Timeline.timeline;
                    Compute_queue.setup cq ~local_mem:0x60000000n
                      ~local_mem_tpc_bytes:0x180000 ();
                    Compute_queue.signal cq ~value:2 tl.Timeline.timeline;
                    Q.dwords (Compute_queue.q cq)
                  in
                  let first_len = 21 * 4 in
                  equal (array int) expected
                    (staged_dwords m
                       ~off:((first_len + 15) / 16 * 16)
                       (Array.length expected))));
          test "out of memory without a fallback propagates" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let qd = queue_desc m in
                  let tl = timeline m in
                  raises_match
                    (function Nv_iface.Out_of_memory _ -> true | _ -> false)
                    (fun () ->
                      Tolk_nv.ensure_has_local_memory dev
                        ~alloc:(fun _ ->
                          raise (Nv_iface.Out_of_memory "scripted"))
                        ~free:(fun _ -> fail "nothing to free")
                        ~num_gpcs:2 ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2
                        ~max_warps_per_sm:48 ~tl ~queue:qd 0x10);
                  equal int 0x20 dev.Tolk_nv.slm_per_thread;
                  is_true (Option.is_none dev.Tolk_nv.shader_local_mem);
                  equal int 1 tl.Timeline.timeline_value;
                  equal int 0 qd.Tolk_nv.Queue_desc.put_value));
        ];
      group "program call"
        [
          test "call stages the arguments, descriptor and stream" (fun () ->
              with_fixture (fun m ->
                  let dev =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x380 m
                  in
                  let prg = load_fixture dev in
                  let qd = queue_desc m in
                  let tl_sig =
                    Signal.make ~is_timeline:true
                      (Buffer.make ~va:0x200000010n ~size:16
                         ~view:(Mmio.view m ~off:0x3000 ~size:16 ())
                         ~meta:() ())
                  in
                  let kernargs =
                    Kernargs.create
                      (Buffer.make ~va:0x30000000n ~size:0x1000
                         ~view:(Mmio.view m ~off:0x4000 ~size:0x1000 ())
                         ~meta:() ())
                  in
                  let r =
                    Program.call prg ~kernargs ~queue:qd ~timeline:tl_sig
                      ~timeline_value:1
                      ~bufs:[| 0x111100000n; 0x222200000n |]
                      ~vals:[| 7 |] ~global_size:(4, 3, 2)
                      ~local_size:(8, 4, 1) ()
                  in
                  is_none r;
                  (* wait for the previous work, make writes visible,
                     launch; the timeline release rides the descriptor *)
                  equal (array int)
                    [|
                      0x20050017; 0x10; 2; 0; 0; 0x01000003;
                      0x200125a6; 0x1011;
                      0x200120ad; 0x300002; 0x200120b0; 9;
                    |]
                    (staged_dwords m ~off:0 12);
                  (* the argument slot: the driver-parameter words, then
                     the buffer addresses, then the value *)
                  equal (array int)
                    [| 0; 0x7294; 0; 0x7293; 0xfffdc0; 0 |]
                    (staged_dwords m ~off:(0x4000 + 24) 6);
                  equal int64 0x111100000L (Mmio.read64 m (0x4000 + 0x160));
                  equal int64 0x222200000L (Mmio.read64 m (0x4000 + 0x168));
                  equal int32 7l (Mmio.read32 m (0x4000 + 0x170));
                  let q =
                    exec_qmd ~compute_class:dev.Tolk_nv.compute_class m
                      ~kernarg_off:0x4000
                  in
                  equal int 4 (Qmd.read q "cta_raster_width");
                  equal int 3 (Qmd.read q "cta_raster_height");
                  equal int 2 (Qmd.read q "cta_raster_depth");
                  equal int 8 (Qmd.read q "cta_thread_dimension0");
                  equal int 4 (Qmd.read q "cta_thread_dimension1");
                  equal int 1 (Qmd.read q "cta_thread_dimension2");
                  equal int 0x30000000
                    (Qmd.read q "constant_buffer_addr_lower_0");
                  equal int 1 (Qmd.read q "release0_enable");
                  equal int 1 (Qmd.read q "release0_payload_lower");
                  equal int 1 qd.Tolk_nv.Queue_desc.put_value;
                  equal int32 0x1abcdl (Mmio.read32 m 0x1090)));
          test "call with wait brackets the launch and reports the time"
            (fun () ->
              with_fixture (fun m ->
                  let dev =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x380 m
                  in
                  let prg = load_fixture dev in
                  let qd = queue_desc m in
                  let slot_at off va =
                    Buffer.make ~va ~size:16
                      ~view:(Mmio.view m ~off ~size:16 ())
                      ~meta:() ()
                  in
                  let tl_sig =
                    Signal.make ~is_timeline:true (slot_at 0x3000 0x200000010n)
                  in
                  let st = Signal.make (slot_at 0x3010 0x200000020n) in
                  let en = Signal.make (slot_at 0x3020 0x200000030n) in
                  let kernargs =
                    Kernargs.create
                      (Buffer.make ~va:0x30000000n ~size:0x1000
                         ~view:(Mmio.view m ~off:0x4000 ~size:0x1000 ())
                         ~meta:() ())
                  in
                  (* completion and nanosecond clock captures the device
                     would write: 25 ms elapsed *)
                  Mmio.write64 m 0x3000 1L;
                  Mmio.write64 m 0x3018 10_000_000L;
                  Mmio.write64 m 0x3028 35_000_000L;
                  let r =
                    Program.call prg ~kernargs ~queue:qd ~timeline:tl_sig
                      ~timeline_value:1 ~wait:(st, en) ~bufs:[||] ~vals:[||]
                      ~global_size:(1, 1, 1) ~local_size:(1, 1, 1) ()
                  in
                  (match r with
                  | Some dt -> equal (float 1e-9) 0.025 dt
                  | None -> fail "expected an execution time");
                  (* the start capture is in the stream; the end capture
                     and the timeline release ride the descriptor *)
                  equal (array int)
                    [|
                      0x20050017; 0x10; 2; 0; 0; 0x01000003;
                      0x200125a6; 0x1011;
                      0x20050017; 0x20; 2; 0; 0; 0x03100001; 0x20010008; 0;
                      0x200120ad; 0x300002; 0x200120b0; 9;
                    |]
                    (staged_dwords m ~off:0 20);
                  let q =
                    exec_qmd ~compute_class:dev.Tolk_nv.compute_class m
                      ~kernarg_off:0x4000
                  in
                  equal int 1 (Qmd.read q "release0_enable");
                  equal int 1 (Qmd.read q "release1_enable")));
          test "launch limits are enforced before staging" (fun () ->
              with_fixture (fun m ->
                  let dev =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x380 m
                  in
                  let prg = load_fixture dev in
                  let qd = queue_desc m in
                  let tl_sig = signal m in
                  let kernargs =
                    Kernargs.create
                      (Buffer.make ~va:0x30000000n ~size:0x1000
                         ~view:(Mmio.view m ~off:0x4000 ~size:0x1000 ())
                         ~meta:() ())
                  in
                  let call ?(prg = prg) ~global_size ~local_size () =
                    Program.call prg ~kernargs ~queue:qd ~timeline:tl_sig
                      ~timeline_value:1 ~bufs:[||] ~vals:[||] ~global_size
                      ~local_size ()
                  in
                  raises_match (failure_with "Invalid global/local dims")
                    (fun () ->
                      call ~global_size:(1, 1, 1) ~local_size:(1, 1, 65) ());
                  raises_match (failure_with "Invalid global/local dims")
                    (fun () ->
                      call ~global_size:(1, 0x10000, 1) ~local_size:(1, 1, 1)
                        ());
                  raises_match (failure_with "Too many resources") (fun () ->
                      call ~global_size:(1, 1, 1) ~local_size:(16, 16, 8) ());
                  (* a register-hungry kernel caps the block size *)
                  let hungry =
                    load_fixture ~lib:(cubin_fixture ~regcount:256 ()) dev
                  in
                  equal int 256 hungry.Program.max_threads;
                  raises_match (failure_with "Too many resources") (fun () ->
                      call ~prg:hungry ~global_size:(1, 1, 1)
                        ~local_size:(32, 32, 1) ());
                  (* a device not sized for the kernel's local memory *)
                  let small =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x240 m
                  in
                  let prg_small = load_fixture small in
                  raises_match (failure_with "Too many resources") (fun () ->
                      Program.call prg_small ~kernargs ~queue:qd
                        ~timeline:tl_sig ~timeline_value:1 ~bufs:[||]
                        ~vals:[||] ~global_size:(1, 1, 1)
                        ~local_size:(1, 1, 1) ());
                  (* nothing was staged or submitted *)
                  equal nativeint 0x30000000n
                    (Buffer.va (Kernargs.alloc kernargs 8));
                  equal int 0 qd.Tolk_nv.Queue_desc.put_value));
          test "blackwell programs use the wide driver-parameter layout"
            (fun () ->
              with_fixture (fun m ->
                  let prg =
                    load_fixture
                      (nv_dev ~compute_class:Defs.blackwell_compute_b
                         ~sass_version:0xa4 ~slm_per_thread:0x240 m)
                  in
                  equal int 224 (Array.length prg.Program.cbuf_0);
                  equal (array int)
                    [| 0; 0x7294; 0; 0x7293 |]
                    (Array.sub prg.Program.cbuf_0 188 4);
                  equal int 0xfffdc0 prg.Program.cbuf_0.(223)));
        ];
      group "cubin fixture"
        [
          test "the recorded nvrtc kernel parses to its recorded fields"
            (fun () ->
              let dir = "../fixtures/nv" in
              let cubin = Filename.concat dir "simple_add_sm89.cubin" in
              let fields_file = Filename.concat dir "simple_add_sm89.fields" in
              if not (Sys.file_exists cubin && Sys.file_exists fields_file)
              then
                skip
                  ~reason:
                    "no cubin fixture (generate it with \
                     test/fixtures/nv/generate_fixture.py on a box with the \
                     CUDA toolkit)"
                  ();
              let fields =
                List.filter_map
                  (fun line ->
                    match
                      String.split_on_char ' ' (String.trim line)
                    with
                    | [ k; v ] -> Some (k, v)
                    | _ -> None)
                  (String.split_on_char '\n'
                     (In_channel.with_open_bin fields_file
                        In_channel.input_all))
              in
              let fint k = int_of_string (List.assoc k fields) in
              let lib =
                Bytes.of_string
                  (In_channel.with_open_bin cubin In_channel.input_all)
              in
              with_fixture (fun m ->
                  let dev =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x2000 m
                  in
                  let alloc size =
                    Buffer.make ~va:0x100000n ~size ~view:(anon_mmio size)
                      ~meta:() ()
                  in
                  let prg =
                    Program.load dev ~alloc
                      ~ensure_local_memory:(fun _ -> ())
                      ~name:(List.assoc "name" fields)
                      lib
                  in
                  equal int (fint "regs_usage") prg.Program.regs_usage;
                  equal int (fint "shmem_usage") prg.Program.shmem_usage;
                  equal int (fint "lcmem_usage") prg.Program.lcmem_usage;
                  equal int (fint "constbuf0_size")
                    prg.Program.params.Tolk_nv.cbuf0_size;
                  equal int
                    (fint "kernargs_alloc_size")
                    prg.Program.kernargs_alloc_size));
        ];
      group "device"
        [
          test "the interface opens and enumerates" (fun () ->
              let i = nvk_iface () in
              is_true ~msg:"count" (i.Nv_iface.count >= 1);
              is_true ~msg:"root" (i.Nv_iface.root <> 0);
              is_true ~msg:"instance" (i.Nv_iface.gpu_instance >= 0);
              is_true ~msg:"kernel driver" (not (Nv_iface.is_nvd i)));
        ];
    ]
