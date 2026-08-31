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
module Tables = Tolk_nv.Nv_tables
module Defs = Tolk_nv.Nv_tables.Defs
module Qmd = Tolk_nv.Qmd
module Compute_queue = Tolk_nv.Compute_queue
module Copy_queue = Tolk_nv.Copy_queue
module Nv_iface = Tolk_nv.Nv_iface
module Nvk_iface = Tolk_nv.Nvk_iface

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

let nv_dev ?(compute_class = Defs.ada_compute_a) ?(cmdq_size = 0x1000) m =
  Tolk_nv.device ~compute_class ~dma_class:Defs.ampere_dma_copy_b
    ~gpfifo_class:Defs.ampere_channel_gpfifo_a ~sass_version:0x59
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
