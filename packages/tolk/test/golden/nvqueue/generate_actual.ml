(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Dumps the dword stream of every queue-builder method and the QMD images
   for the two golden chip configs. The device wiring, fake constants, and
   stream table mirror generate_expected.py exactly; see README for the
   contract. *)

open Tolk_nv
module Hcq = Tolk_hcq.Hcq
module Defs = Nv_tables.Defs

let out_dir = Sys.argv.(1)

(* Fake constants; every value mirrors the CONFIG block of
   generate_expected.py. *)
let prog_addr = 0x100000
let prog_sz = 0x1800
let regs_usage = 32
let shmem_usage = 0x480
let slm_per_thread = 0x240
let constbufs = [ (0, (0x110000n, 0x160)); (3, (0x118000n, 0x200)) ]
let kernarg_va = 0x300000n
let kernarg2_va = 0x301000n
let kernarg_size = 0x1000
let signal_va = 0x400000n
let write_va = 0x600000n
let poll_va = 0x700000n
let local_mem_va = 0x800000n
let local_mem_tpc_bytes = 0x30000
let copy_src_va = 0x10000000n
let copy_dst_va = 0x20000000n
let shared_mem_window = 0x729400000000n
let local_mem_window = 0x729300000000n
let signal_value = 0x100000042
let dma_signal_value = 0x42
let wait_value = 0x42
let write32_value = 0x12345678L
let write64_value = 0x1122334455667788L
let poll_mask = 0x1
let global_size = (4, 3, 2)
let local_size = (8, 4, 1)
let copy_small = 0x1000
let copy_large = (2 * 0x80000000) + 0x400

(* exec stages the QMD image into the kernarg page at this offset
   (constbuf 0 size 0x160 rounded up to 256). *)
let kernarg_qmd_off = 0x200

let anon_mmio size =
  let addr =
    Hcq.File_io.mmap ~addr:0n ~size
      ~prot:(Hcq.File_io.prot_read lor Hcq.File_io.prot_write)
      ~flags:(Hcq.File_io.map_private lor Hcq.File_io.map_anonymous)
      ~fd:(-1) ~offset:0L
  in
  Hcq.Mmio.make ~addr ~size

(* Kernarg pages need real backing: exec writes the QMD image through the
   CPU view. The device address stays the fake constant. *)
let backed_buf va size =
  Hcq.Buffer.make ~va ~size ~view:(anon_mmio size) ~meta:() ()

let make_dev ~compute_class ~dma_class ~gpfifo_class ~sass_version =
  device ~compute_class ~dma_class ~gpfifo_class ~sass_version ~slm_per_thread
    ~shared_mem_window ~local_mem_window
    ~cmdq_page:(backed_buf 0x900000n 0x10000)
    ~gpu_mmio:(anon_mmio 0x1000) ()

let dump_dwords name chip dwords =
  let oc = open_out (Filename.concat out_dir (name ^ "_" ^ chip ^ ".actual")) in
  Array.iter (fun v -> Printf.fprintf oc "%08x\n" v) dwords;
  close_out oc

let mmio_dwords view ~off ~size =
  Array.init (size / 4) (fun i ->
      Int32.to_int (Hcq.Mmio.read32 view (off + (4 * i))) land 0xffffffff)

let bytes_dwords b =
  Array.init (Bytes.length b / 4) (fun i ->
      Int32.to_int (Bytes.get_int32_le b (4 * i)) land 0xffffffff)

(* The fixed program descriptor of the goldens. This mirrors the
   REPLICATED BLOCK of generate_expected.py (the reference program-load
   descriptor construction); re-sync both copies together. *)
let make_prog dev =
  let qmd =
    Qmd.create
      ~view:(anon_mmio (Qmd.sizeof ~compute_class:dev.compute_class))
      ~compute_class:dev.compute_class
  in
  let version_fields =
    if Qmd.version qmd = 5 then
      let pa4 = prog_addr lsr 4 in
      [
        ("qmd_major_version", 5);
        ("qmd_type", Defs.nvcec0_qmdv05_00_qmd_type_grid_cta);
        ("program_address_upper_shifted4", pa4 lsr 32);
        ("program_address_lower_shifted4", pa4 land 0xffffffff);
        ("register_count", regs_usage);
        ("shared_memory_size_shifted7", shmem_usage lsr 7);
        ("shader_local_memory_high_size_shifted4", dev.slm_per_thread lsr 4);
      ]
    else
      [
        ("qmd_major_version", 3);
        ("sm_global_caching_enable", 1);
        ("program_address_upper", prog_addr lsr 32);
        ("program_address_lower", prog_addr land 0xffffffff);
        ("shared_memory_size", shmem_usage);
        ("register_count_v", regs_usage);
        ("shader_local_memory_high_size", dev.slm_per_thread);
      ]
  in
  let smem_cfg =
    (List.find (fun c -> c * 1024 >= shmem_usage) [ 32; 64; 100 ] * 1024 / 4096)
    + 1
  in
  Qmd.write qmd
    (version_fields
    @ [
        ("qmd_group_id", 0x3f);
        ("invalidate_texture_header_cache", 1);
        ("invalidate_texture_sampler_cache", 1);
        ("invalidate_texture_data_cache", 1);
        ("invalidate_shader_data_cache", 1);
        ("api_visible_call_limit", 1);
        ("sampler_index", 1);
        ("barrier_count", 1);
        ("cwd_membar_type", Defs.nvc6c0_qmdv03_00_cwd_membar_type_l1_sysmembar);
        ("constant_buffer_invalidate_0", 1);
        ("min_sm_config_shared_mem_size", smem_cfg);
        ("target_sm_config_shared_mem_size", smem_cfg);
        ("max_sm_config_shared_mem_size", 0x1a);
        ("program_prefetch_size", min (prog_sz lsr 8) 0x1ff);
        ("sass_version", dev.sass_version);
        ("program_prefetch_addr_upper_shifted", prog_addr lsr 40);
        ("program_prefetch_addr_lower_shifted", prog_addr lsr 8);
      ]);
  List.iter
    (fun (i, (addr, sz)) ->
      Qmd.set_constant_buf_addr qmd i addr;
      Qmd.write qmd
        [
          (Printf.sprintf "constant_buffer_size_shifted4_%d" i, sz);
          (Printf.sprintf "constant_buffer_valid_%d" i, 1);
        ])
    constbufs;
  { dev; qmd; cbuf0_size = 0x160 }

let compute_streams chip dev =
  let prog = make_prog dev in
  let qmd_size = Qmd.sizeof ~compute_class:dev.compute_class in
  let sg = Hcq.Signal.make (backed_buf signal_va 16) in
  let wbuf = Hcq.Buffer.make ~va:write_va ~size:0x1000 ~meta:() () in
  let pbuf = Hcq.Buffer.make ~va:poll_va ~size:0x1000 ~meta:() () in
  let build name f =
    let q = Compute_queue.create dev in
    f q;
    dump_dwords name chip (Hcq.Q.dwords (Compute_queue.q q))
  in
  let dump_kernarg_qmd name buf =
    dump_dwords name chip
      (mmio_dwords (Hcq.Buffer.cpu_view buf) ~off:kernarg_qmd_off
         ~size:qmd_size)
  in
  build "setup" (fun q ->
      Compute_queue.setup q ~compute_class:dev.compute_class
        ~local_mem_window:dev.local_mem_window
        ~shared_mem_window:dev.shared_mem_window ());
  build "setup_local_mem" (fun q ->
      Compute_queue.setup q ~local_mem:local_mem_va ~local_mem_tpc_bytes ());
  build "memory_barrier" (fun q -> Compute_queue.memory_barrier q);
  build "wait" (fun q -> Compute_queue.wait q ~value:wait_value sg);
  build "timestamp" (fun q -> Compute_queue.timestamp q sg);
  build "signal_no_qmd" (fun q -> Compute_queue.signal q ~value:signal_value sg);
  build "write32" (fun q -> Compute_queue.write q wbuf write32_value);
  build "write64" (fun q -> Compute_queue.write q ~b64:true wbuf write64_value);
  build "poll_bit_set" (fun q ->
      Compute_queue.poll_bit q pbuf ~value:poll_mask ~mask:poll_mask);
  build "poll_bit_clear" (fun q ->
      Compute_queue.poll_bit q pbuf ~value:0 ~mask:poll_mask);
  let kernargs = backed_buf kernarg_va kernarg_size in
  build "exec" (fun q ->
      Compute_queue.exec q prog ~kernargs ~global_size ~local_size);
  dump_kernarg_qmd "exec_qmd" kernargs;
  (* The second exec emits no packets: it links itself into the first
     QMD's dependent-launch fields. *)
  let kernargs1 = backed_buf kernarg_va kernarg_size in
  let kernargs2 = backed_buf kernarg2_va kernarg_size in
  build "exec_chained" (fun q ->
      Compute_queue.exec q prog ~kernargs:kernargs1 ~global_size ~local_size;
      Compute_queue.exec q prog ~kernargs:kernargs2 ~global_size ~local_size);
  dump_kernarg_qmd "exec_chained_qmd0" kernargs1;
  dump_kernarg_qmd "exec_chained_qmd1" kernargs2;
  (* The signal emits no packets: it patches the active QMD's release
     semaphore. *)
  let kernargs3 = backed_buf kernarg_va kernarg_size in
  build "signal_after_exec" (fun q ->
      Compute_queue.exec q prog ~kernargs:kernargs3 ~global_size ~local_size;
      Compute_queue.signal q ~value:signal_value sg);
  dump_kernarg_qmd "signal_after_exec_qmd" kernargs3;
  dump_dwords "qmd_init" chip (bytes_dwords (Qmd.to_bytes prog.qmd))

let dma_streams chip dev =
  let sg = Hcq.Signal.make (backed_buf signal_va 16) in
  let src = Hcq.Buffer.make ~va:copy_src_va ~size:0 ~meta:() () in
  let dst = Hcq.Buffer.make ~va:copy_dst_va ~size:0 ~meta:() () in
  let build name f =
    let q = Copy_queue.create dev in
    f q;
    dump_dwords name chip (Hcq.Q.dwords (Copy_queue.q q))
  in
  build "dma_setup" (fun q -> Copy_queue.setup q ~copy_class:dev.dma_class ());
  build "dma_copy_small" (fun q -> Copy_queue.copy q ~dest:dst ~src copy_small);
  build "dma_copy_large" (fun q -> Copy_queue.copy q ~dest:dst ~src copy_large);
  build "dma_signal" (fun q -> Copy_queue.signal q ~value:dma_signal_value sg);
  build "dma_wait" (fun q -> Copy_queue.wait q ~value:wait_value sg);
  build "dma_timestamp" (fun q -> Copy_queue.timestamp q sg)

let () =
  let chips =
    [
      ( "ada",
        make_dev ~compute_class:Defs.ada_compute_a
          ~dma_class:Defs.ampere_dma_copy_b
          ~gpfifo_class:Defs.ampere_channel_gpfifo_a ~sass_version:0x89 );
      ( "blackwell",
        make_dev ~compute_class:Defs.blackwell_compute_b
          ~dma_class:Defs.blackwell_dma_copy_b
          ~gpfifo_class:Defs.blackwell_channel_gpfifo_a ~sass_version:0xa4 );
    ]
  in
  List.iter
    (fun (chip, dev) ->
      compute_streams chip dev;
      dma_streams chip dev)
    chips
