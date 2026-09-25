(* Inspect the shared NV encoder at fixed link addresses; never open a GPU. *)
open Tolk_nv
open Tolk_uop
module U = Uop
module D = Dtype
module F = Queue_fixture
module Hcq = Tolk_hcq.Hcq
module Defs = Nv_tables.Defs
let out_dir = Sys.argv.(1)
let u32 n = U.const (Const.int D.uint32 n)
let u64 = F.uint
let bytes words =
  let b = Bytes.create (4 * List.length words) in
  List.iteri (fun i w -> Bytes.set_int32_le b (4*i) (Int32.of_int w)) words;
  b
let nvm sub method_ words = ((2 lsl 28) lor (List.length words lsl 16) lor (sub lsl 13) lor (method_ lsr 2)) :: words
let data64 n = [n lsr 32; n land 0xffffffff]
let raw words = U.ins ~mnemonic:"nv" ~operands:[U.binary (Bytes.to_string (bytes words))] ()

let run chip compute_class dma_class sass_version =
  let name = "NV:" ^ chip in
  let page = Hcq.File_io.mmap ~addr:0n ~size:4096
      ~prot:(Hcq.File_io.prot_read lor Hcq.File_io.prot_write)
      ~flags:(Hcq.File_io.map_private lor Hcq.File_io.map_anonymous) ~fd:(-1) ~offset:0L in
  Fun.protect ~finally:(fun () -> Hcq.File_io.munmap page ~size:4096) (fun () ->
      let dev = device ~compute_class ~dma_class ~gpfifo_class:Defs.ampere_channel_gpfifo_a
          ~sass_version ~slm_per_thread:0x240 ~shared_mem_window:0x729400000000n
          ~local_mem_window:0x729300000000n ~gpu_mmio:(Hcq.Mmio.make ~addr:page ~size:4096) () in
      let ptr tag dtype size address = F.pointer ~device:name ~tag ~dtype ~size ~address in
      let signal = ptr "signal" D.uint64 2 0x400000 in
      let copy_size = (2 * 0x80000000) + 0x400 in
      let src = ptr "source" D.uint8 copy_size 0x10000000 in
      let dst = ptr "destination" D.uint8 copy_size 0x20000000 in
      let binary = F.read "../../fixtures/nv/simple_add_sm89.cubin" in
      let call = F.program ~device:name ~dtype:D.float32 ~binary in
      let encode compute nodes =
        let kind = if compute then "compute_0" else "copy_0" in
        let submit = U.custom_function ~name:("submit_nv_" ^ kind) ~srcs:[U.linear nodes; U.group []] in
        Option.get (Encoded_queue.encode dev ~name ~compute_entries:256 ~copy_entries:256
          ~compute_token:1 ~copy_token:2 submit) in
      let build label compute nodes =
        let encoded = encode compute nodes in
        (* Tolk appends a six-dword progress release for safe command-storage
           reuse. That separate ownership protocol has runtime tests; these
           operation fixtures compare the target builder's pre-submit payload. *)
        let tag = if compute then "cmdbuf_compute" else "cmdbuf_copy" in
        let bits (_, lo) value = value lsl lo in
        let tail = if compute then
            [Some (List.hd (nvm 0 Defs.nvc56f_sem_addr_lo [0;0;0;0;0]));
              None; None; None; Some 0;
              Some (bits Defs.nvc56f_sem_execute_operation Defs.nvc56f_sem_execute_operation_release
                lor bits Defs.nvc56f_sem_execute_release_wfi Defs.nvc56f_sem_execute_release_wfi_en)]
          else [Some (List.hd (nvm 4 Defs.nvc6b5_set_semaphore_a [0;0;0])); None; None; None;
            Some (List.hd (nvm 4 Defs.nvc6b5_launch_dma [0]));
            Some (bits Defs.nvc6b5_launch_dma_flush_enable 1 lor bits Defs.nvc6b5_launch_dma_semaphore_type 1)] in
        F.dump out_dir label chip (F.blob ~tail tag encoded);
        encoded in
      let operation label compute mnemonic operands =
        ignore (build label compute [U.ins ~mnemonic ~operands ()]) in
      operation "setup" true "nv" [U.binary (Bytes.to_string (bytes (
        nvm 1 Defs.nvc6c0_set_object [compute_class] @
        nvm 1 Defs.nvc6c0_set_shader_local_memory_window_a (data64 0x729300000000) @
        nvm 1 Defs.nvc6c0_set_shader_shared_memory_window_a (data64 0x729400000000))))];
      ignore (build "setup_local_mem" true [raw (
        nvm 1 Defs.nvc6c0_set_shader_local_memory_a (data64 0x800000) @
        nvm 1 Defs.nvc6c0_set_shader_local_memory_non_throttled_a (data64 0x30000 @ [0xff]))]);
      operation "memory_barrier" true "barrier" [];
      operation "wait" true "wait" [signal; u64 0x42];
      operation "timestamp" true "timestamp" [signal];
      operation "signal_no_qmd" true "store" [signal; u64 0x100000042];
      let qmd_size = Qmd.sizeof ~compute_class in
      let descriptor label i count encoded =
        let arena = F.blob "qmd" encoded in
        if Bytes.length arena mod count <> 0 then failwith "invalid descriptor stride";
        F.dump out_dir label chip (Bytes.sub arena (i * (Bytes.length arena / count)) qmd_size) in
      descriptor "exec_qmd" 0 1 (build "exec" true [call]);
      let chain = build "exec_chained" true [call; call] in
      descriptor "exec_chained_qmd0" 0 2 chain;
      descriptor "exec_chained_qmd1" 1 2 chain;
      descriptor "signal_after_exec_qmd" 0 1 (build "signal_after_exec" true
        [call; U.ins ~mnemonic:"store" ~operands:[signal; u64 0x100000042] ()]);
      let data = Program.image ~name:"simple_add" binary in
      dev.slm_per_thread <- max dev.slm_per_thread ((data.lcmem_usage + 31) / 32 * 32);
      let qmd, _ = Program.template dev data in
      F.dump out_dir "qmd_init" chip (Qmd.to_bytes qmd);
      ignore (build "dma_setup" false [raw (nvm 4 Defs.nvc6c0_set_object [dma_class])]);
      List.iter (fun (label, size) ->
          let shrink buf = U.shrink ~src:buf ~offset:(F.int 0) ~size:(F.int size) in
          ignore (build label false [U.store_call ~dst:(shrink dst) ~src:(shrink src)]))
        ["dma_copy_small",0x1000; "dma_copy_large",copy_size];
      operation "dma_signal" false "store" [signal; u32 0x42];
      operation "dma_wait" false "wait" [signal; u64 0x42];
      operation "dma_timestamp" false "timestamp" [signal])

let () =
  run "ada" Defs.ada_compute_a Defs.ampere_dma_copy_b 0x89;
  run "blackwell" Defs.blackwell_compute_b Defs.blackwell_dma_copy_b 0xa4
