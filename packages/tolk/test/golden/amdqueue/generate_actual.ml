(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Dumps the dword stream of every queue-builder method for the two golden
   chip configs. The device wiring, fake constants, and stream table mirror
   generate_expected.py exactly; see README for the contract. *)

open Tolk_amd

let out_dir = Sys.argv.(1)

(* Fake constants; every value mirrors the CONFIG block of
   generate_expected.py. *)
let prog_addr = 0x100000n
let scratch_va = 0x200000n
let scratch_size = 0x80000
let kernarg_va = 0x300000n
let signal_va = 0x400000n
let mailbox_ptr = 0x500000n
let event_id = 0x2a
let write_va = 0x600000n
let poll_va = 0x700000n
let copy_src_va = 0x10000000n
let copy_dst_va = 0x20000000n
let tmpring_size = 0x00200008
let rsrc1 = 0x1111
let rsrc2 = 0x2222
let rsrc3 = 0x3333
let signal_value = 0x42
let wait_value = 0x42
let write32_value = 0x12345678L
let write64_value = 0x1122334455667788L
let poll_mask = 0x1
let global_size = (4, 3, 2)
let local_size = (8, 4, 1)
let copy_small = 0x1000
let copy_large_extra = 0x400

let make_dev ~target ~xccs ~gc_version ~nbio_version ~sdma_version =
  device ~target ~xccs ~gc_version ~nbio_version ~sdma_version ~tmpring_size
    ~scratch:(Hcq.Buffer.make ~va:scratch_va ~size:scratch_size ~meta:() ())
    ~is_am:false ~queue_event_mailbox_ptr:mailbox_ptr
    ~queue_event:{ event_id } ()

(* Signals need a CPU-mapped slot; its device address stays the fake
   constant. *)
let signal_slot =
  let size = 4096 in
  let addr =
    Hcq.File_io.mmap ~addr:0n ~size
      ~prot:(Hcq.File_io.prot_read lor Hcq.File_io.prot_write)
      ~flags:(Hcq.File_io.map_private lor Hcq.File_io.map_anonymous)
      ~fd:(-1) ~offset:0L
  in
  Hcq.Buffer.make ~va:signal_va ~size:16
    ~view:(Hcq.Mmio.view (Hcq.Mmio.make ~addr ~size) ~off:0 ~size:16 ())
    ~meta:() ()

let dump name chip q =
  let oc = open_out (Filename.concat out_dir (name ^ "_" ^ chip ^ ".actual")) in
  Array.iter (fun v -> Printf.fprintf oc "%08x\n" v) (Hcq.Q.dwords q);
  close_out oc

let compute_streams chip dev =
  let prog enable_private_segment_sgpr =
    {
      dev;
      prog_addr;
      rsrc1;
      rsrc2;
      rsrc3;
      wave32 = true;
      enable_private_segment_sgpr;
      enable_dispatch_ptr = false;
    }
  in
  let kernargs = Hcq.Buffer.make ~va:kernarg_va ~size:0x1000 ~meta:() () in
  let sg = Hcq.Signal.make signal_slot in
  let sg_tl = Hcq.Signal.make ~is_timeline:true ~owner:dev signal_slot in
  let wbuf = Hcq.Buffer.make ~va:write_va ~size:0x1000 ~meta:() () in
  let pbuf = Hcq.Buffer.make ~va:poll_va ~size:0x1000 ~meta:() () in
  let build name f =
    let q = Compute_queue.create dev in
    f q;
    dump name chip (Compute_queue.q q)
  in
  build "exec" (fun q ->
      Compute_queue.exec q (prog false) ~kernargs ~global_size ~local_size);
  (* exec rejects the flat-scratch descriptor on multi-die devices, so the
     scratch variant exists only for single-die chips. *)
  if dev.xccs = 1 then
    build "exec_scratch" (fun q ->
        Compute_queue.exec q (prog true) ~kernargs ~global_size ~local_size);
  build "signal" (fun q -> Compute_queue.signal q ~value:signal_value sg);
  build "signal_timeline" (fun q ->
      Compute_queue.signal q ~value:signal_value sg_tl);
  build "wait" (fun q -> Compute_queue.wait q ~value:wait_value sg);
  build "timestamp" (fun q -> Compute_queue.timestamp q sg);
  build "write32" (fun q -> Compute_queue.write q wbuf write32_value);
  build "write64" (fun q -> Compute_queue.write q ~b64:true wbuf write64_value);
  build "poll_bit" (fun q ->
      Compute_queue.poll_bit q pbuf ~value:poll_mask ~mask:poll_mask);
  build "memory_barrier" (fun q -> Compute_queue.memory_barrier q)

let sdma_streams chip dev =
  let sg = Hcq.Signal.make signal_slot in
  let sg_tl = Hcq.Signal.make ~is_timeline:true ~owner:dev signal_slot in
  let src = Hcq.Buffer.make ~va:copy_src_va ~size:0 ~meta:() () in
  let dst = Hcq.Buffer.make ~va:copy_dst_va ~size:0 ~meta:() () in
  let wbuf = Hcq.Buffer.make ~va:write_va ~size:0x1000 ~meta:() () in
  let build name f =
    let q = Copy_queue.create dev in
    f q;
    dump name chip (Copy_queue.q q)
  in
  build "sdma_copy_small" (fun q -> Copy_queue.copy q ~dest:dst ~src copy_small);
  build "sdma_copy_large" (fun q ->
      Copy_queue.copy q ~dest:dst ~src
        ((2 * dev.max_copy_size) + copy_large_extra));
  build "sdma_signal" (fun q -> Copy_queue.signal q ~value:signal_value sg);
  build "sdma_signal_timeline" (fun q ->
      Copy_queue.signal q ~value:signal_value sg_tl);
  build "sdma_wait" (fun q -> Copy_queue.wait q ~value:wait_value sg);
  build "sdma_timestamp" (fun q -> Copy_queue.timestamp q sg);
  build "sdma_write32" (fun q -> Copy_queue.write q wbuf write32_value);
  build "sdma_write64" (fun q -> Copy_queue.write q ~b64:true wbuf write64_value)

let () =
  let chips =
    [
      ( "gfx1100",
        make_dev ~target:(11, 0, 0) ~xccs:1 ~gc_version:(11, 0, 0)
          ~nbio_version:(4, 3, 0) ~sdma_version:(6, 0, 0) );
      ( "gfx942",
        make_dev ~target:(9, 4, 2) ~xccs:8 ~gc_version:(9, 4, 3)
          ~nbio_version:(7, 9, 0) ~sdma_version:(4, 4, 2) );
    ]
  in
  List.iter
    (fun (chip, dev) ->
      compute_streams chip dev;
      sdma_streams chip dev)
    chips
