(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Boot_defs

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n lsr 1)

(* The register [name] of [g]'s GC, which the GC of every descriptor layout
   has. *)
let register g name =
  match Register.find g name with
  | Some r -> r
  | None ->
      let a, b, c = g.Gpu.gc in
      invalid_argf "Gfx: GC %d.%d.%d has no register %s" a b c name

(* Queue descriptors *)

type queue = {
  ring : int;
  ring_bytes : int;
  read : int;
  write : int;
  eop : int;
  eop_bytes : int;
  doorbell : int;
}

(* The descriptor layouts are the kernel's v9_structs.h, v11_structs.h and
   v12_structs.h. *)
let layout_of_gc = function
  | 9, _, _ -> D.mqd_v9
  | 11, _, _ -> D.mqd_v11
  | 12, _, _ -> D.mqd_v12
  | a, b, c -> invalid_argf "Gfx.mqd: GC %d.%d.%d has no queue descriptor" a b c

(* The values the kernel's MQD initialisation gives a compute queue
   (gfx_v11_0_compute_mqd_init and its siblings). *)
let mqd_header = 0xC031_0800
let pipe_priority = 2
let queue_priority = 0xf
let quantum = 0x111
let hq_status0 = 0x2000_4000
let all_units = 0xffff_ffff

(* The stride between the descriptors of a queue's dies. *)
let die_stride = 0x1000

let mqd (g : Gpu.t) q ~base kind ~xcc =
  let { Gpu.gc; xccs; _ } = g in
  let kiq = kind = `Kiq and aql = kind = `Aql in
  let (f : D.mqd) = layout_of_gc gc in
  let b = Bytes.make f.sizeof '\000' in
  let set (off, _) v = Bytes.set_int32_le b off (Int32.of_int (lo32 v)) in
  let enc name kvs = Register.encode (register g name) kvs in
  set f.header mqd_header;
  set f.cp_mqd_base_addr_lo (lo32 base);
  set f.cp_mqd_base_addr_hi (hi32 base);
  set f.cp_hqd_pipe_priority pipe_priority;
  set f.cp_hqd_queue_priority queue_priority;
  set f.cp_hqd_quantum quantum;
  set f.cp_hqd_persistent_state
    (enc "regCP_HQD_PERSISTENT_STATE"
       [ ("preload_size", 0x55); ("preload_req", 1) ]);
  set f.cp_hqd_pq_base_lo (lo32 (q.ring lsr 8));
  set f.cp_hqd_pq_base_hi (hi32 (q.ring lsr 8));
  set f.cp_hqd_pq_rptr_report_addr_lo (lo32 q.read);
  set f.cp_hqd_pq_rptr_report_addr_hi (hi32 q.read);
  set f.cp_hqd_pq_wptr_poll_addr_lo (lo32 q.write);
  set f.cp_hqd_pq_wptr_poll_addr_hi (hi32 q.write);
  set f.cp_hqd_pq_doorbell_control
    (enc "regCP_HQD_PQ_DOORBELL_CONTROL"
       [ ("doorbell_offset", q.doorbell * 2); ("doorbell_en", 1) ]);
  set f.cp_hqd_pq_control
    (enc "regCP_HQD_PQ_CONTROL"
       ([
          ("rptr_block_size", 5);
          ("unord_dispatch", 0);
          ("queue_size", log2 (q.ring_bytes / 4) - 1);
        ]
       @ (if kiq then [ ("priv_state", 1); ("kmd_queue", 1) ] else [])
       @
       if aql then
         [
           ("queue_full_en", 1);
           ("slot_based_wptr", 2);
           ("no_update_rptr", Bool.to_int (xcc <> 0 || xccs = 1));
         ]
       else []));
  set f.cp_hqd_ib_control
    (enc "regCP_HQD_IB_CONTROL" [ ("min_ib_avail_size", 3) ]);
  set f.cp_hqd_hq_status0 hq_status0;
  set f.cp_mqd_control (enc "regCP_MQD_CONTROL" [ ("priv_state", 1) ]);
  set f.cp_hqd_vmid 0;
  set f.cp_hqd_aql_control (Bool.to_int aql);
  set f.cp_hqd_eop_base_addr_lo (lo32 (q.eop lsr 8));
  set f.cp_hqd_eop_base_addr_hi (hi32 (q.eop lsr 8));
  set f.cp_hqd_eop_control
    (enc "regCP_HQD_EOP_CONTROL" [ ("eop_size", log2 (q.eop_bytes / 4) - 1) ]);
  (* The thread management words first: on GFX9 four of them share their place
     with the fields of a queue across dies, set after them, as the kernel's
     kfd_mqd_manager_v9 does. *)
  List.iter (fun w -> set w all_units) f.compute_static_thread_mgmt;
  if aql && xccs > 1 then begin
    Option.iter (fun w -> set w 1) f.compute_tg_chunk_size;
    Option.iter (fun w -> set w xcc) f.compute_current_logic_xcc_id;
    Option.iter (fun w -> set w die_stride) f.cp_mqd_stride_size
  end;
  Bytes.to_string b

(* Shader arrays *)

(* GC 10 renamed the shader array fields, SH to SA. *)
let index (g : Gpu.t) sel =
  let reg = register g "regGRBM_GFX_INDEX" in
  let sa = if g.gc < (10, 0, 0) then "sh" else "sa" in
  Register.encode reg
    (match sel with
    | `Array (se, a) ->
        [
          ("se_index", se); (sa ^ "_index", a); ("instance_broadcast_writes", 1);
        ]
    | `All ->
        [
          ("se_broadcast_writes", 1);
          (sa ^ "_broadcast_writes", 1);
          ("instance_broadcast_writes", 1);
        ])
