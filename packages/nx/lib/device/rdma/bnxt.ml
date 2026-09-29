(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A Broadcom NetXtreme network adapter driven without its kernel driver: its
   firmware's requests (HWRM), the RoCE engine's commands (RCFW) and completions
   (CREQ), memory regions, and reliable connected queue pairs whose sends and
   receives the host posts and polls. *)

module D = Bnxt_defs
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci

(* The work queues: 128-byte WQEs, rings of 4096, and 4096-byte packets. *)
let wqe_size = 128
let ring_entries = 4096
let cq_entries = 4096
let mtu = 4096

(* The firmware's mailbox, at the start of BAR 0. *)
let chimp_comm = 0x0
let chimp_comm_trigger = 0x100

(* The context types the RoCE engine keeps in host memory, and the entries each
   takes beyond its minimum; type 15 closes the configuration. *)
let backing_store =
  [
    (0, 64);
    (1, 0);
    (2, 128);
    (3, 0);
    (4, 2);
    (5, 0);
    (6, 0);
    (14, 1024);
    (15, 0);
  ]

let access =
  D.cmdq_modify_qp_access_local_write lor D.cmdq_modify_qp_access_remote_write

let ( ||| ) = List.fold_left ( lor ) 0

let init_mask =
  ( ||| )
    D.
      [
        cmdq_modify_qp_modify_mask_state;
        cmdq_modify_qp_modify_mask_access;
        cmdq_modify_qp_modify_mask_pkey;
      ]

let rtr_mask =
  ( ||| )
    D.
      [
        init_mask;
        cmdq_modify_qp_modify_mask_dgid;
        cmdq_modify_qp_modify_mask_sgid_index;
        cmdq_modify_qp_modify_mask_hop_limit;
        cmdq_modify_qp_modify_mask_dest_mac;
        cmdq_modify_qp_modify_mask_path_mtu;
        cmdq_modify_qp_modify_mask_rq_psn;
        cmdq_modify_qp_modify_mask_min_rnr_timer;
        cmdq_modify_qp_modify_mask_max_dest_rd_atomic;
        cmdq_modify_qp_modify_mask_dest_qp_id;
      ]

let rts_mask =
  ( ||| )
    D.
      [
        cmdq_modify_qp_modify_mask_state;
        cmdq_modify_qp_modify_mask_access;
        cmdq_modify_qp_modify_mask_timeout;
        cmdq_modify_qp_modify_mask_retry_cnt;
        cmdq_modify_qp_modify_mask_rnr_retry;
        cmdq_modify_qp_modify_mask_max_rd_atomic;
        cmdq_modify_qp_modify_mask_sq_psn;
      ]

let ceildiv a b = (a + b - 1) / b

(* Structures, by the generated (byte offset, bytes) of their fields; fields of
   8 bytes hold at most 62 bits, which every address and size here does. *)

let set b (off, n) v =
  match n with
  | 1 -> Bytes.set_uint8 b off (v land 0xff)
  | 2 -> Bytes.set_uint16_le b off (v land 0xffff)
  | 4 -> Bytes.set_int32_le b off (Int32.of_int v)
  | 8 -> Bytes.set_int64_le b off (Int64.of_int v)
  | n -> invalid_arg (Printf.sprintf "Bnxt.set: a field of %d bytes" n)

let get s (off, n) =
  match n with
  | 1 -> Char.code s.[off]
  | 2 -> String.get_uint16_le s off
  | 4 -> Int32.to_int (String.get_int32_le s off) land 0xffff_ffff
  | 8 -> Int64.to_int (String.get_int64_le s off)
  | n -> invalid_arg (Printf.sprintf "Bnxt.get: a field of %d bytes" n)

(* Element [i] of the array field (offset, bytes of an element, elements). *)
let elt (off, n, _) i = (off + (i * n), n)

(* Waits until [ready ()], polling, for at most [timeout_ms]. *)
let wait_until ~timeout_ms what ready =
  let t0 = Unix.gettimeofday () in
  let rec go () =
    if not (ready ()) then
      if (Unix.gettimeofday () -. t0) *. 1000. > float_of_int timeout_ms then
        failwith (what ^ ": no answer in time")
      else go ()
  in
  go ()

(* Memory *)

(* A ring of entries of [stride] bytes in system memory, and the page list the
   adapter reads it through. [aux] follows the ring with 8 bytes an entry. *)
type queue = {
  ring : Mmio.t;
  stride : int;
  size : int; (* the ring's bytes, without [aux] *)
  pbl_level : int;
  pbl_addr : int;
  mutable write_idx : int;
  mutable read_idx : int;
}

type t = {
  pci : Pci.t;
  bar0 : Mmio.t; (* the mailboxes *)
  db : Mmio.t; (* the doorbells *)
  resp : Mmio.t; (* the firmware's answers *)
  resp_pa : int;
  mutable seq : int;
  mutable mac : int;
  mutable db_off : int;
  mutable creq : queue option;
  mutable cmdq : queue option;
  mutable creq_id : int;
  mutable nq_id : int;
  mutable rcfw_first : bool;
  mutable gid : string;
  mutable gid_id : int;
}

(* System memory of the adapter's machine, for the adapter's life. *)
let sysmem t n = Pci.alloc_sysmem t.pci n

(* The page list of [pages]: a level of the address alone, one table, or two; a
   queue's last two entries are marked. *)
let build_pbl t ?(queue = false) pages =
  match pages with
  | [ p ] -> (0, p)
  | _ ->
      let n = List.length pages in
      let values =
        List.mapi
          (fun i p ->
            let v = p lor D.ptu_pte_valid in
            if queue && i = n - 1 then v lor D.ptu_pte_last
            else if queue && i = n - 2 then v lor D.ptu_pte_next_to_last
            else v)
          pages
      in
      let words l =
        let b = Bytes.create (8 * List.length l) in
        List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) l;
        Bytes.unsafe_to_string b
      in
      let table, table_pages = sysmem t (ceildiv n 512 * 0x1000) in
      Mmio.write table 0 (words values);
      if List.length table_pages = 1 then (1, List.hd table_pages)
      else if List.length table_pages > 512 then
        failwith (Printf.sprintf "a page list of %d pages needs larger pages" n)
      else
        let top, top_pages = sysmem t 0x1000 in
        Mmio.write top 0
          (words (List.map (fun p -> p lor D.ptu_pte_valid) table_pages));
        (2, List.hd top_pages)

let alloc_queue t ?(stride = 16) ?(aux = false) ?entries () =
  let entries = Option.value entries ~default:(0x1000 / stride) in
  let size = entries * stride in
  let ring, pages = sysmem t (size + if aux then entries * 8 else 0) in
  let pbl_level, pbl_addr = build_pbl t ~queue:true pages in
  { ring; stride; size; pbl_level; pbl_addr; write_idx = 0; read_idx = 0 }

let slot q i = i mod (q.size / q.stride) * q.stride
let read_entry q i = Mmio.read q.ring (slot q i) q.stride
let write_entry q i data = Mmio.write q.ring (slot q i) data

(* The 8 bytes of [aux] for entry [i]. *)
let write_aux q i data =
  Mmio.write q.ring (q.size + (i mod (q.size / q.stride) * 8)) data

(* Doorbells *)

(* A doorbell's 64 bits: the queue, its path, type and validity in the high
   word, whose top bits the types use, then the index and the epoch. *)
let db_value xid typ index epoch =
  let high =
    xid land D.dbc_dbc_xid_mask lor D.dbc_dbc_path_roce lor typ
    lor D.bnxt_qplib_dbr_valid
  in
  Int64.logor
    (Int64.shift_left (Int64.of_int high) 32)
    (Int64.of_int
       (index land D.dbc_dbc_index_mask
       lor (epoch lsl D.bnxt_qplib_dbr_epoch_shift)))

let doorbell t xid typ index epoch =
  Mmio.barrier ();
  Mmio.set64 t.db t.db_off (db_value xid typ index epoch)

(* Firmware requests *)

(* Sends the request [req_type], filled by [fill], and is the answer's bytes
   once the firmware wrote them whole. *)
let hwrm t ?(timeout_ms = 10_000) name ~req_type fill =
  t.seq <- (t.seq + 1) land 0xffff;
  let b = Bytes.make D.hwrm_max_req_len '\000' in
  set b D.Hwrm_cmd_hdr.req_type req_type;
  set b D.Hwrm_cmd_hdr.cmpl_ring D.bnxt_hwrm_no_cmpl_ring;
  set b D.Hwrm_cmd_hdr.seq_id t.seq;
  set b D.Hwrm_cmd_hdr.target_id D.bnxt_hwrm_target;
  set b D.Hwrm_cmd_hdr.resp_addr t.resp_pa;
  fill b;
  Mmio.fill t.resp 0 (Mmio.length t.resp) '\000';
  Mmio.barrier ();
  Mmio.write t.bar0 chimp_comm (Bytes.unsafe_to_string b);
  Mmio.set32 t.bar0 chimp_comm_trigger 1;
  let header () = Mmio.read t.resp 0 D.Hwrm_resp_hdr.sizeof in
  let answered () =
    let h = header () in
    let n = get h D.Hwrm_resp_hdr.resp_len in
    n > 0
    && get h D.Hwrm_resp_hdr.seq_id = t.seq
    && Mmio.get8 t.resp (n - 1) <> 0
  in
  wait_until ~timeout_ms ("HWRM " ^ name) answered;
  let out = Mmio.read t.resp 0 (get (header ()) D.Hwrm_resp_hdr.resp_len) in
  match get out D.Hwrm_resp_hdr.error_code with
  | 0 -> out
  | e -> failwith (Printf.sprintf "HWRM %s: error %d" name e)

(* RoCE engine commands *)

(* Writes the command [opcode] of [size] bytes, filled by [fill], into the
   command queue, rings it, and is its completion's 16 bytes. The completions of
   asynchronous events, such as a queue pair's error, are skipped. *)
let rcfw t ?(timeout_ms = 20_000) name ~opcode ~size fill =
  let cmdq = Option.get t.cmdq and creq = Option.get t.creq in
  let slots = ceildiv size 16 in
  let b = Bytes.make (slots * 16) '\000' in
  set b D.Cmdq_base.opcode opcode;
  set b D.Cmdq_base.cmd_size slots;
  fill b;
  for i = 0 to slots - 1 do
    write_entry cmdq (cmdq.write_idx + i) (Bytes.sub_string b (i * 16) 16)
  done;
  cmdq.write_idx <- cmdq.write_idx + slots;
  let prod = cmdq.write_idx land 0xffff in
  let prod =
    if t.rcfw_first then begin
      t.rcfw_first <- false;
      prod lor (1 lsl D.firmware_first_flag)
    end
    else prod
  in
  Mmio.barrier ();
  Mmio.set32 t.bar0
    (D.rcfw_comm_base_offset + D.rcfw_pf_vf_comm_prod_offset)
    prod;
  Mmio.set32 t.bar0
    (D.rcfw_comm_base_offset + D.rcfw_comm_trig_offset)
    D.rcfw_cmdq_trig_val;
  let rec next () =
    let ready () =
      Char.code (read_entry creq creq.read_idx).[fst D.Creq_base.v]
      land D.creq_base_v
      <> creq.read_idx / 256 land 1
    in
    wait_until ~timeout_ms ("RCFW " ^ name) ready;
    let r = read_entry creq creq.read_idx in
    creq.read_idx <- creq.read_idx + 1;
    (* Arming the queue publishes its consumer index, which frees room for the
       next command. *)
    doorbell t t.creq_id D.dbc_dbc_type_nq_arm (creq.read_idx land 255)
      (creq.read_idx / 256 land 1);
    if
      get r D.Creq_base.type_ = D.creq_base_type_qp_event
      && get r D.Creq_base.event < D.creq_qp_event_event_qp_error_notification
    then r
    else next ()
  in
  let r = next () in
  match get r D.Creq_create_qp_resp.status with
  | 0 -> r
  | e -> failwith (Printf.sprintf "RCFW %s: status %d" name e)

(* Every completion of a command is laid out as [creq_create_qp_resp]. *)
let xid r = get r D.Creq_create_qp_resp.xid

(* Opening *)

(* The context memory the RoCE engine keeps its queue pairs, memory regions and
   completion queues in, per type and instance. *)
let setup_backing_store t =
  let counts = Hashtbl.create 16 in
  List.iter
    (fun (typ, extra) ->
      let caps =
        hwrm t "func_backing_store_qcaps_v2"
          ~req_type:D.hwrm_func_backing_store_qcaps_v2 (fun b ->
            set b D.Hwrm_func_backing_store_qcaps_v2_input.type_ typ)
      in
      let module Q = D.Hwrm_func_backing_store_qcaps_v2_output in
      let size = get caps Q.entry_size in
      let splits =
        List.filteri
          (fun i _ -> i < get caps Q.subtype_valid_cnt)
          [ Q.split_entry_0; Q.split_entry_1; Q.split_entry_2; Q.split_entry_3 ]
        |> List.map (get caps)
      in
      let n =
        if typ = 15 then Hashtbl.find counts 0
        else
          Int.max
            (get caps Q.min_num_entries)
            (List.fold_left ( + ) extra splits)
      in
      Hashtbl.replace counts typ n;
      let map = get caps Q.instance_bit_map in
      let instances =
        match
          List.filter (fun i -> (map lsr i) land 1 = 1) (List.init 8 Fun.id)
        with
        | [] -> [ 0 ]
        | l -> l
      in
      List.iter
        (fun instance ->
          let bytes = ceildiv (n * size) 0x1000 * 0x1000 in
          let mem, pages = sysmem t bytes in
          (match get caps Q.ctx_init_value with
          | 0 -> ()
          | v ->
              let init = Bytes.make bytes '\000' in
              let off = ref (get caps Q.ctx_init_offset) in
              while !off < bytes do
                Bytes.set_uint8 init !off v;
                off := !off + size
              done;
              Mmio.write mem 0 (Bytes.unsafe_to_string init));
          let level, base = build_pbl t pages in
          let module C = D.Hwrm_func_backing_store_cfg_v2_input in
          ignore
            (hwrm t "func_backing_store_cfg_v2"
               ~req_type:D.hwrm_func_backing_store_cfg_v2 (fun b ->
                 set b C.type_ typ;
                 set b C.instance instance;
                 set b C.entry_size size;
                 set b C.num_entries n;
                 set b C.page_dir base;
                 set b C.page_size_pbl_level level;
                 set b C.subtype_valid_cnt (List.length splits);
                 set b C.flags
                   (if typ = 15 then
                      D.func_backing_store_cfg_v2_req_flags_bs_cfg_all_done
                    else 0);
                 List.iteri
                   (fun j v ->
                     set b
                       (List.nth
                          [
                            C.split_entry_0;
                            C.split_entry_1;
                            C.split_entry_2;
                            C.split_entry_3;
                          ]
                          j)
                       v)
                   splits)))
        instances)
    backing_store

let ring_alloc t name fill =
  get
    (hwrm t name ~req_type:D.hwrm_ring_alloc fill)
    D.Hwrm_ring_alloc_output.ring_id

(* The engine's command queue and completion queue, and its notification queue,
   which is never armed nor serviced but which completion queues require. *)
let open_rcfw t =
  let module R = D.Hwrm_ring_alloc_input in
  let creq = alloc_queue t () and cmdq = alloc_queue t () in
  t.creq <- Some creq;
  t.cmdq <- Some cmdq;
  t.rcfw_first <- true;
  t.creq_id <-
    ring_alloc t "ring_alloc" (fun b ->
        set b R.ring_type D.ring_alloc_req_ring_type_nq;
        set b R.page_tbl_addr creq.pbl_addr;
        set b R.page_size 12;
        set b R.page_tbl_depth creq.pbl_level;
        set b R.length 256;
        set b R.int_mode D.ring_alloc_req_int_mode_msix);
  doorbell t t.creq_id D.dbc_dbc_type_nq_arm 0 0;
  let init = Bytes.make D.Cmdq_init.sizeof '\000' in
  set init D.Cmdq_init.cmdq_pbl cmdq.pbl_addr;
  set init D.Cmdq_init.creq_ring_id t.creq_id;
  set init D.Cmdq_init.cmdq_size_cmdq_lvl (256 lsl D.cmdq_init_cmdq_size_sft);
  Mmio.barrier ();
  Mmio.write t.bar0 D.rcfw_comm_base_offset (Bytes.unsafe_to_string init);
  let _, stats = sysmem t 0x1000 in
  let module S = D.Hwrm_stat_ctx_alloc_input in
  let stat_ctx =
    get
      (hwrm t "stat_ctx_alloc" ~req_type:D.hwrm_stat_ctx_alloc (fun b ->
           set b S.stats_dma_addr (List.hd stats);
           set b S.stats_dma_length 176))
      D.Hwrm_stat_ctx_alloc_output.stat_ctx_id
  in
  ignore
    (rcfw t "initialize_fw" ~opcode:D.cmdq_base_opcode_initialize_fw
       ~size:D.Cmdq_initialize_fw.sizeof (fun b ->
         set b D.Cmdq_initialize_fw.stat_ctx_id stat_ctx;
         set b D.Cmdq_base.flags
           D.cmdq_initialize_fw_flags_hw_requester_retx_supported));
  let nq = alloc_queue t () in
  t.nq_id <-
    ring_alloc t "ring_alloc" (fun b ->
        set b R.ring_type D.ring_alloc_req_ring_type_nq;
        set b R.page_tbl_addr nq.pbl_addr;
        set b R.page_size 12;
        set b R.page_tbl_depth nq.pbl_level;
        set b R.length 16;
        set b R.logical_id 1;
        set b R.int_mode D.ring_alloc_req_int_mode_msix)

(* The L2 receive path, which RoCE ingress requires though no Ethernet receive
   buffer is ever posted. *)
let open_l2 t =
  let module R = D.Hwrm_ring_alloc_input in
  let cq = alloc_queue t () in
  let ci =
    ring_alloc t "ring_alloc" (fun b ->
        set b R.enables D.ring_alloc_req_enables_nq_ring_id_valid;
        set b R.ring_type D.ring_alloc_req_ring_type_l2_cmpl;
        set b R.page_tbl_addr cq.pbl_addr;
        set b R.page_size 12;
        set b R.page_tbl_depth cq.pbl_level;
        set b R.length 16;
        set b R.nq_ring_id t.nq_id)
  in
  let rx = alloc_queue t () in
  let ri =
    ring_alloc t "ring_alloc" (fun b ->
        set b R.enables
          (D.ring_alloc_req_enables_nq_ring_id_valid
         lor D.ring_alloc_req_enables_rx_buf_size_valid);
        set b R.ring_type D.ring_alloc_req_ring_type_rx;
        set b R.page_tbl_addr rx.pbl_addr;
        set b R.page_size 12;
        set b R.page_tbl_depth rx.pbl_level;
        set b R.length 16;
        set b R.rx_buf_size 640;
        set b R.nq_ring_id t.nq_id)
  in
  let vnic =
    get
      (hwrm t "vnic_alloc" ~req_type:D.hwrm_vnic_alloc ignore)
      D.Hwrm_vnic_alloc_output.vnic_id
  in
  let module V = D.Hwrm_vnic_cfg_input in
  ignore
    (hwrm t "vnic_cfg" ~req_type:D.hwrm_vnic_cfg (fun b ->
         set b V.enables
           (D.vnic_cfg_req_enables_mru
          lor D.vnic_cfg_req_enables_default_rx_ring_id
          lor D.vnic_cfg_req_enables_default_cmpl_ring_id);
         set b V.vnic_id vnic;
         set b V.mru 9018;
         set b V.default_rx_ring_id ri;
         set b V.default_cmpl_ring_id ci));
  let module F = D.Hwrm_cfa_l2_filter_alloc_input in
  ignore
    (hwrm t "cfa_l2_filter_alloc" ~req_type:D.hwrm_cfa_l2_filter_alloc (fun b ->
         set b F.flags D.cfa_l2_filter_alloc_req_flags_path_rx;
         set b F.enables
           (D.cfa_l2_filter_alloc_req_enables_l2_addr
          lor D.cfa_l2_filter_alloc_req_enables_l2_addr_mask
          lor D.cfa_l2_filter_alloc_req_enables_dst_id);
         for i = 0 to 5 do
           set b (elt F.l2_addr i) ((t.mac lsr (8 * (5 - i))) land 0xff);
           set b (elt F.l2_addr_mask i) 0xff
         done;
         set b F.dst_id vnic))

(* The GID the adapter is known by on the fabric: the IPv4-mapped address
   10.x.y.z from the last three bytes of its MAC address. *)
let gid_of_mac mac =
  String.make 10 '\000' ^ "\xff\xff\x0a"
  ^ String.init 3 (fun i -> Char.chr ((mac lsr (8 * (2 - i))) land 0xff))

let boot pci =
  Pci.reset pci;
  let command = 0x04 and master = 0x04 in
  Pci.write_config pci command 2 (Pci.read_config pci command 2 lor master);
  let bar0 = Pci.map_bar pci 0 and db = Pci.map_bar pci 2 in
  let resp, resp_pages = Pci.alloc_sysmem pci 0x1000 in
  let t =
    {
      pci;
      bar0;
      db;
      resp;
      resp_pa = List.hd resp_pages;
      seq = 0;
      mac = 0;
      db_off = 0;
      creq = None;
      cmdq = None;
      creq_id = 0;
      nq_id = 0;
      rcfw_first = true;
      gid = "";
      gid_id = 0;
    }
  in
  ignore (hwrm t "ver_get" ~req_type:D.hwrm_ver_get ignore);
  ignore
    (hwrm t ~timeout_ms:40_000 "func_reset" ~req_type:D.hwrm_func_reset ignore);
  let caps =
    hwrm t "func_qcaps" ~req_type:D.hwrm_func_qcaps (fun b ->
        set b D.Hwrm_func_qcaps_input.fid 0xffff)
  in
  let module Q = D.Hwrm_func_qcaps_output in
  t.mac <-
    List.fold_left
      (fun acc i -> (acc lsl 8) lor get caps (elt Q.mac_address i))
      0 (List.init 6 Fun.id);
  ignore (hwrm t "func_drv_rgtr" ~req_type:D.hwrm_func_drv_rgtr ignore);
  let cfg =
    hwrm t "func_qcfg" ~req_type:D.hwrm_func_qcfg (fun b ->
        set b D.Hwrm_func_qcfg_input.fid 0xffff)
  in
  t.db_off <- get cfg D.Hwrm_func_qcfg_output.legacy_l2_db_size_kb * 1024;
  setup_backing_store t;
  open_rcfw t;
  open_l2 t;
  t.gid <- gid_of_mac t.mac;
  let module G = D.Cmdq_add_gid in
  t.gid_id <-
    xid
      (rcfw t "add_gid" ~opcode:D.cmdq_base_opcode_add_gid ~size:G.sizeof
         (fun b ->
           (* The GID's big-endian words, last first; the MAC's big-endian
              halves. *)
           for i = 0 to 3 do
             set b (elt G.gid i)
               (Int32.to_int (String.get_int32_be t.gid (4 * (3 - i)))
               land 0xffff_ffff)
           done;
           for i = 0 to 2 do
             set b (elt G.src_mac i)
               ((t.mac lsr (8 * (4 - (2 * i)))) land 0xffff)
           done));
  t

let fini t =
  ignore (hwrm t "func_drv_unrgtr" ~req_type:D.hwrm_func_drv_unrgtr ignore)

(* Memory regions *)

(* A memory region over [pages] of [1 lsl log_page] bytes: the key under which
   the adapter reads and writes [va, va + size) as those pages. *)
let register_mem t pages ~size ~log_page ~va =
  let pages =
    List.filteri (fun i _ -> i < ceildiv size (1 lsl log_page)) pages
  in
  let level, base = build_pbl t pages in
  let module M = D.Cmdq_register_mr in
  xid
    (rcfw t "register_mr" ~opcode:D.cmdq_base_opcode_register_mr ~size:M.sizeof
       (fun b ->
         set b M.flags D.cmdq_register_mr_flags_alloc_mr;
         set b M.log2_pg_size_lvl
           ((level lsl D.cmdq_register_mr_lvl_sft)
           lor (log_page lsl D.cmdq_register_mr_log2_pg_size_sft));
         set b M.access
           (D.cmdq_register_mr_access_local_write
          lor D.cmdq_register_mr_access_remote_write);
         set b M.log2_pbl_pg_size 12;
         set b M.pbl base;
         set b M.va va;
         set b M.mr_size size))

let unregister_mem t key =
  ignore
    (rcfw t "deregister_mr" ~opcode:D.cmdq_base_opcode_deregister_mr
       ~size:D.Cmdq_deregister_mr.sizeof (fun b ->
         set b D.Cmdq_deregister_mr.lkey key))

(* The page size of a region over [ranges] at [va]: the largest the adapter has
   to which every address and size is aligned, so that the page list is
   shortest. *)
let log_page ~va ranges =
  let align = List.fold_left (fun a (p, n) -> a lor p lor n) va ranges in
  List.fold_left
    (fun best l -> if align land ((1 lsl l) - 1) = 0 then l else best)
    12
    [ 12; 13; 16; 18; 20; 21; 22; 30 ]

(* The pages of [1 lsl log] bytes that [ranges] cover, in order. *)
let pages_of ~log ranges =
  List.concat_map
    (fun (p, n) -> List.init (n lsr log) (fun i -> p + (i lsl log)))
    ranges

(* Queue pairs *)

(* A reliable connected queue pair, its rings and completion queues, and the
   words it keeps in the adapter's memory: the sequence numbers of its sends and
   receives, and the PSN of its next send. Whoever posts on it, the host or a
   GPU, reads and bumps them, and a completion queue's entry [n] completes the
   WQE [n]. *)
type qp = {
  nic : t;
  sq : queue;
  rq : queue;
  scq : queue;
  rcq : queue;
  scq_id : int;
  rcq_id : int;
  qpn : int;
  words : Mmio.t; (* [sq_seq; rq_seq; psn] *)
}

let sq_seq = 0
let rq_seq = 8
let psn_word = 16

let modify_qp qp ~state ~mask ?(network = 0) fill =
  let module M = D.Cmdq_modify_qp in
  ignore
    (rcfw qp.nic "modify_qp" ~opcode:D.cmdq_base_opcode_modify_qp ~size:M.sizeof
       (fun b ->
         set b M.qp_cid qp.qpn;
         set b M.modify_mask mask;
         set b M.network_type_en_sqd_async_notify_new_state (state lor network);
         fill b))

let create_qp t =
  let cq () = alloc_queue t ~stride:D.Cq_base.sizeof ~entries:cq_entries () in
  let scq = cq () and rcq = cq () in
  let create q =
    let module C = D.Cmdq_create_cq in
    xid
      (rcfw t "create_cq" ~opcode:D.cmdq_base_opcode_create_cq ~size:C.sizeof
         (fun b ->
           set b C.cq_size cq_entries;
           set b C.pbl q.pbl_addr;
           set b C.pg_size_lvl q.pbl_level;
           set b C.cq_fco_cnq_id t.nq_id))
  in
  let scq_id = create scq in
  let rcq_id = create rcq in
  let sq = alloc_queue t ~stride:wqe_size ~aux:true ~entries:ring_entries () in
  let rq = alloc_queue t ~stride:wqe_size ~entries:ring_entries () in
  let module Q = D.Cmdq_create_qp in
  let qpn =
    xid
      (rcfw t "create_qp" ~opcode:D.cmdq_base_opcode_create_qp ~size:Q.sizeof
         (fun b ->
           set b Q.type_ D.cmdq_create_qp_type_rc;
           set b Q.scq_cid scq_id;
           set b Q.rcq_cid rcq_id;
           set b Q.sq_size ring_entries;
           set b Q.sq_fwo_sq_sge 6;
           set b Q.sq_pbl sq.pbl_addr;
           set b Q.sq_pg_size_sq_lvl sq.pbl_level;
           set b Q.rq_size ring_entries;
           set b Q.rq_fwo_rq_sge 6;
           set b Q.rq_pbl rq.pbl_addr;
           set b Q.rq_pg_size_rq_lvl rq.pbl_level))
  in
  let words, _ = sysmem t 0x1000 in
  Mmio.fill words 0 24 '\000';
  let qp = { nic = t; sq; rq; scq; rcq; scq_id; rcq_id; qpn; words } in
  modify_qp qp ~state:D.cmdq_modify_qp_new_state_init ~mask:init_mask (fun b ->
      set b D.Cmdq_modify_qp.access access;
      set b D.Cmdq_modify_qp.pkey 0xffff);
  qp

(* Connects [qp] to the queue pair [qpn] of the adapter at [gid] and [mac], RoCE
   v2 over IPv4. A send to a receive not yet posted is retried until one is. *)
let connect qp ~qpn ~gid ~mac =
  let module M = D.Cmdq_modify_qp in
  let network = D.cmdq_modify_qp_network_type_rocev2_ipv4 in
  modify_qp qp ~state:D.cmdq_modify_qp_new_state_rtr ~mask:rtr_mask ~network
    (fun b ->
      set b M.qp_type D.cmdq_modify_qp_qp_type_rc;
      set b M.access access;
      set b M.pkey 0xffff;
      for i = 0 to 3 do
        set b (elt M.dgid i)
          (Int32.to_int (String.get_int32_le gid (4 * i)) land 0xffff_ffff)
      done;
      set b M.sgid_index qp.nic.gid_id;
      set b M.hop_limit 64;
      for i = 0 to 2 do
        let hi = (mac lsr (8 * (5 - (2 * i)))) land 0xff
        and lo = (mac lsr (8 * (4 - (2 * i)))) land 0xff in
        set b (elt M.dest_mac i) (hi lor (lo lsl 8))
      done;
      set b M.min_rnr_timer 1;
      set b M.path_mtu_pingpong_push_enable D.cmdq_modify_qp_path_mtu_mtu_4096;
      set b M.max_dest_rd_atomic 4;
      set b M.dest_qp_id qpn);
  modify_qp qp ~state:D.cmdq_modify_qp_new_state_rts ~mask:rts_mask ~network
    (fun b ->
      set b M.qp_type D.cmdq_modify_qp_qp_type_rc;
      set b M.access access;
      set b M.max_rd_atomic 1;
      set b M.rnr_retry 7;
      set b M.retry_cnt 7;
      set b M.timeout 14)

(* A WQE of one SGE: the send or receive of [size] bytes at [va] of the region
   [key]. *)
let wqe ~send ~va ~key size =
  let b = Bytes.make 48 '\000' in
  if send then begin
    set b D.Sq_send_hdr.wqe_type D.sq_base_wqe_type_send;
    set b D.Sq_send_hdr.flags D.sq_send_flags_signal_comp;
    set b D.Sq_send_hdr.length size
  end
  else set b D.Rq_wqe_hdr.wqe_type D.rq_wqe_wqe_type_rcv;
  set b D.Sq_send_hdr.wqe_size 3;
  let sge (off, n) = (D.Sq_send_hdr.sizeof + off, n) in
  set b (sge D.Sq_sge.va_or_pa) va;
  set b (sge D.Sq_sge.l_key) key;
  set b (sge D.Sq_sge.size) size;
  Bytes.unsafe_to_string b

(* The packets of a send of [size] bytes: one per [mtu] bytes, at least one. *)
let packets size = Int.max 1 (ceildiv size mtu)

(* The MSN table entry of the send [idx] from [psn]: its slot, the PSN after it
   and its first. *)
let msn_entry idx psn size =
  let psn = psn land 0xffffff in
  ((idx mod ring_entries) lsl D.sq_msn_search_start_idx_sft)
  lor (((psn + packets size) land 0xffffff) lsl D.sq_msn_search_next_psn_sft)
  lor psn

let word w off = Int64.to_int (Mmio.get64 w off)
let set_word w off v = Mmio.set64 w off (Int64.of_int v)

(* Rings [xid]'s doorbell of type [typ] with the entry after [n] of a queue of
   [entries], and the parity of its pass. *)
let ring qp ~xid ~typ ~entries n =
  let n = n + 1 in
  doorbell qp.nic xid typ (n mod entries) (n / entries land 1)

(* Posts the send of [size] bytes at [va] of the region [key], and is its
   sequence number. *)
let post_send qp ~va ~key size =
  let n = word qp.words sq_seq and psn = word qp.words psn_word in
  write_entry qp.sq n (wqe ~send:true ~va ~key size);
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int (msn_entry n psn size));
  write_aux qp.sq n (Bytes.unsafe_to_string b);
  set_word qp.words sq_seq (n + 1);
  set_word qp.words psn_word (psn + packets size);
  ring qp ~xid:qp.qpn ~typ:D.dbc_dbc_type_sq ~entries:ring_entries n;
  n

let post_recv qp ~va ~key size =
  let n = word qp.words rq_seq in
  write_entry qp.rq n (wqe ~send:false ~va ~key size);
  set_word qp.words rq_seq (n + 1);
  ring qp ~xid:qp.qpn ~typ:D.dbc_dbc_type_rq ~entries:ring_entries n;
  n

(* Whether the completion entry [cqe] of a pass over a queue is written: its
   toggle bit differs from the parity of the pass that consumes it. *)
let cqe_ready cqe n =
  get cqe D.Cq_base.cqe_type_toggle land D.cq_base_toggle
  <> n / cq_entries land 1

(* Waits for the completion of the WQE [n] of [cq], acknowledges it, and raises
   if it failed. *)
let poll qp ~send ~timeout_ms n =
  let cq, id = if send then (qp.scq, qp.scq_id) else (qp.rcq, qp.rcq_id) in
  let what = if send then "BNXT send" else "BNXT receive" in
  wait_until ~timeout_ms what (fun () -> cqe_ready (read_entry cq n) n);
  let cqe = read_entry cq n in
  ring qp ~xid:id ~typ:D.dbc_dbc_type_cq ~entries:cq_entries n;
  match get cqe D.Cq_base.status with
  | 0 -> ()
  | s -> failwith (Printf.sprintf "%s: completion status %d" what s)
