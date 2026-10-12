(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The boot follows Linux v6.18's drivers/net/ethernet/broadcom/bnxt (the
   firmware's requests and context memory) and drivers/infiniband/hw/bnxt_re
   (the RoCE engine's command queue, queue pairs and doorbells), as tinygrad's
   runtime/support/rdma/bnxtdev.py does it without a kernel driver. *)

module D = Defs
module Machine = Rig_pci.Machine
module Function = Rig_pci.Function
module Window = Rig_pci.Window
module Gpus = Rig_pci.Gpus

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* A request the firmware or the system refuses, caught at the library's
   boundary as an [Error]. *)
exception Refused of string

let fail fmt = Printf.ksprintf (fun s -> raise (Refused s)) fmt
let guard f = match f () with v -> Ok v | exception Refused why -> Error why

(* Linux's include/linux/bnxt/hsi.h and bnxt/bnxt_hwrm.h are under the GPL
   alone, so their values are written here by hand: the fields of a doorbell's
   high word, and the target and completion ring of a request answered in memory
   alone. *)
let db_xid_mask = 0xfffff
let db_index_mask = 0xffffff
let db_path_roce = 0
let db_sq = 0
let db_rq = 0x1000_0000
let db_cq = 0x4000_0000
let db_nq_arm = 0xb000_0000
let hwrm_target = 0xffff
let no_ring = 0xffff

(* The work queues: 128-byte entries, rings and completion queues of 4096, and
   4096-byte packets. *)
let wqe_size = 128
let ring_entries = 4096
let mtu = 4096

(* The largest message: InfiniBand's, which the 32-bit length of an SGE
   holds. *)
let max_message = 1 lsl 31

(* A page of the NIC's page lists and rings. *)
let page = 0x1000

(* The firmware's mailbox, at the start of BAR 0; BAR 2 holds the doorbells. *)
let chimp_comm = 0x0
let chimp_comm_trigger = 0x100
let doorbell_bar = 2

(* How long the firmware may take to answer a request or a command. *)
let hwrm_us = 10_000_000
let reset_us = 40_000_000
let rcfw_us = 20_000_000

(* The context types the RoCE engine keeps in host memory, and the entries each
   takes beyond its minimum: queue pairs first, whose count type 15 repeats to
   close the configuration. *)
let qp_contexts = (0, 64)
let contexts = [ (1, 0); (2, 128); (3, 0); (4, 2); (5, 0); (6, 0); (14, 1024) ]
let last_type = 15
let ceildiv a b = (a + b - 1) / b

(* Structures *)

(* Fields are the generated (byte offset, bytes); fields of 8 bytes hold at most
   62 bits, which every address and size here does. *)

let set b (off, n) v =
  match n with
  | 1 -> Bytes.set_uint8 b off (v land 0xff)
  | 2 -> Bytes.set_uint16_le b off (v land 0xffff)
  | 4 -> Bytes.set_int32_le b off (Int32.of_int v)
  | 8 -> Bytes.set_int64_le b off (Int64.of_int v)
  | n -> invalid_argf "Rig_bnxt: a field of %d bytes" n

let get s (off, n) =
  match n with
  | 1 -> Char.code s.[off]
  | 2 -> String.get_uint16_le s off
  | 4 -> Int32.to_int (String.get_int32_le s off) land 0xffff_ffff
  | 8 -> Int64.to_int (String.get_int64_le s off)
  | n -> invalid_argf "Rig_bnxt: a field of %d bytes" n

(* The firmware's channel *)

type hw = {
  fn : Function.t;
  bar0 : Window.t;
  resp : Window.t; (* the firmware's answers *)
  resp_bus : int;
  lock : Mutex.t; (* the requests and commands, and [allocs] *)
  mutable seq : int;
  mutable allocs : Window.t list; (* system memory, freed at close *)
}

let alloc fn n =
  match Function.alloc_dma fn n with
  | Error why -> fail "%s" why
  | Ok None -> fail "no system memory for %d bytes" n
  | Ok (Some m) -> m

(* [n] bytes of system memory the NIC reaches, and the runs it reaches them
   at. *)
let sysmem hw n =
  let w, runs = alloc hw.fn n in
  hw.allocs <- w :: hw.allocs;
  (w, runs)

(* The addresses of the pages of [1 lsl log] bytes that hold the first [bytes]
   of [runs], in order. The machine's page may be larger than the NIC's, so
   [runs] may cover more. *)
let pages ?(log = 12) ~bytes runs =
  List.concat_map
    (fun (p, n) -> List.init (n lsr log) (fun i -> p + (i lsl log)))
    runs
  |> List.filteri (fun i _ -> i < ceildiv bytes (1 lsl log))

let free hw w =
  hw.allocs <- List.filter (fun a -> a != w) hw.allocs;
  Function.free_dma hw.fn w

let wait hw ~us what ready =
  match Function.wait hw.fn ~us what ready with
  | Ok () -> ()
  | Error why -> fail "%s" why

(* Writes [s], whose length is a multiple of 4, to the registers at [off] of
   [w], a 32-bit store each. *)
let write_registers w off s =
  for i = 0 to (String.length s / 4) - 1 do
    Window.set32 w
      (off + (4 * i))
      (Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)
  done

let words values =
  let b = Bytes.create (8 * List.length values) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) values;
  Bytes.unsafe_to_string b

(* The page list of the pages at [addrs], its level and base: the address alone,
   one table, or a table of tables. A queue's last two entries are marked. The
   list's own memory comes last. *)
let page_list hw ?(queue = false) addrs =
  match addrs with
  | [ p ] -> (0, p, [])
  | _ -> (
      let n = List.length addrs in
      let entry i p =
        let v = p lor D.ptu_pte_valid in
        if queue && i = n - 1 then v lor D.ptu_pte_last
        else if queue && i = n - 2 then v lor D.ptu_pte_next_to_last
        else v
      in
      let bytes = ceildiv n 512 * page in
      let table, runs = sysmem hw bytes in
      Window.write table 0 (words (List.mapi entry addrs));
      match pages ~bytes runs with
      | [ t ] -> (1, t, [ table ])
      | tables when List.length tables > 512 ->
          fail "a page list of %d pages is longer than two levels hold" n
      | tables ->
          let top, runs = sysmem hw page in
          Window.write top 0
            (words (List.map (fun p -> p lor D.ptu_pte_valid) tables));
          (2, fst (List.hd runs), [ table; top ]))

(* Sends the request [req_type], filled by [fill], and is the answer once the
   firmware has written it whole. *)
let hwrm hw ?(us = hwrm_us) name ~req_type fill =
  hw.seq <- (hw.seq + 1) land 0xffff;
  let b = Bytes.make D.hwrm_max_req_len '\000' in
  let module H = D.Hwrm_cmd_hdr in
  set b H.req_type req_type;
  set b H.cmpl_ring no_ring;
  set b H.seq_id hw.seq;
  set b H.target_id hwrm_target;
  set b H.resp_addr hw.resp_bus;
  fill b;
  Window.fill hw.resp 0 (Window.length hw.resp) '\000';
  Window.barrier ();
  write_registers hw.bar0 chimp_comm (Bytes.unsafe_to_string b);
  Window.set32 hw.bar0 chimp_comm_trigger 1;
  let module R = D.Hwrm_resp_hdr in
  let header () = Window.read hw.resp 0 R.sizeof in
  (* The answer's last byte is its valid flag, written last. *)
  let answered () =
    let h = header () in
    let n = get h R.resp_len in
    n > 0
    && n <= Window.length hw.resp
    && get h R.seq_id = hw.seq
    && Window.get8 hw.resp (n - 1) <> 0
  in
  wait hw ~us ("HWRM " ^ name) answered;
  let out = Window.read hw.resp 0 (get (header ()) R.resp_len) in
  match get out R.error_code with
  | 0 -> out
  | e -> fail "HWRM %s: error %d" name e

let ring_alloc hw fill =
  get
    (hwrm hw "ring_alloc" ~req_type:D.hwrm_ring_alloc fill)
    D.Hwrm_ring_alloc_output.ring_id

(* Doorbells *)

(* A doorbell's 64 bits: the queue, the RoCE path, the type and the valid bit in
   the high word, then the index and the parity of its pass over the ring. *)
let db_value xid typ index epoch =
  let high =
    xid land db_xid_mask lor db_path_roce lor typ lor D.bnxt_qplib_dbr_valid
  in
  Int64.logor
    (Int64.shift_left (Int64.of_int high) 32)
    (Int64.of_int
       (index land db_index_mask lor (epoch lsl D.bnxt_qplib_dbr_epoch_shift)))

let doorbell db xid typ index epoch =
  Window.barrier ();
  Window.set64 db 0 (db_value xid typ index epoch)

(* Rings [xid]'s doorbell of type [typ] for the entries before [n] of a ring of
   [ring_entries]. *)
let ring db ~xid ~typ n =
  doorbell db xid typ (n mod ring_entries) (n / ring_entries land 1)

(* Queues *)

(* A ring of [entries] of [stride] bytes in system memory, and its page list.
   [aux] follows the ring with 8 bytes an entry. *)
type queue = {
  ring : Window.t;
  stride : int;
  entries : int;
  level : int;
  pbl : int;
}

let queue hw ?(stride = 16) ?(aux = false) ?(entries = page / 16) () =
  let size = entries * stride in
  let bytes = size + if aux then entries * 8 else 0 in
  let ring, runs = sysmem hw bytes in
  let level, pbl, _ = page_list hw ~queue:true (pages ~bytes runs) in
  { ring; stride; entries; level; pbl }

let slot q i = i mod q.entries * q.stride

(* The fields of a ring's request that place [q]. *)
let ring_fields b (q : queue) =
  let module R = D.Hwrm_ring_alloc_input in
  set b R.page_tbl_addr q.pbl;
  set b R.page_size 12;
  set b R.page_tbl_depth q.level

(* The RoCE engine's commands *)

type rcfw = {
  cmdq : queue;
  creq : queue;
  creq_id : int;
  db : Window.t; (* the NIC's doorbell *)
  mutable prod : int; (* the command slots written *)
  mutable cons : int; (* the completions read *)
  mutable first : bool;
}

(* Writes the command [opcode] of [size] bytes, filled by [fill], into the
   command queue, rings it, and is its completion. The completions of
   asynchronous events, such as a queue pair's error, are skipped. *)
let rcfw hw rc ?(us = rcfw_us) name ~opcode ~size fill =
  let slots = ceildiv size 16 in
  let b = Bytes.make (slots * 16) '\000' in
  set b D.Cmdq_base.opcode opcode;
  set b D.Cmdq_base.cmd_size slots;
  fill b;
  for i = 0 to slots - 1 do
    Window.blit_string (Bytes.unsafe_to_string b) (i * 16) rc.cmdq.ring
      (slot rc.cmdq (rc.prod + i))
      16
  done;
  rc.prod <- rc.prod + slots;
  let prod = rc.prod land 0xffff in
  let prod =
    if rc.first then prod lor (1 lsl D.firmware_first_flag) else prod
  in
  rc.first <- false;
  Window.barrier ();
  Window.set32 hw.bar0
    (D.rcfw_comm_base_offset + D.rcfw_pf_vf_comm_prod_offset)
    prod;
  Window.set32 hw.bar0
    (D.rcfw_comm_base_offset + D.rcfw_comm_trig_offset)
    D.rcfw_cmdq_trig_val;
  let creq = rc.creq in
  let rec next () =
    let ready () =
      Window.get8 creq.ring (slot creq rc.cons + fst D.Creq_base.v)
      land D.creq_base_v
      <> rc.cons / creq.entries land 1
    in
    wait hw ~us ("RCFW " ^ name) ready;
    Window.barrier ();
    let r = Window.read creq.ring (slot creq rc.cons) creq.stride in
    rc.cons <- rc.cons + 1;
    (* Arming the queue publishes its consumer index, which frees room for the
       next command. *)
    doorbell rc.db rc.creq_id db_nq_arm (rc.cons mod creq.entries)
      (rc.cons / creq.entries land 1);
    if
      get r D.Creq_base.type_ = D.creq_base_type_qp_event
      && get r D.Creq_base.event < D.creq_qp_event_event_qp_error_notification
    then r
    else next ()
  in
  let r = next () in
  match get r D.Creq_create_qp_resp.status with
  | 0 -> r
  | e -> fail "RCFW %s: status %d" name e

(* Every completion of a command is laid out as [creq_create_qp_resp]. *)
let xid r = get r D.Creq_create_qp_resp.xid

(* Booting *)

(* The context memory the RoCE engine keeps its queue pairs, regions and
   completion queues in, per type and instance. *)
let backing hw =
  let module Q = D.Hwrm_func_backing_store_qcaps_v2_output in
  let module C = D.Hwrm_func_backing_store_cfg_v2_input in
  let qsplits =
    [ Q.split_entry_0; Q.split_entry_1; Q.split_entry_2; Q.split_entry_3 ]
  in
  let csplits =
    [ C.split_entry_0; C.split_entry_1; C.split_entry_2; C.split_entry_3 ]
  in
  let configure ?count (typ, extra) =
    let caps =
      hwrm hw "func_backing_store_qcaps_v2"
        ~req_type:D.hwrm_func_backing_store_qcaps_v2 (fun b ->
          set b D.Hwrm_func_backing_store_qcaps_v2_input.type_ typ)
    in
    let size = get caps Q.entry_size in
    let splits =
      List.filteri (fun i _ -> i < get caps Q.subtype_valid_cnt) qsplits
      |> List.map (get caps)
    in
    let n =
      match count with
      | Some n -> n
      | None ->
          Int.max
            (get caps Q.min_num_entries)
            (List.fold_left ( + ) extra splits)
    in
    let map = get caps Q.instance_bit_map in
    let instances =
      match
        List.filter (fun i -> (map lsr i) land 1 = 1) (List.init 8 Fun.id)
      with
      | [] -> [ 0 ]
      | l -> l
    in
    let instance i =
      let bytes = ceildiv (n * size) page * page in
      let mem, runs = sysmem hw bytes in
      (match get caps Q.ctx_init_value with
      | 0 -> ()
      | v ->
          let off = ref (get caps Q.ctx_init_offset) in
          while !off < bytes do
            Window.set8 mem !off v;
            off := !off + size
          done);
      let level, dir, _ = page_list hw (pages ~bytes runs) in
      ignore
        (hwrm hw "func_backing_store_cfg_v2"
           ~req_type:D.hwrm_func_backing_store_cfg_v2 (fun b ->
             set b C.type_ typ;
             set b C.instance i;
             set b C.entry_size size;
             set b C.num_entries n;
             set b C.page_dir dir;
             set b C.page_size_pbl_level level;
             set b C.subtype_valid_cnt (List.length splits);
             if typ = last_type then
               set b C.flags
                 D.hwrm_func_backing_store_cfg_v2_input_flags_bs_cfg_all_done;
             List.iteri (fun j v -> set b (List.nth csplits j) v) splits))
    in
    List.iter instance instances;
    n
  in
  let qps = configure qp_contexts in
  List.iter (fun c -> ignore (configure c)) contexts;
  ignore (configure ~count:qps (last_type, 0))

(* The engine's command queue and its completions, then the notification queue
   that completion queues need, which is never armed nor serviced. Is the
   channel and the notification queue's ring. *)
let open_rcfw hw db =
  let module R = D.Hwrm_ring_alloc_input in
  let creq = queue hw () and cmdq = queue hw () in
  let creq_id =
    ring_alloc hw (fun b ->
        set b R.ring_type D.hwrm_ring_alloc_input_ring_type_nq;
        ring_fields b creq;
        set b R.length creq.entries;
        set b R.int_mode D.hwrm_ring_alloc_input_int_mode_msix)
  in
  doorbell db creq_id db_nq_arm 0 0;
  let init = Bytes.make D.Cmdq_init.sizeof '\000' in
  set init D.Cmdq_init.cmdq_pbl cmdq.pbl;
  set init D.Cmdq_init.creq_ring_id creq_id;
  set init D.Cmdq_init.cmdq_size_cmdq_lvl
    ((cmdq.entries lsl D.cmdq_init_cmdq_size_sft) lor cmdq.level);
  Window.barrier ();
  write_registers hw.bar0 D.rcfw_comm_base_offset (Bytes.unsafe_to_string init);
  let rc = { cmdq; creq; creq_id; db; prod = 0; cons = 0; first = true } in
  let _, stats = sysmem hw page in
  let module S = D.Hwrm_stat_ctx_alloc_input in
  let stat_ctx =
    get
      (hwrm hw "stat_ctx_alloc" ~req_type:D.hwrm_stat_ctx_alloc (fun b ->
           set b S.stats_dma_addr (fst (List.hd stats));
           set b S.stats_dma_length 176))
      D.Hwrm_stat_ctx_alloc_output.stat_ctx_id
  in
  ignore
    (rcfw hw rc "initialize_fw" ~opcode:D.cmdq_base_opcode_initialize_fw
       ~size:D.Cmdq_initialize_fw.sizeof (fun b ->
         set b D.Cmdq_initialize_fw.stat_ctx_id stat_ctx;
         set b D.Cmdq_base.flags
           D.cmdq_initialize_fw_flags_hw_requester_retx_supported));
  let nq = queue hw () in
  let nq_id =
    ring_alloc hw (fun b ->
        set b R.ring_type D.hwrm_ring_alloc_input_ring_type_nq;
        ring_fields b nq;
        set b R.length 16;
        set b R.logical_id 1;
        set b R.int_mode D.hwrm_ring_alloc_input_int_mode_msix)
  in
  (rc, nq_id)

(* The Ethernet receive path, which RoCE's ingress needs though no Ethernet
   buffer is ever posted: a completion ring, a receive ring, a virtual NIC over
   them, and a filter that steers the NIC's MAC address to it. *)
let open_l2 hw ~nq_id ~mac =
  let module R = D.Hwrm_ring_alloc_input in
  let cq = queue hw () in
  let cq_id =
    ring_alloc hw (fun b ->
        set b R.enables D.hwrm_ring_alloc_input_enables_nq_ring_id_valid;
        set b R.ring_type D.hwrm_ring_alloc_input_ring_type_l2_cmpl;
        ring_fields b cq;
        set b R.length 16;
        set b R.nq_ring_id nq_id)
  in
  let rx = queue hw () in
  let rx_id =
    ring_alloc hw (fun b ->
        set b R.enables
          (D.hwrm_ring_alloc_input_enables_nq_ring_id_valid
         lor D.hwrm_ring_alloc_input_enables_rx_buf_size_valid);
        set b R.ring_type D.hwrm_ring_alloc_input_ring_type_rx;
        ring_fields b rx;
        set b R.length 16;
        set b R.rx_buf_size 640;
        set b R.nq_ring_id nq_id)
  in
  let vnic =
    get
      (hwrm hw "vnic_alloc" ~req_type:D.hwrm_vnic_alloc ignore)
      D.Hwrm_vnic_alloc_output.vnic_id
  in
  let module V = D.Hwrm_vnic_cfg_input in
  ignore
    (hwrm hw "vnic_cfg" ~req_type:D.hwrm_vnic_cfg (fun b ->
         set b V.enables
           (D.hwrm_vnic_cfg_input_enables_mru
          lor D.hwrm_vnic_cfg_input_enables_default_rx_ring_id
          lor D.hwrm_vnic_cfg_input_enables_default_cmpl_ring_id);
         set b V.vnic_id vnic;
         set b V.mru 9018;
         set b V.default_rx_ring_id rx_id;
         set b V.default_cmpl_ring_id cq_id));
  let module F = D.Hwrm_cfa_l2_filter_alloc_input in
  ignore
    (hwrm hw "cfa_l2_filter_alloc" ~req_type:D.hwrm_cfa_l2_filter_alloc
       (fun b ->
         set b F.flags D.hwrm_cfa_l2_filter_alloc_input_flags_path_rx;
         set b F.enables
           (D.hwrm_cfa_l2_filter_alloc_input_enables_l2_addr
          lor D.hwrm_cfa_l2_filter_alloc_input_enables_l2_addr_mask
          lor D.hwrm_cfa_l2_filter_alloc_input_enables_dst_id);
         Bytes.blit_string mac 0 b (fst F.l2_addr) 6;
         Bytes.fill b (fst F.l2_addr_mask) 6 '\xff';
         set b F.dst_id vnic))

(* The address the NIC is known by on the fabric: the IPv4 address 10.x.y.z,
   from the last three bytes of its MAC address, mapped into IPv6. *)
let gid_of_mac mac = String.make 10 '\000' ^ "\xff\xff\x0a" ^ String.sub mac 3 3

(* NICs *)

type t = {
  name : string;
  hold : Gpus.hold;
  hw : hw;
  rc : rcfw;
  nq_id : int;
  mac : string;
  gid : string;
  gid_index : int;
  mutable closed : bool;
}

let name nic = nic.name
let bus nic = Function.bus nic.hw.fn

let live nic what =
  if nic.closed then invalid_argf "Rig_bnxt.%s: %s is closed" what nic.name

(* [f ()] with the NIC's channel to its firmware held, a refusal as an
   [Error]. *)
let command nic what f =
  live nic what;
  Mutex.protect nic.hw.lock (fun () -> guard f)

let is_bnxt (id : Machine.id) =
  id.vendor = 0x14e4 && id.class_ lsr 16 = 0x02 && id.device = 0x1760

(* The kernel driver serves the NIC through network interfaces, which hold no
   file, and its RoCE driver through files with a [dev] file under the
   function's directory, which {!Gpus.detach} finds itself. *)
let nics =
  Gpus.make ~name:"BNXT" ~noun:"NIC" ~memory_bar:None ~siblings:Keep
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:10_000 ~reset:Function.reset is_bnxt

let buses ?(machine = Machine.this) () = Gpus.buses nics machine
let count ?machine () = List.length (buses ?machine ())

let map fn i =
  match Function.map ~combine:false fn i with
  | Ok w -> w
  | Error why -> fail "%s" why

(* Resets the NIC, registers with its firmware, gives the RoCE engine its
   context memory and command queue, opens the Ethernet receive path, and adds
   the NIC's address. [registered] is set once the firmware knows the process,
   which the NIC's stop then undoes. *)
let boot ~name ~channel ~registered h fn =
  (match Function.reset fn with Ok () -> () | Error why -> fail "%s" why);
  Function.set_bus_master fn true;
  let bar0 = map fn 0 and dbs = map fn doorbell_bar in
  let resp, runs = alloc fn page in
  let hw =
    {
      fn;
      bar0;
      resp;
      resp_bus = fst (List.hd runs);
      lock = Mutex.create ();
      seq = 0;
      allocs = [ resp ];
    }
  in
  channel := Some hw;
  ignore (hwrm hw "ver_get" ~req_type:D.hwrm_ver_get ignore);
  ignore (hwrm hw ~us:reset_us "func_reset" ~req_type:D.hwrm_func_reset ignore);
  let caps =
    hwrm hw "func_qcaps" ~req_type:D.hwrm_func_qcaps (fun b ->
        set b D.Hwrm_func_qcaps_input.fid 0xffff)
  in
  let mac = String.sub caps (fst D.Hwrm_func_qcaps_output.mac_address) 6 in
  ignore (hwrm hw "func_drv_rgtr" ~req_type:D.hwrm_func_drv_rgtr ignore);
  registered := true;
  let cfg =
    hwrm hw "func_qcfg" ~req_type:D.hwrm_func_qcfg (fun b ->
        set b D.Hwrm_func_qcfg_input.fid 0xffff)
  in
  let db_off = get cfg D.Hwrm_func_qcfg_output.legacy_l2_db_size_kb * 1024 in
  let db = Window.sub dbs db_off 8 in
  backing hw;
  let rc, nq_id = open_rcfw hw db in
  open_l2 hw ~nq_id ~mac;
  let gid = gid_of_mac mac in
  let module G = D.Cmdq_add_gid in
  let gid_index =
    xid
      (rcfw hw rc "add_gid" ~opcode:D.cmdq_base_opcode_add_gid ~size:G.sizeof
         (fun b ->
           (* The GID's bytes in reverse order, and the MAC's with each 16-bit
              half swapped. *)
           for i = 0 to 15 do
             Bytes.set b (fst G.gid + i) gid.[15 - i]
           done;
           for i = 0 to 5 do
             Bytes.set b (fst G.src_mac + i) mac.[i lxor 1]
           done))
  in
  { name; hold = h; hw; rc; nq_id; mac; gid; gid_index; closed = false }

(* The NIC's stop: a NIC registered with its firmware, whose machine still
   answers, unregisters, and the hold then turns its bus mastering off. A NIC
   stopped otherwise is reset at its next open, as every open does. *)
let stop channel registered () =
  match !channel with
  | Some hw when !registered && Function.failed hw.fn = None -> (
      Mutex.protect hw.lock @@ fun () ->
      match
        hwrm hw "func_drv_unrgtr" ~req_type:D.hwrm_func_drv_unrgtr ignore
      with
      | _ -> `Clean
      | exception Refused _ -> `Lost)
  | Some _ | None -> `Lost

(* Frees the system memory of a NIC whose function is released, which reaches it
   no more. *)
let release hw =
  Mutex.protect hw.lock (fun () ->
      List.iter (Function.free_dma hw.fn) hw.allocs;
      hw.allocs <- [])

let index what i =
  if i < 0 then invalid_argf "Rig_bnxt.%s: NIC %d is negative" what i

let open_ ?(machine = Machine.this) i =
  index "open_" i;
  let name = Gpus.name nics i in
  let channel = ref None and registered = ref false in
  let start h fn =
    Gpus.set_stop h (stop channel registered);
    guard (fun () -> boot ~name ~channel ~registered h fn)
  in
  match Gpus.open_ nics machine i start with
  | Ok nic -> Ok nic
  | Error why ->
      Option.iter release !channel;
      Error why
  | exception e ->
      Option.iter release !channel;
      raise e

let close nic =
  if not nic.closed then begin
    nic.closed <- true;
    ignore (Gpus.stop nic.hold);
    release nic.hw
  end

let detach i =
  index "detach" i;
  Gpus.detach nics Machine.this i

let attach i =
  index "attach" i;
  Gpus.attach nics Machine.this i

(* Regions *)

module Region = struct
  type nonrec t = {
    nic : t;
    key : int;
    va : int;
    bytes : int;
    memory : Window.t list; (* its page list's, then any it was made over *)
    mutable gone : (unit, string) result option;
  }

  (* The page size of a region over [runs] at [va]: the largest the NIC has to
     which every address and size is aligned, so that its page list is
     shortest. *)
  let log_page ~va runs =
    let align = List.fold_left (fun a (p, n) -> a lor p lor n) va runs in
    List.fold_left
      (fun best l -> if align land ((1 lsl l) - 1) = 0 then l else best)
      12
      [ 12; 13; 16; 18; 20; 21; 22; 30 ]

  (* The region over the pages of [runs], under which the NIC reads and writes
     [va, va + bytes), [va] being the first run's address. [own] is memory the
     region frees at its deregistration, or now if the firmware refuses it. *)
  let register nic runs ~bytes ~own =
    let va = fst (List.hd runs) in
    let log = log_page ~va runs in
    let level, pbl, list = page_list nic.hw (pages ~log ~bytes runs) in
    let memory = list @ own in
    let module M = D.Cmdq_register_mr in
    match
      rcfw nic.hw nic.rc "register_mr" ~opcode:D.cmdq_base_opcode_register_mr
        ~size:M.sizeof (fun b ->
          set b M.flags D.cmdq_register_mr_flags_alloc_mr;
          set b M.log2_pg_size_lvl
            ((level lsl D.cmdq_register_mr_lvl_sft)
            lor (log lsl D.cmdq_register_mr_log2_pg_size_sft));
          set b M.access
            (D.cmdq_register_mr_access_local_write
           lor D.cmdq_register_mr_access_remote_write);
          set b M.log2_pbl_pg_size 12;
          set b M.pbl pbl;
          set b M.va va;
          set b M.mr_size bytes)
    with
    | r -> { nic; key = xid r; va; bytes; memory; gone = None }
    | exception e ->
        List.iter (free nic.hw) memory;
        raise e

  let host nic n =
    if n <= 0 then invalid_argf "Rig_bnxt.Region.host: %d bytes" n;
    command nic "Region.host" @@ fun () ->
    let w, runs = sysmem nic.hw n in
    (register nic runs ~bytes:(Window.length w) ~own:[ w ], w)

  let device nic runs =
    if runs = [] then invalid_arg "Rig_bnxt.Region.device: no runs";
    List.iter
      (fun (a, n) ->
        if a < 0 || n <= 0 || a land (page - 1) <> 0 || n land (page - 1) <> 0
        then invalid_argf "Rig_bnxt.Region.device: the run (%#x, %d)" a n)
      runs;
    command nic "Region.device" @@ fun () ->
    if Function.addressing nic.hw.fn = Machine.Iommu then
      fail
        "the NIC is behind an IOMMU, which maps no other device's memory for it";
    let bytes = List.fold_left (fun s (_, n) -> s + n) 0 runs in
    register nic runs ~bytes ~own:[]

  let deregister r =
    match r.gone with
    | Some result -> result
    | None ->
        let result =
          command r.nic "Region.deregister" @@ fun () ->
          ignore
            (rcfw r.nic.hw r.nic.rc "deregister_mr"
               ~opcode:D.cmdq_base_opcode_deregister_mr
               ~size:D.Cmdq_deregister_mr.sizeof (fun b ->
                 set b D.Cmdq_deregister_mr.lkey r.key));
          List.iter (free r.nic.hw) r.memory
        in
        r.gone <- Some result;
        result
end

(* Queue pairs *)

module Qp = struct
  type endpoint = { qp : int; gid : string; mac : string }
  type side = Send | Receive

  (* The states {!connect} takes a queue pair through, so that a connect the
     firmware refused part-way resumes where it stopped. *)
  type state = Initialized | Ready_to_receive | Ready_to_send

  type nonrec t = {
    nic : t;
    qpn : int;
    sq : queue;
    rq : queue;
    scq : queue;
    rcq : queue;
    scq_id : int;
    rcq_id : int;
    mutable state : state;
    mutable failed : string option;
        (* why the queue pair failed: its entries without a completion then
           complete with this error *)
    lengths : int array; (* the bytes of each send in the ring *)
    mutable sent : int; (* sends posted *)
    mutable sends_done : int; (* their completions taken *)
    mutable received : int;
    mutable receives_done : int;
    mutable psn : int; (* the first packet of the next send *)
  }

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

  let modify nic qpn ~state ~mask ?(network = 0) fill =
    let module M = D.Cmdq_modify_qp in
    ignore
      (rcfw nic.hw nic.rc "modify_qp" ~opcode:D.cmdq_base_opcode_modify_qp
         ~size:M.sizeof (fun b ->
           set b M.qp_cid qpn;
           set b M.modify_mask mask;
           set b M.network_type_en_sqd_async_notify_new_state (state lor network);
           fill b))

  let make nic =
    command nic "Qp.make" @@ fun () ->
    let hw = nic.hw in
    let cq () = queue hw ~stride:D.Cq_base.sizeof ~entries:ring_entries () in
    let create (q : queue) =
      let module C = D.Cmdq_create_cq in
      xid
        (rcfw hw nic.rc "create_cq" ~opcode:D.cmdq_base_opcode_create_cq
           ~size:C.sizeof (fun b ->
             set b C.cq_size q.entries;
             set b C.pbl q.pbl;
             set b C.pg_size_lvl q.level;
             set b C.cq_fco_cnq_id nic.nq_id))
    in
    let scq = cq () and rcq = cq () in
    let scq_id = create scq in
    let rcq_id = create rcq in
    let sq = queue hw ~stride:wqe_size ~aux:true ~entries:ring_entries () in
    let rq = queue hw ~stride:wqe_size ~entries:ring_entries () in
    let module Q = D.Cmdq_create_qp in
    let qpn =
      xid
        (rcfw hw nic.rc "create_qp" ~opcode:D.cmdq_base_opcode_create_qp
           ~size:Q.sizeof (fun b ->
             set b Q.type_ D.cmdq_create_qp_type_rc;
             set b Q.scq_cid scq_id;
             set b Q.rcq_cid rcq_id;
             set b Q.sq_size ring_entries;
             set b Q.sq_fwo_sq_sge 6;
             set b Q.sq_pbl sq.pbl;
             set b Q.sq_pg_size_sq_lvl sq.level;
             set b Q.rq_size ring_entries;
             set b Q.rq_fwo_rq_sge 6;
             set b Q.rq_pbl rq.pbl;
             set b Q.rq_pg_size_rq_lvl rq.level))
    in
    modify nic qpn ~state:D.cmdq_modify_qp_new_state_init ~mask:init_mask
      (fun b ->
        set b D.Cmdq_modify_qp.access access;
        set b D.Cmdq_modify_qp.pkey 0xffff);
    {
      nic;
      qpn;
      sq;
      rq;
      scq;
      rcq;
      scq_id;
      rcq_id;
      state = Initialized;
      failed = None;
      lengths = Array.make ring_entries 0;
      sent = 0;
      sends_done = 0;
      received = 0;
      receives_done = 0;
      psn = 0;
    }

  let endpoint qp = { qp = qp.qpn; gid = qp.nic.gid; mac = qp.nic.mac }

  let connect qp (e : endpoint) =
    if String.length e.gid <> 16 || String.length e.mac <> 6 then
      invalid_arg "Rig_bnxt.Qp.connect: a GID is 16 bytes, a MAC address 6";
    command qp.nic "Qp.connect" @@ fun () ->
    let module M = D.Cmdq_modify_qp in
    let nic = qp.nic in
    let network = D.cmdq_modify_qp_network_type_rocev2_ipv4 in
    if qp.state = Ready_to_send then
      invalid_argf "Rig_bnxt.Qp.connect: queue pair %d is connected" qp.qpn;
    if qp.state = Initialized then begin
      modify nic qp.qpn ~state:D.cmdq_modify_qp_new_state_rtr ~mask:rtr_mask
        ~network (fun b ->
          set b M.qp_type D.cmdq_modify_qp_qp_type_rc;
          set b M.access access;
          set b M.pkey 0xffff;
          Bytes.blit_string e.gid 0 b (fst M.dgid) 16;
          set b M.sgid_index nic.gid_index;
          set b M.hop_limit 64;
          Bytes.blit_string e.mac 0 b (fst M.dest_mac) 6;
          set b M.min_rnr_timer 1;
          set b M.path_mtu_pingpong_push_enable
            D.cmdq_modify_qp_path_mtu_mtu_4096;
          set b M.max_dest_rd_atomic 4;
          set b M.dest_qp_id e.qp);
      qp.state <- Ready_to_receive
    end;
    modify nic qp.qpn ~state:D.cmdq_modify_qp_new_state_rts ~mask:rts_mask
      ~network (fun b ->
        set b M.qp_type D.cmdq_modify_qp_qp_type_rc;
        set b M.access access;
        set b M.max_rd_atomic 1;
        set b M.rnr_retry 7;
        set b M.retry_cnt 7;
        set b M.timeout 14);
    qp.state <- Ready_to_send

  (* A ring keeps one slot free, as the kernel's driver does, so that its
     completion queue, of as many entries, has room for the entry that reports
     the queue pair's failure. *)
  let room qp = function
    | Send -> ring_entries - 1 - (qp.sent - qp.sends_done)
    | Receive -> ring_entries - 1 - (qp.received - qp.receives_done)

  (* Writes the SGE of the entry in slot [slot] of [q]: the [n] bytes at [at] of
     [r]. The caller writes the entry's header. *)
  let entry (q : queue) slot (r : Region.t) at n =
    let sge = (slot * q.stride) + D.Sq_send_hdr.sizeof in
    Window.set64 q.ring (sge + fst D.Sq_sge.va_or_pa) (Int64.of_int (r.va + at));
    Window.set32 q.ring (sge + fst D.Sq_sge.l_key) r.key;
    Window.set32 q.ring (sge + fst D.Sq_sge.size) n

  let check qp what side (r : Region.t) at n =
    live qp.nic what;
    if r.nic != qp.nic then
      invalid_argf "Rig_bnxt.%s: a region of another NIC" what;
    if Option.is_some r.gone then
      invalid_argf "Rig_bnxt.%s: a deregistered region" what;
    if qp.state <> Ready_to_send then
      invalid_argf "Rig_bnxt.%s: queue pair %d is not connected" what qp.qpn;
    if at < 0 || n < 0 || n > max_message || at > r.bytes - n then
      invalid_argf "Rig_bnxt.%s: %d bytes at %d of a region of %d" what n at
        r.bytes;
    if room qp side = 0 then invalid_argf "Rig_bnxt.%s: the ring is full" what

  (* The packets of a send of [n] bytes: one per [mtu] bytes, at least one. *)
  let packets n = Int.max 1 (ceildiv n mtu)

  (* A post on a failed queue pair writes nothing: its completion is the queue
     pair's failure. *)
  let send qp r ~at n =
    check qp "Qp.send" Send r at n;
    let q = qp.sq and i = qp.sent in
    let s = i mod ring_entries in
    qp.lengths.(s) <- n;
    if Option.is_none qp.failed then begin
      let base = s * q.stride in
      Window.fill q.ring base (D.Sq_send_hdr.sizeof + D.Sq_sge.sizeof) '\000';
      Window.set8 q.ring
        (base + fst D.Sq_send_hdr.wqe_type)
        D.sq_base_wqe_type_send;
      Window.set8 q.ring
        (base + fst D.Sq_send_hdr.flags)
        D.sq_send_flags_signal_comp;
      Window.set8 q.ring (base + fst D.Sq_send_hdr.wqe_size) 3;
      Window.set32 q.ring (base + fst D.Sq_send_hdr.length) n;
      entry q s r at n;
      (* The retransmission table's entry of the send: its slot, the packet
         after it and its first packet. *)
      let psn = qp.psn land 0xffffff in
      let next = (psn + packets n) land 0xffffff in
      Window.set64 q.ring
        ((ring_entries * q.stride) + (s * 8))
        (Int64.of_int
           ((s lsl D.sq_msn_search_start_idx_sft)
           lor (next lsl D.sq_msn_search_next_psn_sft)
           lor psn));
      qp.psn <- next;
      ring qp.nic.rc.db ~xid:qp.qpn ~typ:db_sq (i + 1)
    end;
    qp.sent <- i + 1

  let receive qp r ~at n =
    check qp "Qp.receive" Receive r at n;
    let q = qp.rq and i = qp.received in
    if Option.is_none qp.failed then begin
      let s = i mod ring_entries in
      let base = s * q.stride in
      Window.fill q.ring base (D.Rq_wqe_hdr.sizeof + D.Sq_sge.sizeof) '\000';
      Window.set8 q.ring
        (base + fst D.Rq_wqe_hdr.wqe_type)
        D.rq_wqe_wqe_type_rcv;
      Window.set8 q.ring (base + fst D.Rq_wqe_hdr.wqe_size) 3;
      entry q s r at n;
      ring qp.nic.rc.db ~xid:qp.qpn ~typ:db_rq (i + 1)
    end;
    qp.received <- i + 1

  let send_statuses =
    D.
      [
        (cq_req_status_bad_response_err, "the peer's response was malformed");
        (cq_req_status_local_length_err, "its length exceeds the region");
        (cq_req_status_local_qp_operation_err, "the queue pair failed");
        ( cq_req_status_local_protection_err,
          "it names a byte outside its region" );
        (cq_req_status_memory_mgt_operation_err, "a region operation failed");
        ( cq_req_status_remote_invalid_request_err,
          "the peer refused it, such as for a receive too short for it" );
        (cq_req_status_remote_access_err, "the peer refused access");
        (cq_req_status_remote_operation_err, "the peer failed to complete it");
        ( cq_req_status_rnr_nak_retry_cnt_err,
          "the peer posted no receive in time" );
        ( cq_req_status_transport_retry_cnt_err,
          "the peer did not acknowledge it after 7 retries" );
        ( cq_req_status_work_request_flushed_err,
          "the queue pair failed before it ran" );
      ]

  let receive_statuses =
    D.
      [
        (cq_res_rc_status_local_access_error, "its region refused the bytes");
        (cq_res_rc_status_local_length_err, "the message is longer than it");
        ( cq_res_rc_status_local_protection_err,
          "it names a byte outside its region" );
        (cq_res_rc_status_local_qp_operation_err, "the queue pair failed");
        (cq_res_rc_status_memory_mgt_operation_err, "a region operation failed");
        ( cq_res_rc_status_remote_invalid_request_err,
          "the peer sent an invalid request" );
        ( cq_res_rc_status_work_request_flushed_err,
          "the queue pair failed before it ran" );
        (cq_res_rc_status_hw_flush_err, "the NIC flushed it");
      ]

  let what = function Send -> "send" | Receive -> "receive"

  (* Marks [qp] failed, for the reason the completion of [side] gives, and is
     that reason. *)
  let fail_with qp side why =
    let why =
      strf "%s: queue pair %d: a %s failed: %s" qp.nic.name qp.qpn (what side)
        why
    in
    qp.failed <- Some why;
    Error why

  let cq qp = function Send -> qp.scq | Receive -> qp.rcq
  let next qp = function Send -> qp.sends_done | Receive -> qp.receives_done

  let complete qp side n =
    match side with
    | Send -> qp.sends_done <- n + 1
    | Receive -> qp.receives_done <- n + 1

  (* Whether the completion [n] of [q] is written: its toggle bit differs from
     the parity of the pass over the queue that reads it. *)
  let written (q : queue) n =
    Window.get8 q.ring (slot q n + fst D.Cq_base.cqe_type_toggle)
    land D.cq_base_toggle
    <> n / q.entries land 1

  (* Takes the written completion [n] of [side], acknowledges it, and is the
     bytes its entry moved. An entry that failed, or a completion of another
     type, such as the one that reports the queue pair's failure, fails the
     queue pair. So does a machine that failed, whose reads are all ones. *)
  let take qp side n =
    let q = cq qp side in
    Window.barrier ();
    let at = slot q n in
    let kind =
      Window.get8 q.ring (at + fst D.Cq_base.cqe_type_toggle)
      land D.cq_base_cqe_type_mask
    in
    let status = Window.get8 q.ring (at + fst D.Cq_base.status) in
    let bytes =
      match side with
      | Send -> qp.lengths.(n mod ring_entries)
      | Receive -> Window.get32 q.ring (at + fst D.Cq_res_rc.length)
    in
    complete qp side n;
    let id = match side with Send -> qp.scq_id | Receive -> qp.rcq_id in
    ring qp.nic.rc.db ~xid:id ~typ:db_cq (n + 1);
    let expected =
      match side with
      | Send -> D.cq_base_cqe_type_req
      | Receive -> D.cq_base_cqe_type_res_rc
    in
    match Machine.failed (Function.machine qp.nic.hw.fn) with
    | Some why -> fail_with qp side why
    | None when kind = D.cq_base_cqe_type_terminal ->
        fail_with qp side "the queue pair failed"
    | None when kind <> expected ->
        fail_with qp side (strf "a completion of type %d" (kind lsr 1))
    | None when status <> 0 ->
        let statuses =
          match side with Send -> send_statuses | Receive -> receive_statuses
        in
        fail_with qp side
          (match List.assoc_opt status statuses with
          | Some why -> why
          | None -> strf "status %d" status)
    | None -> Ok bytes

  (* Completes the entry [n] of [side] of the failed queue pair. *)
  let flush qp side n why =
    complete qp side n;
    Error why

  (* Why [qp] failed, if it has or its machine has. *)
  let failure qp side =
    match qp.failed with
    | Some _ as failed -> failed
    | None -> (
        match Machine.failed (Function.machine qp.nic.hw.fn) with
        | None -> None
        | Some why ->
            ignore (fail_with qp side why);
            qp.failed)

  let poll qp side =
    live qp.nic "Qp.poll";
    let n = next qp side in
    if room qp side = ring_entries - 1 then None
    else if written (cq qp side) n then Some (take qp side n)
    else
      match failure qp side with
      | Some why -> Some (flush qp side n why)
      | None -> None

  let wait qp side ~us =
    if us < 0 then invalid_argf "Rig_bnxt.Qp.wait: %d us" us;
    live qp.nic "Qp.wait";
    if room qp side = ring_entries - 1 then
      invalid_arg "Rig_bnxt.Qp.wait: no entry is due";
    let n = next qp side and q = cq qp side in
    let ended () = written q n || Option.is_some qp.failed in
    let what = strf "%s: queue pair %d's %s" qp.nic.name qp.qpn (what side) in
    match Function.wait qp.nic.hw.fn ~us what ended with
    | Error why -> Error why
    | Ok () when written q n -> take qp side n
    | Ok () -> flush qp side n (Option.get qp.failed)
end
