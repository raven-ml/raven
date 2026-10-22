(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The boot follows Linux v6.18's drivers/net/ethernet/broadcom/bnxt (the
   firmware's requests and context memory) and drivers/infiniband/hw/bnxt_re
   (the RoCE engine's command queue, queue pairs, doorbells and the handling of
   a failed queue pair), as tinygrad's runtime/support/rdma/bnxtdev.py does it
   without a kernel driver. *)

module D = Defs
module Machine = Rig_pci.Machine
module Function = Rig_pci.Function
module Window = Rig_pci.Window
module Driver = Rig_pci.Driver

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* A request the firmware or the system refuses, caught at the library's
   boundary as an [Error]. *)
exception Refused of string

let fail fmt = Printf.ksprintf (fun s -> raise (Refused s)) fmt
let guard f = match f () with v -> Ok v | exception Refused why -> Error why

(* Values of Linux's include/linux/bnxt/hsi.h and bnxt/bnxt_hwrm.h, which are
   under the GPL alone and which no header under another licence defines,
   written here by hand: the doorbell type that arms a notification queue, the
   path of a RoCE doorbell, and the target and completion ring of a request
   answered in memory alone. *)
let db_nq_arm = 0xb
let db_path_roce = 0
let hwrm_target = 0xffff
let no_ring = 0xffff

(* The work queues: 128-byte entries, rings and completion queues of 4096, and
   4096-byte packets. *)
let wqe_size = 128
let ring_entries = 4096
let mtu = 4096

(* The largest message: InfiniBand's, which the 32-bit length of an SGE
   holds, and BNXT_RE_MAX_MSG_SIZE, which Linux's bnxt_re refuses a work
   request past. *)
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

(* The most queue pairs, shared receive queues, regions and address vectors
   the context memory holds beyond the firmware's own where the firmware
   states no limits of its own: bnxt_alloc_ctx_mem's in Linux's
   drivers/net/ethernet/broadcom/bnxt/bnxt.c (v6.18). *)
let most_qps = 65536
let most_srqs = 8192
let most_regions = 256 * 1024
let most_avs = 128 * 1024

(* A terminal completion's consumer index when it names no send. *)
let no_index = 0xffff
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

(* Locks

   A NIC's lock counts among the locks its thread holds (rig_hold in
   rig_edge.h), from before its take to after its give, so that a call from a
   signal handler, finaliser or Gc.Memprof callback inside it, of rig's or of
   this library's, is refused as it starts and never waits for the lock its own
   thread holds. It nests inside rig.pci's driver and hold locks ([Driver]),
   whose stop takes it, and holds rig.pci's function and memory locks. It is held
   across a request to the firmware, which waits for its answer: the
   firmware's channel takes one request at a time. *)

external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]

external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]
external reraise : exn -> 'a = "%reraise"

(* [take m] takes [m], counted from before the take, so that a signal handler
   that runs inside a contended take sees it held. An exception that ends the
   take gives the count back first, having taken nothing: one a handler
   raised, or the error-checking [Mutex]'s [Sys_error] for a take by the
   thread that holds [m], which only a call that skipped its check makes and
   which raises [Invalid_argument]. *)
let take m =
  hold ();
  match Mutex.lock m with
  | () -> ()
  | exception e -> (
      unhold ();
      match e with
      | Sys_error why when String.starts_with ~prefix:"Mutex.lock:" why ->
          invalid_arg
            "Rig_bnxt: a lock taken again by the thread that holds it, from a \
             signal handler, finaliser or Gc.Memprof callback"
      | e -> reraise e)

(* Gives [m] back, then its count, with nothing between that allocates or
   polls. *)
let give m =
  Mutex.unlock m;
  unhold ()

(* [protect m f] is [f ()] holding [m], taken as the last step before the
   handler and given back as the first step of both exits, where nothing
   allocates or polls, so an exception anywhere in [f], an asynchronous one
   too, leaves [m] free and the count right. *)
let protect m f =
  take m;
  match f () with
  | x ->
      give m;
      x
  | exception e ->
      give m;
      reraise e

(* Raises [Invalid_argument] naming [fn] while the thread holds a lock: the
   call comes from a signal handler, finaliser or Gc.Memprof callback inside
   that lock, and would wait for it or see its section half done. A public
   call that may take a lock starts with it. *)
let check fn =
  if holds () <> 0 then
    invalid_arg
      ("Rig_bnxt." ^ fn
     ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
        inside a lock of the calling thread's")

(* The firmware's channel *)

(* [lock] serializes the requests and commands and guards [allocs] and
   [pins]. Every allocation is freed, and every pin given back, at the NIC's
   stop, once it masters the bus no more. *)
type hw = {
  fn : Function.t;
  bar0 : Window.t;
  dbs : Window.t; (* the doorbell BAR *)
  resp : Window.t; (* the firmware's answers *)
  resp_bus : int;
  lock : Mutex.t;
  mutable seq : int;
  mutable allocs : Window.t list;
  mutable pins : (int * int) list; (* the memory [Region.pages] pinned *)
  mutable registered : bool; (* with the firmware, which stop undoes *)
  mutable relaxed_at : int option;
      (* the Device Control register, whose relaxed ordering open turned off
         and stop turns back on *)
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

(* Gives back one pin of [hw]'s, which another pin of the same pages may
   leave in [pins]. *)
let unpin hw ((a, n) as pin) =
  let rec drop = function
    | [] -> []
    | p :: rest -> if p = pin then rest else p :: drop rest
  in
  hw.pins <- drop hw.pins;
  Function.unpin hw.fn a n

(* [f ()], with [w] given back if [f] is refused. *)
let freeing hw w f =
  match f () with
  | v -> v
  | exception (Refused _ as e) ->
      free hw w;
      raise e

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
      freeing hw table @@ fun () ->
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
   firmware has written it whole: the answer page, zeroed past the answer, so
   that a field a short answer lacks reads as zero. *)
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
  (* The answer's last byte is its valid flag, written last. *)
  let answered () =
    let h = Window.read hw.resp 0 R.sizeof in
    let n = get h R.resp_len in
    n > 0
    && n <= Window.length hw.resp
    && get h R.seq_id = hw.seq
    && Window.get8 hw.resp (n - 1) <> 0
  in
  wait hw ~us ("HWRM " ^ name) answered;
  let out = Window.read hw.resp 0 (Window.length hw.resp) in
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
    xid land D.bnxt_re_db_qid_mask lor db_path_roce
    lor (typ lsl D.bnxt_re_db_typ_shift)
    lor (1 lsl D.bnxt_re_db_valid_shift)
  in
  Int64.logor
    (Int64.shift_left (Int64.of_int high) 32)
    (Int64.of_int
       (index land D.bnxt_re_db_indx_mask
       lor (epoch lsl D.bnxt_re_db_epoch_shift)))

let doorbell db xid typ index epoch =
  Window.barrier ();
  Window.set64 db 0 (db_value xid typ index epoch)

(* Rings [xid]'s doorbell of type [typ] for the entries before [n] of a ring of
   [ring_entries]. *)
let ring db ~xid ~typ n =
  doorbell db xid typ (n mod ring_entries) (n / ring_entries land 1)

(* Queues *)

(* A ring of [entries] of [stride] bytes in system memory, its page list, and
   the memory both take. [aux] follows the ring with 8 bytes an entry. *)
type queue = {
  ring : Window.t;
  stride : int;
  entries : int;
  level : int;
  pbl : int;
  memory : Window.t list;
}

let queue hw ?(stride = 16) ?(aux = false) ?(entries = page / 16) () =
  let size = entries * stride in
  let bytes = size + if aux then entries * 8 else 0 in
  let ring, runs = sysmem hw bytes in
  let level, pbl, list =
    freeing hw ring (fun () -> page_list hw ~queue:true (pages ~bytes runs))
  in
  { ring; stride; entries; level; pbl; memory = ring :: list }

let slot q i = i mod q.entries * q.stride

(* The fields of a ring's request that place [q]. *)
let ring_fields b (q : queue) =
  let module R = D.Hwrm_ring_alloc_input in
  set b R.page_tbl_addr q.pbl;
  set b R.page_size 12;
  set b R.page_tbl_depth q.level

(* The RoCE engine's commands *)

(* [stopped] once a command went unanswered: the channel's state is then
   unknown, so no command is sent on it again, and the memory such a command
   named stays until the NIC stops. *)
type rcfw = {
  cmdq : queue;
  creq : queue;
  creq_id : int;
  db : Window.t; (* the NIC's doorbell *)
  mutable prod : int; (* the command slots written *)
  mutable cons : int; (* the completions read *)
  mutable first : bool;
  mutable stopped : bool;
}

(* Writes the command [opcode] of [size] bytes, filled by [fill], into the
   command queue, rings it, and is its completion. Its cookie, the low bits of
   its first slot's index, tells its completion from those of earlier commands
   and of asynchronous events, such as a queue pair's error, which are
   skipped. *)
let rcfw hw rc ?(us = rcfw_us) name ~opcode ~size fill =
  if rc.stopped then
    fail "RCFW %s: the firmware stopped answering; close the NIC and open it"
      name;
  let slots = ceildiv size 16 in
  let cookie = rc.prod land 0xffff in
  let b = Bytes.make (slots * 16) '\000' in
  set b D.Cmdq_base.opcode opcode;
  set b D.Cmdq_base.cmd_size slots;
  set b D.Cmdq_base.cookie cookie;
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
  let ready () =
    Window.get8 creq.ring (slot creq rc.cons + fst D.Creq_base.v)
    land D.creq_base_v
    <> rc.cons / creq.entries land 1
  in
  let rec next () =
    (match Function.wait hw.fn ~us ("RCFW " ^ name) ready with
    | Ok () -> ()
    | Error why ->
        rc.stopped <- true;
        fail "%s" why);
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
      && get r D.Creq_create_qp_resp.cookie = cookie
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

(* What the firmware states of a context type that the NIC keeps in host
   memory: an entry's bytes, the range and the multiple of its entries, its
   splits, its instances, and the byte each entry of an instance starts with
   and its offset in bytes, where the type asks for one. *)
type context = {
  typ : int;
  size : int;
  least : int;
  most : int;
  multiple : int;
  splits : int list;
  instances : int list;
  init : (int * int) option;
}

let context hw typ =
  let module Q = D.Hwrm_func_backing_store_qcaps_v2_output in
  let caps =
    hwrm hw "func_backing_store_qcaps_v2"
      ~req_type:D.hwrm_func_backing_store_qcaps_v2 (fun b ->
        set b D.Hwrm_func_backing_store_qcaps_v2_input.type_ typ)
  in
  let splits =
    [ Q.split_entry_0; Q.split_entry_1; Q.split_entry_2; Q.split_entry_3 ]
    |> List.filteri (fun i _ -> i < get caps Q.subtype_valid_cnt)
    |> List.map (get caps)
  in
  let map = get caps Q.instance_bit_map in
  let instances =
    match List.filter (fun i -> (map lsr i) land 1 = 1) (List.init 32 Fun.id)
    with
    | [] -> [ 0 ]
    | l -> l
  in
  let init =
    if
      get caps Q.flags
      land D.hwrm_func_backing_store_qcaps_v2_output_flags_enable_ctx_kind_init
      <> 0
    then Some (get caps Q.ctx_init_value, 4 * get caps Q.ctx_init_offset)
    else None
  in
  {
    typ;
    size = get caps Q.entry_size;
    least = get caps Q.min_num_entries;
    most = get caps Q.max_num_entries;
    multiple = get caps Q.entry_multiple;
    splits;
    instances;
    init;
  }

(* The [i]th split of [c], 0 where the firmware states none. *)
let split c i = Option.value (List.nth_opt c.splits i) ~default:0

(* Gives the firmware the memory of [entries] entries of [c] in each of its
   instances, with [splits]: [entries] rounded up to [c]'s multiple and held
   to its range, as bnxt_setup_ctxm_pg_tbls does. An instance of no entries
   gets none. [last]'s last instance closes the configuration. *)
let configure hw ?(last = false) c ~splits entries =
  let module C = D.Hwrm_func_backing_store_cfg_v2_input in
  let csplits =
    [ C.split_entry_0; C.split_entry_1; C.split_entry_2; C.split_entry_3 ]
  in
  if c.size = 0 then
    fail "the firmware states no entry size for context type %d" c.typ;
  let n =
    if c.multiple > 0 then ceildiv entries c.multiple * c.multiple
    else entries
  in
  let n = Int.min c.most (Int.max c.least n) in
  let final = List.length c.instances - 1 in
  let instance j i =
    let bytes = ceildiv (n * c.size) page * page in
    let mem, runs = sysmem hw bytes in
    Option.iter
      (fun (v, at) ->
        let off = ref at in
        while !off < bytes do
          Window.set8 mem !off v;
          off := !off + c.size
        done)
      c.init;
    let level, dir, _ = page_list hw (pages ~bytes runs) in
    ignore
      (hwrm hw "func_backing_store_cfg_v2"
         ~req_type:D.hwrm_func_backing_store_cfg_v2 (fun b ->
           set b C.type_ c.typ;
           set b C.instance i;
           set b C.entry_size c.size;
           set b C.num_entries n;
           set b C.page_dir dir;
           set b C.page_size_pbl_level level;
           set b C.subtype_valid_cnt (List.length splits);
           if last && j = final then
             set b C.flags
               D.hwrm_func_backing_store_cfg_v2_input_flags_bs_cfg_all_done;
           List.iteri (fun k v -> set b (List.nth csplits k) v) splits))
  in
  if n > 0 then List.iteri instance c.instances

(* The context memory the RoCE engine keeps its queue pairs, regions and
   completion queues in, sized as bnxt_alloc_ctx_mem sizes it for a function
   with RoCE. The firmware keeps queue pairs of its own for L2 and QP1 (the
   first two splits of the queue pairs' type). Beyond them the memory holds
   every queue pair and shared receive queue the firmware holds, where it
   states that it enforces limits of its own ([limits]), and otherwise
   [most_qps] queue pairs, plus those it destroys fast (the third split), and
   [most_srqs] shared receive queues. Each queue pair gets two completion
   queues, a TIM entry and two entries of each TQM ring. Half the MRAV
   entries, at most [most_regions], hold regions, and as many, at most
   [most_avs], address vectors, whose count is the type's one split, unless
   the firmware states both. Every type is asked for first, then each is
   configured in the order of its number, the TIM type last. *)
let backing hw ~limits =
  let context = context hw in
  let qp = context D.hwrm_func_backing_store_qcaps_v2_input_type_qp in
  let srq = context D.hwrm_func_backing_store_qcaps_v2_input_type_srq in
  let cq = context D.hwrm_func_backing_store_qcaps_v2_input_type_cq in
  let vnic = context D.hwrm_func_backing_store_qcaps_v2_input_type_vnic in
  let stat = context D.hwrm_func_backing_store_qcaps_v2_input_type_stat in
  let stqm =
    context D.hwrm_func_backing_store_qcaps_v2_input_type_sp_tqm_ring
  in
  let ftqm =
    context D.hwrm_func_backing_store_qcaps_v2_input_type_fp_tqm_ring
  in
  let mrav = context D.hwrm_func_backing_store_qcaps_v2_input_type_mrav in
  let tim = context D.hwrm_func_backing_store_qcaps_v2_input_type_tim in
  let l2 = split qp 0 and qp1 = split qp 1 and srqs = split srq 0 in
  let extra_qps, extra_srqs =
    if limits then (qp.most - l2 - qp1, srq.most - srqs)
    else
      ( Int.min most_qps (qp.most - l2 - qp1) + split qp 2,
        Int.min most_srqs (srq.most - srqs) )
  in
  let qps = l2 + qp1 + extra_qps in
  let regions, avs =
    if limits && List.length mrav.splits = 1 then
      (mrav.most - split mrav 0, split mrav 0)
    else
      let regions = Int.min (mrav.most / 2) most_regions in
      (regions, Int.min regions most_avs)
  in
  let av = match split mrav 0 with 0 -> avs | n -> Int.min n avs in
  let tqm = l2 + (2 * (extra_qps + qp1)) in
  let own c = configure hw c ~splits:c.splits in
  own qp qps;
  own srq (srqs + extra_srqs);
  own cq (split cq 0 + (2 * extra_qps));
  own vnic vnic.most;
  own stat stat.most;
  own stqm (split vnic 0 + tqm + stqm.least);
  own ftqm tqm;
  configure hw mrav ~splits:[ av ] (regions + avs);
  configure hw ~last:true tim ~splits:tim.splits qps

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
  let rc =
    {
      cmdq;
      creq;
      creq_id;
      db;
      prod = 0;
      cons = 0;
      first = true;
      stopped = false;
    }
  in
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
  hold : Driver.hold;
  hw : hw;
  rc : rcfw;
  nq_id : int;
  mac : string;
  gid : string;
  gid_index : int;
  random : Random.State.t; (* the queue pairs' first packets, under the lock *)
  mutable closed : bool;
}

let name nic = nic.name
let bus nic = Function.bus nic.hw.fn

(* Whether [mac] and [gid] are [nic]'s addresses on the fabric. *)
let is_at nic ~mac ~gid = String.equal mac nic.mac && String.equal gid nic.gid

let live nic what =
  if nic.closed then invalid_argf "Rig_bnxt.%s: %s is closed" what nic.name

(* [f ()] with the NIC's channel to its firmware held, a refusal as an
   [Error]. *)
let command nic what f =
  check what;
  live nic what;
  protect nic.hw.lock (fun () -> guard f)

let is_bnxt (id : Machine.id) =
  id.vendor = 0x14e4 && id.class_ lsr 16 = 0x02 && id.device = 0x1760

(* The kernel driver serves the NIC through network interfaces, which hold no
   file, and its RoCE driver through files with a [dev] file under the
   function's directory, which {!Driver.detach} finds itself. *)
let nics =
  Driver.make ~name:"BNXT" ~memory_bar:None ~siblings:Keep
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:10_000 ~reset:Function.reset
    ~ready:(fun _ -> Ok ())
    is_bnxt

let buses ?(machine = Machine.this) () = Driver.buses nics machine
let count ?machine () = Result.map List.length (buses ?machine ())

let map fn i =
  match Function.map ~combine:false fn i with
  | Ok w -> w
  | Error why -> fail "%s" why

(* The NIC's registers and the page its firmware answers in, before anything is
   written to it. *)
let channel fn =
  let bar0 = map fn 0 and dbs = map fn doorbell_bar in
  let resp, runs = alloc fn page in
  {
    fn;
    bar0;
    dbs;
    resp;
    resp_bus = fst (List.hd runs);
    lock = Mutex.create ();
    seq = 0;
    allocs = [ resp ];
    pins = [];
    registered = false;
    relaxed_at = None;
  }

(* PCI configuration space: the capability list, the PCI Express capability's
   ID, and its Device Control register, whose Enable Relaxed Ordering bit lets
   the function set relaxed ordering on the requests it starts (PCI Express
   Base Specification 7.5.3.4, as Linux's include/uapi/linux/pci_regs.h
   names them: PCI_CAPABILITY_LIST, PCI_CAP_ID_EXP, PCI_EXP_DEVCTL and
   PCI_EXP_DEVCTL_RELAX_EN). *)
let capabilities = 0x34
let pcie_capability = 0x10
let device_control = 0x08
let relaxed_ordering = 0x10

(* The offset of [fn]'s Device Control register. The walk of the capabilities
   is bounded: a dead link can give back pointers for ever. *)
let device_control_at fn =
  let rec walk cap seen =
    if cap = 0 || List.mem cap seen then None
    else if Function.config8 fn cap = pcie_capability then Some cap
    else walk (Function.config8 fn (cap + 1) land 0xfc) (cap :: seen)
  in
  match walk (Function.config8 fn capabilities land 0xfc) [] with
  | None -> fail "the NIC has no PCI Express capability"
  | Some cap -> cap + device_control

let relaxed fn at = Function.config16 fn at land relaxed_ordering <> 0

(* Turns relaxed ordering off for every request the NIC starts, the writes of
   its completions and its firmware's included, as Linux's drivers/pci/probe.c
   (pci_configure_relaxed_ordering) does for a device under a root port that
   mishandles it; vfio-pci passes the bit through. A posted write that is not
   relaxed passes no earlier one, so a receive's or a read's bytes reach
   memory before its completion. *)
let strict_order fn at =
  Function.set_config16 fn at
    (Function.config16 fn at land lnot relaxed_ordering);
  if relaxed fn at then
    fail "the NIC keeps relaxed ordering on after it was turned off"

(* Gives the NIC back the relaxed ordering that open found, for the kernel's
   drivers that take it next. *)
let restore_order fn at =
  Function.set_config16 fn at (Function.config16 fn at lor relaxed_ordering)

(* The NIC's stop: a NIC registered with its firmware, whose machine still
   answers, unregisters. The NIC then masters the bus no more, so its memory is
   freed and its pins given back, and gets back the relaxed ordering open
   found. A NIC stopped otherwise is reset at its next open, as every open
   does. *)
let stop hw () =
  protect hw.lock @@ fun () ->
  let clean =
    hw.registered
    && Function.failed hw.fn = None
    &&
    match hwrm hw "func_drv_unrgtr" ~req_type:D.hwrm_func_drv_unrgtr ignore with
    | _ -> true
    | exception Refused _ -> false
  in
  Function.set_bus_master hw.fn false;
  List.iter (Function.free_dma hw.fn) hw.allocs;
  hw.allocs <- [];
  List.iter (fun (a, n) -> Function.unpin hw.fn a n) hw.pins;
  hw.pins <- [];
  Option.iter (restore_order hw.fn) hw.relaxed_at;
  hw.relaxed_at <- None;
  if clean then `Clean else `Lost

(* Resets the NIC, turns its relaxed ordering off, registers with its
   firmware, gives the RoCE engine its context memory and command queue, opens
   the Ethernet receive path, and adds the NIC's address. *)
let boot ~name h hw =
  let at = device_control_at hw.fn in
  let was = relaxed hw.fn at in
  (match Function.reset hw.fn with Ok () -> () | Error why -> fail "%s" why);
  if was then hw.relaxed_at <- Some at;
  strict_order hw.fn at;
  Function.set_bus_master hw.fn true;
  ignore (hwrm hw "ver_get" ~req_type:D.hwrm_ver_get ignore);
  ignore (hwrm hw ~us:reset_us "func_reset" ~req_type:D.hwrm_func_reset ignore);
  let caps =
    hwrm hw "func_qcaps" ~req_type:D.hwrm_func_qcaps (fun b ->
        set b D.Hwrm_func_qcaps_input.fid 0xffff)
  in
  let mac = String.sub caps (fst D.Hwrm_func_qcaps_output.mac_address) 6 in
  let limits =
    get caps D.Hwrm_func_qcaps_output.flags_ext2
    land D.hwrm_func_qcaps_output_flags_ext2_sw_max_resource_limits_supported
    <> 0
  in
  ignore (hwrm hw "func_drv_rgtr" ~req_type:D.hwrm_func_drv_rgtr ignore);
  hw.registered <- true;
  let cfg =
    hwrm hw "func_qcfg" ~req_type:D.hwrm_func_qcfg (fun b ->
        set b D.Hwrm_func_qcfg_input.fid 0xffff)
  in
  let db_off = get cfg D.Hwrm_func_qcfg_output.legacy_l2_db_size_kb * 1024 in
  if db_off + 8 > Window.length hw.dbs then
    fail "the firmware puts the doorbell at %#x, past BAR %d" db_off
      doorbell_bar;
  let db = Window.sub hw.dbs db_off 8 in
  backing hw ~limits;
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
  {
    name;
    hold = h;
    hw;
    rc;
    nq_id;
    mac;
    gid;
    gid_index;
    random = Random.State.make_self_init ();
    closed = false;
  }

let index what i =
  if i < 0 then invalid_argf "Rig_bnxt.%s: NIC %d is negative" what i

let open_ ?(machine = Machine.this) i =
  check "open_";
  index "open_" i;
  let name = Driver.name nics i in
  Driver.open_ nics machine i (fun h fn ->
      guard @@ fun () ->
      let hw = channel fn in
      Driver.set_stop h (stop hw);
      boot ~name h hw)

let close nic =
  check "close";
  if not nic.closed then begin
    nic.closed <- true;
    ignore (Driver.stop nic.hold)
  end

let detach i =
  check "detach";
  index "detach" i;
  Driver.detach nics Machine.this i

let attach i =
  check "attach";
  index "attach" i;
  Driver.attach nics Machine.this i

(* Regions *)

module Region = struct
  type nonrec t = {
    nic : t;
    key : int; (* its local key, and its remote key *)
    va : int;
    bytes : int;
    host : Window.t option; (* the memory {!host} made *)
    memory : Window.t list; (* freed at deregistration *)
    pinned : (int * int) option; (* the pages {!pages} pinned *)
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

  (* The region over the pages of [runs], under which the NIC writes [va, va +
     bytes), [va] being the first run's address. [host] is memory it was made
     over, which it frees at its deregistration. A refused registration frees
     its memory, unless the firmware stopped answering and may still read the
     page list.

     Every region allows remote reads, which only a queue pair connected to
     one of its own NIC serves ([Qp.connect]): its key, which serves as its
     remote key as in Linux's bnxt_re, never leaves the process. The NIC
     writes it in order ([strict_order]). *)
  let register nic runs ~bytes ~host ~pinned =
    let va = fst (List.hd runs) in
    let log = log_page ~va runs in
    let level, pbl, list = page_list nic.hw (pages ~log ~bytes runs) in
    let memory = list @ Option.to_list host in
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
            lor D.cmdq_register_mr_access_remote_read);
          set b M.log2_pbl_pg_size 12;
          set b M.pbl pbl;
          set b M.va va;
          set b M.mr_size bytes)
    with
    | r -> { nic; key = xid r; va; bytes; host; memory; pinned; gone = None }
    | exception e ->
        if not nic.rc.stopped then List.iter (free nic.hw) memory;
        raise e

  let host nic n =
    if n <= 0 then invalid_argf "Rig_bnxt.Region.host: %d bytes" n;
    command nic "Region.host" @@ fun () ->
    let w, runs = sysmem nic.hw n in
    register nic runs ~bytes:(Window.length w) ~host:(Some w) ~pinned:None

  (* The pin is given back if the registration is refused, unless the firmware
     stopped answering and may still read the page list, and at deregistration
     otherwise, or at the NIC's stop. *)
  let pages nic a n =
    let page = Machine.page (Function.machine nic.hw.fn) in
    if n <= 0 then invalid_argf "Rig_bnxt.Region.pages: %d bytes" n;
    if a < 0 || a land (page - 1) <> 0 then
      invalid_argf "Rig_bnxt.Region.pages: %#x is not on a page" a;
    command nic "Region.pages" @@ fun () ->
    let runs =
      match Function.pin nic.hw.fn a n with
      | Ok (Some runs) -> runs
      | Ok None -> fail "no room to pin %d bytes at %#x for the NIC" n a
      | Error why -> fail "%s" why
    in
    nic.hw.pins <- (a, n) :: nic.hw.pins;
    match register nic runs ~bytes:n ~host:None ~pinned:(Some (a, n)) with
    | r -> r
    | exception (Refused _ as e) ->
        if not nic.rc.stopped then unpin nic.hw (a, n);
        raise e

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
    register nic runs ~bytes ~host:None ~pinned:None

  let length r = r.bytes

  (* The memory {!host} made of [r], whose [n] bytes at [at] it holds. *)
  let memory r what at n =
    live r.nic what;
    if Option.is_some r.gone then
      invalid_argf "Rig_bnxt.%s: a deregistered region" what;
    if at < 0 || n < 0 || at > r.bytes - n then
      invalid_argf "Rig_bnxt.%s: %d bytes at %d of a region of %d" what n at
        r.bytes;
    match r.host with
    | Some w -> w
    | None ->
        invalid_argf "Rig_bnxt.%s: a region over another device's memory" what

  let read r at n = Window.read (memory r "Region.read" at n) at n

  let write r at s =
    Window.write (memory r "Region.write" at (String.length s)) at s

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
          List.iter (free r.nic.hw) r.memory;
          Option.iter (unpin r.nic.hw) r.pinned
        in
        r.gone <- Some result;
        result
end

(* Queue pairs *)

module Qp = struct
  type endpoint = { qp : int; psn : int; gid : string; mac : string }
  type side = Send | Receive

  (* A ring of a queue pair, its completion queue, and how far each has gone.
     [cq_read] counts the completions read, which a terminal completion puts
     ahead of the entries completed. Once the queue pair fails, the
     completions written before [cq_until] are read, then the entries before
     [ok_until] complete without error, and the others with the failure.
     Only the ring's own side, its poll and wait, writes the mutable fields
     but [posted], which its posts write. *)
  type ring = {
    entries : queue;
    cq : queue;
    cq_id : int;
    typ : int; (* its doorbell's *)
    kind : int; (* its completions' *)
    statuses : (int * string) list;
    lengths : int array; (* the bytes of each entry in the ring *)
    waiting : string; (* what a wait for it is called *)
    mutable posted : int;
    mutable taken : int;
    mutable cq_read : int;
    mutable noticed : bool; (* whether its side saw the failure *)
    mutable cq_until : int;
    mutable ok_until : int;
  }

  (* The states {!connect} takes a queue pair through, so that a connect the
     firmware refused part-way resumes where it stopped. *)
  type state = Initialized | Ready_to_receive | Ready_to_send

  type nonrec t = {
    nic : t;
    qpn : int;
    sends : ring;
    receives : ring;
    first_psn : int;
    mutable psn : int; (* the first packet of the next send *)
    mutable state : state;
    mutable peer : endpoint option; (* once ready to receive *)
    failed : string option Atomic.t;
    terminal : int Atomic.t;
        (* the send index of a terminal completion, or [no_index] *)
    mutable destroyed : (unit, string) result option;
  }

  let rings qp = function Send -> qp.sends | Receive -> qp.receives
  let access = D.cmdq_modify_qp_access_local_write

  (* A queue pair connected to one of its own NIC serves its peer's reads,
     which are this process's own: the remote keys they name never leave
     it. *)
  let own nic (e : endpoint) = is_at nic ~mac:e.mac ~gid:e.gid

  let access_to nic e =
    if own nic e then access lor D.cmdq_modify_qp_access_remote_read
    else access
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

  let destroy_cq nic id =
    ignore
      (rcfw nic.hw nic.rc "destroy_cq" ~opcode:D.cmdq_base_opcode_destroy_cq
         ~size:D.Cmdq_destroy_cq.sizeof (fun b ->
           set b D.Cmdq_destroy_cq.cq_cid id))

  let destroy_qp nic qpn =
    ignore
      (rcfw nic.hw nic.rc "destroy_qp" ~opcode:D.cmdq_base_opcode_destroy_qp
         ~size:D.Cmdq_destroy_qp.sizeof (fun b ->
           set b D.Cmdq_destroy_qp.qp_cid qpn))

  (* A refused step gives back what the steps before took: the firmware's
     objects, then, once each is gone, the rings' memory. Memory an object the
     firmware kept may name stays until the NIC closes. *)
  let make nic =
    command nic "Qp.make" @@ fun () ->
    let hw = nic.hw in
    let made = ref [] and memory = ref [] in
    let undo () =
      let gone d = match d () with () -> true | exception Refused _ -> false in
      if List.fold_left (fun all d -> gone d && all) true !made then
        List.iter (free hw) !memory
    in
    let alloc_queue ?aux ~stride () =
      let q = queue hw ?aux ~stride ~entries:ring_entries () in
      memory := q.memory @ !memory;
      q
    in
    let create (q : queue) =
      let module C = D.Cmdq_create_cq in
      let id =
        xid
          (rcfw hw nic.rc "create_cq" ~opcode:D.cmdq_base_opcode_create_cq
             ~size:C.sizeof (fun b ->
               set b C.cq_size q.entries;
               set b C.pbl q.pbl;
               set b C.pg_size_lvl q.level;
               set b C.cq_fco_cnq_id nic.nq_id))
      in
      made := (fun () -> destroy_cq nic id) :: !made;
      id
    in
    try
      let scq = alloc_queue ~stride:D.Cq_base.sizeof () in
      let rcq = alloc_queue ~stride:D.Cq_base.sizeof () in
      let scq_id = create scq in
      let rcq_id = create rcq in
      let sq = alloc_queue ~stride:wqe_size ~aux:true () in
      let rq = alloc_queue ~stride:wqe_size () in
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
      made := (fun () -> destroy_qp nic qpn) :: !made;
      modify nic qpn ~state:D.cmdq_modify_qp_new_state_init ~mask:init_mask
        (fun b ->
          set b D.Cmdq_modify_qp.access access;
          set b D.Cmdq_modify_qp.pkey 0xffff);
      let ring entries cq cq_id typ kind statuses what =
        {
          entries;
          cq;
          cq_id;
          typ;
          kind;
          statuses;
          lengths = Array.make ring_entries 0;
          waiting = strf "%s: queue pair %d's %s" nic.name qpn what;
          posted = 0;
          taken = 0;
          cq_read = 0;
          noticed = false;
          cq_until = 0;
          ok_until = 0;
        }
      in
      let first_psn = Random.State.bits nic.random land 0xffffff in
      {
        nic;
        qpn;
        sends =
          ring sq scq scq_id D.bnxt_re_que_type_sq D.cq_base_cqe_type_req
            send_statuses "send";
        receives =
          ring rq rcq rcq_id D.bnxt_re_que_type_rq D.cq_base_cqe_type_res_rc
            receive_statuses "receive";
        first_psn;
        psn = first_psn;
        state = Initialized;
        peer = None;
        failed = Atomic.make None;
        terminal = Atomic.make no_index;
        destroyed = None;
      }
    with Refused _ as e ->
      undo ();
      raise e

  let usable qp what =
    live qp.nic what;
    if Option.is_some qp.destroyed then
      invalid_argf "Rig_bnxt.%s: queue pair %d is destroyed" what qp.qpn

  let endpoint qp =
    { qp = qp.qpn; psn = qp.first_psn; gid = qp.nic.gid; mac = qp.nic.mac }

  let connect qp (e : endpoint) =
    if String.length e.gid <> 16 || String.length e.mac <> 6 then
      invalid_arg "Rig_bnxt.Qp.connect: a GID is 16 bytes, a MAC address 6";
    usable qp "Qp.connect";
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
          set b M.access (access_to nic e);
          set b M.pkey 0xffff;
          Bytes.blit_string e.gid 0 b (fst M.dgid) 16;
          set b M.sgid_index nic.gid_index;
          set b M.hop_limit 64;
          Bytes.blit_string e.mac 0 b (fst M.dest_mac) 6;
          set b M.min_rnr_timer 1;
          set b M.path_mtu_pingpong_push_enable
            D.cmdq_modify_qp_path_mtu_mtu_4096;
          set b M.rq_psn (e.psn land 0xffffff);
          set b M.max_dest_rd_atomic 4;
          set b M.dest_qp_id e.qp);
      qp.state <- Ready_to_receive;
      qp.peer <- Some e
    end;
    let peer = Option.value ~default:e qp.peer in
    modify nic qp.qpn ~state:D.cmdq_modify_qp_new_state_rts ~mask:rts_mask
      ~network (fun b ->
        set b M.qp_type D.cmdq_modify_qp_qp_type_rc;
        set b M.access (access_to nic peer);
        set b M.max_rd_atomic 1;
        set b M.rnr_retry 7;
        set b M.retry_cnt 7;
        set b M.timeout 14;
        set b M.sq_psn qp.first_psn);
    qp.state <- Ready_to_send

  (* A ring keeps one slot free, as the kernel's driver does, so that its
     completion queue, of as many entries, has room for the completion that
     reports the queue pair's failure. *)
  let room qp side =
    let r = rings qp side in
    ring_entries - 1 - (r.posted - r.taken)

  (* Writes the SGE of the entry in slot [slot] of [q]: the [n] bytes at [at] of
     [r]. The caller writes the entry's header. *)
  let sge (q : queue) slot (r : Region.t) at n =
    let sge = (slot * q.stride) + D.Sq_send_hdr.sizeof in
    Window.set64 q.ring (sge + fst D.Sq_sge.va_or_pa) (Int64.of_int (r.va + at));
    Window.set32 q.ring (sge + fst D.Sq_sge.l_key) r.key;
    Window.set32 q.ring (sge + fst D.Sq_sge.size) n

  let check_region qp what (r : Region.t) at n =
    if r.nic != qp.nic then
      invalid_argf "Rig_bnxt.%s: a region of another NIC" what;
    if Option.is_some r.gone then
      invalid_argf "Rig_bnxt.%s: a deregistered region" what;
    if at < 0 || n < 0 || n > max_message || at > r.bytes - n then
      invalid_argf "Rig_bnxt.%s: %d bytes at %d of a region of %d" what n at
        r.bytes

  let check_post qp what side (r : Region.t) at n =
    usable qp what;
    check_region qp what r at n;
    if qp.state <> Ready_to_send then
      invalid_argf "Rig_bnxt.%s: queue pair %d is not connected" what qp.qpn;
    if room qp side = 0 then invalid_argf "Rig_bnxt.%s: the ring is full" what

  (* The packets of a send of [n] bytes: one per [mtu] bytes, at least one. *)
  let packets n = Int.max 1 (ceildiv n mtu)

  (* What an entry of the send ring does: send, or read the bytes at an offset
     of a region. *)
  type op = Sending | Reading of Region.t * int

  (* Writes the entry of [op] that moves the [n] bytes at [at] of [r] in the
     next slot of the send ring, then its retransmission entry, and rings it.
     The headers of both entries are 32 bytes, the SGE after them. A post on a
     failed queue pair writes nothing: its completion is the queue pair's
     failure. *)
  let post_send qp op (r : Region.t) at n =
    let s = qp.sends in
    let q = s.entries and i = s.posted in
    let slot = i mod ring_entries in
    s.lengths.(slot) <- n;
    if Option.is_none (Atomic.get qp.failed) then begin
      let base = slot * q.stride and w = q.ring in
      Window.fill w base (D.Sq_send_hdr.sizeof + D.Sq_sge.sizeof) '\000';
      (match op with
      | Sending ->
          let module H = D.Sq_send_hdr in
          Window.set8 w (base + fst H.wqe_type) D.sq_base_wqe_type_send;
          Window.set8 w (base + fst H.flags) D.sq_send_flags_signal_comp;
          Window.set8 w (base + fst H.wqe_size) 3;
          Window.set32 w (base + fst H.length) n
      | Reading (src, from) ->
          let module H = D.Sq_rdma_hdr in
          Window.set8 w (base + fst H.wqe_type) D.sq_base_wqe_type_read_wqe;
          Window.set8 w (base + fst H.flags) D.sq_rdma_hdr_flags_signal_comp;
          Window.set8 w (base + fst H.wqe_size) 3;
          Window.set32 w (base + fst H.length) n;
          Window.set64 w
            (base + fst H.remote_va)
            (Int64.of_int (src.va + from));
          Window.set32 w (base + fst H.remote_key) src.key);
      sge q slot r at n;
      (* The retransmission table's entry of the send: its slot, the packet
         after it and its first packet. A read takes as many packets of
         responses, as Linux's bnxt_re counts them. *)
      let psn = qp.psn land 0xffffff in
      let next = (psn + packets n) land 0xffffff in
      Window.set64 q.ring
        ((ring_entries * q.stride) + (slot * 8))
        (Int64.of_int
           ((slot lsl D.sq_msn_search_start_idx_sft)
           lor (next lsl D.sq_msn_search_next_psn_sft)
           lor psn));
      qp.psn <- next;
      ring qp.nic.rc.db ~xid:qp.qpn ~typ:s.typ (i + 1)
    end;
    s.posted <- i + 1

  let send qp r ~at n =
    check_post qp "Qp.send" Send r at n;
    post_send qp Sending r at n

  let read qp ~src:((s : Region.t), from) ~dst:(r, at) n =
    check_post qp "Qp.read" Send r at n;
    check_region qp "Qp.read" s from n;
    (match qp.peer with
    | Some e when own qp.nic e -> ()
    | _ ->
        invalid_argf "Rig_bnxt.Qp.read: queue pair %d's peer is on another NIC"
          qp.qpn);
    post_send qp (Reading (s, from)) r at n

  let receive qp r ~at n =
    check_post qp "Qp.receive" Receive r at n;
    let s = qp.receives in
    let q = s.entries and i = s.posted in
    let slot = i mod ring_entries in
    s.lengths.(slot) <- n;
    if Option.is_none (Atomic.get qp.failed) then begin
      let base = slot * q.stride in
      Window.fill q.ring base (D.Rq_wqe_hdr.sizeof + D.Sq_sge.sizeof) '\000';
      Window.set8 q.ring
        (base + fst D.Rq_wqe_hdr.wqe_type)
        D.rq_wqe_wqe_type_rcv;
      Window.set8 q.ring (base + fst D.Rq_wqe_hdr.wqe_size) 3;
      sge q slot r at n;
      ring qp.nic.rc.db ~xid:qp.qpn ~typ:s.typ (i + 1)
    end;
    s.posted <- i + 1

  (* Whether [q]'s completion [n] is written: its toggle bit differs from the
     parity of the pass over the queue that reads it. *)
  let written (q : queue) n =
    Window.get8 q.ring (slot q n + fst D.Cq_base.cqe_type_toggle)
    land D.cq_base_toggle
    <> n / q.entries land 1

  let failure qp why =
    strf "%s: queue pair %d failed: %s" qp.nic.name qp.qpn why

  (* Fails [qp] for [why], once, with no command to the firmware. *)
  let fail_qp qp why =
    let failed = Some (failure qp why) in
    ignore (Atomic.compare_and_set qp.failed None failed : bool)

  (* [r]'s side notes [qp]'s failure the first time it sees it: the
     completions [r]'s queue holds by then are the NIC's, and are read before
     its entries complete with the failure, a completion the NIC writes later
     is not read, and on the send side a terminal completion's index says
     which sends completed. Each side notes its own ring, so a ring that
     another domain polls keeps what the NIC completed. *)
  (* The sends of [r] before the terminal completion's send [index] completed:
     an index outside the sends due completes none. *)
  let sends_done r index =
    let done_ = (index - r.taken) land (ring_entries - 1) in
    if index <> no_index && done_ <= r.posted - r.taken then
      r.ok_until <- r.taken + done_

  let notice qp r =
    r.noticed <- true;
    let n = ref r.cq_read in
    while !n - r.cq_read < r.cq.entries && written r.cq !n do
      incr n
    done;
    r.cq_until <- !n;
    if r == qp.sends then sends_done r (Atomic.get qp.terminal)

  (* Completes the oldest due entry of [r] of the failed queue pair: one before
     [ok_until] moved its bytes. *)
  let flush r why =
    let n = r.taken in
    r.taken <- n + 1;
    if n < r.ok_until then Ok r.lengths.(n mod ring_entries) else Error why

  (* A terminal completion, read from [r]'s queue, reports the queue pair's
     failure, and the sends before the one at its consumer index, which
     completed ([sends_done]). The index is kept before the failure is, so
     the send side reads it once it sees the failure; a send side that saw a
     failure raised elsewhere first applies the index as it reads it. *)
  let terminal qp r at =
    let index = Window.get32 r.cq.ring (at + fst D.Cq_terminal.sq_cons_idx) in
    let index = index land 0xffff in
    if index <> no_index then
      ignore (Atomic.compare_and_set qp.terminal no_index index : bool);
    if r == qp.sends && r.noticed then sends_done r index;
    fail_qp qp "the NIC reported its failure"

  (* Reads [r]'s written completion and acknowledges it: the completion of the
     oldest due entry, which fails the queue pair if the entry failed, or
     [None] for a terminal completion, which fails it, and for one of a kind
     the ring does not take, which the library fails it for. *)
  let take qp r =
    let q = r.cq and n = r.cq_read in
    Window.barrier ();
    let at = slot q n in
    let kind =
      Window.get8 q.ring (at + fst D.Cq_base.cqe_type_toggle)
      land D.cq_base_cqe_type_mask
    in
    let status = Window.get8 q.ring (at + fst D.Cq_base.status) in
    let length = Window.get32 q.ring (at + fst D.Cq_res_rc.length) in
    r.cq_read <- n + 1;
    ring qp.nic.rc.db ~xid:r.cq_id ~typ:D.bnxt_re_que_type_cq (n + 1);
    if kind = D.cq_base_cqe_type_terminal then begin
      terminal qp r at;
      None
    end
    else if kind <> r.kind then begin
      fail_qp qp (strf "a completion of type %d" (kind lsr 1));
      None
    end
    else begin
      let i = r.taken in
      r.taken <- i + 1;
      if status = 0 then
        if r == qp.receives then Some (Ok length)
        else Some (Ok r.lengths.(i mod ring_entries))
      else
        let why =
          match List.assoc_opt status r.statuses with
          | Some why -> why
          | None -> strf "status %d" status
        in
        fail_qp qp why;
        Some (Error (failure qp why))
    end

  (* The completion of [r]'s oldest due entry, if it has one, which a failed
     queue pair or machine gives at once, once the completions written before
     the failure are read. *)
  let rec next qp r =
    match Atomic.get qp.failed with
    | Some why ->
        if not r.noticed then notice qp r;
        if r.cq_read < r.cq_until then taken qp r else Some (flush r why)
    | None when written r.cq r.cq_read -> taken qp r
    | None -> (
        match Machine.failed (Function.machine qp.nic.hw.fn) with
        | None -> None
        | Some why ->
            fail_qp qp why;
            next qp r)

  and taken qp r = match take qp r with Some c -> Some c | None -> next qp r

  let due r = r.posted > r.taken

  let poll qp side =
    check "Qp.poll";
    usable qp "Qp.poll";
    let r = rings qp side in
    if due r then next qp r else None

  let wait qp side ~us =
    check "Qp.wait";
    if us < 0 then invalid_argf "Rig_bnxt.Qp.wait: %d us" us;
    usable qp "Qp.wait";
    let r = rings qp side in
    if not (due r) then invalid_arg "Rig_bnxt.Qp.wait: no entry is due";
    let ended () =
      written r.cq r.cq_read || Option.is_some (Atomic.get qp.failed)
    in
    (match Function.wait qp.nic.hw.fn ~us r.waiting ended with
    | Ok () -> ()
    | Error _ -> Option.iter (fail_qp qp) (Function.failed qp.nic.hw.fn));
    next qp r

  let destroy qp =
    match qp.destroyed with
    | Some result -> result
    | None ->
        let result =
          command qp.nic "Qp.destroy" @@ fun () ->
          let nic = qp.nic in
          destroy_qp nic qp.qpn;
          List.iter (fun r -> destroy_cq nic r.cq_id) [ qp.sends; qp.receives ];
          List.iter
            (fun r -> List.iter (free nic.hw) (r.entries.memory @ r.cq.memory))
            [ qp.sends; qp.receives ]
        in
        qp.destroyed <- Some result;
        result
end
