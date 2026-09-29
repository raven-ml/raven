(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GSP: the GPU's system processor, which runs NVIDIA's resource manager in
   its firmware. The process boots it, then allocates and controls the GPU's
   objects through remote procedure calls over two message queues in system
   memory: commands to the GSP, and its answers and events back. *)

module D = Nv_defs
module P = Params
module Mmio = Nx_device_support.Mmio
module Elf = Nx_device_elf
module Pci = Nx_device_support.Pci
module Page_table = Nx_device_support.Page_table

let round_up n a = (n + a - 1) / a * a
let round_down n a = n / a * a

(* The release of the GSP's firmware, whose layouts its messages carry. *)
let release = D.release 570

(* Message queues *)

type queue = {
  view : Mmio.t; (* its header, then its entries *)
  msg_size : int;
  msg_count : int;
  entry_off : int;
  mutable rx : Mmio.t option; (* where its reader keeps its position *)
  mutable seq : int;
}

(* An element: its header, the message's header, then the message. *)
let element_header = fst D.Queue_element.rpc
let payload_off = element_header + D.Rpc_header.sizeof
let tx q f = Mmio.get32 q.view (fst f)

let queue ?(timeout_ms = 10_000) view =
  let module H = D.Msgq_tx_header in
  Nvdev.wait_until ~timeout_ms "the GSP's message queue" (fun () ->
      Mmio.get32 view (fst H.entry_off) = 0x1000);
  {
    view;
    msg_size = Mmio.get32 view (fst H.msg_size);
    msg_count = Mmio.get32 view (fst H.msg_count);
    entry_off = Mmio.get32 view (fst H.entry_off);
    rx = None;
    seq = 0;
  }

(* Reads or writes [n] bytes at byte [off] of the entries, across the end of the
   ring. *)
let ring_read q off n =
  let size = q.msg_size * q.msg_count in
  let off = off mod size in
  let first = Int.min n (size - off) in
  Mmio.read q.view (q.entry_off + off) first
  ^ if first < n then Mmio.read q.view q.entry_off (n - first) else ""

let ring_write q off s =
  let size = q.msg_size * q.msg_count in
  let n = String.length s in
  let first = Int.min n (size - off) in
  Mmio.write q.view (q.entry_off + off) (String.sub s 0 first);
  if first < n then
    Mmio.write q.view q.entry_off (String.sub s first (n - first))

(* The XOR of the message's 64-bit words, folded to 32 bits. *)
let checksum s =
  let s = s ^ String.make ((8 - (String.length s mod 8)) mod 8) '\000' in
  let x = ref 0L in
  for i = 0 to (String.length s / 8) - 1 do
    x := Int64.logxor !x (String.get_int64_le s (8 * i))
  done;
  Int64.to_int
    (Int64.logand
       (Int64.logxor !x (Int64.shift_right_logical !x 32))
       0xffff_ffffL)

(* The element holding one record of [func] with [msg]. *)
let element ~msg_size ~seq func msg =
  let module R = D.Rpc_header in
  let module E = D.Queue_element in
  let h = Bytes.make R.sizeof '\000' in
  P.write h 0 R.signature D.nv_vgpu_msg_signature_valid;
  P.write h 0 R.rpc_result D.nv_vgpu_msg_result_rpc_pending;
  P.write h 0 R.rpc_result_private D.nv_vgpu_msg_result_rpc_pending;
  P.write h 0 R.header_version (3 lsl 24);
  P.write h 0 R.function_ func;
  P.write h 0 R.length (String.length msg + R.sizeof);
  let body = Bytes.to_string h ^ msg in
  let count = (element_header + String.length body + msg_size - 1) / msg_size in
  let e = Bytes.make element_header '\000' in
  P.write e 0 E.elem_count count;
  P.write e 0 E.seq_num seq;
  let whole = Bytes.to_string e ^ body in
  P.write e 0 E.check_sum (checksum whole);
  let whole = Bytes.to_string e ^ body in
  (count, whole ^ String.make ((count * msg_size) - String.length whole) '\000')

(* The number of elements written and not yet read. *)
let pending q =
  match q.rx with
  | None -> 0
  | Some rx ->
      let module H = D.Msgq_tx_header in
      (tx q H.write_ptr - Mmio.get32 rx 0 + q.msg_count) mod q.msg_count

let send_record q ~doorbell func msg =
  let module H = D.Msgq_tx_header in
  let count, e = element ~msg_size:q.msg_size ~seq:q.seq func msg in
  Nvdev.wait_until "room in the GSP's command queue" (fun () ->
      pending q + count < q.msg_count);
  let wp = tx q H.write_ptr in
  ring_write q (wp * q.msg_size) e;
  Mmio.set32 q.view (fst H.write_ptr) ((wp + count) mod q.msg_count);
  Mmio.barrier ();
  q.seq <- q.seq + 1;
  doorbell ()

(* Sends [msg], as records of at most 16 elements: [func], then
   continuations. *)
let send q ~doorbell func msg =
  let max = (q.msg_size * 16) - element_header - D.Rpc_header.sizeof in
  let n = String.length msg in
  send_record q ~doorbell func (String.sub msg 0 (Int.min n max));
  let rec rest off =
    if off < n then begin
      send_record q ~doorbell D.nv_vgpu_msg_function_continuation_record
        (String.sub msg off (Int.min max (n - off)));
      rest (off + max)
    end
  in
  rest max

(* The next message of [q], as (function, result, message), if any. *)
let receive q =
  let module H = D.Msgq_tx_header in
  let module R = D.Rpc_header in
  let rx = Option.get q.rx in
  Mmio.barrier ();
  let rp = Mmio.get32 rx 0 in
  if rp = tx q H.write_ptr then None
  else begin
    let off = rp * q.msg_size in
    let header = ring_read q off payload_off in
    let count = P.read header 0 D.Queue_element.elem_count in
    let get f = P.read header element_header f in
    let msg = ring_read q (off + payload_off) (get R.length - R.sizeof) in
    Mmio.set32 rx 0 ((rp + count) mod q.msg_count);
    Mmio.barrier ();
    Some (get R.function_, get R.rpc_result, msg)
  end

(* The GSP *)

type ctx_buffer = { size : int; virt : bool; phys : bool; local : bool }

type t = {
  d : Nvdev.t;
  flcn : Falcon.t;
  cmd : queue;
  stat_view : Mmio.t;
  mutable stat : queue option;
  libos : int; (* the libos arguments, as the GSP reads them *)
  wpr_meta : int;
  mutable next_handle : int;
  mutable runlists : (int * int) list; (* runlist of each engine *)
  channels : (int, int) Hashtbl.t; (* runlist of each channel *)
  mutable ctx_buffers : (int * ctx_buffer) list;
  mutable device : int;
  mutable subdevice : int;
  gpfifo_class : int;
  compute_class : int;
  dma_class : int;
  mutable error : string option; (* the first error the GSP reported *)
  lock : Mutex.t;
}

let priv_root = 0xc1e00004
let doorbell g () = Nvdev.write g.d ~i:0 "NV_PGSP_QUEUE_HEAD" []

let stat g =
  match g.stat with Some q -> q | None -> failwith "the GSP is not running"

(* Runs the register sequence the GSP asks the CPU to run. *)
let run_cpu_sequencer g msg =
  let d = g.d in
  let module S = D.Rpc_cpu_sequencer in
  let n = P.read msg 0 S.cmd_index in
  let off, _, _ = S.command_buffer in
  let words = Array.init n (fun i -> P.read msg (off + (4 * i)) (0, 4)) in
  let save = Array.make 8 0 in
  let legacy () =
    match g.flcn with
    | Falcon.Legacy _ -> ()
    | Falcon.Cot _ ->
        failwith "the GSP asked to operate its falcon, which the FSP runs"
  in
  let rec go i =
    if i < n then
      let op = words.(i) and arg k = words.(i + k) in
      if op = D.gsp_seq_buf_opcode_reg_write then (
        Nvdev.wreg d (arg 1) (arg 2);
        go (i + 3))
      else if op = D.gsp_seq_buf_opcode_reg_modify then begin
        let a = arg 1 and v = arg 2 and m = arg 3 in
        Nvdev.wreg d a (Nvdev.rreg d a land lnot m lor (v land m));
        go (i + 4)
      end
      else if op = D.gsp_seq_buf_opcode_reg_poll then begin
        let a = arg 1 and m = arg 2 and v = arg 3 in
        Nvdev.wait_until (Printf.sprintf "register 0x%x" a) (fun () ->
            Nvdev.rreg d a land m = v);
        go (i + 6)
      end
      else if op = D.gsp_seq_buf_opcode_delay_us then begin
        Unix.sleepf (float (arg 1) /. 1e6);
        go (i + 2)
      end
      else if op = D.gsp_seq_buf_opcode_reg_store then begin
        save.(arg 2) <- Nvdev.rreg d (arg 1);
        go (i + 3)
      end
      else if op = D.gsp_seq_buf_opcode_core_reset then begin
        legacy ();
        Falcon.reset d Falcon.gsp_falcon;
        Falcon.disable_ctx_req d Falcon.gsp_falcon;
        go (i + 1)
      end
      else if op = D.gsp_seq_buf_opcode_core_start then begin
        legacy ();
        Falcon.start_cpu d Falcon.gsp_falcon;
        go (i + 1)
      end
      else if op = D.gsp_seq_buf_opcode_core_wait_for_halt then begin
        legacy ();
        Falcon.wait_cpu_halted d Falcon.gsp_falcon;
        go (i + 1)
      end
      else if op = D.gsp_seq_buf_opcode_core_resume then begin
        legacy ();
        Falcon.reset d ~riscv:true Falcon.gsp_falcon;
        Nvdev.write d ~value:(Falcon.lo32 g.libos) "NV_PGSP_FALCON_MAILBOX0" [];
        Nvdev.write d ~value:(Falcon.hi32 g.libos) "NV_PGSP_FALCON_MAILBOX1" [];
        Falcon.start_cpu d Falcon.sec2;
        Nvdev.wait_until "SEC2 handing off to the GSP" (fun () ->
            Nvdev.read_field d "NV_PGC6_BSI_SECURE_SCRATCH_14"
              "boot_stage_3_handoff"
            = 1);
        let m = Nvdev.read d ~base:Falcon.sec2 "NV_PFALCON_FALCON_MAILBOX0" in
        if m <> 0 then failwith (Printf.sprintf "SEC2 failed: mailbox 0x%08x" m);
        go (i + 1)
      end
      else
        failwith
          (Printf.sprintf "the GSP's CPU sequence has an unknown opcode %d" op)
  in
  go 0

let error_log msg =
  let module L = D.Rpc_os_error_log in
  let off, _, n = L.err_string in
  let text = String.sub msg off (Int.min n (String.length msg - off)) in
  let text =
    match String.index_opt text '\000' with
    | Some i -> String.sub text 0 i
    | None -> text
  in
  Printf.sprintf "GSP error (channel %d): %s" (P.read msg 0 L.chid)
    (String.trim text)

(* Handles the messages of the status queue, and is the answer to [want] if one
   came. Events are handled as they come: the CPU sequences run, and the first
   error is kept. *)
let rec drain g ~want =
  match receive (stat g) with
  | None -> None
  | Some (func, result, msg) ->
      if func = D.nv_vgpu_msg_event_gsp_run_cpu_sequencer then
        run_cpu_sequencer g msg
      else if func = D.nv_vgpu_msg_event_os_error_log then
        begin if g.error = None then g.error <- Some (error_log msg)
        end
      else if func = D.nv_vgpu_msg_event_mmu_fault_queued then
        begin if g.error = None then
          g.error <- Some "the GSP reported an MMU fault"
        end;
      if result <> 0 then
        failwith (Printf.sprintf "the GSP's call %d failed: 0x%x" func result);
      if Some func = want then Some msg else drain g ~want

let wait_answer ?(timeout_ms = 10_000) g func =
  let t0 = Unix.gettimeofday () in
  let rec go () =
    match drain g ~want:(Some func) with
    | Some msg -> msg
    | None ->
        if (Unix.gettimeofday () -. t0) *. 1000. > float timeout_ms then
          failwith
            (Printf.sprintf "timed out waiting for the GSP's answer to call %d"
               func);
        Domain.cpu_relax ();
        go ()
  in
  go ()

let call g func msg =
  Mutex.protect g.lock (fun () ->
      send g.cmd ~doorbell:(doorbell g) func msg;
      wait_answer g func)

(* Handles the GSP's pending messages; raises its first reported error. *)
let poll g =
  Mutex.protect g.lock (fun () -> ignore (drain g ~want:None));
  Option.iter failwith g.error

let handle g =
  let h = g.next_handle in
  g.next_handle <- h + 1;
  h

let check what s = Rm.check release what s
let mm g = Nvdev.mm g.d

let valloc g n =
  match Page_table.alloc ~contiguous:true (mm g) n with
  | Some m -> m
  | None -> failwith "no GPU memory for the GSP"

let paddr (m : Page_table.mapping) = fst (List.hd m.pages)

let set_desc p field ~base ~size ~space =
  let module M = D.Memory_desc in
  let at f = (fst field + fst f, snd f) in
  P.set p (at M.base) base;
  P.set p (at M.size) size;
  P.set p (at M.address_space) space;
  P.set p (at M.cache_attrib) 0

(* The number of entries of the root table. *)
let root_entries (d : Nvdev.t) =
  let levels = Nvdev.levels d.mmu_ver in
  let bits = if d.mmu_ver = 3 then 56 else 48 in
  1 lsl (bits + 1 - List.nth levels (List.length levels - 1))

let rec rm_control g ?(client = priv_root) obj cmd params =
  let module C = D.Rpc_rm_control in
  let a = Bytes.make C.sizeof '\000' in
  let n = Option.fold ~none:0 ~some:P.length params in
  P.write a 0 C.h_client client;
  P.write a 0 C.h_object obj;
  P.write a 0 C.cmd cmd;
  P.write a 0 C.params_size n;
  let msg = Bytes.to_string a ^ Option.fold ~none:"" ~some:P.to_string params in
  let r = call g D.nv_vgpu_msg_function_gsp_rm_control msg in
  check (Printf.sprintf "command 0x%x" cmd) (P.read r 0 C.status);
  Option.iter (fun p -> P.blit_string (String.sub r C.sizeof n) p 0) params;
  if cmd = D.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token then
    Option.iter
      (fun p ->
        let module W = D.Work_submit_token in
        let blackwell = String.sub (Nvdev.chip_name g.d) 0 3 = "GB2" in
        P.set p W.work_submit_token
          (P.get p W.work_submit_token
          lor (Option.value ~default:0 (Hashtbl.find_opt g.channels obj) lsl 16)
          lor if blackwell then 1 lsl 30 else 0))
      params

(* Promotes the context buffers [bufs] of the channel [obj]: those it has
   ([have]) or new ones, each set up physically and virtually as [phys] and
   [virt] say, defaulting to the buffer's description. *)
and promote_ctx g ~client ~subdevice obj bufs ?(have = []) ?virt ?phys
    ?(engine = 1) () =
  let module Pr = D.Promote_ctx in
  let module E = D.Promote_entry in
  let p = P.create Pr.sizeof in
  P.set p Pr.entry_count (List.length bufs);
  P.set p Pr.engine_type engine;
  P.set p Pr.h_chan_client client;
  P.set p Pr.h_object obj;
  let made =
    List.mapi
      (fun i (id, b) ->
        let v = Option.value virt ~default:b.virt
        and ph = Option.value phys ~default:b.phys in
        let x =
          match List.assoc_opt id have with
          | Some x -> x
          | None -> valloc g b.size
        in
        let f = P.elt_field Pr.promote_entry i in
        P.set p (f E.buffer_id) id;
        P.set p (f E.gpu_virt_addr) (if v then x.Page_table.va else 0);
        P.set p (f E.b_initialize) (Bool.to_int ph);
        P.set p (f E.gpu_phys_addr) (if ph then paddr x else 0);
        P.set p (f E.size) (if ph then b.size else 0);
        P.set p (f E.phys_attr) (if ph then 0x4 else 0);
        P.set p (f E.b_nonmapped) (Bool.to_int (ph && not v));
        (id, x))
      bufs
  in
  rm_control g ~client subdevice D.nv2080_ctrl_cmd_gpu_promote_ctx (Some p);
  made

and rm_alloc g ?(client = priv_root) ~parent cls params =
  let module A = D.Rpc_rm_alloc in
  let module R = (val release : D.RELEASE) in
  let module G = R.Gpfifo_alloc in
  if cls = g.gpfifo_class then
    Option.iter
      (fun p ->
        let ramfc = valloc g 0x1000 in
        set_desc p G.ramfc_mem ~base:(paddr ramfc) ~size:0x200 ~space:2;
        set_desc p G.instance_mem ~base:(paddr ramfc) ~size:0x1000 ~space:2;
        (match Nvdev.boot_mem g.d ~sysmem:false 0x5000 with
        | _, Some pa, _ ->
            set_desc p G.mthdbuf_mem ~base:pa ~size:0x5000 ~space:2
        | _ -> assert false (* the GPU's memory has an address *));
        if client <> priv_root && P.get p G.h_object_error <> 0 then begin
          set_desc p G.error_notifier_mem ~base:0 ~size:0xecc ~space:0;
          let userd =
            P.get p (P.elt G.h_userd_memory 0)
            + P.get p (P.elt G.userd_offset 0)
          in
          set_desc p G.userd_mem ~base:userd ~size:0x400 ~space:2
        end)
      params;
  let obj = handle g in
  let a = Bytes.make A.sizeof '\000' in
  P.write a 0 A.h_client client;
  P.write a 0 A.h_parent parent;
  P.write a 0 A.h_object obj;
  P.write a 0 A.h_class cls;
  P.write a 0 A.params_size (Option.fold ~none:0 ~some:P.length params);
  let msg = Bytes.to_string a ^ Option.fold ~none:"" ~some:P.to_string params in
  let r = call g D.nv_vgpu_msg_function_gsp_rm_alloc msg in
  check (Printf.sprintf "allocating class 0x%x" cls) (P.read r 0 A.status);
  if cls = g.gpfifo_class then
    Option.iter
      (fun p ->
        let e = P.get p G.engine_type in
        let key = e + if e >= D.nv2080_engine_type_nvdec0 then 10 else 0 in
        Hashtbl.replace g.channels obj
          (Option.value ~default:0 (List.assoc_opt key g.runlists)))
      params;
  if client <> priv_root then begin
    if cls = D.fermi_vaspace_a then
      set_page_directory g ~client ~device:parent obj;
    if cls = D.nv01_device_0 then g.device <- obj;
    if cls = g.compute_class then begin
      let bufs =
        List.filter (fun (k, _) -> List.mem k [ 0; 1; 2 ]) g.ctx_buffers
      in
      let phys =
        promote_ctx g ~client ~subdevice:g.subdevice parent bufs ~virt:false ()
      in
      ignore
        (promote_ctx g ~client ~subdevice:g.subdevice parent bufs ~have:phys
           ~phys:false ())
    end
  end;
  if cls = D.nv20_subdevice_0 then g.subdevice <- obj;
  if cls = D.nv01_root then client else obj

and set_page_directory g ~client ~device vaspace =
  let module S = D.Rpc_set_page_directory in
  let module Pd = D.Set_page_directory in
  let a = Bytes.make S.sizeof '\000' in
  let at f = (fst S.params + fst f, snd f) in
  P.write a 0 S.h_client client;
  P.write a 0 S.h_device device;
  P.write a 0 S.pasid 0xffff_ffff;
  P.write a 0 (at Pd.phys_address) (Page_table.root (mm g));
  P.write a 0 (at Pd.num_entries) (root_entries g.d);
  P.write a 0 (at Pd.flags) 0x8;
  P.write a 0 (at Pd.h_va_space) vaspace;
  P.write a 0 (at Pd.pasid) 0xffff_ffff;
  P.write a 0 (at Pd.sub_device_id) 1;
  P.write a 0 (at Pd.ch_id) 0;
  ignore (call g D.nv_vgpu_msg_function_set_page_directory (Bytes.to_string a))

(* The resource manager, through the GSP, under the client [root]. *)
let rm g ~root =
  {
    Rm.root;
    alloc =
      (fun ~parent cls params -> rm_alloc g ~client:root ~parent cls params);
    control = (fun obj cmd params -> rm_control g ~client:root obj cmd params);
    free =
      (fun ~parent:_ _ -> failwith "the GSP's objects live as long as the GPU");
  }

(* Boot *)

(* The page counts of the GSP's image of [len] bytes in radix-3 form: three
   levels of 64-bit page addresses, then the image's pages. *)
let radix3_pages len =
  let n3 = round_up len 0x1000 / 0x1000 in
  let per = D.libos_memory_region_radix_page_log2 - 3 in
  let n2 = ((n3 - 1) lsr per) + 1 in
  let n1 = ((n2 - 1) lsr per) + 1 in
  let n0 = ((n1 - 1) lsr per) + 1 in
  [| n0; n1; n2; n3 |]

type layout = {
  image : string; (* the GSP's image, in radix-3 order *)
  signature : string;
  bootloader : string;
  desc : string; (* the bootloader's descriptor *)
}

(* The images of [gsp_fw] and [bootloader_fw] the GPU [chip] runs. *)
let images ~chip ~gsp_fw ~bootloader_fw =
  let o = Elf.load gsp_fw in
  let section n = Falcon.section o n in
  let module B = D.Bin_header in
  let hdr = P.read bootloader_fw 0 B.header_offset in
  {
    image = section ".fwimage";
    signature =
      section
        (".fwsignature_" ^ String.lowercase_ascii (String.sub chip 0 4) ^ "x");
    bootloader =
      String.sub bootloader_fw
        (P.read bootloader_fw 0 B.data_offset)
        (P.read bootloader_fw 0 B.data_size);
    desc = String.sub bootloader_fw hdr D.Riscv_ucode_desc.sizeof;
  }

(* The GPU memory the GSP's firmware reserves at the top, laid out as the CPU
   describes it on Ampere and Ada, from the sizes of the images: the WPR
   metadata's fields. *)
let wpr ~vram ~boot ~image =
  let module W = D.Wpr_meta in
  let vga = 0x100000 and frts = 0x100000 and heap = 0x8100000 in
  let non_wpr = 0x100000 in
  let vga_off = vram - vga in
  let frts_off = vga_off - frts in
  let boot_off = frts_off - boot in
  let gsp_off = round_down (boot_off - image) 0x10000 in
  let heap_off = round_down (gsp_off - heap) 0x100000 in
  let wpr_start = round_down (heap_off - 0x1000) 0x100000 in
  let non_wpr_off = round_down (wpr_start - non_wpr) 0x100000 in
  [
    (W.vga_workspace_size, vga);
    (W.vga_workspace_offset, vga_off);
    (W.gsp_fw_wpr_end, vga_off);
    (W.frts_size, frts);
    (W.frts_offset, frts_off);
    (W.boot_bin_offset, boot_off);
    (W.gsp_fw_offset, gsp_off);
    (W.gsp_fw_heap_size, heap);
    (W.fb_size, vram);
    (W.gsp_fw_heap_offset, heap_off);
    (W.gsp_fw_wpr_start, wpr_start);
    (W.non_wpr_heap_size, non_wpr);
    (W.non_wpr_heap_offset, non_wpr_off);
    (W.gsp_fw_rsvd_start, non_wpr_off);
  ]

(* The sizes the FMC boot passes: the FMC lays the region out itself. *)
let fmc_sizes =
  let module W = D.Wpr_meta in
  [
    (W.vga_workspace_size, 0x20000);
    (W.pmu_reserved_size, 0x1820000);
    (W.non_wpr_heap_size, 0x220000);
    (W.gsp_fw_heap_size, 0x8700000);
    (W.frts_size, 0x100000);
  ]

(* The top of the memory the process may manage: below the region the GSP
   reserves, which it must never hand out, and 64 MiB below the top. *)
let managed_top ~vram ~fmc ~boot ~image =
  let bound =
    if fmc then
      vram
      - List.fold_left
          (fun n (_, s) -> n + s)
          (boot + image + 0x100000)
          fmc_sizes
    else List.assoc D.Wpr_meta.gsp_fw_rsvd_start (wpr ~vram ~boot ~image)
  in
  Int.min (vram - (64 lsl 20)) (round_down bound (2 lsl 20))

let bdf bus =
  match String.split_on_char ':' bus with
  | [ _; b; df ] -> (
      match String.split_on_char '.' df with
      | [ dev; fn ] ->
          (int_of_string ("0x" ^ b) lsl 8)
          lor (int_of_string ("0x" ^ dev) lsl 3)
          lor int_of_string ("0x" ^ fn)
      | _ -> 0)
  | _ -> 0

let registry = [ ("RMForcePcieConfigSave", 1); ("RMSecBusResetEnable", 1) ]

let registry_table () =
  let module T = D.Registry_table in
  let module E = D.Registry_entry in
  let n = List.length registry in
  let names_off = T.sizeof + (n * E.sizeof) in
  let entries = Bytes.make (n * E.sizeof) '\000' in
  let names = Buffer.create 64 in
  List.iteri
    (fun i (k, v) ->
      let base = i * E.sizeof in
      P.write entries base E.name_offset (names_off + Buffer.length names);
      P.write entries base E.type_ D.registry_table_entry_type_dword;
      P.write entries base E.data v;
      P.write entries base E.length 4;
      Buffer.add_string names k;
      Buffer.add_char names '\000')
    registry;
  let h = Bytes.make T.sizeof '\000' in
  P.write h 0 T.size (names_off + Buffer.length names);
  P.write h 0 T.num_entries n;
  Bytes.to_string h ^ Bytes.to_string entries ^ Buffer.contents names

let classes d =
  match String.sub (Nvdev.chip_name d) 0 2 with
  | "GB" ->
      ( D.blackwell_channel_gpfifo_a,
        D.blackwell_compute_b,
        D.blackwell_dma_copy_b )
  | "AD" -> (D.ampere_channel_gpfifo_a, D.ada_compute_a, D.ampere_dma_copy_b)
  | _ -> (D.ampere_channel_gpfifo_a, D.ampere_compute_b, D.ampere_dma_copy_b)

let u64s l =
  let b = Bytes.create (8 * List.length l) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) l;
  Bytes.to_string b

(* Prepares the GSP's boot: its queues, arguments, image and WPR metadata, and
   the first commands it reads. *)
let init_sw (d : Nvdev.t) flcn (l : layout) =
  let queue_size = 0x40000 in
  let queue_pages = queue_size * 2 / 0x1000 in
  let ptes = queue_pages + (round_up (queue_pages * 8) 0x1000 / 0x1000) in
  let pt_size = round_up (ptes * 8) 0x1000 in
  let queues, _, pages =
    Nvdev.boot_mem d ~sysmem:true (pt_size + (queue_size * 2))
  in
  Mmio.write queues 0 (u64s pages);
  let module Q = D.Queue_init_args in
  let module A = D.Gsp_arguments in
  let args = Bytes.make A.sizeof '\000' in
  let qa = fst A.message_queue_init_arguments in
  P.write args qa Q.shared_mem_phys_addr (List.hd pages);
  P.write args qa Q.page_table_entry_count ptes;
  P.write args qa Q.cmd_queue_offset pt_size;
  P.write args qa Q.stat_queue_offset (pt_size + queue_size);
  P.write args 0 A.b_dmem_stack 1;
  let _, _, rm_args = Nvdev.boot_mem d ~data:(Bytes.to_string args) A.sizeof in
  let cmd_view = Mmio.sub queues pt_size queue_size in
  let stat_view = Mmio.sub queues (pt_size + queue_size) queue_size in
  let module H = D.Msgq_tx_header in
  let h = Bytes.make H.sizeof '\000' in
  P.write h 0 H.size queue_size;
  P.write h 0 H.entry_off 0x1000;
  P.write h 0 H.msg_size 0x1000;
  P.write h 0 H.msg_count ((queue_size - 0x1000) / 0x1000);
  P.write h 0 H.flags 1;
  P.write h 0 H.rx_hdr_off H.sizeof;
  Mmio.write cmd_view 0 (Bytes.to_string h);
  (* libos: its log buffers and its arguments *)
  let _, _, logs = Nvdev.boot_mem d (2 lsl 20) in
  let module L = D.Libos_region in
  let region id ~pa ~size =
    let r = Bytes.make L.sizeof '\000' in
    let id8 = String.fold_left (fun n c -> (n lsl 8) lor Char.code c) 0 id in
    P.write r 0 L.kind D.libos_memory_region_contiguous;
    P.write r 0 L.loc D.libos_memory_region_loc_sysmem;
    P.write r 0 L.size size;
    P.write r 0 L.id8 id8;
    P.write r 0 L.pa pa;
    Bytes.to_string r
  in
  let regions =
    List.mapi
      (fun i n ->
        region ("LOG" ^ n) ~pa:(List.hd logs + (0x10000 * i)) ~size:0x10000)
      [ "INIT"; "INTR"; "RM"; "MNOC"; "KRNL" ]
    @ [ region "RMARGS" ~pa:(List.hd rm_args) ~size:0x1000 ]
  in
  let _, _, libos = Nvdev.boot_mem d ~data:(String.concat "" regions) 0x1000 in
  (* the image, as radix-3 tables over its pages *)
  let n = radix3_pages (String.length l.image) in
  let offsets =
    Array.init 4 (fun i -> Array.fold_left ( + ) 0 (Array.sub n 0 i) * 0x1000)
  in
  let radix, _, addrs =
    Nvdev.boot_mem d (offsets.(3) + String.length l.image)
  in
  let addrs = Array.of_list addrs in
  Mmio.write radix offsets.(3) l.image;
  for i = 0 to 2 do
    let cur = Array.fold_left ( + ) 0 (Array.sub n 0 (i + 1)) in
    Mmio.write radix offsets.(i)
      (u64s (Array.to_list (Array.sub addrs cur n.(i + 1))))
  done;
  let _, _, sign =
    Nvdev.boot_mem d ~data:l.signature (String.length l.signature)
  in
  let _, _, boot =
    Nvdev.boot_mem d ~data:l.bootloader (String.length l.bootloader)
  in
  (* the WPR metadata *)
  let module W = D.Wpr_meta in
  let module U = D.Riscv_ucode_desc in
  let m = Bytes.make W.sizeof '\000' in
  let set (f, v) = P.write m 0 f v in
  let boot_size = String.length l.bootloader
  and image_size = String.length l.image in
  P.write m 0 W.size_of_bootloader boot_size;
  P.write m 0 W.sysmem_addr_of_bootloader (List.hd boot);
  P.write m 0 W.size_of_radix3_elf image_size;
  P.write m 0 W.sysmem_addr_of_radix3_elf addrs.(0);
  P.write m 0 W.size_of_signature 0x1000;
  P.write m 0 W.sysmem_addr_of_signature (List.hd sign);
  P.write m 0 W.bootloader_code_offset (P.read l.desc 0 U.monitor_code_offset);
  P.write m 0 W.bootloader_data_offset (P.read l.desc 0 U.monitor_data_offset);
  P.write m 0 W.bootloader_manifest_offset (P.read l.desc 0 U.manifest_offset);
  P.write m 0 W.revision D.gsp_fw_wpr_meta_revision;
  Bytes.set_int64_le m (fst W.magic) D.gsp_fw_wpr_meta_magic;
  if d.fmc_boot then List.iter set fmc_sizes
  else begin
    let layout = wpr ~vram:d.vram_size ~boot:boot_size ~image:image_size in
    List.iter set layout;
    if List.assoc W.frts_offset layout <> Falcon.frts_offset d then
      failwith "the FRTS region is not where the WPR metadata puts it"
  end;
  let _, _, meta = Nvdev.boot_mem d ~data:(Bytes.to_string m) W.sizeof in
  let gpfifo_class, compute_class, dma_class = classes d in
  let g =
    {
      d;
      flcn;
      cmd =
        {
          view = cmd_view;
          msg_size = 0x1000;
          msg_count = (queue_size - 0x1000) / 0x1000;
          entry_off = 0x1000;
          rx = None;
          seq = 0;
        };
      stat_view;
      stat = None;
      libos = List.hd libos;
      wpr_meta = List.hd meta;
      next_handle = 0xcf000000;
      runlists = [];
      channels = Hashtbl.create 8;
      ctx_buffers = [];
      device = 0;
      subdevice = 0;
      gpfifo_class;
      compute_class;
      dma_class;
      error = None;
      lock = Mutex.create ();
    }
  in
  (* the system's description, and the registry, for the GSP to read at boot *)
  let module S = D.System_info in
  let pci = d.pci in
  let s = Bytes.make S.sizeof '\000' in
  P.write s 0 S.gpu_phys_addr (fst (Pci.bar pci 0));
  P.write s 0 S.gpu_phys_fb_addr (fst (Pci.bar pci 1));
  P.write s 0 S.gpu_phys_inst_addr (fst (Pci.bar pci 3));
  P.write s 0 S.pci_config_mirror_base (if d.fmc_boot then 0x92000 else 0x88000);
  P.write s 0 S.pci_config_mirror_size 0x1000;
  P.write s 0 S.nv_domain_bus_device_func (bdf (Pci.bus pci));
  P.write s 0 S.b_is_passthru 1;
  P.write s 0 S.pci_device_id (Pci.read_config pci 0x00 4);
  P.write s 0 S.pci_sub_device_id (Pci.read_config pci 0x2c 4);
  P.write s 0 S.pci_revision_id (Pci.read_config pci 0x08 1);
  P.write s 0 S.max_user_va 0x7ffffffff000;
  send g.cmd ~doorbell:(doorbell g) D.nv_vgpu_msg_function_gsp_set_system_info
    (Bytes.to_string s);
  send g.cmd ~doorbell:(doorbell g) D.nv_vgpu_msg_function_set_registry
    (registry_table ());
  g

(* Reserves the golden context: a channel of the privileged client whose context
   buffers the GSP promotes, which later channels' contexts copy. *)
let golden_context g =
  let ignore_handle (_ : int) = () in
  let alloc = rm_alloc g ~client:priv_root in
  ignore_handle
    (alloc ~parent:0 D.nv01_root (Some (P.create D.Nv0000_alloc.sizeof)));
  let dp = P.create D.Nv0080_alloc.sizeof in
  P.set dp D.Nv0080_alloc.h_client_share priv_root;
  let dev = alloc ~parent:priv_root D.nv01_device_0 (Some dp) in
  let subdev =
    alloc ~parent:dev D.nv20_subdevice_0 (Some (P.create D.Nv2080_alloc.sizeof))
  in
  let module R = (val release : D.RELEASE) in
  let vaspace =
    alloc ~parent:dev D.fermi_vaspace_a (Some (P.create R.Vaspace_alloc.sizeof))
  in
  let module T = D.Device_info_table in
  let module E = D.Device_entry in
  let t = P.create T.sizeof in
  rm_control g subdev D.nv2080_ctrl_cmd_fifo_get_device_info_table (Some t);
  g.runlists <-
    List.init (P.get t T.num_entries) (fun i ->
        let data k =
          P.get t (P.elt_field T.entries i (P.elt E.engine_data k))
        in
        (data 2, data 3));
  (* the reserved page directory entries of 512 MiB *)
  let mm = mm g in
  let res = 512 lsl 20 in
  let va =
    match Page_table.Space.alloc ~align:res (Page_table.space mm) res with
    | Some va -> va
    | None -> failwith "no addresses for the GSP"
  in
  let module Rp = D.Reserved_pdes in
  let p = P.create Rp.sizeof in
  P.set p Rp.page_size res;
  P.set p Rp.num_levels_to_copy 3;
  P.set p Rp.virt_addr_lo va;
  P.set p Rp.virt_addr_hi (va + res - 1);
  let levels = List.rev (Nvdev.levels g.d.mmu_ver) in
  List.iteri
    (fun i table ->
      let f = P.elt_field Rp.levels i in
      P.set p (f Rp.levels_phys_address) table;
      P.set p (f Rp.levels_size)
        (if i = 0 then root_entries g.d * 8 else 0x1000);
      P.set p (f Rp.levels_page_shift) (List.nth levels i);
      P.set p (f Rp.levels_aperture) 1)
    (Page_table.tables mm ~va res);
  rm_control g vaspace D.nv90f1_ctrl_cmd_vaspace_copy_server_reserved_pdes
    (Some p);
  (* a channel of 32 entries *)
  let area = valloc g 0x1000 in
  let module G = R.Gpfifo_alloc in
  let gp = P.create G.sizeof in
  P.set gp G.gp_fifo_offset area.va;
  P.set gp G.gp_fifo_entries 32;
  P.set gp G.engine_type 1;
  P.set gp G.cid 3;
  P.set gp G.h_va_space vaspace;
  P.set gp (P.elt G.userd_offset 0) 0x100;
  set_desc gp G.userd_mem ~base:(paddr area + 0x100) ~size:0x20 ~space:2;
  P.set gp G.internal_flags 0x1a;
  P.set gp G.flags 0x200320;
  let channel = alloc ~parent:dev g.gpfifo_class (Some gp) in
  let module C = D.Context_buffers_info in
  let module Cb = D.Context_buffers in
  let module B = D.Context_buffer in
  let c = P.create C.sizeof in
  rm_control g subdev
    D.nv2080_ctrl_cmd_internal_static_kgr_get_context_buffers_info (Some c);
  let info ?(add = 0) ?align idx =
    let f x =
      P.get c
        (P.elt_field C.engine_context_buffers_info 0
           (P.elt_field Cb.engine idx x))
    in
    round_up (f B.size + add) (Option.value align ~default:(f B.alignment))
  in
  let graphics =
    info ~add:0x40000
      D.nv0080_ctrl_fifo_get_engine_context_properties_engine_id_graphics
  in
  let patch =
    info
      D.nv0080_ctrl_fifo_get_engine_context_properties_engine_id_graphics_patch
  in
  let cfg x = info ?align:(if x = 5 then Some (2 lsl 20) else None) (x + 14) in
  let buf ?(local = false) ~phys ~virt size = { size; phys; virt; local } in
  g.ctx_buffers <-
    [
      (0, buf graphics ~phys:true ~virt:true);
      (1, buf patch ~phys:true ~virt:true ~local:true);
      (2, buf patch ~phys:true ~virt:true);
    ]
    @ List.map (fun x -> (x, buf (cfg x) ~phys:false ~virt:true)) [ 3; 4; 5; 6 ]
    @ [
        (9, buf (cfg 9) ~phys:true ~virt:true);
        (10, buf (cfg 10) ~phys:true ~virt:false);
        (11, buf (cfg 10) ~phys:true ~virt:true);
      ];
  ignore
    (promote_ctx g ~client:priv_root ~subdevice:subdev channel
       (List.filter (fun (_, b) -> not b.local) g.ctx_buffers)
       ());
  ignore_handle (alloc ~parent:channel g.compute_class None);
  ignore_handle (alloc ~parent:channel g.dma_class None)

(* Waits for the booted GSP, then sets up its golden context. *)
let init_hw g =
  let stat = queue g.stat_view in
  let module H = D.Msgq_tx_header in
  stat.rx <-
    Some (Mmio.sub g.cmd.view (Mmio.get32 g.cmd.view (fst H.rx_hdr_off)) 4);
  g.cmd.rx <-
    Some (Mmio.sub g.stat_view (Mmio.get32 g.stat_view (fst H.rx_hdr_off)) 4);
  g.stat <- Some stat;
  Mutex.protect g.lock (fun () ->
      ignore (wait_answer g D.nv_vgpu_msg_event_gsp_init_done));
  Nvdev.write g.d "NV_PBUS_BAR1_BLOCK"
    [ ("mode", 0); ("target", 0); ("ptr", 0) ];
  if g.d.fmc_boot then
    Nvdev.write g.d "NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR"
      [ ("mode", 0); ("target", 0); ("ptr", 0) ];
  golden_context g

(* Tells the GSP the driver is unloading, so the next boot finds it idle. *)
let fini g =
  let module U = D.Rpc_unloading in
  let u = Bytes.make U.sizeof '\000' in
  P.write u 0 U.new_level (1 lsl 6);
  ignore
    (call g D.nv_vgpu_msg_function_unloading_guest_driver (Bytes.to_string u))
