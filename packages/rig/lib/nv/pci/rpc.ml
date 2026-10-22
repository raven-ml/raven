(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An element's header holds a checksum, a sequence number and an element
   count; the RPC's header a signature, a function, a result and a length.
   A body longer than 16 elements continues in records of the function
   CONTINUATION_RECORD. A record's checksum makes the XOR of its 64-bit
   words folded to 32 bits zero (message_queue_cpu.c). The bodies are
   g_rpc-structures.h's and rmgspseq.h's. [fault] names an exception type as
   nverror.h does, and leaves out OS_ERROR_LOG, since the GSP stops a
   channel itself and says so with RC_TRIGGERED (_kgspRpcRCTriggered). *)

open Field

let strf = Printf.sprintf

module E = Gsp_defs.Queue_element
module R = Gsp_defs.Rpc_header

(* Records *)

let element_size = 0x1000

(* A record's elements hold the element's header up to its RPC header, then the
   RPC header and the body. *)
let element_header = fst E.rpc
let header_size = element_header + R.sizeof

(* The most elements a record takes (message_queue_cpu.c's
   GSP_MSG_QUEUE_RECORD_MAX = 16). *)
let record_elements = 16
let record_max = (element_size * record_elements) - header_size

(* The XOR of [s]'s little-endian 64-bit words, [s] padded with zeros to a
   multiple of 8 bytes, folded to 32 bits. *)
let checksum s =
  let x = ref 0L in
  let n = String.length s in
  for i = 0 to ((n + 7) / 8) - 1 do
    let w =
      if (8 * i) + 8 <= n then String.get_int64_le s (8 * i)
      else
        let b = Bytes.make 8 '\000' in
        Bytes.blit_string s (8 * i) b 0 (n - (8 * i));
        Bytes.get_int64_le b 0
    in
    x := Int64.logxor !x w
  done;
  Int64.(to_int (logand (logxor !x (shift_right_logical !x 32)) 0xffff_ffffL))

(* The record of function [fn] with [body] and sequence number [seq], whole
   elements long, with its checksum. *)
let element ~seq fn body =
  let n = header_size + String.length body in
  let count = (n + element_size - 1) / element_size in
  let b = Bytes.make (count * element_size) '\000' in
  set b E.seq_num seq;
  set b E.elem_count count;
  let version =
    Gsp_defs.nv_vgpu_msg_header_version_major_tot
    lsl fst Gsp_defs.nv_vgpu_msg_header_version_major
    lor Gsp_defs.nv_vgpu_msg_header_version_minor_tot
        lsl fst Gsp_defs.nv_vgpu_msg_header_version_minor
  in
  let rpc (off, n) = (element_header + off, n) in
  set b (rpc R.header_version) version;
  set b (rpc R.signature) Gsp_defs.nv_vgpu_msg_signature_valid;
  set b (rpc R.length) (R.sizeof + String.length body);
  set b (rpc R.function_) fn;
  set b (rpc R.rpc_result) Gsp_defs.nv_vgpu_msg_result_rpc_pending;
  set b (rpc R.rpc_result_private) Gsp_defs.nv_vgpu_msg_result_rpc_pending;
  Bytes.blit_string body 0 b header_size (String.length body);
  set b E.check_sum (checksum (Bytes.unsafe_to_string b));
  Bytes.unsafe_to_string b

let records ~seq fn body =
  let n = String.length body in
  let rec go acc seq off =
    if off >= n then List.rev acc
    else
      let len = Int.min record_max (n - off) in
      let e =
        element ~seq Gsp_defs.nv_vgpu_msg_function_continuation_record
          (String.sub body off len)
      in
      go (e :: acc) (seq + 1) (off + len)
  in
  let first = Int.min record_max n in
  element ~seq fn (String.sub body 0 first) :: go [] (seq + 1) first

let elements s = get s E.elem_count

type message = { fn : int; result : int; body : string }

let message s =
  if String.length s < header_size then
    Error "a GSP message shorter than its header"
  else
    let rpc (off, n) = (element_header + off, n) in
    let count = get s E.elem_count in
    let len = get s (rpc R.length) - R.sizeof in
    if get s (rpc R.signature) <> Gsp_defs.nv_vgpu_msg_signature_valid then
      Error
        (strf "a GSP message without the RPC signature, 0x%x"
           (get s (rpc R.signature)))
    else if len < 0 || header_size + len > String.length s || count < 1 then
      Error (strf "a GSP message of %d elements longer than them" count)
    else
      Ok
        ( {
            fn = get s (rpc R.function_);
            result = get s (rpc R.rpc_result);
            body = String.sub s header_size len;
          },
          count )

let fault m =
  if m.fn = Gsp_defs.nv_vgpu_msg_event_rc_triggered then
    let module T = Gsp_defs.Rpc_rc_triggered in
    let b = m.body in
    if String.length b < T.sizeof then Some "the GSP stopped a channel"
    else
      let e = get b T.except_type in
      let name =
        Option.value ~default:"an unknown error"
          (List.assoc_opt e Gsp_defs.robust_channel_errors)
      in
      Some
        (strf "the GSP stopped channel %d: %s (Xid %d)" (get b T.chid) name e)
  else if m.fn = Gsp_defs.nv_vgpu_msg_event_mmu_fault_queued then
    Some "the GPU's MMU queued a fault"
  else None

(* The text of the character array [(off, 1, n)] of [s], which holds it, to
   its first zero byte. *)
let text s (off, _, n) =
  let t = String.sub s off n in
  match String.index_opt t '\000' with Some i -> String.sub t 0 i | None -> t

(* CPU-RM files the GSP's journal (NOCAT) records into its own
   (_kgspRpcGspPostNocatRecord): of a record, the source and the error code
   say what failed. *)
let report m =
  if m.fn = Gsp_defs.nv_vgpu_msg_event_os_error_log then
    let module L = Gsp_defs.Rpc_os_error_log in
    let b = m.body in
    if String.length b < L.sizeof then
      Some ("an error log", "an error log the GSP cut short")
    else
      let xid = strf "Xid %d" (get b L.except_type) in
      Some (xid, strf "%s: %s" xid (text b L.err_string))
  else if m.fn = Gsp_defs.nv_vgpu_msg_event_gsp_post_nocat_record then
    let module N = Gsp_defs.Nocat_record in
    let b = m.body in
    if String.length b < N.sizeof then
      Some ("a journal record", "a journal record the GSP cut short")
    else
      (* The error code takes 64 bits, which [get] would cut to 63. *)
      let code = String.get_int64_le b (fst N.error_code) in
      let source = text b N.source in
      Some (source, strf "%s, error 0x%Lx" source code)
  else None

(* Calls *)

let rm_alloc ~client ~parent ~obj ~cls params =
  let module A = Gsp_defs.Rpc_rm_alloc in
  record A.sizeof (fun b ->
      set b A.h_client client;
      set b A.h_parent parent;
      set b A.h_object obj;
      set b A.h_class cls;
      set b A.params_size (String.length params))
  ^ params

let rm_control ~client ~obj ~cmd params =
  let module C = Gsp_defs.Rpc_rm_control in
  record C.sizeof (fun b ->
      set b C.h_client client;
      set b C.h_object obj;
      set b C.cmd cmd;
      set b C.params_size (String.length params))
  ^ params

(* The parent is NV01_NULL_OBJECT, as rpcRmApiFree_GSP gives it: the RM finds
   the object by its handle. *)
let free ~client ~obj =
  let module P = Gsp_defs.Free_params in
  let p f = at Gsp_defs.Rpc_free.params f in
  record Gsp_defs.Rpc_free.sizeof (fun b ->
      set b (p P.h_root) client;
      set b (p P.h_object_parent) 0;
      set b (p P.h_object_old) obj)

let rm_answer kind body =
  let size, status, params_size =
    match kind with
    | `Alloc -> Gsp_defs.Rpc_rm_alloc.(sizeof, status, params_size)
    | `Control -> Gsp_defs.Rpc_rm_control.(sizeof, status, params_size)
  in
  if String.length body < size then Error "an RM answer shorter than its header"
  else
    let n = get body params_size in
    if size + n > String.length body then
      Error (strf "an RM answer of %d bytes of parameters, which it lacks" n)
    else Ok (get body status, String.sub body size n)

let page_directory ~client ~device ~vaspace ~root ~entries =
  let module S = Gsp_defs.Rpc_set_page_directory in
  let module P = Gsp_defs.Set_page_directory in
  let p f = at S.params f in
  record S.sizeof (fun b ->
      set b S.h_client client;
      set b S.h_device device;
      (* No PASID: the address space is the GPU's own. *)
      set b S.pasid 0xffff_ffff;
      set b (p P.phys_address) root;
      set b (p P.num_entries) entries;
      (* NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_VIDMEM: the root is
         in the GPU's memory. *)
      set b (p P.flags) 0x8;
      set b (p P.h_va_space) vaspace;
      set b (p P.pasid) 0xffff_ffff;
      set b (p P.sub_device_id) 1)

(* The GSP's range of an address space whose tables the process keeps: 512 MiB
   from 4 GiB (g_gpu_vaspace_nvoc.h's SPLIT_VAS_SERVER_RM_MANAGED_VA_START and
   _SIZE). The GSP computes the same range itself (gvaspaceReserveSplitVaSpace)
   and takes the tables over at its start (gvaspaceCopyServerReservedPdes). *)
let reserved = (1 lsl 32, 512 lsl 20)

(* GMMU_APERTURE_VIDEO: the tables are in the GPU's memory. *)
let aperture_video = 1

(* The bytes of an entry, dev_mmu.h's NV_MMU_VER2_PDE__SIZE and
   NV_MMU_VER2_DUAL_PDE__SIZE, the same on version 3. *)
let pde_size = 8
let dual_pde_size = 16

(* Each table, root first, as _gvaspacePopulatePDEentries describes it: its
   address, its bytes and the bit its level indexes. *)
let reserved_pdes v tables =
  let module R = Defs.Reserved_pdes in
  let shifts = Array.of_list (List.rev (Page_entry.levels v)) in
  let n = Array.length shifts - 1 in
  if List.length tables <> n then
    invalid_arg
      (strf "Rpc.reserved_pdes: %d tables, expected %d" (List.length tables) n);
  let first, size = reserved in
  let level (off, bytes, _) i (f, w) = (off + (i * bytes) + f, w) in
  record R.sizeof (fun b ->
      (* The lowest table is the dual level's, whose 16-byte entries map
         2 MiB each. *)
      set b R.page_size (1 lsl shifts.(n - 1));
      set b R.virt_addr_lo first;
      set b R.virt_addr_hi (first + size - 1);
      set b R.num_levels_to_copy n;
      List.iteri
        (fun i pa ->
          let f = level R.levels i in
          let above = if i = 0 then Page_entry.bits v else shifts.(i - 1) in
          let entry = if i = n - 1 then dual_pde_size else pde_size in
          set b (f R.levels_phys_address) pa;
          set b (f R.levels_size) ((1 lsl (above - shifts.(i))) * entry);
          set b (f R.levels_page_shift) shifts.(i);
          set b (f R.levels_aperture) aperture_video)
        tables)

(* Unloading to level 6, NV2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_3's bit (RM's
   unload to the deepest state the GSP keeps). *)
let unload_level = 1 lsl 6

let unloading =
  record Gsp_defs.Rpc_unloading.sizeof (fun b ->
      set b Gsp_defs.Rpc_unloading.new_level unload_level)

let registry keys =
  let module T = Gsp_defs.Registry_table in
  let module E = Gsp_defs.Registry_entry in
  let n = List.length keys in
  let names_at = T.sizeof + (n * E.sizeof) in
  let names = String.concat "" (List.map (fun (k, _) -> k ^ "\000") keys) in
  let entries =
    record (n * E.sizeof) (fun b ->
        ignore
          (List.fold_left
             (fun (i, name) (k, v) ->
               let f (off, w) = ((i * E.sizeof) + off, w) in
               set b (f E.name_offset) name;
               set b (f E.type_) Gsp_defs.registry_table_entry_type_dword;
               set b (f E.data) v;
               set b (f E.length) 4;
               (i + 1, name + String.length k + 1))
             (0, names_at) keys))
  in
  record T.sizeof (fun b ->
      set b T.size (names_at + String.length names);
      set b T.num_entries n)
  ^ entries ^ names

(* CPU sequences *)

(* The GSP's falcon restarted with its libos arguments at the bus address
   [libos], once SEC2 hands it over. *)
let resume ~libos =
  let module L = Defs.Legacy in
  let gsp = Falcon.gsp and sec2 = Falcon.sec2 in
  let handoff = L.nv_pgc6_bsi_secure_scratch_14_boot_stage_3_handoff in
  Falcon.reset gsp `Riscv
  @ [
      Falcon.Write (L.nv_pgsp_falcon_mailbox0, libos land 0xffff_ffff);
      Write (L.nv_pgsp_falcon_mailbox1, libos lsr 32);
    ]
  @ Falcon.start sec2
  @ [
      Poll
        ( "SEC2 to hand the GSP over",
          L.nv_pgc6_bsi_secure_scratch_14,
          mask handoff,
          Is
            (Defs.nv_pgc6_bsi_secure_scratch_14_boot_stage_3_handoff_value_done
            lsl fst handoff) );
      Expect
        ("SEC2 failed", sec2 + L.nv_pfalcon_falcon_mailbox0, 0xffff_ffff, Is 0);
    ]

(* The GSP's falcon reset, able to read physical memory without a context. *)
let core_reset () =
  let module L = Defs.Legacy in
  let gsp = Falcon.gsp in
  Falcon.reset gsp `Falcon
  @ [
      Modify
        ( gsp + L.nv_pfalcon_fbif_ctl,
          mask L.nv_pfalcon_fbif_ctl_allow_phys_no_ctx,
          1 lsl fst L.nv_pfalcon_fbif_ctl_allow_phys_no_ctx );
      Write (gsp + L.nv_pfalcon_falcon_dmactl, 0);
    ]

let sequence ~libos body =
  let module S = Gsp_defs.Rpc_cpu_sequencer in
  let base, _, _ = S.command_buffer in
  if String.length body < base then
    Error "a CPU sequence shorter than its header"
  else
    let n = get body S.cmd_index in
    if base + (4 * n) > String.length body then
      Error (strf "a CPU sequence of %d words, longer than its message" n)
    else
      let word i = get body (base + (4 * i), 4) in
      let rec go i acc =
        if i >= n then Ok (List.concat (List.rev acc))
        else
          let op = word i in
          (* A field of the payload after the opcode. *)
          let at (off, _) = word (i + 1 + (off / 4)) in
          let take size f =
            let k = size / 4 in
            if i + k >= n then
              Error (strf "a CPU sequence ending inside opcode %d" op)
            else go (i + k + 1) (f () :: acc)
          in
          if op = Gsp_defs.gsp_seq_buf_opcode_reg_write then
            let module P = Gsp_defs.Seq_reg_write in
            take P.sizeof (fun () -> [ Falcon.Write (at P.addr, at P.val_) ])
          else if op = Gsp_defs.gsp_seq_buf_opcode_reg_modify then
            let module P = Gsp_defs.Seq_reg_modify in
            (* A falcon modify writes the mask's bits only: the value's bits
               outside the mask join it. *)
            take P.sizeof (fun () ->
                let m = at P.mask and x = at P.val_ in
                [ Falcon.Modify (at P.addr, m lor x, x) ])
          else if op = Gsp_defs.gsp_seq_buf_opcode_reg_poll then
            let module P = Gsp_defs.Seq_reg_poll in
            (* The poll's timeout and error are the runner's. *)
            take P.sizeof (fun () ->
                let r = at P.addr in
                let x = at P.val_ in
                [ Falcon.Poll (strf "register 0x%x" r, r, at P.mask, Is x) ])
          else if op = Gsp_defs.gsp_seq_buf_opcode_delay_us then
            let module P = Gsp_defs.Seq_delay_us in
            take P.sizeof (fun () -> [ Falcon.Delay (at P.val_) ])
          else if op = Gsp_defs.gsp_seq_buf_opcode_reg_store then
            (* The GSP does not read its save area back from the CPU. *)
            take Gsp_defs.Seq_reg_store.sizeof (fun () -> [])
          else if op = Gsp_defs.gsp_seq_buf_opcode_core_reset then
            take 0 core_reset
          else if op = Gsp_defs.gsp_seq_buf_opcode_core_start then
            take 0 (fun () -> Falcon.start Falcon.gsp)
          else if op = Gsp_defs.gsp_seq_buf_opcode_core_wait_for_halt then
            take 0 (fun () -> Falcon.wait_halt Falcon.gsp)
          else if op = Gsp_defs.gsp_seq_buf_opcode_core_resume then
            take 0 (fun () -> resume ~libos)
          else Error (strf "a CPU sequence of unknown opcode %d" op)
      in
      go 0 []
