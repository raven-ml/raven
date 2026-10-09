(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Field

let strf = Printf.sprintf

module E = Defs.Queue_element
module R = Defs.Rpc_header

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
    Defs.nv_vgpu_msg_header_version_major_tot
    lsl fst Defs.nv_vgpu_msg_header_version_major
    lor Defs.nv_vgpu_msg_header_version_minor_tot
        lsl fst Defs.nv_vgpu_msg_header_version_minor
  in
  let rpc (off, n) = (element_header + off, n) in
  set b (rpc R.header_version) version;
  set b (rpc R.signature) Defs.nv_vgpu_msg_signature_valid;
  set b (rpc R.length) (R.sizeof + String.length body);
  set b (rpc R.function_) fn;
  set b (rpc R.rpc_result) Defs.nv_vgpu_msg_result_rpc_pending;
  set b (rpc R.rpc_result_private) Defs.nv_vgpu_msg_result_rpc_pending;
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
        element ~seq Defs.nv_vgpu_msg_function_continuation_record
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
    if get s (rpc R.signature) <> Defs.nv_vgpu_msg_signature_valid then
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
  if m.fn = Defs.nv_vgpu_msg_event_rc_triggered then
    let module T = Defs.Rpc_rc_triggered in
    let b = m.body in
    if String.length b < T.sizeof then Some "the GSP stopped a channel"
    else
      let e = get b T.except_type in
      let name =
        Option.value ~default:"an unknown error"
          (List.assoc_opt e Defs.robust_channel_errors)
      in
      Some
        (strf "the GSP stopped channel %d: %s (Xid %d)" (get b T.chid) name e)
  else if m.fn = Defs.nv_vgpu_msg_event_mmu_fault_queued then
    Some "the GPU's MMU queued a fault"
  else None

(* Calls *)

let rm_alloc ~client ~parent ~obj ~cls params =
  let module A = Defs.Rpc_rm_alloc in
  record A.sizeof (fun b ->
      set b A.h_client client;
      set b A.h_parent parent;
      set b A.h_object obj;
      set b A.h_class cls;
      set b A.params_size (String.length params))
  ^ params

let rm_control ~client ~obj ~cmd params =
  let module C = Defs.Rpc_rm_control in
  record C.sizeof (fun b ->
      set b C.h_client client;
      set b C.h_object obj;
      set b C.cmd cmd;
      set b C.params_size (String.length params))
  ^ params

let rm_answer kind body =
  let size, status, params_size =
    match kind with
    | `Alloc -> Defs.Rpc_rm_alloc.(sizeof, status, params_size)
    | `Control -> Defs.Rpc_rm_control.(sizeof, status, params_size)
  in
  if String.length body < size then Error "an RM answer shorter than its header"
  else
    let n = get body params_size in
    if size + n > String.length body then
      Error (strf "an RM answer of %d bytes of parameters, which it lacks" n)
    else Ok (get body status, String.sub body size n)

let page_directory ~client ~device ~vaspace ~root ~entries =
  let module S = Defs.Rpc_set_page_directory in
  let module P = Defs.Set_page_directory in
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

(* Unloading to level 6, NV2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_3's bit (RM's
   unload to the deepest state the GSP keeps). *)
let unload_level = 1 lsl 6

let unloading =
  record Defs.Rpc_unloading.sizeof (fun b ->
      set b Defs.Rpc_unloading.new_level unload_level)

let registry keys =
  let module T = Defs.Registry_table in
  let module E = Defs.Registry_entry in
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
               set b (f E.type_) Defs.registry_table_entry_type_dword;
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

type step =
  | Write of int * int
  | Modify of int * int * int
  | Poll of int * int * int
  | Delay_us of int
  | Store of int * int
  | Core_reset
  | Core_start
  | Core_wait_for_halt
  | Core_resume

let sequence body =
  let module S = Defs.Rpc_cpu_sequencer in
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
        if i >= n then Ok (List.rev acc)
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
          if op = Defs.gsp_seq_buf_opcode_reg_write then
            let module P = Defs.Seq_reg_write in
            take P.sizeof (fun () -> Write (at P.addr, at P.val_))
          else if op = Defs.gsp_seq_buf_opcode_reg_modify then
            let module P = Defs.Seq_reg_modify in
            take P.sizeof (fun () -> Modify (at P.addr, at P.mask, at P.val_))
          else if op = Defs.gsp_seq_buf_opcode_reg_poll then
            let module P = Defs.Seq_reg_poll in
            take P.sizeof (fun () -> Poll (at P.addr, at P.mask, at P.val_))
          else if op = Defs.gsp_seq_buf_opcode_delay_us then
            let module P = Defs.Seq_delay_us in
            take P.sizeof (fun () -> Delay_us (at P.val_))
          else if op = Defs.gsp_seq_buf_opcode_reg_store then
            let module P = Defs.Seq_reg_store in
            take P.sizeof (fun () -> Store (at P.addr, at P.index))
          else if op = Defs.gsp_seq_buf_opcode_core_reset then
            take 0 (fun () -> Core_reset)
          else if op = Defs.gsp_seq_buf_opcode_core_start then
            take 0 (fun () -> Core_start)
          else if op = Defs.gsp_seq_buf_opcode_core_wait_for_halt then
            take 0 (fun () -> Core_wait_for_halt)
          else if op = Defs.gsp_seq_buf_opcode_core_resume then
            take 0 (fun () -> Core_resume)
          else Error (strf "a CPU sequence of unknown opcode %d" op)
      in
      go 0 []
