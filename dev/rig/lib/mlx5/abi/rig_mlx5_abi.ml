(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

type buffer =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let strf = Printf.sprintf

(* Big-endian fields *)

external get16 : buffer -> int -> int = "%caml_bigstring_get16"
external get32 : buffer -> int -> int32 = "%caml_bigstring_get32"
external set32 : buffer -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : buffer -> int -> int64 -> unit = "%caml_bigstring_set64"
external swap16 : int -> int = "%bswap16"
external swap32 : int32 -> int32 = "%bswap_int32"
external swap64 : int64 -> int64 = "%bswap_int64"

let be16 b at = if Sys.big_endian then get16 b at else swap16 (get16 b at)

let be32 b at =
  let v = get32 b at in
  Int32.to_int (if Sys.big_endian then v else swap32 v) land 0xffff_ffff

let byte b at = Char.code (Bigarray.Array1.get b at)
let set_byte b at v = Bigarray.Array1.set b at (Char.unsafe_chr v)

let set_be32 b at v =
  let v = Int32.of_int v in
  set32 b at (if Sys.big_endian then v else swap32 v)

let set_be64 b at v =
  let v = Int64.of_int v in
  set64 b at (if Sys.big_endian then v else swap64 v)

(* A field's byte offset, from a layout's (offset, bytes). *)
let off = fst

let check_bytes fn b at n =
  if at < 0 || at + n > Bigarray.Array1.dim b then
    invalid_arg
      (strf "Rig_mlx5_abi.%s: bytes %d to %d outside the buffer" fn at (at + n))

let in_range fn what lo hi v =
  if v < lo || v > hi then
    invalid_arg (strf "Rig_mlx5_abi.%s: %s %d not in [%d;%d]" fn what v lo hi)

let is_pow2 n = n > 0 && n land (n - 1) = 0

module Entry = struct
  let size = D.mlx5_send_wqe_bb

  type local = { address : int; bytes : int; key : int }
  type remote = { address : int; key : int }

  type op =
    | Write of { src : local; dst : remote }
    | Write_inline of { data : string; dst : remote }
    | Read of { src : remote; dst : local }

  type t = { op : op; signal : bool }

  let ctrl = 0
  let raddr = ctrl + D.Ctrl_seg.sizeof
  let data = raddr + D.Raddr_seg.sizeof

  (* An entry's data starts after its inline segment's 4-byte header. *)
  let max_inline = size - data - D.Inline_seg.sizeof

  (* The 16-byte units of an entry, which its control segment counts. *)
  let unit = 16
  let max_key = 0xffff_ffff
  let max_length = 0x7fff_ffff

  let check_remote (r : remote) =
    in_range "Entry.write" "address" 0 max_int r.address;
    in_range "Entry.write" "key" 0 max_key r.key

  let check_local (l : local) =
    in_range "Entry.write" "address" 0 max_int l.address;
    in_range "Entry.write" "key" 0 max_key l.key;
    in_range "Entry.write" "length" 0 max_length l.bytes

  let write_remote b at (r : remote) =
    set_be64 b (at + off D.Raddr_seg.raddr) r.address;
    set_be32 b (at + off D.Raddr_seg.rkey) r.key

  let write_local b at (l : local) =
    set_be32 b (at + off D.Data_seg.byte_count) l.bytes;
    set_be32 b (at + off D.Data_seg.lkey) l.key;
    set_be64 b (at + off D.Data_seg.addr) l.address

  let write b at ~qp ~index e =
    check_bytes "Entry.write" b at size;
    if at mod size <> 0 then
      invalid_arg
        (strf "Rig_mlx5_abi.Entry.write: byte %d is not a multiple of %d" at
           size);
    in_range "Entry.write" "queue pair" 0 0xff_ffff qp;
    in_range "Entry.write" "index" 0 max_int index;
    (match e.op with
    | Write { src = l; dst = r } | Read { src = r; dst = l } ->
        check_local l;
        check_remote r
    | Write_inline { data = s; dst } ->
        if String.length s > max_inline then
          invalid_arg
            (strf "Rig_mlx5_abi.Entry.write: %d inline bytes, more than %d"
               (String.length s) max_inline);
        check_remote dst);
    for i = 0 to (size / 8) - 1 do
      set64 b (at + (8 * i)) 0L
    done;
    let opcode, bytes =
      match e.op with
      | Write { src; dst } ->
          write_remote b (at + raddr) dst;
          write_local b (at + data) src;
          (D.mlx5_opcode_rdma_write, data + D.Data_seg.sizeof)
      | Write_inline { data = s; dst } ->
          let n = String.length s in
          write_remote b (at + raddr) dst;
          set_be32 b
            (at + data + off D.Inline_seg.byte_count)
            (n lor D.mlx5_inline_seg);
          let start = at + data + D.Inline_seg.sizeof in
          String.iteri (fun i c -> Bigarray.Array1.set b (start + i) c) s;
          (D.mlx5_opcode_rdma_write, data + D.Inline_seg.sizeof + n)
      | Read { src; dst } ->
          write_remote b (at + raddr) src;
          write_local b (at + data) dst;
          (D.mlx5_opcode_rdma_read, data + D.Data_seg.sizeof)
    in
    let units = (bytes + unit - 1) / unit in
    set_be32 b
      (at + off D.Ctrl_seg.opmod_idx_opcode)
      (((index land 0xffff) lsl 8) lor opcode);
    set_be32 b (at + off D.Ctrl_seg.qpn_ds) ((qp lsl 8) lor units);
    set_byte b
      (at + off D.Ctrl_seg.fm_ce_se)
      (if e.signal then D.mlx5_wqe_ctrl_cq_update else 0)
end

module Completion = struct
  let size = D.Cqe64.sizeof

  type error =
    | Local_length
    | Local_qp_operation
    | Local_protection
    | Flushed
    | Bad_response
    | Local_access
    | Remote_invalid_request
    | Remote_access
    | Remote_operation
    | Retry_exceeded
    | Remote_aborted
    | Other_error of int

  type status =
    | Done
    | Failed of { error : error; vendor : int }
    | Unexpected of int

  type t = { qp : int; index : int; status : status }

  let errors =
    [
      (D.mlx5_cqe_syndrome_local_length_err, Local_length);
      (D.mlx5_cqe_syndrome_local_qp_op_err, Local_qp_operation);
      (D.mlx5_cqe_syndrome_local_prot_err, Local_protection);
      (D.mlx5_cqe_syndrome_wr_flush_err, Flushed);
      (D.mlx5_cqe_syndrome_bad_resp_err, Bad_response);
      (D.mlx5_cqe_syndrome_local_access_err, Local_access);
      (D.mlx5_cqe_syndrome_remote_inval_req_err, Remote_invalid_request);
      (D.mlx5_cqe_syndrome_remote_access_err, Remote_access);
      (D.mlx5_cqe_syndrome_remote_op_err, Remote_operation);
      (D.mlx5_cqe_syndrome_transport_retry_exc_err, Retry_exceeded);
      (D.mlx5_cqe_syndrome_remote_aborted_err, Remote_aborted);
    ]

  let error_of s =
    match List.assoc_opt s errors with Some e -> e | None -> Other_error s

  let opcode op_own = op_own lsr 4

  let owned b at ~count ~entries =
    check_bytes "Completion.owned" b at size;
    if count < 0 then
      invalid_arg
        (strf "Rig_mlx5_abi.Completion.owned: count %d is negative" count);
    if not (is_pow2 entries) then
      invalid_arg
        (strf "Rig_mlx5_abi.Completion.owned: %d entries is not a power of two"
           entries);
    let op_own = byte b (at + off D.Cqe64.op_own) in
    opcode op_own <> D.mlx5_cqe_invalid
    && op_own land D.mlx5_cqe_owner_mask = Bool.to_int (count land entries <> 0)

  let read b at =
    check_bytes "Completion.read" b at size;
    let qp = be32 b (at + off D.Cqe64.sop_drop_qpn) land 0xff_ffff in
    let index = be16 b (at + off D.Cqe64.wqe_counter) in
    let op = opcode (byte b (at + off D.Cqe64.op_own)) in
    let status =
      if op = D.mlx5_cqe_req then Done
      else if op = D.mlx5_cqe_req_err || op = D.mlx5_cqe_resp_err then
        Failed
          {
            error = error_of (byte b (at + off D.Err_cqe.syndrome));
            vendor = byte b (at + off D.Err_cqe.vendor_err_synd);
          }
      else Unexpected op
    in
    { qp; index; status }

  let invalidate b at =
    check_bytes "Completion.invalidate" b at size;
    set_byte b (at + off D.Cqe64.op_own) (D.mlx5_cqe_invalid lsl 4)

  let describe = function
    | Local_length -> "local length error"
    | Local_qp_operation -> "local queue pair operation error"
    | Local_protection -> "local protection error"
    | Flushed -> "flushed"
    | Bad_response -> "bad response"
    | Local_access -> "local access error"
    | Remote_invalid_request -> "remote invalid request"
    | Remote_access -> "remote access error"
    | Remote_operation -> "remote operation error"
    | Retry_exceeded -> "retry counter exceeded"
    | Remote_aborted -> "remote aborted"
    | Other_error s -> strf "syndrome 0x%x" s

  let message c =
    let what =
      match c.status with
      | Done -> "completed"
      | Failed { error; vendor } ->
          strf "%s (vendor syndrome 0x%x)" (describe error) vendor
      | Unexpected op -> strf "unexpected completion opcode 0x%x" op
    in
    strf "queue pair 0x%x, entry %d: %s" c.qp c.index what
end

module Doorbell = struct
  let word = 4
  let size = 2 * word
  let send = D.mlx5_snd_dbr * word
  let consumed = D.mlx5_cq_set_ci * word
  let armed = D.mlx5_cq_arm_db * word

  (* The arm word: the sequence's low 2 bits from bit 28, the command, then the
     consumer count's low 24 bits. *)
  let arm ~sequence ~count ~cq =
    in_range "Doorbell.arm" "sequence" 0 max_int sequence;
    in_range "Doorbell.arm" "count" 0 max_int count;
    in_range "Doorbell.arm" "completion queue" 0 0xff_ffff cq;
    ( ((sequence land 3) lsl 28)
      lor D.mlx5_cq_db_req_not lor (count land 0xff_ffff),
      cq )
end

module Uar = struct
  let per_region = D.mlx5_bfregs_per_uar

  let mapping ~page i =
    in_range "Uar.mapping" "page" 0 255 i;
    if not (is_pow2 page) then
      invalid_arg (strf "Rig_mlx5_abi.Uar.mapping: page size %d" page);
    (D.mlx5_ib_mmap_nc_page lsl D.mlx5_ib_mmap_cmd_shift) lor i * page

  let register ~per_page ~size r =
    in_range "Uar.register" "register" 0 max_int r;
    if r mod per_region >= D.mlx5_non_fp_bfregs_per_uar then
      invalid_arg
        (strf "Rig_mlx5_abi.Uar.register: register %d is never rung" r);
    if per_page < 1 then
      invalid_arg
        (strf "Rig_mlx5_abi.Uar.register: %d regions per page" per_page);
    if size < 0 then
      invalid_arg (strf "Rig_mlx5_abi.Uar.register: size %d" size);
    let region = r / per_region in
    ( region / per_page,
      (region mod per_page * D.mlx5_adapter_page_size)
      + D.mlx5_bf_offset
      + (r mod per_region * size) )

  let cq_doorbell = D.mlx5_cq_doorbell
end
