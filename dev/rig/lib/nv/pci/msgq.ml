(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_pci

let strf = Printf.sprintf

module E = Defs.Queue_element
module R = Defs.Rpc_header
module H = Defs.Msgq_tx_header

(* Records *)

(* The size of an element, which the process sets for both queues. *)
let element_size = 0x1000

(* A record's elements hold the element's header up to its RPC header, then the
   RPC header and the body. *)
let element_header = fst E.rpc
let body_at = element_header + R.sizeof

(* The most elements a record takes (message_queue_cpu.c's
   GSP_MSG_QUEUE_RECORD_MAX = 16). *)
let record_elements = 16
let record_max = (element_size * record_elements) - body_at

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

let records fn body =
  let n = String.length body in
  let rec go acc off =
    if off >= n then List.rev acc
    else
      let len = Int.min record_max (n - off) in
      go
        ((Defs.nv_vgpu_msg_function_continuation_record, String.sub body off len)
        :: acc)
        (off + len)
  in
  let first = Int.min record_max n in
  (fn, String.sub body 0 first) :: go [] first

let set b (off, n) x =
  match n with
  | 4 -> Bytes.set_int32_le b off (Int32.of_int x)
  | _ -> Bytes.set_int64_le b off (Int64.of_int x)

let element ~seq fn body =
  let n = body_at + String.length body in
  let count = (n + element_size - 1) / element_size in
  let b = Bytes.make (count * element_size) '\000' in
  let header = element_header in
  set b E.seq_num seq;
  set b E.elem_count count;
  let version =
    Defs.nv_vgpu_msg_header_version_major_tot
    lsl fst Defs.nv_vgpu_msg_header_version_major
    lor Defs.nv_vgpu_msg_header_version_minor_tot
        lsl fst Defs.nv_vgpu_msg_header_version_minor
  in
  let rpc (off, n) = (header + off, n) in
  set b (rpc R.header_version) version;
  set b (rpc R.signature) Defs.nv_vgpu_msg_signature_valid;
  set b (rpc R.length) (R.sizeof + String.length body);
  set b (rpc R.function_) fn;
  set b (rpc R.rpc_result) Defs.nv_vgpu_msg_result_rpc_pending;
  set b (rpc R.rpc_result_private) Defs.nv_vgpu_msg_result_rpc_pending;
  Bytes.blit_string body 0 b body_at (String.length body);
  set b E.check_sum (checksum (Bytes.unsafe_to_string b));
  Bytes.unsafe_to_string b

type message = { fn : int; result : int; body : string }

let get s (off, _) = Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

let message s =
  if String.length s < body_at then
    Error "a GSP message shorter than its header"
  else
    let rpc (off, n) = (element_header + off, n) in
    let count = get s E.elem_count in
    let len = get s (rpc R.length) - R.sizeof in
    if get s (rpc R.signature) <> Defs.nv_vgpu_msg_signature_valid then
      Error
        (strf "a GSP message without the RPC signature, 0x%x"
           (get s (rpc R.signature)))
    else if len < 0 || body_at + len > String.length s || count < 1 then
      Error (strf "a GSP message of %d elements longer than them" count)
    else
      Ok
        ( {
            fn = get s (rpc R.function_);
            result = get s (rpc R.rpc_result);
            body = String.sub s body_at len;
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

(* Queues *)

type queue = {
  w : Window.t; (* its header, then its elements *)
  count : int; (* its elements *)
}

type t = {
  cmd : queue;
  stat_window : Window.t;
  doorbell : Window.t;
  mutable stat : queue option;
  mutable cmd_read : Window.t option; (* the GSP's position in [cmd] *)
  mutable stat_read : Window.t option; (* the process's position in [stat] *)
  mutable seq : int;
}

(* The elements start after a page of header. *)
let entries_at = 0x1000

let create w ~doorbell =
  let size = Window.length w / 2 in
  let cmd = Window.sub w 0 size and stat_window = Window.sub w size size in
  let count = (size - entries_at) / element_size in
  let h = Bytes.make H.sizeof '\000' in
  set h H.size size;
  set h H.msg_size element_size;
  set h H.msg_count count;
  set h H.entry_off entries_at;
  set h H.flags 1;
  set h H.rx_hdr_off H.sizeof;
  Window.write cmd 0 (Bytes.unsafe_to_string h);
  {
    cmd = { w = cmd; count };
    stat_window;
    doorbell;
    stat = None;
    cmd_read = None;
    stat_read = None;
    seq = 0;
  }

let header w f = Window.get32 w (fst f)

let ready q =
  match q.stat with
  | Some _ -> true
  | None ->
      Window.barrier ();
      if header q.stat_window H.entry_off <> entries_at then false
      else begin
        (* Each reader keeps its position in the other queue's header page. *)
        q.stat_read <- Some (Window.sub q.cmd.w (header q.cmd.w H.rx_hdr_off) 4);
        q.cmd_read <-
          Some (Window.sub q.stat_window (header q.stat_window H.rx_hdr_off) 4);
        q.stat <-
          Some { w = q.stat_window; count = header q.stat_window H.msg_count };
        true
      end

(* Reading and writing [n] bytes at element [e] of [q], across its end. *)
let ring_write q e s =
  let off = entries_at + (e * element_size) in
  let first = Int.min (String.length s) ((q.count - e) * element_size) in
  Window.blit_string s 0 q.w off first;
  if first < String.length s then
    Window.blit_string s first q.w entries_at (String.length s - first)

let ring_read q e n =
  let off = entries_at + (e * element_size) in
  let first = Int.min n ((q.count - e) * element_size) in
  let a = Window.read q.w off first in
  if first < n then a ^ Window.read q.w entries_at (n - first) else a

let pending q read =
  (header q.w H.write_ptr - Window.get32 read 0 + q.count) mod q.count

let send q fn body =
  let elements = records fn body in
  let counts =
    List.map
      (fun (_, b) ->
        (body_at + String.length b + element_size - 1) / element_size)
      elements
  in
  let used = match q.cmd_read with None -> 0 | Some r -> pending q.cmd r in
  if used + List.fold_left ( + ) 0 counts >= q.cmd.count then false
  else begin
    List.iter
      (fun (f, b) ->
        let e = element ~seq:q.seq f b in
        let wp = header q.cmd.w H.write_ptr in
        ring_write q.cmd wp e;
        Window.barrier ();
        Window.set32 q.cmd.w (fst H.write_ptr)
          ((wp + (String.length e / element_size)) mod q.cmd.count);
        Window.barrier ();
        q.seq <- q.seq + 1;
        Window.set32 q.doorbell 0 0)
      elements;
    true
  end

let receive q =
  match (q.stat, q.stat_read) with
  | Some stat, Some read ->
      Window.barrier ();
      let rp = Window.get32 read 0 in
      if rp = header stat.w H.write_ptr then None
      else begin
        let head = ring_read stat rp body_at in
        let count = get head E.elem_count in
        let r =
          if count < 1 || count >= stat.count then
            Error
              (strf "a GSP message of %d elements, in a queue of %d" count
                 stat.count)
          else
            Result.map fst (message (ring_read stat rp (count * element_size)))
        in
        (* A message the process cannot read is consumed whole, or the queue's
           one element when its count is out of range. *)
        let count = if count < 1 || count >= stat.count then 1 else count in
        Window.barrier ();
        Window.set32 read 0 ((rp + count) mod stat.count);
        Window.barrier ();
        Some r
      end
  | _ -> invalid_arg "Msgq.receive: the GSP has not started its queue"
