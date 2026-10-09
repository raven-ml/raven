(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_pci
open Field

let strf = Printf.sprintf

module H = Defs.Msgq_tx_header

let element_size = Rpc.element_size

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
  let records = Rpc.records ~seq:q.seq fn body in
  let counts = List.map (fun e -> String.length e / element_size) records in
  let used = match q.cmd_read with None -> 0 | Some r -> pending q.cmd r in
  if used + List.fold_left ( + ) 0 counts >= q.cmd.count then false
  else begin
    List.iter
      (fun e ->
        let wp = header q.cmd.w H.write_ptr in
        ring_write q.cmd wp e;
        Window.barrier ();
        Window.set32 q.cmd.w (fst H.write_ptr)
          ((wp + (String.length e / element_size)) mod q.cmd.count);
        Window.barrier ();
        q.seq <- q.seq + 1;
        Window.set32 q.doorbell 0 0)
      records;
    true
  end

let receive q =
  match (q.stat, q.stat_read) with
  | Some stat, Some read ->
      Window.barrier ();
      let rp = Window.get32 read 0 in
      if rp = header stat.w H.write_ptr then None
      else begin
        let count = Rpc.elements (ring_read stat rp Rpc.header_size) in
        let r =
          if count < 1 || count >= stat.count then
            Error
              (strf "a GSP message of %d elements, in a queue of %d" count
                 stat.count)
          else
            Result.map fst
              (Rpc.message (ring_read stat rp (count * element_size)))
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
