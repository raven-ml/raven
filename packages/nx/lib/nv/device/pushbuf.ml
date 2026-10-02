(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The runtime's own work on the GPU's channels: channel setup, local memory and
   copies, as methods in command segments of a ring of its own, which the
   channels' GPFIFOs point to. *)

module D = Nv_defs
module P = Params
module Mmio = Nx_device_support.Mmio

let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

(* Methods *)

(* The subchannels the engines are bound to. *)
let host = 0
let compute = 1
let copy_engine = 4

(* Incrementing methods of [subc] from [mthd], one per value. *)
let methods subc mthd vals =
  ((2 lsl 28) lor (List.length vals lsl 16) lor (subc lsl 13) lor (mthd lsr 2))
  :: vals

(* Waits until the 64-bit semaphore at [addr] is at least [v]. *)
let acquire addr v =
  methods host D.nvc56f_sem_addr_lo
    [
      lo32 addr;
      hi32 addr;
      lo32 v;
      hi32 v;
      P.bits D.nvc56f_sem_execute_operation
        D.nvc56f_sem_execute_operation_acq_circ_geq
      lor P.bits D.nvc56f_sem_execute_payload_size
            D.nvc56f_sem_execute_payload_size_64bit;
    ]

(* Writes the 64-bit [v] at [addr] once the channel's earlier work is done. *)
let release addr v =
  methods host D.nvc56f_sem_addr_lo
    [
      lo32 addr;
      hi32 addr;
      lo32 v;
      hi32 v;
      P.bits D.nvc56f_sem_execute_operation
        D.nvc56f_sem_execute_operation_release
      lor P.bits D.nvc56f_sem_execute_release_wfi
            D.nvc56f_sem_execute_release_wfi_en
      lor P.bits D.nvc56f_sem_execute_payload_size
            D.nvc56f_sem_execute_payload_size_64bit;
    ]
  @ methods host D.nvc56f_non_stall_interrupt [ 0 ]

(* The copy engine: a copy of [n] bytes, in lines of at most 2 GiB. *)
let copy ~dst ~src n =
  let step = 1 lsl 31 in
  let launch =
    P.bits D.nvc6b5_launch_dma_data_transfer_type
      D.nvc6b5_launch_dma_data_transfer_type_non_pipelined
    lor P.bits D.nvc6b5_launch_dma_src_memory_layout
          D.nvc6b5_launch_dma_src_memory_layout_pitch
    lor P.bits D.nvc6b5_launch_dma_dst_memory_layout
          D.nvc6b5_launch_dma_dst_memory_layout_pitch
  in
  let rec go off acc =
    if off >= n then List.concat (List.rev acc)
    else
      let s = src + off and d = dst + off in
      go (off + step)
        ((methods copy_engine D.nvc6b5_offset_in_upper
            [ hi32 s; lo32 s; hi32 d; lo32 d ]
         @ methods copy_engine D.nvc6b5_line_length_in
             [ Int.min step (n - off) ]
         @ methods copy_engine D.nvc6b5_launch_dma [ launch ])
        :: acc)
  in
  go 0 []

(* The copy engine writes 32 bits per semaphore, so the 64-bit [v] goes as its
   low word, then its high word when the low one wrapped to 0: mid-write, the
   word reads no higher than before, so no wait passes early. A high word
   written late carries the value every later writer has too, so it never takes
   the word back, as rewriting it on every release would once another channel's
   release lands between the two words. *)
let copy_release addr v =
  let one a w =
    methods copy_engine D.nvc6b5_set_semaphore_a [ hi32 a; lo32 a; w ]
    @ methods copy_engine D.nvc6b5_launch_dma
        [
          P.bits D.nvc6b5_launch_dma_flush_enable
            D.nvc6b5_launch_dma_flush_enable_true
          lor P.bits D.nvc6b5_launch_dma_semaphore_type
                D.nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore;
        ]
  in
  one addr (lo32 v) @ if lo32 v = 0 then one (addr + 4) (hi32 v) else []

(* The copy engine's timestamp: a four-word semaphore release at the 16 bytes at
   [addr] writes a payload of 0 into the first 8 and the GPU timer, in
   nanoseconds, into the last 8. *)
let copy_stamp addr =
  methods copy_engine D.nvc6b5_set_semaphore_a [ hi32 addr; lo32 addr; 0 ]
  @ methods copy_engine D.nvc6b5_launch_dma
      [
        P.bits D.nvc6b5_launch_dma_flush_enable
          D.nvc6b5_launch_dma_flush_enable_true
        lor P.bits D.nvc6b5_launch_dma_semaphore_type
              D.nvc6b5_launch_dma_semaphore_type_release_four_word_semaphore;
      ]

(* Channels *)

type channel = {
  ring : Mmio.t;
  entries : int;
  gp_get : Mmio.t;
  gp_put : Mmio.t;
  put : Mmio.t; (* the 64-bit count of entries written *)
  doorbell : Mmio.t;
  token : int;
}

(* The GPFIFO entry of the segment of [words] words at [addr]. *)
let entry addr words =
  let e0 = P.bits D.nvc56f_gp_entry0_get (addr lsr 2) in
  let e1 =
    P.bits D.nvc56f_gp_entry1_get_hi (addr lsr 32)
    lor P.bits D.nvc56f_gp_entry1_level D.nvc56f_gp_entry1_level_subroutine
    lor P.bits D.nvc56f_gp_entry1_length words
  in
  Int64.logor (Int64.of_int e0) (Int64.shift_left (Int64.of_int e1) 32)

(* Appends the segment of [words] words at [addr] to [ch], after waiting until
   the GPU has fetched enough of its entries to make room, for at most
   [timeout_ms]. *)
let submit ch ~timeout_ms addr words =
  if addr >= 1 lsl 40 then failwith "a command segment above 2^40";
  let put = Int64.to_int (Mmio.get64 ch.put 0) in
  let unfetched () =
    (put - Mmio.get32 ch.gp_get 0 + ch.entries) mod ch.entries
  in
  Nvdev.wait_until ~timeout_ms "room in the GPU channel" (fun () ->
      unfetched () < ch.entries - 1);
  Mmio.set64 ch.ring (8 * (put mod ch.entries)) (entry addr words);
  Mmio.barrier ();
  Mmio.set64 ch.put 0 (Int64.of_int (put + 1));
  Mmio.set32 ch.gp_put 0 ((put + 1) mod ch.entries);
  Mmio.barrier ();
  Mmio.set32 ch.doorbell 0 ch.token

(* The runtime's segments *)

type ring = {
  mem : Mmio.t; (* as the host writes it *)
  gpu : int; (* its address for the GPU *)
  mutable head : int;
  mutable tags : (int * int * int) list; (* (start, end, value) of segments *)
}

let ring mem ~gpu = { mem; gpu; head = 0; tags = [] }

(* Writes [words] into a new segment for the work of timeline value [v], and is
   its address. The segment's bytes are reused once the work of every segment
   that held them has signaled, as [signaled] reads: values complete in order,
   so waiting for the latest is enough. *)
let segment r ~signaled ~timeout_ms v words =
  let n = 4 * List.length words in
  if n > Mmio.length r.mem then
    failwith "a command segment larger than the ring";
  if r.head + n > Mmio.length r.mem then r.head <- 0;
  let start = r.head and stop = r.head + n in
  let last =
    List.fold_left
      (fun m (s, e, tv) -> if s < stop && start < e then Int.max m tv else m)
      0 r.tags
  in
  Nvdev.wait_until ~timeout_ms "a command segment the GPU still reads"
    (fun () -> signaled () >= last);
  r.tags <- (start, stop, v) :: List.filter (fun (_, _, tv) -> tv > last) r.tags;
  List.iteri (fun i w -> Mmio.set32 r.mem (start + (4 * i)) w) words;
  r.head <- stop;
  r.gpu + start
