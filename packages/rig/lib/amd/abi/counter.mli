(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Performance counters, which count events of a kernel's run.

    A counter is an event of one block of a GPU's GC, such as ["SQ_WAVES"]
    or ["GRBM_GUI_ACTIVE"], which one of the block's counter registers
    selects. A run counts it in each instance of its block, and an SQ
    counter in each shader engine and, after GFX9, in each shader array and
    work-group processor of the engine. The run writes each value as a
    64-bit word into its samples. For the counters a profile asks for,
    {!val-layout} gives the register each takes and where its values lie,
    which the programs that count ({!section-counting}) write and the host
    reads.

    The counters are rocprofiler's for gfx942, gfx950, and GFX11 and GFX12
    GPUs. A GPU of another processor counts none. *)

type t = {
  name : string;  (** Its name, such as ["SQ_WAVES"]. *)
  block : string;  (** Its block: ["GRBM"], ["GL2C"], ["TCC"] or ["SQ"]. *)
  event : int;  (** The event its block's counter register selects. *)
  register : int;
      (** Which of its block's counter registers counts it. The counters of
          a block take its registers in the profile's order, from [0]. *)
  instances : int;  (** The instances of its block in one die. *)
  engines : int;  (** The shader engines it is counted in. *)
  arrays : int;  (** The shader arrays of an engine it is counted in. *)
  wgps : int;
      (** The work-group processors of an array it is counted in: every one
          the array is built with ({!Gpu.t.array_units}), those that run no
          work included. *)
  offset : int;  (** The byte offset of its values in a run's samples. *)
}
(** The type for a counter as a run counts it. Its values are
    [xccs * instances * engines * arrays * wgps] 64-bit words, one per die,
    instance, engine, array and work-group processor, the last varying
    fastest. *)

type layout = {
  counters : t list;  (** The counters, in the profile's order. *)
  bytes : int;  (** The bytes of a run's samples. *)
}
(** The type for a run's samples: each counter's values, one after
    another. *)

val names : Gpu.t -> string list
(** [names g] is the counters [g] counts, in increasing order. *)

val layout : Gpu.t -> string list -> (layout, string) result
(** [layout g names] is the samples of a run of [g] counting [names], in
    order. The result is [Error msg]:
    {ul
    {- if [g] counts no counter of [names], naming it, as
       ["gfx1201 counts no SQ_FOO"];}
    {- if [names] lists a counter twice, naming it, as
       ["SQ_WAVES is listed twice"];}
    {- if [names] has more counters of a block than the block has counter
       registers, naming the block, as
       ["gfx1201 counts at most 4 GL2C counters at once"].}} *)

(** {1:counting Counting}

    A queue counts the SQ counters of a layout over the work between the
    packets of {!val-start} and those of {!val-stop}, which a submission
    places on the queue ([Rig.Submission.words]). The GPU has one set of
    counters, which all its queues and processes share.

    Counting is filtered by VMID. A process's queues on a GPU share one
    VMID, which a wave reads in bits 27 to 24 of its [HW_ID2] register and
    which can change while the process runs. The filter counts the waves of
    every queue of the process whose VMID it holds. Whether a wave counts is
    decided when it starts, and holds for its life.

    The counts are exact when the filter word holds the process's VMID while
    it counts, the process's other queues run nothing while it counts, and
    no other process counts on the GPU. A preemption of the
    queue while it counts adds the waves its restore relaunches and the
    instructions of their trap handler, so ["SQ_WAVES"] exceeds the waves
    the queue launched.

    Each work-group processor's value saturates at [2{^32} - 1], so a value
    of [2{^32} - 1] means at least that many.

    The counters count only in the GPU's stable power state, which
    {!Capability.t.trace} holds. Outside it, ["SQ_INSTS_VALU"] stays [0]. *)

val filter : Gpu.t -> vmid:int -> int
(** [filter g ~vmid] is the 32-bit filter word that counts the waves of VMID
    [vmid] alone: [1 lor (1 lsl (vmid + 1))]. A kernel of the process may
    compute it from its own [HW_ID2] and store it for {!val-start}.

    Raises [Invalid_argument] if [vmid] is not in \[[0];[15]\], or if [g] is
    a GPU other than a GFX12 of one die. *)

val start : Gpu.t -> layout -> filter_address:int -> string
(** [start g l ~filter_address] is the packets, as little-endian 32-bit words, that
    wait until the queue's dispatches before them complete, reset every
    counter of [g], select [l]'s, and start counting the waves that the
    filter word at [filter_address] selects ({!val-filter}). They read that word
    once those dispatches complete, so a dispatch before them may write it.
    Addresses are as the device's work addresses memory
    ([Rig.Submission.address]).

    Raises [Invalid_argument] if [filter_address] is not a non-negative
    multiple of 4, if [l] holds a counter of a block other than SQ, if [g] is a GPU
    other than a GFX12 of one die, or if [g]'s GC lacks a register the
    program writes. *)

val stop : Gpu.t -> wgps:int array array -> layout -> address:int -> string
(** [stop g ~wgps l ~address] is the packets, as little-endian 32-bit
    words, that wait until the queue's dispatches before them complete,
    stop counting, and store [l]'s values from [address] as {!t} lays them
    out, [0] for each work-group processor that [wgps] says runs no work.
    They then clear the filter, so that no wave started after them counts.
    [l] must be the layout {!val-start} took. [wgps] is as
    {!Capability.t}'s. Each store is confirmed before the next packet
    starts.

    Raises [Invalid_argument] if [address] is not a non-negative multiple
    of 8, if [wgps] does not hold [g.shader_engines] engines, if [l] holds a
    counter of a block other than SQ, if [g] is a GPU other than a GFX12 of
    one die, or if [g]'s GC lacks a register the program reads or writes. *)

(**/**)

(* [start_packets g l ~filter] and [stop_packets g ~wgps l ~at ~zero] are
   {!val-start} and {!val-stop} as packets over any value: [filter] the
   filter word's address, [at o] the address [o] bytes into the samples, and
   [zero] the constant [0], so that a driver may leave the addresses for
   later ({!Packet.template}). They raise as {!val-start} and {!val-stop},
   but for the addresses' checks. *)
val start_packets : Gpu.t -> layout -> filter:'v -> 'v Packet.t

val stop_packets :
  Gpu.t ->
  wgps:int array array ->
  layout ->
  at:(int -> 'v) ->
  zero:'v ->
  'v Packet.t

