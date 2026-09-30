(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** AMD's command queues, as batches encode them.

    An AMD GPU runs a batch's programs from its compute queue and its copies
    from its SDMA copy queues. The compute queue takes PM4 packets, or, on a GPU
    of several compute dies, AQL packets: a dispatch packet for each program,
    and the PM4 packets between them run as indirect buffers. Each queue has a
    ring in memory that the host and the GPU share, and three 64-bit words: the
    position its engine reads up to, the position after the last packet written,
    and a doorbell that wakes the engine. Positions grow without wrapping: they
    count dwords on a PM4 queue, 64-byte packets on an AQL queue and bytes on an
    SDMA queue, and a packet's place in the ring is its position modulo the
    ring's size.

    A batch's host program appends each queue's packets to its ring at its
    position, stores the new position, and rings the doorbell. It reads and
    writes the rings and words through placeholders of the device that the
    engine binds to the device's queues ({!storage}). The packets of a program
    name its code object and its scratch memory through placeholders too.

    {b Signal words.} A device's signal word ({!Hcq2.signal_word}) holds 64-bit
    values, and a queue compares 32-bit words. A queue waits on a device's
    signal word only for the value the device's work before the batch signals
    ({!Hcq2.submitted}), and only the batch's own work signals the value after
    it, so the wait is exact: it waits until the word's low 32 bits equal the
    value's. A queue's signal of a device's value ({!Hcq2.value}) writes the
    whole word: in one 64-bit write on the compute queue; on a copy queue, which
    writes 32 bits at a time, its low 32 bits, then its high 32 bits only when
    the low ones are [0]. A wait on the low half thus never passes before the
    work it waits for is complete, and a high half that lands after a later
    value's write holds that value's high half: the word never goes back. A wait
    on a queue's signal in the batch's slots, whose values are small, waits
    until the low 32 bits are at least the value, and a signal of one writes the
    low 32 bits. *)

(** {1:gpus GPUs} *)

type gpu = {
  target : int * int * int;
      (** The graphics target, such as [(11, 0, 0)] for ["gfx1100"]. *)
  sdma : int * int * int;  (** The version of the copy engine. *)
  xccs : int;  (** The number of compute dies. *)
  shader_engines : int;  (** The number of shader engines of one die. *)
  compute_units : int;  (** The number of compute units of one die. *)
  scratch_slots_per_cu : int;  (** The scratch wave slots of a compute unit. *)
  aql : bool;
      (** Whether the compute queue takes AQL packets rather than PM4 packets.
          It does on a GPU of several dies. *)
  compute_ring : int;  (** The size in bytes of the compute queue's ring. *)
  copy_rings : int list;
      (** The size in bytes of each copy queue's ring, ["COPY:0"] first. *)
}
(** The type for the AMD GPUs a compiler encodes work for. *)

(** {1:queues Command queues} *)

val queues : host:string -> reaches:(string -> bool) -> gpu -> Hcq2.queues
(** [queues ~host ~reaches gpu] is the command queues of a device that is the
    GPU [gpu], whose batches are submitted by host programs of [host], and whose
    queues address the memory of the devices [reaches] names. It copies on its
    copy queues if it has any. Its commands are:
    - [exec call prg] on the compute queue, which lays out [prg]'s arguments,
      the addresses of its buffers on the device then its variables, in its
      kernel argument segment, followed by its dispatch packet if it reads one,
      in a buffer of the device. It then invalidates the caches the kernel reads
      through, other than its instructions' and the L2, and dispatches [prg]'s
      kernel from its code object with its scratch memory, its local size and
      its grid of work-groups, at most the setting [WAVES_PER_SH] waves on each
      shader array when it is not [0] ({!Helpers.getenv}, read at each command),
      then waits for it to finish. With AQL, the dispatch is a dispatch packet,
      which waits for the packets before it;
    - [copy dst src n] on a copy queue, which copies [n] bytes in pieces of at
      most the copy engine's largest copy: 1 GiB from version 4.4.2 below 5.0
      and from 5.2, 4 MiB otherwise;
    - [wait word v], which waits as the signal words above say;
    - [signal word v], which writes [v] as the signal words above say once the
      work before it is complete and its writes are visible, and interrupts the
      host;
    - [timestamp slot], which writes the GPU's 100 MHz clock into the second
      word of [slot] once the work before it is complete;
    - [memory_barrier ()], which on the compute queue flushes the host data
      path, making the host's writes to the GPU's memory visible, then
      invalidates the GPU's caches; it does nothing on a copy queue;
    - [submit cmdbuf], whose host program appends the queue's packets to its
      ring. The compute queue's are one indirect buffer packet that runs
      [cmdbuf] with PM4, and the dispatch packets and indirect buffers of
      [cmdbuf] with AQL. A copy queue's are [cmdbuf] itself, held in [host]'s
      memory: if it does not fit before the end of the ring, the rest of the
      ring is zeroed and it starts at the ring's beginning.

    [exec] and [copy] raise [Invalid_argument] on a queue that does not run
    them, and a command of a copy queue [gpu] lacks raises [Invalid_argument]. A
    submission writes at most half of each ring, the room the device leaves it:
    [submit] raises {!Hcq2.Over_capacity} if a copy queue's packets exceed a
    quarter of its ring, since zeroing the ring's tail can double what they
    take, or if the AQL packets exceed half of theirs, and the batch is then
    split into several submissions.

    Raises [Invalid_argument] if [gpu] has several dies and its compute queue
    takes PM4 packets, or if [gpu.target] is none of [(9, 4, 2)], [(9, 5, 0)]
    and the targets of major version 11 and 12. *)

(** {1:linking What the engine links} *)

(** The type for the storage that the placeholders of AMD's commands name. The
    queues are named ["COMPUTE:0"] and ["COPY:i"], as {!Hcq2.Queue.name} names
    them. *)
type storage =
  | Ring of string  (** The ring of the queue. *)
  | Write_ptr of string
      (** The 64-bit position the queue's engine reads up to. *)
  | Put of string
      (** The 64-bit position after the last packet written in the queue's ring,
          where the next writer appends. *)
  | Doorbell of string  (** The 64-bit doorbell of the queue. *)
  | Program of { binary : string; name : string }
      (** The code object [binary], loaded on the device, of which the command
          runs the kernel [name]. Its descriptor and its first instruction are
          at the offsets of the code object's image that the ELF object [binary]
          lays out. *)
  | Scratch of int
      (** The device's scratch memory, for kernels of up to that many bytes of
          scratch per lane. *)

val storage : Ops.t -> storage option
(** [storage u] is the storage of the placeholder [u] of a batch if AMD's
    commands name it, and [None] for the others, which the engine allocates. *)
