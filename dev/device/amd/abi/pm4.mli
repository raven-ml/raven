(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PM4 packets: the commands of a compute queue.

    A compute queue runs its packets in order. It starts each packet once the
    one before it has started, never once it has completed: a dispatch's waves
    may still run when the packets after it start. {!run} is the rule that
    orders kernels: every dispatch goes inside one.

    A PM4 ring may wrap: a packet's words continue at the ring's start. *)

(** {1:runs Runs} *)

val dispatch :
  Gpu.t ->
  Code_object.kernel ->
  program:'v ->
  scratch:'v ->
  args:'v ->
  packet:'v ->
  threads:'v * 'v * 'v ->
  groups:'v * 'v * 'v ->
  ?waves_per_array:int ->
  unit ->
  'v Packet.t
(** [dispatch g k ~program ~scratch ~args ~packet ~threads ~groups
     ~waves_per_array ()] launches the kernel [k], whose first instruction is at
    address [program] (its image's address plus [k.entry]), on one die, over a
    grid of [groups] workgroups of [threads] work-items each. The packets after
    it may start before its waves complete.

    It sets the [COMPUTE_*] registers from [k]'s descriptor: its code, its
    resource words, with the privilege GFX11 runs kernels with and the LDS its
    workgroups take, the scratch of [k]'s private segment ({!Scratch.tmpring})
    in the buffer at [scratch], and the user SGPRs [k] enables, in their order:
    a buffer descriptor of the scratch at [scratch], the address [packet] of its
    dispatch packet ({!Aql.dispatch}), and the address [args] of its arguments.
    [packet] is read only if [k.dispatch_ptr]. At most [waves_per_array] waves
    run at once on each shader array; without it, as many as fit.

    [program] and [scratch] are 256-byte aligned. Raises [Invalid_argument] if
    [g]'s GC has no register of a dispatch. *)

val run : Gpu.t -> 'v Packet.t -> 'v Packet.t
(** [run g p] runs [p], the words of a dispatch and of what goes with it, such
    as its profiling: it invalidates the data caches above the L2 before [p], so
    that [p]'s kernels read what the work before them wrote, and waits for [p]'s
    dispatches to complete after it ([CS_PARTIAL_FLUSH]), so that the packets
    after it start once they have. *)

(** {1:memory Memory and registers} *)

(** The type for what a packet reads or writes: a register, at its address
    ({!Register.address}), or memory, at the caller's address. *)
type 'v location = Register of int | Memory of 'v

val set_reg : int -> 'v Packet.t -> 'v Packet.t
(** [set_reg a ws] writes the words [ws] to the consecutive registers from the
    one at address [a]: an SH register's packet or a UCONFIG register's.

    Raises [Invalid_argument] if [a] is in neither range. *)

val write_data : 'v location -> 'v -> 'v Packet.t
(** [write_data loc v] writes the low 32 bits of [v] to [loc]; a write to memory
    completes before the next packet starts. *)

(** The type for how a copy completes: posted, or confirmed before the next
    packet starts. *)
type write = Posted | Confirmed

(** The type for what a copy reads: a 32-bit counter register, at its address,
    or the GPU's clock, a 64-bit count at 100 MHz. *)
type source = Counter of int | Clock

val copy_data : write -> source -> 'v -> 'v Packet.t
(** [copy_data w src addr] copies [src] to memory at [addr] through the L2, as
    [w] says, when the queue reaches the packet: once the packets before it have
    started, before those after it start. *)

(** {1:sync Waits, caches and signals} *)

val wait :
  Gpu.t ->
  'v location ->
  Packet.comparison ->
  'v ->
  ?mask:int ->
  ?interval:int ->
  unit ->
  'v Packet.t
(** [wait g loc cmp v ~mask ~interval ()] waits until the 32 bits at [loc],
    masked by [mask], compare to the low 32 bits of [v] as [cmp] says, reading
    them again every [interval] clocks of the packet's poll timer. [mask]
    defaults to all 32 bits, [interval] to [4]. *)

(** The type for the caches an acquire invalidates: the data caches above the L2
    (scalar, vector and L1), or all of them, the instruction cache too, and the
    L2, written back first. *)
type caches = Data_caches | All_caches

val acquire_mem : Gpu.t -> caches -> 'v Packet.t
(** [acquire_mem g c] invalidates the caches [c] of all memory before the next
    packet starts. *)

(** The type for the signal a release writes: the low 32 bits of a value, or all
    64. *)
type 'v data = Low_32 of 'v | Data_64 of 'v

val release_mem : Gpu.t -> 'v -> 'v data -> 'v Packet.t
(** [release_mem g addr d] writes [d] to memory at [addr] once the work before
    it has completed, at the end of the pipe, after writing back and
    invalidating the GPU's caches so that whoever reads the signal sees the
    work's writes, then raises an interrupt. The next packet starts without
    waiting for it. A 64-bit write is one write: a reader sees the value whole
    or not at all. *)

(** The type for the events a queue signals. *)
type event =
  | Cs_partial_flush  (** Waits for the dispatches before it to complete. *)
  | Thread_trace_marker  (** Marks the thread trace ({!Thread_trace}). *)
  | Thread_trace_finish  (** Finishes the thread trace. *)

val event_write : event -> 'v Packet.t
(** [event_write e] signals [e]. *)

(** {1:control Control} *)

val pred_exec : xcc_mask:int -> 'v Packet.t -> 'v Packet.t
(** [pred_exec ~xcc_mask p] runs [p] only on the dies of [xcc_mask], bit [i] for
    die [i], of a queue that runs on several.

    Raises [Invalid_argument] if [xcc_mask] is not in \[[0];[255]\] or
    [Packet.size p] exceeds 16383 words. *)

val indirect_buffer : 'v -> dwords:int -> 'v Packet.t
(** [indirect_buffer addr ~dwords] runs the [dwords] words of PM4 packets at
    [addr], then the packets after it. *)
