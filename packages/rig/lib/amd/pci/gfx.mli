(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's GC, whose compute micro-engines (MEC) load each hardware queue
    from a queue descriptor (MQD).

    It also drives the GC's clock gating, reads which work-group processors
    run work, and on a virtual function runs the KIQ, through which the
    function asks for what its host keeps. Calls for one GPU run one at a
    time. *)

(** {1:formats Formats}

    The formats of a GPU's GC that its driver writes: the descriptors (MQD)
    its compute queues load, and the index that selects the shader arrays
    its registers reach. Every value is pure, and any domain may call it. *)

(** {2:mqd Queue descriptors} *)

type queue = {
  ring : int;  (** The ring's GPU address, 256-byte aligned. *)
  ring_bytes : int;  (** Its size, a power of two. *)
  read : int;  (** Where the queue writes its read position. *)
  write : int;  (** Where it polls its write position. *)
  eop : int;  (** Its end-of-pipe buffer's GPU address, 256-byte aligned. *)
  eop_bytes : int;  (** Its size, a power of two. *)
  doorbell : int;  (** Its doorbell index. *)
}
(** The type for what a compute queue is made of. *)

val mqd :
  Rig_amd_abi.Gpu.t ->
  queue ->
  base:int ->
  [ `Kiq | `Pm4 | `Aql ] ->
  xcc:int ->
  string
(** [mqd g q ~base kind ~xcc] is the descriptor of queue [q] on die [xcc] of
    [g]'s dies, laid out for [g]'s GC. The hardware reads it at the address
    the memory controller gives [base]. The queue is of [kind]:
    {ul
    {- [`Pm4] reads PM4 packets.}
    {- [`Kiq] reads PM4 packets, and is privileged and of the kernel
       driver's kind.}
    {- [`Aql] reads AQL packets, and a GPU of several dies spreads its work
       across them, its dies' descriptors 4 KiB apart.}}

    Raises [Invalid_argument] if [g]'s GC is not of major 9, 11 or 12. *)

(** {2:arrays Shader arrays} *)

val index : Rig_amd_abi.Gpu.t -> [ `Array of int * int | `All ] -> int
(** [index g sel] is the value of [g]'s [GRBM_GFX_INDEX] that directs
    register accesses to shader array [a] of shader engine [e] on every
    instance ([`Array (e, a)]), or broadcasts them to all ([`All]). *)

(** {1:gc The GC} *)

type t
(** The type for a GPU's GC. *)

val make :
  Regs.t -> Gmc.t -> Rig_pci.Window.t -> Rig_pci.Window.t -> mqds:int array -> t
(** [make r gmc vram doorbells ~mqds] is the GC, whose queue descriptors are
    at the physical addresses [mqds] of the GPU's memory, one page per die
    each: the compute queue's, and on a virtual function the KIQ's, whose
    doorbell it rings through [doorbells]. *)

val wait_autoload : t -> unit
(** [wait_autoload g] waits for the RLC to have loaded the GC's firmware, on
    a GC that loads it so. The load resets the GC's hub, so a caller starts
    the hub after it ({!Gmc.start_hub}).

    Raises {!Regs.Stuck} if the load does not complete within 10 s. *)

val start :
  t -> Rig_pci.Memory.t -> Firmware.t -> partial:bool -> unit
(** [start g m images ~partial] starts the micro-engines, the RS64 ones from
    the start addresses of [images], and enables the interrupt a release of
    the compute queue raises. The RLC has loaded the GC's firmware
    ({!wait_autoload}) and the GC's hub is started ({!Gmc.start_hub}).
    {ul
    {- A full boot configures the engines and their doorbell ranges, and on
       a virtual function the KIQ, in host memory of [m].}
    {- A partial boot dequeues the queues the last boot left, then resets
       and restarts the engines.}}

    Raises {!Regs.Stuck} if a step does not complete. *)

val queue :
  t ->
  [ `Pm4 | `Aql ] ->
  ring:int ->
  bytes:int ->
  read:int ->
  write:int ->
  eop:int ->
  int
(** [queue g kind ~ring ~bytes ~read ~write ~eop] programs compute queue 0 of
    [kind], with an end-of-pipe buffer of 4 KiB at [eop], and is its doorbell
    index. An AQL queue runs on every die, a PM4 queue on the first. A
    hardware queue left active is dequeued before its registers are written.

    Raises {!Regs.Stuck} if that dequeue does not complete within 10 s. *)

val dequeue : t -> wait:bool -> bool
(** [dequeue g ~wait] removes every queue this library programs. With
    [wait] it waits up to 10 s for each to leave, and is [false] if one did
    not, as when a wave the reset cannot stop holds it. Without [wait] it is
    [true]. *)

val halt : t -> unit
(** [halt g] halts the micro-engines. *)

val untrace : t -> unit
(** [untrace g] turns off a thread trace that may run: every shader engine's,
    then the compute queues' trace enable. It does nothing on a virtual
    function, which traces nothing. *)

val gate : t -> unit
(** [gate g] enables the GC's clock gating, under the RLC's safe mode. *)

val ungate : t -> unit
(** [ungate g] disables the clock gating {!gate} enables, under the RLC's
    safe mode, so that the GC's clocks run steadily. *)

val wgps : t -> int array array
(** [wgps g] is the processors that run work, as [Rig_amd.path]'s [wgps]:
    per shader engine, engines numbered across dies, and per shader array, a
    bit per work-group processor (GC 10 on) or compute unit (GC 9) that
    neither the GPU's fuses nor its configuration mark inactive. *)

val invalidate : t -> unit
(** [invalidate g] asks the KIQ of each die to invalidate both hubs' TLBs,
    and waits for each.

    Raises [Invalid_argument] if [g] has no KIQ, which only a virtual
    function has, and {!Regs.Stuck} if a KIQ does not respond within
    10 s. *)
