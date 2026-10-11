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
  t -> Rig_pci.Memory.t -> Rig_amd_abi.Firmware.t -> partial:bool -> unit
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
