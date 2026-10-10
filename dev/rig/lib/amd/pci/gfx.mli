(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's GC: its compute micro-engines (MEC) and the memory queue
    descriptors (MQD) their hardware queues load, the doorbells that wake them,
    the KIQ through which a virtual function asks for what its host keeps, clock
    gating, and which work-group processors run work.

    The GPU's owner serializes calls. *)

(** {1:mqd Queue descriptors} *)

type queue = {
  ring : int;  (** The ring's GPU address, on 256 bytes. *)
  ring_bytes : int;  (** Its size, a power of two. *)
  read : int;  (** Where the queue writes its read position. *)
  write : int;  (** Where it polls its write position. *)
  eop : int;  (** Its end-of-pipe buffer's GPU address, on 256 bytes. *)
  eop_bytes : int;  (** Its size, a power of two. *)
  doorbell : int;  (** Its doorbell index. *)
}
(** The type for what a compute queue is made of. *)

(* CR: Derive the die count from Regs.gpu l and remove ~xccs. AQL queues
   use every live die; program passes back the count this layout supplied.
   Keep ~xcc and PM4/KIQ's instance selection. Make the multidie test's
   discovery layout contain eight live GC instances instead of pairing a
   one-die layout with ~xccs:8. *)
val mqd :
  Regs.layout ->
  queue ->
  base:int ->
  kiq:bool ->
  aql:bool ->
  xcc:int ->
  xccs:int ->
  string
(** [mqd l q ~base ~kiq ~aql ~xcc ~xccs] is the descriptor of queue [q] on die
    [xcc] of [xccs], as the kernel's MQD structs of [l]'s GC lay it out
    ([v9_structs.h], [v11_structs.h], [v12_structs.h]), which the hardware reads
    at the address the memory controller gives [base]: privileged and of the
    kernel driver's kind if [kiq], an AQL queue if [aql], whose work a GPU of
    several dies spreads across them. Pure. *)

(** {1:gc The GC} *)

type t
(** The type for a GPU's GC. *)

val make :
  Regs.t -> Gmc.t -> Rig_pci.Window.t -> Rig_pci.Window.t -> mqds:int array -> t
(** [make r gmc vram doorbells ~mqds] is the GC, whose queue descriptors are at
    the physical addresses [mqds] of the GPU's memory, one page per die each:
    the compute queue's, and the KIQ's on a virtual function, whose doorbell it
    rings through [doorbells]. *)

val wait_autoload : t -> unit
(** [wait_autoload g] waits for the RLC to have loaded the GC's firmware, on a
    GC that loads it so. The load resets the GC's hub, so the hub is started
    after it ({!Gmc.start_hub}). *)

val start : t -> Rig_pci.Memory.t -> Images.t -> partial:bool -> unit
(** [start g m images ~partial] starts the micro-engines, once the RLC loaded
    the GC's firmware ({!wait_autoload}), the RS64 ones from the start addresses
    of [images]: a full boot configures them and their doorbell ranges, and a
    virtual function its KIQ, in system memory of [m]; a partial boot dequeues
    the queues the last boot left and resets the engines. Either enables the
    interrupt a release of the compute queue raises. The GC's hub is started
    before ({!Gmc.start_hub}). *)

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
    [kind], on every die for AQL, on the first for PM4, from an end-of-pipe
    buffer of 4 KiB at [eop]. It is the queue's doorbell index. *)

val dequeue : t -> wait:bool -> bool
(** [dequeue g ~wait] removes every queue it programmed. With [wait], it waits
    for each to leave, and is [false] if one did not: a wave the reset could not
    stop holds it. *)

val halt : t -> unit
(** [halt g] halts the micro-engines. *)

val untrace : t -> unit
(** [untrace g] turns off a thread trace that may run: every shader engine's,
    then the compute queues' trace enable. A virtual function traces nothing. *)

val gate : t -> unit
(** [gate g] enables the GC's clock gating, under the RLC's safe mode. *)

val ungate : t -> unit
(** [ungate g] disables the clock gating {!gate} enables, under the RLC's safe
    mode, so that the GC's clocks run steadily. *)

val index : Regs.layout -> [ `Array of int * int | `All ] -> int
(** [index l sel] is the value of [l]'s GPU's GRBM_GFX_INDEX that directs
    register accesses to shader array [a] of shader engine [e] on every instance
    ([`Array (e, a)]), or broadcasts them to all ([`All]). Pure. *)

val wgps : t -> int array array
(** [wgps g] is the processors that run work, as [Rig_amd.path]'s [wgps]: per
    shader engine, engines numbered across dies, and shader array, a bit per
    work-group processor (GC 10 on) or compute unit (GC 9) that neither the
    GPU's fuses nor its configuration mark inactive. *)

val invalidate : t -> unit
(** [invalidate g] asks the KIQ of each die to invalidate both hubs' TLBs. On a
    virtual function only. *)
