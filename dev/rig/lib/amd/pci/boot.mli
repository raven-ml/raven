(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** An AMD GPU this process booted over PCI: its BARs, its blocks brought up,
    its page tables, its memory, and the session mark it leaves for the next
    open.

    {v
    take the function
      -> read: BARs, discovery, blocks supported, mark, firmware   (state unchanged)
      -> ASPM off, VF access, page tables, boot pool
      -> full boot:    MM hub, IH, PSP and its firmware, SMU
         partial boot: the TMR the last boot left, MEC reset
      -> GC hub and MEC, SDMA, clocks, gating, mark
    v}

    A full boot needs a GPU that runs no firmware; a partial one a GPU that
    carries this library's clean mark. A boot marks the GPU dirty before it
    touches a block, so one that fails midway is booted fully next time.

    One mutex per GPU serializes the register sequences and page-table edits of
    every domain; [sleep] waits for interrupts outside it. *)

(** {1:sessions Sessions} *)

val session : int
(** [session] is the mark a GPU this library booted carries in its seventh GC
    scratch register. *)

val plan :
  mark:int ->
  dirty:int ->
  fault:int ->
  gc:Discovery.version ->
  alive:bool ->
  [ `Partial | `Full | `Booted ]
(** [plan ~mark ~dirty ~fault ~gc ~alive] is how a GPU boots that carries [mark]
    in its seventh scratch register and [dirty] in its sixth, with [fault] in
    GC's protection fault status, and whose security processor and power manager
    run iff [alive]:
    - [`Partial] if it carries {!session}, [dirty] and [fault] are [0], or its
      GC is 9.5.0, whose full boot over live state can stall its fabric;
    - [`Full] otherwise if no firmware runs;
    - [`Booted] otherwise: firmware this library did not leave runs on it, and
      only a reset stops it. Pure. *)

(** {1:gpus GPUs} *)

type t
(** The type for GPUs this process booted. *)

val space : Rig_pci.Space.t
(** [space] is the GPU addresses every GPU this library boots shares, in the
    process and on its machine, which {!start}'s caller reserves
    ({!Rig_pci.Machine.reserve}): 2{^ 44} bytes from [0x2000_0000_0000]. *)

val start :
  Rig_pci.Function.t ->
  (string -> digest:string -> (string, string) result) ->
  (t, [ `Refused of string | `Lost of string ]) result
(** [start f find] boots the GPU of [f], whose function the caller took and
    whose machine has {!space} reserved, its firmware read with [find]: a
    partial or full boot as {!plan} says. The GPU masters the bus only once
    booted, its hubs' faults reaching a page of system memory the boot owns.

    [Error (`Refused msg)], no register written, if a BAR cannot be mapped, if
    its discovery table is refused, if a block has a version this library does
    not boot, if firmware is missing, if it is [`Booted], saying that a reset
    stops it, if it is in a fabric left running, if its memory controller's
    window does not hold its memory ({!Gmc.window}), or if the machine has no
    memory for its page tables or fault page. [Error (`Lost msg)] if a block
    does not answer, naming the step: the GPU is then stopped ({!stop}). An
    exception raised during the boot stops it too, and passes through. *)

val reset : Rig_pci.Function.t -> (unit, string) result
(** [reset f] resets the GPU of [f] as [Rig_amd_pci.reset] states, its bus
    mastering off: if its security processor and power manager run, it stops the
    compute queues, lowers the clocks, halts the engines and resets the GPU
    whole (mode 1), waiting until the function answers again, its configuration
    restored. A GPU of a fabric (XGMI) is only stopped: its GPUs reset together,
    as their kernel driver does when it takes them back. It then turns the
    interrupt rings off. [Ok ()] means no engine or interrupt ring runs, nor
    firmware but on a fabric's GPU: a GPU nothing ran on, or one of blocks this
    library does not boot, which it never wrote to, is left as it is.
    [Error msg] if the GPU does not answer, if its configuration differs after
    the reset, or if its security processor's OS or a ring still runs after it,
    as when its power manager is hung. *)

val gpu : t -> Rig_amd_abi.Gpu.t
val gc : t -> Discovery.gc
val mec : t -> int
val wgps : t -> int array array

val memory : t -> Rig_pci.Memory.t
(** [memory g] is the GPU's memory, placed by kind. *)

val hive : t -> bool
(** [hive g] is {!Gmc.hive}. *)

val budget : t -> int
(** [budget g] is the bytes of the GPU's memory it hands out: its page tables'
    main pool. *)

val reaches : t -> t -> bool
(** [reaches g o] is [true] iff [g] reaches all of [o]'s memory, both GPUs of
    one machine: over their fabric, or through [o]'s memory BAR as large as its
    memory, both at physical addresses no IOMMU translates. *)

val protect : t -> (unit -> 'a) -> 'a
(** [protect g f] is [f ()] with [g]'s register sequences and page-table edits
    held off, as {!Rig_pci.Memory} calls on [g]'s memory need. *)

val queue :
  t ->
  [ `Pm4 | `Aql | `Sdma ] ->
  ring:int ->
  bytes:int ->
  read:int ->
  write:int ->
  (int, string) result
(** [queue g kind ~ring ~bytes ~read ~write] is [Rig_amd.path]'s [queue]: the
    host address of the queue's doorbell. A compute queue's end-of-pipe buffer
    is the GPU's own memory. [Error msg] if the doorbell BAR is not mapped into
    the process, or no GPU memory is left for the buffer. *)

val hdp : t -> int option
(** [hdp g] is the host address of the HDP flush register, if the register BAR
    is mapped into the process. *)

val sleep : t -> ms:int -> unit
(** [sleep g ~ms] is [Rig_amd.path]'s [sleep]: it waits at most [ms]
    milliseconds for the interrupt ring to move, then reads it. It raises
    [Rig_amd.Fault] with every report of a fault, a fatal hardware error and the
    GPU's machine-check banks, or the function's or machine's failure. Once
    raised, every later sleep raises the same report. *)

val give_back : t -> unit
(** [give_back g] gives back the access a virtual function asked its host for,
    once the device's queues are made: a host resets a VF that holds access
    long. It does nothing on a physical function. *)

val stop : t -> [ `Clean | `Lost | `Unknown ]
(** [stop g] stops the GPU's queues and leaves it for the next open, clocks
    lowered and its bus mastering off, so that it reaches no memory outside its
    own but over a fabric:
    - [`Clean] if every queue left and no fault was reported: the mark is clean,
      and the next open boots partially;
    - [`Lost] if a fault was reported or a queue did not leave: the mark is
      dirty, and only a reset recovers it;
    - [`Unknown] if its machine failed, or a queue of a GPU in a fabric did not
      leave, whose writes to its peers bus mastering does not stop.

    [`Clean] and [`Lost] answer [Rig_amd]'s [`Stopped]. A later stop does
    nothing and answers the same. *)
