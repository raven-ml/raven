(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The boot of an AMD GPU over PCI, and the GPU this process then holds
    until it stops it, leaving a mark for the next open.

    {v
    take the function
      -> read, nothing written: BARs, discovery table, layout, firmware,
         the last session's mark
      -> page tables, boot pool, the hubs' fault page
      -> full boot:    SoC, MM hub, interrupt rings, PSP and firmware, SMU
         partial boot: the last session's engines stopped, rings, its TMR
      -> GC hub, MEC, SDMA, clocks, gating, mark
      -> bus mastering on, once both hubs translate
    v}

    A full boot needs a GPU that runs no firmware, and a partial boot a GPU
    that carries this library's clean mark. A partial boot first stops the
    engines the last session may have left running, as {!stop} does, before
    its hubs take this boot's tables. A boot marks the GPU dirty before it
    touches a block, so a boot that fails midway leaves a full boot for the
    next open. One mutex per GPU holds the register sequences and page-table
    edits of every domain apart, and {!sleep} waits for interrupts outside
    it. *)

(** {1:sessions Sessions} *)

val session : int
(** [session] is the mark a GPU this library booted carries in its GC's
    [SCRATCH_REG7]. *)

val plan :
  gc:int * int * int ->
  mark:int ->
  dirty:int ->
  fault:int ->
  os:bool ->
  [ `Partial | `Full | `Booted ]
(** [plan ~gc ~mark ~dirty ~fault ~os] is how a GPU boots whose GC, of
    version [gc], holds [mark] in [SCRATCH_REG7], [dirty] in [SCRATCH_REG6]
    and [fault] in its protection fault status, and whose security
    processor's OS runs iff [os]. Pure.
    {ul
    {- [`Partial] if it carries {!session}, and [dirty] and [fault] are [0],
       as a session that stopped cleanly leaves them, or [gc] is 9.5.0,
       whose fabric a mode 1 reset over live state can stop until a power
       cycle;}
    {- [`Full] otherwise, if no OS runs;}
    {- [`Booted] otherwise, when firmware runs whose state this library
       cannot vouch for, whoever left it. Only a reset stops it.}} *)

(** {1:gpus GPUs} *)

type t
(** The type for GPUs this process booted. *)

val space : Rig_pci.Space.t
(** [space] is the GPU addresses every GPU this library boots shares, in the
    process and on its machine: 2{^44} bytes from [0x2000_0000_0000].
    {!start}'s caller reserves it ({!Rig_pci.Machine.val-reserve}). *)

val start :
  gpus:int Lazy.t ->
  Rig_pci.Function.t ->
  (Discovery.t ->
  (Regs.layout * Firmware.t, string) result) ->
  (t, [ `Refused of string | `Running | `Lost of string ]) result
(** [start ~gpus f first] boots the GPU of [f], partially or fully as
    {!plan} says. The caller took [f], reserved {!space} on its machine, and
    counts [gpus] AMD GPUs there. [first d] returns the registers' layout
    and the firmware from the GPU's discovery table [d], which the boot reads
    before it writes anything. The GPU's memory joins its fabric as
    {!Gmc.link} says.

    The GPU masters the bus only once it is booted, every flush has been
    confirmed ({!confirm}), and both hubs read back as translating
    ({!Gmc.translates}) with their faults sent to a page of system memory
    the boot owns.

    The result is [Error (`Refused msg)], before any block's register is
    written, if:
    {ul
    {- a BAR cannot be mapped, or its register or doorbell BAR is not mapped
       into the process, as through a transport;}
    {- its firmware has not laid out its memory, the GPU reports no memory,
       or its discovery table is refused;}
    {- [first] returns [Error], with its message;}
    {- it is in a fabric left running;}
    {- its memory controller's window does not hold its memory
       ({!Gmc.covers});}
    {- the machine has no memory for its page tables or fault page.}}
    It is [Error `Running], before any block's register is written and with
    its BARs unmapped, if {!plan} returns [`Booted] outside a fabric. The
    caller then resets the GPU ({!reset}) and starts again. With either
    error, a virtual function gives back the access it asked its host for.

    It is [Error (`Lost msg)] if a block does not respond, naming the step,
    or a hub does not translate, and the GPU is then stopped ({!stop}). An
    exception raised during the boot stops the GPU the same way and passes
    through. *)

val confirm :
  Rig_pci.Function.t -> string option ref -> (unit -> unit) -> unit -> bool
(** [confirm f fault invalidate] is the hubs' flush of the GPU of [f]:
    [confirm f fault invalidate ()] runs [invalidate] and is [true] iff it
    completes. If [invalidate] raises {!Regs.Stuck}, it turns [f]'s bus
    mastering off, records the first such reason in [fault] and is [false],
    since the hubs may still reach memory through a translation the flush
    removed. {!start} gives every flush of its tables this way, and turns bus
    mastering on only while [fault] is [None]. *)

val writes : pcie:int option -> rebars:int list -> (int * int) list
(** [writes ~pcie ~rebars] is the configuration a reset restores, as
    [(offset, bytes)] in the order it writes them, for a function whose PCI
    Express capability is at [pcie] and whose resizable BARs' controls are
    at [rebars]. The PCI Express controls come first, then the resizable
    BARs' controls, which clear their BARs' addresses when written, then the
    header with the BARs, and the command register last. Pure. *)

val reset : Rig_pci.Function.t -> (unit, string) result
(** [reset f] resets the GPU of [f] as [Rig_amd_pci.reset] states, its bus
    mastering off. If the GPU's security processor's OS and power manager
    run, it stops the compute queues, lowers the clocks where the power
    manager lets it, halts the engines, resets the GPU whole (mode 1),
    restores its configuration and waits until its bootloader and memory
    respond. A refusal to lower the clocks is printed on standard error and
    does not stop the reset. Then it turns the interrupt rings off.
    A GPU of blocks this library does not boot, which it never wrote to, is
    left as it is.

    [Ok ()] means no engine or interrupt ring runs and, outside a fabric, no
    firmware runs and the bootloader is ready for its first command. The
    result is [Error msg]:
    {ul
    {- if the GPU does not respond, or its configuration differs after the
       reset;}
    {- if its security processor's OS or an interrupt ring still runs after
       it, as when its power manager is hung;}
    {- if its bootloader is not ready with no OS running, as after a failed
       load. Only a power cycle recovers that GPU.}} *)

val gpu : t -> Rig_amd_abi.Gpu.t
(** [gpu g] is {!Regs.gpu} of [g]'s layout, counting the compute units that
    run work, as {!wgps} reads them. *)

val gc : t -> Discovery.gc
(** [gc g] is the shape of [g]'s GC, from its discovery table. *)

val mec : t -> int
(** [mec g] is the version of the compute queues' firmware [g] booted
    with. *)

val wgps : t -> int array array
(** [wgps g] is {!Gfx.wgps} of [g]'s GC. *)

val memory : t -> Rig_pci.Memory.t
(** [memory g] is the GPU's memory, placed by kind. *)

val hive : t -> bool
(** [hive g] is {!Gmc.hive}. *)

val budget : t -> int
(** [budget g] is the bytes of the GPU's memory that its page tables' main
    pool hands out. *)

val lock : t -> Rig_amd.Lock.t
(** [lock g] is held over [g]'s register sequences and page-table edits: calls
    on {!Rig_pci.Memory} of [g]'s memory take it. *)

val queue :
  t ->
  [ `Pm4 | `Aql | `Sdma ] ->
  ring:int ->
  bytes:int ->
  read:int ->
  write:int ->
  (int, string) result
(** [queue g kind ~ring ~bytes ~read ~write] is [Rig_amd.path]'s [queue]:
    the host address of the queue's doorbell. A compute queue's end-of-pipe
    buffer is the GPU's own memory. The result is [Error msg] if the doorbell
    BAR is not mapped into the process, or no GPU memory is left for the
    buffer. *)

val hdp : t -> int option
(** [hdp g] is the host address of the HDP flush register, if the register
    BAR is mapped into the process. *)

val sleep : t -> ms:int -> unit
(** [sleep g ~ms] is [Rig_amd.path]'s [sleep]. It waits at most [ms]
    milliseconds for the interrupt ring to move, then reads it. A sleep that
    a {!stop} overtakes returns without reading the ring again.

    Raises [Rig_amd.Fault] with every report of a fault, a fatal hardware
    error with the GPU's machine-check banks, or the function's or machine's
    failure. Once it has raised, every later sleep raises the same
    report. *)

val faulted : t -> string -> unit
(** [faulted g why] records [why] as a fault of the GPU's work that its
    interrupt ring did not report, such as work that stopped making
    progress, unless a fault was recorded already. {!stop} then returns
    [`Lost], and {!sleep} raises the first fault. *)

val stable_power : t -> (unit, string) result
(** [stable_power g] is [Rig_amd.path]'s [stable_power]. The boot holds the
    GPU's clocks at their highest, and this turns the GC's clock gating off
    for the rest of the session. The result is [Error msg] on a virtual
    function, whose host holds its clocks.

    Raises {!Regs.Stuck} if the RLC does not respond. *)

val give_back : t -> unit
(** [give_back g] gives back the access a virtual function asked its host
    for, once the device's queues are made, since a host resets a virtual
    function that holds its access long. It does nothing on a physical
    function. *)

val stop : t -> [ `Clean | `Lost | `Unknown ]
(** [stop g] stops the GPU's queues and leaves it for the next open, its
    clocks lowered and its bus mastering off, so that it reaches no memory
    outside its own but over a fabric.
    {ul
    {- [`Clean] if every queue left and no fault was reported. The mark is
       clean, and the next open boots partially.}
    {- [`Lost] if a fault was reported, a queue did not leave, or a step of
       the stop failed. The mark is dirty, and only a reset recovers the
       GPU.}
    {- [`Unknown] if its machine failed, or if a queue of a GPU in a fabric
       did not leave, since bus mastering does not stop its writes to its
       peers.}}
    [`Clean] and [`Lost] give [Rig_amd]'s [`Stopped]. A later stop does
    nothing and returns the same. *)
