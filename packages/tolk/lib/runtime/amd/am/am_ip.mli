(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hardware-IP bring-up protocols for driver-less AMD GPU devices.

    Each module drives one IP block of an {!Amdev.t} over its
    registers: {!Soc} routes doorbells and interrupt sources at the
    die level, {!Gmc} programs the memory hubs and their address
    translation, {!Psp} feeds firmware to the security processor,
    which alone can start the other engines, {!Smu} speaks the
    power-management message protocol, {!Gfx} and {!Sdma} bring up the
    compute and DMA engines and their queues, and {!Ih} owns the
    interrupt rings. All address registers by name through the device,
    poll them against the device clock ({!Amdev.now_ms}), and place
    their in-memory structures in the device's boot memory region, so
    a device built with {!Amdev.make} runs the full protocols over
    scripted register access.

    Blocks read each other's state where the hardware couples them;
    those neighbours are explicit parameters, wired by the boot
    sequence. The register layer resolves die instance 0 only, so
    per-instance programming (several memory hubs or compute dies)
    addresses every instance through the same registers; this is exact
    on single-die parts. *)

exception Timeout_error of string
(** Raised when a polled register fails to reach its expected value
    within a protocol's deadline, measured against the device clock. *)

(** {1:soc Die-level control}

    The die-level block owns what is neither memory nor an engine: the
    doorbell routing between the host bus and the engines, and the
    naming of interrupt clients and sources used to decode interrupt
    entries. *)

module Soc : sig
  type t
  (** The type for the die-level block of one device. *)

  val create : Amdev.t -> t
  (** [create adev] resolves the device's generation constants and
      interrupt naming tables. Raises [Invalid_argument] when the
      graphics-core generation is unknown. *)

  val init_hw : t -> vmhubs:int -> unit
  (** [init_hw t ~vmhubs] enables doorbell routing for the device:
      opens the doorbell aperture and, per generation, either lifts
      the function's soft-reset strap or programs the multi-die
      doorbell fences of each of the [vmhubs] dies (see
      {!Gmc.vmhubs}). *)

  val set_clockgating_state : t -> unit
  (** [set_clockgating_state t] enables the host-data-path memory
      power features on generations that have them. *)

  val doorbell_enable :
    t ->
    port:int ->
    ?awid:int ->
    ?awaddr_31_28_value:int ->
    ?offset:int ->
    ?size:int ->
    ?aid:int ->
    unit ->
    unit
  (** [doorbell_enable t ~port ()] routes a doorbell aperture range to
      the engine behind [port]: [offset] and [size] select the range,
      [awid] and [awaddr_31_28_value] the bus write route. [aid]
      addresses a die other than the first on multi-die parts. All
      default to [0]. *)

  (** {2:constants Generation constants} *)

  val mtype_uc : t -> int
  (** [mtype_uc t] is the generation's uncached memory-type encoding. *)

  val sh_mem_address_mode_64 : t -> int
  (** [sh_mem_address_mode_64 t] is the generation's 64-bit
      shader-memory address-mode encoding. *)

  val sh_mem_alignment_mode_unaligned : t -> int
  (** [sh_mem_alignment_mode_unaligned t] is the generation's
      unaligned shader-memory alignment-mode encoding. *)

  (** {2:ih Interrupt naming} *)

  val ih_client_name : t -> int -> string option
  (** [ih_client_name t client] is the name of the interrupt client id
      [client], when the generation's client table lists it. *)

  val ih_src_name : t -> client:int -> src:int -> string
  (** [ih_src_name t ~client ~src] is the name of interrupt source
      [src] of the client [client], or [""] when unnamed. *)
end

(** {1:gmc Memory hubs}

    The graphics and memory hubs translate engine addresses through
    the device's page tables. The block programs both hubs' system
    apertures, translation caches and page-table roots, and issues the
    TLB invalidations that make mapping changes visible. *)

module Gmc : sig
  type t
  (** The type for the memory-hub block of one device. *)

  type hub = Mm | Gc
      (** The type for the two hub kinds: the memory hub ([Mm], serving
          the DMA and multimedia engines) and the graphics hub ([Gc],
          serving the compute engines). *)

  val create : Amdev.t -> t
  (** [create adev] reads the framebuffer window and virtual-space
      bounds and allocates the fault scratch pages from boot memory.
      Requires [adev] to be booting. *)

  val vmhubs : t -> int
  (** [vmhubs t] is the number of memory-hub instances the device
      discovered (one per die). *)

  val init_hw : t -> soc:Soc.t -> unit
  (** [init_hw t ~soc] programs the memory hub on every die: see
      {!init_hub}. *)

  val init_hub : t -> soc:Soc.t -> hub -> inst_cnt:int -> unit
  (** [init_hub t ~soc hub ~inst_cnt] programs [inst_cnt] instances of
      [hub]: the system and AGP apertures, the fault default pages,
      the translation caches, the level-2 cache geometry, address
      translation over the device's root page table for VM context 0,
      and the invalidation engines' address ranges. After it returns,
      {!flush_tlb} reaches the hub. *)

  val flush_hdp : Amdev.t -> unit
  (** [flush_hdp adev] flushes the host-data-path write buffer, making
      host writes to device memory visible to the engines. *)

  val flush_tlb : t -> ?flush_type:int -> xccs:int -> hub -> vmid:int -> unit
  (** [flush_tlb t ~xccs hub ~vmid] invalidates the translation caches
      of [hub] for the VM context [vmid] on every instance ({!vmhubs}
      of them for [Mm], [xccs] — see {!Gfx.xccs} — for [Gc]), after a
      host-data-path flush. A hub whose {!init_hub} has not run yet is
      skipped. [flush_type] defaults to [0]. This is the flush boot
      installs as the memory manager's after-mapping hook (see
      {!Amdev.set_on_range_mapped}). Raises {!Timeout_error} when the
      hub does not acknowledge. *)

  val pf_status_reg : t -> hub -> string
  (** [pf_status_reg t hub] is the name of [hub]'s protection-fault
      status register for the device's generation. *)
end

(** {1:smu Power management}

    The management processor accepts one message at a time through a
    register triple: the response register is cleared, the argument and
    message-id registers are written, and completion is polled as the
    response register turning [1]. Message ids differ per firmware
    interface version; the device's discovered version selects its
    table (see {!Amd_tables.smu}). *)

module Smu : sig
  type t
  (** The type for the power-management block of one device. *)

  val create : Amdev.t -> t
  (** [create adev] prepares the block: selects the message table for
      the device's discovered management-processor version and
      allocates the driver table (the shared-memory buffer table
      transfers fill) from boot memory. Requires [adev] to be booting.
      Raises [Invalid_argument] when no message table covers the
      discovered version. *)

  val driver_table_paddr : t -> int
  (** [driver_table_paddr t] is the device-local physical address of
      the driver table. *)

  val init_hw : t -> unit
  (** [init_hw t] points the firmware at the driver table and enables
      all firmware features. Raises {!Timeout_error} if a message goes
      unanswered for 10 seconds. *)

  val is_smu_alive : t -> bool
  (** [is_smu_alive t] is [true] iff the firmware answers a version
      query within 100ms. *)

  val mode1_reset : t -> unit
  (** [mode1_reset t] asks the firmware to reset the whole device,
      through the message the device's generation expects, and gives
      the hardware 500ms to settle (skipped on multi-die fabrics, which
      reset as a group elsewhere). *)

  val read_table : t -> size:int -> int -> bytes
  (** [read_table t ~size arg] asks the firmware to export the table
      selected by [arg] and is the driver table's first [size] bytes
      after the transfer. *)

  val read_clocks : t -> int list -> (int * int list) list
  (** [read_clocks t clks] is the supported frequency levels of each
      clock domain in [clks] that reports any, queried once and cached
      per [clks]. *)

  val set_clocks : t -> level:int option -> unit
  (** [set_clocks t ~level] pins every adjustable clock domain to the
      frequency at index [level] of its level list, counting from the
      end when negative ([-1] is the highest level). [None] lifts the
      limits instead. Domains whose limit messages go unanswered are
      skipped. *)

  val set_power_limit : t -> float -> unit
  (** [set_power_limit t watts] limits the device's power draw to
      [watts], rounded to whole watts and at least 1. *)

  val aca_read_banks : t -> ue:bool -> int64 list list
  (** [aca_read_banks t ~ue] dumps the hardware-error banks the
      firmware reports: 16 registers per bank, uncorrectable errors
      when [ue], correctable otherwise. The empty list when the
      device's interface version cannot report them. *)
end

(** {1:psp Security processor}

    The security processor gates all engine firmware. Its bring-up has
    two stages, both fed through a 1MB staging buffer in device
    memory: the bootloader mailbox loads the components of the secure
    OS, and once that OS runs, commands submitted over a shared-memory
    ring load every other engine's firmware and set up the trusted
    memory region. Each ring submission is a fixed-size command frame
    whose completion is a fence dword the firmware writes back. *)

module Psp : sig
  type t
  (** The type for the security-processor block of one device. *)

  val create : Amdev.t -> fw:Amdev.Firmware.t -> t
  (** [create adev ~fw] prepares the block for loading the firmware set
      [fw]: picks the mailbox register family of the device's
      security-processor generation and allocates the staging buffer,
      command buffer, fence buffer, submission ring and, on generations
      whose trusted memory region is not reserved by the boot firmware,
      that region too, all from boot memory. Requires [adev] to be
      booting. *)

  val is_sos_alive : t -> bool
  (** [is_sos_alive t] is [true] iff the secure OS reports itself
      running. *)

  val init_hw : t -> unit
  (** [init_hw t] runs the bring-up: loads the secure-OS components
      through the bootloader mailbox unless the OS already runs,
      creates the submission ring, sizes and sets up the trusted memory
      region as the generation requires, and loads every firmware image
      of the set in order (power management first, then the remaining
      engines, ending with the graphics-core loader or its autoload
      trigger). Raises {!Timeout_error} when a mailbox or ring response
      never arrives, [Failure] when the firmware rejects a command. *)

  val spatial_partition_cmd : t -> int -> unit
  (** [spatial_partition_cmd t mode] asks the secure OS to partition
      the device's compute dies into [mode] partitions. *)

  (** {2:layout Boot-memory layout}

      Device-local physical addresses of the block's structures, fixed
      at {!create} time. *)

  val msg1_paddr : t -> int
  (** [msg1_paddr t] is the 1MB firmware staging buffer. *)

  val cmd_paddr : t -> int
  (** [cmd_paddr t] is the command frame buffer. *)

  val fence_paddr : t -> int
  (** [fence_paddr t] is the fence buffer the firmware completes
      submissions into. *)

  val ring_paddr : t -> int
  (** [ring_paddr t] is the submission ring. *)

  val tmr_paddr : t -> int
  (** [tmr_paddr t] is the trusted memory region, or [0] when the boot
      firmware reserves it instead. *)
end

(** {1:gfx Compute engines}

    The compute engines execute queues described by in-memory queue
    descriptors. The block boots the engine processors from the loaded
    firmware, opens their doorbell routes, programs the per-context
    shader-memory configuration, and creates hardware queues by
    building a descriptor in device memory and mirroring it into the
    queue-bringup registers. *)

module Gfx : sig
  type t
  (** The type for the compute-engine block of one device. *)

  val create : Amdev.t -> t
  (** [create adev] allocates the block's queue descriptors from boot
      memory, one per hardware queue slot on every compute die.
      Requires [adev] to be booting. *)

  val xccs : t -> int
  (** [xccs t] is the number of compute dies the device discovered. *)

  val init_hw :
    t ->
    soc:Soc.t ->
    gmc:Gmc.t ->
    psp:Psp.t ->
    fw:Amdev.Firmware.t ->
    partial_boot:bool ->
    unit
  (** [init_hw t ~soc ~gmc ~psp ~fw ~partial_boot] brings the engines
      up: waits for the firmware autoload, programs the graphics hub
      (see {!Gmc.init_hub}), and either resets the compute processors
      ([partial_boot], when the device kept its state from a previous
      session) or runs the full bring-up — pointing the engine
      processors at their instruction start addresses from [fw],
      loading the golden register values, opening the doorbell routes
      through [soc], configuring shader memory for every VM context,
      starting the processors and, on multi-die parts, asking the
      secure OS through [psp] for a single partition. Raises
      {!Timeout_error} when the firmware or a processor does not come
      up. *)

  val reset_mec : t -> fw:Amdev.Firmware.t -> unit
  (** [reset_mec t ~fw] drains and resets the compute processors and
      starts them again from the instruction start addresses in [fw]. *)

  val setup_ring :
    t ->
    ring_addr:int ->
    ring_size:int ->
    rptr_addr:int ->
    wptr_addr:int ->
    eop_addr:int ->
    eop_size:int ->
    idx:int ->
    aql:bool ->
    int
  (** [setup_ring t ~ring_addr ~ring_size ~rptr_addr ~wptr_addr
      ~eop_addr ~eop_size ~idx ~aql] creates hardware queue [idx] over
      the ring at [ring_addr] ([ring_size] bytes, a power of two) with
      the given read/write-pointer and end-of-pipe buffer addresses,
      writing the queue descriptor and activating the queue ([aql]
      selects the architected queuing-language format, and on
      multi-die parts replicates the queue on every die). Returns the
      queue's doorbell index. *)

  val set_clockgating_state : t -> unit
  (** [set_clockgating_state t] enables coarse-grained clock gating on
      the engines. Raises {!Timeout_error} when the firmware does not
      acknowledge safe mode. *)

  val fini_hw : t -> unit
  (** [fini_hw t] drains and deactivates the active hardware queues.
      Raises {!Timeout_error} when a queue does not drain on a healthy
      device. *)
end

(** {1:ih Interrupt rings}

    The interrupt handler block owns the rings the hardware writes
    interrupt entries into. Entries are 8 dwords carrying the client
    and source ids and per-source context; the block decodes and
    reports them, records faults on the device, and keeps the ring's
    read pointer in step. *)

module Ih : sig
  type t
  (** The type for the interrupt block of one device. *)

  val create : Amdev.t -> t
  (** [create adev] allocates two 256KB interrupt rings and their
      pointer buffers from boot memory. Requires [adev] to be
      booting. *)

  val init_hw : t -> unit
  (** [init_hw t] programs and enables both rings: base and size,
      write-pointer writeback for the first ring, overflow behavior,
      the storm-control settings of generations that have them, and
      finally the ring-enable and interrupt-enable bits. *)

  val drain : t -> unit
  (** [drain t] advances the first ring's read pointer to its write
      pointer, discarding pending entries, and clears the overflow
      flag when the ring overflowed. *)

  val interrupt_handler : t -> soc:Soc.t -> gmc:Gmc.t -> smu:Smu.t -> unit
  (** [interrupt_handler t ~soc ~gmc ~smu] drains the first ring,
      decoding and reporting every pending entry with the names from
      [soc]: shader interrupts report their encoding and error type,
      translation faults read and clear the graphics hub's fault
      status through [gmc], and any other source marks the device
      errored (see {!Amdev.is_err_state}). Afterwards checks the bus
      fault lines and, when raised, dumps the hardware-error banks
      through [smu] and clears them. *)
end

(** {1:sdma DMA engines}

    The DMA engines execute copy queues over rings in device memory.
    The block configures the engines' translation and watchdog
    behavior, routes their doorbells, and programs the ring registers
    of each queue. *)

module Sdma : sig
  type t
  (** The type for the DMA-engine block of one device. *)

  val create : Amdev.t -> t
  (** [create adev] prepares the block for the device's DMA-engine
      generation. *)

  val init_hw : t -> soc:Soc.t -> unit
  (** [init_hw t ~soc] configures every engine — watchdog, translation
      client, cache policy, trap enable — releases its halt, and
      routes its doorbell range through [soc]. *)

  val setup_ring :
    t ->
    ring_addr:int ->
    ring_size:int ->
    rptr_addr:int ->
    wptr_addr:int ->
    idx:int ->
    int
  (** [setup_ring t ~ring_addr ~ring_size ~rptr_addr ~wptr_addr ~idx]
      programs queue [idx] over the ring at [ring_addr] ([ring_size]
      bytes, a power of two) with the given read/write-pointer
      addresses, and enables it. Returns the queue's doorbell index.
      Raises [Failure] for a queue index the generation does not
      have. *)

  val fini_hw : t -> unit
  (** [fini_hw t] disables every queue {!setup_ring} created and, on
      generations that need it, pulses the engine soft-reset. *)
end
