(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** AMD GPU runtime.

    {!Program} prepares kernel images and {!Encoded_queue} translates shared
    calls into compute and DMA submissions. Hardware tables are provided by
    {!Amd_tables}; {!Kfd_iface} and {!Pci_iface} allocate memory and create
    mapped queues ({!Queue_desc}). *)

module Hcq = Tolk_hcq.Hcq
module Compiler_amd = Compiler_amd
module System = Tolk_hcq.System
module Amd_tables = Amd_tables
module Amdev = Amdev
module Am_ip = Am_ip
module Am_boot = Am_boot

(** {1:devices Devices} *)

type queue_event = { event_id : int }
(** An interrupt event registered with the driver. [event_id] names the
    event slot a packet can fire to wake waiters. *)

type queue_type =
  | Compute  (** A compute-engine queue fed with type-3 packets. *)
  | Compute_aql  (** A compute-engine queue fed with HSA dispatch packets. *)
  | Sdma  (** A DMA-engine queue fed with byte-granular packets. *)
      (** The type for hardware queue flavors. *)

type ip_versions = {
  gc : int * int * int;  (** Graphics-core version. *)
  sdma : int * int * int;  (** DMA-engine version. *)
  nbif : int * int * int;  (** Bus-interface version. *)
}
(** The type for discovered hardware-block versions. *)

type 'meta device = {
  target : int * int * int;  (** Target graphics version, e.g. [(11, 0, 0)]. *)
  xccs : int;  (** Number of accelerated-compute dies; 1 on consumer chips. *)
  is_aql : bool; (** Whether compute uses HSA dispatch packets. *)
  soc : (module Amd_tables.Soc);  (** Event ids for the generation. *)
  pm4 : (module Amd_tables.Pm4);  (** Compute-packet constants. *)
  sdma : (module Amd_tables.Sdma);  (** DMA-packet constants. *)
  gc : Amd_tables.Ip.t;  (** Graphics-core register family. *)
  nbio : Amd_tables.Ip.t;  (** Bus-interface register family. *)
  max_copy_size : int;
      (** Largest byte count a single DMA copy packet can move. *)
  mutable tmpring_size : int;
      (** Scratch-ring size register value, written verbatim on launch. *)
  mutable scratch : 'meta Hcq.Buffer.t;
      (** Backing store for kernel scratch (private segments). *)
  mutable max_private_segment_size : int;
      (** Largest per-work-item private segment the scratch buffer has
          been sized for; starts at [0]. *)
  is_am : bool;
      (** [true] when this process drives the GPU directly rather than
          through the kernel driver; such devices have no queue event. *)

}
(** The device description the queue builders read. ['meta] is the
    driver metadata carried by the device's buffers. *)

val device :
  target:int * int * int ->
  xccs:int ->
  gc_version:int * int * int ->
  nbio_version:int * int * int ->
  sdma_version:int * int * int ->
  ?is_aql:bool ->
  tmpring_size:int ->
  scratch:'meta Hcq.Buffer.t ->
  is_am:bool ->
  unit ->
  'meta device
(** [device ~target ~xccs ~gc_version ~nbio_version ~sdma_version ...]
    resolves the hardware tables for a chip: event ids and packet
    constants from [target], register families from the discovered
    [gc_version] and [nbio_version] (the [nbif] family on generation 12
    and later), the DMA packet format from [sdma_version], and
    [max_copy_size] ([0x40000000] for DMA revisions from 4.4.2 up to
    5.0.0 exclusive, and from 5.2.0; [0x400000] otherwise).

    Raises [Invalid_argument] when a version has no table. *)

val ensure_has_local_memory :
  'meta device ->
  props:(string * int) list ->
  alloc:(int -> 'meta Hcq.Buffer.t) ->
  free:('meta Hcq.Buffer.t -> unit) ->
  int ->
  unit
(** [ensure_has_local_memory dev ~props ~alloc ~free size] grows [dev]'s
    scratch buffer to cover [max size 128] private bytes per work-item and
    stores the matching register value in [dev.tmpring_size]. Does nothing
    when [dev.max_private_segment_size] already covers that requirement.

    Allocation uses the device topology in [props]: ["simd_count"],
    ["simd_per_cu"], ["array_count"], ["simd_arrays_per_engine"], and
    ["max_slots_scratch_cu"]. A successful replacement retires the previous
    nonempty buffer through [free]. Allocation failure leaves the previous
    backing and sizing state unchanged and propagates to the caller.

    Raises [Failure] when a property is missing or allocation fails. *)

(** {1:queue_desc Mapped queues} *)

(** Hardware queues mapped into the process.

    A descriptor bundles the mappings a submission needs: the command
    ring, the pointers through which producer and consumer positions are
    exchanged, and the doorbell that tells the hardware new work
    arrived. Descriptors come from {!Kfd_iface.create_queue}; tests may
    build them over any mapped memory. *)
module Queue_desc : sig
  type aql = {
    descriptor : Hcq.Mmio.t;
  }
  (** Mapped HSA queue descriptor. *)
  type t = {
    ring : Hcq.Mmio.t;  (** The command ring. *)
    aql : aql option; (** HSA queue descriptor; absent on PM4 and DMA queues. *)
    read_ptr : Hcq.Mmio.t;
        (** 64-bit consumer position, advanced by the device. *)
    write_ptr : Hcq.Mmio.t;
        (** 64-bit producer position used by compiled submission:
            a dword count for PM4, a packet count for AQL and a byte count for DMA. *)
    doorbell : Hcq.Mmio.t;  (** 64-bit doorbell slot of the queue. *)
    hdp_flush : Hcq.Mmio.t option;
        (** Flushes the host-data-path write buffer, run before every
            doorbell write; queues on devices driven without the kernel
            driver need it so host stores to device memory reach the
            engines. *)
    resetup : (unit -> unit) option;
        (** Re-creates the hardware queue with its original parameters;
            fault recovery replays it after the engines were reset. *)
  }
  (** The type for mapped queues. *)

end

(** {1:iface Device interfaces}

    The device runtime reaches the GPU through one of two
    interchangeable interfaces: the Linux kernel driver
    ({!Kfd_iface}) or the driver-less PCI path ({!Pci_iface}). An
    {!Iface.t} is the record of every call the shared runtime makes on
    the selected one. *)

module Iface : sig
  type 'mem t = {
    props : (string * int) list;
        (** Topology properties, e.g. ["simd_count"]. *)
    ip_versions : ip_versions;  (** Discovered hardware-block versions. *)
    is_am : bool;
        (** [true] when the interface drives the GPU directly rather
            than through the kernel driver. *)
    queue_event : queue_event;
        (** The completion event queues fire; unused when [is_am]. *)
    queue_event_mailbox_ptr : nativeint;
        (** The completion event's mailbox slot; [0n] when [is_am]. *)
    alloc :
      ?host:bool ->
      ?uncached:bool ->
      ?cpu_access:bool ->
      int ->
      'mem Hcq.Buffer.t;
        (** Allocates mapped device memory (see {!Kfd_iface.alloc} for
            the flags). *)
    free : 'mem Hcq.Buffer.t -> unit;  (** Releases an allocation. *)
    kind : 'mem Hcq.Buffer.t Type.Id.t;
        (** Shared identity of this interface's raw storage representation. *)
    map : Tolk.Device.Buffer.t -> 'mem Hcq.Buffer.t;
        (** Maps source storage while its owner retains the allocation. *)
    unmap : 'mem Hcq.Buffer.t -> unit;
        (** Releases only an import's mapping or registration. *)
    empty_scratch : 'mem Hcq.Buffer.t;
        (** The zero-sized placeholder a fresh device's scratch starts
            as, before the first sizing. *)
    create_queue :
      queue_type ->
      ring:'mem Hcq.Buffer.t ->
      gart:'mem Hcq.Buffer.t ->
      rptr:int ->
      wptr:int ->
      ?eop_buffer:'mem Hcq.Buffer.t ->
      ?cwsr_buffer:'mem Hcq.Buffer.t ->
      ?ctl_stack_size:int ->
      ?ctx_save_restore_size:int ->
      ?idx:int -> unit -> Queue_desc.t;
        (** Creates a hardware queue (see {!Kfd_iface.create_queue}). [idx]
            selects the PCI SDMA ring and defaults to [0]. KFD assigns its
            hardware ring independently of this logical index. *)
    sleep : int -> unit;
        (** Called from stalled signal waits with the milliseconds since
            the last observed progress; may block briefly, and may raise
            the device's fault report. *)
    on_device_hang : unit -> unit;
        (** Raises [Failure] with the device's fault report; run after a
            wait stalls. May recover the device first. *)
    register :
      (compute_queue:Queue_desc.t ->
      tl:('mem, 'mem device) Hcq.Timeline.t ->
      submission:Hcq.Submission.t ->
      sdma_queues:(unit -> Queue_desc.t list) ->
      unit)
      option;
        (** Hands the interface the device's compute queue, timeline and native submission state
            once they exist, for interrupt collection and fault
            recovery. *)
    after_sync : (unit -> unit) option;
        (** Run after every successful device synchronization. *)
    device_fini : (unit -> unit) option;
        (** Shuts the device down; run at process exit. *)
  }
  (** The type for device interfaces whose allocations carry metadata
      ['mem]. *)
end

(** {1:programs Programs} *)

(** Kernel images prepared for shared queue submission. *)
module Program : sig
  type data = {
    desc_offset : int;
        (** Kernel descriptor byte offset in the image. *)
    entry_offset : int;
        (** Entry point byte offset in the image. *)
    rsrc1 : int;
    rsrc2 : int;
    rsrc3 : int;
        (** Compute resource registers, including architecture and LDS settings. *)
    wave32 : bool;
    enable_private_segment_sgpr : bool;
    enable_dispatch_ptr : bool;
        (** Wave size and enabled user-SGPR inputs from the kernel descriptor. *)
    group_segment_size : int;
        (** Static workgroup-local memory in bytes. *)
    private_segment_size : int;
        (** Per-work-item scratch in bytes. *)
    kernargs_segment_size : int;
        (** Kernel argument segment size in bytes. *)
  }
  (** The descriptor fields used to encode a kernel launch. *)

  val image :
    target:int * int * int ->
    props:(string * int) list -> Bytes.t -> data * Bytes.t
  (** [image ~target ~props lib] is the descriptor and relocated image of
      compiled kernel object [lib], without allocating device storage.
      The image is zero-padded to a four-byte boundary. Queue linking owns
      the image buffer and its lifetime.

      [props] must carry ["lds_size_in_kb"], bounding workgroup-local memory.
      Raises [Failure] if the object has no [.rodata] section, uses a
      relocation other than the 64-bit location-relative form, refers to an
      undefined symbol, or requests more workgroup-local memory than the
      device has; [Invalid_argument] if [lib] is not a loadable object
      (see {!Tolk.Elf.load}). *)
end

(** Compiled host submission over AMD packet templates. *)
module Encoded_queue : sig
  val lower : string -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
  (** [lower name u] replaces timeline polling with bounded native polling
      for the device [name]. *)

  val encode :
    'meta device -> props:(string * int) list -> name:string ->
    compute_ring_size:int -> copy_ring_size:(int -> int option) ->
    Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
  (** [encode dev ~props ~name ~compute_ring_size ~copy_ring_size u] lowers
      queue submission [u] to host code over packet templates. Ring sizes are
      bytes; [copy_ring_size index] supplies the indexed SDMA ring, or [None]
      when unavailable. Returns [None] for other operations. *)
end

(** {1:kfd Kernel-driver interface} *)

(** GPU access through the Linux kernel driver.

    A {!Kfd_iface.t} owns one GPU node: it allocates and maps device
    memory, creates hardware queues, and carries the interrupt events
    used to sleep on completions and to detect faults. Construction and
    every operation raise [Failure] with the system error when the
    driver rejects a request; on systems without the driver,
    {!Kfd_iface.create} and {!Kfd_iface.count} raise [Failure]. *)

module Kfd_iface : sig
  type mem
  (** The type for driver metadata of an allocation: the kernel memory
      handle and the owning GPU. Sub-buffers share their root's
      metadata. *)

  type t
  (** The type for driver interfaces. One value per GPU node. *)

  val count : unit -> int
  (** [count ()] is the number of usable GPU nodes on the system. *)

  val create : device_id:int -> t
  (** [create ~device_id] opens the [device_id]th usable GPU node (in
      stable node order), acquires its virtual-memory space, and
      registers the completion and fault events. Raises [Failure] if
      [device_id] names no node. Failed setup releases the descriptor and
      per-device events. The process-wide event page and its mappings remain
      cached. An ambiguous page-registration failure is latched: KFD may
      already retain the page even when event creation fails. *)

  val props : t -> (string * int) list
  (** [props t] are the node's topology properties, e.g.
      ["simd_count"]. *)

  val ip_versions : t -> ip_versions
  (** [ip_versions t] are the node's discovered hardware-block
      versions. *)

  val queue_event : t -> queue_event
  (** [queue_event t] is the auto-reset event queues fire to wake
      {!sleep}. *)

  val queue_event_mailbox_ptr : t -> nativeint
  (** [queue_event_mailbox_ptr t] is the device address of
      {!queue_event}'s mailbox slot. *)

  (** {2:memory Memory} *)

  val alloc :
    t ->
    ?host:bool ->
    ?uncached:bool ->
    ?cpu_access:bool ->
    ?cpu_addr:nativeint ->
    int ->
    mem Hcq.Buffer.t
  (** [alloc t size] allocates [size] bytes of device memory, maps them
      into this GPU, and is the resulting region. The CPU mapping lives
      at the same virtual address as the device mapping.

      [host] allocates pinned system memory instead of device memory
      (registering the pages at [cpu_addr] when given, fresh ones
      otherwise); [uncached] allocates device-coherent, CPU-uncached
      memory for descriptors and rings; [cpu_access] requests
      host-visible device memory. The buffer carries a CPU view exactly
      when [cpu_access] or [host] is set (all default to [false]).

      Raises [Failure] when memory is exhausted, or, for host-visible
      device memory, when the device's visible aperture is too small
      (resizable BAR disabled). *)

  val free : t -> mem Hcq.Buffer.t -> unit
  (** [free t buf] unmaps [buf]'s root allocation from this GPU and,
      when this GPU owns it, releases the CPU mapping and the memory
      itself. *)

  val map : t -> mem Hcq.Buffer.t -> mem Hcq.Buffer.t
  (** [map t buf] maps a region allocated through another interface
      into this GPU and is the region as visible to it (without a CPU
      view). *)

  (** {2:queues Queues} *)

  val create_queue :
    t ->
    queue_type ->
    ring:mem Hcq.Buffer.t ->
    gart:mem Hcq.Buffer.t ->
    rptr:int ->
    wptr:int ->
    ?eop_buffer:mem Hcq.Buffer.t ->
    ?cwsr_buffer:mem Hcq.Buffer.t ->
    ?ctl_stack_size:int ->
    ?ctx_save_restore_size:int ->
    ?xcc_id:int ->
    unit ->
    Queue_desc.t
  (** [create_queue t kind ~ring ~gart ~rptr ~wptr ()] creates a
      hardware queue of [kind] over the [ring] buffer, with its read and
      write pointers at byte offsets [rptr] and [wptr] of the [gart]
      buffer, and is its mapped descriptor.

      [eop_buffer] backs end-of-pipe events (required for compute
      queues that signal); [cwsr_buffer], [ctl_stack_size] and
      [ctx_save_restore_size] back compute-wave save/restore for
      preemption; [xcc_id] selects the die on multi-die devices
      (defaults to [0]). The queue's priority is read from the
      [AMD_KFD_QUEUE_PRIORITY] environment variable (defaults to
      [7]). *)

  (** {2:events Completion and faults} *)

  val sleep : t -> timeout_ms:int -> unit
  (** [sleep t ~timeout_ms] blocks until a queue fires the completion
      event, a fault is reported, or the timeout elapses. Raises
      [Failure] with the fault report when the device reported a memory
      or hardware fault, now or on a previous call. *)

  val on_device_hang : t -> 'a
  (** [on_device_hang t] raises [Failure] describing the fault the
      device reported, polling for a pending report first. *)

  val iface : t -> mem Iface.t
  (** [iface t] is [t] as the interface record the device runtime
      drives. *)
end

(** {1:pci Driver-less PCI interface} *)

(** GPU access over PCI, with no kernel driver.

    A {!Pci_iface.t} owns one GPU claimed straight from the PCI bus
    (see {!System.Pci_iface_base}): creating it unbinds any kernel
    driver, maps the device's BARs, and boots it — firmware loading,
    memory hubs, security processor, engines (see {!Am_boot}). Device
    memory then comes from the device's own memory manager, hardware
    queues are written directly into engine registers, and faults are
    read from the device's interrupt rings, with an engine reset in
    place of the driver's recovery.

    Everything here needs Linux, root or equivalent capabilities, and
    the device firmware on disk; construction raises [Failure]
    otherwise. The device runtime uses it when explicitly selected or
    when kernel-driver interface initialization fails (see {!create}). *)
module Pci_iface : sig
  type mem = System.Pci_iface_base.mem
  (** The type for driver metadata of an allocation. *)

  type t
  (** The type for driver-less interfaces. One value per GPU. *)

  val vendor : int
  (** [vendor] is the PCI vendor id the interface probes for:
      [0x1002]. *)

  val pci_ids : (int * int list) list
  (** [pci_ids] are the supported PCI device ids as [(mask, ids)]
      pairs (see {!System.pci_scan_bus}): the RDNA3 and RDNA4 consumer
      parts. *)

  val create : device_id:int -> t
  (** [create ~device_id] claims and boots the [device_id]th supported
      GPU on the PCI bus (in bus-address order). Raises [Failure] when
      no such device exists, the device cannot be claimed, or boot
      fails. *)

  val compute_props :
    gc_info:Amdev.gc_info ->
    gc_ver:int * int * int ->
    xccs:int ->
    (string * int) list
  (** [compute_props ~gc_info ~gc_ver ~xccs] synthesizes the topology
      properties the kernel driver would publish for a device with the
      discovered graphics-core geometry [gc_info], graphics-core
      version [gc_ver] and [xccs] compute dies: the [props] of the
      interface record. *)

  val register :
    am:Am_boot.t ->
    compute_queue:Queue_desc.t ->
    tl:('mem, 'mem device) Hcq.Timeline.t ->
    submission:Hcq.Submission.t ->
    sdma_queues:(unit -> Queue_desc.t list) ->
    unit
  (** [register ~am ~compute_queue ~tl ~submission ~sdma_queues] makes a booted device visible
      to {!collect_interrupts}: its interrupt rings are serviced on
      every collection pass, and on recovery its compute queue is
      rebuilt (via [resetup]), its abandoned timeline epoch retired, and
      its native submission failure cleared. [sdma_queues ()] lists existing
      copy queues; recovery requires them to be idle because compute reset
      cannot cancel outstanding copies. The
      device runtime registers each device once its queues exist. *)

  val unregister : Am_boot.t -> unit
  (** [unregister am] removes [am]'s registrations. Failed runtime setup
      unregisters its device before retiring queues. Scripted devices must
      also leave the registry before their mappings do. *)

  val collect_interrupts : ?reset:Am_boot.t -> ?drain_only:bool -> unit -> unit
  (** [collect_interrupts ()] services the interrupt rings of every
      registered device (see {!register}): decoding and reporting
      pending entries ([drain_only] discards them instead), and with
      [reset], recovering only that device: resetting its compute processors,
      re-creating its compute queue, and retiring its abandoned timeline epoch.
      No device is reset by default. [drain_only] defaults to [false]. *)

  val sleep : Am_boot.t -> timeout_ms:int -> unit
  (** [sleep am ~timeout_ms] parks a stalled wait: it blocks on the
      device's interrupt route for at most [timeout_ms] milliseconds
      (see {!System.Pci_device.wait_irq}; skipped for devices without
      one), then collects pending interrupts for every registered
      device. Raises [Failure] when [am] is in the error state
      afterwards, or with a protocol fault report when collection
      itself fails, so waits abort onto the recovery path. *)

  val on_device_hang : Am_boot.t -> 'a
  (** [on_device_hang am] handles a stalled or faulted wait: collects
      pending interrupts, recovers [am] (see {!collect_interrupts}),
      and raises [Failure].
      Recovered devices keep working; the raise reports the hang to
      the caller whose work was lost. *)

  val iface : t -> mem Iface.t
  (** [iface t] is [t] as the interface record the device runtime
      drives. *)
end

(** {1:runtime Device runtime} *)

val create : string -> Tolk.Device.t
(** [create name] opens the AMD GPU named [name] — ["AMD"] for the
    first usable GPU, ["AMD:n"] for the [n]th — and is its device
    runtime. Kernels are compiled with {!Compiler_amd} for the
    discovered architecture (overridable with [DEV=AMD:HIP:gfx1100])
    and dispatched by the shared executor through compiled hardware
    compute queues; host transfers ride the DMA engine when the device
    provides one, and fall back to host-visible device memory
    otherwise.

    With no interface in [DEV], opens the kernel-driver interface
    ({!Kfd_iface}) first and falls back to direct PCI access ({!Pci_iface})
    if interface initialization fails. [DEV=KFD+AMD] or [DEV=PCI+AMD]
    selects only that interface. A later runtime initialization failure
    does not trigger fallback.

    Raises [Failure] when the selected interface cannot open the GPU
    (no driver, no such device, or a failed driver-less boot), when
    the GPU is unsupported (supported: gfx942, gfx950, and the gfx11
    and gfx12 generations, single-die only; the PCI interface covers
    the RDNA3/RDNA4 consumer parts of {!Pci_iface.pci_ids}); [Invalid_argument]
    for an unknown interface, malformed device index, or the deprecated
    [AMD_IFACE] environment variable. After a fault or a stalled
    wait, {!Tolk.Device.synchronize} raises [Failure] with the
    device's fault report; the kernel-driver interface keeps raising
    (the device does not recover), while the driver-less interface
    resets the engines and the device keeps working. *)
