(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** AMD GPU runtime.

    Building blocks for driving AMD GPUs through their hardware command
    queues: generic queue machinery ({!Hcq}), kernel compilation
    ({!Compiler_amd}), hardware tables ({!Amd_tables}), the
    command-stream builders ({!Compute_queue}, {!Copy_queue}) that
    translate work into the packet formats the compute and DMA engines
    execute, and the kernel-driver interface ({!Kfd_iface}) that
    allocates device memory and creates the hardware queues.

    The builders are pure: they read a {!type-device} description and
    append dwords to an in-memory {!Hcq.Q.t}. Their [submit] functions
    copy the accumulated stream into a mapped queue ({!Queue_desc}) and
    ring its doorbell. *)

module Hcq = Hcq
module Compiler_amd = Compiler_amd
module Amd_tables = Amd_tables

(** {1:devices Devices} *)

type queue_event = { event_id : int }
(** An interrupt event registered with the driver. [event_id] names the
    event slot a packet can fire to wake waiters. *)

type 'meta device = {
  target : int * int * int;  (** Target graphics version, e.g. [(11, 0, 0)]. *)
  xccs : int;  (** Number of accelerated-compute dies; 1 on consumer chips. *)
  soc : (module Amd_tables.Soc);  (** Event ids for the generation. *)
  pm4 : (module Amd_tables.Pm4);  (** Compute-packet constants. *)
  sdma : (module Amd_tables.Sdma);  (** DMA-packet constants. *)
  gc : Amd_tables.Ip.t;  (** Graphics-core register family. *)
  nbio : Amd_tables.Ip.t;  (** Bus-interface register family. *)
  max_copy_size : int;
      (** Largest byte count a single DMA copy packet can move. *)
  sqtt_enabled : bool;  (** Thread-trace capture; not supported yet. *)
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
  queue_event_mailbox_ptr : nativeint;
      (** Address the driver polls for the queue event's payload. *)
  queue_event : queue_event;
      (** Event fired to signal completion interrupts. *)
}
(** The device description the queue builders read. ['meta] is the
    driver metadata carried by the device's buffers. *)

val device :
  target:int * int * int ->
  xccs:int ->
  gc_version:int * int * int ->
  nbio_version:int * int * int ->
  sdma_version:int * int * int ->
  ?sqtt_enabled:bool ->
  tmpring_size:int ->
  scratch:'meta Hcq.Buffer.t ->
  is_am:bool ->
  queue_event_mailbox_ptr:nativeint ->
  queue_event:queue_event ->
  unit ->
  'meta device
(** [device ~target ~xccs ~gc_version ~nbio_version ~sdma_version ...]
    resolves the hardware tables for a chip: event ids and packet
    constants from [target], register families from the discovered
    [gc_version] and [nbio_version] (the [nbif] family on generation 12
    and later), the DMA packet format from [sdma_version], and
    [max_copy_size] ([0x40000000] for DMA engines of major version 5 and
    later, [0x400000] before). [sqtt_enabled] defaults to [false].

    Raises [Invalid_argument] when a version has no table. *)

val ensure_has_local_memory :
  'meta device ->
  props:(string * int) list ->
  alloc:(int -> 'meta Hcq.Buffer.t) ->
  free:('meta Hcq.Buffer.t -> unit) ->
  int ->
  unit
(** [ensure_has_local_memory dev ~props ~alloc ~free size] grows [dev]'s
    scratch buffer until it covers a private segment of [size] bytes per
    work-item, and stores the matching scratch-ring register value in
    [dev.tmpring_size]. Does nothing when the scratch already covers
    [size] (per [dev.max_private_segment_size]).

    Growing frees the old buffer through [free] (unless it is empty,
    the state of a freshly created device) and allocates the new one
    through [alloc], sized from the device's compute topology in
    [props]: ["simd_count"], ["simd_per_cu"], ["array_count"],
    ["simd_arrays_per_engine"], and ["max_slots_scratch_cu"]. When
    [alloc] raises [Failure] for the grown size, the old size is
    allocated again and the sizing state is left unchanged, so the
    device stays usable.

    Raises [Failure] when a property is missing or when allocation
    fails without a previous size to fall back to. *)

type 'meta program = {
  dev : 'meta device;  (** Device the program was loaded on. *)
  prog_addr : nativeint;  (** Machine-code address, 256-byte aligned. *)
  rsrc1 : int;  (** COMPUTE_PGM_RSRC1 register value. *)
  rsrc2 : int;  (** COMPUTE_PGM_RSRC2 register value. *)
  rsrc3 : int;  (** COMPUTE_PGM_RSRC3 register value. *)
  wave32 : bool;  (** Dispatch in wave32 mode (generation 10 and later). *)
  enable_private_segment_sgpr : bool;
      (** The kernel expects a flat-scratch descriptor in its first user
          registers. *)
  enable_dispatch_ptr : bool;
      (** The kernel expects a dispatch-packet pointer; not supported
          yet. *)
}
(** The launch parameters of a loaded kernel. *)

(** {1:queue_desc Mapped queues} *)

(** Hardware queues mapped into the process.

    A descriptor bundles the mappings a submission needs: the command
    ring, the pointers through which producer and consumer positions are
    exchanged, and the doorbell that tells the hardware new work
    arrived. Descriptors come from {!Kfd_iface.create_queue}; tests may
    build them over any mapped memory. *)
module Queue_desc : sig
  type t = {
    ring : Hcq.Mmio.t;  (** The command ring. *)
    read_ptr : Hcq.Mmio.t;
        (** 64-bit consumer position, advanced by the device. *)
    write_ptr : Hcq.Mmio.t;
        (** 64-bit producer position, published from [put_value]. *)
    doorbell : Hcq.Mmio.t;  (** 64-bit doorbell slot of the queue. *)
    mutable put_value : int;
        (** End of the submitted stream: a dword count for compute
            rings, a byte count for DMA rings. *)
  }
  (** The type for mapped queues. *)

  val signal_doorbell : t -> unit
  (** [signal_doorbell t] publishes [put_value] to the device: it writes
      the write pointer, fences so all prior ring stores are visible,
      then writes the doorbell. *)
end

(** {1:queues Queue builders} *)

(** Compute-engine command streams.

    Each function appends one logical command to the queue's dword
    stream; {!Compute_queue.q} exposes the accumulated stream for
    submission. Values that do not fit their 32-bit dword raise
    [Invalid_argument] (see {!Hcq.Q.push}). *)
module Compute_queue : sig
  type 'meta t
  (** The type for compute command streams under construction. *)

  val create : 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  (** {2:commands Commands} *)

  val exec :
    'meta t ->
    'meta program ->
    kernargs:'a Hcq.Buffer.t ->
    global_size:int * int * int ->
    local_size:int * int * int ->
    unit
  (** [exec t prg ~kernargs ~global_size ~local_size] launches [prg]
      over a [global_size] grid of [local_size] workgroups, with the
      kernel arguments staged at [kernargs]. The launch invalidates
      stale caches first and drains the pipeline afterwards, so
      successive launches see each other's writes.

      Raises [Invalid_argument] if [prg] wants a dispatch pointer or
      thread-trace capture (neither is supported), or if it wants a
      private-segment descriptor on a multi-die device. *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] flushes caches and writes [value] (defaults to [0])
      to [sg]'s value slot once all prior work retired. For a timeline
      signal owned by a driver-managed device, also fires the owner's
      queue event so blocked waiters wake up. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the queue until [sg]'s value reaches [value]
      (defaults to [0]). *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] records the GPU clock counter in [sg]'s
      timestamp slot once all prior work retired. *)

  val write : 'meta t -> ?b64:bool -> 'a Hcq.Buffer.t -> int64 -> unit
  (** [write t buf v] writes [v] to the start of [buf] once all prior
      work retired: the full 64 bits when [b64] is [true], the low 32
      otherwise (defaults to [false]). *)

  val poll_bit : 'meta t -> 'a Hcq.Buffer.t -> value:int -> mask:int -> unit
  (** [poll_bit t buf ~value ~mask] stalls the queue until the first
      dword of [buf], masked with [mask], equals [value]. *)

  val memory_barrier : 'meta t -> unit
  (** [memory_barrier t] flushes the host-data-path caches and
      invalidates every GPU cache, making host writes visible to
      subsequent commands. *)

  val submit : 'meta t -> Queue_desc.t -> unit
  (** [submit t qd] copies the accumulated stream into [qd]'s ring at
      [put_value], wrapping dword by dword at the ring end, then
      advances [put_value] and rings the doorbell. The stream is kept:
      submitting again replays it.

      On multi-die devices the stream is placed behind an in-ring
      indirect-buffer packet, padded so its body never straddles the
      wrap point, because predication only takes effect inside indirect
      buffers. *)

  (** {2:packets Packet-level interface}

      Building blocks for commands not covered above. *)

  val pkt3 : 'meta t -> int -> int array -> unit
  (** [pkt3 t op payload] appends a type-3 packet: the header for [op]
      sized to [payload], then [payload] itself. *)

  val wreg : 'meta t -> Amd_tables.Reg.t -> int array -> unit
  (** [wreg t reg vals] sets [vals] into the register file starting at
      [reg]: consecutive dwords land in consecutive registers. The
      packet type follows [reg]'s range (shader or universal config).
      Raises [Invalid_argument] for a register outside both ranges. *)

  val wreg_fields : 'meta t -> Amd_tables.Reg.t -> (string * int) list -> unit
  (** [wreg_fields t reg fields] is {!wreg} with the value assembled
      from named fields via {!Amd_tables.Reg.encode}. *)

  val pred_exec : 'meta t -> xcc_mask:int -> (unit -> unit) -> unit
  (** [pred_exec t ~xcc_mask f] runs [f] with the commands it emits
      predicated to the dies selected by [xcc_mask]. On single-die
      devices the commands are emitted unpredicated. *)

  val acquire_mem :
    'meta t ->
    ?addr:nativeint ->
    ?sz:int64 ->
    ?gli:int ->
    ?glm:int ->
    ?glk:int ->
    ?glv:int ->
    ?gl1:int ->
    ?gl2:int ->
    unit ->
    unit
  (** [acquire_mem t ()] stalls until prior work retired and invalidates
      the selected caches for the [sz] bytes at [addr] (defaults: the
      whole address space, every cache). Each [gl*] flag selects one
      cache level; pass [0] to leave it untouched. *)

  val release_mem :
    'meta t ->
    ?address:nativeint ->
    ?value:int64 ->
    ?data_sel:int ->
    ?int_sel:int ->
    ?ctxid:int ->
    ?cache_flush:bool ->
    unit ->
    unit
  (** [release_mem t ()] appends an end-of-pipe event: once prior work
      retired, write the datum selected by [data_sel] ([value], or the
      GPU clock) to [address] and raise the interrupt selected by
      [int_sel], optionally flushing caches first. *)

  val wait_reg_mem :
    'meta t ->
    ?mask:int ->
    ?mem:nativeint ->
    ?reg:int ->
    ?reg_done:int ->
    ?op:int ->
    int ->
    unit
  (** [wait_reg_mem t value] stalls the queue until a masked location
      compares against [value] under [op] (defaults to
      {!wait_reg_mem_function_geq}): the dword at address [mem] when
      given, the register [reg] otherwise. Without [mem], a non-zero
      [reg_done] register is written back when the wait completes. *)

  val wait_reg_mem_function_eq : int
  (** Comparison: masked location equals the reference value. *)

  val wait_reg_mem_function_geq : int
  (** Comparison: masked location is at least the reference value. *)
end

(** DMA-engine command streams.

    Each function appends one packet to the queue's dword stream;
    {!Copy_queue.q} exposes the accumulated stream and
    {!Copy_queue.cmd_sizes} its packet boundaries, which submission
    needs to split the stream across a ring's wrap point. *)
module Copy_queue : sig
  type 'meta t
  (** The type for DMA command streams under construction. *)

  val create : ?max_copy_size:int -> 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. [max_copy_size] caps
      the bytes per copy packet and defaults to the device's. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  val cmd_sizes : 'meta t -> int list
  (** [cmd_sizes t] is the dword count of each packet, in stream
      order. *)

  val copy :
    'meta t -> dest:'a Hcq.Buffer.t -> src:'b Hcq.Buffer.t -> int -> unit
  (** [copy t ~dest ~src size] copies [size] bytes from the start of
      [src] to the start of [dest], split into as many packets as the
      queue's [max_copy_size] requires. *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] writes [value] (defaults to [0]) to [sg]'s value
      slot once prior packets completed. For a timeline signal owned by
      a driver-managed device, also writes the owner's event mailbox and
      fires its queue event. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the engine until [sg]'s value reaches [value]
      (defaults to [0]). *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] records the global engine clock in [sg]'s
      timestamp slot. *)

  val write : 'meta t -> ?b64:bool -> 'a Hcq.Buffer.t -> int64 -> unit
  (** [write t buf v] writes [v] to the start of [buf]: the full 64 bits
      when [b64] is [true], the low 32 otherwise (defaults to
      [false]). *)

  val submit : 'meta t -> Queue_desc.t -> unit
  (** [submit t qd] copies the accumulated packets into [qd]'s ring at
      [put_value], advances [put_value] (in bytes) and rings the
      doorbell. The engine fetches packets as units, so a packet never
      straddles the ring end: when the next packet would, the remaining
      tail is zero-filled and the stream continues at the ring start.
      Blocks until the device has consumed enough of the ring for the
      stream to fit. The stream is kept: submitting again replays it.

      Raises [Invalid_argument] if the stream cannot fit in the ring at
      all. *)
end

(** {1:programs Programs} *)

(** Loaded kernels.

    Loading lays a compiled kernel object out in device memory and
    derives its launch parameters; {!Program.call} then stages the
    arguments for one launch and drives it through a mapped compute
    queue. *)
module Program : sig
  type 'meta t = {
    params : 'meta program;
        (** Launch parameters, as consumed by {!Compute_queue.exec}. *)
    name : string;  (** Kernel name, for diagnostics. *)
    lib_gpu : 'meta Hcq.Buffer.t;
        (** Device memory holding the kernel image. *)
    group_segment_size : int;
        (** Static workgroup-local memory, in bytes. *)
    private_segment_size : int;
        (** Per-work-item scratch the kernel needs, in bytes. The
            device's scratch must be sized for it (see
            {!ensure_has_local_memory}) before launching. *)
    kernargs_segment_size : int;
        (** Kernel-argument bytes the kernel reads. *)
    kernargs_alloc_size : int;
        (** Bytes staged per launch: the argument segment plus space for
            launch metadata. *)
  }
  (** The type for loaded kernels. *)

  val load :
    'meta device ->
    alloc:(int -> 'meta Hcq.Buffer.t) ->
    props:(string * int) list ->
    name:string ->
    Bytes.t ->
    'meta t
  (** [load dev ~alloc ~props ~name lib] loads the compiled kernel
      object [lib] (a shared object, as produced by {!Compiler_amd})
      onto [dev]: it lays the object's sections out into a flat image,
      resolves the object's internal relocations, copies the image into
      device memory obtained from [alloc] (which must return a
      CPU-mapped buffer of at least the requested size, a multiple of
      [0x1000]), and parses the kernel descriptor at the start of the
      object's [.rodata] section into launch parameters.

      [props] must carry ["lds_size_in_kb"], bounding the
      workgroup-local memory a kernel may request.

      Raises [Failure] if the object has no [.rodata] section, uses a
      relocation other than the 64-bit location-relative form, refers to
      an undefined symbol, or requests more workgroup-local memory than
      the device has; [Invalid_argument] if [lib] is not a loadable
      object (see {!Tolk.Elf.load}). *)

  val free : free:('meta Hcq.Buffer.t -> unit) -> 'meta t -> unit
  (** [free ~free t] releases the device memory holding [t]'s image
      through [free]. [t] must have no launches in flight. *)

  val call :
    'meta t ->
    kernargs:'a Hcq.Kernargs.t ->
    queue:Queue_desc.t ->
    timeline:('b, 'meta device) Hcq.Signal.t ->
    timeline_value:int ->
    ?wait:('c, 'meta device) Hcq.Signal.t * ('d, 'meta device) Hcq.Signal.t ->
    ?timeout_ms:int ->
    bufs:nativeint array ->
    vals:int array ->
    global_size:int * int * int ->
    local_size:int * int * int ->
    unit ->
    float option
  (** [call t ~kernargs ~queue ~timeline ~timeline_value ~bufs ~vals
      ~global_size ~local_size ()] enqueues one launch of [t]: it stages
      [bufs] and [vals] into a fresh slot of [kernargs], then submits to
      [queue] a stream that waits for the device's previous work
      ([timeline] reaching [timeline_value - 1]), makes host writes
      visible, launches the kernel over a [global_size] grid of
      [local_size] workgroups, and signals [timeline] with
      [timeline_value] once the launch retired. [timeline_value] must be
      at least [1]; the caller owns the counter and submits the next
      launch with the next value.

      [wait], when given, brackets the launch with clock captures into
      the two signals, blocks until [timeline] reaches [timeline_value]
      ([timeout_ms] bounds the wait, see {!Hcq.Signal.wait}), and
      returns the seconds elapsed between the two captures. Otherwise
      the call returns [None] without blocking.

      Raises [Invalid_argument] if [t] expects a dispatch-packet pointer
      (not supported), or from the argument and queue builders when a
      value does not fit its slot. *)
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

  type ip_versions = {
    gc : int * int * int;  (** Graphics-core version. *)
    sdma : int * int * int;  (** DMA-engine version. *)
    nbif : int * int * int;  (** Bus-interface version. *)
  }
  (** The type for discovered hardware-block versions. *)

  val count : unit -> int
  (** [count ()] is the number of usable GPU nodes on the system. *)

  val create : device_id:int -> t
  (** [create ~device_id] opens the [device_id]th usable GPU node (in
      stable node order), acquires its virtual-memory space, and
      registers the completion and fault events. Raises [Failure] if
      [device_id] names no node. *)

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

  type queue_type =
    | Compute  (** A compute-engine queue fed with type-3 packets. *)
    | Sdma  (** A DMA-engine queue fed with byte-granular packets. *)
  (** The type for hardware queue flavors. *)

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
end
