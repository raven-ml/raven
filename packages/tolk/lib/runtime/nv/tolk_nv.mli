(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPU runtime.

    Building blocks for driving NVIDIA GPUs through their hardware
    command queues: generic queue machinery ({!Hcq}), the generated
    driver tables ({!Nv_tables}), kernel launch descriptors ({!Qmd}),
    the command-stream builders ({!Compute_queue}, {!Copy_queue}) that
    translate work into the method streams the compute and copy engines
    execute, the driver interface ({!Nv_iface}) with its kernel-driver
    implementation ({!Nvk_iface}), and kernel loading and dispatch
    ({!Program}).

    The builders are pure: they read a {!type-device} description,
    append dwords to an in-memory {!Hcq.Q.t}, and patch launch
    descriptors through their CPU mappings. Their [submit] functions
    stage the accumulated stream in the device's command buffer, point
    a channel ring entry ({!Queue_desc}) at it, and ring the
    work-submission doorbell. *)

module Hcq = Tolk_hcq.Hcq
module Nv_tables = Nv_tables
module Nvdev = Nvdev
module Ip = Ip

(** {1:qmd Launch descriptors} *)

(** Kernel launch descriptors.

    The compute engine consumes launches as fixed-size descriptors of
    packed bitfields: grid and block geometry, program and constant
    buffer addresses, dependency links, and release semaphores. A
    program template lives in managed host bytes; each launch copies it
    into mapped device memory. A descriptor accesses either backing through
    fields addressed by the names of
    the generated tables ({!Nv_tables.Defs}). *)
module Qmd : sig
  type t
  (** The type for launch descriptors. *)

  val sizeof : compute_class:int -> int
  (** [sizeof ~compute_class] is the descriptor size in bytes for the
      layout [compute_class] consumes: [0x100] before Blackwell,
      [0x180] from Blackwell on. *)

  val create : view:Hcq.Mmio.t -> compute_class:int -> t
  (** [create ~view ~compute_class] is the descriptor stored in the
      first [sizeof ~compute_class] bytes of [view], read and written
      in place. Raises [Invalid_argument] if [view] is smaller than the
      descriptor. *)

  val version : t -> int
  (** [version t] is the descriptor layout version: [3] before
      Blackwell, [5] from Blackwell on. *)

  val read : t -> string -> int
  (** [read t name] is the value of field [name]. Field names are
      case-insensitive; per-slot fields carry their slot index as a
      suffix (for example ["release0_enable"]). Raises
      [Invalid_argument] on an unknown name. *)

  val write : t -> (string * int) list -> unit
  (** [write t fields] sets each named field to its value, in list
      order. Raises [Invalid_argument] on an unknown name or a value
      that does not fit the field's width. *)

  val field_offset : t -> string -> int
  (** [field_offset t name] is the byte offset of the first byte of
      field [name] within the descriptor. Raises [Invalid_argument] on
      an unknown name. *)

  val set_constant_buf_addr : t -> int -> nativeint -> unit
  (** [set_constant_buf_addr t i addr] binds constant buffer slot [i]
      to device address [addr]. Version 5 descriptors store the
      address shifted right by 6, so [addr] must be 64-byte aligned
      there. *)

  val to_bytes : t -> bytes
  (** [to_bytes t] is a copy of the descriptor's bytes. *)
end

(** {1:devices Devices} *)

type 'meta device = {
  compute_class : int;
      (** Compute engine class id; selects the descriptor layout. *)
  dma_class : int;  (** Copy engine class id. *)
  gpfifo_class : int;  (** Channel class id. *)
  sass_version : int;  (** Shader ISA version of the chip. *)
  mutable slm_per_thread : int;
      (** Per-thread local-memory bytes the device is currently sized
          for; starts at [0]. *)
  mutable shader_local_mem : Tolk.Device.Buffer.t option;
      (** Backing store for kernel local memory; absent until
          {!ensure_has_local_memory} first grows it. *)
  shared_mem_window : nativeint;
      (** Virtual-address window shared-memory accesses go through. *)
  local_mem_window : nativeint;
      (** Virtual-address window local-memory accesses go through. *)
  cmdq_page : 'meta Hcq.Buffer.t;
      (** CPU-mapped device memory where submissions stage their
          command streams. *)
  mutable cmdq_position : int;
      (** Absolute byte position for the next direct command stream. *)
  cmdq_pending : (int * Hcq.Mmio.t * int64) Stdlib.Queue.t;
      (** Staging regions and channel sequences still awaiting retirement. *)
  submission : Hcq.Submission.t;
      (** Failure state shared by direct and compiled submissions. *)
  cmdq : Hcq.Mmio.t;  (** CPU view of [cmdq_page]. *)
  gpu_mmio : Hcq.Mmio.t;
      (** Usermode register region carrying the work-submission
          doorbell. *)
}
(** The device description the queue builders read. ['meta] is the
    driver metadata carried by the device's buffers. *)

val device :
  compute_class:int ->
  dma_class:int ->
  gpfifo_class:int ->
  sass_version:int ->
  ?slm_per_thread:int ->
  shared_mem_window:nativeint ->
  local_mem_window:nativeint ->
  cmdq_page:'meta Hcq.Buffer.t ->
  gpu_mmio:Hcq.Mmio.t ->
  unit ->
  'meta device
(** [device ~compute_class ~dma_class ~gpfifo_class ~sass_version
    ~shared_mem_window ~local_mem_window ~cmdq_page ~gpu_mmio ()] is a
    device description over the given engine classes and mappings. The
    command-stream allocator wraps over [cmdq_page], whose CPU view
    must exist. [slm_per_thread] defaults to [0] and [shader_local_mem]
    starts absent.

    Raises [Invalid_argument] if [cmdq_page] has no CPU view. *)

(** {1:programs Programs} *)

type 'meta program = {
  dev : 'meta device;  (** Device the program was loaded on. *)
  qmd : Qmd.t;
      (** Launch descriptor template: the geometry-independent fields,
          filled at load time. {!Compute_queue.exec} copies it behind
          the staged arguments and patches the per-launch fields into
          the copy. *)
  cbuf0_size : int;
      (** Size of the kernel's first constant buffer in bytes. Kernel
          arguments are staged in it, and the descriptor copy lands at
          the next 256-byte boundary after it. *)
}
(** The launch parameters of a loaded kernel. *)

(** {1:queue_desc Mapped queues} *)

(** Hardware channels mapped into the process.

    A descriptor bundles what a submission needs: the channel's entry
    ring, its put pointer, and the token that identifies the channel
    to the work-submission doorbell. Tests may build descriptors over
    any mapped memory. *)
module Queue_desc : sig
  type t = {
    ring : Hcq.Mmio.t;
        (** The channel's entry ring: 64-bit entries, each pointing at
            a staged command stream. *)
    gpput : Hcq.Mmio.t;
        (** 32-bit producer position, published after each entry. *)
    progress : Hcq.Mmio.t;
        (** Host submitted sequence at byte 0, GPU completed low dword at byte 8. *)
    progress_addr : nativeint;
        (** GPU address of [progress], which may differ from its CPU mapping. *)
    token : int;
        (** Work-submission token naming the channel to the
            doorbell. *)
  }
  (** The type for mapped channels. *)
end

(** {1:queues Queue builders} *)

(** Compute-engine command streams.

    Each function appends one logical command to the queue's dword
    stream; {!Compute_queue.q} exposes the accumulated stream for
    submission. Values that do not fit their 32-bit dword raise
    [Invalid_argument] (see {!Hcq.Q.push}).

    Successive launches coalesce: while a launch is pending, the next
    {!Compute_queue.exec} links itself into the pending descriptor as
    its dependent instead of appending stream methods, and
    {!Compute_queue.signal} rides in a free release slot of the
    pending descriptor instead of appending a semaphore method.
    {!Compute_queue.wait}, {!Compute_queue.write},
    {!Compute_queue.poll_bit} and {!Compute_queue.memory_barrier} end
    the pending launch, so later commands go back to the stream. *)
module Compute_queue : sig
  type 'meta t
  (** The type for compute command streams under construction. *)

  val create : 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  val setup :
    'meta t ->
    ?compute_class:int ->
    ?local_mem_window:nativeint ->
    ?shared_mem_window:nativeint ->
    ?local_mem:nativeint ->
    ?local_mem_tpc_bytes:int ->
    unit ->
    unit
  (** [setup t ()] appends the engine set-up methods for each argument
      given: bind the compute class to the channel, set the two
      virtual-address windows, and point the engine at the
      local-memory backing store and its per-TPC size. *)

  val exec :
    'meta t ->
    'meta program ->
    kernargs:'a Hcq.Buffer.t ->
    global_size:int * int * int ->
    local_size:int * int * int ->
    unit
  (** [exec t prg ~kernargs ~global_size ~local_size] launches [prg]
      over a [global_size] grid of [local_size] blocks, with the
      kernel arguments staged at the start of [kernargs]. The launch
      descriptor is copied into [kernargs] at the 256-byte boundary
      after the argument bytes ([prg.cbuf0_size]) and its geometry and
      constant-buffer-0 address are patched into the copy, so
      [kernargs] must be CPU-mapped and have room for the descriptor.

      Raises [Invalid_argument] if the descriptor's device address
      does not fit in 40 bits, or if a dimension does not fit its
      field (32 bits per grid dimension, 16 bits for the first two
      block dimensions, 8 bits for the third). *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] writes [value] (defaults to [0]) to [sg]'s value
      slot once all prior work retired. After a launch, the release is
      carried by the launch descriptor when one of its two release
      slots is free; otherwise a semaphore-release method is appended,
      which also stamps [sg]'s timestamp and raises a non-stalling
      interrupt. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the channel until [sg]'s value reaches
      [value] (defaults to [0]), comparing 64-bit values with
      wrap-around. *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] releases [sg] with value [0], stamping its
      timestamp slot. *)

  val write : 'meta t -> ?b64:bool -> 'a Hcq.Buffer.t -> int64 -> unit
  (** [write t buf v] writes [v] to the start of [buf] once all prior
      work retired: the full 64 bits when [b64] is [true], the low 32
      otherwise (defaults to [false]). *)

  val poll_bit : 'meta t -> 'a Hcq.Buffer.t -> value:int -> mask:int -> unit
  (** [poll_bit t buf ~value ~mask] stalls the channel until the bits
      selected by [mask] in the first dword of [buf] are all set
      ([value = mask]) or all clear ([value = 0]). *)

  val memory_barrier : 'meta t -> unit
  (** [memory_barrier t] invalidates the engine's instruction, global
      data, and constant caches, making prior memory writes visible to
      subsequent launches. *)

  val submit : 'meta t -> Queue_desc.t -> unit
  (** [submit t qd] stages the accumulated stream in the device's
      command buffer, writes the next ring entry of [qd] to point at
      it, publishes the new put position, fences, and rings the
      work-submission doorbell with [qd]'s token. A trailing engine release
      retires the channel sequence. FIFO capacity and staging reuse wait for
      retirement with [HCQ_TIMEOUT_MS] (default 30 seconds); timeout latches
      failure without overwriting live commands. The stream is kept:
      submitting again stages it again. Submissions must be serialized. *)
end

(** Copy-engine command streams.

    Each function appends one logical command to the queue's dword
    stream; {!Copy_queue.q} exposes the accumulated stream for
    submission. *)
module Copy_queue : sig
  type 'meta t
  (** The type for copy command streams under construction. *)

  val create : 'meta device -> 'meta t
  (** [create dev] is an empty stream for [dev]. *)

  val q : 'meta t -> Hcq.Q.t
  (** [q t] is the underlying dword stream. *)

  val setup : 'meta t -> ?copy_class:int -> unit -> unit
  (** [setup t ()] binds [copy_class], when given, to the channel. *)

  val copy :
    'meta t -> dest:'a Hcq.Buffer.t -> src:'b Hcq.Buffer.t -> int -> unit
  (** [copy t ~dest ~src size] copies [size] bytes from the start of
      [src] to the start of [dest], split into transfers of at most
      2 GiB. *)

  val signal : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [signal t sg] writes [value] (defaults to [0]) to [sg]'s value
      slot once prior transfers completed, stamping its timestamp
      slot. *)

  val wait : 'meta t -> ?value:int -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [wait t sg] stalls the channel until [sg]'s value reaches
      [value] (defaults to [0]), comparing 64-bit values with
      wrap-around. *)

  val timestamp : 'meta t -> ('a, 'meta device) Hcq.Signal.t -> unit
  (** [timestamp t sg] releases [sg] with value [0], stamping its
      timestamp slot. *)

  val submit : 'meta t -> Queue_desc.t -> unit
  (** [submit t qd] stages the accumulated stream and rings the
      doorbell, exactly as {!Compute_queue.submit}. *)
end

(** {1:iface Driver interfaces} *)

(** The driver interface a device runs on.

    A {!Nv_iface.t} bundles everything the runtime asks of a driver:
    object allocation and control calls, memory allocation and mapping,
    and the channel set-up hooks. The kernel-driver implementation is
    {!Nvk_iface}; further implementations can be added without changing
    any call site. *)
module Nv_iface : sig
  exception Out_of_memory of string
  (** Raised by [rm_alloc] and [alloc] when the driver reports that no
      memory is available, so allocation caches can be flushed and the
      allocation retried. Any other driver error raises [Failure]. *)

  type nvdev = ..
  (** The type for driver-less device implementations. An interface
      that programs the hardware directly carries its implementation
      here; the kernel-driver interface carries none. *)

  type usermode = {
    handle : int;  (** Object handle of the usermode region. *)
    mmio : Hcq.Mmio.t;
        (** CPU mapping of the usermode register region, carrying the
            work-submission doorbell. *)
    compute_class : int;  (** Newest supported compute engine class. *)
    dma_class : int;  (** Newest supported copy engine class. *)
    gpfifo_class : int;  (** Newest supported channel class. *)
  }
  (** The result of [setup_usermode]: the mapped doorbell region and
      the engine classes probed from the device. *)

  type 'mem t = {
    root : int;  (** Handle of the client every driver object hangs off. *)
    gpu_instance : int;  (** Driver instance index of the device. *)
    count : int;
        (** Number of devices this interface kind can open in this
            system. *)
    defs : Nv_defs_versions.t;
        (** Layouts of the parameter structures whose shape depends on
            the driver generation behind the interface (see
            {!Nv_tables.defs_for_driver}). *)
    set_device : nvdevice:int -> subdevice:int -> virtmem:int -> unit;
        (** [set_device ~nvdevice ~subdevice ~virtmem] hands the
            interface the handles of the device, subdevice and virtual
            memory objects allocated on it. Must be called once, before
            [alloc], [free], [map] or any [setup_] function. *)
    rm_alloc : parent:int -> cls:int -> ?params:Nv_tables.blob -> unit -> int;
        (** [rm_alloc ~parent ~cls ?params ()] allocates a driver
            object of class [cls] under object [parent] and is its
            handle. [params] is the class's allocation parameter
            structure, read and updated by the driver in place. Raises
            {!Out_of_memory} when the driver is out of memory,
            [Failure] on any other driver error. *)
    rm_control : obj:int -> cmd:int -> ?params:Nv_tables.blob -> unit -> unit;
        (** [rm_control ~obj ~cmd ?params ()] invokes control command
            [cmd] on object [obj]. [params] is the command's parameter
            structure, read and updated by the driver in place. Raises
            [Failure] on driver errors. *)
    alloc :
      ?host:bool ->
      ?uncached:bool ->
      ?cpu_access:bool ->
      ?contiguous:bool ->
      ?force_devmem:bool ->
      ?map_flags:int ->
      ?cpu_addr:nativeint ->
      int ->
      'mem Hcq.Buffer.t;
        (** [alloc size] allocates [size] bytes of device-visible
            memory, rounded up to the allocation page size, and maps
            them at a fresh virtual address. [host] registers CPU
            memory instead of allocating device memory; [uncached]
            allocates GPU-uncacheable system pages; [cpu_access] also
            maps device memory for the CPU, so the buffer has a view;
            [contiguous] requires physically contiguous pages;
            [force_devmem] prevents the PCI interface from placing CPU-visible
            device allocations in host memory when the BAR is small;
            [map_flags] adds driver flags to the CPU mapping;
            [cpu_addr] reuses the existing CPU mapping at that address
            instead of reserving a fresh range. All default to [false],
            [0] or absent. Raises {!Out_of_memory} and [Failure] as
            [rm_alloc]. *)
    free : 'mem Hcq.Buffer.t -> unit;
        (** [free buf] releases an allocation made by this interface,
            including its virtual range and any owned CPU mapping.
            Use [unmap] for imported storage. *)
    kind : 'mem Hcq.Buffer.t Type.Id.t;
        (** Identity shared by interfaces with compatible raw storage. *)
    hmemory : 'mem Hcq.Buffer.t -> int;
        (** Backing handle or physical address for channel setup. *)
    map : Tolk.Device.Buffer.t -> 'mem Hcq.Buffer.t;
        (** Maps source storage without taking ownership of its allocation. *)
    unmap : 'mem Hcq.Buffer.t -> unit;
        (** Releases an import's GPU mapping or host registration. *)
    setup_usermode : unit -> usermode;
        (** [setup_usermode ()] probes the device's engine classes and
            maps the usermode register region. *)
    setup_vm : vaspace:int -> unit;
        (** [setup_vm ~vaspace] registers the device and its address
            space object [vaspace] with the driver's memory manager. *)
    setup_gpfifo_vm : gpfifo:int -> unit;
        (** [setup_gpfifo_vm ~gpfifo] registers channel [gpfifo] with
            the driver's memory manager. *)
    sleep : int -> unit;
        (** [sleep ms] may yield the CPU to the driver for up to [ms]
            milliseconds while a signal wait spins; on drivers with no
            wait channel it returns immediately. *)
    device_fini : unit -> unit;
        (** [device_fini ()] releases interface resources at device
            shutdown. *)
    nvdev : nvdev option;
        (** The driver-less implementation behind the interface, when
            it is one (see {!is_nvd}). *)
  }
  (** The type for the driver interface of one device. *)

  val is_nvd : 'mem t -> bool
  (** [is_nvd t] is [true] if [t] programs the hardware directly rather
      than going through the kernel driver. *)
end

(** The kernel-driver interface.

    Drives devices through the resident kernel driver: objects and
    memory through control-device escape calls, virtual-address ranges
    and mappings through the memory-manager device. Driver-wide state
    (the device files, the root client, the installed driver's
    parameter-structure generation and the visible cards) is shared by
    every device in the process and set up by the first {!iface} call.

    Driver calls require Linux and the resident driver; elsewhere they
    raise [Failure]. The parameter-structure constructors and the
    virtual-address allocator are pure and run anywhere, so tests can
    pin the wire formats without a device. *)
module Nvk_iface : sig
  type mem
  (** Kernel-driver allocation and import metadata. *)

  val iface : device_id:int -> mem Nv_iface.t
  (** [iface ~device_id] is the kernel-driver interface of the
      [device_id]th visible device. The first call opens the driver:
      it creates the root client, detects the driver generation,
      initializes the memory manager and enumerates the cards. Raises
      [Failure] if the driver is unavailable or [device_id] is out of
      range. *)

  val is_initialized : unit -> bool
  (** [is_initialized ()] is [true] once the kernel driver has been
      opened in this process (see {!iface}). The driver-less interface
      refuses to open after this, since its fixed mappings would overwrite
      the kernel driver's. *)

  val alloc_gpu_vaddr : ?alignment:int -> ?force_low:bool -> int -> nativeint
  (** [alloc_gpu_vaddr size] reserves [size] bytes of the process's
      device virtual address space and is the range's base address, a
      multiple of [alignment] (defaults to 4 KiB). The space splits at
      [0x2000000000]: [force_low] (defaults to [false]) reserves below
      the split, where CPU-visible mappings live; the default reserves
      above it. Addresses are process-global and never reused. Raises
      [Out_of_memory] when the requested range is exhausted. *)

  (** {2:wire Wire formats}

      Constructors for the driver parameter structures whose layout
      the interface composes itself, exposed so tests can pin the
      composition byte for byte. *)

  val driver_version_major : Nv_tables.blob -> int
  (** [driver_version_major b] is the major component of the
      NUL-terminated dotted driver version at the start of the
      build-version parameter structure [b] — [580] for
      ["580.65.06"] — the value {!Nv_tables.defs_for_driver} maps to a
      parameter-structure generation. Raises [Failure] if the version
      does not parse. *)

  val nvos21_params :
    root:int ->
    parent:int ->
    cls:int ->
    ?params:Nv_tables.blob ->
    unit ->
    Nv_tables.blob
  (** [nvos21_params ~root ~parent ~cls ?params ()] is the
      object-allocation envelope: client [root], parent object, class,
      and the address of the class's [params] structure when given.
      Keep [params] live for as long as the envelope may reach the
      driver. *)

  val memory_allocation_params :
    root:int ->
    size:int ->
    page_size:int ->
    uncached:bool ->
    contiguous:bool ->
    read_only:bool ->
    int * Nv_tables.blob
  (** [memory_allocation_params ~root ~size ~page_size ~uncached
      ~contiguous ~read_only] is the memory class to allocate and its
      allocation parameter structure for [size] bytes in pages of
      [page_size]: [uncached] selects GPU-uncacheable system pages
      under the notifier type, otherwise cacheable device pages under
      the image type; a [page_size] above 4 KiB adds the huge-page
      attributes; [read_only] restricts user mappings to reads. *)

  val map_external_params :
    rm_ctrl_fd:int ->
    root:int ->
    va:nativeint ->
    size:int ->
    mem_handle:int ->
    gpu_uuid:bytes ->
    Nv_tables.blob
  (** [map_external_params ~rm_ctrl_fd ~root ~va ~size ~mem_handle
      ~gpu_uuid] is the memory-manager mapping request binding
      [mem_handle]'s [size] bytes at virtual address [va] for the
      device identified by the 16-byte [gpu_uuid], with a single
      mapping-attribute entry. Raises [Invalid_argument] if [gpu_uuid]
      is not 16 bytes. *)
end

(** The driver-less PCI interface.

    Drives a supported NVIDIA GPU with no kernel driver: it takes
    exclusive ownership of the device over PCI, boots the GSP firmware
    (see {!Ip}), and serves the runtime's object, control and memory
    operations as GSP remote procedure calls over the device's own
    memory manager. It fills the same {!Nv_iface.t} the kernel-driver
    interface does, so the rest of the runtime is unchanged on top.

    Opening the device programs its firmware and engine registers
    directly, so it requires Linux and PCI access rights. The runtime uses
    this interface when explicitly selected or when kernel-driver interface
    initialization fails (see {!create}). The device probe below is pure
    and runs anywhere. *)
module Pci_iface : sig
  type t
  (** The type for driver-less PCI interfaces. *)

  val vendor : int
  (** [vendor] is the PCI vendor id the device probe matches (NVIDIA). *)

  val pci_ids : (int * int list) list
  (** [pci_ids] is the device-id allowlist: [(mask, ids)] pairs whose
      [ids] are the masked device ids of the supported consumer parts. *)

  val create : device_id:int -> t
  (** [create ~device_id] takes exclusive ownership of the [device_id]th
      matching device on the bus and boots its GSP firmware end to end.
      Raises [Failure] when the kernel driver is already open in this
      process (its mappings would be corrupted), when no matching device
      has that index, when the device cannot be claimed, or on a firmware
      or boot failure. *)

  val iface : t -> Tolk_hcq.System.Pci_iface_base.mem Nv_iface.t
  (** [iface t] is the runtime interface over [t]: object and control
      calls as GSP remote procedure calls, memory through the device's
      memory manager, the work-submission doorbell in BAR0, and status
      drained on each wait. *)
end

(** {1:loading Kernel images} *)

(** Cubin images and address-independent launch templates. Device storage and
    launch arguments are owned by shared compiled submissions. *)
module Program : sig
  type data = private {
    image : bytes; (** Section contents followed by a zeroed 4 KiB prefetch guard. *)
    relocations : (int * int * int * int) list;
        (** Byte offset, target image offset, byte width and right shift. *)
    prog_offset : int; (** Code offset in [image], in bytes. *)
    prog_size : int; (** Code size, in bytes. *)
    regs_usage : int; (** Registers per thread. *)
    shmem_usage : int; (** Shared-memory bytes per block. *)
    lcmem_usage : int; (** Local-memory bytes per thread. *)
    constbufs : (int * (int * int)) list;
        (** Constant banks with their image offsets and byte sizes. *)
    cbuf0_size : int; (** Driver-parameter prefix size, in bytes. *)
  }
  (** Parsed kernel metadata. Relocations remain relative until linking. *)

  val image : name:string -> bytes -> data
  (** [image ~name lib] parses [name] from the cubin [lib], without allocating
      device storage or changing local-memory backing.

      Raises [Failure] for unsupported relocations, undefined symbols or unknown
      descriptor formats, and [Invalid_argument] for malformed objects. *)

  val template : 'meta device -> data -> Qmd.t * int array
  (** [template dev data] is a fresh descriptor and driver-parameter prefix for
      [data] on [dev]. Image, constant-buffer and launch addresses remain unset;
      the compiled queue patches them when linked. The descriptor uses
      [dev.slm_per_thread] for local-memory sizing. *)
end

module Encoded_queue : sig
  val encode :
    'meta device -> name:string -> compute_entries:int -> copy_entries:int ->
    compute_token:int -> copy_token:int -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
  (** [encode dev ~name ~compute_entries ~copy_entries ~compute_token ~copy_token u]
      lowers a queue submission to relocatable command storage and host calls.
      Device placement and executable addresses are resolved when linked. *)

  val lower : string -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
  (** [lower name u] bounds host timeline polling with the device's submission
      deadline and latches failures for synchronization to report. *)
end

val submit_commands : device:Tolk.Device.t -> queue:string -> int array -> unit
(** [submit_commands ~device ~queue commands] executes raw NV packet dwords
    through the device's compiled queue and waits for completion. [queue] is
    ["COMPUTE:0"] or ["COPY:0"]. The submission shares timeline, command storage
    and FIFO retirement with compiled kernels. *)

val ensure_has_local_memory :
  'meta device ->
  num_gpcs:int ->
  num_tpc_per_gpc:int ->
  num_sm_per_tpc:int ->
  max_warps_per_sm:int ->
  device:Tolk.Device.t ->
  int ->
  unit
(** [ensure_has_local_memory dev ~num_gpcs ~num_tpc_per_gpc
    ~num_sm_per_tpc ~max_warps_per_sm ~device size] grows [dev]'s
    local-memory backing store ([dev.shader_local_mem]) until it covers
    [size] bytes per thread, recording the granted amount, rounded up
    to 32 bytes, in [dev.slm_per_thread]. Does nothing when the store
    already covers [size].

    Growing allocates owned storage with caching disabled, then submits setup
    commands through [device]'s compiled compute queue. The backing and capacity
    change only after confirmed completion; the previous allocation is then
    released. Failure leaves the previous backing and capacity unchanged. If
    completion is uncertain, normal buffer retirement retains the new allocation
    until the device can be drained safely. *)

(** {1:runtime Device runtime} *)

val arch_of_sm_version : int -> string
(** [arch_of_sm_version v] is the architecture string for the reported
    SM version [v]: the generation and revision digits appended to
    ["sm_"], so [0x809] is ["sm_89"]. A revision byte above [0xf] keeps
    only its high nibble, and the [0xa04] report names ["sm_120"]. *)

val sass_of_sm_version : int -> int
(** [sass_of_sm_version v] is the shader ISA revision launch
    descriptors carry for the reported SM version [v]: the generation
    nibbles over the revision nibble, so [0x809] is [0x89]. *)

val query_gpu_info : 'mem Nv_iface.t -> subdevice:int -> int list -> int list
(** [query_gpu_info iface ~subdevice indices] is the value of each
    graphics-engine information row in [indices], in order, queried
    from the device behind [subdevice]. An interface that programs the
    hardware directly answers from its static engine information
    instead of the driver query. *)

val on_device_hang :
  'mem Nv_iface.t -> debugger:int -> debug_channel:int -> unit -> unit
(** [on_device_hang iface ~debugger ~debug_channel ()] reads the per-SM
    error states of the compute channel [debug_channel] through the
    [debugger] object and raises [Failure] carrying the fault report:
    when an MMU fault is recorded, one line per queued fault with its
    address, fault type and access type decoded by name; otherwise one
    line per SM with a latched error state. The device timeline calls
    it when a wait stalls, folding the report into the timeout. *)

val create : string -> Tolk.Device.t
(** [create name] opens the NVIDIA device [name] names — ["NV"] for the
    first visible device, ["NV:1"] for the second, and so on — through
    the selected interface and is its runtime: a compute and a copy channel,
    an allocator staging host transfers through the copy engine,
    kernels compiled to the exact chip's binary format and dispatched
    through shared compiled queues, execution timing from device clocks, and
    fault reports raised through the completion timeline when the
    device hangs.

    With no interface in [DEV], opens the kernel-driver interface first
    and falls back to direct PCI access if interface initialization fails.
    [DEV=NVK+NV] or [DEV=PCI+NV] selects only that interface. A later
    runtime initialization failure does not trigger fallback.

    Raises [Failure] when no requested interface can open the device, and
    [Invalid_argument] for an unknown interface, malformed device index,
    or the deprecated [NV_IFACE] environment variable. *)
