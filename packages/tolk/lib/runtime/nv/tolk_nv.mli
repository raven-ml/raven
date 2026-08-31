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
    execute, and the driver interface ({!Nv_iface}) with its
    kernel-driver implementation ({!Nvk_iface}).

    The builders are pure: they read a {!type-device} description,
    append dwords to an in-memory {!Hcq.Q.t}, and patch launch
    descriptors through their CPU mappings. Their [submit] functions
    stage the accumulated stream in the device's command buffer, point
    a channel ring entry ({!Queue_desc}) at it, and ring the
    work-submission doorbell. *)

module Hcq = Tolk_hcq.Hcq
module Nv_tables = Nv_tables

(** {1:qmd Launch descriptors} *)

(** Kernel launch descriptors.

    The compute engine consumes launches as fixed-size descriptors of
    packed bitfields: grid and block geometry, program and constant
    buffer addresses, dependency links, and release semaphores. A
    descriptor is a lens over caller-provided mapped bytes, either a
    program's long-lived template or the per-launch copy the engine
    fetches from device memory; fields are addressed by the names of
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
  shared_mem_window : nativeint;
      (** Virtual-address window shared-memory accesses go through. *)
  local_mem_window : nativeint;
      (** Virtual-address window local-memory accesses go through. *)
  cmdq_page : 'meta Hcq.Buffer.t;
      (** CPU-mapped device memory where submissions stage their
          command streams. *)
  cmdq_allocator : Tolk.Bump.t;
      (** Wrapping bump allocator handing out stream addresses inside
          [cmdq_page]. *)
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
    must exist. [slm_per_thread] defaults to [0].

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
    token : int;
        (** Work-submission token naming the channel to the
            doorbell. *)
    mutable put_value : int;  (** Number of entries submitted. *)
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
      work-submission doorbell with [qd]'s token. The stream is kept:
      submitting again stages it again. *)
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

  type mem = {
    h_memory : int;  (** Driver handle of the backing memory. *)
    owner_id : int;  (** Index of the device that allocated it. *)
  }
  (** The type for driver metadata carried by device-memory buffers. *)

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

  type t = {
    root : int;  (** Handle of the client every driver object hangs off. *)
    gpu_instance : int;  (** Driver instance index of the device. *)
    count : int;
        (** Number of devices this interface kind can open in this
            system. *)
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
      ?map_flags:int ->
      ?cpu_addr:nativeint ->
      int ->
      mem Hcq.Buffer.t;
        (** [alloc size] allocates [size] bytes of device-visible
            memory, rounded up to the allocation page size, and maps
            them at a fresh virtual address. [host] registers CPU
            memory instead of allocating device memory; [uncached]
            allocates GPU-uncacheable system pages; [cpu_access] also
            maps device memory for the CPU, so the buffer has a view;
            [contiguous] requires physically contiguous pages;
            [map_flags] adds driver flags to the CPU mapping;
            [cpu_addr] reuses the existing CPU mapping at that address
            instead of reserving a fresh range. All default to [false],
            [0] or absent. Raises {!Out_of_memory} and [Failure] as
            [rm_alloc]. *)
    free : mem Hcq.Buffer.t -> unit;
        (** [free buf] releases [buf]'s physical memory, virtual range
            and CPU mapping. A buffer allocated by another device's
            interface is left untouched. *)
    map : mem Hcq.Buffer.t -> mem Hcq.Buffer.t;
        (** [map buf] maps a buffer allocated by another device's
            interface into this device's address space at the same
            virtual address. The result keeps the original owner. *)
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

  val is_nvd : t -> bool
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
  val iface : device_id:int -> Nv_iface.t
  (** [iface ~device_id] is the kernel-driver interface of the
      [device_id]th visible device. The first call opens the driver:
      it creates the root client, detects the driver generation,
      initializes the memory manager and enumerates the cards. Raises
      [Failure] if the driver is unavailable or [device_id] is out of
      range. *)

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
