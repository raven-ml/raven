(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Driver-less PCI device access.

    Talks to PCI devices directly, without a kernel driver: scanning the
    bus, detaching a device from its driver, reading and writing its
    configuration space, mapping its BARs into the process, and pinning
    system memory so devices can address it physically.

    Everything here works through [/sys] and [/proc] and therefore needs
    Linux, and typically root or equivalent capabilities; entry points
    raise [Failure] with an actionable message when the system lacks
    them. On other systems the module loads and its pure entry points
    work, but device probing fails cleanly. *)

exception Rollback_failed of exn * exn list
(** A setup error and the failures encountered during its rollback.
    Distinct from [Failure] so optional-backend fallbacks cannot swallow a
    failed cleanup and proceed with resources still in use. *)

val with_rollback : (((unit -> unit) -> unit) -> 'a) -> 'a
(** [with_rollback f] calls [f register]. Register a release action after each
    successful acquisition. If [f] raises, actions run in reverse order;
    otherwise ownership transfers to its result and no action runs.
    The original exception and backtrace are preserved when cleanup succeeds.
    Cleanup failures do not prevent later actions: they are appended to the
    original error in [Rollback_failed]. Register dependent releases in one action
    when a failed release must prevent the next one. [register] must not escape
    the call to [f]. *)

val with_buffer_setup :
  free:('m Hcq.Buffer.t -> unit) -> stop:(unit -> unit) -> close:(unit -> unit) ->
  (track:('m Hcq.Buffer.t -> 'm Hcq.Buffer.t) ->
   free:('m Hcq.Buffer.t -> unit) -> 'a) -> 'a
(** [with_buffer_setup ~free ~stop ~close f] tracks raw allocations while [f]
    constructs a device. [f] must wrap its allocator with [track] and use the
    supplied [free] for releases, including later storage finalizers.
    Success transfers ownership; the wrappers then stop tracking allocations.
    Failure calls [stop] before releasing pending buffers and calling [close].
    If any cleanup step fails, later steps are skipped and the cleanup error
    is appended as in {!with_rollback}. Finalizers cannot release buffers again
    after failed setup, even if [stop] failed and queues still use them.
    Imported buffers remain outside this allocation scope. *)

val filter_visible_devices : string -> 'a list -> 'a list
(** [filter_visible_devices device devices] selects and orders [devices] using
    the indices of [device]'s [DEV] target. Comma-separated indices preserve
    order and repetitions; [a-b] includes both ends. No indices, or an empty
    range, preserves the inventory.

    Raises [Invalid_argument] for malformed or unavailable indices, or use of
    the deprecated [HCQ_VISIBLE_DEVICES] environment variable. *)

(** {1:constants System constants} *)

val available : bool
(** [available] is [true] when the operating system supports driver-less
    PCI access (Linux). When [false], the mmap flags below are all [0]
    and functions touching [/sys] or [/proc] raise [Failure]. *)

val page_size : int
(** [page_size] is the system's virtual-memory page size in bytes. *)

val map_locked : int
(** Lock the mapping's pages in memory; [0] on systems without the
    flag. *)

val map_populate : int
(** Populate the mapping's page tables up front; [0] on systems without
    the flag. *)

val map_hugetlb : int
(** Back the mapping with huge pages; [0] on systems without the
    flag. *)

val map_fixed_noreplace : int
(** Place the mapping exactly at the given address but fail instead of
    replacing an existing mapping; [0] on systems without the flag. *)

(** {1:memory Memory} *)

val memory_barrier : unit -> unit
(** [memory_barrier ()] is a full memory barrier: memory accesses
    sequenced before it complete before any access sequenced after
    it. *)

val reserve_va : va_start:nativeint -> va_size:int -> unit
(** [reserve_va ~va_start ~va_size] reserves the address range
    [\[va_start, va_start + va_size)] with an inaccessible mapping so
    nothing else in the process lands there before fixed mappings claim
    parts of it. Reserving the same range again is a no-op. Raises
    [Failure] if the range cannot be reserved, in particular when part
    of it is already mapped. *)

val lock_memory : addr:nativeint -> size:int -> unit
(** [lock_memory ~addr ~size] locks the [size] bytes at [addr] into
    physical memory. Raises [Failure] if the pages cannot be locked. *)

val system_paddrs : ?pagemap:int -> vaddr:nativeint -> int -> int list
(** [system_paddrs ~vaddr size] is the physical address of each page
    backing the [size] bytes of mapped memory at [vaddr], in order, read
    from [pagemap] (a file descriptor in the kernel's page-map format;
    defaults to this process's page map, which only root can read fully).
    The pages must be resident, so lock or populate the mapping first.
    [vaddr] and [size] must be multiples of {!page_size}. Raises
    [Failure] if the page map cannot be read. *)

(** {1:sysfs Sysfs} *)

val write_sysfs : ?expected:string -> string -> value:string -> msg:string -> unit
(** [write_sysfs path ~value ~msg] ensures the sysfs file at [path]
    reads back [expected] (defaults to [value]) on its first line. If it
    does not, the write is attempted through [sudo] in a subshell; if
    the file still does not read back as expected, raises [Failure] with
    [msg] and the exact command to run manually. *)

val pci_scan_bus :
  ?sysfs:string -> ?base_class:int -> vendor:int -> (int * int list) list ->
  string list
(** [pci_scan_bus ~vendor devices] is the sorted bus addresses (for
    example ["0000:03:00.0"]) of every PCI device with vendor id
    [vendor] whose device id, masked by a [(mask, ids)] pair of
    [devices], appears in that pair's [ids]. [base_class] restricts the
    scan to devices of that class (the high byte of the sysfs class
    value). [sysfs] is the sysfs mount point (defaults to ["/sys"]).
    Raises [Failure] if the system exposes no PCI bus. *)

(** {1:locking Device locking} *)

val flock_acquire : string -> int
(** [flock_acquire name] takes the exclusive advisory lock on the file
    [name] in the system's temporary directory, creating it
    world-writable if needed, and is the open file descriptor holding
    the lock. The lock lasts until the descriptor is closed, normally
    for the life of the process. Raises [Failure] if another process
    holds the lock. *)

(** {1:device PCI devices} *)

(** A PCI device detached from its kernel driver.

    Creating a device takes an exclusive cross-process lock, unbinds the
    kernel driver, and {b hot-removes the device's sibling PCI
    functions} (for example a GPU's audio function): they disappear from
    the system until a PCI rescan or reboot. *)
module Pci_device : sig
  type t
  (** The type for driver-less PCI devices. *)

  val create : ?sysfs:string -> devpref:string -> string -> t
  (** [create ~devpref pcibus] takes exclusive ownership of the device
      at bus address [pcibus]: acquires the [devpref]-prefixed device
      lock (see {!flock_acquire}), unbinds any kernel driver,
      hot-removes sibling functions, enables the device, and opens its
      configuration space. [sysfs] is the sysfs mount point (defaults to
      ["/sys"]). Raises [Failure] if the device is inaccessible (with
      root and capability guidance on permission errors), if the driver
      cannot be unbound, or if another process holds the device lock.
      Failed acquisition closes its descriptors and releases its lock;
      driver unbinding and sibling removal are not undone.

      With [VFIO=1] in the environment and a usable [vfio] kernel
      module, the device is bound to the [vfio-pci] driver in no-iommu
      mode and its MSI interrupt vector is routed to an event
      descriptor, so {!wait_irq} can sleep until the device raises an
      interrupt. When the module, [/dev/vfio], or the permission for
      either is missing the device is enabled without the route and
      {!wait_irq} degrades to returning immediately; a failure after
      that probe (the group cannot be opened or the vector cannot be
      routed) raises [Failure]. *)

  val pcibus : t -> string
  (** [pcibus t] is the device's bus address. *)

  val lock_fd : t -> int
  (** [lock_fd t] is the file descriptor holding the device's exclusive
      lock; it stays open for the life of the process. *)

  val alloc_sysmem :
    ?vaddr:nativeint -> ?contiguous:bool -> int -> Hcq.Mmio.t * int list
  (** [alloc_sysmem size] allocates [size] bytes of page-locked,
      populated system memory that devices can target by physical
      address, and is the CPU mapping paired with the physical address
      of each 4 KiB block, in order. [vaddr] places the mapping at a
      fixed address. [contiguous] rounds [size] up to a whole page and
      backs multi-page allocations with a single huge page, making the
      blocks physically contiguous. Raises [Invalid_argument] if
      [contiguous] is set with [size] over 2 MiB, and [Failure] if
      memory cannot be allocated or its physical layout cannot be
      resolved. *)

  val read_config : t -> offset:int -> size:int -> int
  (** [read_config t ~offset ~size] is the little-endian value of the
      [size] bytes at [offset] in the device's configuration space.
      Raises [Failure] if the read fails. *)

  val write_config : t -> offset:int -> value:int -> size:int -> unit
  (** [write_config t ~offset ~value ~size] stores [value] as [size]
      little-endian bytes at [offset] in the device's configuration
      space. Raises [Failure] if the write fails. *)

  val write_config_flush : t -> offset:int -> value:int -> size:int -> unit
  (** [write_config_flush t ~offset ~value ~size] is {!write_config}
      followed by a read of the same range, forcing the write to reach
      the device before returning. *)

  val disable_aspm : t -> unit
  (** [disable_aspm t] clears L0s/L1 power-management bits in the PCIe link
      control register, preserving other bits and flushing the write. Missing
      PCIe capabilities and cyclic capability lists leave configuration intact.
      Raises [Failure] if configuration access fails. *)

  val bar_info : t -> int -> int * int
  (** [bar_info t bar] is the bus address and byte size of the device's
      BAR number [bar]. Raises [Failure] if the device does not expose
      that BAR. *)

  val map_bar : t -> ?off:int -> ?addr:nativeint -> ?size:int -> int -> Hcq.Mmio.t
  (** [map_bar t bar] maps BAR number [bar] into the process and is the
      mapped region, starting [off] bytes into the BAR and covering
      [size] bytes (to the end of the BAR by default). [addr] places the
      mapping at a fixed address. The mapping is excluded from forked
      children. Raises [Failure] if the BAR cannot be mapped. *)

  val resize_bar : t -> int -> unit
  (** [resize_bar t bar] resizes BAR number [bar] to the largest size
      the device supports. Raises [Failure] if the device or platform
      does not support resizing it. *)

  val reset : t -> unit
  (** [reset t] requests a function-level reset of the device through
      [sudo] in a subshell. Best effort: failures are not reported. *)

  val wait_irq : t -> timeout_ms:int -> unit
  (** [wait_irq t ~timeout_ms] blocks until the device raises an
      interrupt, for at most [timeout_ms] milliseconds, and consumes the
      pending interrupt event. Returns immediately on a device without
      the interrupt route (see {!create}). *)
end

(** {1:iface Driver-less device interfaces} *)

(** The device-independent half of a driver-less PCI GPU interface.

    Owns what every such interface shares: probing the bus for a
    supported device and taking exclusive ownership of it, reserving
    the GPU virtual address range, and serving memory requests over the
    device implementation's memory manager — device memory from its
    physical allocator, host memory as pinned system pages mapped
    through the GPU page tables, and peer memory through the owning
    device's memory BAR.

    A vendor backend supplies the device-implementation constructor and
    everything hardware-specific (queues, interrupts, shutdown) on
    top. *)
module Pci_iface_base : sig
  type ('impl, 'pt) t
  (** The type for interface bases over a device implementation
      ['impl] whose memory manager writes ['pt] page tables. *)

  type mem
  (** Allocation or import metadata. Allocations retain their owning memory
      manager independently of its vendor and page-table representation. *)

  val kind : mem Hcq.Buffer.t Type.Id.t
  (** [kind] identifies raw PCI storage across device interfaces. *)

  val empty : mem Hcq.Buffer.t
  (** [empty] is the zero-sized scratch placeholder. It owns no memory. *)

  val hmemory : mem Hcq.Buffer.t -> int
  (** [hmemory b] is the first backing physical address of an allocation,
      used by NV channel setup. Raises [Invalid_argument] for an import. *)

  val create :
    name:string ->
    devpref:string ->
    dev_id:int ->
    vendor:int ->
    devices:(int * int list) list ->
    ?base_class:int ->
    vram_bar:int ->
    va_start:nativeint ->
    va_size:int ->
    dev_impl:(Pci_device.t -> 'impl) ->
    mm:('impl -> 'pt Tolk.Memory.t) ->
    unit ->
    ('impl, 'pt) t
  (** [create ~name ~devpref ~dev_id ~vendor ~devices ~vram_bar
      ~va_start ~va_size ~dev_impl ~mm ()] opens the [dev_id]th PCI
      device (in bus order) matching [vendor], the [(mask, ids)] pairs
      of [devices] and [base_class] (see {!pci_scan_bus}): it takes
      exclusive ownership of the device (see {!Pci_device.create},
      with [devpref] prefixing the device lock), reserves the GPU
      virtual range ([va_start], [va_size]) in this process, grows the
      device-memory BAR [vram_bar] to its full size where the platform
      allows, and hands the device to [dev_impl], whose result — the
      booted device implementation — serves all further operations
      through the memory manager [mm] selects from it.

      [name] names the backend in messages. Raises [Failure] when no
      matching device has index [dev_id], when the device cannot be
      claimed, or when [dev_impl] fails. *)

  val pci_dev : ('impl, 'pt) t -> Pci_device.t
  (** [pci_dev t] is the owned PCI device. *)

  val dev_impl : ('impl, 'pt) t -> 'impl
  (** [dev_impl t] is the device implementation built by {!create}. *)

  val mm : ('impl, 'pt) t -> 'pt Tolk.Memory.t
  (** [mm t] is the device implementation's memory manager. *)

  val count : ('impl, 'pt) t -> int
  (** [count t] is the number of matching devices on the bus. *)

  val is_bar_small : ('impl, 'pt) t -> bool
  (** [is_bar_small t] is [true] iff the device-memory BAR kept the
      legacy 256MB window, so only a slice of device memory is
      CPU-reachable. *)

  val alloc :
    ('impl, 'pt) t ->
    ?host:bool ->
    ?uncached:bool ->
    ?cpu_access:bool ->
    ?contiguous:bool ->
    ?force_devmem:bool ->
    int ->
    mem Hcq.Buffer.t
  (** [alloc t size] allocates [size] bytes for the device and is the
      mapped region. Device memory is the default; [host] allocates
      pinned system pages instead, mapped into the GPU as coherent
      uncached memory. [uncached] requests uncached device memory;
      [cpu_access] host-visible memory — the buffer carries a CPU view
      exactly when the memory is host-backed or [cpu_access] is set.
      CPU-visible requests on a small-BAR device use system memory unless
      [force_devmem] insists on device memory. Large-BAR devices retain
      device memory even for uncached CPU-visible requests. [contiguous] makes
      host-backed memory physically contiguous. All default to
      [false].

      Device memory is sized in whole huge pages from 8MB up so the
      tail does not fall back to 4KB translations; smaller requests
      and system memory round to the page size.

      Raises [Failure] when memory is exhausted or pages cannot be
      pinned. *)

  val free : ('impl, 'pt) t -> mem Hcq.Buffer.t -> unit
  (** [free t b] releases an allocation of [t], including its virtual
      range, device memory and CPU mapping, whose address may differ from
      the GPU address. Imports are only unmapped from [t]'s page tables.
      Raises [Invalid_argument] for an allocation owned by another interface. *)

  val map : ('impl, 'pt) t -> Tolk.Device.Buffer.t -> mem Hcq.Buffer.t
  (** [map t b] maps the source allocation into [t]. PCI device memory is
      reached through its owner's BAR; host storage is pinned and mapped by
      its physical pages. The source range must fit the receiver's GPU
      address range; host addresses must also be page aligned.
      System-memory allocations remain importable when
      their owner has a small BAR. Raises {!Tolk_uop.Storage.Mapping_unavailable}
      for unsupported storage, an incompatible source address, or device memory
      behind a small BAR. Driver failures propagate. The caller must retain
      [b] until unmapping. *)

  val unmap : ('impl, 'pt) t -> mem Hcq.Buffer.t -> unit
  (** [unmap t b] releases an import's page tables without releasing the
      source allocation or its CPU mapping. Raises [Invalid_argument] for
      an owned allocation. *)

  val p2p_paddrs :
    ('impl, 'pt) t -> (int * int) list -> (int * int) list * Tolk.Memory.addr_space
  (** [p2p_paddrs t paddrs] are the device-local [(paddr, size)] ranges
      [paddrs] as a peer on the bus reaches them: offset into [t]'s
      device-memory BAR, addressed as system memory. *)
end
