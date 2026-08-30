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
      cannot be unbound, or if another process holds the device lock. *)

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
end
