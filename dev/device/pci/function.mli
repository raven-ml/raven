(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PCI functions the process has taken.

    A driver that drives a GPU itself takes the GPU's PCI function: it reads and
    writes its configuration space, maps its BARs, the windows through which the
    process reaches its registers and memory, waits for its interrupts, and
    allocates the system memory it reaches by DMA. Taking a function changes
    nothing on its machine; only {!Gpus.detach} and {!Gpus.attach} do.

    A function of {!Machine.this} is taken one of two ways, which follow from
    the machine's state and which {!addressing} reports:
    - {e Behind an IOMMU}, through VFIO, without root. An administrator binds
      the function to [vfio-pci] ([driverctl set-override BUS vfio-pci]) on a
      machine whose IOMMU is on and grants the user its group's file
      [/dev/vfio/N]. The function then reaches only the memory the process maps
      for it, at device addresses the process chooses. That memory counts
      against the process's locked-memory limit ([ulimit -l]), and a container
      holds at most [dma_entry_limit] mappings (a parameter of
      [vfio_iommu_type1], 65,535 by default).
    - {e Physically}, through [/sys/bus/pci], unbound from any driver. Mapping
      its BARs needs write access to their files and a kernel that is not locked
      down ([/sys/kernel/security/lockdown], which Secure Boot turns on in
      several distributions); configuration space past its first 64 bytes needs
      [CAP_SYS_ADMIN]; its interrupts, through VFIO's no-IOMMU mode, need it
      bound to [vfio-pci] and [CAP_SYS_RAWIO]. The function reaches system
      memory at its physical addresses, which the IOMMU, if any, must not
      translate.

    A function of another machine is taken through its transport
    ({!Machine.make}), {!Physical}ly, and has no interrupts.

    One owner calls a function's operations at a time, except that {!pin},
    {!unpin}, {!alloc_dma} and {!free_dma} may be called from any domain.
    Accesses that fail under the process raise [Failure]
    ({{!Device_pci.errors}errors}). *)

(** {1:functions Functions} *)

type t
(** The type for functions the process has taken. *)

(** The type for how a function reaches system memory. *)
type addressing = Machine.addressing =
  | Physical
      (** At physical addresses of its machine, which no IOMMU translates. *)
  | Iommu
      (** Through an IOMMU, at device addresses the process maps for it alone.
      *)

val take : Machine.t -> lock:string -> string -> (t, string) result
(** [take m ~lock bus] takes the function at [bus] on [m] for this process. It
    locks the function through two files of the temporary directory:
    [nx_BUS.lock], which every process of this library takes, and
    [LOCK_BUS.lock], which other programs driving such a GPU take with the same
    [lock] name. A lock file that is a link or no regular file is refused.
    Taking a function changes nothing on [m].

    [Error why] if [bus] is no function of [m], if it is bound to a driver other
    than [vfio-pci] ({!Gpus.detach}), if another process holds it, if the IOMMU
    translates its addresses and it is not bound to [vfio-pci], if the kernel is
    locked down and it is not behind an IOMMU, or if the process may not access
    it. [why] names what to change, such as the udev rule that grants
    [/dev/vfio/N], or the [driverctl] commands that bind the other functions of
    its IOMMU group to [vfio-pci]. *)

val release : t -> unit
(** [release f] gives [f] back: it unmaps its BAR windows, closes the process's
    files for it and unlocks it. Behind an IOMMU, [f] no longer reaches the
    memory mapped for it, which stays allocated until {!free_dma}. Releasing it
    again does nothing. *)

val machine : t -> Machine.t
(** [machine f] is the machine [f] is on. *)

val bus : t -> string
(** [bus f] is [f]'s bus address on its machine. *)

val addressing : t -> addressing
(** [addressing f] is how [f] reaches system memory. *)

val released : t -> bool
(** [released f] is [true] iff {!release} gave [f] back. *)

(** {1:config Configuration space} *)

val config : t -> int -> int -> int
(** [config f off n] is the [n]-byte little-endian value at byte [off] of [f]'s
    configuration space, [n] being 1, 2 or 4.

    Raises [Invalid_argument] if [n] is not 1, 2 or 4, and [Failure] naming
    [CAP_SYS_ADMIN] if [off] is past the first 64 bytes and [f], taken
    physically, may not read them. *)

val set_config : t -> int -> int -> int -> unit
(** [set_config f off n x] writes the low [n] bytes of [x] at byte [off] of
    [f]'s configuration space, then reads them back, so the write has reached
    the function when it returns.

    Raises as {!config} does. *)

(** {1:bars BARs} *)

val bar : t -> int -> (int * int) option
(** [bar f i] is the bus address of [f]'s BAR [i], as its BAR register holds it
    and as another function on the bus reaches it, with its size in bytes.
    [None] if [f] has no BAR [i], such as the upper index of a 64-bit BAR. *)

val map : ?off:int -> ?length:int -> t -> int -> Window.t
(** [map f i] is a window on [length] bytes (defaults to the rest of the BAR) of
    [f]'s BAR [i] from byte [off] (defaults to [0]), until {!unmap} or
    {!release}. Child processes do not inherit it.

    Raises [Invalid_argument] if the bytes do not lie in the BAR, and [Failure]
    if VFIO or the kernel does not let the process map them. *)

val unmap : t -> Window.t -> unit
(** [unmap f w] unmaps [w].

    Raises [Invalid_argument] if [w] is no window {!map} made on [f] and has not
    unmapped. *)

(** {1:events Interrupts and reset} *)

val interrupt : t -> int -> bool
(** [interrupt f ms] waits at most [ms] milliseconds for an interrupt of [f],
    with the OCaml runtime released, and is [true] iff one arrived. Without
    VFIO, or on another machine, it is [false] at once. *)

val reset : t -> unit
(** [reset f] resets [f] with the reset Linux has for it, and waits at most a
    second for it to answer again. It clears the state a previous driver left.

    Raises [Failure] naming the file if the process may not reset [f] or Linux
    has none for it, and if [f] does not answer in time. *)

(** {1:dma System memory}

    The system memory of [f]'s machine that [f] reaches, at the addresses it
    reaches it at, as (address, bytes) runs in order: physical addresses for a
    function taken {!Physical}ly, one run per page unless contiguous; device
    addresses behind an IOMMU, which the process maps for [f] alone, as one run.
    Physical addresses need Linux, the privileges for [/proc/self/pagemap] and
    [mlock], and the kernel setting [vm.compact_unevictable_allowed = 0],
    without which the kernel may move locked pages. *)

val alloc_dma :
  ?contiguous:bool -> ?va:int -> t -> int -> Window.t * (int * int) list
(** [alloc_dma f n] is [n] bytes, rounded up to {!Machine.page}, of new, zeroed
    memory of [f]'s machine, locked where it is, at [va] inside a range
    {!Machine.reserve} reserved or where the machine chooses without [va], with
    the runs at which [f] reaches it. With [~contiguous:true] (defaults to
    [false]) the memory is one run of at most 2 MiB; reached physically and
    larger than a page, it is a huge page the system must have free
    ([vm.nr_hugepages]).

    Raises [Invalid_argument] if [va] is not on a page, if [contiguous] memory
    is larger than 2 MiB, or if [va] is not on 2 MiB for a huge page. Raises
    [Failure] naming the limit if the machine cannot: no free memory or huge
    page, a missing privilege or setting, the locked-memory limit, or the
    container's [dma_entry_limit]. Those limits are reached by use, yet raise: a
    default [ulimit -l] would otherwise make every allocation fail without a
    word of why. *)

val free_dma : t -> Window.t -> unit
(** [free_dma f w] frees [w], and [f] reaches it no more.

    Raises [Invalid_argument] if [w] is no window {!alloc_dma} returned for [f]
    and has not freed. *)

val pin : t -> int -> int -> (int * int) list
(** [pin f a n] locks the [n] bytes at [a] of [f]'s machine, which start on a
    page, and is the runs at which [f] reaches them. Pins are counted: memory
    stays locked until each pin is {!unpin}ned, and behind an IOMMU [(a, n)]
    stays mapped for [f] as long.

    Raises [Invalid_argument] if [a] is not on a page, and [Failure] if the
    pages cannot be locked or their addresses read, or as {!alloc_dma} does for
    the limits behind an IOMMU. *)

val unpin : t -> int -> int -> unit
(** [unpin f a n] releases one pin of the [n] bytes at [a].

    Raises [Invalid_argument] if [(a, n)] is not pinned for [f]. *)
