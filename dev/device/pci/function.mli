(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PCI functions the process has taken.

    A taken function gives the process what a kernel driver has of it: its
    configuration space, its BARs mapped as windows on its registers and memory,
    its interrupts and reset, and the system memory it reaches by DMA. Taking a
    function changes nothing on its machine; only {!Gpus.detach} and
    {!Gpus.attach} do.

    A function of {!Machine.this} is taken in one of two ways, which follow from
    the machine's state and which {!addressing} reports:
    - {e Behind an IOMMU}, through VFIO, without root. An administrator turns
      the IOMMU on, binds the function to [vfio-pci]
      ([driverctl set-override BUS vfio-pci]) and grants the user its group's
      file [/dev/vfio/N]. The function reaches only the memory the process maps
      for it, at device addresses the process chooses.
    - {e Physically}, through [/sys/bus/pci], with write access to its BAR files
      and no driver bound other than [vfio-pci]. The function reaches system
      memory at its physical addresses, which the IOMMU, if any, must not
      translate.

    {!take} lists what refuses a take; an operation that needs more states it. A
    function of another machine is taken through its transport
    ({!Machine.make}), which reaches it as that machine's way allows.

    One owner calls a function's operations at a time, except that {!pin},
    {!unpin}, {!alloc_dma} and {!free_dma} may be called from any domain. After
    {!release}, every operation but {!free_dma}, {!unpin} and the observers
    raises [Invalid_argument].

    Windows are values: a window equal to a live one that {!map} or {!alloc_dma}
    gave names it, and a window of another function or machine is never equal to
    one of [f]'s. *)

(** {1:functions Functions} *)

type t
(** The type for functions the process has taken. *)

val take : Machine.t -> string -> (t, string) result
(** [take m bus] takes the function at [bus] on [m] for this process alone:
    behind an IOMMU, VFIO opens its group for one process at a time; taken
    physically, [m] locks it. Taking a function changes nothing on [m].

    [Error why] if [bus] is no function of [m], if [m] failed, if a process,
    this one included, holds it, or if neither way is open:
    - it is bound to a driver other than [vfio-pci] ({!Gpus.detach});
    - the IOMMU translates its addresses and it is not bound to [vfio-pci];
    - it is not behind an IOMMU and the kernel is locked down
      ([/sys/kernel/security/lockdown], which Secure Boot turns on in several
      distributions), which refuses every mapping of a BAR;
    - the process may not open its files.

    [why] names what to change, such as the udev rule that grants [/dev/vfio/N],
    or the [driverctl] commands that bind the other functions of its IOMMU group
    to [vfio-pci]. *)

val release : t -> unit
(** [release f] gives [f] back: it turns [f]'s bus mastering off, so that [f]
    reaches system memory by DMA no more, unmaps its BAR windows, closes the
    process's files for it and unlocks it. The memory allocated for [f] stays
    allocated until {!free_dma}. Releasing it again does nothing.

    A function of {!Machine.this} the process still holds when it exits loses
    its bus mastering then, once the exit functions of the libraries above this
    one, such as its driver's, have run; a child of [fork] exiting changes
    nothing. A process killed by a signal stops nothing: behind an IOMMU, Linux
    then stops the function's DMA as it closes the process's files; taken
    physically, the function keeps mastering the bus. *)

val machine : t -> Machine.t
(** [machine f] is the machine [f] is on. *)

val bus : t -> string
(** [bus f] is [f]'s bus address on its machine. *)

val addressing : t -> Machine.addressing
(** [addressing f] is how [f] reaches system memory. *)

val released : t -> bool
(** [released f] is [true] iff {!release} gave [f] back. *)

val failed : t -> string option
(** [failed f] is why [f] can no longer be reached, if it cannot: its machine
    failed ({!Machine.failed}), or [f] left the bus, its vendor ID reading
    [0xffff]. It reads configuration space, so a driver asks it where it acts on
    what it read, off its hot path ({{!Device_pci.errors}errors}).

    Raises [Invalid_argument] if [f] was released. *)

(** {1:config Configuration space}

    A function's configuration space is 4096 bytes of little-endian values.
    Linux shows a process its first 64 bytes alone, unless the process took the
    function behind an IOMMU or has [CAP_SYS_ADMIN].

    Each access raises [Invalid_argument] if its bytes are not in the
    configuration space, and nothing else. A read the function or the system
    does not answer gives all ones: on a function that left the bus or a failed
    machine, and past the first 64 bytes of a function taken physically by a
    process without [CAP_SYS_ADMIN]. A write reads its bytes back, so it has
    reached the function when it returns; one the system refuses is dropped. *)

val config8 : t -> int -> int
(** [config8 f off] is the byte at [off] of [f]'s configuration space. *)

val config16 : t -> int -> int
(** [config16 f off] is the 16-bit value at byte [off] of [f]'s configuration
    space. *)

val config32 : t -> int -> int
(** [config32 f off] is the 32-bit value at byte [off] of [f]'s configuration
    space. *)

val set_config8 : t -> int -> int -> unit
(** [set_config8 f off x] writes the low 8 bits of [x] at byte [off] of [f]'s
    configuration space. *)

val set_config16 : t -> int -> int -> unit
(** [set_config16 f off x] writes the low 16 bits of [x] at byte [off]. *)

val set_config32 : t -> int -> int -> unit
(** [set_config32 f off x] writes the low 32 bits of [x] at byte [off]. *)

(** {1:bars BARs} *)

val bar : t -> int -> (int * int) option
(** [bar f i] is the bus address of [f]'s BAR [i], as its BAR register holds it
    and as another function on the bus reaches it, with its size in bytes.
    [None] if [f] has no BAR [i], such as the upper index of a 64-bit BAR or an
    index past 5.

    Raises [Invalid_argument] if [i < 0]. *)

val map :
  ?combine:bool ->
  ?off:int ->
  ?length:int ->
  t ->
  int ->
  (Window.t, string) result
(** [map f i] is a window on [length] bytes (defaults to the rest of the BAR) of
    [f]'s BAR [i] from byte [off] (defaults to [0]), until {!unmap} or
    {!release}. Child processes do not inherit it.

    With [~combine:true] (defaults to [false]) stores through the window may
    merge and reach the function in another order until a {!Window.barrier},
    where the machine allows it: a prefetchable BAR of a function taken
    physically. Behind an IOMMU, or on a BAR that is not prefetchable, it is the
    window [map] makes without it. Map registers and doorbells without it.

    [Error why] if VFIO or the kernel does not let the process map them.

    Raises [Invalid_argument] if [i < 0], the bytes do not lie in the BAR, or a
    live window of [f] maps BAR [i] with the other [combine]: the processor maps
    the same addresses one way at a time. *)

val unmap : t -> Window.t -> unit
(** [unmap f w] unmaps [w].

    Raises [Invalid_argument] if [w] is no live window {!map} made on [f]. *)

(** {1:events Interrupts and reset} *)

val interrupt : t -> int -> bool
(** [interrupt f ms] waits at most [ms] milliseconds for an interrupt of [f],
    with the OCaml runtime released, and is [true] iff one arrived. Taken
    physically, [f] has interrupts only bound to [vfio-pci] in VFIO's no-IOMMU
    mode, whose files need [CAP_SYS_RAWIO]. Without interrupts it is [false] at
    once.

    Raises [Invalid_argument] if [ms < 0]. *)

val reset : t -> (unit, string) result
(** [reset f] resets [f] with the reset Linux has for it, and waits at most a
    second for it to answer again. It clears the state a previous driver left.

    [Error why] naming the file if the process may not reset [f] or Linux has
    none for it, if [f] does not answer in time, or if its machine failed. *)

(** {1:dma System memory}

    A function reaches system memory of its machine at addresses given as
    (address, bytes) {e runs}, in order.
    - Behind an IOMMU they are device addresses the process maps for the
      function alone, as one run. The memory counts against the process's
      locked-memory limit ([ulimit -l]), and the function's VFIO container holds
      at most [dma_entry_limit] mappings (a parameter of [vfio_iommu_type1],
      65,535 by default).
    - Taken {!Machine.Physical}ly, they are physical addresses, one run per page
      unless contiguous. Reading them needs the privileges for
      [/proc/self/pagemap] and [mlock], and the kernel setting
      [vm.compact_unevictable_allowed = 0], without which the kernel may move
      locked pages. *)

val alloc_dma :
  ?contiguous:bool ->
  ?va:int ->
  t ->
  int ->
  (Window.t * (int * int) list, string) result
(** [alloc_dma f n] is [n] bytes, rounded up to {!Machine.page}, of new, zeroed
    memory of [f]'s machine, locked where it is, at [va] inside a range
    {!Machine.reserve} reserved or where the machine chooses without [va], with
    the runs at which [f] reaches it. With [~contiguous:true] (defaults to
    [false]) the memory is one run of at most 2 MiB; reached physically and
    larger than a page, it is a huge page the system must have free
    ([vm.nr_hugepages]).

    [Error why] naming what is missing if the machine cannot: free memory or a
    huge page, or one of the privileges, settings and limits above.

    Raises [Invalid_argument] if [n <= 0] or rounding it up overflows, if [va]
    is not on a page or in no range {!Machine.reserve} reserved, if [contiguous]
    memory is larger than 2 MiB, or if [va] is not on 2 MiB for a huge page. *)

val free_dma : t -> Window.t -> unit
(** [free_dma f w] frees [w], and [f] reaches it no more.

    Raises [Invalid_argument] if [w] is no live window {!alloc_dma} returned for
    [f]. *)

val pin : t -> int -> int -> ((int * int) list, string) result
(** [pin f a n] locks the [n] bytes at [a] of [f]'s machine, which start on a
    page, and is the runs at which [f] reaches them. Pins are counted: memory
    stays locked until each pin is {!unpin}ned, and behind an IOMMU [(a, n)]
    stays mapped for [f] as long.

    [Error why] as {!alloc_dma}, or if the pages cannot be locked or their
    addresses read.

    Raises [Invalid_argument] if [a] is not on a page or [n <= 0]. *)

val unpin : t -> int -> int -> unit
(** [unpin f a n] releases one pin of the [n] bytes at [a].

    Raises [Invalid_argument] if [(a, n)] is not pinned for [f]. *)
