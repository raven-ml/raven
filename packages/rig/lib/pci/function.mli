(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PCI functions the process has taken.

    A taken function gives the process what a kernel driver has of it: its
    configuration space, its BARs mapped as windows on its registers and memory,
    its interrupts and reset, and the system memory it reaches by DMA.

    A function of a machine reached without a transport, such as
    {!Machine.this}, is taken in one of two ways. The machine's state decides
    which, and {!addressing} reports it.
    - {e Behind an IOMMU}, through VFIO, without root. An administrator turns
      the IOMMU on, binds the function to [vfio-pci] (its [driver_override]) and
      grants the user its group's file [/dev/vfio/N]. The function reaches only
      the memory the process maps for it, at device addresses the process
      chooses.
    - {e Physically}, through [/sys/bus/pci], with write access to the
      function's files there, for a function bound to no driver or to
      [vfio-pci] whose addresses no IOMMU translates. It reaches system memory
      at physical addresses, and nothing confines its DMA. An address that its
      work or page tables get wrong reads or writes whatever memory of the
      machine lies there, the kernel's included.

    {!take} lists what refuses a take. An operation that needs a further
    privilege, such as {!reset}, names it. A function of another machine is
    taken through its transport ({!Machine.make}), which reaches it as that
    machine allows.

    The life of a taken function goes like this:
    + {!take} takes it for this process alone.
    + The driver maps its BARs ({!map}), gives it system memory ({!alloc_dma},
      {!pin}) and lets it master the bus ({!set_bus_master}).
    + The driver runs the device through its windows and waits for it with
      {!wait}.
    + {!release} turns its bus mastering off and gives it back.

    A function of a machine reached without a transport that the process still
    holds when it exits loses its bus mastering then, after the exit functions
    of the libraries above this one, such as its driver's, have run. A child of
    [fork] exiting changes nothing. A process killed by a signal runs no exit
    function. A process that exits from a callback inside a lock of this
    library's leaves the function whose release takes that lock as a killed
    process leaves it, prints why on standard error, and releases the
    others. Behind an IOMMU, Linux stops the DMA of a function so left as it
    closes the process's files. Taken physically, the function keeps
    writing, into memory that stays until its GPU is reset
    ({{!dma}system memory}), and its legacy interrupts stay off until a kernel
    driver enables the function.

    Behind an IOMMU, Linux turns a function's bus mastering off when the last
    descriptor of its files closes. A child of [fork] that has not run [exec]
    holds copies of them, so the function keeps mastering the bus, after its
    {!release} too, until that child closes them or exits.

    A take changes the function's legacy interrupts until its release
    ({!take}), and a release turns bus mastering off. A function taken
    physically also gets memory that outlives the process
    ({{!dma}system memory}). {!Driver.detach} and {!Driver.attach} are the
    calls that change the machine.

    One owner makes a function's calls one at a time, except that {!pin},
    {!unpin}, {!alloc_dma}, {!free_dma} and {!wait} may be called from any
    domain. A pin, an unpin, an allocation, a free, the read of configuration
    space at a wait's bound and each request that writes the GPU's tables
    ({!Memory}) are the function's {e uses}. Each counts among the locks its
    thread holds while it runs ({{!Rig_pci.errors}errors}). Once a {!release}
    has begun, no use starts. A pin or an allocation then raises
    [Invalid_argument], and a free or an unpin gives memory back as
    {{!dma}system memory} states. The function is {e given back} once its
    release has begun and no use runs. The release gives it back itself if no
    use runs as it begins, and otherwise the last use to end gives it back. A
    use refused after the release, or {!finish_releases}, gives back what an
    exception left of that. A process that exits from a callback inside a use
    leaves the function as a killed process leaves it. After {!release}, every
    call but {!free_dma}, {!unpin}, {!wait}, {!machine}, {!bus}, {!addressing}
    and {!released} raises [Invalid_argument], and so does {!set_bus_master}
    once the function is given back. {!unmap} and {!free_dma} find a window by
    equality ({!Window.t}). *)

(** {1:functions Functions} *)

type t
(** The type for functions the process has taken. *)

val take : Machine.t -> string -> (t, string) result
(** [take m bus] takes the function at [bus] on [m] for this process alone.
    Behind an IOMMU, VFIO opens its group for one process at a time; taken
    physically, [m] locks it. Taken physically without VFIO's interrupts, its
    legacy interrupts (INTx) are turned off, since nothing handles them, until
    {!release} sets them back as it found them. A physical take first deletes
    the memory files that dead processes left and no function reaches.

    [Error why] if [bus] is not a function of [m], if [m] has failed, if a
    process, this one included, holds it, if this process gave the memory
    its BAR reaches to another function ({!Memory.bus}), until the process
    exits, or if neither way is open:
    - it is bound to a driver other than [vfio-pci] ({!Driver.detach});
    - the IOMMU translates its addresses and it is not bound to [vfio-pci];
    - not behind an IOMMU, it is disabled, or it shares its device with
      another function and Linux resets it through its bus, which Linux
      refuses while another function shares the bus. Linux resets it so when
      the first method of its [reset_method] is [bus] or [cxl_bus], or the
      kernel lists none and the function has a [reset] ({!Driver.detach});
    - it is not behind an IOMMU and the kernel is locked down
      ([/sys/kernel/security/lockdown], which Secure Boot turns on in several
      distributions), which refuses every mapping of a BAR;
    - the process may not open its files. Bound to [vfio-pci] without an IOMMU,
      the function's no-IOMMU group file needs [CAP_SYS_RAWIO].

    [why] names what to change, such as the user's access to [/dev/vfio/N], or
    the other functions of its IOMMU group, which must be bound to [vfio-pci]
    too. *)

val release : t -> unit
(** [release f] gives [f] back. Giving it back turns [f]'s bus mastering off,
    which ends its DMA, restores the legacy interrupts {!take} turned off,
    unmaps [f]'s BAR windows, closes the process's files for [f] and unlocks
    it. Memory from {!alloc_dma} stays until {!free_dma}
    ({{!dma}system memory}). While uses of [f] run, on other threads or
    below a callback on this one, the release returns immediately, and the
    last of them to end gives [f] back. A release called once one has begun
    returns immediately.

    An exception that ends the give-back part-way, such as an asynchronous
    one ({!Sys.Break}), passes to the release or the use that made it, and
    leaves the rest to a use refused after it, to {!finish_releases}, or to
    the process's exit, which at worst leaves [f] as a killed process
    does. *)

val finish_releases : unit -> unit
(** [finish_releases ()] gives back each function of this process whose
    {!release} has begun and that is not given back, such as one whose
    give-back an exception ended part-way: what it had not given back, and
    nothing twice. It first waits for the uses of such a function that run,
    and for a give-back of it that runs on another thread, so each is given
    back as it returns. {!Driver} calls [finish_releases] before each open,
    reset, detach and attach.

    Raises [Invalid_argument], having done nothing, if the calling thread
    holds a lock or is inside a use, as a callback inside one does
    ({{!Rig_pci.errors}errors}). Once each function is finished, it raises
    the first exception that ended a finish, and leaves that function to its
    next call. *)

val machine : t -> Machine.t
(** [machine f] is the machine [f] is on. *)

val bus : t -> string
(** [bus f] is [f]'s bus address on its machine. *)

val addressing : t -> Machine.addressing
(** [addressing f] is how [f] reaches system memory. *)

val released : t -> bool
(** [released f] is [true] iff a {!release} of [f] has begun. *)

val failed : t -> string option
(** [failed f] is why [f] can no longer be reached, if it cannot: its machine
    has failed ({!Machine.failed}), or [f] has left the bus and its vendor ID
    reads [0xffff]. It reads configuration space, so a driver calls it off its
    hot path, at the checks {{!Rig_pci.errors}errors} lists.

    Raises [Invalid_argument] if [f] was released. *)

val wait : t -> us:int -> string -> (unit -> bool) -> (unit, string) result
(** [wait f ~us what cond] calls [cond], at least once, until it is [true] or at
    least [us] microseconds have passed on a monotonic clock. It is [Ok ()] iff
    [cond] became [true] while [f]'s machine had not failed. The machine's
    state is read after each call of [cond], so an answer computed from the all
    ones of a failed machine is not trusted. A long wait sleeps between calls
    and holds no core.

    [Error why] otherwise, with [why] starting with [what]. At the bound [why]
    is {!failed}'s reason if [f] has failed, or says that [what] did not
    complete in time and names the bound. A function that leaves the bus while
    its machine does not fail is noticed only at the bound, since noticing it
    reads configuration space. A released [f] waits as its machine does.

    Raises [Invalid_argument] if [us < 0]. *)

val set_bus_master : t -> bool -> unit
(** [set_bus_master f on] lets [f] master the bus, reaching system memory by
    DMA, iff [on]. It sets or clears the bus master bit of [f]'s command
    register (PCI Express Base Specification, 7.5.1.1.3) and keeps its other
    bits. {!release} clears it. The owner must not call it once it has
    released [f]. A use's call, which may follow another thread's release, is
    valid until [f] is given back, which no use outlives.

    Raises [Invalid_argument] if [f] was given back ({!release}). *)

(** {1:config Configuration space}

    A function's configuration space is 4096 bytes of little-endian values.
    Linux shows a process only its first 64 bytes, unless the process took the
    function behind an IOMMU or has [CAP_SYS_ADMIN].

    A read the function or the system does not answer gives all ones. This
    happens on a function that has left the bus or a failed machine, and past
    the first 64 bytes of a function taken physically by a process without
    [CAP_SYS_ADMIN]. A write reads its bytes back, so it has reached the
    function when it returns. A write the system refuses is dropped.

    Each access raises [Invalid_argument] if its bytes are not in the
    configuration space or the function was released. An error the system gives
    only for a bug in this library raises [Failure], naming it. *)

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
    and as another function on the bus reaches it, with its size in bytes. It
    is [None] if [f] has no BAR [i], such as the upper index of a 64-bit BAR or
    an index past 5.

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

    With [~combine:true] the window combines ({!Window.accesses}) where the
    machine allows it, which is on a prefetchable BAR of a function taken
    physically. Elsewhere it is the window [~combine:false] makes. Map
    registers and doorbells with [~combine:false]. Without [combine], the
    window maps BAR [i] as its live windows do, or with [~combine:false] if it
    has none. A window mapped into the process on a BAR whose register marks
    it prefetchable, memory into which host bridges may merge the processor's
    writes (PCI Local Bus Specification 2.2, 6.2.5.1), takes copies wider than
    a word, whatever [combine] ({!Window.accesses}).

    [Error why] if VFIO or the kernel does not let the process map the BAR, or
    if [f] is taken physically and the file of its BAR [i] is shorter than the
    BAR, naming the file. An access past a mapped file's end would end the
    process.

    Raises [Invalid_argument] if [i < 0], [f] has no BAR [i], the bytes do not
    lie in the BAR, or a live window of [f] maps BAR [i] with the other
    [combine]. The processor maps the same addresses one way at a time. *)

val unmap : t -> Window.t -> unit
(** [unmap f w] unmaps [w].

    Raises [Invalid_argument] if [w] is not a live window {!map} made on [f]. *)

(** {1:events Interrupts and reset} *)

val interrupt : t -> int -> bool
(** [interrupt f ms] waits at most [ms] milliseconds for an interrupt of [f],
    with the OCaml runtime released, and is [true] iff one arrived. A function
    taken physically has interrupts only when bound to [vfio-pci], in VFIO's
    no-IOMMU mode. Without interrupts it is [false] immediately.

    Raises [Invalid_argument] if [ms < 0]. *)

val reset : t -> (unit, string) result
(** [reset f] resets [f] with the reset Linux has for it, which clears the
    state a previous driver left, and waits at most a second for [f] to answer
    again.

    [Error why] naming the file if the process may not reset [f] or Linux has
    no reset for it, if [f] does not answer in time, or if its machine failed.
*)

(** {1:dma System memory}

    A function reaches system memory of its machine at addresses given as
    (address, bytes) {e runs}, in order. How it reaches the memory, and what
    becomes of the memory when the process dies, depend on how it was taken.

    Behind an IOMMU, the runs are device addresses that the process maps for
    the function, as one run, from 4 GiB up to [2{^44}]. Every function
    that rig drives reaches this range, and an address truncated to 32 bits
    falls outside it. The functions the process takes behind one machine's
    IOMMU share one VFIO container where Linux lets them, so memory that
    several of them reach maps and counts once, and each of them reaches the
    memory given to the others. The memory counts against the process's
    locked-memory limit ([memlock]), and a container holds at most
    [dma_entry_limit] mappings (a parameter of [vfio_iommu_type1], 65,535 by
    default) for all its functions. The memory belongs to the process. VFIO
    ends the function's access when the process's files close, at its death
    too.

    Taken physically, the runs are physical addresses, and reading them needs
    [CAP_SYS_ADMIN] for [/proc/self/pagemap]. The function keeps writing its
    memory after its process dies by a signal or a crash. So the memory lies in
    a file [/dev/hugepages/rig-pci-*], on a hugetlbfs of 2 MiB pages, which
    keeps its pages when the process dies and which the kernel neither swaps
    nor compacts. Each 2 MiB block of the process's addresses that holds memory
    maps one huge page of the file, which the memory of every function of the
    machine in that block shares. Each run lies in one block, and the runs of
    neighbouring blocks merge where their pages follow each other. The system
    needs a free huge page ([vm.nr_hugepages]) for each new block.

    Such a file goes once no function can write it any more:
    - The process's own file goes once it holds no memory and each function
      that reaches it has been released with its bus mastering off. At the
      process's exit, which turns off the bus mastering of the functions it
      still holds, the file goes whatever memory it holds, unless a
      function's bus mastering stayed on.
    - A file a dead process left goes once every function that reached it has
      been released with its bus mastering off, or it has been reset
      ({!Driver.reset}, which {!Driver.attach} runs too, as does
      {!Driver.open_} of a function such a file names). It goes at that
      reset, or at the next {!take} of a function of the machine.
    - A pinned block ({!pin}) stays, after the process's death too, until each
      pin is {!unpin}ned or the pinning function is reset ({!Driver.reset})
      or renewed ({!Driver.renew}).

    From a release's start until the function is given back, and after a
    release that could not turn its bus mastering off because the system
    refused the write, as a kernel locked down since the take does,
    {!free_dma} and {!unpin} give no memory back, since the function may
    still reach it. That memory stays the process's until it exits, and
    taken physically, past its exit.

    Either way, a function reaches only the addresses its engines hold,
    which its driver states with {!set_dma_bits}, as Linux's drivers do with
    [dma_set_mask].

    Without an IOMMU, the kernel can still move a page a function writes: to
    allocate a contiguous area (CMA), for virtio-mem, and to take memory
    offline or a failing page out of service. A kernel booted by kexec, without
    the platform's reset, inherits a GPU still writing. *)

val set_dma_bits : t -> int -> unit
(** [set_dma_bits f n] states that [f] reaches system memory at addresses below
    [2{^n}]. From then on {!alloc_dma} and {!pin} give [f] memory only where
    its runs lie below [2{^n}]. Until the first call [f] reaches 64 bits.
    Memory given before a call is kept.
    - Behind an IOMMU, the process places the runs below [2{^n}], and below
      [2{^44}] whatever [n], from 4 GiB up, so with [n <= 32] nothing fits.
      Memory that another function of the container maps already past
      [2{^n}] is [Ok None] until that function lets go of it.
    - Taken physically, the runs are where the kernel placed its huge pages,
      which no process chooses. Memory past [2{^n}] is [Ok None], and stays
      so while the kernel gives only pages that high.

    Raises [Invalid_argument] if [n < 1], [n > 64] or [f] was released. *)

val alloc_dma :
  ?contiguous:bool ->
  ?va:int ->
  t ->
  int ->
  ((Window.t * (int * int) list) option, string) result
(** [alloc_dma f n] is [Some (w, runs)], with [w] on [n] new, zeroed bytes of
    [f]'s machine, rounded up to {!Machine.val-page}, and [runs] the addresses
    at which [f] reaches them. The memory stays where it is: at [va], inside a
    range {!Machine.val-reserve} reserved, or where the machine chooses without
    [va]. With [~contiguous:true] (defaults to [false]) the memory is one run
    of at most 2 MiB, which starts on 2 MiB if [f] is taken physically and the
    memory is larger than a page.

    [Ok None] if the machine has no free memory for it now, within the limits
    {{!dma}system memory} names, or if the memory it gave lies past [f]'s
    addresses ({!set_dma_bits}). Freeing memory makes room, but for memory
    past [f]'s addresses taken physically.

    [Error why], naming what is missing, if the machine refuses: a privilege or
    a file system named above, or, taken physically, a 2 MiB block of [va] that
    holds memory of another machine.

    Raises [Invalid_argument] if [n <= 0] or rounding it up overflows, if [va]
    is not on a page, if the bytes from [va] lie in no reserved range (taken
    physically, the 2 MiB blocks that hold them), if [contiguous] memory is
    larger than 2 MiB, or if [va] is not on 2 MiB for contiguous memory larger
    than a page on a function taken physically. *)

val free_dma : t -> Window.t -> unit
(** [free_dma f w] frees [w], and [f] reaches it no more. From a release's
    start until [f] is given back, or after a release that could not turn
    [f]'s bus mastering off, [w] stays instead ({{!dma}system memory}).

    Raises [Invalid_argument] if [w] is not a live window {!alloc_dma} returned
    for [f]. *)

val pin : t -> int -> int -> ((int * int) list option, string) result
(** [pin f a n] locks the [n] bytes at [a] of [f]'s machine, which start on a
    page, and is [Some runs], with [runs] the addresses at which [f] reaches
    them. Pins are counted. The memory stays locked until each pin is
    {!unpin}ned, and behind an IOMMU [(a, n)] stays mapped for [f] as long.
    Behind an IOMMU, a child of [fork] gets copies of pinned pages of private
    mappings, and the parent keeps the pinned pages.

    Taken physically, [f] pins only memory that {!alloc_dma} gave on [f]'s
    machine, for [f] or another function, since the process's own pages go back
    to the system when it dies with [f] still writing them. The pinned memory
    then outlives the process ({{!dma}system memory}).

    [Ok None] if the machine has no room to pin them now, within the limits
    {{!dma}system memory} names, or if they lie past [f]'s addresses
    ({!set_dma_bits}). Unpinning or freeing memory makes room, but for pages
    past [f]'s addresses taken physically.

    [Error why] if the system refuses to lock the pages or to read their
    addresses, or if [f] is taken physically and the pages are not memory
    {!alloc_dma} gave on [f]'s machine.

    Raises [Invalid_argument] if [a] is not on a page or [n <= 0]. *)

val unpin : t -> int -> int -> unit
(** [unpin f a n] releases one pin of the [n] bytes at [a]. From a release's
    start until [f] is given back, or after a release that could not turn
    [f]'s bus mastering off, the memory stays locked instead
    ({{!dma}system memory}).

    Raises [Invalid_argument] if [(a, n)] is not pinned for [f]. *)

(**/**)

(* [take_locked m l] is {!take} of the function [l] locks, for a lock of [m]'s
   files ({!Machine.files}): the take holds the lock [l] is, which a physical
   take keeps until its release. *)
val take_locked : Machine.t -> 's Sysfs.lock -> (t, string) result

(* [set_own_length n] makes each range of this process's own addresses that
   {!alloc_dma} reserves from then on, for memory of a function taken
   physically without [va], [n] bytes, or as many as a request needs, and is
   the length it replaces. A test shrinks it to pass a range with little
   memory. *)
val set_own_length : int -> int

(* [use f] counts a use of [f], as {!pin} and {!alloc_dma} are, and is
   [true], or is [false], counting nothing, if a release of [f] has begun. A
   [true] use must end with {!unuse}, on every exit: a caller installs its
   handler as the next step, with nothing between that allocates or polls. A
   use may write [f]'s BAR windows, which [f]'s give-back unmaps once no use
   runs. *)
val use : t -> bool

(* [unuse f] ends a use {!use} counted. The last use to end once a release
   has begun gives [f] back ({!release}), and raises what ends that. *)
val unuse : t -> unit

(* [unmap_used f w] is {!unmap} of [w] from inside a use ({!use}), which a
   release begun meanwhile does not refuse: the give-back lists the windows
   left only once no use runs. *)
val unmap_used : t -> Window.t -> unit

(* [reaching f] is [true] iff [f] was released and may still reach the
   memory it was given: it is not given back yet, or its release could not
   turn [f]'s bus mastering off. {!free_dma} and {!unpin} give nothing back
   then. *)
val reaching : t -> bool

(* [finish_held ()] is {!finish_releases} without its check of the calling
   thread's locks, for {!Driver}, which checks at its entries and then holds
   its own lock. *)
val finish_held : unit -> unit

(* [inherited f] is the [inherited] of [f]'s take ({!Machine.fn}) until
   {!forget} returns [Ok ()], and [false] after. *)
val inherited : t -> bool

(* [keep f] keeps [f] taken until the process exits, since another function
   may write the memory its BAR reaches ({!Memory.bus}): from then on
   {!release} turns [f]'s bus mastering off and gives back nothing else, and
   {!take} of [f]'s bus on its machine, in this process or a child of fork,
   is [Error]. The process's exit gives [f] back as it gives back every
   function it holds. Keeping it again does nothing.

   Raises [Invalid_argument] if [f] was released. *)
val keep : t -> unit

(* [forget f] is the [forget] of [f]'s take ({!Machine.fn}). The memory this
   process allocated stays listed for [f], so that the process's own death
   still leaves [f] to be reset.

   Raises [Invalid_argument] if [f] was released. *)
val forget : t -> (unit, string) result
