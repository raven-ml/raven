(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA devices, without CUDA.

    Opens NVIDIA GPUs as {!Nx_device.t}s through one of two interfaces, which
    their names tell apart:
    - {!Kernel}, the resource manager interface of NVIDIA's Linux kernel driver
      ([/dev/nvidiactl] and [/dev/nvidia-uvm]) of the releases 570, 580, 610 and
      615. The driver owns the GPU and shares it with other programs.
    - {!Pci}, the runtime's own driver: it takes the GPU's PCI function, which
      {!detach} detached from its kernel driver, boots the GPU's security
      processors and its GSP with their firmware, and manages the GPU's memory
      and page tables itself. The GPU is this process's until it exits.

    The two differ in what a program observes: who else may use the GPU, what a
    fault or a crash leaves behind, how much memory the host maps, and which
    copies go direct. Nothing chooses between them for the caller: {!get} takes
    the interface.

    {b Numbering.} GPU [i] is the [i]th of the machine's NVIDIA GPUs, its
    display controllers in bus order ({!Nx_device_support.Pci.compare_address}),
    under both interfaces: index [i] names the same GPU through either, and
    taking a GPU over {!Pci} renumbers none. Through {!Kernel} the GPUs are
    named ["NV"], ["NV:1"], ["NV:2"], ..., and through {!Pci} ["NV-PCI"],
    ["NV-PCI:1"], .... A process uses one interface, chosen by its first open.
    Both need Linux: elsewhere {!count} is [0] and {!get} says why.

    {b Other machines.} Given the host of another machine ([nx.remote.device]),
    {!count} and {!get} reach that machine's GPUs, over {!Pci} through the
    machine's server: they are named ["NV-PCI@HOST:PORT"],
    ["NV-PCI:1@HOST:PORT"], ..., their host memory is that machine's memory,
    every register access and pushbuffer write crosses the network, and they
    have no interrupts. Such a GPU copies directly only to GPUs of its machine.
    The machine needs what {!Pci} needs here, and the process none of it.

    {b NV and CUDA devices.} [nx.cuda.device] opens GPUs through NVIDIA's CUDA
    driver library as ["CUDA"], ["CUDA:1"], ...; this library opens them without
    it. An NV device and a CUDA device are two devices even when they are the
    same GPU: each has its own memory, timeline and budget, and
    {!Nx_device.Buffer.copy} between them goes through the host's staging
    memory. Both number GPUs in bus order, so ["NV:1"] and ["CUDA:1"] are the
    same GPU when CUDA sees every NVIDIA GPU of the machine. CUDA cannot open a
    GPU {!detach} detached.

    {b Memory.} Buffers are GPU memory, which the host does not address.
    {!Nx_device.Buffer.copy} moves their bytes on the GPU's copy engine:
    directly from and to host memory the GPU addresses, and through the host's
    staging memory from and to other host memory.

    {!Nx_device.Buffer.create}[ ~memory:Mapped] allocates GPU memory that the
    host also addresses, through the GPU's window onto it (BAR1). The host's
    writes through BAR1 are write-combined; the full fence before a doorbell
    drains them, so the work the doorbell submits sees them. NV needs no flush
    besides; work that reads memory the host rewrote since the device's previous
    work starts by invalidating the compute engine's instruction, data and
    constant caches. Without a BAR that covers the GPU's memory, under {!Pci},
    or once BAR1 is full, under {!Kernel}, mapped memory is pinned memory.

    Host memory the GPU addresses is coherent for it:
    - {!Nx_device.Buffer.create}[ ~memory:Pinned] allocates it, system memory
      the GPU snoops, and it counts in the device's budget, which defaults to
      the size of the GPU's memory heap;
    - {!Nx_device.Buffer.borrow} maps host memory, whole pages of it, and
      page-locks it while mapped. It counts in no budget;
    - the host's staging memory, 128 MiB, is mapped at the first copy that needs
      it and kept for the life of the process. Between two NV GPUs that reach
      each other's memory, through peer access under {!Kernel} or a memory BAR
      as large as the GPU's memory under {!Pci} with no IOMMU between them, the
      source's copy engine writes the destination; otherwise the bytes go
      through the staging memory.

    {b Timestamps} count the GPU's timer in nanoseconds
    ({!Nx_device.Driver.Device_clock}); the copy engine stamps it for the
    runtime's profiles.

    {b Programs} are functions of cubins, the ELF objects NVIDIA's compilers
    make for the device's {!Nx_device.arch}: the function [name] is the code of
    the section [.text.name], and a cubin may hold several, each with its own
    registers, stack and constant bank 0. PTX is not loaded: compile it to a
    cubin first. A cubin is uploaded with its relocations applied, once while a
    program of it or its {!kernel}'s image is reachable
    ({!Nx_device.Program.load}).

    {b Faults and hangs.} A fault the GPU reports, such as a page fault or an
    error of a streaming multiprocessor, loses the device ({!Nx_device.Lost})
    with the GPU's report when a wait finds it. Work that does not signal within
    the device's {!Nx_device.timeout}, 30 seconds unless
    {!Nx_device.set_timeout} sets another, loses it too. Nothing recovers a lost
    device in the process. Under {!Pci}, an open refuses a GPU that a failed or
    earlier boot left with its secure region up, until {!reset} resets it.

    {b Under {!Pci}}, opening a GPU changes nothing outside the process. The GPU
    must be detached from its kernel driver ({!detach}), reset if it was booted
    before ({!reset}), and its firmware at hand ({!fetch_firmware}): {!get}
    fails otherwise, naming the call. A GPU an administrator bound to [vfio-pci]
    on a machine with an IOMMU opens without root, given access to its IOMMU
    group's [/dev/vfio/N] and a locked-memory limit that holds its system memory
    ({!Nx_device_support.Pci}). Otherwise opening needs root, or the
    capabilities and file permissions to take PCI functions and to lock memory
    and read its physical addresses ({!Nx_device_support.Sysmem}). It supports
    the GPUs of the Ampere (GA10x), Ada (AD10x) and Blackwell (GB20x) families,
    and boots them with the firmware they were validated with, verified by
    digest ({!get}). Booting takes seconds. *)

(** The type for the interfaces that reach NVIDIA GPUs. *)
type interface =
  | Kernel  (** NVIDIA's kernel driver, [/dev/nvidiactl]. *)
  | Pci  (** The runtime's own driver, over PCI. *)

val count : ?host:Nx_device.t -> unit -> int
(** [count ()] is the number of NVIDIA GPUs of the machine of [host] (defaults
    to {!Nx_device.host}), whatever driver holds them: the indices [0], ...,
    [count () - 1] of both interfaces. [0] on systems other than Linux.

    Raises [Invalid_argument] if [host] is no host, and {!Nx_device.Lost} with
    [host] if [host]'s machine cannot be reached. *)

val get :
  ?host:Nx_device.t ->
  interface:interface ->
  ?firmware:string ->
  int ->
  (Nx_device.t, string) result
(** [get ~interface i] is NVIDIA GPU [i] through [interface], opened by the
    first call that succeeds; every later call returns the same value. The first
    successful open fixes the process's interface. Given another machine's
    [host] (defaults to {!Nx_device.host}), it is that machine's GPU [i], which
    only {!Pci} reaches.

    Under {!Pci}, each of the GPU's firmware images must have the SHA-256 digest
    it was validated with. An image is read from the directory [firmware], if
    given, which suits a machine without network access; then from
    [/lib/firmware], plain or compressed, when the distribution ships that
    version; then from the user's cache, where {!fetch_firmware} downloads it. A
    file of [firmware] with another digest is refused. Under {!Kernel},
    [firmware] is ignored: the kernel driver loads the firmware. An open
    downloads nothing.

    [Error msg] says why the GPU cannot be opened, for example that
    [i >= count ()], that the process's interface is the other one, that another
    machine's GPU was asked for under {!Kernel}, that the kernel driver does not
    hold the GPU or is of another release, naming it, that {!Pci} does not
    support the GPU's family, that a privilege is missing, or that a firmware
    image differs, naming the file. A precondition of {!Pci} that does not hold
    names the call that establishes it: a GPU not detached names {!detach}, one
    booted before {!reset}, a missing firmware image {!fetch_firmware}. [msg]
    starts with the GPU's name under [interface], such as
    ["NV:2: no GPU 2; there are 2 NVIDIA GPUs"]. A failed open gives back what
    it took, so a later [get] may open the GPU.

    Raises [Invalid_argument] if [i < 0] or if [host] is no host, and
    {!Nx_device.Lost} with [host] if [host]'s machine cannot be reached. *)

(** {1:machine Changes to the machine}

    Each function below makes one change to this machine, which persists after
    the process, and prepares or undoes the preconditions of {!Pci}. Each acts
    on GPU [i] of this machine, numbered as for {!get}, refuses a GPU a device
    of the process holds, and needs Linux. [Error msg] says why it failed, and
    starts with the GPU's name: under {!Pci}, but under {!Kernel} for {!attach}.
    Another machine's GPUs are detached and reset by a program on that machine;
    they boot with this machine's firmware images. *)

val detach : int -> (unit, string) result
(** [detach i] detaches GPU [i] from its kernel driver, for {!Pci}: it unbinds
    the driver, unless that is [vfio-pci], removes the GPU's other PCI
    functions, such as its audio function, enables the GPU's function, and makes
    its memory window (BAR1) as large as the platform allows. The kernel
    driver's users lose the GPU, a screen it drives among them, and so does
    {!Kernel}. This persists until {!attach} or a reboot. It needs root, or
    write access to the files {!Nx_device_support.Pci.detach} writes. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] back to its kernel driver, for {!Kernel}: Linux
    rescans the PCI bus, which brings back the functions {!detach} removed, and
    binds the GPU's driver. This persists. It needs root, or write access to
    [/sys/bus/pci/rescan] and [/sys/bus/pci/drivers_probe]. [Error msg] says if
    no driver takes the GPU, such as when the [nvidia] module is not loaded. *)

val reset : int -> (unit, string) result
(** [reset i] resets GPU [i], which must be detached, with its PCI function's
    reset: it stops what runs on the GPU and clears what a boot left in it, by
    its kernel driver or an earlier process. {!Pci} opens a GPU only once it is
    reset. This persists. It needs root, or the permissions of
    {!Nx_device_support.Pci.reset}. *)

val fetch_firmware : int -> (unit, string) result
(** [fetch_firmware i] downloads the firmware images GPU [i] boots with under
    {!Pci} into the user's cache, [$XDG_CACHE_HOME/raven/firmware]
    ([~/.cache/raven/firmware] by default, [$RAVEN_CACHE_ROOT/firmware] when
    set), from linux-firmware with the system's [libcurl], verified by digest.
    Images [/lib/firmware] or the cache already holds are not downloaded. This
    persists. It needs network access and write access to the cache, and no
    privilege. [Error msg] says if {!Pci} does not support the GPU's family, or
    which download failed. *)

(** {1:low Low-level}

    For the libraries that submit work to an NV device, inside
    {!Nx_device.submit}: they write methods into pushbuffers and hand the
    pushbuffers to the device's channels. Every address below is the GPU's.

    Work for the value [v] first acquires, on its channel, the semaphore at the
    signal word ({!Nx_device.signal_word}), with the 64-bit circular
    greater-or-equal test against [v - 1], and ends by writing [v] into the
    signal word: with one 64-bit semaphore release, or with a one-word release
    of its low 32 bits followed, when they are [0], by one of its high 32 bits.
    Mid-write the word never reads above its old value, and a late high word
    never takes it back. The values thus complete in order across the channels.
    The device's own copies follow the same rule on the copy channel. Work waits
    for another device's pair of {!Nx_device.Submission.waits} the same way, on
    that device's signal word. The memory the device allocates lies below
    [2{^40}], and a semaphore that records a timestamp writes 16 bytes, which
    start on 16 bytes. *)

type t
(** The type for the channels and properties of a device. *)

val of_device : Nx_device.t -> t option
(** [of_device d] is the channels and properties of [d], if [d] is an NV device.
*)

type channel = {
  ring : Nx_device.Buffer.t;
      (** The channel's GPFIFO: 64-bit entries, each naming a pushbuffer
          segment. The host writes it at its GPU address too. *)
  gp_get : Nx_device.Buffer.t;
      (** The 32-bit index of the next entry the GPU fetches, which the GPU
          stores and the host reads. *)
  gp_put : Nx_device.Buffer.t;
      (** The 32-bit index after the last entry written, which the GPU reads. *)
  put : Nx_device.Buffer.t;
      (** The 64-bit count of the entries ever written, which grows without
          wrapping. *)
  doorbell : Nx_device.Buffer.t;
      (** The 32-bit doorbell that wakes the channel. *)
  token : int;  (** The work submit token the doorbell takes. *)
}
(** The type for channels, the GPU's hardware queues. A writer, with the device
    taken:
    - waits until [gp_get] leaves room: at most [entries - 1] entries written
      and not fetched, where [entries] is the length of [ring]. Inside
      {!Nx_device.submit}, each channel has room for half its ring, which
      [submit] waits for: work submitted there writes at most half a ring into
      each channel and need not wait;
    - writes its entry at [put mod entries]: the address of its segment, which
      lies below [2{^40}], and the segment's length in 32-bit words, in the
      entry format of the channel's class ({!props}[.gpfifo_class]);
    - stores [put + 1] into [put] and [(put + 1) mod entries] into [gp_put],
      then [token] into [doorbell].

    A segment stays unchanged until the work it holds has signaled its value. *)

type props = {
  sm_version : int;
      (** The streaming multiprocessors' version, such as [0x806], whose
          {!Nx_device.arch} is ["sm_86"]. *)
  sass_version : int;  (** The version of the machine code they run. *)
  gpcs : int;  (** The number of graphics processing clusters. *)
  tpcs_per_gpc : int;  (** The number of texture processing clusters of one. *)
  sms_per_tpc : int;  (** The number of multiprocessors of one of those. *)
  warps_per_sm : int;  (** The most warps a multiprocessor runs at once. *)
  gpfifo_class : int;  (** The class of the channels. *)
  compute_class : int;  (** The class of the compute engine. *)
  dma_class : int;  (** The class of the copy engine. *)
}
(** The type for the GPU's properties that its work depends on. *)

val compute : t -> channel
(** [compute n] is the compute channel, bound to [compute_class]. *)

val copy : t -> channel
(** [copy n] is the copy channel, bound to [dma_class], on which
    {!Nx_device.Buffer.copy} runs. *)

val shared_window : t -> nativeint
(** [shared_window n] is the address at which kernels' shared memory appears. *)

val local_window : t -> nativeint
(** [local_window n] is the address at which kernels' local memory appears. *)

val props : t -> props
(** [props n] is the GPU's properties. *)

type kernel = {
  image : Nx_device.Buffer.t;
      (** The uploaded cubin: its image, as the ELF object lays it out with
          sections aligned to 128 bytes ({!Nx_device_elf.load}), relocated, then
          zeros up to the next multiple of 4 KiB and 4 KiB more, which the GPU's
          instruction prefetch may read past the code. It is memory of the
          device the host writes, which keeps the cubin loaded while it is
          reachable ({!Nx_device.Program.code}). *)
  entry : nativeint;  (** The address of the function's first instruction. *)
}
(** The type for kernels: a function of an uploaded cubin. A launch reads the
    rest of what it needs, such as registers and constant banks, from the cubin
    itself. *)

val kernel : Nx_device.Program.t -> kernel option
(** [kernel p] is the kernel of [p], if [p] is loaded on an NV device. Its
    {!Nx_device.Program.handle} is its [entry]. *)

type local_memory = {
  address : nativeint;  (** The local memory's first byte. *)
  bytes : int;  (** Its size. *)
  per_thread : int;  (** The bytes of each thread, a multiple of 32. *)
}
(** The type for the local memory of a device's kernels. *)

val local_memory : t -> int -> local_memory
(** [local_memory n bytes] is the device's local memory for kernels of up to
    [bytes] bytes per thread, grown if smaller: its address is set on the
    compute channel as work on the device's timeline. Kernels that need none
    ([bytes <= 0]) get no memory, of 0 bytes at address 0, until a larger
    [bytes] allocates some. Memory it replaces returns to the device once
    unreachable. Call it before {!Nx_device.submit}, not inside.

    Raises {!Nx_device.Out_of_memory} if the device cannot allocate it. *)

val invalidate_caches : t -> unit
(** [invalidate_caches n] writes back and invalidates the device's caches, so
    that the next work starts with them cold, as timing kernels in isolation
    needs. *)
