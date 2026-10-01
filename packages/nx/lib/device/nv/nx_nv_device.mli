(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA devices, without CUDA.

    Opens NVIDIA GPUs as {!Nx_device.t}s named ["NV"], ["NV:1"], ["NV:2"], ...
    through one of two interfaces:
    - {!Kernel}, the resource manager interface of NVIDIA's Linux kernel driver
      ([/dev/nvidiactl] and [/dev/nvidia-uvm]) of the releases 570, 580 and 610.
      The driver owns the GPU and shares it with other programs.
    - {!Pci}, the runtime's own driver: it takes the GPU's PCI function from its
      kernel driver, boots the GPU's security processors and its GSP with their
      firmware, and manages the GPU's memory and page tables itself. The GPU is
      this process's until it exits.

    A process uses one interface, chosen by its first open. Both need Linux:
    elsewhere {!count} is [0] and {!get} says why.

    {b Other machines.} Given the host of another machine ([nx.remote.device]),
    {!count} and {!get} reach that machine's GPUs, over {!Pci} through the
    machine's server: they are named ["NV@HOST:PORT"], ["NV:1@HOST:PORT"], ...,
    their host memory is that machine's memory, every register access and
    pushbuffer write crosses the network, and they have no interrupts. Such a
    GPU copies directly only to GPUs of its machine. The machine needs what
    {!Pci} needs here, and the process none of it.

    {b NV and CUDA devices.} [nx.cuda.device] opens GPUs through NVIDIA's CUDA
    driver library as ["CUDA"], ["CUDA:1"], ...; this library opens them without
    it. An NV device and a CUDA device are two devices even when they are the
    same GPU: each has its own memory, timeline and budget, and
    {!Nx_device.Buffer.copy} between them goes through the host's staging
    memory. Each library numbers GPUs its own way, so ["NV:1"] and ["CUDA:1"]
    need not be the same GPU. Under {!Pci} the kernel driver lets go of the GPU,
    which CUDA then cannot open.

    {b Memory.} Buffers are GPU memory, which the host does not address.
    {!Nx_device.Buffer.copy} moves their bytes on the GPU's copy engine:
    directly from and to host memory the GPU addresses, and through the host's
    staging memory from and to other host memory. Host memory the GPU addresses
    is coherent for it:
    - {!Nx_device.Buffer.create}[ ~pinned:true] allocates it, system memory the
      GPU snoops, and it counts in the device's budget, which defaults to the
      size of the GPU's memory heap;
    - {!Nx_device.Buffer.borrow} maps host memory, whole pages of it, and
      page-locks it while mapped. It counts in no budget;
    - the host's staging memory, 128 MiB, is mapped at the first copy that needs
      it and kept for the life of the process. Between two NV GPUs that reach
      each other's memory, through peer access under {!Kernel} or a memory BAR
      as large as the GPU's memory under {!Pci}, the source's copy engine writes
      the destination; otherwise the bytes go through the staging memory.

    {b Timestamps} count the GPU's timer in nanoseconds
    ({!Nx_device.Driver.Device_clock}); the copy engine stamps it for the
    runtime's profiles.

    {b Programs} are functions of cubins, the ELF objects NVIDIA's compilers
    make for the device's {!Nx_device.arch}: the function [name] is the code of
    the section [.text.name], and a cubin may hold several, each with its own
    registers, stack and constant bank 0. PTX is not loaded: compile it to a
    cubin first. A cubin is uploaded once per device and kept, with its
    relocations applied.

    {b Faults and hangs.} A fault the GPU reports, such as a page fault or an
    error of a streaming multiprocessor, loses the device ({!Nx_device.Lost})
    with the GPU's report when a wait finds it. Work that does not signal within
    the device's {!Nx_device.timeout}, 30 seconds unless
    {!Nx_device.set_timeout} sets another, loses it too. Nothing recovers a lost
    device in the process. Under {!Pci}, the next process that opens the GPU
    resets it before booting it.

    {b Under {!Pci}}, opening a GPU needs root, or the capabilities and file
    permissions to take and reset PCI functions ({!Nx_device_support.Pci}) and
    to lock memory and read its physical addresses
    ({!Nx_device_support.Sysmem}). It detaches the GPU's kernel driver,
    including the display driver of a GPU that drives a screen. It supports the
    GPUs of the Ampere (GA10x), Ada (AD10x) and Blackwell (GB20x) families, and
    boots them with the firmware they were validated with, which it downloads
    once and verifies by digest ({!get}). Booting takes seconds. *)

(** The type for the interfaces that reach NVIDIA GPUs. *)
type interface =
  | Kernel  (** NVIDIA's kernel driver, [/dev/nvidiactl]. *)
  | Pci  (** The runtime's own driver, over PCI. *)

val count : ?host:Nx_device.t -> ?interface:interface -> unit -> int
(** [count ()] is the number of NVIDIA GPUs that [interface] reaches: under
    {!Kernel}, those of NVIDIA's kernel driver; under {!Pci}, the PCI functions
    of the GPUs it supports, whatever driver they have. [interface] defaults to
    the process's interface once a device is open, and before that to {!Kernel}
    if [/dev/nvidiactl] exists and {!Pci} otherwise. [0] on systems other than
    Linux. Given another machine's [host], it is the number of that machine's
    GPUs under {!Pci}.

    Raises [Invalid_argument] if [host] is no host, and {!Nx_device.Lost} with
    [host] if [host]'s machine cannot be reached. *)

val get :
  ?host:Nx_device.t ->
  ?interface:interface ->
  ?firmware:string ->
  int ->
  (Nx_device.t, string) result
(** [get i] is the NVIDIA GPU [i] of [interface] (defaults as for {!count}),
    opened by the first call that succeeds; every later call returns the same
    value. The first successful open fixes the process's interface. Given
    another machine's [host] (defaults to {!Nx_device.host}), it is that
    machine's GPU [i], under {!Pci}.

    Under {!Pci}, each of the GPU's firmware images must have the SHA-256 digest
    it was validated with. An image is read from the directory [firmware], if
    given, which suits a machine without network access; then from
    [/lib/firmware], plain or compressed, when the distribution ships that
    version; then from the user's cache, [$XDG_CACHE_HOME/raven/firmware]
    ([~/.cache/raven/firmware] by default, [$RAVEN_CACHE_ROOT/firmware] when
    set), where it is otherwise downloaded from linux-firmware, once, with the
    system's [libcurl]. A file of [firmware] or a download with another digest
    is refused. Under {!Kernel}, [firmware] is ignored: the kernel driver loads
    the firmware.

    [Error msg] says why the GPU cannot be opened, for example that
    [i >= count ~interface ()], that the process's interface is the other one,
    that another machine's GPU was asked for under {!Kernel}, that the kernel
    driver is of another release, naming it, that a privilege is missing, or
    that a firmware image is missing or differs, naming the file. [msg] starts
    with the GPU's name, such as ["NV:2: no GPU 2; there are 2 NVIDIA GPUs"].
    Under {!Kernel}, a failed open gives back what it took, so a later [get] may
    open the GPU.

    Raises [Invalid_argument] if [i < 0] or if [host] is no host, and
    {!Nx_device.Lost} with [host] if [host]'s machine cannot be reached. *)

val v :
  ?host:Nx_device.t ->
  ?interface:interface ->
  ?firmware:string ->
  int ->
  Nx_device.t
(** [v i] is like {!get} but raises [Failure] with [get]'s message when the GPU
    cannot be opened. *)

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

val interface : t -> interface
(** [interface n] is the interface the device was opened through. *)

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
          instruction prefetch may read past the code. It is in memory of the
          device that the device keeps for as long as it lives. *)
  entry : nativeint;  (** The address of the function's first instruction. *)
  code_bytes : int;  (** The size of the function's code. *)
  registers : int;  (** The registers a thread uses. *)
  shared_bytes : int;
      (** Shared memory per block, with the 1 KiB the hardware reserves. *)
  local_bytes : int;  (** Local memory per thread, with its stack. *)
  param_offset : int;
      (** The offset of the parameters in constant bank 0, after the words the
          launch fills itself. *)
  banks : (int * nativeint * int) list;
      (** The constant banks, as (index, address, bytes): those of the cubin at
          their uploaded address, and bank 0 at the cubin's defaults, which a
          launch replaces with its own parameters. *)
  max_threads : int;
      (** The most threads a block may have, given [registers]. *)
}
(** The type for the launch parameters of a kernel. *)

val kernel : Nx_device.Program.t -> kernel option
(** [kernel p] is the launch parameters of [p], if [p] is loaded on an NV
    device. Its {!Nx_device.Program.handle} is its [entry]. *)

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
