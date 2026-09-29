(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD devices.

    Opens AMD GPUs as {!Nx_device.t}s named ["AMD"], ["AMD:1"], ["AMD:2"], ...
    through one of two interfaces:
    - {!Kernel}, the compute interface of Linux's [amdgpu] kernel driver. The
      driver owns the GPU and shares it with other programs.
    - {!Pci}, the runtime's own driver: it takes the GPU's PCI function from its
      kernel driver, loads the GPU's firmware, boots its blocks, and manages its
      memory and page tables itself. The GPU is this process's until it exits.

    A process uses one interface, chosen by its first open. Both need Linux:
    elsewhere {!count} is [0] and {!get} says why.

    {b Other machines.} Given the host of another machine ([nx.remote.device]),
    {!count} and {!get} reach that machine's GPUs, over {!Pci} through the
    machine's server: they are named ["AMD@HOST:PORT"], ["AMD:1@HOST:PORT"],
    ..., their host memory is that machine's memory, every register access and
    ring write crosses the network, and they have no interrupts. Such a GPU
    copies directly only to GPUs of its machine. The machine needs what {!Pci}
    needs here, and the process none of it.

    {b Memory.} Buffers are GPU memory, which the host does not address.
    {!Nx_device.Buffer.copy} moves their bytes on the GPU's copy engine (SDMA):
    directly from and to host memory the GPU addresses, and through the host's
    staging memory from and to other host memory. Host memory the GPU addresses
    is registered with it and coherent for it:
    - {!Nx_device.Buffer.create}[ ~host:true] allocates it, and it counts in the
      device's budget, which defaults to the GPU's memory size;
    - {!Nx_device.Buffer.borrow} maps host memory, whole pages of it. It counts
      in no budget;
    - the host's staging memory, 128 MiB, is mapped at the first copy that needs
      it and kept for the life of the process. Between two AMD GPUs that reach
      each other's memory, over a direct link or a large memory BAR, the
      source's copy engine writes the destination; otherwise the bytes go
      through the staging memory.

    {b Timestamps} count the GPU's global clock, 100 MHz
    ({!Nx_device.Device_clock}); the copy engine stamps it for the runtime's
    profiles.

    {b Programs} are functions of AMD GPU code objects, ELF objects compiled for
    the device's {!Nx_device.arch}: the function [name] is the kernel whose
    descriptor is the symbol [name ^ ".kd"]. A code object is uploaded once per
    device and kept.

    {b Faults and hangs.} A fault the GPU reports, such as a page fault, fails
    the device with the driver's report when a wait finds it. Work that does not
    signal within the device's {!Nx_device.timeout}, 30 seconds unless
    {!Nx_device.set_timeout} sets another, fails it too. Nothing recovers a
    failed device in the process. Under {!Pci}, the process stops the engines of
    a failed GPU at exit and takes its bus mastering away, so that it cannot
    reach the memory the process releases; the next process that opens the GPU
    finds it was left failed, and resets it before booting it.

    {b Under {!Pci}}, opening a GPU needs root, or the capabilities and file
    permissions to take PCI functions ({!Nx_device_support.Pci}) and to lock
    memory and read its physical addresses ({!Nx_device_support.Sysmem}). It
    detaches the GPU's kernel driver, including the display driver of a GPU that
    drives a screen. It boots the GPU with the firmware it was validated with,
    which it downloads once and verifies by digest ({!get}). A GPU that the
    previous process closed cleanly boots in a fraction of a second; otherwise
    it is reset and fully booted, which takes seconds. *)

(** The type for the interfaces that reach AMD GPUs. *)
type interface =
  | Kernel
      (** The compute interface of the [amdgpu] kernel driver, [/dev/kfd]. *)
  | Pci  (** The runtime's own driver, over the GPU's PCI function. *)

val count : ?host:Nx_device.t -> ?interface:interface -> unit -> int
(** [count ()] is the number of AMD GPUs that [interface] reaches: under
    {!Kernel}, those of the [amdgpu] driver; under {!Pci}, the PCI functions of
    the GPUs it supports, whatever driver they have. [interface] defaults to the
    process's interface once a device is open, and before that to {!Kernel} if
    [/dev/kfd] exists and {!Pci} otherwise. [0] on systems other than Linux.
    Given another machine's [host], it is the number of that machine's GPUs
    under {!Pci}.

    Raises [Invalid_argument] if [host] is no host, and [Failure] if [host]'s
    machine cannot be reached. *)

val get :
  ?host:Nx_device.t ->
  ?interface:interface ->
  ?firmware:string ->
  int ->
  (Nx_device.t, string) result
(** [get i] is the AMD GPU [i] of [interface] (defaults as for {!count}), opened
    by the first call that succeeds; every later call returns the same value.
    The first successful open fixes the process's interface. Given another
    machine's [host] (defaults to {!Nx_device.host}), it is that machine's GPU
    [i], under {!Pci}.

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
    that another machine's GPU was asked for under {!Kernel}, that a privilege
    is missing, or that a firmware image is missing or differs, naming the file.

    Raises [Invalid_argument] if [i < 0] or if [host] is no host. *)

val v :
  ?host:Nx_device.t ->
  ?interface:interface ->
  ?firmware:string ->
  int ->
  Nx_device.t
(** [v i] is like {!get} but raises [Invalid_argument] with [get]'s message when
    the GPU cannot be opened. *)

val interface : Nx_device.t -> interface
(** [interface d] is the interface [d] was opened through.

    Raises [Invalid_argument] if [d] is not an AMD device. *)

(** {1:low Low-level}

    For the libraries that submit work to an AMD device, inside
    {!Nx_device.submit}: they write packets into its queues. Every address below
    is the same for the host and for the GPU. Work begins with a flush of the
    host data path (HDP), since the host's writes to GPU memory through the BAR,
    such as uploaded programs, are not otherwise visible to it.

    Work for the timeline value [v] first waits on its queue until the low 32
    bits of the signal word, the first word of {!Nx_device.timeline}, equal
    [v - 1], and ends by writing [v] into it: all 64 bits in one write, or its
    low 32 bits and then, only when they are [0], its high 32 bits. The values
    thus complete in order across the queues, and a high word written late never
    takes the word back. The device's own copies follow the same rule on the
    SDMA queue. *)

type queue = {
  ring : nativeint;  (** The ring's first byte. *)
  ring_bytes : int;  (** The ring's size in bytes, a power of two. *)
  read_ptr : nativeint;
      (** The 64-bit position up to which the engine has read the ring. *)
  write_ptr : nativeint;  (** The 64-bit position the engine reads up to. *)
  put : nativeint;
      (** The 64-bit position after the last packet written, where the next
          writer appends. *)
  doorbell : nativeint;  (** The 64-bit doorbell that wakes the engine. *)
}
(** The type for hardware queues. Positions count dwords on a PM4 queue, 64-byte
    packets on an AQL queue, and bytes on an SDMA queue, and grow without
    wrapping; a packet's place in the ring is its position modulo the ring's
    size. A writer, with the device taken:
    - waits until [read_ptr] leaves room for its packets;
    - writes them from [put] on. On an SDMA queue a packet never wraps: if the
      packets do not fit before the ring's end, the writer zeroes the rest of
      the ring and writes them from its start;
    - stores the new position into [write_ptr] and [put], then into [doorbell],
      minus one on an AQL queue. *)

type props = {
  target : int * int * int;
      (** The graphics target, such as [(11, 0, 0)] for ["gfx1100"]. *)
  gc : int * int * int;  (** The version of the graphics and compute block. *)
  sdma : int * int * int;  (** The version of the copy engine. *)
  nbio : int * int * int;  (** The version of the bus interface block. *)
  xccs : int;  (** The number of compute dies (XCCs). *)
  shader_engines : int;  (** The number of shader engines of one XCC. *)
  compute_units : int;  (** The number of compute units of one XCC. *)
  waves_per_cu : int;  (** The most waves a compute unit runs at once. *)
  lds_bytes : int;  (** The local data share of a work-group, in bytes. *)
  scratch_slots_per_cu : int;  (** The scratch wave slots of a compute unit. *)
}
(** The type for the GPU's properties that its work depends on. *)

type handles = {
  compute : queue;  (** The compute queue. *)
  aql : bool;
      (** [true] iff the compute queue takes AQL packets, which it does on GPUs
          of several XCCs; it takes PM4 packets otherwise. *)
  sdma : queue list;
      (** The SDMA copy queues: one, or one per GPU of the machine up to eight
          on a virtual function. {!Nx_device.Buffer.copy} uses the first. *)
  signal : nativeint;  (** The address of the timeline's signal word. *)
  props : props;  (** The GPU's properties. *)
}
(** The type for the queues and properties of a device. *)

val handles : Nx_device.t -> handles
(** [handles d] is the queues and properties of [d].

    Raises [Invalid_argument] if [d] is not an AMD device. *)

type kernel = {
  code : nativeint;  (** The address of the uploaded code object. *)
  descriptor : nativeint;  (** The address of the kernel descriptor. *)
  entry : nativeint;  (** The address of the kernel's first instruction. *)
  rsrc1 : int;  (** [COMPUTE_PGM_RSRC1] as dispatched. *)
  rsrc2 : int;  (** [COMPUTE_PGM_RSRC2] as dispatched, with the LDS size. *)
  rsrc3 : int;  (** [COMPUTE_PGM_RSRC3]. *)
  wave32 : bool;  (** [true] iff the kernel runs in waves of 32 lanes. *)
  private_segment : int;  (** Scratch bytes per lane. *)
  group_segment : int;  (** LDS bytes per work-group. *)
  kernarg_segment : int;  (** Argument bytes. *)
  dispatch_ptr : bool;  (** [true] iff it reads a dispatch packet pointer. *)
  private_segment_buffer : bool;
      (** [true] iff it reads a scratch buffer descriptor. *)
}
(** The type for the dispatch parameters of a kernel. *)

val kernel : Nx_device.Program.t -> kernel
(** [kernel p] is the dispatch parameters of [p]. Its
    {!Nx_device.Program.handle} is its [descriptor].

    Raises [Invalid_argument] if [p] is not loaded on an AMD device. *)

type scratch = {
  address : nativeint;  (** The scratch memory's first byte. *)
  bytes : int;  (** Its size, split evenly among the XCCs. *)
  tmpring_size : int;  (** The [COMPUTE_TMPRING_SIZE] value for it. *)
}
(** The type for the scratch memory of a device's kernels. *)

val scratch : Nx_device.t -> int -> scratch
(** [scratch d n] is [d]'s scratch memory for kernels of up to [n] scratch bytes
    per lane, grown if smaller, and on an AQL queue written into the queue's
    descriptor. Memory it replaces returns to [d] once unreachable. Call it
    before {!Nx_device.submit}, not inside.

    Raises [Invalid_argument] if [d] is not an AMD device, and
    {!Nx_device.Out_of_memory} if [d] cannot allocate it. *)
