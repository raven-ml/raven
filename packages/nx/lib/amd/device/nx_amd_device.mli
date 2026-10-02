(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD devices.

    Opens AMD GPUs as {!Nx_device.t}s through one of two interfaces, which their
    names tell apart:
    - {!Kernel}, the compute interface of Linux's [amdgpu] kernel driver. The
      driver owns the GPU and shares it with other programs.
    - {!Pci}, the runtime's own driver: it takes the GPU's PCI function from its
      kernel driver, loads the GPU's firmware, boots its blocks, and manages its
      memory and page tables itself. The GPU is this process's until it exits.

    Through {!Kernel} the GPUs are named ["AMD"], ["AMD:1"], ["AMD:2"], ..., and
    through {!Pci} ["AMD-PCI"], ["AMD-PCI:1"], .... A process uses one
    interface, chosen by its first open. Both need Linux: elsewhere {!count} is
    [0] and {!get} says why.

    {b Other machines.} Given the host of another machine ([nx.remote.device]),
    {!count} and {!get} reach that machine's GPUs, over {!Pci} through the
    machine's server: they are named ["AMD@HOST:PORT"], ["AMD:1@HOST:PORT"],
    ..., their host memory is that machine's memory, every register access and
    ring write crosses the network, and they have no interrupts. Such a GPU
    copies directly only to GPUs of its machine. The machine needs what {!Pci}
    needs here, and the process none of it.

    {b Memory.} Buffers are GPU memory, which the host does not address, but for
    mapped memory. {!Nx_device.Buffer.copy} moves their bytes on the GPU's copy
    engine (SDMA): directly from and to host memory the GPU addresses, and
    through the host's staging memory from and to other host memory. Host memory
    the GPU addresses is registered with it and coherent for it:
    - {!Nx_device.Buffer.create}[ ~memory:Pinned] allocates it, uncached system
      memory, and it counts in the device's budget, which defaults to the GPU's
      memory size;
    - {!Nx_device.Buffer.borrow} maps host memory, whole pages of it. It counts
      in no budget;
    - the host's staging memory, 128 MiB, is mapped at the first copy that needs
      it and kept for the life of the process. Between two AMD GPUs that reach
      each other's memory, over a direct link or a large memory BAR, the
      source's copy engine writes the destination; otherwise the bytes go
      through the staging memory.

    {!Nx_device.Buffer.create}[ ~memory:Mapped] allocates GPU memory that the
    host also writes, through the GPU's memory BAR, write-combined. The host's
    writes reach the GPU's memory once the host data path (HDP) is flushed: the
    runtime flushes it before each copy of its own, and work submitted to the
    device flushes it before it starts (see {!section-low}). Without a BAR that
    covers the GPU's memory (Resizable BAR), or under the [amdgpu] driver when
    it does not give the process the HDP's flush register, mapped memory is
    pinned memory.

    {b Timestamps} count the GPU's global clock, 100 MHz
    ({!Nx_device.Driver.Device_clock}); the copy engine stamps it for the
    runtime's profiles.

    {b Programs} are functions of AMD GPU code objects, ELF objects compiled for
    the device's {!Nx_device.arch}: the function [name] is the kernel whose
    descriptor is the symbol [name ^ ".kd"]. A code object is uploaded once
    while a program of it or its {!kernel}'s code is reachable
    ({!Nx_device.Program.load}). A code object larger than the memory the host
    can map is refused, naming Resizable BAR, which enlarges it.

    {b Faults and hangs.} A fault the GPU reports, such as a page fault, loses
    the device ({!Nx_device.Lost}) with the driver's report when a wait finds
    it. Work that does not signal within the device's {!Nx_device.timeout}, 30
    seconds unless {!Nx_device.set_timeout} sets another, loses it too. Nothing
    recovers a lost device in the process. Under {!Pci}, the process stops the
    engines of a lost GPU at exit and takes its bus mastering away, so that it
    cannot reach the memory the process releases; the next process that opens
    the GPU finds it was not closed cleanly, and resets it before booting it.

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

    Raises [Invalid_argument] if [host] is no host, and {!Nx_device.Lost} with
    [host] if [host]'s machine cannot be reached. *)

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
    [msg] starts with the GPU's name, such as
    ["AMD:2: no GPU 2; there are 2 AMD GPUs"].

    Raises [Invalid_argument] if [i < 0] or if [host] is no host, and
    {!Nx_device.Lost} with [host] if [host]'s machine cannot be reached. *)

(** {1:low Low-level}

    For the libraries that submit work to an AMD device, inside
    {!Nx_device.submit}: they write packets into its queues. Every address below
    is the same for the host and for the GPU. The host's writes to GPU memory
    through the BAR, such as mapped memory and uploaded programs, are visible to
    work only after a flush of the host data path (HDP): a compute queue's work
    begins with one, in PM4 packets, and a library flushes it from the host with
    {!flush_hdp} before it submits work to a copy queue, which has no such
    packet.

    Work for the value [v] first waits on its queue until the low 32 bits of the
    signal word ({!Nx_device.signal_word}) equal [v - 1], and ends by writing
    [v] into it: all 64 bits in one write, or its low 32 bits and then, only
    when they are [0], its high 32 bits. The values thus complete in order
    across the queues, and a high word written late never takes the word back.
    The device's own copies follow the same rule on the SDMA queue.

    A queue compares 32-bit words, and a test that the low half is at least a
    value passes early once the value's low half wraps. So work waits on its
    queue for another device [d'] only when [d'] is a device of the same
    submission, and only for {!Nx_device.submitted}[ d'], the value of [d']'s
    work before the submission: only the submission's own work on [d'] signals
    past it, and it does so after the waiting work, so the wait is exact, a test
    that the low 32 bits of [d']'s signal word equal the value's. Any other pair
    of {!Nx_device.Submission.waits} is waited for on the host
    ({!Nx_device.Submission.wait}). *)

type t
(** The type for the queues and properties of a device. *)

val of_device : Nx_device.t -> t option
(** [of_device d] is the queues and properties of [d], if [d] is an AMD device.
*)

type queue = {
  ring : Nx_device.Buffer.t;
      (** The ring, whose size ({!Nx_device.Buffer.nbytes}) is a power of two.
      *)
  read_ptr : Nx_device.Buffer.t;
      (** The 64-bit position up to which the engine has read the ring. *)
  write_ptr : Nx_device.Buffer.t;
      (** The 64-bit position the engine reads up to. *)
  put : Nx_device.Buffer.t;
      (** The 64-bit position after the last packet written, where the next
          writer appends. *)
  doorbell : Nx_device.Buffer.t;
      (** The 64-bit doorbell that wakes the engine. *)
}
(** The type for hardware queues. Positions count dwords on a PM4 queue, 64-byte
    packets on an AQL queue, and bytes on an SDMA queue, and grow without
    wrapping; a packet's place in the ring is its position modulo the ring's
    size. A writer, with the device taken:
    - waits until [read_ptr] leaves room for its packets. Inside
      {!Nx_device.submit}, each queue has room for half its ring, which [submit]
      waits for: work submitted there writes at most half a ring into each queue
      and need not wait;
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

val compute : t -> queue
(** [compute a] is the compute queue. *)

val aql : t -> bool
(** [aql a] is [true] iff the compute queue takes AQL packets, which it does on
    GPUs of several XCCs; it takes PM4 packets otherwise. *)

val sdma : t -> queue list
(** [sdma a] is the SDMA copy queues: one, or one per GPU of the machine up to
    eight on a virtual function. {!Nx_device.Buffer.copy} uses the first. *)

val props : t -> props
(** [props a] is the GPU's properties. *)

val flush_hdp : t -> unit
(** [flush_hdp a] flushes the GPU's host data path (HDP) from the host, after a
    full fence, so that the host's earlier writes through the BAR reach the
    GPU's memory. Under the [amdgpu] driver without the HDP's flush register,
    the device has no mapped memory and [flush_hdp] does nothing. *)

type kernel = {
  code : Nx_device.Buffer.t;
      (** The uploaded code object: its image, as the ELF object lays it out
          ({!Nx_device_elf.load}), relocated, in memory of the device the host
          writes, which keeps the code object loaded while it is reachable
          ({!Nx_device.Program.code}). *)
  descriptor : nativeint;  (** The address of the kernel descriptor. *)
  private_segment : int;
      (** Scratch bytes per lane, which {!scratch} must provide. *)
}
(** The type for kernels: a kernel descriptor of an uploaded code object. A
    dispatch reads the rest of what it needs, such as its resource words, from
    the descriptor itself. *)

val kernel : Nx_device.Program.t -> kernel option
(** [kernel p] is the kernel of [p], if [p] is loaded on an AMD device. Its
    {!Nx_device.Program.handle} is its [descriptor]. *)

val scratch : t -> int -> Nx_device.Buffer.t
(** [scratch a n] is the device's scratch memory for kernels of up to [n]
    scratch bytes per lane, split evenly among the XCCs: the memory of the last
    call, if large enough, or new memory, which on an AQL queue is written into
    the queue's descriptor. Memory it replaces returns to the device once
    unreachable. Call it before {!Nx_device.submit}, not inside.

    Raises {!Nx_device.Out_of_memory} if the device cannot allocate it. *)

(** {2:profiling Profiling}

    A profile that asks for counters or traces ({!Nx_device.Profile.start}) has
    each run of a kernel on the compute queue count the counters, and each
    shader engine trace its threads, into a slot of the device's profiling. *)

type counter = {
  name : string;  (** The counter's name, such as ["SQ_BUSY_CYCLES"]. *)
  block : string;
      (** The hardware block that counts it: ["GRBM"], ["GL2C"], ["TCC"] or
          ["SQ"]. *)
  event : int;  (** The event its block's counter selects. *)
  register : int;
      (** Which of its block's counter registers counts it: the counters of a
          block take its registers in order. *)
  instances : int;  (** The instances of its block in one XCC. *)
  engines : int;  (** The shader engines it is counted in. *)
  arrays : int;  (** The shader arrays of an engine it is counted in. *)
  wgps : int;  (** The work-group processors of an array it is counted in. *)
  offset : int;  (** The byte offset of its values in a run's samples. *)
}
(** The type for a counter as a kernel's run counts it. Its values are 64-bit
    words, one per XCC, instance, engine, array and work-group processor, the
    last varying fastest. *)

type counting = {
  samples : Nx_device.Buffer.t;
      (** [slots] runs of [size] bytes, the values of [counters]. *)
  counters : counter list;  (** The counters, in the profile's order. *)
  size : int;  (** The bytes of a run's samples. *)
  wgp_active : engine:int -> array:int -> wgp:int -> bool;
      (** Whether a work-group processor is active. A run does not count an
          inactive one, whose values stay [0]. *)
}
(** The type for the counting of a device's runs. Each submission of a compute
    queue resets the counters and selects them at its start, and a run writes
    the values of [counters] into the [slot]th [size] bytes of [samples] once it
    completes. *)

type tracing = {
  traces : Nx_device.Buffer.t;
      (** The traces, in mapped memory: [slots] windows of [window] bytes for
          each shader engine, the engine's windows first. *)
  ends : Nx_device.Buffer.t;
      (** [slots * engines] [Int32]: where each run's trace of each engine ends,
          the run's engines first, as the engine's write pointer reads after the
          run. *)
  window : int;  (** The bytes of a run's trace of one engine. *)
  engines : int;  (** The shader engines of all the XCCs. *)
}
(** The type for the tracing of a device's runs. Each run traces the waves of
    every shader engine into the engine's window of its slot, and the
    instructions of shader engines 0 and 1. A trace that fills its window is cut
    short there. *)

type profiling = {
  slots : int;  (** The runs it keeps until a synchronization reads them. *)
  log : Nx_device.Buffer.t;
      (** [1 + 3 * slots] [UInt64]: the runs taken so far, then for each slot
          the kernel descriptor address of its run, and when the run started and
          when it stopped, on the GPU's clock. *)
  counting : counting option;  (** The counting, if the profile counts. *)
  tracing : tracing option;  (** The tracing, if the profile traces. *)
}
(** The type for the profiling of a device's runs. A run of a kernel takes the
    slot [(r + k) mod slots], where [r] is the first word of [log] when the
    submission's host program runs and [k] the runs the submission took before
    it: the host program writes the kernel's descriptor address into the word
    [1 + 3 * slot] of [log], and adds the submission's runs to the first word
    once it wrote the command buffer. The queue writes the time into the next
    word before the kernel and into the one after once the kernel completed. The
    device reads the runs at each synchronization while the profile is taken,
    and when it stops: their {!Nx_device.Profile.Counters} and the
    {!Nx_device.Profile.Trace} of each shader engine, named after the kernel's
    function and timed by the run; on GFX11 and GFX12 the spans of each trace's
    waves ({!Thread_trace}), on a lane of their engine, compute unit, SIMD and
    slot; and as {!Nx_device.Profile.Overwritten} the runs taken over before it
    read them. *)

val profiling : t -> profiling option
(** [profiling a] is the profiling of the counters and traces of the profile
    being taken ({!Nx_device.Profile.counters}, {!Nx_device.Profile.traced}), or
    [None] if it asks for neither. The device keeps the profiling of each
    request it was asked for, for its life: work encoded for a request writes
    that request's buffers whenever it runs, and the device reads them all.

    Raises [Invalid_argument] if the GPU does not count a counter, naming it and
    the counters it has, and [Failure] if, under {!Kernel}, a GPU other than a
    GFX9 one is not in its stable power state, which counts and traces need: the
    message says to run [amd-smi set -l stable_std]. *)

(** {2:traces Thread traces}

    A thread trace is a stream of packets a shader engine writes while it runs
    waves: when each wave starts and ends, the instructions it issues, and, from
    time to time, markers of the GPU's realtime clock. Its times count the
    shader engine's cycles from the start of the trace. *)

module Thread_trace : sig
  type wave = {
    cu : int;
        (** The compute unit, numbered within its shader engine; on GFX11 and
            GFX12, its work-group processor and shader array. *)
    simd : int;  (** The SIMD of the compute unit. *)
    slot : int;  (** The SIMD's wave slot. *)
    start : int;  (** When the wave started, in shader cycles. *)
    stop : int;  (** When it ended, in shader cycles. *)
  }
  (** The type for a wave of a trace. *)

  val waves : string -> wave list
  (** [waves trace] is the waves that start and end in [trace], in the order
      they end. A wave's start is paired with the next end of the same compute
      unit, SIMD and slot. *)

  val clock : string -> (int -> int) option
  (** [clock trace] maps a shader time of [trace] to the GPU's 100 MHz realtime
      clock, which its timestamps count, through the trace's realtime markers:
      on the line through the two markers around it, or through the first two or
      last two outside them. It is [None] for a trace of fewer than two markers
      of distinct shader times, such as a GFX9 GPU's, whose traces have none. *)
end
