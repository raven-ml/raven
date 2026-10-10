(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs, driven once open.

    A device of this library is one AMD GPU that a {e path} opened: a library
    that reaches the GPU one way, such as through Linux's [amdgpu] driver, and
    gives this one the GPU's memory, queues and interrupts as a {!type-path}
    ({!make}). Whichever path opened it, the device runs work on two hardware
    queues, ["COMPUTE:0"] and ["COPY:0"], whose rings this library alone writes,
    and makes completion observable through its {e timeline word}, 64 bits of
    host memory: the word holds [v] once the work of every value up to [v]
    completed. Its caller numbers the work it hands over: the first submission
    is value [1], each next one the value after it, and value [v] runs after
    every value below it, whichever queue each ran on.

    A program opens a GPU through a path and hands the device to [rig], which
    submits its work through the device's C entries ({!section-this}):
    {[
    let d =
      Rig.open_
        (module Rig_amd)
        ~name:(Rig_amd_amdgpu.device_name 0)
        (fun () -> Rig_amd_amdgpu.open_ 0)
      |> Result.get_ok
    in
    let s = Rig.Submission.make ~reads:0 ~writes:0 d [||] in
    let run = Rig.Submission.Run.make () in
    let p = Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||] in
    Rig.Point.wait p
    ]}

    {b GPUs.} The device drives GPUs of GFX 9.4.2, 9.5.0, 11 and 12. Its compute
    queue reads PM4 packets on a GPU of one die and AQL packets on a GPU of
    several ({!Rig_amd_abi.Capability.compute}); its copy queue reads SDMA
    packets.

    {b Memory and caches.} Every submission's work on the compute queue starts
    by invalidating the GPU's caches, its L2 included, and the release of a
    value writes the L2 back: work reads what the host, the copy queue and other
    devices wrote before the values it runs after, and they read what it wrote
    once its value is reached.

    {b Faults.} The path reports a fault of a device's work, such as a page
    fault or a reset of the GPU, at a later {!sleep}. Work that runs long is no
    fault: only the path's report is. A path that bounds progress states the
    bound ([hang_ms]), and rig loses the device past it ({!Rig_edge.facts}). A
    function that calls the path answers its refusal of the arguments as its
    result ([None], [Error]) and raises {!Fault} for any other failure.
    {!signaled}, {!free} and {!stop} never raise it.

    {b Domains.} Every value may be called from any domain, at the same time as
    others, except the C entries and {!stop}. The C entries run one call at a
    time, in value order, which their caller ensures; {!sleep} may run beside
    them. {!stop} is called once, after every other call returned; after it only
    {!free}, {!unload} and {!signaled} are called.

    {b References.}
    - LLVM's
      {{:https://llvm.org/docs/AMDGPUUsage.html}User Guide for AMDGPU Backend}:
      code objects, kernel descriptors, the memory model of each GFX version.
    - The Linux kernel's amdgpu driver: the PM4 and SDMA packet headers
      ([nvd.h], [soc15d.h], [sdma_v*_0_pkt_open.h]), its fences ([gfx_v12_0.c],
      [sdma_v7_0.c]) and its flush of the host data path
      ([amdgpu_hdp_generic_flush]).
    - ROCm's runtime: [amd_queue.h] (the AQL queue's descriptor) and
      [amd_aql_queue.cpp]. *)

(** {1:driver Driver} *)

include Rig_edge.Driver

val capability : t -> Rig_amd_abi.Capability.t
(** [capability g] is the record of [g]'s [capability] fact ({!facts}). *)

(** {1:this This driver}

    {t
      | Fact | Value |
      |------|-------|
      | [arch] | The processor the GPU runs code objects of, as LLVM names it ({!Rig_amd_abi.Gpu.processor}), such as ["gfx1201"]. |
      | [budget] | The bytes of the GPU's own memory. |
      | [queues] | ["COMPUTE:0"] runs [Words], [Fill] and [Launch]; ["COPY:0"] runs [Words], [Fill] and [Copy]. Index [0] and [1] in C. |
      | [completion] | [Store]: the queue that releases a value writes the word. |
      | [waits] | [stores] and [hosts] iff the compute queue compares 64-bit words, never [objects]; [most] is [255]. |
      | [may_block] | [false]: the C entries write memory and call no system function. |
      | [maps_host] | [true] iff the path maps host memory ([map_host] of {!type-path}). |
      | [host_addresses] | [false]: the host does not address [Device] memory. |
      | [capability] | A {!Rig_amd_abi.Capability.t} under {!Rig_amd_abi.Capability.key}. |
      | [word] | Eight bytes of host memory. |
    }

    {b Waits.} The compute queue compares 64-bit words where this library
    knows, for the GPU's GC, the first firmware that does, and the GPU's ([mec]
    of {!type-path}) is that version or later. A submission's reserved ring
    space holds [255] waits.

    {b Capability.} The record holds the GPU, its clock, its compute queue's
    packets, the C functions a fill calls, its active work-group processors,
    and its trace buffers, made at the first [trace]. On an AQL queue its
    [scratch] grows the device's scratch memory, which the queue hands every
    kernel; on a PM4 queue, the dispatch of each launch names it.

    {b Word.} The word holds, as an unsigned 64-bit integer in the host's byte
    order, the last value [v] such that the work of every value up to [v]
    completed. The queue that releases [v] writes it whole, after its work's
    writes reached memory. The device's C state holds the word, and neither is
    ever freed: other devices may read the word after the device is gone.

    {b Memory.}
    - [Device] memory is GPU memory, which the host does not address.
    - [Pinned] memory is host memory that the GPU addresses, which the host
      reads as fast as other memory.
    - [Mapped] memory is GPU memory the host addresses through the GPU's memory
      BAR, write-combined: fast to write and slow to read. The host's writes
      reach the GPU's work submitted after them. {!alloc} answers [None] where
      the path has no such memory, or cannot flush the host's writes into it.

    {!alloc} answers [None] if the GPU or the host lacks the memory. {!alloc}
    and {!map_host} raise [Invalid_argument] for fewer than one byte. Freeing a
    mapping ends the mapping alone; the memory it maps stays. A mapping of
    another device's memory is freed before that memory.

    {!locate} answers the GPU address of a region's first byte, and that
    address again as the handle: the GPU names memory by address. The host
    address is [None] iff the region is [Device] memory or a {!map_peer} view
    of [Device] memory.

    {!peer} and {!map_peer} answer [true] and a view at the same address iff
    the path that opened both devices maps it and this GPU reaches the other's
    memory: over a link between them, or through a memory BAR as large as the
    other GPU's memory. {!map_peer} also answers [None] for [Mapped] memory of
    the other device when this one already flushes the host data path of seven
    other GPUs, the most it keeps. {!map_host} answers a region the GPU's work
    addresses at its {!locate} address, or [None] if the path maps no host
    memory or refuses the pages, such as read-only ones. The kernel driver's
    path maps a page once per GPU, for every device of the GPU the process
    opens: a region whose pages all lie within a region {!map_host} gave a
    device of the same GPU shares that mapping, which lasts until the last
    region over it is freed, also after those devices stopped. It refuses
    pages of which some, but not all, lie within such a region.

    {b Images.} {!image} answers [Place (n, lay)] for a code object, whose
    image takes [n] bytes ({!Rig_amd_abi.Code_object.size}): code runs from the
    GPU's own memory, so that its fetches never cross the bus. It answers
    [Error msg] if the binary is not a code object, with
    {!Rig_amd_abi.Code_object.of_string}'s message, if it is for another
    processor, as ["a code object for gfx90a; the GPU is gfx1201"], or if a
    kernel takes more local data share than the GPU has, as
    ["kernel reduce takes 98304 bytes of local data share; the GPU has 65536"].
    {!entry} answers as [code] the address of the descriptor of the kernel,
    the symbol [f ^ ".kd"], which a dispatch names. It also makes the
    kernel's launch, and grows the device's scratch memory to the kernel's
    private segment. On a GPU of one die the launch is the kernel's PM4
    dispatch ({!Rig_amd_abi.Pm4.dispatch}), which names the scratch; on a GPU
    of several, its AQL kernel dispatch packet ({!Rig_amd_abi.Aql.dispatch}),
    whose queue hands the kernel the scratch and the packet. It raises
    [Invalid_argument] for a kernel that reads an implicit argument that is a
    runtime's service ([Other]), or, on a GPU of one die, its dispatch
    packet, which a launch does not write; or whose scratch the GPU has no
    memory for. {!unload} frees the launches; the code region is rig's.

    {b Sleep and faults.} {!sleep} blocks on the path's interrupt, which every
    release raises, and lets other domains run while it waits. It may return
    early on an interrupt of an earlier release or of other work of the GPU. It
    raises {!Fault} with the path's report if the device's work met a fault.

    {b Hand-over.} [edge]'s room check and hand-over are [rig_amd_room] and
    [rig_amd_submit], which [rig_amd.h] declares; its commit does nothing. A
    part is:
    - words, whole packets of the queue's kind: PM4, or AQL in multiples of 16
      words, on ["COMPUTE:0"]; SDMA on ["COPY:0"];
    - a fill, a C function that places packets on the queue during the
      hand-over, as {!Rig_amd_abi.Capability} states. It places at most its
      ring units of words and takes at most its segment bytes of the device's
      argument segment;
    - on ["COMPUTE:0"], a launch. The hand-over writes the kernel's
      arguments into the argument segment: the launch's parameters, each
      ref's address added, then the implicit arguments the kernel reads
      ({!Rig_amd_abi.Code_object.hidden}), its grid of whole groups starting
      at work-item 0, over any parameter bytes they share. It places the
      kernel's dispatch after invalidating the caches above the L2, on a PM4
      queue, or as an AQL packet, which waits for the packets before it and
      acquires at system scope, so that a launch reads what the launches
      before it wrote. The kernel's groups take its group segment plus the
      launch's shared memory of LDS;
    - on ["COPY:0"], a copy between two GPU addresses, its handles, whose
      ranges do not overlap.

    Parts on one queue run in array order. A part's [after] indices, each below
    its own, order it after parts of the other queue; parts of the two queues
    that [after] does not order may run at once. A wait is [RIG_WORD]: the
    compute queue holds the submission back until the aligned 64-bit word at
    its address, which the device's work addresses, holds at least its value,
    as unsigned integers. [handles] is ignored: the GPU's work names its memory
    by address.

    The room check answers [RIG_NEVER] if the parts exceed what the device
    holds when it is idle: the compute ring, half the copy ring and half the
    argument segment (a submission's copy packets and segment bytes never
    wrap), or 512 parts. It also answers [RIG_NEVER] for a part the device does
    not run: a copy on ["COMPUTE:0"], words on an AQL queue that are not whole
    packets, a launch whose grid or group has an empty axis, whose groups
    have more work-items than its kernel's bound
    ({!Rig_amd_abi.Code_object.kernel}'s [max_threads]) or more shared memory
    than the GPU's local data share less the kernel's group segment, or, as
    an AQL packet, whose grid has [2{^32}] work-items or more along an axis,
    an [after] index not below its own part's.

    The queues read none of a submission's packets before all of them are
    placed, so its work never waits on the host. A submission of no parts
    releases its value alone. The hand-over answers [RIG_FAILED] if a fill
    failed, as ["a fill on COMPUTE:0 failed with 1"], or if it waits on more
    words than the device holds or where it cannot wait. The queues then run
    none of its parts, and the word still reaches its value once the earlier
    values completed. Every later hand-over answers the same failure and hands
    nothing over. Each value's hand-over writes its release and answers
    [RIG_COMMITTED].

    {b Stop.} {!stop} destroys the device's queues through the path: once none
    runs, it writes the last value the hand-over was given into the word. If
    the path could not destroy every queue, the word reaches that value only if
    the queues still run and complete their work. The path's stop gets the
    device's first fault: the one {!sleep} raised, else [fault]. A path that
    keeps the GPU's state for later opens, as {!Rig_amd_pci}'s does, then skips
    its wait for the queues to leave and leaves the GPU lost, for its next open
    to reset. A later {!stop} does nothing. *)

(** {1:paths Paths}

    For the libraries that open GPUs. A path reaches a GPU one way, reads what
    the device needs, and gives it all to {!make} as a {!type-path}. {!make}
    makes the device's queues, rings and timeline word through the path. *)

type 'm memory = {
  address : int;  (** The GPU address of its first byte. *)
  host : int option;
      (** The host address of its first byte, if the host addresses it. *)
  data : 'm;  (** The path's own data for it. *)
}
(** The type for memory a path gives a device. *)

type 'm path = {
  key : 'm Type.Id.t;
      (** The path's key: devices whose paths share [key] map each other's
          memory with [map_peer]. *)
  index : int;
      (** The GPU's number among the machine's AMD GPUs in bus order
          ({!is_gpu}). *)
  gpu : Rig_amd_abi.Gpu.t;  (** The GPU, as its formats depend on it. *)
  waves : int;  (** The most waves a compute unit runs at once. *)
  lds : int;  (** The local data share of a workgroup, in bytes. *)
  clock_hz : int;  (** The frequency of the GPU's clock, in hertz. *)
  mec : int;  (** The version of the firmware of its compute queues. *)
  wgps : int array array;
      (** The work-group processors that run work: [wgps.(e).(a)] has bit [w]
          set iff processor [w] of shader array [a] of shader engine [e] does,
          engines numbered across dies. A die whose processors the path cannot
          read has every processor of its arrays set. *)
  budget : int;  (** The bytes of the GPU's own memory. *)
  alloc : [ `Gpu | `Bar | `System ] -> int -> 'm memory option;
      (** [alloc k n] is [n] new bytes: [`Gpu], GPU memory the host does not
          address; [`Bar], GPU memory the host addresses through the GPU's
          memory BAR; [`System], host memory the GPU addresses, cached for the
          host and snooped by the GPU, which the kernel driver owns, so that no
          unmap of host memory takes it from the GPU. It is [None] if the memory
          of [k] is exhausted, or [`Bar] memory does not exist. *)
  map_host : (int -> int -> 'm memory option) option;
      (** [Some map] if the GPU maps host memory: [map a n] maps for the GPU the
          pages that hold the [n] bytes of host memory at [a], the result's
          [address] where the GPU reaches the byte at [a] and its [host]
          [Some a], or is [None] where the path refuses the pages, such as
          read-only ones. [None] if the GPU maps no host memory, as one taken
          without an IOMMU, whose pages would go back to the system at the
          process's death while the GPU still writes them. *)
  reaches : int -> bool;
      (** [reaches j] is [true] iff [map_peer] maps the GPU memory the path
          gives a device of GPU [j], numbered as [index]: this GPU, or one it
          reaches over a link or through a memory BAR as large as that GPU's
          memory. *)
  map_peer : 'm memory -> 'm memory option;
      (** [map_peer m] maps for the GPU the memory [m] that the path gave
          another of its devices, at the same address, or is [None] if the GPU
          does not reach that device's GPU's memory. Memory of a device of the
          same GPU is in its address space already: the view maps nothing, and
          [free] of it gives nothing back. *)
  free : 'm memory -> unit;
      (** [free m] gives back what [alloc], [map_host] or [map_peer] gave. *)
  queue :
    [ `Pm4 | `Aql | `Sdma ] ->
    ring:int ->
    bytes:int ->
    read:int ->
    write:int ->
    (int, string) result;
      (** [queue k ~ring ~bytes ~read ~write] makes a hardware queue that reads
          packets of [k] from the [bytes] bytes of [`System] memory at [ring],
          writes its read position to the 64-bit word at [read] and reads its
          write position from the word at [write], all GPU addresses. It is the
          host address of the queue's 64-bit doorbell, or [Error msg]. *)
  hdp : int option;
      (** The host address of the 32-bit register whose store flushes the GPU's
          host data path, after which host writes through the memory BAR are in
          the GPU's memory; [None] if the host does not reach it. *)
  interrupt : int;
      (** The context, not [0], that the interrupt of a value's release carries,
          which wakes [sleep]. *)
  hang_ms : int option;
      (** [Some n] if work whose word has not moved for [n] milliseconds hung,
          where nothing else bounds the GPU's work; [None] where the path's
          kernel driver does. It is the device's [hang_ms] fact
          ({!Rig_edge.facts}). *)
  sleep : ms:int -> unit;
      (** [sleep ~ms] returns once the GPU interrupts, or after [ms]
          milliseconds, and raises {!Fault} with the path's report of a fault of
          the GPU's work. *)
  stable_power : unit -> (unit, string) result;
      (** [stable_power ()] holds the GPU's clocks and shader engines steady, as
          tracing needs, for the life of the process. *)
  stop : fault:string option -> [ `Stopped | `Unknown ];
      (** [stop ~fault] destroys the queues [queue] made: [`Stopped] once none
          of them runs or none can write memory outside the GPU, [`Unknown] if
          one may. [fault] is the fault rig lost the device for, if any: the
          path's own report, another fault of a call, or a hang past
          [hang_ms] ({!val-stop}). A path that keeps
          the GPU's state for later opens records it as its own; one that keeps
          none ignores it. *)
}
(** The type for what a path gives a device. A path fills it with functions over
    its own state, which this library never names. Each function answers the
    GPU's refusal as its result and raises {!Fault} for any other failure of the
    GPU or of the path's kernel driver. Any domain may call them at the same
    time, except [stop], which the device calls once, after which it calls only
    [free]. A [sleep] that another domain began before [stop] may still be
    waiting when [stop] runs: it returns without reading the GPU once [stop]
    began. *)

val make : 'm path -> (t, string) result
(** [make p] is a device of the GPU [p] reaches: its queues, timeline word and
    argument segment, made through [p]. The result is [Error msg] if this
    library does not drive the GPU's family, or knows no registers for its GC,
    naming the GPU's processor; if [p] refuses a queue, with [p]'s message; if
    [p] lacks the memory they need; or if [p]'s [`System] memory has no host
    address, which a path's contract rules out. A failed [make] gives back what
    it took: it frees the memory [p] gave and calls [p.stop] if [p] made a
    queue.

    Raises [Invalid_argument] if [p.interrupt] is [0]. *)

val is_gpu : vendor:int -> class_:int -> bool
(** [is_gpu ~vendor ~class_] is [true] iff a PCI function of vendor [vendor] and
    24-bit class code [class_] is an AMD GPU: AMD's vendor [0x1002] and a
    display controller or a processing accelerator, base class [0x03] or [0x12].
    Every path numbers a machine's GPUs [0], [1], ... in bus order among such
    functions. *)

(**/**)

(* [renumber ~age g v] makes [v] the value after [g]'s last one: the timeline
   word and the last value given become [v - 1], and every slot word is as if
   last written [age] (defaults to [0]) values earlier, at [v - 1 - age]. [g] is
   idle: its word holds the last value it was given. Tests reach the values a
   long run reaches, such as those past 2^31 and 2^32, and the slots' refresh
   past 2^31 values of age, without making that many submissions.

   Raises [Invalid_argument] if [g] is not idle, if [v - 1] is below its last
   value, or if [age] is negative or above [v - 1]. *)
val renumber : ?age:int -> t -> int -> unit

(* [dispatch g k ~base ~lds args ~shared] is the bytes of the words a
   hand-over places for a launch of [k], of an image at [base], on a GPU [g]
   whose groups take at most [lds] bytes of LDS, given the hand-over's 8
   arguments [args], the address of its parameters and of the scratch, its
   threads per group and its groups along x, y and z, and its groups' [shared]
   bytes of shared memory: on a GPU of one die, its PM4 dispatch; on a GPU of
   several, its AQL kernel dispatch packet, which names no scratch. Tests
   compare it with Rig_amd_abi.Pm4.dispatch and Aql.dispatch on every GPU this
   library drives, which no machine has all of.

   Raises [Invalid_argument] as those do for [k], or if [args] does not hold 8
   arguments. *)
val dispatch :
  Rig_amd_abi.Gpu.t ->
  Rig_amd_abi.Code_object.kernel ->
  base:int ->
  lds:int ->
  int array ->
  shared:int ->
  string
