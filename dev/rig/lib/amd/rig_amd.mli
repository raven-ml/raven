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
    submits its work through the device's C entries ({!room_entry},
    {!submit_entry}):
    {[
    let d =
      Rig.open_
        (module Rig_amd)
        ~name:(Rig_amd_amdgpu.device_name 0)
        (fun () -> Rig_amd_amdgpu.open_ 0)
      |> Result.get_ok
    in
    let s = Rig.Submission.make ~reads:0 ~writes:0 ~waits:0 d [||] in
    Rig.wait d (Rig.Point.value (Rig.submit s))
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
    fault: only the path's report is, or a bound the path states ([hang_ms]). A
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

(** {1:facts Facts}

    Fixed when the device is made. *)

type t
(** The type for open AMD GPUs: the path that opened one, its queues and its
    timeline word. *)

val key : t Type.Id.t
(** [key] tells devices of this library apart from others'. *)

val arch : t -> string
(** [arch g] is the processor the GPU runs code objects of, as LLVM names it
    ({!Rig_amd_abi.Gpu.processor}), such as ["gfx1201"]. *)

val budget : t -> int
(** [budget g] is the bytes of the GPU's own memory. *)

val queues : t -> string list
(** [queues g] is [["COMPUTE:0"; "COPY:0"]]: the compute queue and the copy
    queue, index [0] and [1] in C. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion g] is [`Store]: [g]'s queues write its timeline word. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on g c] is [true] iff [c] is [`Store] or [`Host] and [g]'s compute
    queue compares 64-bit words: this library knows, for the GPU's GC, the first
    firmware that does, and the GPU's ([mec] of {!type-path}) is that version or
    later. *)

val max_waits : t -> int
(** [max_waits g] is [255]: a submission's reserved ring space holds that many
    waits. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks g] is [`Returns]: the C entries write memory and call no system
    function. *)

val maps_host : t -> bool
(** [maps_host g] is [true] iff [g]'s path maps host memory ([map_host] of
    {!type-path}). *)

type capability = Rig_amd_abi.Capability.t
(** The type for what compiled code needs from a device. *)

val capability : t -> capability
(** [capability g] is the GPU, its clock, its compute queue's packets, the C
    functions a fill calls, its active work-group processors, and its trace
    buffers, made at the first [trace]. On an AQL queue its [scratch] grows the
    device's scratch memory. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Rig_amd_abi.Capability.key}. *)

(** {1:memory Memory} *)

type region
(** The type for memory a device's work addresses: an allocation of the device,
    host memory it maps, or another device's memory it maps. *)

val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
(** [alloc g kind n] is [Some r] with [r] [n] new bytes:
    - [`Device], GPU memory, which the host does not address;
    - [`Pinned], host memory that the GPU addresses, which the host reads as
      fast as other memory;
    - [`Mapped], GPU memory the host addresses through the GPU's memory BAR,
      write-combined: fast to write and slow to read. The host's writes reach
      the GPU's work submitted after them. Where the path has no such memory or
      cannot flush the host's writes into it, it is host memory, as [`Pinned].

    It is [None] if the GPU or the host lacks the memory.

    Raises [Invalid_argument] if [n < 1]. *)

val free : t -> region -> unit
(** [free g r] gives back [r], which {!alloc}, {!map_peer} or {!map_host} gave.
    Freeing a mapping ends the mapping alone; the memory it maps stays. The
    caller frees [r] once no work that uses it runs, and frees a mapping of
    another device's memory before that memory.

    Raises [Invalid_argument] if [r] is no such region of [g] nor its {!word},
    or was freed. *)

val address : region -> int option
(** [address r] is [Some a], [a] the GPU address of [r]'s first byte. *)

val handle : region -> nativeint
(** [handle r] is {!address}[ r]: the GPU names memory by address. *)

val host : region -> int option
(** [host r] is [Some a], [a] the host address of [r]'s first byte. It is [None]
    iff [r] is [`Device] memory or a {!map_peer} view of [`Device] memory. *)

val peer : t -> t -> bool
(** [peer g g'] is [true] iff {!map_peer}[ g g'] maps [`Device] memory of [g']:
    the path that opened both reaches [g']'s GPU from [g]'s. It is [false] for
    [g] itself. *)

val map_peer : t -> t -> region -> region option
(** [map_peer g g' r] is [Some r'] with [r'] a new region of [g] over the memory
    of [g']'s region [r], at the same address, if the path that opened both maps
    it and [g]'s GPU reaches [g']'s memory: over a link between them, or through
    a memory BAR as large as [g']'s memory. It is [None] otherwise, for devices
    two paths opened, and for [`Mapped] memory of [g'] when [g] already flushes
    the host data path of seven other GPUs, the most it keeps.

    Raises [Invalid_argument] if [g'] is [g], or if [r] is no region of [g'] or
    was freed. *)

val map_host : t -> int -> int -> region option
(** [map_host g a n] is [Some r] with [r] the [n] bytes of host memory at [a],
    which [g]'s work addresses at {!address}[ r]. The path maps the pages that
    hold them, which must stay mapped in the process until [r] is freed. It is
    [None] if the path maps no host memory ({!maps_host}) or refuses these
    pages, such as read-only ones. The kernel driver's path also refuses pages
    that a region {!map_host} gave a device of the same GPU maps, as it maps a
    page at most once per GPU.

    Raises [Invalid_argument] if [n < 1]. *)

(** {1:images Images} *)

type image
(** The type for code objects loaded on a device. *)

val image :
  t ->
  string ->
  ( [ `Loaded of image | `Place of int * (region -> image * string) ],
    string )
  result
(** [image g bin] is [Ok (`Place (n, lay))] for the code object [bin], whose
    image takes [n] bytes ({!Rig_amd_abi.Code_object.size}). The caller
    allocates them as [`Device] memory [r] of [g], at least [n] bytes: [lay r]
    is the code object laid over [r] and the image's bytes, which the caller
    copies to [r]'s start before work runs a kernel of it. Code runs from the
    GPU's own memory, so that its fetches never cross the bus. [image] makes
    nothing on the GPU, and [lay] calls nothing and raises nothing. It is never
    [`Loaded]: the device's library places no code itself.

    The result is [Error msg] if [bin] is not a code object, with
    {!Rig_amd_abi.Code_object.of_string}'s message, if it is for another
    processor, as ["a code object for gfx90a; the GPU is gfx1201"], or if a
    kernel takes more local data share than the GPU has, as
    ["kernel reduce takes 98304 bytes of local data share; the GPU has 65536"].
*)

val entry : image -> string -> int option
(** [entry m f] is [Some a], [a] the address of the descriptor of [m]'s kernel
    [f], the symbol [f ^ ".kd"], which a dispatch names. It is [None] if [m] has
    no kernel [f].

    Raises [Invalid_argument] if [m] was unloaded. *)

val unload : t -> image -> unit
(** [unload g m] ends [m]: its kernels' addresses are no longer valid. The
    caller unloads it once no work that runs its kernels runs, and frees the
    region it was laid over after.

    Raises [Invalid_argument] if [m] is another device's or was unloaded. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word g] is [g]'s timeline word: eight bytes of host memory holding, as an
    unsigned 64-bit integer in the host's byte order, the last value [v] such
    that the work of every value up to [v] completed. The queue that releases
    [v] writes it whole, after its work's writes reached memory; it never
    decreases. Other devices may map it and wait on it. It lives until
    {!free}, which the caller calls once [g] is stopped, the word holds its
    last value, and no other device's work reads it. *)

val signaled : t -> int
(** [signaled g] is the value in {!word}, read with acquire order: the work of
    every value up to it completed, and its writes are visible to the reader. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep g ~seen ~still_ms] returns once [g]'s timeline word differs from
    [seen], at once if it does already, and at the latest after [still_ms]
    milliseconds. [still_ms >= 0]. It may return earlier with the word still
    [seen]: on an interrupt of an earlier release, or of other work of the GPU,
    or when the hang bound's clock restarts. A caller reads the word again after
    every return. It blocks on the path's interrupt, which every release raises,
    and lets other domains run while it waits.

    Raises {!Fault} with the path's report if [g]'s work met a fault. Where the
    path bounds progress ([hang_ms] is [Some n]), it also raises {!Fault} once
    work is outstanding and the word has not moved for [n] milliseconds. That
    clock runs only while the last value given is above the word: it starts at
    the later of the word's last move and the first [sleep] after the device was
    idle, as [sleep] observes them, so an idle device never hangs and the report
    may come late but never early. Once [sleep] raised {!Fault}, every later
    call on [g] raises it again. *)

(** {1:work Work}

    Work reaches a device in C, through {!room_entry} and {!submit_entry}, over
    [rig_edge.h]'s structures. A part names its queue by its index in {!queues}.
    A part is:
    - words, whole packets of the queue's kind: PM4, or AQL in multiples of 16
      words, on ["COMPUTE:0"]; SDMA on ["COPY:0"];
    - a fill, a C function that places packets on the queue during the submit,
      as {!Rig_amd_abi.Capability} states. It places at most its ring units of
      words and takes at most its segment bytes of the device's argument
      segment;
    - on ["COPY:0"], a copy between two GPU addresses, its handles, whose ranges
      do not overlap.

    Parts on one queue run in array order. A part's [after] indices, each below
    its own, order it after parts of the other queue; parts of the two queues
    that [after] does not order may run at once. A wait is [RIG_WORD]: the
    compute queue holds the submission back until the aligned 64-bit word at its
    address, which the device's work addresses, holds at least its value, as
    unsigned integers. The device waits only where {!waits_on} says so, and on
    at most {!max_waits} words per submission. [handles] is ignored: the GPU's
    work names its memory by address.

    The room check answers [RIG_NEVER] if the parts exceed what the device holds
    when it is idle: the compute ring, half the copy ring and half the argument
    segment (a submission's copy packets and segment bytes never wrap), or 512
    parts. It also answers [RIG_NEVER] for a part the device does not run: a
    copy on ["COMPUTE:0"], words on an AQL queue that are not whole packets, an
    [after] index not below its own part's.

    The submission of [v] runs after every earlier value and after its waits;
    once it completed, the timeline word holds [v]. The queues read none of a
    submission's packets before all of them are placed, so its work never
    waits on the host. A submission of no parts releases [v] alone. The submit
    answers [RIG_FAILED] if a fill failed, as
    ["a fill on COMPUTE:0 failed with 1"], or if it waits on more words than the
    device holds or where it cannot wait. The queues then run none of its parts,
    and the word still reaches [v] once the earlier values completed. Every
    later submit answers the same failure and hands nothing over. *)

val room_entry : nativeint
(** [room_entry] is the address of the C function [rig_amd_room], which
    [rig_amd.h] declares. *)

val submit_entry : nativeint
(** [submit_entry] is the address of the C function [rig_amd_submit], which
    [rig_amd.h] declares. *)

val commit_entry : nativeint
(** [commit_entry] is the address of the device's commit, in the shape
    [rig_commit_fn] of [rig_edge.h]. Each value's hand-over writes its release
    and answers [RIG_COMMITTED]: [commit_entry] does nothing. *)

val self : t -> nativeint
(** [self g] is the address of [g]'s state, the first argument of [rig_amd_room]
    and [rig_amd_submit]. It is valid while the process runs: a device's C state
    holds its {!word}, which other devices may read after [g] is gone, so
    neither is ever freed. *)

(** {1:loss Loss} *)

exception Fault of string
(** [Fault why] reports a fault of a device's work: the path's report, or the
    end of the progress bound ({!val-sleep}). *)

val stop : t -> unit
(** [stop g] stops [g] for good, never waiting. It destroys [g]'s queues: once
    none runs, it writes the last value the submit entry was given into the
    timeline word, with release order, so work of other devices that waits on it
    runs on. If the path could not destroy every queue, the word reaches that
    value only if the queues still run and complete their work. A later [stop]
    does nothing. *)

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
      (** [Some n] if work whose word has not moved for [n] milliseconds is a
          fault ({!val-sleep}), where nothing else bounds the GPU's work; [None]
          where the path's kernel driver does. [n] is positive. *)
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
          one may. [fault] is the device's first fault, if any: the path's own
          report or the end of the progress bound, a hang ({!val-sleep}). A
          path that keeps the GPU's state for later opens records it as its
          own; one that keeps none ignores it. *)
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

    Raises [Invalid_argument] if [p.interrupt] is [0] or [p.hang_ms] is [Some n]
    with [n < 1]. *)

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
