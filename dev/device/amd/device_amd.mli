(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs, driven once open.

    A device of this library is one AMD GPU that a {e path} opened: a library
    that reaches the GPU one way, such as through Linux's [amdgpu] driver, and
    gives this one the GPU's memory, queues and interrupts as a {!path}
    ({!make}). Whichever path opened it, the device runs work on two hardware
    queues, ["COMPUTE:0"] and ["COPY:0"], whose rings this library alone
    writes, and makes completion observable through its {e timeline word}, 64
    bits of host memory: the word holds [v] once the work of every value up to
    [v] completed. Its caller numbers the work it hands over: the first
    submission is value [1], each next one the value after it, and value [v]
    runs after every value below it, whichever queue each ran on.

    A program that uses a GPU alone opens it through a path, allocates and
    submits, then waits for the word:
    {[
    let g = Result.get_ok (Device_amd_amdgpu.open_ 0) in
    let src = Option.get (Device_amd.alloc g `Pinned 4096) in
    let dst = Option.get (Device_amd.alloc g `Device 4096) in
    let copy = `Copy ((dst, 0), (src, 0), 4096) in
    let p = Device_amd.part g ~queue:"COPY:0" copy in
    match Device_amd.submit g ~v:1 ~waits:[||] ~handles:[||] [| p |] with
    | `Ok ->
        let rec wait () =
          let seen = Device_amd.signaled g in
          if seen < 1 then (
            Device_amd.sleep g ~seen ~still_ms:200;
            wait ())
        in
        wait ()
    | `Failed why -> prerr_endline why
    ]}

    {b GPUs.} The device drives GPUs of GFX 9.4.2, 9.5.0, 11 and 12. Its compute
    queue reads PM4 packets on a GPU of one die and AQL packets on a GPU of
    several ({!Device_amd_abi.Capability.compute}); its copy queue reads SDMA
    packets.

    {b Memory and caches.} Every submission's work on the compute queue starts
    by invalidating the GPU's caches, its L2 included, and the release of a
    value writes the L2 back: work reads what the host, the copy queue and
    other devices wrote before the values it runs after, and they read what it
    wrote once its value is reached.

    {b Faults.} The path reports a fault of a device's work, such as a page
    fault or a reset of the GPU, at a later {!sleep}. Work that runs long is no
    fault: only the path's report is. A function that calls the path answers
    its refusal of the arguments as its result ([None], [Error]) and raises
    {!Fault} for any other failure. {!signaled}, {!free}, {!unmap} and {!stop}
    never raise it.

    {b Domains.} Every value may be called from any domain, at the same time as
    others, with three exceptions. {!room} and {!submit} run one call at a time,
    in value order: their caller serialises them. {!stop} is called once, after
    every other call returned; after it only {!free} and {!unmap} are called.
    {!sleep} may run while another domain submits.

    {b References.}
    - LLVM's
      {{:https://llvm.org/docs/AMDGPUUsage.html}User Guide for AMDGPU Backend}:
      code objects, kernel descriptors, the memory model of each GFX version.
    - The Linux kernel's amdgpu driver: the PM4 and SDMA packet headers
      ([nvd.h], [soc15d.h], [sdma_v*_0_pkt_open.h]), its fences
      ([gfx_v12_0.c], [sdma_v7_0.c]) and its flush of the host data path
      ([amdgpu_hdp_generic_flush]).
    - ROCm's runtime: [amd_queue.h] (the AQL queue's descriptor) and
      [amd_aql_queue.cpp]. *)

(** {1:facts Facts} *)

type t
(** The type for open AMD GPUs: the path that opened one, its queues and its
    timeline word. *)

val key : t Type.Id.t
(** [key] tells devices of this library apart from others'. *)

val arch : t -> string
(** [arch g] is the processor the GPU runs code objects of, as LLVM names it
    ({!Device_amd_abi.Gpu.processor}), such as ["gfx1201"]. *)

val machine : t -> string option
(** [machine g] is [None]: [g]'s GPU is on this machine. *)

val budget : t -> int
(** [budget g] is the bytes of the GPU's own memory. *)

val queues : t -> string list
(** [queues g] is [["COMPUTE:0"; "COPY:0"]]: the compute queue and the copy
    queue, index [0] and [1] in C. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion g] is [`Store]: [g]'s queues write its timeline word. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on g c] is [true] for [`Store] and [`Host] if [g]'s compute queue
    compares 64-bit words, which the firmware of a GPU's compute queues does
    from a version this library knows for the GPU's GC (the [mec] of its
    {!path}); it is [false] otherwise, and for [`Object]. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks g] is [`Returns]: {!submit} writes memory and calls no system
    function. *)

type capability = Device_amd_abi.Capability.t
(** The type for what compiled code needs from a device. *)

val capability : t -> capability
(** [capability g] is the GPU, its clock, its compute queue's packets, and the C
    functions a fill calls. On an AQL queue its [scratch] grows the device's
    scratch memory. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Device_amd_abi.Capability.key}. *)

val self : t -> nativeint
(** [self g] is the address of [g]'s state, the first argument of
    [device_amd_room] and [device_amd_submit]. It is valid while the process
    runs: a device's C state holds its {!word}, which other devices may read
    after [g] is gone, so neither is ever freed. *)

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
      the GPU's work submitted after them. Where the path has no such memory
      or cannot flush the host's writes into it, it is host memory, as
      [`Pinned].

    It is [None] if the GPU or the host has not the memory.

    Raises [Invalid_argument] if [n < 1]. *)

val free : t -> region -> unit
(** [free g r] frees the allocation [r]. The caller frees it once no work that
    uses it runs.

    Raises [Invalid_argument] if [r] is no allocation of [g], or was freed. *)

val address : region -> int option
(** [address r] is [Some a], [a] the GPU address of [r]'s first byte. *)

val handle : region -> nativeint
(** [handle r] is {!address}[ r]: the GPU names memory by address. *)

val host : region -> nativeint option
(** [host r] is [Some a], [a] the host address of [r]'s first byte, unless [r]
    is [`Device] memory or another GPU's. *)

val map_peer : t -> t -> region -> region option
(** [map_peer g g' r] is [Some r'] with [r'] a new region of [g] over the memory
    of [g']'s region [r], at the same address, if the path that opened both
    maps it and [g]'s GPU reaches [g']'s memory: over a link between them, or
    through a memory BAR as large as [g']'s memory. It is [None] otherwise,
    and for devices two paths opened. {!unmap} of [r'] ends only [r'], and
    {!free} refuses it.

    Raises [Invalid_argument] if [g'] is [g], or if [r] is no region of [g'] or
    was freed or unmapped. *)

val map_host : t -> nativeint -> int -> region option
(** [map_host g a n] is [Some r] with [r] the [n] bytes of host memory at [a],
    which [g]'s work addresses at [a]. The path maps the pages that hold them,
    which must stay mapped in the process until [r] is unmapped. It is [None]
    if the path refuses them, such as read-only pages, or pages that a region
    {!map_host} gave a device of the same GPU maps: a GPU maps a page at most
    once.

    Raises [Invalid_argument] if [n < 1]. *)

val unmap : t -> region -> unit
(** [unmap g r] ends the region [r] that {!map_peer} or {!map_host} gave. The
    caller unmaps it once no work that uses it runs.

    Raises [Invalid_argument] if [r] is an allocation, a region of another
    device, or was unmapped. *)

(** {1:images Images} *)

type image
(** The type for code objects loaded on a device. *)

val image : t -> string -> (image * (region * string) option, string) result
(** [image g bin] is [Ok (m, Some (r, b))] with [m] the code object [bin], [r]
    new [`Device] memory for its image and [b] the image's bytes
    ({!Device_amd_abi.Code_object}), which the caller copies into [r] before
    work runs a kernel of [m]. Code runs from the GPU's own memory, so that its
    fetches never cross the bus.

    The result is [Error msg] if [bin] is not a code object, with
    {!Device_amd_abi.Code_object.of_string}'s message, if it is for another
    processor, as ["a code object for gfx90a; the GPU is gfx1201"], if a kernel
    takes more local data share than the GPU has, as ["kernel reduce takes
    98304 bytes of local data share; the GPU has 65536"], or if the GPU has not
    the memory for its image. *)

val entry : image -> string -> int option
(** [entry m f] is [Some a], [a] the address of the descriptor of [m]'s kernel
    [f], the symbol [f ^ ".kd"], which a dispatch names. It is [None] if [m]
    has no kernel [f].

    Raises [Invalid_argument] if [m] was unloaded. *)

val unload : t -> image -> unit
(** [unload g m] frees [m]'s image. The caller unloads it once no work that
    runs its kernels runs.

    Raises [Invalid_argument] if [m] is another device's or was unloaded. *)

(** {1:work Work} *)

type part
(** The type for work for one queue of a device. *)

val part :
  t ->
  queue:string ->
  ?after:int array ->
  [ `Words of int array
  | `Fill of nativeint * nativeint * int * int
  | `Copy of (region * int) * (region * int) * int ] ->
  part
(** [part g ~queue ~after w] is the work [w] for [g]'s queue [queue], one of
    {!queues}. [after] (defaults to [[||]]) holds the indices, in the array
    given to {!submit}, of the parts of its submission that it runs after, each
    smaller than its own. Parts on one queue run in array order; parts on two
    queues that [after] does not order may run at once. [w] is:
    - [`Words ws], whole packets of the queue's kind, each integer's low 32
      bits a word: PM4 or AQL ([Array.length ws] a multiple of 16) on
      ["COMPUTE:0"], SDMA on ["COPY:0"];
    - [`Fill (f, arg, units, bytes)], a C function [f] that places packets on
      the queue, called during {!submit} with [arg] as
      {!Device_amd_abi.Capability} states. It places at most [units] words and
      takes at most [bytes] bytes of the device's argument segment;
    - [`Copy ((dst, o), (src, o'), n)], on ["COPY:0"], a copy of the [n] bytes
      of [src] at offset [o'] to [dst] at offset [o], any two regions of [g],
      whose ranges do not overlap.

    Raises [Invalid_argument] if [queue] is not a queue of [g], if words on an
    AQL queue are not a multiple of 16, if [units] or [bytes] is negative, if a
    copy is on ["COMPUTE:0"] or its range lies outside its region, if a region
    is of another device or was freed or unmapped, or if an index of [after] is
    negative. *)

val room : t -> part array -> [ `Fits | `Later | `Never ]
(** [room g ps] is [`Fits] if [ps] fit [g]'s rings and argument segment now,
    [`Later] if they fit once work [g] was given completes, as it stands when
    [room] reads the timeline word, and [`Never] if they exceed a ring when
    it is empty (half the copy ring, whose packets never wrap), the argument
    segment, or 512 parts. Its C form, [device_amd_room], also answers
    [NX_NEVER] for a part {!part} refuses. *)

val submit :
  t ->
  v:int ->
  waits:([ `Word | `Equal | `Object ] * int * int) array ->
  handles:nativeint array ->
  part array ->
  [ `Ok | `Failed of string ]
(** [submit g ~v ~waits ~handles ps] hands over [ps], which {!room} answered
    [`Fits] for, as [g]'s value [v], the value after the last one [g] was
    given. Each wait [(k, a, w)] holds the work back until the aligned 64-bit
    word at address [a], which [g]'s work addresses, holds at least [w]
    ([`Word]), as unsigned integers; the compute queue waits. The work
    runs after every earlier value of [g] and after the waits; once it
    completed, the timeline word holds [v]. A submission of no parts writes [v]
    after its waits and after every earlier value. [handles] is ignored: the
    GPU's work names its memory by address.

    The result is [`Ok] once the queues were given every part, or [`Failed why]
    if a fill failed, as ["a fill on COMPUTE:0 failed with 1"]. The queues then
    run none of [ps], and the timeline word still reaches [v] once the earlier
    values completed. Every later [submit] answers the same [`Failed] and
    hands nothing over.

    Raises [Invalid_argument] if [v] is not the value after the last one, if a
    part is another device's, if a part's [after] names a part at or after its
    own index, if [waits] is not empty while {!waits_on}[ g `Store] is
    [false], or if a wait is [`Equal] or [`Object]: the device waits only on
    other devices' timeline words.
*)

val room_entry : nativeint
(** [room_entry] is the address of the C function [device_amd_room], {!room} for
    C, which [device_amd.h] declares. *)

val submit_entry : nativeint
(** [submit_entry] is the address of the C function [device_amd_submit],
    {!submit} for C, which [device_amd.h] declares. It is called with or
    without the domain lock, and calls no function of the OCaml runtime. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word g] is [g]'s timeline word: eight bytes of host memory holding, as an
    unsigned 64-bit integer in the host's byte order, the last value [v] such
    that the work of every value up to [v] completed. The queue that releases
    [v] writes it whole, after its work's writes reached memory; it never
    decreases. Other devices may map it and wait on it. It is never freed:
    another device's work may still read it after [g] is stopped or collected.
*)

val signaled : t -> int
(** [signaled g] is the value in {!word}, read with acquire order: the work of
    every value up to it completed, and its writes are visible to the reader. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep g ~seen ~still_ms] returns once [g]'s timeline word differs from
    [seen], at once if it does already, or after [still_ms] milliseconds. It
    blocks on the path's interrupt, which every release raises, and lets other
    domains run while it waits. It may run while {!submit} does.

    Raises {!Fault} with the path's report if [g]'s work met a fault. *)

(** {1:loss Loss} *)

exception Fault of string
(** The exception for a fault of a device's work, with the path's report. *)

val stop : t -> [ `Stopped | `Unknown ]
(** [stop g] stops [g] for good. It destroys [g]'s queues: once none runs, it
    raises the timeline word to the last value {!submit} was given, so work of
    other devices that waits on it runs on, and is [`Stopped]. It is [`Unknown]
    if the path could not destroy a queue; the word then reaches the last value
    only if every queue still runs. After [stop], only {!free} and {!unmap} may be called
    on [g]. *)

(** {1:paths Paths}

    For the libraries that open GPUs. A path reaches a GPU one way, reads what
    the device needs, and gives it all to {!make} as a {!path}. The device's
    queues, rings and timeline word {!make} makes through the path. *)

type 'm memory = {
  address : int;  (** The GPU address of its first byte. *)
  host : nativeint option;
      (** The host address of its first byte, if the host addresses it. *)
  path : 'm;  (** The path's own value for it. *)
}
(** The type for memory a path gives a device. *)

type 'm path = {
  id : 'm Type.Id.t;
      (** The path's key. Devices whose paths have one key map each other's
          memory with [map_peer]. *)
  gpu : Device_amd_abi.Gpu.t;  (** The GPU, as its formats depend on it. *)
  lds : int;  (** The local data share of a workgroup, in bytes. *)
  clock_hz : int;  (** The frequency of the GPU's clock, in hertz. *)
  mec : int;  (** The version of the firmware of its compute queues. *)
  wgps : int array array;
      (** The work-group processors that run work: [wgps.(e).(a)] has bit [w]
          set iff processor [w] of shader array [a] of shader engine [e] does,
          engines numbered across dies. *)
  budget : int;  (** The bytes of the GPU's own memory. *)
  alloc : [ `Gpu | `Bar | `System ] -> int -> 'm memory option;
      (** [alloc k n] is [n] new bytes: [`Gpu], GPU memory the host does not
          address; [`Bar], GPU memory the host addresses through the GPU's
          memory BAR; [`System], host memory the GPU addresses, cached for the
          host and snooped by the GPU, which the kernel driver owns, so that no
          unmap of host memory takes it from the GPU. It is [None] if the
          memory of [k] is exhausted, or [`Bar] memory does not exist. *)
  map_host : nativeint -> int -> 'm memory option;
      (** [map_host a n] maps for the GPU the pages that hold the [n] bytes of
          host memory at [a], at their host address: the result's [address]
          is [a]. It is [None] where the path refuses the pages, such as
          read-only ones. *)
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
    (nativeint, string) result;
      (** [queue k ~ring ~bytes ~read ~write] makes a hardware queue that reads
          packets of [k] from the [bytes] bytes of [`System] memory at [ring],
          writes its read position to the 64-bit word at [read] and reads its
          write position from the word at [write], all GPU addresses. It is the
          host address of the queue's 64-bit doorbell, or [Error msg]. *)
  hdp : nativeint option;
      (** The host address of the 32-bit register whose store flushes the
          GPU's host data path, after which host writes through the memory
          BAR are in the GPU's memory; [None] if the host does not reach it. *)
  interrupt : int;
      (** The context, not [0], that the interrupt of a value's release
          carries, which wakes [sleep]. *)
  sleep : ms:int -> unit;
      (** [sleep ~ms] returns once the GPU interrupts, or after [ms]
          milliseconds, and raises {!Fault} with the path's report of a fault
          of the GPU's work. *)
  stable_power : unit -> (unit, string) result;
      (** [stable_power ()] holds the GPU's clocks and shader engines steady,
          as tracing needs, for the life of the process. *)
  stop : unit -> [ `Stopped | `Unknown ];
      (** [stop ()] destroys the queues [queue] made: [`Stopped] once none of
          them runs, [`Unknown] if one may. *)
}
(** The type for what a path gives a device. A path fills it with functions
    over its own state, which this library never names. Each function answers
    the GPU's refusal as its result and raises {!Fault} for any other failure
    of the GPU or of the path's kernel driver. Any domain may call them at the
    same time, except [stop], which the device calls once, after every other
    call returned, and after which it calls only [free]. *)

val make : 'm path -> (t, string) result
(** [make p] is a device of the GPU [p] reaches: its queues, timeline word and
    argument segment, made through [p]. The result is [Error msg] if this
    library does not drive the GPU's family, naming it, if [p] refuses a queue,
    with [p]'s message, or if [p] has not the memory they need. A failed
    [make] gives back what it took: it frees the memory [p] gave and calls
    [p.stop] if [p] made a queue.

    Raises [Invalid_argument] if [p.interrupt] is [0]. *)

val is_gpu : vendor:int -> class_:int -> bool
(** [is_gpu ~vendor ~class_] is [true] iff a PCI function of vendor [vendor] and
    24-bit class code [class_] is an AMD GPU: AMD's vendor [0x1002] and a
    display controller or a processing accelerator, base class [0x03] or
    [0x12]. Every path numbers a machine's GPUs [0], [1], ... in bus order
    among such functions. *)
