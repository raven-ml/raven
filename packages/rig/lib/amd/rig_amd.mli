(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The driver of AMD GPUs.

    A device of this driver is one AMD GPU. Paths find and take GPUs, and
    the driver runs them. A {e path} is a library that reaches a GPU in one
    way and gives the driver what it found there: the GPU's description,
    and functions that allocate the GPU's memory, make its hardware queues,
    wait for its interrupt and stop it ({!type-path}). [Rig_amd_amdgpu]
    reaches a GPU through Linux's [amdgpu] kernel driver, and [Rig_amd_pci]
    drives the GPU's PCI function itself, with no kernel driver. Each offers
    an opener whose [make] fills a path and calls {!make}.

    From any path, {!make} builds the same device. Two queues take rig's
    work. ["COMPUTE:0"] is a ring of PM4 or AQL packets that the GPU's
    compute engine reads, and ["COPY:0"] a ring of SDMA packets that a copy
    engine reads. Both rings lie in [Pinned] memory, so the driver's
    [submit] places packets and rings the queues' doorbells with plain
    stores. The {e argument segment} is [Pinned] memory into which [submit]
    copies each launch's arguments. The last packet of a value's work is
    its {e release}, which writes the value into the timeline word, the
    eight bytes of host memory that the opener's [make] is given, once the
    work has completed, and raises the
    path's interrupt, on which {!val-sleep} blocks. What differs between
    paths, such as a bound on how long work runs or whether the GPU maps
    host memory, is a field of the path, which the device's facts carry
    ({!section-facts}). The sections below hold on every path.

    {v
    rig --room, submit--> COMPUTE:0 ring, PM4 or AQL --+
                          COPY:0 ring, SDMA -----------+--> GPU
                                                             |
                              release of value v, interrupt  |
    rig <--signaled, sleep-- timeline word, host memory <----+

    path: the GPU's memory, its hardware queues, its interrupt, its stop
    v}

    A device's life goes like this:
    {ol
    {- A path's opener reaches the GPU and fills a {!type-path}.}
    {- {!make} makes the device's rings and argument segment, and maps its
       timeline word, through the path. [Rig.open_] runs these two steps through the
       opener's [make].}
    {- rig gives each submission to the driver's [room] and [submit]
       ({!section-c}). The GPU runs its parts and releases its value.}
    {- Once rig loses or closes the device, {!val-stop} ends it
       ({!section-stop}).}}

    Unless a value says otherwise, a function that calls the path returns
    the path's refusal of its arguments as [None] or [Error], and raises
    {!Fault} for any other failure of the GPU or the path. {!signaled},
    {!val-free} and {!val-stop} never raise it. Any domain may call any
    value concurrently with others, except the C functions and {!val-stop},
    which rig calls as {!Rig_edge.Driver} states. {!val-sleep} may run
    beside the C functions. The [scratch] and [trace] of the device's
    {!capability}, and an opener's [make], called from a callback inside a
    lock ([Rig]'s domains section), raise [Invalid_argument], having changed
    nothing ({!Rig_edge.section-drivers}).

    A program names a GPU through a path's opener:
    {[
    match Rig.open_ (Rig_amd_amdgpu.gpu 0) with
    | Error why -> prerr_endline why
    | Ok d -> print_endline (Rig.arch d) (* such as "gfx1201" *)
    ]}

    {!Rig_edge.section-calls} states how rig hands a driver work for its
    queues, and {!Rig_edge} what its timeline word shows. [Rig_pci]'s
    section on what an open needs states what an open of a GPU with no
    kernel driver needs. *)

(** {1:gpus GPUs}

    An AMD GPU has two version numbers. Its processor
    ({!Rig_amd_abi.Gpu.processor}), such as gfx942, names the code objects
    it runs. Its GC version, such as 9.4.3, names the block that runs its
    compute queues, whose registers and packets follow it. The Instinct MI300
    series is processor gfx942 with GC 9.4.3. This driver names GPUs by
    processor, and [Rig_amd_pci], which programs the GC, by GC version.

    The driver drives gfx942, gfx950 (the MI350 series), and the processors
    of GFX11 and GFX12. On a GPU of one die its compute queue reads PM4
    packets. On a GPU of several, such as the MI300X, it reads AQL packets,
    which every die runs ({!Rig_amd_abi.Capability.type-compute}). The copy
    queue reads SDMA packets.

    On a GPU of several dies, the AQL queue hands every kernel the device's
    scratch memory.

    The driver has run on gfx1201 alone. Its AQL queue, with the dispatch,
    scratch and room of GPUs of several dies, has never run on hardware. *)

(** {1:order Order}

    ["COMPUTE:0"] runs words ({!Rig.Submission.words}) and launches, and
    ["COPY:0"], the copy engine, runs words and copies. The parts of one queue
    run in array order. A part's [after] indices, which rig derives
    ({!Rig_edge.type-queue}), order it after parts of the other queue, and two
    parts of different queues that [after] does not order may run in
    parallel.

    A copy runs on the copy engine or on the compute engine, and keeps its
    place in the order of ["COPY:0"] whichever engine runs it, as does a copy
    of no bytes.

    Value [v]'s work starts once the work of [v - 1] has completed, on both
    queues. A submission of no parts releases its value once [v - 1] is
    reached.

    A wait for another device's point holds the submission's work back until
    the aligned 64-bit word at the wait's address holds at least the point's
    value, compared as unsigned integers. The compute queue makes such waits
    only on a GPU whose GC and compute firmware ([mec] of {!type-path}) this
    library knows to compare 64-bit words. On other GPUs the driver's
    [waits] lists neither [stores] nor [hosts] ({!Rig_edge.type-waits}), and
    rig waits on the host.

    {b Caches.} The first part of a submission on the compute engine, and any
    part after a wait for the other queue, follows an invalidation of every
    cache of the GPU, its L2 included, after the L2 is written back. Any
    other launch follows at least an invalidation of the caches above the
    L2. A launch therefore reads what the host, the copy queue, other devices
    and the launches before it wrote. A value's release writes the L2 back,
    so the host and other devices read what its work wrote once the value is
    reached. *)

(** {1:device_memory Memory}

    {ul
    {- [Device] memory is GPU memory, which the host does not address.}
    {- [Pinned] memory is host memory that the GPU addresses, which the host
       reads as fast as other memory.}
    {- [Mapped] memory is GPU memory that the host addresses through the
       GPU's memory BAR, write-combined, so the host writes it fast and reads
       it slowly. The host's writes reach the work submitted after them.
       {!val-alloc} returns [None] for it on a path without such memory or
       whose [hdp] is [None] ({!type-path}).}}

    {!val-alloc} returns [None] if the GPU or the host lacks the memory.
    {!val-alloc} and {!val-map_host} raise [Invalid_argument] for fewer than
    one byte. Freeing a mapping ends the mapping alone, and the memory it
    maps stays.

    {!locate} returns the GPU address of a region's first byte, and that
    address again as the handle, since the GPU names memory by address. The
    host address is [None] iff the region is [Device] memory or a
    {!val-map_peer} view of [Device] memory.

    {!peer} and {!val-map_peer} return [true] and a view at the same GPU
    address iff paths of one [key] opened both devices and this device's
    path reaches the other GPU's memory ([reaches] of {!type-path}). A
    device flushes the host writes of at most 7 other GPUs, so
    {!val-map_peer} returns [None] for [Mapped] memory of an eighth. A view
    of [Mapped] memory flushes the other GPU's host writes before this
    device's later work until the other device stops, and {!val-map_peer} of
    a stopped device's [Mapped] memory is [None]. Peer
    access between two GPUs, with the flushes of each other's host writes,
    is untested, and so is a path whose [hdp] is [None].

    {!val-map_host} returns a region the GPU addresses, or [None] if the
    path maps no host memory or refuses the pages. Both paths map host
    pages for writing, so it returns [None] for pages mapped read-only,
    such as an io device's pages over a read-only file.
    [Rig_amd_amdgpu]'s kernel driver faults the pages in for writing, and
    [Rig_amd_pci] pins them for writing through VFIO. *)

(** {1:images Images}

    {!val-image} returns [Place (n, lay)] for a code object, whose image
    takes [n] bytes ({!Rig_amd_abi.Code_object.size}) of the GPU's own
    memory, from which its code runs. It returns [Error msg]:
    {ul
    {- if the binary is not a code object, with
       {!Rig_amd_abi.Code_object.of_string}'s message;}
    {- if it is for another processor, as
       ["a code object for gfx90a; the GPU is gfx1201"];}
    {- if a kernel takes more local data share than the GPU has, naming
       the kernel and both sizes.}}

    {!entry} returns the address of the kernel's descriptor, the symbol
    [f ^ ".kd"], as [code], which a dispatch names, the kernel's
    [explicit_size] as [params] and its [max_threads] as [threads]
    ({!Rig_amd_abi.Code_object.type-kernel}). It makes the kernel's
    launch and grows the device's scratch memory to the kernel's private
    segment. It answers [No_memory n], having made nothing, if the GPU has no
    memory for the [n] bytes of that scratch. It raises [Invalid_argument]
    for a kernel:
    {ul
    {- that reads an implicit argument that is a runtime's service ([Other]
       of {!Rig_amd_abi.Code_object.type-hidden});}
    {- that reads its dispatch packet, on a GPU of one die.}}
    {!unload} frees the launches. *)

(** {1:driver Driver}

    The sections above state what this driver adds to each value's
    contract. *)

include Rig_edge.Driver

val capability : t -> Rig_amd_abi.Capability.t
(** [capability g] is the record [g]'s facts carry as [capability]
    ({!val-facts}). *)

(** {1:this This driver}

    {2:facts Facts}

    The device's facts ({!Rig_edge.type-facts}) are:
    {ul
    {- [arch], the GPU's processor ({!Rig_amd_abi.Gpu.processor}), such as
       ["gfx1201"];}
    {- [budget], the bytes of the GPU's own memory;}
    {- [queues], ["COMPUTE:0"], which runs [Words] and [Launch], and
       ["COPY:0"], which runs [Words] and [Copy], index [0] and [1] in C;}
    {- [copy_engine], [Some 1], ["COPY:0"];}
    {- [completion], [Store], since the queue that releases a value writes
       the word;}
    {- [waits], [stores] and [hosts] iff the compute queue compares 64-bit
       words ({!section-order}), and [most] of [255];}
    {- [may_block], [false], since the C functions write memory and call no
       system function;}
    {- [hang_ms], the path's [hang_ms], [None] on a path whose kernel driver
       bounds how long work runs;}
    {- [maps_host], [true] iff the path maps host memory ([map_host] of
       {!type-path});}
    {- [host_addresses], [false], since the host does not address [Device]
       memory;}
    {- [alignment], [4096], since both paths allocate whole 4 KiB pages of
       the GPU, at addresses that start on one;}
    {- [capability], {!capability}, declared with
       {!Rig_amd_abi.declare};}
    {- [word], the device's timeline word, as {b Word.} below states.}}

    {b Capability.} The record holds the GPU, its clock, its compute queue's
    packets, its work-group processors that run work, and its trace buffers,
    made at the first [trace]. On an AQL queue its [scratch] grows the
    device's scratch memory, which the queue hands every kernel, and
    returns [Error] once {!val-stop} has begun. On a PM4 queue the dispatch
    of each launch names the scratch.

    {b Word.} The device maps rig's whole array of words through the path
    ([map_host] of {!type-path}). The path's first map of the array on a GPU
    makes the mapping, and every later one counts against it: a GPU maps the
    array once, at the host's addresses, while a device of it lives. On a
    path that maps no host memory, or when the word [make] was given lies
    outside the array, the word is eight bytes of the device's own
    [Pinned] memory, started at the given word's value, which rig reads
    through {!signaled}. The device's values start after that value. The
    word holds an unsigned 64-bit integer in the host's byte order. A
    release writes it after its work's writes reached memory. A release on
    ["COMPUTE:0"] writes all 64 bits, and one on ["COPY:0"] the low 32
    bits, so the driver releases on ["COMPUTE:0"] every value whose low 32
    bits are [0], the only values whose high 32 bits change.

    {b Sleep.} {!val-sleep} blocks on the path's interrupt, which every
    release raises, and lets other domains run while it waits. It may return
    on the interrupt of an earlier release or of other work of the GPU. The
    path reports a fault of the device's work, such as a page fault or a
    reset of the GPU, and {!val-sleep} raises it as {!Fault} with the path's
    report.

    {2:c C functions}

    [edge]'s [room], [submit] and [settle] are [rig_amd_room],
    [rig_amd_submit] and [rig_amd_settle], declared in [rig_amd.h]. Its
    [commit] does nothing, since [submit] places each value's release and
    returns [RIG_COMMITTED]. A part is:
    {ul
    {- words, whole packets of the queue's kind: PM4, or AQL in multiples of
       16 words, on ["COMPUTE:0"], and SDMA on ["COPY:0"];}
    {- on ["COMPUTE:0"], a launch. [submit] writes the kernel's arguments
       into the argument segment: the launch's parameters, each ref's
       address added, then the implicit arguments the kernel reads
       ({!Rig_amd_abi.Code_object.type-hidden}) for a grid of whole groups
       that starts at work-item 0. A group takes the kernel's group segment
       plus the launch's shared memory of local data share;}
    {- on ["COPY:0"], a copy between two GPU addresses, its handles, whose
       ranges do not overlap.}}

    [room] returns [RIG_NEVER] if the parts exceed what the device holds
    when it is idle, or if a part is one it does not run. An idle device
    holds 512 parts, and the words and argument bytes that fit in the
    compute ring, half the copy ring and half the argument segment, since a
    submission's SDMA packets and argument bytes never wrap. The device's
    [most_parts] ({!Rig_edge.type-facts}) is the most parts that an idle
    device holds, each a launch whose arguments take at most 4352 bytes,
    which every kernel of at most 4096 bytes of explicit arguments and code
    object version 5's 256 bytes of implicit ones meets, or a copy of at
    most {!Rig_amd_abi.Sdma.max_copy} bytes (1 GiB; 4 MiB on SDMA engines
    before 4.4.2 and on 5.0 and 5.1). A launch of more arguments, a longer
    copy and words take the room their sizes take. It does not run:
    {ul
    {- a copy of one byte or more on ["COMPUTE:0"];}
    {- words on an AQL queue that are not whole packets;}
    {- a launch whose parameters end before its kernel's explicit arguments
       do ([explicit_size] of {!Rig_amd_abi.Code_object.type-kernel});}
    {- a launch whose grid or group has an empty axis, whose groups have more
       work-items than its kernel's [max_threads], or whose shared memory
       exceeds the GPU's local data share less the kernel's group segment;}
    {- on a GPU of several dies, a launch of [2{^32}] work-items or more
       along an axis;}
    {- a part with an [after] index not below its own.}}

    On PM4 and SDMA rings, the queues read none of a submission's packets
    before all of them are placed, since [submit] moves the write position
    and rings the doorbell last. An AQL queue may read a packet as soon as
    its header is placed, and [submit] places each header after the rest of
    its packet. rig gives [submit] at most 255 waits, and none where the
    compute queue cannot wait, as the device's facts state
    ({!Rig_edge.type-facts}).

    {b Times.} While a profile is taken, each queue a value's work enters
    writes the GPU's clock after the value's waits, and the queue that
    releases the value writes it before the release: on ["COMPUTE:0"] once
    the dispatches before it have completed ({!Rig_amd_abi.Pm4.copy_data}
    of the clock), on ["COPY:0"] after a fence has waited for its copies
    ({!Rig_amd_abi.Sdma.timestamp}). [settle], or a later [submit],
    converts the earliest start and the end to the host's clock, in value
    order, with each no earlier than the end of the value converted before
    it, since a value's work starts once every earlier value's work has
    completed. A stamp of a completed value that the work never wrote makes
    the next [settle] fail, which loses the device.

    A device measures the clock against the host's clock as the path's
    {!type-clock} says, and converts a stamp along the line through its
    first measure, at open, and its latest, once they lie a second apart,
    and at [clock_hz] from the latest before. A measure through a {!Source}
    takes the time of a host clock between two readings of it. A re-measure
    whose readings lie more than 10 µs apart is dropped, and the measure at
    open keeps the first of sixteen reads within 10 µs, or the narrowest,
    which the first re-measure within 10 µs replaces. A time is thus within
    half its measure's interval, 5 µs once a measure within the bound is
    taken, plus the line's rate error over the time since the latest
    measure, at most 62.5 ms. That rate error is at most 10 µs over the
    measures' distance, 10 ppm, and the clock's own until they are a second
    apart. Through KFD, KFD reads its clock right after the GPU's, and
    the device maps that clock to the host's with a pair of readings of
    its own. With {!Round_trip} the device measures at open only, and its
    times drift from the host's clock at the clock's rate error since
    open.

    {2:stop Stop}

    {!val-stop} passes [fault] to the path's [stop], which destroys the
    device's queues. Once none runs, it writes the last value given to
    [submit] into the word and gives back the device's own memory. If the
    path could not destroy every queue, the device keeps its memory, and
    the word reaches that value only if the queues still run and complete
    their work.

    Either way the stop frees the C state the device ran work with, its
    rings' and timed values' records, which the queues never read. The edge
    that reads the word, 24 bytes, stays for the process. A word of the
    device's own goes back when rig frees it ({!Rig_edge.type-facts}).

    Once the word shows the last value and {!val-stop} has returned, no work
    of the device writes it: the queues release values in order, and the
    last value's release is the last write; the stop writes the word itself
    only once the path destroyed every queue. *)

(** {1:paths Paths}

    Only path libraries use these values. A path reaches a GPU in one way,
    reads what the device needs, and gives it all to {!make} as a
    {!type-path}. What the driver does when a path refuses or fails a call,
    such as a [stop] that answers [`Unknown] or a [free] that raises, is
    untested. *)

type 'm memory = {
  address : int;  (** The GPU address of its first byte. *)
  host : int option;
      (** The host address of its first byte, if the host addresses it. *)
  data : 'm;  (** The path's own data for it. *)
}
(** The type for memory a path gives a device. *)

(** The type for how a device reads the GPU's clock, the clock of its
    stamps, against the host's clock ({!section-c}). *)
type clock =
  | Source of int
      (** The address of the path's [struct rig_amd_clock] ([rig_amd.h]),
          which the path keeps while a device may read it. A device reads
          it at open and again in [rig_amd_settle], at most every
          62.5 ms. *)
  | Round_trip
      (** The device measures at open only, with a stamp its compute queue
          writes between two readings of the host's clock, the closest of
          eight. *)

type 'm path = {
  key : 'm Type.Id.t;
      (** The path's key: devices whose paths share [key] are of one
          machine, and map each other's memory with [map_peer]. *)
  index : int;
      (** The GPU's number among the machine's AMD GPUs in bus order
          ({!is_gpu}). *)
  gpu : Rig_amd_abi.Gpu.t;  (** The GPU, as its formats depend on it. *)
  waves : int;  (** The most waves a compute unit runs concurrently. *)
  lds : int;  (** The local data share of a workgroup, in bytes. *)
  clock_hz : int;  (** The frequency of the GPU's clock, in hertz. *)
  clock : clock;
      (** How a device reads the GPU's clock against the host's clock. *)
  mec : int;  (** The version of the firmware of its compute queues. *)
  wgps : int array array;
      (** The work-group processors that run work: [wgps.(e).(a)] has bit [w]
          set iff processor [w] of shader array [a] of shader engine [e] does,
          engines numbered across dies. A die whose processors the path cannot
          read has every processor of its arrays set. *)
  budget : int;  (** The bytes of the GPU's own memory. *)
  alloc : Rig_edge.memory -> int -> 'm memory option;
      (** [alloc k n] is [n] new bytes of memory [k]:
          {ul
          {- [Device], GPU memory the host does not address;}
          {- [Mapped], GPU memory the host addresses through the GPU's memory
             BAR;}
          {- [Pinned], host memory the GPU addresses, cached for the host and
             snooped by the GPU. The path owns it, so that no unmap of the
             process's memory takes it from the GPU.}}
          It is [None] if the memory of [k] is exhausted, or if [Mapped]
          memory does not exist. *)
  map_host : (int -> int -> 'm memory option) option;
      (** [Some map] if the GPU maps host memory. [map a n] maps for the GPU
          the pages that hold the [n] bytes of host memory at [a]. The
          result's [address] is where the GPU reaches the byte at [a], and its
          [host] is [Some a]. It is [None] if the path refuses the pages, such
          as read-only ones. [map_host] is [None] if the GPU maps no host
          memory. *)
  reaches : int -> bool;
      (** [reaches j] is [true] iff [map_peer] maps the GPU memory the path
          gives a device of GPU [j], numbered as [index]. It is [true] for
          this GPU. *)
  map_peer : 'm memory -> 'm memory option;
      (** [map_peer m] maps for the GPU the memory [m] that the path gave
          another of its devices, at the same address, or is [None] if the
          GPU does not reach that device's GPU's memory. Memory of a device of
          the same GPU is in its address space already, so the view maps
          nothing, and [free] of it gives nothing back. *)
  free : 'm memory -> unit;
      (** [free m] gives back what [alloc], [map_host] or [map_peer] gave. *)
  queue :
    [ `Pm4 | `Aql | `Sdma ] ->
    ring:int ->
    bytes:int ->
    read:int ->
    write:int ->
    (int, string) result;
      (** [queue k ~ring ~bytes ~read ~write] makes a hardware queue that
          reads packets of [k] from the [bytes] bytes of [Pinned] memory at
          [ring], writes its read position to the 64-bit word at [read] and
          reads its write position from the word at [write], all GPU
          addresses. It is the host address of the queue's 64-bit doorbell, or
          [Error msg]. *)
  hdp : int option;
      (** The host address of the 32-bit register whose store flushes the
          GPU's host data path, after which host writes through the memory BAR
          are in the GPU's memory; [None] if the host does not reach it. The
          register must stay mapped until [stop] is called. Devices that view
          the GPU's [Mapped] memory ([map_peer]) store to it too, until
          {!val-stop}, which waits for their stores before it calls
          [stop]. *)
  interrupt : int;
      (** The nonzero context that the interrupt of a value's release
          carries, which wakes [sleep]. *)
  hang_ms : int option;
      (** [Some n] if work whose word has not moved for [n] milliseconds hung,
          where nothing else bounds the GPU's work; [None] where the path's
          kernel driver does. It is the [hang_ms] of the device's facts
          ({!Rig_edge.type-facts}). *)
  sleep : ms:int -> unit;
      (** [sleep ~ms] returns once the GPU interrupts, or after [ms]
          milliseconds. It raises {!Fault} with the path's report of a fault
          of the GPU's work. *)
  stable_power : unit -> (unit, string) result;
      (** [stable_power ()] holds the GPU's clocks and shader engines steady,
          as tracing needs, for the life of the process. *)
  stop : fault:string option -> [ `Stopped | `Unknown ];
      (** [stop ~fault] destroys the queues [queue] made. It returns
          [`Stopped] once none of them runs or none can write memory outside
          the GPU, and [`Unknown] if one may. [fault] is the reason rig lost
          the device for, if it lost it for a fault or a hang
          ({!Rig_edge.Driver.stop}). A path that keeps the GPU's state for
          later opens records it; a path that keeps none ignores it. *)
}
(** The type for what a path gives a device. A path fills it with functions
    over its own state, which this library never names.
    {ul
    {- Each function returns the GPU's refusal as [None] or [Error], and raises
       {!Fault} for any other failure of the GPU or of the path.}
    {- Any domain may call them at the same time, except [stop], which the
       device calls once. After [stop] the device calls only [free] and
       [reaches]. It keeps [free] until it has given back all the memory
       the path gave, and [reaches] for the life of the process.}
    {- A [sleep] that another domain began before [stop] may still be waiting
       when [stop] runs. It returns without reading the GPU once [stop] has
       begun.}
    {- A function that holds a lock across OCaml code must take it with
       {!Lock.protect} ({!section-locks}).}} *)

val make : word:nativeint -> 'm path -> (t, string) result
(** [make ~word p] is a device of the GPU [p] reaches, whose timeline word is
    the 8 bytes at the host address [word] ({!Rig_edge.type-opener}). It
    makes, through [p], the device's queues and argument segment, maps its
    word, and loads the kernel the device copies with. A failed [make] gives back what it took once
    [p]'s [stop] returns [`Stopped], and keeps it otherwise. It calls [stop]
    only if it made a queue.

    The result is [Error msg]:
    {ul
    {- if this library does not drive the GPU's processor, or knows no
       registers for its GC, naming the processor;}
    {- if [p] refuses a queue, with [p]'s message;}
    {- if [p] lacks the memory the device needs, naming what it was for;}
    {- if [p]'s [Pinned] memory has no host address;}
    {- if this library's copy kernel for the GPU's processor does not load,
       naming the kernel and the processor and saying why;}
    {- if [p.clock] cannot read the GPU's clock, or, with {!Round_trip},
       the GPU writes no stamp of its clock within a second.}}

    It takes its first measure of the GPU's clock as [p.clock] says.

    Raises [Invalid_argument] if [p.interrupt] is [0], if [p.clock_hz] is
    below [1], or if a count of [p.gpu] ({!Rig_amd_abi.Gpu.t}) is below
    [1]. *)

val is_gpu : vendor:int -> class_:int -> bool
(** [is_gpu ~vendor ~class_] is [true] iff a PCI function of vendor [vendor]
    and 24-bit class code [class_] is an AMD GPU: AMD's vendor [0x1002] and a
    display controller or a processing accelerator, base class [0x03] or
    [0x12]. Every path numbers a machine's GPUs [0], [1], … in bus order among
    such functions. *)

(** {2:locks A path's locks}

    For a path's implementation. A path that holds a lock across OCaml code
    must take it with {!Lock.protect}, which counts it among its thread's
    locks ([rig_hold] in [rig_edge.h]), so that a call of rig's or of the
    driver's from a callback inside it is refused as the call starts and
    does not wait for good.

    {b Warning.} The path must take its locks in the order {!Lock.create}
    states. Nothing checks it, and two threads that take two locks in
    opposite orders wait for each other for good. Only a take by the thread
    that holds the lock raises. A path's entries must call {!Lock.check}
    themselves, before they change anything; one that does not may raise
    [Invalid_argument] from a lock it takes later, with what it changed
    before left changed. *)

module Lock : sig
  type t
  (** The type for a path's locks. *)

  val create : unit -> t
  (** [create ()] is a new lock. A thread may take it while it holds an
      image's lock, a device's, or rig.pci's vendor lock or a hold's
      ([Rig_pci.Gpus]), and may take rig.pci's memory locks while it holds
      it. *)

  val protect : t -> (unit -> 'a) -> 'a
  (** [protect l f] is [f ()] run holding [l], which it gives back as [f]
      returns or raises.

      Raises [Invalid_argument], having taken nothing, if the calling thread
      holds [l], as a callback that interrupts a section does. *)

  val check : string -> unit
  (** [check fn] raises [Invalid_argument], naming [fn], if the calling
      thread holds a lock of rig's or of a driver's. A path's opener's
      [make] and its public functions that may take a lock call it as they
      start, before they change anything ({!Rig_edge.section-drivers}). *)
end

(** {1:references References}

    {ul
    {- LLVM's
       {{:https://llvm.org/docs/AMDGPUUsage.html}User Guide for AMDGPU
       Backend}: code objects, kernel descriptors, the memory model of each
       GFX version.}} *)

(**/**)

(* [renumber ~age g v] makes [v] the value after [g]'s last one: the timeline
   word and the last value given become [v - 1], and every slot word is as if
   last written [age] (defaults to [0]) values earlier, at [v - 1 - age]. [g] is
   idle: its word holds the last value it was given. Tests reach the values a
   long run reaches, such as those past 2^31 and 2^32, and the slots' refresh
   past 2^31 values of age, without making that many submissions.

   Raises [Invalid_argument] if [g] has stopped or is not idle, if [v - 1] is
   below its last value, if [age] is negative or above [v - 1], or from a
   callback inside a lock ({!Lock.check}). *)
val renumber : ?age:int -> t -> int -> unit

(* [ring g q] is [(n, size)]: the 32-bit words [g] placed on the ring of its
   queue [q] since it made the ring, and the ring's size in words. A word's
   place on the ring is its count modulo [size]. Tests check that their
   submissions wrapped a ring.

   Raises [Invalid_argument] if [g] has stopped, if it has no queue [q], or
   from a callback inside a lock ({!Lock.check}). *)
val ring : t -> string -> int * int
