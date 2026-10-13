(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The driver of NVIDIA GPUs.

    A device of this library is one NVIDIA GPU. The GPU's resource manager
    (RM) is the part of NVIDIA's software that makes the GPU's objects, such
    as its channels, its memory and its address spaces, for its clients. A
    {e path} is a library that reaches the RM one way and gives this library
    what it reached, as a {!type-path}. [Rig_nv_nvidia] reaches it through
    NVIDIA's kernel driver, and [Rig_nv_pci] through the GPU's own system
    processor, with no kernel driver. Each path's opener calls {!make}.

    The device runs its work on two hardware {e channels}, one for compute
    and one for copies. A channel fetches 64-bit entries from its ring, the
    GPFIFO, in order, and runs the 32-bit words of the segment each entry
    names ({!Rig_nv_abi.Method}). rig hands the device a submission through
    the two C functions of [rig_nv.h], [rig_nv_room] and [rig_nv_submit].
    [rig_nv_submit] writes the submission's parts into the channels' rings
    and wakes the channels through the GPU's doorbell register, with stores
    to memory alone. Each submission ends with a release that writes its
    value into the timeline word.

    {v
    Rig.submit -> rig_nv_room, rig_nv_submit      C, stores only
                      |                |
                      v                v
              compute channel     copy channel    ring entries name words
                      |                |
                      +-- release v ---+
                              |
                              v
                        timeline word             read by Rig_nv.sleep
    v}

    A device lives from {!make}, which allocates its channels and their
    memory, until rig calls {!val-stop}, which frees the channels and asks
    the path to let go of the GPU. Between them rig submits, waits in
    {!sleep}, allocates and frees memory, and loads images.

    Any domain may call any function, at the same time as others, within
    the rules rig keeps for every driver ({!Rig_edge.Driver}). A C caller
    other than rig must call [rig_nv_room] before each [rig_nv_submit], with
    no other call of either on the device between them. A function that
    calls the RM returns the RM's refusal of its arguments as its result,
    [None] or [Error], and raises {!Fault} for any other failure;
    {!signaled}, {!val-free} and {!val-stop} never raise it, and {!make}
    returns it as [Error].

    Compiled code finds the record its launch descriptors depend on under
    {!Rig_nv_abi.Gpu.key}:
    {[
    let d = Result.get_ok (Rig.open_ (Rig_nv_nvidia.gpu 0)) in
    let g = Option.get (Rig.capability d Rig_nv_abi.Gpu.key) in
    Printf.printf "%s, %d SMs\n" (Rig.arch d)
      (g.gpcs * g.tpcs_per_gpc * g.sms_per_tpc)
    ]}

    {!Rig_edge.section-calls} states how rig hands a driver work. *)

(** {1:driver Driver} *)

include Rig_edge.Driver

val capability : t -> Rig_nv_abi.Gpu.t
(** [capability g] is [g]'s GPU as compiled code sees it, which [g]'s facts
    declare ({!Rig_nv_abi.Gpu.declare}). *)

(** {1:facts Facts}

    {t
      | Fact | Value |
      |------|-------|
      | [arch] | The multiprocessors' architecture, such as ["sm_89"]. |
      | [budget] | The path's {!field-budget}. |
      | [queues] | ["COMPUTE:0"] and ["COPY:0"] ({!section-submitting}). |
      | [copy_engine] | [Some 1], ["COPY:0"]. |
      | [completion] | [Store]: the channels' releases write the word. |
      | [waits] | [stores] and [hosts]; [most] is [256]. |
      | [may_block] | [false]: its C functions only store to memory. |
      | [hang_ms] | The path's {!field-hang_ms}. |
      | [maps_host] | [true] iff the path maps host memory. |
      | [host_addresses] | [false]: the host does not address [Device] memory. |
      | [capability] | {!val-capability} ({!Rig_nv_abi.Gpu.declare}). |
      | [word] | Eight bytes of [Pinned] memory. |
    }

    The word holds, as an unsigned 64-bit integer in the host's byte order,
    the last value [v] such that the work of every value up to [v] has
    completed. A channel writes it once its work has completed and that
    work's writes are visible to the system. A read returns the value before
    a release or the value after it. This assumes the copy engine writes a
    64-bit release as one store.

    The capability's [local] grows the local memory the compute channel
    gives kernels ({!Rig_nv_abi.Gpu.t.local}). A growth takes effect at the
    next submission that uses the compute channel, after its waits and
    before its parts. After {!val-stop}, [local] returns [Error]. The record
    also holds the classes and counts the path read of the GPU, and the
    windows onto shared and local memory the device set on its compute
    channel. *)

(** {1:submitting Submitting}

    ["COMPUTE:0"] runs [Words] and [Launch] parts, and ["COPY:0"] runs
    [Words] and [Copy] parts. A part of [Words] is ring entries that
    compiled code wrote ({!Rig_nv_abi.Gpfifo.entry}).

    A value's work starts once the work of every value before it has
    completed, and the release of [v] waits for the work of [v] to complete,
    so two submissions never overlap on the GPU. Each queue runs its parts
    in array order: a part starts once the parts before it on its queue have
    completed, and reads what they wrote. [Words] run on their queue's
    channel. A copy runs on either channel and keeps its place in its
    queue's order, even a copy of zero bytes.

    A [RIG_WORD] wait names an aligned 64-bit word that the device's work
    addresses, and a value [w]. It holds the submission's work back until
    the word holds at least [w] in circular order
    ({!Rig_nv_abi.Method.acquire}), whoever wrote it.

    [rig_nv_room] returns [RIG_NEVER] for parts that would not fit in the
    rings even when empty, for more than 65,535 parts, and for a part the
    device does not run:
    - a part on neither queue, a launch on ["COPY:0"], or a copy of one byte
      or more on ["COMPUTE:0"];
    - ring entries of an odd number of words;
    - an [after] index that is not below the part's own;
    - a launch whose grid or group has an axis of size [0] or exceeds
      {!Rig_nv_abi.Qmd.max_size}, whose group has more threads than
      {!Rig_nv_abi.Launch.max_threads}, or whose dynamic shared memory,
      rounded up to 128 bytes, exceeds {!Rig_nv_abi.Launch.dynamic_shared}.

    The device's [most_parts] ({!Rig_edge.type-facts}) is the most parts
    that the empty rings hold, each a launch, whose constant bank 0 {!entry}
    bounds, or a copy of at most 2 GiB. A longer copy and words take the
    room their sizes take.

    It reads the word first, so [RIG_LATER] means that a value the device
    was given is not yet reached. [rig_nv_submit] returns [RIG_COMMITTED],
    since the channels run work as soon as they fetch it, and rig never asks
    the device to commit. A submission of no parts writes its value after
    its waits and every earlier value.

    While a profile is taken, each channel a value's work enters writes a
    stamp of the GPU's timer after the value's waits, and the channel that
    releases the value writes one before the release
    ({!Rig_nv_abi.Method.release_stamp}). On COMPUTE a stamp waits for the
    channel's earlier work to complete. [rig_nv_settle], the [edge]'s
    [settle], or a later [rig_nv_submit] converts the earliest start and
    the end to the host's clock, in value order, with each no earlier than
    the end of the value converted before it, since a value's work starts
    once every earlier value's work has completed. A stamp of a completed
    value that the work never wrote makes the next [rig_nv_settle] fail,
    which loses the device.

    A device measures the timer against the host's clock through the
    path's {!type-clock}, and converts a stamp along the line through its
    first measure, at open, and its latest, once they lie a second apart,
    and at the timer's nominal rate of a nanosecond a tick before. A
    measure takes the time of a host clock between two readings of it. A
    re-measure whose readings lie more than 10 µs apart is dropped, and the
    measure at open keeps the first of sixteen reads within 10 µs, or the
    narrowest, which the first re-measure within 10 µs replaces. A time is
    thus within half its measure's interval, 5 µs once a measure within the
    bound is taken, plus the line's rate error over the time since the
    latest measure, at most 62.5 ms. That rate error is at most 10 µs over
    the measures' distance, 10 ppm, and the timer's own until they are a
    second apart. Through NVIDIA's kernel driver ({!Source}) the driver
    takes the readings around its read of the timer's registers, and the
    device takes one more pair, to map the driver's clock to the host's.
    With {!Registers} the device reads the registers itself, at open only,
    and its times drift from the host's clock at the timer's rate error
    since open.

    A launch with dynamic shared memory runs in the multiprocessors'
    configuration of the most shared memory ({!Rig_nv_abi.Qmd.t.shared}). *)

(** {1:regions Memory}

    - [Device] memory is the GPU's, which the host does not address.
    - [Pinned] memory is host memory that the GPU reads and writes
      coherently with the host.
    - [Mapped] memory is the GPU's, which the host also addresses through
      the GPU's memory BAR. The host's stores to it reach the GPU before the
      work of any later submission of the device reads it.

    {!val-alloc} returns [None] if the GPU or the host lacks the memory, and
    for [Mapped] once the BAR has no room, or if it had none for the path's
    flush word ({!field-flush}). {!val-alloc} and {!val-map_host}
    raise [Invalid_argument] for fewer than one byte. The device itself
    holds [Pinned] memory for its rings and its word, and 1 MiB for the
    descriptors and constant banks of its launches, in the BAR where the BAR
    had room at {!make} and in [Pinned] memory otherwise.

    {!locate} returns the address of a region's first byte in the GPU's
    virtual address space ({!Rig_nv_abi.Gpu}). Its handle is the same
    address, and its host address is [None] for [Device] memory. Freeing a
    region leaves the memory it maps, and other regions over that memory, in
    place.

    {!peer} is [true] iff paths of one [key] opened both devices and this
    device's path reaches the other GPU's memory ([reaches] of
    {!type-path}), and {!val-map_peer} then maps it. Peer
    access between GPUs is untested. {!val-map_host} returns [None] if the
    path maps no host memory or refuses the pages ({!field-map_host}). *)

(** {1:images Images}

    An image is a cubin ({!Rig_nv_abi.Cubin}). {!val-image} returns
    [Place (n, lay)] for a cubin that {!Rig_nv_abi.Cubin.of_string} accepts,
    where [n] is its {!Rig_nv_abi.Cubin.size}, and [Error msg] otherwise.
    [lay r] relocates the cubin for the address of [r]. Every relocation
    must name a symbol the image defines, so CUDA's driver-supplied device
    functions, such as [__assertfail] and [vprintf], are unavailable: an
    image that refers to one is refused even if the kernel to launch never
    calls it. A device opened with [Rig_cuda.gpu] runs such images.

    The first submission on the compute channel after a [lay] invalidates
    the compute engine's instruction cache before its parts, since a cubin
    may load where another one's code was.

    {!entry} returns the address of the kernel's first instruction; the
    image starts there less the kernel's [code] offset
    ({!Rig_nv_abi.Cubin.type-kernel}). Its [params] are the kernel's
    [params_bytes], and its [threads] the launch's
    {!Rig_nv_abi.Launch.max_threads}. It sets the function up for launch
    ({!Rig_nv_abi.Launch.make}) and grows the compute channel's local memory
    to serve it ({!Rig_nv_abi.Gpu.t.local}). It answers [No_memory n],
    having made nothing, if the GPU has no memory for the [n] bytes of that
    local memory, and raises [Invalid_argument] with the reason if the
    function takes more than a launch can, such as parameters placed past 1
    KiB of its constant bank 0, or a constant bank 0 of more than 5120
    bytes. {!unload} frees the launch
    descriptors of the image's functions, and leaves its code region to
    rig. *)

(** {1:faults Faults and stopping}

    The RM stops a channel that faults, such as on a read its page tables
    refuse, and writes the error into the channel's notifier, a page of
    [Pinned] memory. The multiprocessors and the MMU report their own
    faults to the RM, and a path may learn of others ({!field-check}).

    {!sleep} reads the word and the notifiers every millisecond, and lets
    other domains run while it waits. As it starts to wait, and once a
    notifier holds an error, it asks the path and the RM for their reports,
    and raises {!Fault} if either has one. If the path has a report,
    {!Fault} carries it. Otherwise {!Fault} has a line for each distinct
    notifier error, by number and name, then one for each fault the multiprocessors or the MMU
    reported; an MMU line names the faulting address and access. A wait for
    work that runs long lasts until the work ends, unless the path bounds
    progress ({!field-hang_ms}): rig then loses the device past that bound.

    {!val-stop} ignores its [fault]. It ends the path's registration of the
    channels ({!field-unregister}) and frees them, which makes the RM
    preempt their work, then asks the path to stop ({!field-stop}). If the
    path unregistered and the RM freed the channels, or the path returns
    [`Stopped], the word holds the last submitted value before {!val-stop}
    returns, and the memory the device holds itself goes back to the path.
    Otherwise the word reaches that value when the channels stop or finish,
    and that memory stays allocated for the life of the process. Regions rig
    allocated return as {!Rig_edge.Driver.val-stop} states, in both cases.

    The word, and the device's C state, which reading it needs, stay for
    the process. *)

(** {1:paths Paths}

    A path reaches a GPU's RM, allocates the RM's device, subdevice and
    virtual address space for the GPU, reads what the device needs, and
    gives it all to {!make} as a {!type-path}. {!make} allocates everything
    else the device holds, its channels and their memory, through the
    path. What the driver does when a path or the RM refuses or fails a
    call, such as a [stop] that answers [`Unknown] or a [free] that raises,
    is untested. *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for the parameters of an RM call: bytes outside the OCaml heap,
    which the RM reads and writes in place. *)

type rm = {
  release : int;
      (** The release of the RM's interface, such as [615], which sets the
          layout of parameters that changed between releases. *)
  client : int;  (** The client, which the device's objects descend from. *)
  alloc : parent:int -> int -> params option -> (int, string) result;
      (** [alloc ~parent cls p] is a new object of class [cls] under [parent],
          made with the parameters [p], or the RM's refusal. *)
  control : int -> int -> params option -> (unit, string) result;
      (** [control obj cmd p] runs the command [cmd] on [obj], which reads and
          writes [p] in place, or is the RM's refusal. *)
  free : parent:int -> int -> (unit, string) result;
      (** [free ~parent obj] frees [obj] and every object under it, or is the
          RM's refusal. *)
}
(** The type for a GPU's resource manager, as a path reaches it. Any domain may
    call its functions at any time. *)

type gpu = {
  channel_class : int;  (** The class of its channels. *)
  compute_class : int;  (** The class of its compute engine. *)
  copy_class : int;  (** The class of its copy engine. *)
  sm_version : int;
      (** The version of its multiprocessors, as the RM reports it
          ([NV2080_CTRL_GR_INFO_INDEX_SM_VERSION]), such as [0x809]. *)
  gpcs : int;  (** Its graphics processing clusters. *)
  tpcs_per_gpc : int;  (** The texture processing clusters of one. *)
  sms_per_tpc : int;  (** The streaming multiprocessors of one of those. *)
  warps_per_sm : int;  (** The most warps a multiprocessor runs concurrently. *)
}
(** The type for what a path reads of a GPU, from which {!make} builds its
    {!Rig_nv_abi.Gpu.t}. *)

type 'm memory = {
  address : int;
      (** The GPU address of its first byte. The memory lies below [2{^40}],
          its last byte included ({!Rig_nv_abi.Gpu}). A path may map host
          memory at a GPU address of its own, as the host's may lie
          higher. *)
  host : int option;
      (** The host address of its first byte, if the host addresses it. *)
  handle : int;  (** The RM's name for it, which channel allocations take. *)
  data : 'm;  (** The path's own data for it. *)
}
(** The type for memory a path gives a device. *)

(** The type for how a device reads the GPU's timer, the clock of its
    stamps, against the host's clock ({!section-submitting}). *)
type clock =
  | Source of int
      (** The address of the path's [struct rig_nv_clock] ([rig_nv.h]),
          which the path keeps while a device may read it. A device reads
          it at open and again in [rig_nv_settle], at most every
          62.5 ms. *)
  | Registers of int
      (** The host address at which the GPU's BAR 0 is mapped, whose timer
          registers ([NV_PTIMER_TIME_0] and [_1]) a device reads at open
          only. *)

type 'm path = {
  key : 'm Type.Id.t;
      (** The path's key. Devices whose paths have one key are of one
          machine, and map each other's memory with [map_peer]. *)
  index : int;  (** The GPU's number ({!is_gpu}). *)
  rm : rm;  (** The GPU's RM. *)
  device : int;  (** The RM's device object of the GPU. *)
  subdevice : int;  (** Its subdevice object. *)
  vaspace : int;  (** The virtual address space the device's channels use. *)
  gpu : gpu;  (** The GPU's classes and counts. *)
  budget : int;  (** The GPU's memory its work may allocate, in bytes. *)
  doorbell : int;
      (** The host address of the 4-byte register into which the device
          stores a channel's work submit token to wake the channel
          ([NVC361_NOTIFY_CHANNEL_PENDING]). *)
  flush : int option;
      (** [Some a] if the host address [a] is a 32-bit word of the GPU's
          [Mapped] memory, which a device reads before waking its channels
          so that the host's writes through the GPU's BAR reach the GPU
          first; [None] if the BAR had no room for it, and the device then
          has no [Mapped] memory. The word must stay mapped while a device
          may read it: this GPU's devices, and the devices that view its
          [Mapped] memory ([map_peer]), until their stop. [Rig_nv_nvidia]
          keeps it for the life of the process. [Rig_nv_pci] unmaps it at
          the GPU's stop, with the BAR windows, so a view of a stopped GPU's
          [Mapped] memory on another device is not covered. *)
  alloc : Rig_edge.memory -> int -> 'm memory option;
      (** [alloc k n] is [n] new bytes of the kind [k] ({!section-regions}) at
          an address aligned to 4 KiB, [Mapped] memory mapped uncached for the
          host, or [None] if the GPU or the host lacks the memory. *)
  map_host : (int -> int -> 'm memory option) option;
      (** [Some map] if the GPU maps host memory: [map a n] is the [n] bytes of
          host memory at [a], mapped for the GPU, or [None] if the path refuses
          them. The path maps whole pages and keeps them mapped until every
          memory it gave over them is freed. [None] if the GPU maps no host
          memory. *)
  reaches : int -> bool;
      (** [reaches i] is [true] iff this GPU's work addresses the GPU memory of
          another GPU of this path's machine, GPU [i], numbered as [index]. *)
  map_peer : 'm memory -> 'm memory option;
      (** [map_peer m] is the memory [m] of another device of this path, mapped
          for this GPU, or [None] if this GPU cannot address it. *)
  free : 'm memory -> unit;
      (** [free m] gives back what [alloc], [map_host] or [map_peer] gave. *)
  register : int -> (unit, string) result;
      (** [register c] makes the channel [c], which the device allocated under
          [device], ready to run, as the path requires. *)
  unregister : int -> (unit, string) result;
      (** [unregister c] gives back what [register c] took, before the device
          frees [c]. *)
  check : unit -> unit;
      (** [check ()] raises {!Fault} with the path's report of a fault of the
          GPU's work that the RM sent the path, or of the GPU's failure, such as
          a lost function. *)
  hang_ms : int option;
      (** [Some n] if work whose timeline word has not moved for [n]
          milliseconds hung, where nothing else bounds the GPU's work; [None]
          where the path's kernel driver bounds it. It is the [hang_ms] of
          the device's facts ({!Rig_edge.type-facts}). *)
  clock : clock;
      (** How a device reads the GPU's timer against the host's clock. *)
  stop : unit -> [ `Stopped | `Unknown ];
      (** [stop ()] ends the path's hold of the GPU after the device has freed
          its channels. It must return [`Stopped] only once none of the
          device's work runs or can write memory outside the GPU, and
          [`Unknown] if some may. It takes no fault: no path keeps GPU state
          that a fault spoils for later opens, since NVIDIA's kernel driver
          recovers its GPUs itself and a GPU held over PCI is reset by its
          next open whatever ended its last. *)
}
(** The type for what a path gives a device. Each function must return the
    RM's refusal as its result and raise {!Fault} for any other failure, and
    must allow any domain to call it at the same time as others. After the
    device's {!val-stop}, the device calls only [free] and [reaches]. It keeps
    [free] until it has given back all the memory the path gave, and
    [reaches] for the life of the process. *)

val make : 'm path -> (t, string) result
(** [make p] is the device of the GPU [p] reaches. Under [p.device] it
    allocates a channel group with one context share, the two channels and
    their memory, and asks the RM to run the GPU at its highest clocks.

    It then takes its first measure of the GPU's timer through [p.clock].

    [Error msg] if [p.rm.release] is none of [570], [580], [610] and [615], if
    the RM refuses an object, if [p] lacks the memory, or if [p.clock]
    cannot read the timer. A failed [make] frees
    what it allocated. [p]'s own objects stay allocated, and another [make] may
    use them once this one has failed or its device has stopped. *)

val is_gpu : vendor:int -> class_:int -> bool
(** [is_gpu ~vendor ~class_] is [true] iff a PCI function of vendor [vendor] and
    24-bit class code [class_] is an NVIDIA GPU: NVIDIA's vendor [0x10de] and a
    display controller, base class [0x03].

    Every path numbers a machine's NVIDIA GPUs [0], [1], … in bus order among
    such functions, whichever kernel driver holds each: GPU [i] of one path is
    GPU [i] of every other. *)
