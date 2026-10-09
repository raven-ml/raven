(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs driven through their resource manager.

    A device of this library is one NVIDIA GPU, which a {e path} opened: a
    library that reaches the GPU's resource manager (RM) one way, such as
    [Rig_nv_nvidia] through NVIDIA's kernel driver, and fills a {!path} record.
    The device itself is the same whichever path opened it.

    A device runs work on two {e channels}, the GPU's hardware queues: its
    queues ["COMPUTE:0"] and ["COPY:0"]. Its work is one sequence of
    {e submissions}, which the caller numbers [1], [2], …, the {e values} of the
    device's {e timeline}. The device makes each observable in its
    {e timeline word}, 64 bits of host memory that hold [v] once the work of
    every value up to [v] completed, whichever channel ran it.

    A program opens a GPU through [Rig], which submits by calling this
    library's C functions ([rig_nv.h]):
    {[
    let d =
      Result.get_ok
        (Rig.open_ (module Rig_nv) ~name:"NV" (fun () -> Rig_nv_nvidia.open_ 0))
    in
    let src = Rig.Buffer.create ~memory:Pinned d 4096 in
    let dst = Rig.Buffer.create d 4096 in
    let copy =
      let work = Rig.Submission.Copy { src; dst } in
      { Rig.Submission.queue = "COPY:0"; after = [||]; work }
    in
    let s = Rig.Submission.make ~reads:0 ~writes:0 ~waits:0 d [| copy |] in
    Rig.wait d (Rig.Point.value (Rig.submit s))
    ]}

    {b Submissions.} A submission is the work of one value: {e parts}, each for
    one channel ([rig_nv.h]), and the waits on other devices' words it starts
    after. The device writes each part into its channel's ring, a GPFIFO, after
    a wait for the value before it and before the release of its own value, and
    wakes the channel. A part is ring entries that compiled code wrote
    ({!Rig_nv_abi.Gpfifo.entry}), or a copy the device encodes. Writing a
    submission makes no system call and calls no function of the OCaml runtime.

    {b Order.} A channel runs its work in order. The device makes a value's work
    start after the work of every value before it, and releases [v] into the
    word once its work completed: the release waits for the channel to be idle.
    Two submissions therefore never overlap on the GPU.

    {b Faults.} The RM stops a channel that faults, such as on a read its page
    tables refuse, and writes the error into the channel's notifier; the GPU's
    multiprocessors report their own errors to the RM. {!sleep} reads both
    reports and the path's ({!field-check}), and raises {!Fault} with them. Work
    that runs long is no fault unless the path bounds it ({!field-hang_ms}): a
    wait lasts until the work ends.

    A function that calls the RM answers its refusal of the arguments as its
    result ([None], [Error]) and raises {!Fault} for any other failure.
    {!signaled}, {!free} and {!stop} never raise it, and {!make} answers it as
    [Error].

    {b Domains.} Any domain may call any function, at the same time as others,
    under two rules. [rig_nv_room] and [rig_nv_submit] run one at a time: the
    caller holds the device's {e turn} from [rig_nv_room] to the end of
    [rig_nv_submit], while {!sleep} may run in another domain. {!stop} is called
    once, after every other call returned; after it, only {!free}, {!unload},
    {!signaled} and the capability's [local] are called.

    {b References.}
    - NVIDIA's
      {{:https://github.com/NVIDIA/open-gpu-kernel-modules}
       open-gpu-kernel-modules}, releases 570.144, 580, 610 and 615: the RM's
      classes and controls in [src/common/sdk/nvidia/inc] ([cl0080.h],
      [cl2080.h], [cla06c.h], [clc36f.h], [cl83de.h], [ctrl2080gr.h],
      [ctrl2080perf.h], [ctrl83debug.h], [ctrlc36f.h], [nvos.h]), and
      [nvstatuscodes.h].
    - NVIDIA's {{:https://github.com/NVIDIA/open-gpu-doc}open-gpu-doc}:
      [manuals/ampere/ga100/dev_pbdma.ref.txt] (GPFIFO entries, semaphores,
      [RELEASE_WFI]) and [dev_ram.ref.txt] (USERD, the doorbell). *)

(** {1:driver Driver} *)

include Rig_edge.Driver

(** {1:this This driver}

    {t
      | Fact | Value |
      |------|-------|
      | [arch] | The multiprocessors' architecture, as ["sm_89"]. |
      | [budget] | What the path reports the GPU's work may allocate. |
      | [queues] | ["COMPUTE:0"] runs [Words], ["COPY:0"] [Words] and [Copy]. |
      | [completion] | [Store]: the channels write the word. |
      | [waits] | [stores] and [hosts], not [objects]; [most] is [256]. |
      | [may_block] | [false]: the room check and hand-over store to memory. |
      | [maps_host] | Whether the path maps host memory ({!type-path}). |
      | [capability] | A {!Rig_nv_abi.Gpu.t} under {!Rig_nv_abi.Gpu.key}. |
      | [word] | Eight bytes of host memory. |
    }

    {b Queues.} The parts of a submission on one queue run in array order: a
    part starts once the parts before it on its queue completed, and reads what
    they wrote.

    {b Waits.} The channels wait on any 64-bit word the device maps, whoever
    writes it; [rig_nv_room] keeps room for [256] waits.

    {b Capability.} The record holds the GPU as its formats depend on it: the
    classes and counts its path read, the shared and local memory windows the
    device set on its compute channel, and [local], which grows the local
    memory the compute channel gives kernels ({!Rig_nv_abi.Gpu.field-local}).
    A growth takes effect at the next submission that uses ["COMPUTE:0"],
    before its parts; the memory it replaces stays allocated until the work
    before that submission completed. After {!stop}, [local] is [Error].

    {b Word.} The word holds, as an unsigned 64-bit integer in the host's byte
    order, the last value [v] such that the work of every value up to [v]
    completed. The channels write it with one 64-bit store, after waiting for
    their work and making its writes visible to the system.

    {b Memory.}
    - [Device] memory is GPU memory, which the host does not address.
    - [Pinned] memory is host memory that the GPU reads and writes coherently
      with the host.
    - [Mapped] memory is GPU memory that the host also addresses, through the
      GPU's memory BAR, while the BAR has room; {!alloc} answers [None] once it
      has none. The host's stores to it reach the GPU before the work of any
      later submission of the device reads it.

    {!alloc} answers [None] if the GPU or the host lacks the memory. {!alloc}
    and {!map_host} raise [Invalid_argument] for fewer than one byte. Freeing a
    mapping ends only that region: the memory it maps, and other regions over
    it, stay.

    {!locate} answers the address of a region's first byte in the GPU's
    virtual address space, below [2{^40}], and that address again as the
    handle: the device names memory by address. The host address is [None]
    for [Device] memory.

    {!peer} and {!map_peer} answer [true] and a mapping iff the same path
    opened both devices and it reaches the other GPU's memory from this one,
    as the path of two GPUs with peer access does. {!map_host} answers [None]
    if the path maps no host memory or refuses the pages. The path maps whole
    pages and keeps them mapped until every region over them is freed.

    {b Images.} {!image} answers [Place (n, lay)] for a cubin, whose image is
    [n] bytes ({!Rig_nv_abi.Cubin.size}), and [Error msg] for a binary that is
    no cubin ({!Rig_nv_abi.Cubin.of_string}). [lay r] relocates the cubin for
    [r]'s address. The device invalidates its compute engine's instruction
    cache at its next submission that uses ["COMPUTE:0"] after [lay], before
    its parts: a cubin may load where another one's code was. {!entry} answers
    the address of the first instruction of the kernel; the cubin's image
    starts there minus the kernel's code offset ({!Rig_nv_abi.Cubin.kernel}).
    An image holds nothing of the device but its code region: {!unload} does
    nothing.

    {b Sleep and faults.} {!sleep} reads the word and the channels' error
    notifiers every millisecond, and the multiprocessors' errors and the
    path's {!field-check} as it starts and once a notifier holds an error. It
    lets other domains run while it waits. Its {!Fault} report has a line for
    each error the RM wrote into the channels' notifiers, by its number and
    name, then one for each fault the multiprocessors or the MMU reported, the
    MMU's with the faulting address and access.

    If the path bounds progress ([hang_ms] is [Some n]), {!sleep} also raises
    {!Fault} once work is outstanding and the word has not moved for [n]
    milliseconds, and may return early when that clock runs out. The clock
    runs while a value given is above the word, from the later of the word's
    last move and the first [sleep] after the device was idle, as [sleep]
    observes them: an idle device never hangs, and the report may come late
    but never early.

    {b Hand-over.} [edge]'s room check, [rig_nv_room], answers [RIG_NEVER] for
    parts that exceed the device's empty rings, more than 65,535 parts, or a
    part the device does not run. It reads the word first, so [RIG_LATER]
    means a value the device was given is not yet reached. The hand-over,
    [rig_nv_submit], writes its release with each value and answers
    [RIG_COMMITTED]: the device is never asked to commit. A submission of no
    parts writes its value after its waits and after every earlier value. A
    [RIG_WORD] wait names an aligned 64-bit word at an address below
    [2{^40}] that the device's work addresses, and a value [w]. It holds the
    work back until the word holds at least [w], compared circularly: [x] is
    at least [w] iff [x - w], as a signed 64-bit integer, is not negative. The
    device's C state is never freed.

    {b Stop.} {!stop} ignores [fault]. It ends its path's registration of the
    channels ({!field-unregister}) and frees them, and the RM preempts what
    they run; then it asks the path to stop ({!field-stop}). The word holds the
    last value before {!stop} returns if the RM freed the channels, the path
    answered [`Stopped] or the RM had stopped the channels on a fault;
    otherwise the channels' own releases write it as their work ends. The
    device's own memory goes back to the path in the first two cases. *)

(** {1:paths Paths}

    For the libraries that open GPUs. A path reaches a GPU's RM, allocates the
    RM's device, subdevice and virtual address space for it, reads what the
    device needs, and gives it all to {!make} as a {!path}. Everything else a
    device holds, its channels and their memory, {!make} allocates through the
    path. *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for the parameters of an RM call: bytes outside the OCaml heap,
    which may point to other such bytes the caller keeps alive. *)

type rm = {
  release : int;
      (** The release of the RM's interface, such as [615]: it sets the layout
          of parameters that changed between releases. *)
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
  warps_per_sm : int;  (** The most warps a multiprocessor runs at once. *)
}
(** The type for what a path reads of a GPU, from which {!make} builds its
    {!Rig_nv_abi.Gpu.t}. *)

type 'm memory = {
  address : int;
      (** The GPU address of its first byte. The memory lies below [2{^40}], its
          last byte included: rings and semaphores take addresses of 40 bits,
          and the device packs its local memory's address in 40 bits and keeps
          kernels' windows onto shared and local memory above. A path may map
          host memory at a GPU address of its own, as the host's may lie higher.
          A device raises [Invalid_argument] for memory a path answers above. *)
  host : int option;
      (** The host address of its first byte, if the host addresses it. *)
  handle : int;  (** The RM's name for it, which channel allocations take. *)
  data : 'm;  (** The path's own data for it. *)
}
(** The type for memory a path gives a device. *)

type 'm path = {
  key : 'm Type.Id.t;
      (** The path's key. Devices whose paths have one key map each other's
          memory with [map_peer]. *)
  index : int;
      (** The GPU's number among the machine's NVIDIA GPUs in bus order
          ({!is_gpu}). *)
  rm : rm;  (** The GPU's RM. *)
  device : int;  (** The RM's device object of the GPU. *)
  subdevice : int;  (** Its subdevice object. *)
  vaspace : int;  (** The virtual address space the device's channels use. *)
  gpu : gpu;  (** The GPU's classes and counts. *)
  budget : int;  (** The GPU's memory its work may allocate, in bytes. *)
  doorbell : int;
      (** The host address of the 4-byte register into which the device stores a
          channel's work submit token to wake the channel
          ([NVC361_NOTIFY_CHANNEL_PENDING]). *)
  alloc : [ `Gpu | `Bar | `System ] -> int -> 'm memory option;
      (** [alloc k n] is [n] new bytes at an address aligned to 4 KiB, or [None]
          if the GPU or the host lacks the memory:
          - [`Gpu], GPU memory, which the host does not address;
          - [`Bar], GPU memory that the host also addresses, through the GPU's
            memory BAR, mapped uncached;
          - [`System], host memory that the GPU reads and writes coherently with
            the host. *)
  map_host : (int -> int -> 'm memory option) option;
      (** [Some map] if the GPU maps host memory: [map a n] is the [n] bytes of
          host memory at [a], mapped for the GPU, or [None] if the path refuses
          them. The path maps whole pages and keeps them mapped until every
          memory it gave over them is freed. [None] if the GPU maps no host
          memory, as one taken without an IOMMU, whose pages would go back to
          the system at the process's death while the GPU still writes them. *)
  reaches : int -> bool;
      (** [reaches i] is [true] iff this GPU's work addresses the GPU memory of
          GPU [i] of this path, another GPU. *)
  map_peer : 'm memory -> 'm memory option;
      (** [map_peer m] is the memory [m] of another GPU of this path, mapped for
          this one, or [None] if this GPU cannot address it. *)
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
          milliseconds is a fault ({!val-sleep}), where nothing else bounds the
          GPU's work; [None] where the path's kernel driver does. [n] is
          positive. *)
  stop : unit -> [ `Stopped | `Unknown ];
      (** [stop ()] ends the path's hold of the GPU after the device freed its
          channels: [`Stopped] once none of the device's work runs or can write
          memory outside the GPU, [`Unknown] if some may. *)
}
(** The type for what a path gives a device. Each function answers the GPU's
    refusal as its result and raises {!Fault} for any other failure. Any domain
    may call them at the same time. After the device's {!stop} it calls only
    [free]. *)

val make : 'm path -> (t, string) result
(** [make p] is the device of the GPU [p] reaches. Under [p.device] it allocates
    a channel group with one context share, the channels ["COMPUTE:0"] and
    ["COPY:0"] and their memory, and asks the RM to run the GPU at its highest
    clocks. The result is [Error msg] if [p.rm.release] is none of [570], [580],
    [610] and [615], if the RM refuses an object, or if [p] lacks the memory. A
    failed [make] frees what it allocated; [p]'s own objects stay [p]'s, and
    another [make] may use them once this one's device is stopped or failed.

    Raises [Invalid_argument] if [p.hang_ms] is [Some n] with [n < 1]. *)

val is_gpu : vendor:int -> class_:int -> bool
(** [is_gpu ~vendor ~class_] is [true] iff a PCI function of vendor [vendor] and
    24-bit class code [class_] is an NVIDIA GPU: NVIDIA's vendor [0x10de] and a
    display controller, base class [0x03]. Every path numbers a machine's GPUs
    [0], [1], … in bus order among such functions. *)
