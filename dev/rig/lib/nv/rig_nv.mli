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
    {e timeline word} ({!word}), 64 bits of host memory that hold [v] once the
    work of every value up to [v] completed, whichever channel ran it.

    A program opens a GPU through the device core, [Rig], which submits by
    calling this library's C functions ({!room_entry}, {!submit_entry}):
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
    once, after every other call returned; after it, only {!free}, {!signaled}
    and the capability's [local] are called.

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

(** {1:facts Facts}

    What a device is, read once when it opens. *)

type t
(** The type for open NVIDIA GPUs: a GPU's channels, its memory and its timeline
    word. *)

val key : t Type.Id.t
(** [key] tells devices of this library apart from others'. *)

val arch : t -> string
(** [arch g] is the architecture of the GPU's multiprocessors, as ["sm_89"]. *)

val budget : t -> int
(** [budget g] is the GPU's memory that its work may allocate, in bytes, as its
    path reports it. *)

val queues : t -> string list
(** [queues g] is [["COMPUTE:0"; "COPY:0"]], one channel each. The parts of a
    submission on one queue run in array order: a part starts once the parts
    before it on its queue completed, and reads what they wrote. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion g] is [`Store]: [g]'s channels write its timeline word. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on g c] is [true] for [`Store] and [`Host]: [g]'s channels wait on
    any 64-bit word [g] maps, whoever writes it. It is [false] for [`Object]. *)

val max_waits : t -> int
(** [max_waits g] is [256]: a submission's channels take that many waits, for
    which [rig_nv_room] keeps room. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks g] is [`Returns]: [rig_nv_room] and [rig_nv_submit] store to memory
    and never block. *)

val maps_host : t -> bool
(** [maps_host g] is [true] iff [g]'s path maps host memory ([map_host] of
    {!type-path}). *)

type capability = Rig_nv_abi.Gpu.t
(** The type for what compiled code needs from a device. *)

val capability : t -> capability
(** [capability g] is [g]'s GPU as its formats depend on it: the classes and
    counts its path read, the shared and local memory windows [g] set on its
    compute channel, and [local], which grows the local memory [g]'s compute
    channel gives kernels ({!Rig_nv_abi.Gpu.field-local}). A growth takes effect
    at the next submission that uses ["COMPUTE:0"], before its parts; the memory
    it replaces stays allocated until the work before that submission completed.
    After {!stop}, [local] is [Error]. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Rig_nv_abi.Gpu.key}. *)

(** {1:memory Memory} *)

type region
(** The type for memory a device's work addresses: an allocation of the device,
    host memory it maps, or another device's memory it maps. *)

val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
(** [alloc g kind n] is [Some r] with [r] [n] new bytes:
    - [`Device], GPU memory, which the host does not address;
    - [`Pinned], host memory that the GPU reads and writes coherently with the
      host;
    - [`Mapped], GPU memory that the host also addresses, through the GPU's
      memory BAR, while the BAR has room; then [`Pinned] memory. The host's
      stores to it reach the GPU before the work of any later submission of [g]
      reads it.

    It is [None] if the GPU or the host lacks the memory.

    Raises [Invalid_argument] if [n < 1]. *)

val free : t -> region -> unit
(** [free g r] gives back [r], an allocation or a mapping {!map_peer} or
    {!map_host} gave. Freeing a mapping ends only that region: the memory it
    maps, and other regions over it, stay. The caller frees [r] once no work
    that uses it runs.

    Raises [Invalid_argument] if [r] is not a region of [g], is the timeline
    word, or was freed. *)

val address : region -> int option
(** [address r] is [Some a], [a] the address of [r]'s first byte in the GPU's
    virtual address space, below [2{^40}]. *)

val handle : region -> nativeint
(** [handle r] is {!address}[ r] as a [nativeint]: the device names memory by
    address. *)

val host : region -> int option
(** [host r] is [Some a], [a] the host address of [r]'s first byte, if the host
    addresses [r], and [None] for [`Device] memory. *)

val peer : t -> t -> bool
(** [peer g g'] is [true] iff {!map_peer}[ g g'] maps [`Device] memory of [g']:
    if the same path opened both devices and it reaches [g']'s GPU memory from
    [g]'s GPU, as the path of two GPUs with peer access does. *)

val map_peer : t -> t -> region -> region option
(** [map_peer g g' r] is [Some r'] with [r'] a new region of [g] over the memory
    of [g']'s region [r], if [g]'s work can address it: if the same path opened
    both devices and it maps [r]'s memory for [g]'s GPU, as the path of two GPUs
    with peer access does. It is [None] otherwise.

    Raises [Invalid_argument] if [g'] is [g], or if [r] is no region of [g'] or
    was freed. *)

val map_host : t -> int -> int -> region option
(** [map_host g a n] is [Some r] with [r] the [n] bytes of host memory at [a],
    mapped for [g]'s GPU, or [None] if its path maps no host memory
    ({!maps_host}) or refuses these pages. The path maps whole pages, and keeps
    the pages mapped until every region over them is freed. The host memory must
    stay mapped until [r] is freed.

    Raises [Invalid_argument] if [n < 1]. *)

(** {1:images Images} *)

type image
(** The type for cubins a device loaded. *)

val image :
  t ->
  string ->
  ( [ `Loaded of image | `Place of int * (region -> image * string) ],
    string )
  result
(** [image g bin] is [Ok (`Place (n, lay))] for the cubin [bin], whose image is
    [n] bytes ({!Rig_nv_abi.Cubin.size}). The caller allocates a region [r] of
    [`Device] memory of [g] of at least [n] bytes; [lay r] is the loaded cubin
    over [r] and the [n] bytes of its image, relocated for [r]'s address, which
    the caller writes to [r]'s start, such as with a [`Copy] part, before work
    runs its kernels. [image] makes nothing on [g], and [lay] calls no driver
    function and raises nothing. [r] stays the caller's: it frees [r] after
    {!unload}.

    The device invalidates its compute engine's instruction cache at its next
    submission that uses ["COMPUTE:0"] after [lay], before its parts: a cubin
    may load where another one's code was.

    The result is [Error msg] if [bin] is no cubin
    ({!Rig_nv_abi.Cubin.of_string}). *)

val entry : image -> string -> int option
(** [entry c f] is [Some a], [a] the address of the first instruction of the
    kernel [f] of [c], or [None] if [c] has no kernel [f]. The cubin's image
    starts at [a] minus the kernel's code offset ({!Rig_nv_abi.Cubin.kernel}).

    Raises [Invalid_argument] if [c] was unloaded. *)

val unload : t -> image -> unit
(** [unload g c] ends [c]: its code region goes back to the caller, who frees
    it. The caller unloads it once no work that runs its kernels runs, and never
    after {!stop}: an image of a stopped device needs no ending.

    Raises [Invalid_argument] if [c] is another device's or was unloaded. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word g] is [g]'s timeline word: eight bytes of host memory holding, as an
    unsigned 64-bit integer in the host's byte order, the last value [v] such
    that the work of every value up to [v] completed. [g]'s channels write it
    with one 64-bit store, after waiting for their work and making its writes
    visible to the system; it never decreases. Other devices may map it and wait
    on it. It is never freed: another device's work may still read it after [g]
    is stopped or collected. *)

val signaled : t -> int
(** [signaled g] is the value in {!word}, read with acquire order: the work of
    every value up to it completed, and its writes are visible to the reader. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep g ~seen ~still_ms] returns once [g]'s timeline word differs from
    [seen], at once if it already does, and at the latest after [still_ms]
    milliseconds; [still_ms >= 0]. Under a hang bound it may return earlier,
    when the bound's clock runs out. It reads the word and the channels' error
    notifiers every millisecond, and the multiprocessors' errors and the path's
    {!field-check} as it starts and once a notifier holds an error. It lets
    other domains run while it waits.

    Raises {!Fault} with the report if [g]'s work faulted. If the path bounds
    progress ([hang_ms] is [Some n]), it also raises {!Fault} once work is
    outstanding and the word has not moved for [n] milliseconds. That clock runs
    while a value given is above the word, from the later of the word's last
    move and the first [sleep] after the device was idle, as [sleep] observes
    them: an idle device never hangs, and the report may come late but never
    early. *)

(** {1:work Work}

    The C functions [Rig] submits with, which [rig_nv.h] declares. *)

val room_entry : nativeint
(** [room_entry] is the address of [rig_nv_room]. It answers whether the
    device's rings take a submission's parts now ([RIG_FITS]), once one of the
    device's values is reached ([RIG_LATER]), or never ([RIG_NEVER]): for parts
    that exceed the device's empty rings, more than 65,535 parts, or a part the
    device does not run. It reads the timeline word first, so [RIG_LATER] means
    a value the device was given is not yet reached. *)

val submit_entry : nativeint
(** [submit_entry] is the address of [rig_nv_submit]. It hands over parts that
    [rig_nv_room] answered [RIG_FITS] for as the device's value [v], the value
    after the last one the device was given, and calls no function of the OCaml
    runtime. The work runs after every earlier value of the device and after the
    waits, its parts on one queue in array order ({!queues}). Once it completed,
    the timeline word holds [v]. A submission of no parts writes [v] after its
    waits and after every earlier value.

    A [RIG_WORD] wait names an aligned 64-bit word at an address below [2{^40}]
    that the device's work addresses, and a value [w]. It holds the work back
    until the word holds at least [w], compared circularly: [x] is at least [w]
    iff [x - w], as a signed 64-bit integer, is not negative. *)

val self : t -> nativeint
(** [self g] is the address of [g]'s state, the [self] argument of [rig_nv_room]
    and [rig_nv_submit] for [g]. It is valid while the process runs: the state
    holds the {!word}, which other devices may read after [g] is gone, so
    neither is ever freed. *)

(** {1:loss Loss} *)

exception Fault of string
(** [Fault why] reports that a device's work faulted or its GPU failed: [why] is
    the RM's or the path's report. *)

val stop : t -> unit
(** [stop g] stops [g] for good, never waiting for its work. It ends its path's
    registration of [g]'s channels ({!field-unregister}) and frees them, and the
    RM preempts what they run; then it asks the path to stop ({!field-stop}).

    Once none of [g]'s work runs, the timeline word holds the last value
    [rig_nv_submit] was given, written with release order, so work of other
    devices that waits on it runs on. The word holds it before [stop] returns if
    the RM freed the channels, the path answered [`Stopped] or the RM had
    stopped the channels on a fault; otherwise the channels' own releases write
    it as their work ends. [g]'s memory goes back to the path in the first two
    cases.

    [g]'s images need no {!unload} after [stop]: an image holds nothing of [g]
    but its code region, which the caller frees. *)

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
