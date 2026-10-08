(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs driven through their resource manager.

    A device of this library is one NVIDIA GPU, which a {e path} opened: a
    library that reaches the GPU's resource manager (RM) one way, such as
    {!Device_nv_nvidia} through NVIDIA's kernel driver, and fills a {!path}
    record. The device itself is the same whichever path opened it.

    A device runs work on two {e channels}, the GPU's hardware queues: its
    queues ["COMPUTE:0"] and ["COPY:0"]. Its work is one sequence of
    {e submissions}, which the caller numbers [1], [2], …, the {e values} of the
    device's {e timeline}. The device makes each observable in its
    {e timeline word} ({!word}), 64 bits of host memory that hold [v] once the
    work of every value up to [v] completed, whichever channel ran it.

    A program that uses a GPU alone opens it, allocates and submits, then waits
    for the word:
    {[
    let g = Result.get_ok (Device_nv_nvidia.open_ 0) in
    let src = Option.get (Device_nv.alloc g `Pinned 4096) in
    let dst = Option.get (Device_nv.alloc g `Device 4096) in
    let copy = `Copy ((dst, 0), (src, 0), 4096) in
    let p = Device_nv.part g ~queue:"COPY:0" copy in
    match Device_nv.submit g ~v:1 ~waits:[||] ~handles:[||] [| p |] with
    | `Ok ->
        let rec wait () =
          let seen = Device_nv.signaled g in
          if seen < 1 then (
            Device_nv.sleep g ~seen ~still_ms:200;
            wait ())
        in
        wait ()
    | `Failed why -> prerr_endline why
    ]}

    {b Submissions.} A submission is the work of one value: {e parts}, each for
    one channel ({!part}), and the waits on other devices' words it starts
    after. The device writes each part into its channel's ring, a GPFIFO, after
    a wait for the value before it and before the release of its own value, and
    wakes the channel. A part is ring entries that compiled code wrote
    ({!Device_nv_abi.Gpfifo.entry}), or a copy the device encodes. Writing a
    submission makes no system call and calls no function of the OCaml runtime.

    {b Order.} A channel runs its work in order. The device makes a value's work
    start after the work of every value before it, and releases [v] into the
    word once its work completed: the release waits for the channel to be idle.
    Two submissions therefore never overlap on the GPU.

    {b Faults.} The RM stops a channel that faults, such as on a read its page
    tables refuse, and writes the error into the channel's notifier; the GPU's
    multiprocessors report their own errors to it. {!sleep} reads both, and
    raises {!Fault} with the report. Work that runs long is no fault: a wait
    lasts until the work ends.

    A function that calls the RM answers its refusal of the arguments as its
    result ([None], [Error]) and raises {!Fault} for any other failure.
    {!signaled}, {!free} and {!stop} never raise it.

    {b Domains.} Any domain may call any function, at the same time as others,
    with three exceptions. {!room} and {!submit} are called one at a time: the
    caller holds the device's {e turn} from {!room} to the end of {!submit}.
    {!stop} is called once, after every other call returned; after it only
    {!free} and the capability's [local] are called. {!sleep} may run while
    another domain submits.

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

(** {1:facts Facts} *)

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
(** [queues g] is [["COMPUTE:0"; "COPY:0"]], one channel each. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion g] is [`Store]: [g]'s channels write its timeline word. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on g c] is [true] for [`Store] and [`Host]: [g]'s channels wait on
    any 64-bit word [g] maps, whoever writes it. It is [false] for [`Object]. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks g] is [`Returns]: {!room} and {!submit} store to memory and never
    block. *)

type capability = Device_nv_abi.Gpu.t
(** The type for what compiled code needs from a device. *)

val capability : t -> capability
(** [capability g] is [g]'s GPU as its formats depend on it: the classes and
    counts its path read, the shared and local memory windows [g] set on its
    compute channel, and [local], which grows the local memory [g]'s compute
    channel gives kernels ({!Device_nv_abi.Gpu.field-local}). A growth takes
    effect at the next submission that uses ["COMPUTE:0"], before its parts; the
    memory it replaces stays allocated until the work before that submission
    completed. After {!stop}, [local] is [Error]. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Device_nv_abi.Gpu.key}. *)

val self : t -> nativeint
(** [self g] is the address of [g]'s state, the first argument of
    [device_nv_room] and [device_nv_submit]. It is valid while the process runs:
    the state holds the {!word}, which other devices may read after [g] is gone,
    so neither is ever freed. *)

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
      stores to it reach the GPU before the work of any later {!submit} of [g]
      reads it.

    It is [None] if the GPU or the host has not the memory.

    Raises [Invalid_argument] if [n < 1]. *)

val free : t -> region -> unit
(** [free g r] gives back the region [r]: an allocation, or a mapping
    {!map_peer} or {!map_host} gave, of which it ends only [r]. The caller frees
    it once no work that uses it runs.

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
    mapped for [g]'s GPU, unless its path refuses them. The path maps whole
    pages, and keeps the pages mapped until every region over them is freed. The
    host memory must stay mapped until [r] is freed.

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
    [n] bytes ({!Device_nv_abi.Cubin.size}). The caller allocates a region [r]
    of [`Device] memory of [g] of at least [n] bytes; [lay r] is the loaded
    cubin over [r] and the [n] bytes of its image, relocated for [r]'s address,
    which the caller writes to [r]'s start, such as with a [`Copy] part, before
    work runs its kernels. [image] makes nothing on [g], and [lay] calls no
    driver function and raises nothing. [r] stays the caller's: it frees [r]
    after {!unload}.

    The device invalidates its compute engine's instruction cache at its next
    submission that uses ["COMPUTE:0"] after [lay], before its parts: a cubin
    may load where another one's code was.

    The result is [Error msg] if [bin] is no cubin
    ({!Device_nv_abi.Cubin.of_string}). *)

val entry : image -> string -> int option
(** [entry c f] is [Some a], [a] the address of the first instruction of the
    kernel [f] of [c], or [None] if [c] has no kernel [f]. The cubin's image
    starts at [a] minus the kernel's code offset
    ({!Device_nv_abi.Cubin.kernel}).

    Raises [Invalid_argument] if [c] was unloaded. *)

val unload : t -> image -> unit
(** [unload g c] ends [c]: its code region goes back to the caller, who frees
    it. The caller unloads it once no work that runs its kernels runs.

    Raises [Invalid_argument] if [c] is another device's or was unloaded. *)

(** {1:work Work} *)

type part
(** The type for work for one channel of a device. *)

val part :
  t ->
  queue:string ->
  ?after:int array ->
  [ `Words of int array
  | `Fill of nativeint * nativeint * int * int
  | `Copy of (region * int) * (region * int) * int ] ->
  part
(** [part g ~queue ~after w] is the work [w] for [g]'s channel [queue], one of
    {!queues}. [after] (defaults to [[||]]) holds the indices, in the array
    given to {!submit}, of the parts of its submission that it runs after, each
    smaller than its own. Parts on one channel run in array order; parts on two
    channels that [after] does not order may run at once. [w] is:
    - [`Words ws], ring entries ({!Device_nv_abi.Gpfifo.entry}), each as two
      32-bit words of [ws], low first. Each names a segment of [g]'s memory,
      which the caller keeps unchanged until the work completes;
    - [`Copy ((dst, o), (src, o'), n)], on ["COPY:0"], a copy of the [n] bytes
      of [src] at offset [o'] to [dst] at offset [o], any two regions of [g],
      whose ranges do not overlap.

    Raises [Invalid_argument] if [queue] is not a queue of [g], if [w] is
    [`Fill _], which the device does not run, if [ws] has an odd length, if a
    copy is on ["COMPUTE:0"] or its range lies outside its region, if a region
    is of another device or was freed, or if an index of [after] is negative. *)

val room : t -> part array -> [ `Fits | `Later | `Never ]
(** [room g ps] is [`Fits] if [g]'s rings take [ps] now, [`Later] if they take
    them once one of [g]'s values is reached, and [`Never] if [ps] exceed [g]'s
    empty rings or are more than 65,535 parts. It reads the timeline word first,
    so [`Later] means a value [g] was given is not yet reached. Its C form,
    [device_nv_room], also answers [NX_NEVER] for a part {!part} refuses. *)

val submit :
  t ->
  v:int ->
  waits:([ `Word | `Object ] * int * int) array ->
  handles:nativeint array ->
  part array ->
  [ `Ok | `Failed of string ]
(** [submit g ~v ~waits ~handles ps] hands over [ps], which {!room} answered
    [`Fits] for, as [g]'s value [v], the value after the last one [g] was given.
    Each wait [(`Word, a, w)] holds the work back until the aligned 64-bit word
    at address [a], which [g]'s work addresses, holds at least [w], compared
    circularly: [x] is at least [w] if [x - w], as a signed 64-bit integer, is
    not negative. The work runs after every earlier value of [g] and after the
    waits; once it completed, the timeline word holds [v]. A submission of no
    parts writes [v] after its waits and after every earlier value. [handles] is
    ignored: the device's work names its memory by address.

    The result is [`Ok] once every part is in its channel's ring and the
    channels were woken: stores to this machine's memory cannot fail.

    Raises [Invalid_argument] if [v] is not the value after the last one, if
    {!room} does not answer [`Fits] for [ps], if a part is another device's, if
    a part's [after] names a part at or after its own index, if [waits] holds
    more than 256 waits, or if a wait is [`Object]: the device waits only for
    words to reach a value. *)

val room_entry : nativeint
(** [room_entry] is the address of the C function [device_nv_room], {!room} for
    C, which [device_nv.h] declares. *)

val submit_entry : nativeint
(** [submit_entry] is the address of the C function [device_nv_submit],
    {!submit} for C, which [device_nv.h] declares. It calls no function of the
    OCaml runtime. *)

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
    [seen], at once if it does already, or after [still_ms] milliseconds,
    whichever comes first; [still_ms] is not negative. It reads the word and the
    channels' error notifiers each millisecond, and the RM's report of the
    multiprocessors' errors once per call. It lets other domains run while it
    waits, and may run while {!submit} does.

    Raises {!Fault} with the report if [g]'s work faulted. *)

(** {1:loss Loss} *)

exception Fault of string
(** The exception for a fault of a device's work, with the RM's report. *)

val stop : t -> unit
(** [stop g] stops [g] for good, never waiting for its work: it ends its path's
    registration of [g]'s channels ({!field-unregister}), then frees them, and
    the RM preempts what they run. Once none of [g]'s work runs, the timeline
    word holds the last value {!submit} was given, written with release order,
    so work of other devices that waits on it runs on: before [stop] returns if
    the RM freed the channels or had stopped them on a fault, and otherwise by
    the channels' own releases as their work ends. After [stop], only {!free} is
    called on [g], and it raises no {!Fault}. *)

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
      (** [free ~parent obj] frees [obj] and every object under it. *)
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
    {!Device_nv_abi.Gpu.t}. *)

type 'm memory = {
  address : int;  (** The GPU address of its first byte, below [2{^40}]. *)
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
          if the GPU or the host has not the memory:
          - [`Gpu], GPU memory, which the host does not address;
          - [`Bar], GPU memory that the host also addresses, through the GPU's
            memory BAR, mapped uncached;
          - [`System], host memory that the GPU reads and writes coherently with
            the host. *)
  map_host : int -> int -> 'm memory option;
      (** [map_host a n] is the [n] bytes of host memory at [a], mapped for the
          GPU, or [None] if the path refuses them. The path maps whole pages,
          once per process for the pages of one range, and keeps them mapped
          until every memory it gave over them is freed. *)
  reaches : 'm path -> bool;
      (** [reaches p'] is [true] iff this GPU's work addresses the GPU memory of
          the GPU [p'] reaches, another of this path. *)
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
    [610] and [615], if the RM refuses an object, or if [p] has not the memory.
    A failed [make] frees what it allocated; [p]'s own objects stay [p]'s, and
    another [make] may use them once this one's device is stopped or failed. *)

val is_gpu : vendor:int -> class_:int -> bool
(** [is_gpu ~vendor ~class_] is [true] iff a PCI function of vendor [vendor] and
    24-bit class code [class_] is an NVIDIA GPU: NVIDIA's vendor [0x10de] and a
    display controller, base class [0x03]. Every path numbers a machine's GPUs
    [0], [1], … in bus order among such functions. *)
