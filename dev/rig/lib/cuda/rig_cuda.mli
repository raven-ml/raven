(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs driven through the CUDA library.

    A device of this library is one GPU, opened on its primary context. It runs
    work on two streams, its queues ["COMPUTE:0"] and ["COPY:0"], and makes
    completion observable through its {e timeline word}, 64 bits of page-locked
    host memory: the word holds [v] once the work of every value up to [v]
    completed. Its caller numbers the work it hands over: the first submission
    is value [1], each next one the value after it, and value [v] runs after
    every value below it, whichever queue each ran on.

    A program opens a GPU through rig, which hands it work:
    {[
    let g =
      Rig.open_
        (module Rig_cuda)
        ~name:(Rig_cuda.device_name 0)
        (fun () -> Rig_cuda.open_ 0)
      |> Result.get_ok
    in
    let src = Rig.Buffer.create ~memory:Pinned g 4096 in
    let dst = Rig.Buffer.create g 4096 in
    let copy =
      {
        Rig.Submission.queue = "COPY:0";
        after = [||];
        work = Copy { src; dst };
      }
    in
    let s = Rig.Submission.make ~reads:0 ~writes:0 g [| copy |] in
    Rig.wait g
      (Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||]))
    ]}

    {b The CUDA library} ([libcuda]) is loaded at the first call of {!count} or
    {!open_}, by its standard name from the platform's library search path:
    [libcuda.so.1] (or [libcuda.so]) on Linux, [nvcuda.dll] on Windows,
    [libcuda.dylib] on macOS. A library of that name earlier on the path is
    loaded in its place, for the whole process. Building and linking this
    library needs no CUDA installation.

    {b Requirements.} A CUDA library of CUDA 12.4 or later (driver 550). The
    device writes its timeline from its streams with 64-bit stream memory
    operations, and its memory and the host's share one address space
    (unified addressing). A GPU without either does not open; there is no
    fallback.

    {b Other CUDA libraries} in the process share the primary context. Every
    call that needs a current context makes the device's current on the calling
    thread and restores the thread's own before returning.

    {b Faults.} CUDA reports a fault of a device's work, such as an illegal
    address or a kernel ended by a display's watchdog, at a later call. Work
    that runs long is no fault: only CUDA's report is ({!sleep}). A fault leaves
    its error in the GPU's context for the process: every later call in the
    context answers it, and the GPU no longer opens (Error Handling,
    [CUDA_ERROR_ILLEGAL_ADDRESS]). CUDA then runs none of the context's queued
    work. The reference does not state this; it is observed with the 615 driver,
    and {!stop} relies on it.

    {b Errors.} A function that calls CUDA answers CUDA's refusal as its result
    ([None], [Error]) while [g]'s context is sound, and raises
    {!exception-Fault} with the context's error once a fault left one there. The
    functions that raise it are {!alloc}, {!map_peer}, {!map_host},
    {!val-image}, {!entry}, {!unload} and {!sleep}; the submit answers
    [RIG_FAILED] instead, and the [graph] function of {!val-capability} answers
    [Error] with the context's error. Misuse, such as a region of another
    device, raises [Invalid_argument].

    {b Domains.} Every value may be called from any domain, at the same time as
    others, with three exceptions. The C room and submit run one call at a time,
    in value order: their caller serialises them. {!stop} is called once, after
    every other call returned but the [graph] function of {!val-capability},
    which compiled code may call at any time and which a stop waits for; after
    it only {!free}, {!unload}, the [symbol] and [graph] functions of
    {!val-capability} and the release of a graph are called, also after the GPU
    opened again: the device's modules are in the GPU's primary context, which
    the process keeps. {!sleep} may run while another domain
    submits.

    {b References.}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}: Primary
      Context Management, Stream Management, Event Management, Stream Memory
      Operations ([cuStreamWaitValue64], [cuStreamWriteValue64],
      [cuStreamBatchMemOp], [CUstreamWaitValue_flags]), Memory Management
      ([cuMemHostAlloc], [cuMemHostRegister]), Unified Addressing, Peer Context
      Memory Access, Module Management, Graph Management
      ([cuGraphAddKernelNode], [cuGraphInstantiateWithFlags]), Error Handling.
*)

(** {1:opening Opening} *)

type t
(** The type for open CUDA GPUs: a GPU's primary context, its two streams and
    its timeline word. *)

val count : unit -> int
(** [count ()] is the number of GPUs CUDA sees: [0] if the CUDA library cannot
    be loaded or initialised. [CUDA_VISIBLE_DEVICES] decides which GPUs CUDA
    sees. *)

val device_name : int -> string
(** [device_name i] is the name of GPU [i]: ["CUDA"] for [0], ["CUDA:i"]
    otherwise.

    Raises [Invalid_argument] if [i < 0]. *)

val open_ : int -> (t, string) result
(** [open_ i] opens GPU [i], the [i]th by PCI address among the GPUs CUDA sees,
    whatever [CUDA_DEVICE_ORDER] says: it is the GPU [nvidia-smi] numbers [i]
    when CUDA sees every GPU. The device works on the GPU's primary context,
    with streams of its own.

    The result is [Error msg] if the CUDA library cannot be loaded or
    initialised, or is older than CUDA 12.4 (driver 550), if [i >= count ()],
    if the GPU lacks 64-bit stream memory operations or unified addressing,
    while a device of GPU [i] is open or was stopped while its work still ran
    and that work runs on, or with CUDA's error, such as the error a fault left
    in the GPU's context. A GPU has one device at a time: two would wait on each
    other's words through the context's shared hardware queues, ordering CUDA
    does not see.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:facts Facts} *)

val key : t Type.Id.t
(** [key] tells devices of this library apart from others'. *)

val arch : t -> string
(** [arch g] is the GPU's compute capability, as ["sm_89"]. *)

val budget : t -> int
(** [budget g] is the GPU's memory, in bytes. *)

val queues : t -> string list
(** [queues g] is [["COMPUTE:0"; "COPY:0"]], one stream each. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion g] is [`Store]: [g]'s streams write its timeline word. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on g c] is [true] for [`Store] and [`Host]: [g] waits in its streams
    on any 64-bit word it maps, whoever writes it. It is [false] for [`Object].
*)

val max_waits : t -> int
(** [max_waits g] is [max_int]: a submission's waits go to its stream in
    batches, as many as it carries. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks g] is [`May_block]: the submit calls CUDA, which may block. *)

val maps_host : t -> bool
(** [maps_host g] is [true] iff CUDA page-locks host memory for [g]'s GPU, as
    its [CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED] says. *)

type capability = Rig_cuda_abi.t
(** The type for what compiled code needs from a device. *)

val capability : t -> capability
(** [capability g] is [g]'s record: the functions of the CUDA library [g] was
    opened with, and the maker of graphs in [g]'s context. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Rig_cuda_abi.key}. *)

(** {1:memory Memory} *)

type region
(** The type for memory a device's work addresses: an allocation of the device,
    host memory it maps, or another device's memory it maps. *)

val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
(** [alloc g kind n] is [Some r] with [r] [n] new bytes:
    - [`Device], GPU memory, which the host does not address;
    - [`Pinned] and [`Mapped], page-locked host memory that the host and every
      CUDA device address ([cuMemHostAlloc], portable and mapped). It is not
      write-combined, so the host reads it as fast as other memory. CUDA maps no
      GPU memory for the host, so [`Mapped] is host memory too.

    It is [None] if CUDA refuses the allocation, for lack of memory or
    otherwise.

    Raises [Invalid_argument] if [n < 1]. *)

val free : t -> region -> unit
(** [free g r] gives back [r]: it frees an allocation, and ends a region that
    {!map_peer} or {!map_host} gave. The caller frees it once no work that uses
    it runs. CUDA may wait for all of the GPU's work before it frees an
    allocation. A {!map_host} region whose unregistration CUDA refuses, because
    [g]'s context failed, keeps the pages locked, and every later {!map_host}
    that overlaps them is [None]. The free of the {!word} ends a stopped [g]:
    it destroys the streams a {!stop} that found work running left, and the
    GPU then opens again.

    Raises [Invalid_argument] if [r] is another device's, or was freed. *)

val address : region -> int option
(** [address r] is [Some a], [a] the address of [r]'s first byte in the
    process's unified address space, which every CUDA device's work uses. *)

val handle : region -> nativeint
(** [handle r] is the address by which CUDA names [r]'s allocation: its device
    address for GPU memory, its host address for host memory. *)

val host : region -> int option
(** [host r] is [Some a], [a] the host address of [r]'s first byte, if [r] is
    host memory, and [None] for GPU memory. *)

val peer : t -> t -> bool
(** [peer g g'] is [true] iff CUDA gives [g]'s GPU access to the GPU memory of
    [g']'s, which [peer] then enables for the pair: always for two devices of
    one GPU. It is {!map_peer}'s answer for GPU memory.

    Raises [Invalid_argument] if [g'] is [g]. *)

val map_peer : t -> t -> region -> region option
(** [map_peer g g' r] is [Some r'] with [r'] a new region of [g] over the memory
    of [g']'s region [r], if [g]'s work can address it: always for host memory;
    for GPU memory if CUDA gives [g]'s GPU access to [g']'s, which [map_peer]
    then enables for the pair. It is [None] otherwise. {!free} of [r'] ends only
    [r']. [r'] is [r]'s memory: the caller frees [r'] before it frees [r].

    Raises [Invalid_argument] if [g'] is [g], or if [r] is no region of [g'] or
    was freed. *)

val map_host : t -> int -> int -> region option
(** [map_host g a n] is [Some r] with [r] the [n] bytes of host memory at [a],
    page-locked for every CUDA device and mapped, unless [g] maps no host memory
    ({!maps_host}) or CUDA refuses to page-lock them. CUDA refuses read-only
    memory and memory whose pages another page-locked range shares. The host
    memory must stay mapped until [r] is freed.

    Page-locking is the process's. A range inside one that {!map_host}
    page-locked counts against it: the pages stay locked until every region
    {!map_host} gave over them, on any device, is freed. A range that shares a
    page with one without lying inside it is [None]. Memory that CUDA
    page-locked for another owner, such as an allocation of {!alloc} or of
    another library, is mapped as it is, uncounted, if the range lies inside one
    of its allocations ([None] otherwise), and must stay page-locked until [r]
    is freed.

    [r]'s {!address} is the one CUDA gives for the memory
    ([cuMemHostGetDevicePointer]), which every device's work uses under unified
    addressing.

    Raises [Invalid_argument] if [n < 1]. *)

(** {1:images Images} *)

type image
(** The type for CUDA modules a device loaded. *)

val image :
  t ->
  string ->
  ( [ `Loaded of image | `Place of int * (region -> image * string) ],
    string )
  result
(** [image g bin] is [Ok (`Loaded m)] with [m] the module of [bin], a cubin, a
    fatbin or PTX text, which CUDA compiles for the GPU. CUDA holds the code
    itself: [image] places every function's code before it returns, whatever
    [CUDA_MODULE_LOADING] asks, so {!entry} and a launch place none. CUDA may
    wait for all of the GPU's work before it loads [bin], and [image] lets
    other domains run meanwhile. The result is [Error msg] with CUDA's error
    if CUDA refuses [bin], for instance a cubin for another GPU, or lacks the
    memory for its code. *)

val entry : image -> string -> int option
(** [entry m f] is [Some h], [h] the [CUfunction] of the kernel [f] of [m],
    which compiled code passes to [cuLaunchKernel], or [None] if [m] has no
    kernel [f]. [h] is valid until [m] is unloaded.

    Raises [Invalid_argument] if [m] was unloaded. *)

val unload : t -> image -> unit
(** [unload g m] unloads [m]. The caller unloads it once no work that runs its
    kernels runs. CUDA may wait for all of the GPU's work before it returns, and
    [unload] lets other domains run meanwhile.

    Raises [Invalid_argument] if [m] is another device's or was unloaded. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word g] is [g]'s timeline word: eight bytes of page-locked host memory
    holding, as an unsigned 64-bit integer in the host's byte order, the last
    value [v] such that the work of every value up to [v] completed. [g]'s
    streams write it after a fence that makes the work's writes visible
    ([cuStreamWriteValue64] with [CU_STREAM_WRITE_VALUE_DEFAULT], a system-wide
    memory fence before the write); it never decreases. Other devices may map it
    and wait on it. It lives until
    {!free}, which the caller calls once [g] is stopped, the word holds its
    last value, and no other device's work reads it. *)

val signaled : t -> int
(** [signaled g] is the value in {!word}, read with acquire order: the work of
    every value up to it completed, and its writes are visible to the reader. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep g ~seen ~still_ms] returns once [g]'s timeline word differs from
    [seen], at once if it does already, or after [still_ms] milliseconds;
    [still_ms] is not negative. It asks CUDA each millisecond whether [g]'s
    streams met an error ([cuStreamQuery]), so it finds a fault at most a
    millisecond after CUDA reports it. It lets other domains run while it waits,
    and may run while the submit does.

    Raises {!exception-Fault} with CUDA's error if [g]'s work met one. *)

(** {1:work Work} *)

val room_entry : nativeint
(** [room_entry] is the address of the C function [rig_cuda_room], in the shape
    [rig_room_fn] of [rig_edge.h], which [rig_cuda.h] declares. It answers
    [RIG_NEVER] for a part with words, ring units or segment bytes, or on no
    queue of the device, and [RIG_FITS] otherwise: CUDA's streams take any
    amount of work, and a submit that finds a stream full waits for earlier work
    to free it. *)

val submit_entry : nativeint
(** [submit_entry] is the address of the C function [rig_cuda_submit], in the
    shape [rig_submit_fn] of [rig_edge.h], which [rig_cuda.h] declares. It is
    called without the domain lock, and calls no function of the OCaml runtime.
    It hands over the parts as the value [v] after the last one the device was
    given:
    - A part on queue [0] runs on the stream ["COMPUTE:0"], on queue [1] on
      ["COPY:0"]. Parts on one queue run in array order; parts on two queues
      that [after] does not order may run at once.
    - A fill [f] enqueues its work on its queue's stream: the submit calls it as
      [f stream arg v], with the device's context current on the calling thread,
      as {!Rig_cuda_abi} states. Nothing bounds what a fill enqueues, so it
      declares no room.
    - A copy moves [copy_bytes] bytes between the handles of two regions of the
      device ({!handle}) at their offsets. The ranges are apart: a copy between
      overlapping ranges writes undefined bytes.
    - Each wait holds the work back until the aligned 64-bit word at [at], which
      the device's work addresses, holds at least [value], compared circularly:
      [x] is at least [w] if [x - w], as a signed 64-bit integer, is not
      negative. Its kind is [RIG_WORD].
    - [handles] is ignored: CUDA's work names its memory by address.

    The work runs after every earlier value of the device and after the waits;
    once it completed, the timeline word holds [v]. A submission of no parts
    writes [v] after its waits and after every earlier value. No part writes a
    timeline word, the device's or another's: only the devices' streams write
    their words, and a word the work wrote could move backwards or claim work
    that has not completed.

    It answers [RIG_COMMITTED] once every part is enqueued, or [RIG_FAILED] with
    the step and the error of the first CUDA call that failed, a fill too, as
    ["running a fill: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was
     encountered"]. The parts enqueued before the failure may run; the others
    never do. A failed device stays failed, since CUDA may keep the context's
    error for the process: every later submit enqueues none of its parts and
    answers the same failure. The timeline word still reaches every value, so
    that work waiting on it elsewhere runs on: a failed submit, and every later
    one, writes [v] after its waits, every earlier value and the work enqueued
    for [v]. It writes nothing once the context failed, whose work no longer
    runs, or if CUDA refuses a call that orders the write.

    It may block while a stream is full, until the device's earlier work
    completes. *)

val commit_entry : nativeint
(** [commit_entry] is the address of the device's commit, in the shape
    [rig_commit_fn] of [rig_edge.h]. Each value's hand-over writes its value and
    answers [RIG_COMMITTED]: [commit_entry] does nothing. *)

val self : t -> nativeint
(** [self g] is the address of [g]'s state, the first argument of
    [rig_cuda_room] and [rig_cuda_submit]. It is valid while the process runs: a
    device's C state holds its {!word}, which other devices may read after [g]
    is gone, so neither is ever freed. *)

(** {1:loss Loss} *)

exception Fault of string
(** The exception for a fault of a device's work, with CUDA's error. *)

val stop : t -> unit
(** [stop g] stops [g] for good, without waiting for [g]'s work. If [g]'s work
    no longer writes memory (the work of every value it was given completed, its
    streams are idle, or a fault ended the context's work), the timeline word
    holds at least the last value the submit was given when [stop] returns, so
    work of other devices that waits on it runs on, and [g]'s streams are
    destroyed. Otherwise the timeline word reaches the last value the submit
    was given once that work ends, unless it waits on a word of another device
    that never reaches its value; the GPU opens again once that work ends. It
    releases no region and no image: those end at {!free} and {!unload}, which
    may follow. A [graph] call of {!val-capability} in flight returns before
    [stop] begins, and every later one answers [Error]. After [stop], only
    {!free}, {!unload}, the [symbol] and [graph] functions of
    {!val-capability} and the release of a graph may be called on [g]. *)
