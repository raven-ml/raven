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
    let run = Rig.Submission.Run.make () in
    let p = Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||] in
    Rig.wait g (Rig.Point.value p)
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
    [RIG_FAILED] instead, once it makes a CUDA call (a submission of no parts
    and no waits makes none), and the [graph] function of {!val-capability}
    answers [Error] with the context's error.

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

(** {1:driver Driver} *)

include Rig_edge.Driver with type t := t

val capability : t -> Rig_cuda_abi.t
(** [capability g] is [g]'s ABI record, which its facts carry under
    {!Rig_cuda_abi.key}: the functions of the CUDA library [g] was opened with,
    and the maker of graphs in [g]'s context. *)

(** {1:this This driver}

    {t
      | Fact | Value |
      |------|-------|
      | [arch] | The GPU's compute capability, as ["sm_89"]. |
      | [budget] | The GPU's memory, in bytes. |
      | [queues] | ["COMPUTE:0"] runs [Fill], [Copy], [Launch]; ["COPY:0"] runs [Fill], [Copy]. |
      | [completion] | [Store]: the streams write the word. |
      | [waits] | [stores] and [hosts]; no [objects]; [most] is [max_int]. |
      | [may_block] | [true]: the submit calls CUDA, which may block. |
      | [maps_host] | The GPU's [CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED]. |
      | [capability] | {!capability}, under {!Rig_cuda_abi.key}. |
      | [word] | Eight bytes of page-locked host memory. |
    }

    {b Waits.} The streams wait on any aligned 64-bit word the device's work
    addresses, whoever writes it. A submission's waits go to its stream in
    batches, as many as it carries.

    {b Memory.} A region is an allocation of the device, host memory it maps,
    or another device's memory it maps. {!alloc} answers:
    - [Device], GPU memory, which the host does not address;
    - [Pinned], page-locked host memory that the host and every CUDA device
      address ([cuMemHostAlloc], portable and mapped). It is not
      write-combined, so the host reads it as fast as other memory;
    - [Mapped], [None]: CUDA maps no GPU memory for the host.

    An allocation is [None] if CUDA refuses it, for lack of memory or otherwise.
    {!alloc} and {!map_host} raise [Invalid_argument] for fewer than one byte.

    A region's {!locate} [address] is the address of its first byte in the
    process's unified address space, which every CUDA device's work uses. Its
    [handle] is the address by which CUDA names its allocation: its device
    address for GPU memory, its host address for host memory. Its [host] is
    that host address for host memory, and [None] for GPU memory.

    {!free} frees an allocation, and ends a region that {!map_peer} or
    {!map_host} gave. CUDA may wait for all of the GPU's work before it frees an
    allocation. A {!map_host} region whose unregistration CUDA refuses, because
    [g]'s context failed, keeps the pages locked, and every later {!map_host}
    that overlaps them is [None]. The free of the word ends a stopped [g]: it
    destroys the streams a {!stop} that found work running left, and the GPU
    then opens again.

    {!peer}[ g g'] is [true] iff CUDA gives [g]'s GPU access to the GPU memory
    of [g']'s, which {!peer} then enables for the pair: always for two devices
    of one GPU. {!map_peer} maps host memory always, and GPU memory where CUDA
    gives that access, which it then enables for the pair. The free of a
    {!map_peer} region ends only that region.

    {!map_host}[ g a n] page-locks the [n] bytes at [a] for every CUDA device
    and maps them. It is [None] where [g] does not map host memory, or where
    CUDA refuses to page-lock them: CUDA refuses read-only memory and memory
    whose pages another page-locked range shares. Page-locking is the process's.
    A range inside one that {!map_host} page-locked counts against it: the pages
    stay locked until every region {!map_host} gave over them, on any device, is
    freed. A range that shares a page with one without lying inside it is
    [None]. Memory that CUDA page-locked for another owner, such as an
    allocation of {!alloc} or of another library, is mapped as it is,
    uncounted, if the range lies inside one of its allocations ([None]
    otherwise), and must stay page-locked until the region is freed. The
    region's [address] is the one CUDA gives for the memory
    ([cuMemHostGetDevicePointer]), which every device's work uses under unified
    addressing.

    {b Images.} {!val-image}[ g bin] is [Ok (Loaded m)] with [m] the CUDA module
    of [bin], a cubin, a fatbin or PTX text, which CUDA compiles for the GPU.
    CUDA holds the code itself: {!val-image} places every function's code
    before it returns, whatever [CUDA_MODULE_LOADING] asks, so {!entry} and a
    launch place none. CUDA may wait for all of the GPU's work before it loads
    [bin], and {!val-image} lets other domains run meanwhile. The result is
    [Error msg] with CUDA's error if CUDA refuses [bin], for instance a cubin
    for another GPU, or lacks the memory for its code.

    {!entry}[ m f]'s [code] is the [CUfunction] of the kernel [f] of [m], which
    compiled code passes to [cuLaunchKernel]. Its [launch] is C memory holding
    that function and its limits: the most threads a group of it has
    ([CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]), and the most dynamic shared
    memory a group of it takes, as much as a block of the GPU can have (its
    opt-in maximum, [CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN]),
    less [f]'s static shared memory. Both are valid until [m] is unloaded, and
    a second {!entry} for [f] gives the same.

    {!unload} may wait, in CUDA, for all of the GPU's work, and lets other
    domains run meanwhile.

    {b Timeline.} The word holds its value as an unsigned 64-bit integer in the
    host's byte order. The streams write it after a fence that makes the work's
    writes visible ([cuStreamWriteValue64] with
    [CU_STREAM_WRITE_VALUE_DEFAULT], a system-wide memory fence before the
    write). {!signaled} reads it with acquire order: the work of every value up
    to its answer completed, and its writes are visible to the reader.

    {!sleep} asks CUDA each millisecond whether [g]'s streams met an error
    ([cuStreamQuery]), so it finds a fault at most a millisecond after CUDA
    reports it, and raises {!exception-Fault} with CUDA's error. It lets other
    domains run while it waits.

    {b Work.} The edge's [struct rig_driver] holds the C functions
    [rig_cuda_room], [rig_cuda_submit] and [rig_cuda_commit], which
    [rig_cuda.h] declares. The device's C state holds its word, which other
    devices may read after [g] is gone, so neither is ever freed.

    [rig_cuda_room] answers [RIG_NEVER] for a part that is no fill, copy or
    launch, a fill with ring units or segment bytes, a part on no queue of the
    device, a launch on queue [1], or a launch whose block has a size of [0]
    along an axis, more groups or threads along an axis than the GPU allows
    ([CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X] to [_Z],
    [CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X] to [_Z]: on an [sm_89], [2{^31} - 1]
    groups along x and [65535] along y and z), or more threads per group or
    dynamic shared memory than its function's limits; and [RIG_FITS]
    otherwise: CUDA's streams take any amount of work, and a submit that finds
    a stream full waits for earlier work to free it.

    [rig_cuda_submit] is called without the domain lock, and calls no function
    of the OCaml runtime. It encodes the parts on the device's streams as the
    value [v] after the last one the device was given:
    - A part on queue [0] runs on the stream ["COMPUTE:0"], on queue [1] on
      ["COPY:0"]. Parts on one queue run in array order; parts on two queues
      that [after] does not order may run at once.
    - A fill [f] enqueues its work on its queue's stream: the submit calls it as
      [f stream arg v], with the device's context current on the calling thread,
      as {!Rig_cuda_abi} states. Nothing bounds what a fill enqueues, so it
      declares no room.
    - A copy moves [copy.bytes] bytes between the handles of two regions of the
      device at their offsets. The ranges are apart: a copy between overlapping
      ranges writes undefined bytes.
    - A launch runs its function with [cuLaunchKernel] over the groups, threads
      and dynamic shared memory of its block, its parameters passed as one
      buffer ([CU_LAUNCH_PARAM_BUFFER_POINTER]). The submit copies them first
      and adds to each ref's 8 bytes its slot's address, and CUDA copies the
      result at the call.
    - Each wait holds the work back until the aligned 64-bit word at [at], which
      the device's work addresses, holds at least [value], compared circularly:
      [x] is at least [w] if [x - w], as a signed 64-bit integer, is not
      negative. Its kind is [RIG_WORD].
    - [handles] is ignored: CUDA's work names its memory by address.

    Encoding launches the parts on the device's stream. A commit writes the
    value with [cuStreamWriteValue64] ([rig_cuda_commit]); the device commits on
    its own every 64 values, and commits a value with a copy as it encodes it,
    since the copy hides the write's cost. A submission of no parts ends after
    its waits and after every earlier value. No part writes a timeline word, the
    device's or another's: only the devices' streams write their words, and a
    word the work wrote could move backwards or claim work that has not
    completed.

    Once every part is enqueued it answers [RIG_COMMITTED] if it committed [v],
    and [RIG_OK] otherwise. It answers [RIG_FAILED] with the step and the error
    of the first CUDA call that failed, a fill or a launch too, as
    ["running a fill: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was
     encountered"]. The parts enqueued before the failure may run; the others
    never do. A failed device stays failed, since CUDA may keep the context's
    error for the process: every later submit enqueues none of its parts and
    answers the same failure. The timeline word still reaches every value, so
    that work waiting on it elsewhere runs on: a failed submit, and every later
    one, writes [v] after its waits, every earlier value and the work enqueued
    for [v]. It writes nothing once the context failed, whose work no longer
    runs, or if CUDA refuses a call that orders the write. It may block while a
    stream is full, until the device's earlier work completes.

    [rig_cuda_commit], given [v], writes [v] with [cuStreamWriteValue64] on the
    stream the last value ended on, after the work of every value up to [v],
    unless a commit wrote [v] or a later value. It answers [RIG_FAILED] with the
    device's failure once a submit or a commit failed. It may block while a
    stream is full.

    {b Stopping.} {!stop} ignores [fault]. If [g]'s work no longer writes
    memory (the work of every value it was given completed, its streams are
    idle, or a fault ended the context's work), the word holds at least the last
    value the submit was given when {!stop} returns, and [g]'s streams are
    destroyed. Otherwise it commits the last value the submit was given, and the
    word reaches it once that work ends, unless it waits on a word of another
    device that never reaches its value; the GPU opens again once that work
    ends. A [graph] call of {!val-capability} in flight returns before {!stop}
    begins, and every later one answers [Error]. *)
