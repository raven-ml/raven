(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs, driven through the CUDA library.

    A device of this library is one GPU, opened on its primary context with
    two CUDA streams of its own, its queues ["COMPUTE:0"] and ["COPY:0"].
    ["COMPUTE:0"] runs launches, copies and fills, and ["COPY:0"] runs copies
    and fills. The device's timeline word lies in page-locked host memory,
    and the streams write it themselves. A GPU has one device at a time.

    A kernel reaches the GPU in a cubin, a fatbin or PTX text.
    {!Rig.Image.load} loads it as a CUDA module and places every function's
    code on the GPU. Each {!Rig.submit} enqueues its parts on the streams and
    returns without waiting for them, a launch with [cuLaunchKernel] and a
    copy with [cuMemcpyAsync]. CUDA events order a value's work after the
    previous value's work, on both streams. To commit a value, the device
    enqueues a write of the value into the word ([cuStreamWriteValue64]),
    which runs once the work before it has completed. It commits when a wait
    asks for a value, and on its own ({!section-work}). Work that waits for
    another device's point waits on the GPU, as a stream wait on that
    device's word, where the facts allow it ({!section-facts}).

    Given a device [g] that [Rig.open_ (Rig_cuda.gpu 0)] opened, a buffer [x]
    on it that holds [n] floats, where [n] is a multiple of 256, and a cubin
    [bin] compiled from
    {v
    extern "C" __global__ void twice(float *x) {
      x[blockIdx.x * blockDim.x + threadIdx.x] *= 2;
    }
    v}
    this doubles [x] in place:
    {[
    let image = Rig.Image.load g bin |> Result.get_ok in
    let twice =
      Rig.Submission.(launch ~refs:[| write 0 ~at:0 |] image "twice" ~params:8)
    in
    let s = Rig.Submission.make g [| twice |] in
    let run = Rig.Submission.Run.make () in
    let b = Rig.Submission.block s 0 in
    Rig.Submission.Run.groups run b (n / 256) 1 1;
    Rig.Submission.Run.threads run b 256 1 1;
    Rig.Point.wait (Rig.submit s ~run ~buffers:[| x |])
    ]}
    The kernel's parameters are its one pointer, and the ref puts the address
    of [x]'s first byte in those 8 bytes.

    A device lives until {!Rig.close} or a loss, and a fault of its work ends
    the GPU's use for the rest of the process ({!section-failure}). Other CUDA
    libraries in the process share the primary context. Every call that needs
    a current context makes the device's current on the calling thread, and
    restores the thread's previous context before it returns. Any domain may
    call any function of this module, at the same time as other domains. A
    function that waits for CUDA lets other domains run while it waits.

    {!Rig_edge} states how rig drives a device through its driver, and
    {!Rig_edge.section-calls} how it hands the driver work. *)

(** {1:opening Opening}

    A device needs:
    {ul
    {- the CUDA library of CUDA 12.4 or later (driver 550);}
    {- a GPU with 64-bit stream memory operations, with which its streams
       write and wait on timeline words;}
    {- unified addressing, under which the GPU's memory and the host's share
       one address space.}}

    The CUDA library ([libcuda]) is loaded at the first call of {!count} or
    of an opener's [make], by its standard name from the platform's library
    search path: [libcuda.so.1] (or [libcuda.so]) on Linux, [nvcuda.dll] on
    Windows, [libcuda.dylib] on macOS. A library of that name earlier on the
    path is loaded in its place, for the whole process. A load that fails is
    tried again at the next call. Building and linking this library needs no
    CUDA installation. *)

type t
(** The type for open devices: a GPU's primary context, two streams and a
    timeline word. *)

val count : unit -> int
(** [count ()] is the number of GPUs CUDA sees, or [0] if the CUDA library
    cannot be loaded or initialised. [CUDA_VISIBLE_DEVICES] decides which
    GPUs CUDA sees. The list is read at the first call whose load and listing
    both succeed, and kept for the process. *)

val gpu : int -> t Rig_edge.opener
(** [gpu i] is the opener of GPU [i], the [i]th by PCI address among the GPUs
    CUDA sees, whatever [CUDA_DEVICE_ORDER] says. When CUDA sees every GPU,
    it is the GPU [nvidia-smi] numbers [i]. The opener is named ["CUDA"] for
    [0] and ["CUDA:i"] otherwise. Its [make] opens a device on the GPU's
    primary context, with streams of its own.

    [make] is [Error] with the reason:
    {ul
    {- while a device of GPU [i] is open, or a stopped device's work on GPU
       [i] still runs;}
    {- if the CUDA library cannot be loaded or initialised, or lacks a
       function of CUDA 12.4;}
    {- if [i >= count ()], or the GPU lacks what {!section-opening}
       requires;}
    {- with CUDA's error, such as the error of a fault earlier in the
       process.}}

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:driver The driver}

    rig calls these functions. On CUDA, {!alloc}, {!free}, {!locate},
    {!map_host}, {!peer} and {!map_peer} behave as in {!section-memory};
    {!val-image}, {!entry} and {!unload} as in {!section-code}; {!sleep} and
    {!stop} as in {!section-failure}; and the C functions as in
    {!section-work}. *)

include Rig_edge.Driver with type t := t

val capability : t -> Rig_cuda_abi.t
(** [capability g] is [g]'s record for compiled code, which [g]'s facts
    declare ({!Rig_cuda_abi.declare}): the functions of the CUDA library [g]
    was opened with, and the maker of graphs in [g]'s context. *)

(** {1:facts Facts}

    {t
      | Fact             | Value                                            |
      |------------------|--------------------------------------------------|
      | [arch]           | ["sm_XY"], the GPU's compute capability          |
      | [budget]         | The GPU's memory, in bytes                       |
      | [queues]         | ["COMPUTE:0"] and ["COPY:0"]                     |
      | [copy_engine]    | [Some 1], ["COPY:0"]                             |
      | [completion]     | [Store]                                          |
      | [waits]          | [stores] and [hosts]; [most] [max_int]           |
      | [may_block]      | [true]                                           |
      | [hang_ms]        | [Some 30_000], or [None] under a watchdog        |
      | [maps_host]      | [CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED]    |
      | [host_addresses] | [false]                                          |
      | [capability]     | {!Rig_cuda_abi.t} ({!val-capability})            |
      | [word]           | Eight bytes of page-locked host memory           |
    }

    [arch] is ["sm_89"] for compute capability 8.9. ["COMPUTE:0"] runs
    [Fill], [Copy] and [Launch], and ["COPY:0"] runs [Fill] and [Copy]. The
    streams write the word, and the host does not address [Device] memory.
    The word holds an unsigned 64-bit integer in the host's byte order.

    The streams wait on any aligned 64-bit word the device's work addresses,
    whoever writes it. A submission's waits go to its stream in batches, as
    many as it carries. [hang_ms] is [None] where a display's watchdog
    bounds the GPU's kernels ([CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT]),
    and [Some 30_000], 30 s, where nothing does ({!section-failure}). *)

(** {1:work Work}

    The device's [struct rig_driver] holds [rig_cuda_room], [rig_cuda_submit],
    [rig_cuda_commit] and [rig_cuda_settle], which [rig_cuda.h] declares. A
    part on queue [0] runs on ["COMPUTE:0"], one on queue [1] on
    ["COPY:0"].

    The device commits a value by enqueueing a write of it into the word
    ([cuStreamWriteValue64]) on the stream the last value ended on. The
    write follows a system-wide memory fence ([CU_STREAM_WRITE_VALUE_DEFAULT]),
    so the work's writes are visible once the word shows the value. A
    commit ({!Rig_edge.section-calls}) writes nothing if a commit already
    wrote that value or a later one. [rig_cuda_submit] also commits, and
    returns [RIG_COMMITTED], in two cases:
    {ul
    {- a value 64 values past the last commit, so the device commits each
       value at most 64 values late;}
    {- a value that holds a copy, as it is enqueued, so that a wait for the
       copy waits for no work enqueued after it.}}

    A fill [f] is called as [f stream arg v], with the device's context
    current on the calling thread ({!Rig_cuda_abi}). A copy moves
    [copy.bytes] bytes between the handles of two regions of the device, at
    their offsets. A part must not write a timeline word, the device's or
    another's. Only the devices' streams write their words, and a word that
    work wrote could move backwards or show work that has not completed.

    A wait is a [RIG_WORD]. It holds the work back until the aligned 64-bit
    word at [at] holds at least [value], compared circularly, in the sense
    that [x] is at least [w] if [x - w], as a signed 64-bit integer, is not
    negative.

    [rig_cuda_room] never returns [RIG_LATER], since a stream takes any
    amount of work. [rig_cuda_submit] and [rig_cuda_commit] instead block
    while a stream is full, until the device's earlier work frees it. A
    timed submit that measures blocks for its round trips, tens of
    microseconds, and one timed 1,024 values after an unconverted value
    waits for that value.
    While a profile is taken, [rig_cuda_submit] records a timing event on
    each stream a value's work enters, after the value's waits, and one
    after its work on the stream it ends on, into one of 1,024 stamps,
    whose events it makes at the stamp's first use. It waits for the value
    1,024 values before it, whose events it reuses, if that value was
    timed.
    [rig_cuda_settle], or a later timed [rig_cuda_submit], converts the
    earliest start and the end to the host's clock, in value order, with
    each no earlier than the end of the value converted before it, since a
    value's work starts once every earlier value's work has completed. A
    failed conversion loses the device.

    The device measures its clock against the host's when it opens, and
    again in a timed [rig_cuda_submit] that finds the last measure 8 s old.
    A measure records a reference event on a stream of the device's own
    between two readings of the host's clock, asking CUDA until the event
    has completed, eight times, and keeps the closest; while that is over
    10 µs it tries again, up to sixteen times. An event's time is its
    elapsed time since the reference, a float of milliseconds, plus the
    host's time at the reference. A time is thus within half the closest
    round trip of the host's clock, 5 µs once a round trip within 10 µs is
    seen, plus the GPU timer's rate error since the measure, plus half a
    float step at its distance from the reference, half a microsecond
    within 8 s. [rig_cuda_settle] never waits for the
    device's work.

    {2:launches Launches}

    A launch runs its function with [cuLaunchKernel] over the groups,
    threads and dynamic shared memory of its block, and passes its
    parameters as one buffer ([CU_LAUNCH_PARAM_BUFFER_POINTER]).

    [rig_cuda_room] refuses a launch if:
    {ul
    {- it is on queue [1];}
    {- its block has a size of [0] along an axis;}
    {- it has more groups or threads along an axis than the GPU allows
       ([CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X] to [_Z],
       [CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X] to [_Z]);}
    {- it has more threads per group, or more dynamic shared memory, than
       its function's limits.}} *)

(** {1:memory Memory}

    A region is an allocation of the device, host memory it maps, or another
    device's memory it maps. {!alloc} returns:
    {ul
    {- for [Device], GPU memory, which the host does not address;}
    {- for [Pinned], page-locked host memory that the host and every CUDA
       device address ([cuMemHostAlloc], portable and mapped). It is not
       write-combined, so the host reads it through its caches;}
    {- for [Mapped], [None], since CUDA maps no GPU memory for the host.}}
    An allocation is [None] if CUDA refuses it, such as for lack of memory.
    {!alloc} and {!map_host} raise [Invalid_argument] for fewer than one
    byte.

    A region's {!locate} [address] is the address of its first byte in the
    process's unified address space, which every CUDA device's work uses. Its
    [handle] is the address by which CUDA names its allocation: the device
    address for GPU memory, the host address for host memory. Its [host] is
    that host address for host memory, and [None] for GPU memory.

    {!free} frees an allocation, and ends a region that {!map_peer} or
    {!map_host} gave.

    {!peer}[ g g'] is [true] if [g] and [g'] are devices of one GPU, and
    otherwise iff CUDA gave [g]'s GPU access to the GPU memory of [g']'s
    when the later of the two GPUs first opened. That open enables the
    access both ways where CUDA gives it, for the rest of the process.
    {!map_peer} maps host memory always, and GPU memory where {!peer} is
    [true]. Freeing a {!map_peer} region ends only that region. Peer access
    between two GPUs is untested.

    {!map_host}[ g a n] page-locks the [n] bytes at [a] for every CUDA device
    and maps them, at the address CUDA gives for the memory
    ([cuMemHostGetDevicePointer]). Page-locking belongs to the process, and
    the region's pages stay locked until every region {!map_host} gave over
    them, on any device, is freed. The result depends on what is already
    locked:
    {ul
    {- A range inside one that {!map_host} locked is mapped and counts
       against it.}
    {- A range inside one allocation that CUDA locked for another owner,
       such as an allocation of {!alloc} or of another library, is mapped as
       it is, uncounted. That memory must stay locked until the region is
       freed.}
    {- Any other range is locked, or [None] if CUDA refuses, as it does for
       read-only memory and for a range that overlaps locked memory without
       lying inside one of its ranges.}}
    It is also [None] where [g] does not map host memory ([maps_host]).

    On Linux, CUDA locks host pages for writing, which Linux refuses for a
    shared mapping of a file. A CUDA device therefore cannot borrow a region
    that an io device holds for writing through a shared mapping of a file
    ({!Rig_edge.Io.pages}), and copies to and from that region go through
    staging memory. *)

(** {1:code Code}

    CUDA may wait for all of the GPU's work before {!free} frees an
    allocation, and before {!val-image} or {!unload} loads or unloads a
    module.

    {!val-image}[ g bin] is [Ok (Loaded m)] with [m] the CUDA module of [bin]:
    a cubin, a fatbin, or PTX text, which CUDA compiles for the GPU. It places
    every function's code before it returns, whatever [CUDA_MODULE_LOADING]
    asks, so neither {!entry} nor a launch loads code later. Each function
    may take as much dynamic shared memory as a block of the GPU can have,
    less its static shared memory. The result is [Error] with CUDA's error if
    CUDA refuses [bin], for instance a cubin for another GPU, or lacks the
    memory for its code.

    {!entry}[ m f] is the entry of kernel [f] of [m]. Its [code] is the
    kernel's [CUfunction], which compiled code passes to [cuLaunchKernel].
    Its [launch] is C memory that holds the function and its limits: the most
    threads a group of it has ([CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]),
    and the most dynamic shared memory a group of it takes. Both are valid
    until [m] is unloaded, and a second {!entry} for [f] returns the same.
    Its [params] end at the kernel's last parameter
    ([cuFuncGetParamInfo]), and its [threads] are the most threads above. A
    launch hands CUDA those [params] bytes of its parameters, since CUDA
    refuses a buffer of another size.

    After {!unload}, no new graph may record the image's functions. Its CUDA
    module stays loaded until every graph that recorded one of its functions
    is destroyed ({!Rig_cuda_abi.section-graphs}). *)

(** {1:failure Failure and stopping}

    rig loses a device when CUDA reports a fault of its work, such as an
    illegal address or a kernel that a display's watchdog ended, at a later
    call. A fault leaves the process in an inconsistent state: every later
    CUDA call returns the fault's error, and to use CUDA again the process
    must be terminated and relaunched. So the GPU cannot be opened again.

    On a GPU whose kernels no display's watchdog bounds, rig also loses the
    device once its work has kept the word still for 30 s, even work that
    would have ended later.

    Before a fault, {!alloc}, {!map_host}, {!val-image} and {!entry} return
    CUDA's refusals as their result, [None], [Error] or [Absent], and
    {!unload} raises {!exception-Fault} if CUDA refuses to unload the module.
    After a fault, the calls of {!alloc}, {!map_host}, {!val-image},
    {!entry}, {!unload} and {!sleep} that call CUDA raise {!exception-Fault}
    with its error, and the others answer as before the fault, such as an
    {!entry} for a function already found or a {!sleep} once the word has
    moved. [rig_cuda_submit] and [rig_cuda_commit] then return [RIG_FAILED]
    once they call CUDA, and the [graph] function of {!val-capability}
    returns [Error]. {!free} and {!stop} ignore CUDA's errors and raise
    nothing.

    A [rig_cuda_submit] or [rig_cuda_commit] that fails returns [RIG_FAILED]
    with what it was doing and the error of the first CUDA call that failed,
    including one a fill made, as in ["running a fill:
    CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered"].
    The parts enqueued before the failure may run, and the others never do.

    A failed device stays failed, since CUDA may keep the context's error for
    the process. Every later [rig_cuda_submit] enqueues none of its parts,
    and every later [rig_cuda_submit] and [rig_cuda_commit] returns the same
    failure. The word still reaches every value, so that work waiting on it
    on other devices runs on. A failed [rig_cuda_submit], and every later
    one, writes its value after its waits, every earlier value and the work
    it enqueued. It writes nothing once the context has failed, whose work no
    longer runs, or if CUDA refuses a call that orders the write.

    {!sleep} asks CUDA each millisecond whether [g]'s streams have met an
    error ([cuStreamQuery]), and raises {!exception-Fault} with CUDA's error
    once one has.

    {!stop} ignores [fault]. If [g]'s work has ended (every value's work has
    completed, its streams are idle, or a fault has ended the context's
    work), it writes the last value. Otherwise it commits that value, and
    the word reaches it once the work ends, unless the work waits on a word
    of another device that never reaches its value, and it keeps an event
    behind the work on each stream. Either way it destroys [g]'s streams
    and its other events, which CUDA frees once their work ends. The GPU
    can be opened again once that work ends, by completing or by a fault,
    which then also brings the word to the last value; after a fault,
    [make] answers its error. The stop frees [g]'s C state but for the
    word and what reading it needs, which stay for the process. Compiled
    code may still call the [symbol] and [graph] functions of
    {!val-capability} after {!stop}, and rig's {!free} and {!unload} may
    come after the GPU has opened again, since the device's modules lie in
    the GPU's primary context, which the process keeps. *)

(** {1:references References}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}:
      Primary Context Management, Stream Management, Event Management, Stream
      Memory Operations ([cuStreamWaitValue64], [cuStreamWriteValue64],
      [cuStreamBatchMemOp], [CUstreamWaitValue_flags]), Memory Management
      ([cuMemHostAlloc], [cuMemHostRegister]), Unified Addressing, Peer
      Context Memory Access, Module Management, Graph Management
      ([cuGraphAddKernelNode], [cuGraphInstantiateWithFlags]), Error
      Handling ([CUDA_ERROR_ILLEGAL_ADDRESS]). *)
