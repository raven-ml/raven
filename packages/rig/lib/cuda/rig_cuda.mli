(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs, driven through the CUDA library.

    A device of this library is one GPU, opened on its primary context with
    two CUDA streams of its own, its queues ["COMPUTE:0"] and ["COPY:0"].
    ["COMPUTE:0"] runs launches, copies and fills, and ["COPY:0"] runs copies
    and fills. The device's timeline word is the word its opener's [make] is
    given, in rig's array of words, and the streams write it themselves. The
    first device that opens registers that array for every context
    ([cuMemHostRegister], portable and mapped) in its GPU's primary context,
    which the process keeps, so the registration lasts for the process. A
    GPU has one device at a time.

    A kernel reaches the GPU in a cubin, a fatbin or PTX text.
    {!Rig.Image.load} loads it as a CUDA module and places every function's
    code on the GPU. Each {!Rig.submit} enqueues its parts on the streams and
    returns without waiting for them, a launch with [cuLaunchKernel] or, in a
    run of launches, through a chain of graph nodes ({!section-replay}), and a
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
    let twice = Rig.Image.resolve image "twice" |> Option.get in
    let twice =
      Rig.Submission.(launch ~refs:[| write 0 ~at:0 |] twice ~params:8)
    in
    let s = Rig.Submission.make g [| twice |] in
    let args =
      Rig.Submission.
        { groups = (n / 256, 1, 1); threads = (256, 1, 1); shared = 0;
          params = Bytes.empty }
    in
    Rig.Point.wait (Rig.submit s ~buffers:[| x |] ~args:[| args |])
    ]}
    The kernel's parameters are its one pointer. They are given no bytes, so
    they read the offset [0], and the ref puts the address of [x]'s first
    byte in those 8 bytes.

    A device lives until {!Rig.close} or a loss, and a fault of its work ends
    the GPU's use for the rest of the process ({!section-failure}). Other CUDA
    libraries in the process share the primary context. Every call that needs
    a current context makes the device's current on the calling thread, and
    restores the thread's previous context before it returns. Any domain may
    call any function of this module, at the same time as other domains. A
    function that waits for CUDA lets other domains run while it waits.

    {!count} and an opener's [make], called from a callback inside a lock
    ([Rig]'s domains section), raise [Invalid_argument], having changed
    nothing. rig calls
    the functions of {!section-driver}, but [locate], [peer] and [signaled],
    holding no lock ({!Rig_edge.section-drivers}); one called from such a
    callback raises
    [Invalid_argument], having changed nothing, where it would take a lock
    that the interrupted call holds.

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

    The CUDA library ([libcuda]) is loaded at the first call of {!count},
    {!buses} or an opener's [make], by its standard name from the platform's
    library search path: [libcuda.so.1] (or [libcuda.so]) on Linux,
    [nvcuda.dll] on Windows, [libcuda.dylib] on macOS. A library of that name
    earlier on the path is loaded in its place, for the whole process. A load
    that fails is tried again at the next call. Building and linking this
    library needs no CUDA installation. *)

type t
(** The type for open devices: a GPU's primary context, two streams and a
    timeline word. *)

val count : unit -> (int, string) result
(** [count ()] is the number of GPUs CUDA sees, or [Error why] if the CUDA
    library cannot be loaded or initialised or CUDA cannot list its GPUs. A
    machine with none is [Ok 0]. [CUDA_VISIBLE_DEVICES] decides which GPUs
    CUDA sees. The list is read at the first call whose load and listing both
    succeed, and kept for the process. *)

val buses : unit -> (string list, string) result
(** [buses ()] is the PCI bus addresses of the GPUs CUDA sees, in {!gpu}'s
    order, as Linux writes them under [/sys/bus/pci], such as
    ["0000:01:00.0"], or {!count}'s [Error]. *)

val gpu : int -> t Rig_edge.opener
(** [gpu i] is the opener of GPU [i], the [i]th by PCI address among the GPUs
    CUDA sees, whatever [CUDA_DEVICE_ORDER] says. When CUDA sees every GPU,
    it is the GPU [nvidia-smi] numbers [i]. The opener is named ["CUDA"] for
    [0] and ["CUDA:i"] otherwise. Its [make] opens a device on the GPU's
    primary context, with streams of its own.

    [make] sets the context's [CU_LIMIT_MAX_L2_FETCH_GRANULARITY] to 128
    bytes where it reads 64, the value CUDA reports before any set. On an
    RTX 5000 Ada, until a set, the L2 fetches from memory only the 32-byte
    sectors a miss names. At 128, a kernel that reads a few bytes of each of
    many rows in each iteration of a loop misses a quarter as often, and a
    random scalar read from memory moves four times the bytes it moved
    before. The limit holds for every kernel in the context, other
    libraries' included, after the device closes too. A limit set before
    the open stays unless it is 64, which cannot be told from CUDA's report.
    Each open sets it again, so a program that wants another value sets it
    after every open, with CUDA's [cuCtxSetLimit].

    [make] is [Error] with the reason:
    {ul
    {- while a device of GPU [i] is open, or a stopped device's work on GPU
       [i] still runs;}
    {- if the CUDA library cannot be loaded or initialised, or lacks a
       function of CUDA 12.4;}
    {- if CUDA sees no GPU [i] ({!count}), or the GPU lacks what
       {!section-opening} requires;}
    {- with CUDA's error, such as the error of a fault earlier in the
       process.}}

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:driver The driver}

    rig calls these functions. On CUDA, {!alloc}, {!free}, {!locate},
    {!map_host}, {!peer}, {!map_peer} and {!val-dma} behave as in
    {!section-memory};
    {!val-image}, {!entry} and {!unload} as in {!section-code}; {!grow} as in
    {!section-replay}; {!sleep} and {!stop} as in {!section-failure}; and the
    C functions as in {!section-work}. *)

include Rig_edge.Driver with type t := t

val capability : t -> Rig_cuda_abi.t
(** [capability g] is [g]'s record for compiled code, which [g]'s facts
    declare ({!Rig_cuda_abi.declare}): the functions of the CUDA library [g]
    was opened with. *)

(** {1:facts Facts}

    {t
      | Fact             | Value                                            |
      |------------------|--------------------------------------------------|
      | [arch]           | ["sm_XY"], the GPU's compute capability          |
      | [budget]         | [cuMemGetInfo]'s total at open                   |
      | [queues]         | ["COMPUTE:0"] and ["COPY:0"]                     |
      | [copy_engine]    | [Some 1], ["COPY:0"]                             |
      | [completion]     | [Store]                                          |
      | [waits]          | [stores] and [hosts]; [most] [max_int]           |
      | [may_block]      | [true]                                           |
      | [hang_ms]        | [None]                                           |
      | [maps_host]      | [CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED]    |
      | [host_addresses] | [false]                                          |
      | [alignment]      | [256], as CUDA aligns every allocation           |
      | [cache_bytes]    | [CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE], [None] if 0 |
      | [capability]     | {!Rig_cuda_abi.t} ({!val-capability})            |
      | [word]           | The word [make] was given, registered            |
      | [bus]            | [cuDeviceGetPCIBusId], in lowercase              |
      | [capacity]       | [cuMemGetInfo]'s free and total                  |
    }

    [capacity] ({!Rig.val-capacity}) is [cuMemGetInfo] in the device's context,
    the [budget]'s source (CUDA Driver API, Memory Management): its total is
    the memory available to the context, and its free "the amount of memory
    on the device that is free according to the OS": the total less every
    process's allocations. CUDA states that it "is not guaranteed to be able
    to allocate all of" it, and states no granularity for an allocation.
    Measured on an RTX 5000 Ada, the free falls in whole blocks of 2 MiB,
    which small allocations share: 1 byte, 4 KiB and 64 KiB each take
    2 MiB, 64 allocations of one byte take 2 MiB together, 2 MiB and one
    byte take 4 MiB, and 16 MiB takes 16 MiB. The total there is
    33,881,587,712 bytes, as [cuDeviceTotalMem] also states.

    [arch] is ["sm_89"] for compute capability 8.9. ["COMPUTE:0"] runs
    [Fill], [Copy] and [Launch], and ["COPY:0"] runs [Fill] and [Copy]. The
    streams write the word, and the host does not address [Device] memory.
    The word holds an unsigned 64-bit integer in the host's byte order, and
    the device's values start after the value it holds when [make] returns.
    [alignment] is 256 since the CUDA C++ Programming Guide (Device Memory
    Accesses) states that every address an allocation routine of the driver
    or runtime API returns "is always aligned to at least 256 bytes".
    On a GPU that registers no host memory, or when the word [make] was
    given lies outside rig's array, the word is eight bytes of the device's
    own page-locked memory, which starts at that value and has no host
    address in its location: rig reads it through {!signaled}. That word,
    one 8-byte [cuMemHostAlloc] allocation, goes back when rig frees it
    ({!Rig_edge.type-facts}). Its free, as every free of host memory, may
    wait for the GPU's work.

    The streams wait on any aligned 64-bit word the device's work addresses,
    whoever writes it. A submission's waits go to its stream in batches, as
    many as it carries. On a GPU that drives a display, the display's
    watchdog ends a kernel that runs too long
    ([CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT]). Nothing else bounds the
    GPU's work in time ({!section-failure}). *)

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
    returns [RIG_COMMITTED], for a value 64 values past the last commit, so
    the device commits each value at most 64 values late.

    A fill [f] is called as [f stream arg v], with the device's context
    current on the calling thread ({!Rig_cuda_abi}). A copy part moves its
    rows between the handles of two regions of the device, at their
    offsets, in one 2D copy, or a copy per row where a pitch exceeds the
    GPU's [CU_DEVICE_ATTRIBUTE_MAX_PITCH] or CUDA refuses the pitches, and
    a copy part that fills sets its bytes with one memset. A part must not
    write a timeline word, the device's or another's. Only the devices'
    streams write their words, and a word that work wrote could move
    backwards or show work that has not completed.

    A wait holds the work back until the aligned 64-bit word at [at] holds
    at least [value], compared circularly, in the sense that [x] is at least
    [w] if [x - w], as a signed 64-bit integer, is not negative.

    [rig_cuda_room] never returns [RIG_LATER], since a stream takes any
    amount of work. It returns [RIG_GROW] for parts with a piece whose chain
    the device has not made ({!section-replay}). [rig_cuda_submit] and
    [rig_cuda_commit] instead block while a stream is full, until the
    device's earlier work frees it. A
    timed submit that measures blocks for its round trips, tens of
    microseconds, and one timed 1,024 values after an unconverted value
    waits for that value.
    For a value rig times, [rig_cuda_submit] records a timing event on
    each stream a value's work enters, after the value's waits, and one
    after its work on the stream it ends on, into one of 1,024 stamps,
    whose events it makes at the stamp's first use. It waits for the value
    1,024 values before it, whose events it reuses, if that value was
    timed.
    [rig_cuda_settle], or a later timed [rig_cuda_submit], converts the
    earliest start and the end to the host's clock, in value order, as
    {!Rig_edge.section-calls} requires of [settle]. A failed conversion
    loses the device.

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
    within 8 s.

    {2:launches Launches}

    A launch runs its function over the groups, threads and dynamic shared
    memory of its block, plain with [cuLaunchKernel] or as a node of a chain
    ({!section-replay}), and passes its parameters as one buffer
    ([CU_LAUNCH_PARAM_BUFFER_POINTER]). A launch's
    bytes past those its function reads ({!Rig.Image.params}) are not
    passed.

    [rig_cuda_room] refuses a launch if:
    {ul
    {- it is on queue [1];}
    {- its block has a size of [0] along an axis;}
    {- it has more groups or threads along an axis than the GPU allows
       ([CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X] to [_Z],
       [CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X] to [_Z]);}
    {- it has more threads per group, or more dynamic shared memory, than
       its function's limits.}} *)

(** {1:replay Replay}

    A submission's runs of consecutive launches play through {e chains}:
    CUDA graphs of [n] kernel nodes, each after the one before, of which the
    device holds at most one of each length [n] from 2 to 32. A run ends
    before a part that is no launch or whose function runs plain, and before
    a launch that starts after a part of ["COPY:0"] that ["COMPUTE:0"] has
    not yet waited for, so that only that launch and those after it wait for
    ["COPY:0"]. That wait covers all the work ["COPY:0"] was given before
    the launch, so launches that each start after one copy there play as one
    run. A run of one launch runs plain, with [cuLaunchKernel]. A longer run
    is cut into the fewest pieces of at most 32 launches, whose lengths
    differ by at most one: 33 launches play as 17 then 16. A piece updates
    every node of the chain of its length to its launches
    ([cuGraphExecNodeSetParams]: function, groups, threads, dynamic shared
    memory and parameters), then launches the chain ([cuGraphLaunch]). An
    update reaches the chain's launches after it and none queued before, so
    it never waits for the device's work, and the chain's launch blocks
    while ["COMPUTE:0"] is full, as [cuLaunchKernel] does. A piece of [k]
    launches thus costs the submitting thread [k] updates and one launch
    call where plain launches cost [k] launch calls. An update costs the
    host less than a launch call, more for a larger parameter block and more
    again when the node's function changes. The updates also add GPU time to
    the chain's launch, more for a larger parameter block.

    The device makes a chain at the first submit that needs its length, as
    growth ({!Rig_edge.Driver.val-grow}): [rig_cuda_room] answers
    [RIG_GROW], and {!grow} makes the chain, outside the device's turn, from
    a placeholder kernel that does nothing. The GPU's first {!val-image} in
    the process loads that kernel, PTX in a module of its own that the
    process keeps. Where CUDA refuses it, for lack of memory or because the
    driver compiles no PTX, as under [CUDA_DISABLE_PTX_JIT=1], the image
    loads all the same and the next image tries again: until one succeeds,
    the device makes no chain and plays every piece plain. A make that CUDA
    refuses for lack of memory answers [Short] of 2 MiB, the chunk CUDA
    takes graph memory in, so rig reclaims memory and asks again, then
    raises {!Rig.Out_of_memory} at the submit; each later grow that needs
    the length makes it again. One that CUDA refuses for another reason,
    while the context stands, leaves that length's pieces plain for the
    device's life.

    Two submits that need one length make it once, since a grow keeps the
    device's chains to itself until its makes end. A submit whose room
    finds a length missing, or that plays a piece, while another thread's
    grow runs thus waits for that grow's makes, holding the device's turn,
    so every other submitter of the device waits too; {!val-image} and
    {!unload} wait as well. Each length is made once in the device's life,
    unless CUDA refused its make for lack of memory.

    A chain holds memory of the GPU and of the host for itself and for each
    of its nodes, 527 nodes for the 31 lengths. A node's host memory grows on
    its first update to a larger parameter block than it held, to hold that
    block, and does not shrink while its chain lives. Neither the device's
    budget nor the host's counts this memory, but [capacity]'s free falls
    by its GPU part, which [cuMemGetInfo] counts as any allocation of the
    process. {!stop} destroys the chains, but CUDA keeps a destroyed graph's
    GPU memory for the process's later graphs and never gives it back to
    the GPU, so after a close that memory stays the process's, out of the
    free of the GPU's later devices, and a later device of the GPU reuses
    it.

    A function that needs clusters, whose launch must set a cluster size
    ([CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET]) or which has one of its
    own ([CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH]), runs plain from its
    {!entry}, since CUDA checks clusters only at a launch. A function whose
    update CUDA refuses ([CUDA_ERROR_INVALID_VALUE] or
    [CUDA_ERROR_NOT_SUPPORTED]) runs plain from then on: its piece and the
    rest of its run launch plain, and the refusal loses nothing, though a
    plain launch that CUDA refuses too fails the submit as any launch does.
    Any other error of an update or of a chain's launch fails the submit
    ({!section-failure}).

    {!unload} points every node that may hold a function of the image back at
    the placeholder before CUDA unloads the module, so that no chain names
    an unloaded function. *)

(** {1:memory Memory}

    A region is an allocation of the device, host memory it maps, or another
    device's memory it maps. {!alloc} returns:
    {ul
    {- for [Device], GPU memory, which the host does not address;}
    {- for [Pinned], page-locked host memory that the host and every CUDA
       device address ([cuMemHostAlloc], portable and mapped). It is not
       write-combined, so the host reads it through its caches;}
    {- for [Mapped], [None], since CUDA maps no GPU memory for the host:
       its facts declare no [mapped], and {!flush} does nothing.}}
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
    [true], and is [Never] otherwise. Freeing a {!map_peer} region ends only
    that region. Peer access between two GPUs is untested.

    {!map_host}[ g a n] page-locks the [n] bytes at [a] for every CUDA device
    and maps them, at the address CUDA gives for the memory
    ([cuMemHostGetDevicePointer]). Page-locking belongs to the process, and
    the region's pages stay locked until every region {!map_host} gave over
    them, on any device, is freed. The last free unlocks them. One made
    while a {!map_host} or an open of a CUDA device runs, on another thread
    or in the call its callback interrupted, may instead leave them locked
    until no such call runs. The result depends on what is already locked:
    {ul
    {- A range inside one that {!map_host} locked is mapped and counts
       against it.}
    {- A range inside one allocation that CUDA locked for another owner,
       such as an allocation of {!alloc} or of another library, is mapped as
       it is, uncounted. That memory must stay locked until the region is
       freed.}
    {- Any other range is locked. Where CUDA refuses, it is [No_room] if
       CUDA had no memory to lock or map the range, and [Never] otherwise,
       as for read-only memory, a shared mapping of a file and a range that
       overlaps locked memory without lying inside one of its ranges.}}
    It is also [Never] where [g] does not map host memory ([maps_host]).

    On Linux, CUDA locks host pages for writing, which Linux refuses for a
    shared mapping of a file. A CUDA device therefore cannot borrow a region
    that an io device holds for writing through a shared mapping of a file
    ({!Rig_edge.Io.pages}), and copies to and from that region go through
    staging memory.

    {!val-dma}[ g r ~fn] exports the whole allocation [r] lies in as a
    dma-buf ([cuMemGetHandleForAddressRange], with no flags), at [r]'s
    distance from the allocation's start. CUDA exports [Device] memory only
    where its kernel driver can, as [CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED]
    states: on Linux, with NVIDIA's open kernel modules. It exports
    [Pinned] memory only where [CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED]
    holds, an attribute from CUDA 13 on. The kernel maps the memory for
    [fn] where [fn] registers the dma-buf, and refuses there a topology it
    does not support. With no flags it maps a coherent GPU's memory as the
    host's pages, over the GPU's link to the host, so a NIC behind the
    GPU's own PCIe switch reaches it through the host's memory system, at
    that link's bandwidth, where a BAR1 mapping would cross only the
    switch. [fn] reads what the device's work wrote once that work's value
    is reached, since the word's write ([cuStreamWriteValue64_v2]) fences
    the stream's writes for every reader first. It is [Error], naming the
    cause: for an [fn] that is no bus address or no function under
    [/sys/bus/pci/devices]; where CUDA exports no dma-buf, naming the
    attribute that decides it; and for an allocation whose start or size is
    off the host's page, as CUDA's export requires and its allocations,
    aligned to 256 bytes, need not be. *)

(** {1:code Code}

    CUDA may wait for all of the GPU's work before {!free} frees an
    allocation, before {!val-image} loads a module, and before a module is
    unloaded. CUDA waits as the
    scheduling in the primary context's flags says
    ([cuDevicePrimaryCtxSetFlags]), and an open leaves those flags as it
    finds them. Under CUDA's default, [CU_CTX_SCHED_AUTO], CUDA may
    busy-wait, and a call behind queued work then spends processor time until
    that work completes, however long the device's word stands still. rig's
    own waits sleep once the word has stood still for 200 ms
    ({!Rig.section-hangs}).

    {!val-image}[ g bin] is [Ok (Loaded m)] with [m] the CUDA module of [bin]:
    a cubin, a fatbin, or PTX text, which CUDA compiles for the GPU. It places
    every function's code before it returns, whatever [CUDA_MODULE_LOADING]
    asks, so neither {!entry} nor a launch loads code later. Each function
    may take as much dynamic shared memory as a block of the GPU can have,
    less its static shared memory. Until one has loaded it, each image also
    loads the placeholder kernel of the device's chains, whose refusal
    leaves the image's load as it is ({!section-replay}). The result is
    [Error] with CUDA's error if CUDA refuses [bin], for instance a cubin for
    another GPU or PTX that the driver does not compile, or lacks the memory
    for its code.

    {!entry}[ m f] is the entry of kernel [f] of [m]. Its [code] is the
    kernel's [CUfunction], which compiled code passes to [cuLaunchKernel].
    Its [launch] is C memory that holds the function and its limits: the most
    threads a group of it has ([CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]),
    and the most dynamic shared memory a group of it takes. Both are valid
    until [m] is unloaded, and a second {!entry} for [f] returns the same.
    Its [params] end at the kernel's last parameter
    ([cuFuncGetParamInfo]), and its [threads] are the most threads above. A
    launch hands CUDA those [params] bytes of its parameters, since CUDA
    refuses a buffer of another size. {!entry} raises [Invalid_argument] for
    a kernel of more parameter bytes than CUDA launches on the GPU: 32,764
    on compute capability 7.0 or later, and 4,096 below.

    {!unload} points the device's chain nodes that may hold the image's
    functions back at the placeholder, then unloads the module. *)

(** {1:failure Failure and stopping}

    rig loses a device when CUDA reports a fault of its work, such as an
    illegal address or a kernel that a display's watchdog ended, at a later
    call. A fault leaves the process in an inconsistent state: every later
    CUDA call returns the fault's error, and to use CUDA again the process
    must be terminated and relaunched. So the GPU cannot be opened again,
    and every other CUDA device of the process is lost at its first call
    that CUDA answers with the fault's error ({!Rig.section-loss}).

    Before a fault, {!alloc}, {!map_host}, {!val-image} and {!entry} return
    CUDA's refusals as their result, [None], [Never] or [No_room], [Error]
    or [Absent], and
    {!unload} raises {!exception-Fault} if CUDA refuses to unload the module.
    After a fault, the calls of {!alloc}, {!map_host}, {!val-image},
    {!entry}, {!unload}, {!grow} and {!sleep} that call CUDA raise
    {!exception-Fault}
    with its error, and the others answer as before the fault, such as an
    {!entry} for a function already found or a {!sleep} once the word has
    moved. [rig_cuda_submit] and [rig_cuda_commit] then return [RIG_FAILED]
    once they call CUDA. {!free} and {!stop} ignore CUDA's errors and raise
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
    behind the work on each stream. Either way it destroys [g]'s chains,
    streams and other events, which CUDA frees once their work ends. The GPU
    can be opened again once that work ends, by completing or by a fault.
    CUDA signals no fault, so after one the GPU's next open raises the word
    to the last value, and until then the device does not count as
    stopped; that open's [make] answers the fault's error. The stop frees
    [g]'s C state but for what reading the word needs, the 64-byte edge,
    which stays for the process. Once the word shows the last value and the
    stop has returned, no work of [g] writes it: a stream's write of the
    last value is its last command, and the stop and the next open raise the
    word only once both streams are idle or one has met an error, after
    which CUDA runs none of the context's queued work, as the 615 driver
    does (CUDA's reference states only that such an error is returned to
    every later call). Compiled code may still call the [symbol] function of
    {!val-capability} after {!stop}, and rig's {!free} and
    {!unload} may come after the GPU has opened again, since the device's
    modules lie in the GPU's primary context, which the process keeps. *)

(** {1:references References}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}:
      Primary Context Management, Stream Management, Event Management, Stream
      Memory Operations ([cuStreamWaitValue64], [cuStreamWriteValue64],
      [cuStreamBatchMemOp], [CUstreamWaitValue_flags]), Memory Management
      ([cuMemHostAlloc], [cuMemHostRegister]), Unified Addressing, Peer
      Context Memory Access, Module Management, Graph Management
      ([cuGraphAddNode], [cuGraphInstantiateWithFlags],
      [cuGraphExecNodeSetParams], [cuGraphLaunch]), Execution Control
      ([CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET]), Error
      Handling ([CUDA_ERROR_ILLEGAL_ADDRESS]), Version Management
      ([cuDriverGetVersion]).
    - {{:https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/}
       CUDA 12.1 Supports Large Kernel Parameters}: a kernel's parameters
      take up to 32,764 bytes on Volta and later under a CUDA 12.1 driver or
      newer, and 4,096 below Volta. *)

(**/**)

val replays : t -> int * int
(** [replays g] is [(pieces, plain)]: the pieces [g]'s submits played through
    a chain and the launches they ran plain, since [g] opened; [(0, 0)] once
    [g] is stopped. *)

val made : t -> int
(** [made g] is the number of chains [g]'s grows made since [g] opened; [0]
    once [g] is stopped. *)
