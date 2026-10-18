(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Mac's GPU, driven through Apple's Metal framework.

    Each device of this library owns a Metal command queue on the Mac's GPU.
    A device has one rig queue, ["COMPUTE:0"], which runs launches and fills.
    Its memory is shared memory that the GPU and the host address alike, so
    the host copies it and the device has no copy engine.

    A kernel reaches the GPU in a metallib, the binary Apple's Metal compiler
    makes. {!Rig.Image.load} loads the metallib as a Metal library. The first
    {!Rig.Image.resolve} of one of its functions compiles the function for
    the GPU into a pipeline, which later launches reuse. A Metal command
    buffer holds encoded work, and Metal runs it only once it is committed.
    The device keeps at most one {e open command buffer}, which collects the
    work of consecutive values until the device commits it. Each
    {!Rig.submit} encodes its launches into the open command buffer
    and returns without waiting for them. The device commits the open
    command buffer when a wait asks for one of its values, and on its own
    often enough to keep the GPU busy ({!section-work}). As each command
    buffer completes, Metal calls a handler of the device on one of Metal's
    threads. The handlers write into the device's timeline word the last
    value of each command buffer once it and every buffer committed before
    it have completed without failure.

    This program adds [x] to [y] on the GPU and prints [y[10] = 11]. It
    needs [add.metallib], compiled from
    {v
    struct args { device const float *x; device float *y; };
    kernel void add(constant args &a [[buffer(0)]],
                    uint i [[thread_position_in_grid]]) { a.y[i] += a.x[i]; }
    v}
    {[
    let n = 1024

    let () =
      let ok = function Ok v -> v | Error why -> failwith why in
      let d = ok (Rig.open_ (Rig_metal.gpu 0)) in
      let lib =
        In_channel.with_open_bin "add.metallib" In_channel.input_all
      in
      let image = ok (Rig.Image.load d lib) in
      let x = Rig.Buffer.create d (4 * n)
      and y = Rig.Buffer.create d (4 * n) in
      Rig.Buffer.write x Bigarray.float32 (fun a ->
          for i = 0 to n - 1 do a.{i} <- Float.of_int i done);
      Rig.Buffer.write y Bigarray.float32 (fun a -> Bigarray.Array1.fill a 1.);
      let add = Option.get (Rig.Image.resolve image "add") in
      let add =
        Rig.Submission.(
          launch ~refs:[| read 0 ~at:0; write 1 ~at:8 |] add ~params:16)
      in
      let s = Rig.Submission.make d [| add |] in
      let run = Rig.Submission.Run.make s in
      let b = Rig.Submission.block s 0 in
      Rig.Submission.Run.groups run b (n / 256) 1 1;
      Rig.Submission.Run.threads run b 256 1 1;
      ignore (Rig.submit run ~buffers:[| x; y |] : Rig.Point.t);
      Rig.Buffer.read y Bigarray.float32 (fun a ->
          Printf.printf "y[10] = %g\n" a.{10});
      Rig.close d
    ]}
    The host addresses the device's memory, so it writes [x] and [y] in
    place, after any work on them ({!Rig.Buffer.write}). The program never
    stores the 8 parameter bytes at [0] and [8], so they hold the offset [0],
    and the kernel reads them as the GPU addresses of [x]'s and [y]'s first
    bytes, [a.x] and [a.y]. [n] is a multiple of 256, the threads of a
    group. {!Rig.Buffer.read} of [y] returns once the launch has completed,
    and raises {!Rig.Lost} if its command buffer failed.

    A device lives until {!Rig.close} or a loss ({!section-failure}). Any
    domain may call any function of this module, at the same time as other
    domains. {!sleep} and the first {!entry} for a function let other domains
    run while they wait for Metal.

    An [icb] ({!Rig_metal_abi}) and an opener's [make], called from a
    callback inside a lock ([Rig]'s domains section), raise
    [Invalid_argument], having changed nothing. rig calls the functions of
    {!section-driver}, but [locate], [peer] and [signaled], holding no lock
    ({!Rig_edge.section-drivers}); one called from such a callback raises
    [Invalid_argument], having changed nothing, where it would take a lock
    that the interrupted call holds.

    Compiled code can encode Metal commands of its own, such as indirect
    command buffers, in a fill: a C function that the driver's [submit] calls
    with the open command buffer's encoder ({!Rig_metal_abi}).
    {!Rig_edge} states how rig drives a device through its driver, and
    {!Rig_edge.section-calls} how it hands the driver work. *)

(** {1:opening Opening}

    A device opens on macOS 15 or later, on a GPU of an Apple or a Mac GPU
    family. The library builds on every system, and off macOS no device
    opens. *)

type t
(** The type for open devices. *)

val count : unit -> (int, string) result
(** [count ()] is the number of GPUs: [Ok 1] on a Mac whose GPU supports
    Metal, and [Ok 0] otherwise. A counted GPU opens only where
    {!section-opening} says. *)

val gpu : int -> t Rig_edge.opener
(** [gpu i] is the opener of GPU [i], named ["METAL"] for [0] and ["METAL:i"]
    otherwise. Its [make] opens a new device on each call, with a Metal
    command queue, a timeline and a {!val-capability} of its own, even while
    an earlier device of the GPU is open or its work still runs.

    [make] is [Error] with the reason if the Mac has no GPU [i] ({!count}), if
    the Mac lacks what {!section-opening} requires, or if Metal cannot make the
    device's command queue, fence or residency set.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:driver The driver}

    rig calls these functions. On Metal, {!alloc}, {!free}, {!locate},
    {!map_host}, {!peer} and {!map_peer} behave as in {!section-memory};
    {!val-image}, {!entry} and {!unload} as in {!section-code}; {!sleep} and
    {!stop} as in {!section-failure}; and the C functions as in
    {!section-work}. *)

include Rig_edge.Driver with type t := t

val capability : t -> Rig_metal_abi.t
(** [capability d] is [d]'s record for compiled code, which [d]'s facts
    declare ({!Rig_metal_abi.declare}). A program holds the [Rig.t] that
    {!Rig.open_} returned, never [d], and gets the record under
    {!Rig_metal_abi.key}. Its [align] is [4] on an Apple GPU
    family, the minimum constant buffer offset alignment of Apple's Metal
    feature set tables. The tables give none for Mac families, where [align]
    is [256], a multiple of every smaller power of two, which costs only
    padding. *)

(** {1:facts Facts}

    {t
      | Fact             | Value                                          |
      |------------------|------------------------------------------------|
      | [arch]           | ["AppleN"] or ["Mac2"]                         |
      | [budget]         | [recommendedMaxWorkingSetSize]                 |
      | [queues]         | ["COMPUTE:0"], which runs [Fill], [Launch]     |
      | [copy_engine]    | [None]                                         |
      | [completion]     | [Host]                                         |
      | [waits]          | no [stores] or [hosts]; [most] [0]             |
      | [may_block]      | [true]                                         |
      | [hang_ms]        | [None]                                         |
      | [maps_host]      | [true]                                         |
      | [host_addresses] | [true]                                         |
      | [alignment]      | [256] ({!section-memory})                      |
      | [capability]     | {!Rig_metal_abi.t} ({!val-capability})         |
      | [word]           | The word [make] was given, which no work reads |
    }

    [arch] is the highest Apple GPU family the GPU supports, such as
    ["Apple7"] for an M1, else its Mac family, ["Mac2"]. [budget] is the
    memory, in bytes, that Metal recommends the device keep allocated at most.
    The word holds an unsigned 64-bit integer in the host's byte order, and
    the device's values start after the value it holds when [make] returns.
    Its location has a host address and no device address: only the
    completion handlers and {!stop} write it.

    The device's queue waits only for the device's own work, so rig waits on
    the host for the points a submission waits for, before it hands the
    submission to the driver. [hang_ms] is [None] because Metal ends a command
    buffer it judges to run too long, with [MTLCommandBufferErrorTimeout]. *)

(** {1:work Work}

    The device's [struct rig_driver] holds [rig_metal_room],
    [rig_metal_submit] and [rig_metal_commit], which [rig_metal.h] declares,
    and no [settle]: the command buffers' completion handlers write a
    value's times before they raise the word.
    [rig_metal_submit] encodes the parts of a value, in order, into the open
    command buffer, and opens one if none is open. A command buffer has one
    compute encoder, which runs its commands one after the other
    ([MTLDispatchTypeSerial]), and each command buffer runs after the one
    committed before it.

    The device commits the open command buffer in four cases:
    {ul
    {- a commit ({!Rig_edge.section-calls}), for any value it is given;}
    {- a [rig_metal_submit] that finds fewer than three of the device's
       committed command buffers uncompleted, so that the GPU finds work
       queued behind the one it runs;}
    {- the completion of a command buffer while the open one holds work;}
    {- a [rig_metal_submit] that leaves 256 values in it, so the device
       commits each value at most 256 values late.}}
    A [rig_metal_submit] that commits returns [RIG_COMMITTED].

    Metal calls a handler of the device on one of Metal's threads once a
    command buffer has completed, successfully or not
    ([addCompletedHandler:]). The handlers store into the word in the order
    the command buffers were committed, with release ordering. Once a command
    buffer and every one committed before it have completed, the last value
    whose work it holds is stored. A command buffer that failed stores
    nothing, and from then on no handler stores into the word, even for a
    later command buffer that succeeds, until {!stop}. A value with no parts
    is reached once the work before it has completed.

    A fill receives the open compute encoder, encodes into it, and may start
    new command buffers ({!Rig_metal_abi}).

    [rig_metal_room] returns [RIG_LATER] and [RIG_GROW] for launches'
    parameters alone ({!section-launches}). For command buffers,
    [rig_metal_submit] instead blocks, before it opens a command buffer,
    while the device has 1,024
    command buffers, as many as its Metal queue holds, counting the open one
    and the committed ones not yet completed. [rig_metal_commit] commits the
    open command buffer, which holds the work of every value not yet
    committed.

    {2:launches Launches}

    A launch dispatches its function once, in the open command buffer's
    encoder. It sets the function's pipeline; its parameters as kernel buffer
    [0], unless it has none: up to 4 KB, the most
    [setBytes:length:atIndex:] takes, with it, and longer ones in the
    device's {e parameter arena}, a shared buffer bound with
    [setBuffer:offset:atIndex:]; its threadgroup memory, rounded up to a
    multiple of 16 bytes, as threadgroup memory [0]
    ([setThreadgroupMemoryLength:atIndex:]), unless it is [0]; and its grid
    ([dispatchThreadgroups:threadsPerThreadgroup:]). The function reads its
    parameters as a [constant] structure at [[[buffer(0)]]], and each ref's 8
    bytes as a device pointer, the GPU address of the ref's buffer plus the
    offset.

    The Metal Feature Set Tables bound a buffer's contents by its length
    alone, so the device's [maxBufferLength] bounds a launch's parameters.
    The launches take the arena's bytes in turn, each on 256 bytes, the
    largest constant buffer offset alignment, and the bytes of a command
    buffer's launches come back once its work completes. [rig_metal_room]
    returns [RIG_LATER] for parts whose parameters the arena holds only once
    earlier work completes, and [RIG_GROW] for parts whose parameters exceed
    the empty arena. The growth ({!val-grow}) makes an arena of the least
    power of two of bytes, at least 64 KiB and at most [maxBufferLength],
    that holds twice them, and the next submission takes it; the command
    buffers that use the old one keep it until their work completes. A
    growth Metal has no memory for answers [Short], so a launch rig cannot
    find memory for raises [Rig.Out_of_memory]. The arena never shrinks: it
    stays until the device stops, and counts in no budget, since a budget
    counts buffers, images' code and caches ({!Rig.section-reclaim}),
    as no driver's queues count in one.

    [rig_metal_room] refuses a launch if:
    {ul
    {- an axis of its grid or threadgroup has size [0];}
    {- its threadgroup holds more threads, along an axis or in all, than the
       least of the pipeline's [maxTotalThreadsPerThreadgroup] and the three
       axes of the device's [maxThreadsPerThreadgroup];}
    {- its threadgroup memory, rounded, exceeds the device's
       [maxThreadgroupMemoryLength] less the function's
       [staticThreadgroupMemoryLength].}}
    It also refuses parts whose parameters past 4 KB take more than
    [maxBufferLength] together.

    {2:profiles Profiles}

    While a profile is taken, [rig_metal_submit] commits the open command
    buffer, runs the value's work in command buffers of its own and commits
    them immediately. The value's times are the first one's [GPUStartTime]
    and the last one's [GPUEndTime]. Profiling therefore costs the device its
    batching. *)

(** {1:memory Memory}

    {!alloc} makes the same shared memory ([MTLResourceStorageModeShared])
    for every {!Rig_edge.type-memory}. A region is resident for the work of
    every submission whose submit starts after {!alloc} or {!map_host} has
    returned. Apple documents no alignment for a buffer's first byte. Metal
    places a buffer below a page at a multiple of 256 bytes and a larger one
    at a page, so the device's [alignment] ({!Rig_edge.type-facts}) is 256.
    A buffer whose GPU or host address breaks it faults the device, so
    {!Rig.Buffer.create} raises {!Rig.Lost}, and so does every later use of
    the device and of its memory. {!alloc} and {!map_host} raise
    [Invalid_argument] for fewer than one byte.

    {!locate} returns a region's GPU address ([gpuAddress]), which kernels
    read and write, its host address, and its [MTLBuffer]. {!free} takes the
    region out of the device's residency set, unless the device stopped, and
    releases its [MTLBuffer].

    Metal wraps only memory that starts at a page, so {!map_host}[ d p n]
    wraps the pages that hold the [n] bytes at [p], without a copy, and the
    region's host address is [p] rounded down to a page. It is [None] if Metal
    cannot wrap the memory.

    {!peer} is [false] and {!map_peer} is [None], since a Metal device maps
    only its own memory and the host's. *)

(** {1:code Code}

    {!val-image} loads a metallib as a Metal library, and leaves compiling
    its functions to {!entry}. The result is [Ok (Loaded i)], or [Error] with
    the reason if Metal cannot load the bytes as a library, or if one of its
    functions is not a compute kernel.

    The first {!entry} for a function compiles it into a pipeline
    ([MTLComputePipelineState]) that indirect command buffers may also run.
    Calls for one function from several domains concurrently make one
    pipeline. Every later call returns the same entry, which is valid until
    {!unload}. The entry's [code] is the pipeline's address, and its [launch]
    the address of what a launch reads, the pipeline and its limits. Its
    [threads] are the least of the pipeline's
    [maxTotalThreadsPerThreadgroup] and the device's axes. Its [params] are
    the size, trailing padding included, of the struct the function reads
    as kernel buffer 0, [Some 0] for a function with no buffer 0, and [None]
    where buffer 0 points to another type, such as words, whose length the
    function does not state. {!entry} raises [Invalid_argument] with Metal's
    reason if Metal cannot make a pipeline of the function, such as for a
    function that needs more threadgroup memory than the GPU family has. A
    refusal is not kept, so a later call compiles again.

    {!unload} releases the library, and the pipelines and launches {!entry}
    made of it. An indirect command buffer that records one of these
    pipelines keeps it until its own release ({!Rig_metal_abi.type-icb}). *)

(** {1:failure Failure and stopping}

    A submission fails immediately if a fill returns a failure, if Metal
    cannot make a command buffer or an encoder, or if Metal raises an
    exception. [rig_metal_submit] then drops the open command buffer
    uncommitted, with the work of every value it held, and returns
    [RIG_FAILED] with the reason. It fails later if Metal reports that a
    command buffer holding its work failed, such as one Metal ended because
    it ran too long ([MTLCommandBufferErrorTimeout]) or kept the GPU from the
    display. The reason is then ["the GPU's work failed: "] followed by
    Metal's description of the error. Either way no handler stores into the
    word again ({!section-work}), and the word stays below every value the
    failed command buffer held until {!stop}.

    The device keeps its first failure. Once it has recorded the failure,
    {!sleep} raises {!exception-Fault} with it, every [rig_metal_submit]
    returns [RIG_FAILED] with it and runs nothing, and every
    [rig_metal_commit] returns [RIG_FAILED]. rig loses the device at the
    first such call, which these make:
    {ul
    {- a {!Rig.submit} on the device;}
    {- a wait for the device's work, such as {!Rig.Point.wait},
       {!Rig.Buffer.wait} or a {!Rig.close} that has work to wait for.}}
    Until then the device is live. Work encoded before the failure was
    recorded may still run. While a command buffer runs, a wait lasts until
    the buffer completes or Metal reports its failure.

    {!sleep} blocks on a condition the handlers signal, and uses no processor
    time while it waits.

    {!stop} ignores [fault]. It drops the open command buffer uncommitted, so
    its work never runs. If every command buffer the device committed has
    completed, it writes the last value [rig_metal_submit] received into the
    word, and no work of the device writes memory again. Otherwise the
    committed work may write memory while it runs. Once the last such
    command buffer has completed, its handler writes the last value into the
    word, whatever that work did. The stop gives back what the device runs
    work with: its Metal command queue, fence, residency set and command
    buffer ring, which each committed command buffer keeps until its handler
    returns. Once the word shows the last value and [stop] has returned, no
    handler writes it: the handlers write in order, and the one that
    completes the last command buffer writes the last value. The few bytes
    of C state that read the word stay with the device. {!sleep} on a
    stopped device raises [Invalid_argument]. *)

(** {1:references References}
    - Apple's Metal framework headers (macOS 26 SDK): [MTLCommandQueue.h]
      ([newCommandQueueWithMaxCommandBufferCount:], [addResidencySet:]),
      [MTLCommandBuffer.h] ([addCompletedHandler:], [status], [error],
      [MTLCommandBufferErrorTimeout], [GPUStartTime], [GPUEndTime],
      [computeCommandEncoderWithDispatchType:]), [MTLFence.h],
      [MTLResidencySet.h] ([addAllocation:], [commit]), [MTLDevice.h]
      ([newBufferWithLength:options:],
      [newBufferWithBytesNoCopy:length:options:deallocator:],
      [recommendedMaxWorkingSetSize], [maxBufferLength], [supportsFamily:],
      [maxThreadsPerThreadgroup], [maxThreadgroupMemoryLength]),
      [MTLBuffer.h] ([gpuAddress], [contents]), [MTLLibrary.h]
      ([newLibraryWithData:error:], [functionNames], [newFunctionWithName:]),
      [MTLFunction.h] ([functionType]), [MTLComputePipeline.h]
      ([supportIndirectCommandBuffers], [maxTotalThreadsPerThreadgroup],
      [staticThreadgroupMemoryLength]), [MTLComputeCommandEncoder.h]
      ([setBytes:length:atIndex:], [setBuffer:offset:atIndex:],
      [setThreadgroupMemoryLength:atIndex:],
      [dispatchThreadgroups:threadsPerThreadgroup:]),
      [MTLIndirectCommandBuffer.h].
    - {{:https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets}
       Simplifying GPU resource management with residency sets}: a set added to
      a queue keeps its committed allocations resident for every command buffer
      of the queue.
    - {{:https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)}
       makeBuffer(bytesNoCopy:length:options:deallocator:)}: the wrapped memory
      starts at a page.
    - {{:https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf}Metal
       feature set tables}: GPU families, minimum constant buffer offset
      alignment, the 4 KB of inlined buffer contents [setBytes] takes, and
      buffer contents limited only by the buffer's length. *)
