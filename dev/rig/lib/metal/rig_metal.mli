(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Mac's GPU, driven through Metal.

    A device is the Mac's GPU opened with a command queue of its own. Its work
    is one sequence of {e submissions}. The caller numbers them [1], [2], …, the
    {e values} of the device's {e timeline}, and the device makes each
    observable in its {e timeline word} (the [word] of {!facts}), a host word
    that reads [v] once every submission up to [v] completed. Memory is shared
    with the host: every region the device allocates is the host's memory too,
    at the same bytes.

    A device opened through rig, running an empty submission:
    {[
    let d =
      Rig.open_
        (module Rig_metal)
        ~name:(Rig_metal.device_name 0)
        (fun () -> Rig_metal.open_ 0)
      |> Result.get_ok
    in
    let s = Rig.Submission.make ~reads:0 ~writes:0 d [||] in
    let run = Rig.Submission.Run.make () in
    let p = Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||] in
    Rig.wait d (Rig.Point.value p)
    ]}

    {b Submissions.} Work reaches the device in C, through its room check and
    submit (the [edge] of {!facts}), over [rig_edge.h]'s structures. A
    submission is a list of {e parts} for the device's one queue, ["COMPUTE:0"],
    possibly empty. A part is a {e fill}, a C function that encodes Metal work
    into a compute command encoder the device gives it ({!Rig_metal_abi}); it
    declares no ring units or segment bytes. The device runs no words and no
    copies, and waits on no other device's word. The submit runs the fills in
    order into the device's {e open command buffer}, opening one if none is, in
    one encoder per command buffer, each command buffer after the ones before
    it, and returns. It does not wait for the work. Metal runs a command buffer
    only once it is committed, and the device commits the open one at four
    points: the edge's commit; a submit that finds fewer than three of its
    committed command buffers uncompleted, so that the GPU finds work queued
    behind the one it runs; the completion of one of its command buffers while
    one is open; and a submit that leaves 256 values in it, so the device
    commits each value at most 256 values late. Metal calls a handler of the
    device on one of its own threads once each command buffer completed,
    successfully or not ([addCompletedHandler:]). The handlers write [v] into
    the word once every command buffer of the submissions up to [v] completed
    without failure, whatever order they complete in. Only the handlers and
    {!stop} write the word.

    {b Failures.} A submission fails at once if a fill returns a failure, if
    Metal makes no command buffer or no encoder, or if Metal raises an
    exception; the submit then answers [RIG_FAILED] with the reason. It fails
    later if Metal reports one of its command buffers failed, such as one Metal
    aborts because it ran too long ([MTLCommandBufferErrorTimeout]) or kept the
    GPU from the display. Either way the word stops: it never reaches the failed
    submission's value, nor any later one, until {!stop}. Once the device
    recorded a failure, {!sleep} raises {!exception-Fault} with its reason and
    every submit answers it and runs nothing. A long command buffer is no
    failure: a wait lasts until the work completes or Metal reports it failed.

    {b Compiled code.} Code compiled for the device reaches it through its
    {!val-capability}, the record {!Rig_metal_abi.t}: indirect command buffers,
    the [split] a fill calls to start a new command buffer, and the argument
    alignment. The fill's calling convention is stated there.

    {b Domains.} Any domain may call any function, at the same time as others,
    as {!Rig_edge.Driver} states. Compiled code may also call the [icb] function
    of {!val-capability} at any time, beside {!stop}, which waits for a call in
    flight, and after it; an indirect command buffer may be released after
    {!stop} too, also after the GPU opened again.

    {b Platforms.} A device opens on macOS 15 and later: its queue keeps every
    region resident through a residency set, which Metal has from macOS 15
    ([MTLResidencySet.h]). Elsewhere the library builds, {!count} is [0] off
    macOS, and {!open_} answers [Error].

    {b References.}
    - Apple's Metal framework headers (macOS 26 SDK): [MTLCommandQueue.h]
      ([newCommandQueueWithMaxCommandBufferCount:], [addResidencySet:]),
      [MTLCommandBuffer.h] ([addCompletedHandler:], [status], [error],
      [MTLCommandBufferErrorTimeout], [GPUStartTime], [GPUEndTime],
      [computeCommandEncoderWithDispatchType:]), [MTLFence.h],
      [MTLResidencySet.h] ([addAllocation:], [commit]), [MTLDevice.h]
      ([newBufferWithLength:options:],
      [newBufferWithBytesNoCopy:length:options:deallocator:],
      [recommendedMaxWorkingSetSize], [supportsFamily:]), [MTLBuffer.h]
      ([gpuAddress], [contents]), [MTLLibrary.h] ([newLibraryWithData:error:],
      [functionNames], [newFunctionWithName:]), [MTLFunction.h]
      ([functionType]), [MTLComputePipeline.h] ([supportIndirectCommandBuffers],
      [maxTotalThreadsPerThreadgroup]), [MTLIndirectCommandBuffer.h].
    - {{:https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets}
       Simplifying GPU resource management with residency sets}: a set added to
      a queue keeps its committed allocations resident for every command buffer
      of the queue.
    - {{:https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)}
       makeBuffer(bytesNoCopy:length:options:deallocator:)}: the wrapped memory
      starts at a page.
    - {{:https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf}Metal
       feature set tables} (May 21, 2026): GPU families, minimum constant buffer
      offset alignment. *)

(** {1:opening Opening} *)

type t
(** The type for open devices. *)

val count : unit -> int
(** [count ()] is the number of devices: [1] on a Mac whose GPU supports Metal,
    [0] otherwise. *)

val device_name : int -> string
(** [device_name i] is the name device [i] has once open: ["METAL"] for [0],
    ["METAL:i"] otherwise.

    Raises [Invalid_argument] if [i < 0]. *)

val open_ : int -> (t, string) result
(** [open_ i] opens device [i]: a queue of its own over the Mac's GPU, a fresh
    timeline at [0] and a fresh {!val-capability}. Each call makes a new device,
    whatever devices are open or lost.

    The result is [Error msg] if [i >= count ()], off macOS, on macOS before 15,
    if the GPU belongs to no Apple or Mac GPU family, or if Metal makes no
    queue, fence, residency set or word buffer for it.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:driver The driver} *)

include Rig_edge.Driver with type t := t

val capability : t -> Rig_metal_abi.t
(** [capability d] is [d]'s record, which {!facts} declares under
    {!Rig_metal_abi.key}.

    Its [align] is [4] on Apple GPU families, the feature set tables' minimum
    constant buffer offset alignment. The tables list none for Mac families,
    where it is [256]: an offset that is a multiple of [256] meets every smaller
    power-of-two alignment and costs only padding. A dispatch's offset is also a
    multiple of its kernel arguments' own alignment, that of their structure's
    largest member, such as [8] for a structure that holds device pointers.

    Its indirect command buffers retain their pipelines, so an {!unload} cannot
    end one. *)

(** {1:this This driver}

    {t
      | Fact             | Value                                  |
      |------------------|----------------------------------------|
      | [arch]           | ["AppleN"] or ["Mac2"]                 |
      | [budget]         | [recommendedMaxWorkingSetSize]         |
      | [queues]         | ["COMPUTE:0"], which runs [Fill]       |
      | [completion]     | [Host]                                 |
      | [waits]          | None, [most = 0]                       |
      | [may_block]      | [true]                                 |
      | [maps_host]      | [true]                                 |
      | [host_addresses] | [true]                                 |
      | [capability]     | {!Rig_metal_abi.t} ({!val-capability}) |
      | [word]           | Eight bytes of host memory             |
    }

    {b Facts.} [arch] is the highest Apple GPU family the GPU supports, such as
    ["Apple7"] for an M1, else its Mac family, ["Mac2"]. [budget] is the memory,
    in bytes, Metal recommends the device keep allocated at most. The word is an
    [MTLBuffer] holding an unsigned 64-bit integer in the host's byte order,
    which the device's handlers write from the host.

    {b Queues.} The device lists no copy queue: its memory is the host's
    ([host_addresses]), which copies it. Work that depends on another device
    starts after the caller waited for it on the host.

    {b The C edge.} The edge's functions are [rig_metal_room],
    [rig_metal_submit] and [rig_metal_commit], which [rig_metal.h] declares. The
    edge is valid while the process runs: a device's C state holds its word,
    which other devices may read after [d] is gone, so neither is ever freed.
    - [rig_metal_room] answers [RIG_NEVER] for a part that is no fill on queue
      [0] or declares ring units or segment bytes, and [RIG_FITS] otherwise: the
      submit waits inside for command buffers when the queue is full.
    - [rig_metal_submit] runs the parts as the work of [v] and answers
      [RIG_COMMITTED] if it committed the open command buffer, [RIG_OK] if [v]'s
      work waits in it. With no part, [v] is observable once the work before it
      completed. Its waits are none and its handles are ignored: every region of
      the device is resident. It answers [RIG_FAILED] if the submission failed
      at once, or if the device recorded a failure before; then the parts did
      not run (Failures, above). It may block: it calls Metal, and waits for the
      device's oldest command buffer to complete when 1,024 of them are
      uncommitted or uncompleted, as many as the device's queue holds.
    - [rig_metal_commit] commits the open command buffer, which holds the work
      of every value not yet committed, whatever [v].

    {b Memory.} {!alloc} makes the same shared memory for every
    {!Rig_edge.memory}. A region is resident for the work of every submission
    whose submit starts after {!alloc} or {!map_host} returned. An allocation's
    GPU and host addresses are multiples of 256 bytes: Apple documents no
    alignment for a buffer's first byte, so the device checks the one Metal
    gives, and an allocation that breaks it is given back and raises
    {!exception-Fault}. {!alloc} and {!map_host} raise [Invalid_argument] for
    fewer than one byte.

    {!locate} answers the region's GPU address ([gpuAddress]), the address
    kernels read and write, its host address, and its [MTLBuffer], which lives
    until {!free}, and the word's while the process runs. {!free} releases the
    buffer.

    {!peer} is [false] and {!map_peer} answers [None]: a Mac has one GPU, and a
    region of another device of it is not mapped into [d].

    {!map_host} answers [None] if Metal cannot wrap the memory. It accepts any
    [p]: Metal wraps memory that starts at a page, so the region covers the
    pages holding the [n] bytes at [p]. Its host address is the page holding
    [p], and [p] lies [p - host] bytes in.

    {b Code.} {!image} loads a metallib and makes no pipeline: Metal places the
    code itself, and the result is [Ok (Loaded i)]. It is [Error msg] if the
    bytes are no metallib, or if one of its functions is no compute kernel.

    {!entry} answers the address of the [MTLComputePipelineState] of the
    function, usable from an indirect command buffer. The first call for a
    function makes the pipeline: Metal compiles it for the GPU, 0.1 to 1 s when
    its shader cache does not hold it, and the call releases the domain lock
    meanwhile. Every later call answers the same address, which is valid until
    {!unload}. Calls for one function from several domains at once make one
    pipeline. It raises [Invalid_argument] with Metal's reason if Metal makes no
    pipeline of the function: the function needs what the GPU family lacks
    (Apple's Metal feature set tables), such as more than its 32 KB of
    threadgroup memory on the Apple families. A refusal is not kept: a later
    call compiles again.

    {!unload} releases the image's library and the pipelines {!entry} made of
    it. An indirect command buffer made with one of them keeps it until its own
    release ({!Rig_metal_abi.field-release}).

    {b Timeline.} The handlers write the word with release order, and never
    lower it. {!sleep} blocks on a condition the handlers signal, using no
    processor time while it waits, and releases the domain lock. Once the device
    recorded a failure it raises {!exception-Fault} with its reason: for a
    command buffer Metal reports failed, ["the GPU's work failed: "] followed by
    Metal's description of its error, such as
    ["the GPU's work failed: Impacting Interactivity
     (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"].

    {b Stopping.} {!stop} ignores [fault]. It drops the open command buffer
    uncommitted: its work never runs. If every command buffer the device
    committed completed, it writes the last value the submit received into the
    word and releases the device's queue; no work of the device writes memory
    again. Otherwise the work still in flight may write memory as long as it
    runs, and the device keeps its queue: once the last command buffer it
    committed completed, its handler writes the last value into the word,
    whatever that work did. An [icb] call of {!val-capability} in flight returns
    before {!stop} begins, and every later one answers [Error]. *)
