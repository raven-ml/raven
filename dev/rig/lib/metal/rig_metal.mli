(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Mac's GPU, driven through Metal.

    A device is the Mac's GPU opened with a command queue of its own. Its work
    is one sequence of {e submissions}. The caller numbers them [1], [2], …, the
    {e values} of the device's {e timeline}, and the device makes each
    observable in its {e timeline word} ({!word}), a host word that reads [v]
    once every submission up to [v] completed. Memory is shared with the host:
    every region the device allocates is the host's memory too, at the same
    bytes.

    A device opened through rig, running an empty submission:
    {[
    let d =
      Rig.open_
        (module Rig_metal)
        ~name:(Rig_metal.device_name 0)
        (fun () -> Rig_metal.open_ 0)
      |> Result.get_ok
    in
    let s = Rig.Submission.make ~reads:0 ~writes:0 ~waits:0 d [||] in
    Rig.wait d (Rig.Point.value (Rig.submit s))
    ]}

    {b Submissions.} Work reaches the device in C, through {!room_entry} and
    {!submit_entry}, over [rig_edge.h]'s structures. A submission is a list of
    {e parts} for the device's one queue, ["COMPUTE:0"], possibly empty. A part
    is a {e fill}, a C function that encodes Metal work into a compute command
    encoder the device gives it ({!Rig_metal_abi}); it declares no ring units or
    segment bytes. The device runs no words and no copies, and waits on no other
    device's word. The submit runs the fills in order, each in an encoder that
    waits for the encoders before it, commits the submission's command buffers
    and returns. It does not wait for the work. Metal calls a handler of the
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
    with three exceptions. The C room and submit are called one at a time: the
    caller holds the device's {e turn} from a room check to the end of the
    submit it precedes. {!stop} is called once, after every other call returned.
    After {!stop} only {!free} and the release of an indirect command buffer are
    called. These rules are the caller's; the device does not check them.
    {!sleep} may run while another domain submits. A region or an image is given
    back once: of two {!free}s or {!unload}s of one value, from any domains, one
    gives it back and the other raises [Invalid_argument].

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
      [functionNames]), [MTLComputePipeline.h] ([supportIndirectCommandBuffers],
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

(** {1:facts Facts} *)

val key : t Type.Id.t
(** [key] is the key of this library's devices, by which a caller holding
    devices of several drivers finds which are Metal's. *)

val arch : t -> string
(** [arch d] is [d]'s GPU family as Metal names it: the highest Apple family the
    GPU supports, such as ["Apple7"] for an M1, else its Mac family, ["Mac2"].
*)

val budget : t -> int
(** [budget d] is the memory, in bytes, Metal recommends the device keep
    allocated at most ([recommendedMaxWorkingSetSize]). *)

val queues : t -> string list
(** [queues d] is [["COMPUTE:0"]], the device's one queue. It lists no copy
    queue: the device runs no copies, and its memory is the host's, which copies
    it. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion d] is [`Host]: the device's handlers write {!word} from the
    host. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on d c] is [false]: a Metal queue waits on no other device's word.
    Work that depends on another device starts after the caller waited for it on
    the host. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks d] is [`May_block]: the submit calls Metal, and waits for [d]'s
    oldest command buffer to complete when 1,024 of them are uncommitted or
    uncompleted, as many as [d]'s queue holds. *)

type capability = Rig_metal_abi.t
(** The type for what compiled code needs from the device. *)

val capability : t -> capability
(** [capability d] is [d]'s record.

    Its [align] is [4] on Apple GPU families, the feature set tables' minimum
    constant buffer offset alignment. The tables list none for Mac families,
    where it is [256]: an offset that is a multiple of [256] meets every smaller
    power-of-two alignment and costs only padding. A dispatch's offset is also a
    multiple of its kernel arguments' own alignment, that of their structure's
    largest member, such as [8] for a structure that holds device pointers.

    Its indirect command buffers retain their pipelines, so an {!unload} cannot
    end one. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Rig_metal_abi.key}. *)

val self : t -> nativeint
(** [self d] is the address of [d]'s C state, the [self] argument of
    {!room_entry} and {!submit_entry}. It is valid while the process runs: a
    device's C state holds its word, which other devices may read after [d] is
    gone, so neither is ever freed. *)

(** {1:memory Memory} *)

type region
(** The type for memory of a device: an [MTLBuffer] the host and the GPU share.
*)

val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
(** [alloc d kind n] is a region of [n] bytes of [d], or [None] if Metal has no
    memory for it. Every kind is the same shared memory. The region is resident
    for the work of every submission whose submit starts after [alloc] returned.

    The region's GPU and host addresses are multiples of 256 bytes. Apple
    documents no alignment for a buffer's first byte, so the device checks the
    one Metal gives: an allocation that breaks it is given back and raises
    {!exception-Fault}.

    Raises [Invalid_argument] if [n < 1]. *)

val free : t -> region -> unit
(** [free d r] gives back [r], an allocation or a {!map_host} region of [d]. The
    caller frees a region once no work of [d] that uses it is in flight.

    Raises [Invalid_argument] if [r] is another device's or {!word}, or was
    freed. *)

val address : region -> int option
(** [address r] is [Some a] with [a] the GPU address of [r]'s first byte
    ([gpuAddress]), the address kernels read and write. *)

val handle : region -> nativeint
(** [handle r] is [r]'s [MTLBuffer]. It lives until {!free}, and for {!word}
    while the process runs. *)

val host : region -> int option
(** [host r] is [Some p] with [p] the host address of [r]'s first byte. *)

val peer : t -> t -> bool
(** [peer d d'] is [false]: a Mac has one GPU, and {!map_peer} maps no memory of
    another device of it. *)

val map_peer : t -> t -> region -> region option
(** [map_peer d d' r] is [None]: a Mac has one GPU, and a region of another
    device of it is not mapped into [d].

    Raises [Invalid_argument] if [d'] is [d], or if [r] is no region of [d'] or
    was freed. *)

val map_host : t -> int -> int -> region option
(** [map_host d p n] is a region of [d] over the host memory holding the [n]
    bytes at [p], shared without a copy, or [None] if Metal cannot wrap it.
    Metal wraps memory that starts at a page, so the region covers the pages
    holding the [n] bytes: {!host} of it is the page holding [p], and [p] lies
    [p - host r] bytes in. The memory stays mapped until {!free}. The region is
    resident as {!alloc}'s.

    Raises [Invalid_argument] if [n < 1]. *)

(** {1:images Images} *)

type image
(** The type for loaded code: a Metal library, with a compute pipeline for each
    of its functions. *)

val image :
  t ->
  string ->
  ( [ `Loaded of image | `Place of int * (region -> image * string) ],
    string )
  result
(** [image d b] loads the metallib [b] and makes a compute pipeline, usable from
    an indirect command buffer, for each function it holds. Metal places the
    code itself: the result is [Ok (`Loaded i)].

    The result is [Error msg] if [b] is no metallib, or if Metal makes no
    pipeline of one of its functions. It releases the domain lock while Metal
    compiles. *)

val entry : image -> string -> int option
(** [entry i f] is the address of the [MTLComputePipelineState] of [i]'s
    function [f], or [None] if [i] has no function [f]. The address is valid
    until {!unload}.

    Raises [Invalid_argument] if [i] was unloaded. *)

val unload : t -> image -> unit
(** [unload d i] releases [i]'s pipelines. An indirect command buffer made with
    one of them keeps it until its own release ({!Rig_metal_abi.field-release}).
    The caller unloads once no work that names a pipeline of [i] directly is in
    flight.

    Raises [Invalid_argument] if [i] is another device's or was unloaded. *)

(** {1:work Work} *)

val room_entry : nativeint
(** [room_entry] is the address of [rig_metal_room], in the shape [rig_room_fn]
    of [rig_edge.h]. It answers [RIG_NEVER] for a part that is no fill on queue
    [0] or declares ring units or segment bytes, and [RIG_FITS] otherwise: the
    submit waits inside for command buffers when the queue is full. *)

val submit_entry : nativeint
(** [submit_entry] is the address of [rig_metal_submit], in the shape
    [rig_submit_fn] of [rig_edge.h]: it runs the parts as the work of [v], the
    value after the last one it received, and answers [RIG_OK] once every
    command buffer of [v] is committed; [v] is observable in {!word} once they
    all completed. With no part, [v] is observable once the work before it
    completed. Its waits are none and its handles are ignored: every region of
    the device is resident. [RIG_FAILED] if the submission failed at once, or if
    the device recorded a failure before; then the parts did not run (Failures,
    above). It calls no function of the OCaml runtime: its caller releases the
    domain lock. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word d] is [d]'s timeline word: eight bytes of host memory holding, as an
    unsigned 64-bit integer in the host's byte order, the last value [v] such
    that every submission up to [v] completed without failure. The device's
    handlers write it with release order, and never lower it. Once {!stop} was
    called and no work of [d] is in flight, it holds the last value the submit
    received, whatever that work did. It lives while the process runs. *)

val signaled : t -> int
(** [signaled d] is the value in {!word}, read with acquire order: every
    submission up to it completed, and its writes are visible to the reader,
    unless {!stop} was called. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep d ~seen ~still_ms] returns once {!word} holds a value other than
    [seen], at once if it already does, or after [still_ms] milliseconds of the
    monotonic clock, whichever comes first; [still_ms] is not negative. It
    blocks on a condition the handlers signal, using no processor time while it
    waits, and releases the domain lock.

    Raises {!exception-Fault} if [d] recorded a failure, with its reason: for a
    command buffer Metal reports failed, ["the GPU's work failed: "] followed by
    Metal's description of its error, such as
    ["the GPU's work failed: Impacting Interactivity
     (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"]. *)

(** {1:loss Loss} *)

exception Fault of string
(** [Fault why] is raised by {!sleep} once its device recorded a failure, and by
    {!alloc} when Metal places an allocation off the 256-byte alignment {!alloc}
    promises. *)

val stop : t -> unit
(** [stop d] stops [d] without waiting.

    If every command buffer [d] committed completed, it writes the last value
    the submit received into {!word} and releases [d]'s queue; no work of [d]
    writes memory again.

    Otherwise the work still in flight may write memory as long as it runs, and
    [d] keeps its queue. Metal calls a handler for every committed command
    buffer once it completed, failed ones included; once the last command buffer
    [d] committed completed, its handler writes the last value the submit
    received into {!word}, whatever that work did.

    Either way it releases the pipelines of every image not unloaded, as
    {!unload} would: the command buffers in flight and the indirect command
    buffers that use one keep it until they end. {!unload} is not called after
    [stop]. Regions end at {!free}, which may follow. *)
