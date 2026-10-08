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

    A device used alone, waiting for an empty submission:
    {[
    let d = Result.get_ok (Device_metal.open_ 0) in
    match Device_metal.room d [||] with
    | `Fits -> (
        match Device_metal.submit d ~v:1 ~waits:[||] ~handles:[||] [||] with
        | `Ok ->
            let rec wait () =
              let seen = Device_metal.signaled d in
              if seen < 1 then (
                Device_metal.sleep d ~seen ~still_ms:200;
                wait ())
            in
            wait ()
        | `Failed why -> prerr_endline why)
    | `Later | `Never -> prerr_endline "no room"
    ]}

    {b Submissions.} A submission is a list of {e parts} for the device's one
    queue, ["COMPUTE:0"] ({!part}), possibly empty. A part is a {e fill}, a C
    function that encodes Metal work into a compute command encoder the device
    gives it. {!submit} runs the fills in order, each in an encoder that waits
    for the encoders before it, commits the submission's command buffers and
    returns. It does not wait for the work. When a command buffer of value [v]
    completes, Metal calls the device's handler on one of its threads, which
    writes [v] into the word once every earlier command buffer completed without
    failure. Only the handler and {!stop} write the word.

    {b Failures.} A fill that returns a failure, or a command buffer Metal could
    not make, fails {!submit}. A command buffer that Metal reports failed, such
    as one macOS ends because it kept the GPU from the display, stops the word:
    it never passes the value before the failed one, and {!sleep} raises
    {!Fault} with Metal's reason. A long command buffer is no failure: a wait
    lasts until the work completes or Metal reports it failed. After a failure
    every {!submit} answers the first failure and runs nothing.

    {b Compiled code.} Code compiled for the device reaches it through its
    {!capability}, the record {!Device_metal_abi.t}: indirect command buffers,
    the [split] a fill calls to start a new command buffer, and the argument
    alignment. The fill's calling convention is stated there.

    {b Domains.} Any domain may call any function, at the same time as others,
    with three exceptions. {!room} and {!submit} are called one at a time: the
    caller holds the device's {e turn} from {!room} to the end of {!submit}.
    {!stop} is called once, after every other call returned. After {!stop} only
    {!free} and {!unmap} are called. {!sleep} may run while another domain
    submits.

    {b Platforms.} A device opens on macOS 15 and later, whose command queues
    take residency sets. Elsewhere the library builds, {!count} is [0] off
    macOS, and {!open_} answers [Error].

    {b References.}
    - Apple's Metal framework headers (macOS 26 SDK): [MTLCommandQueue.h]
      ([newCommandQueueWithMaxCommandBufferCount:], [addResidencySet:]),
      [MTLCommandBuffer.h] ([addCompletedHandler:], [status], [error],
      [GPUStartTime], [GPUEndTime], [computeCommandEncoderWithDispatchType:]),
      [MTLFence.h], [MTLResidencySet.h] ([addAllocation:], [commit]),
      [MTLDevice.h] ([newBufferWithLength:options:],
      [newBufferWithBytesNoCopy:length:options:deallocator:],
      [recommendedMaxWorkingSetSize], [supportsFamily:]), [MTLLibrary.h]
      ([newLibraryWithData:error:], [functionNames]), [MTLComputePipeline.h]
      ([supportIndirectCommandBuffers], [maxTotalThreadsPerThreadgroup]),
      [MTLIndirectCommandBuffer.h].
    - {{:https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets}
       Simplifying GPU resource management with residency sets}: a set added to
      a queue keeps its committed allocations resident for every command buffer
      of the queue.
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
    timeline at [0] and a fresh {!capability}. Each call makes a new device,
    whatever devices are open or lost.

    The result is [Error msg] if [i >= count ()], off macOS, on macOS before 15,
    if the GPU belongs to no Apple or Mac GPU family, or if Metal makes no queue
    for it.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:facts Facts} *)

val key : t Type.Id.t
(** [key] is the key of this library's devices, by which a caller holding
    devices of several drivers finds which are Metal's. *)

val arch : t -> string
(** [arch d] is [d]'s GPU family as Metal names it: the highest Apple family the
    GPU supports, such as ["Apple7"] for an M1, else its Mac family, ["Mac2"].
*)

val machine : t -> string option
(** [machine d] is [None]: the device is this machine's. *)

val budget : t -> int
(** [budget d] is the memory, in bytes, Metal recommends the device keep
    allocated at most ([recommendedMaxWorkingSetSize]). *)

val queues : t -> string list
(** [queues d] is [["COMPUTE:0"]], the device's one queue. It lists no copy
    queue: the device runs no copies, and its memory is the host's, which copies
    it. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion d] is [`Host]: the device's handler writes {!word} from the
    host. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on d c] is [false]: a Metal queue waits on no other device's word.
    Work that depends on another device starts after the caller waited for it on
    the host. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks d] is [`May_block]: {!submit} calls Metal, and may wait for [d]'s
    earlier command buffers to complete when [d] holds as many uncompleted
    command buffers as its queue does. *)

type capability = Device_metal_abi.t
(** The type for what compiled code needs from the device. *)

val capability : t -> capability
(** [capability d] is [d]'s record. Its [align] is [4] on Apple GPU families,
    the feature set tables' minimum constant buffer offset alignment, and [256]
    on Mac families, which the tables do not list: a conservative value, since
    an offset that is a multiple of [256] meets any smaller power-of-two
    alignment and costs only padding. A dispatch's offset is also a multiple of
    its kernel arguments' own alignment, an argument structure's largest
    member's, such as [8] for a structure that holds device pointers. Its
    indirect command buffers retain their pipelines, so an {!unload} cannot end
    one, and an indirect command buffer's [release] keeps its objects until
    every command buffer [d] committed before the call completed, so it is safe
    whatever {!stop} answered. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Device_metal_abi.key}. *)

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
(** [alloc d kind n] is a region of [n] bytes of [d], starting at a multiple of
    the host's page size (16 KiB on Apple silicon), or [None] if Metal has no
    memory for it. Every kind is the same shared memory. The region is resident
    for the work of every submission that starts after the call returns.

    Raises [Invalid_argument] if [n < 1]. *)

val free : t -> region -> unit
(** [free d r] gives [r] back to Metal. The caller frees a region once no work
    of [d] that uses it is in flight.

    Raises [Invalid_argument] if [r] is no allocation of [d] (another device's,
    a {!map_host} region or {!word}), or was freed. *)

val address : region -> int option
(** [address r] is [Some a] with [a] the GPU address of [r]'s first byte
    ([gpuAddress]), the address kernels read and write. *)

val handle : region -> nativeint
(** [handle r] is [r]'s [MTLBuffer]. It lives until {!free} or {!unmap}. *)

val host : region -> nativeint option
(** [host r] is [Some p] with [p] the host address of [r]'s first byte. *)

val map_peer : t -> t -> region -> region option
(** [map_peer d d' r] is [None]: a Mac has one GPU, and a region of another
    device of it is not mapped into [d]. *)

val map_host : t -> nativeint -> int -> region option
(** [map_host d p n] is a region of [d] over the host memory holding the [n]
    bytes at [p], shared without a copy, or [None] if Metal cannot wrap it.
    Metal wraps whole pages: the region starts at the page holding [p], so
    {!host} of it is that page and [p] lies [p - host r] bytes in. The memory
    must stay mapped until {!unmap}. The region is resident as {!alloc}'s.

    Raises [Invalid_argument] if [n < 1]. *)

val unmap : t -> region -> unit
(** [unmap d r] ends the mapping {!map_host} made. The caller unmaps once no
    work of [d] that uses [r] is in flight.

    Raises [Invalid_argument] if [r] is no {!map_host} region of [d], or was
    unmapped. *)

(** {1:images Images} *)

type image
(** The type for loaded code: a Metal library, with a compute pipeline for each
    of its functions. *)

val image : t -> string -> (image * (region * string) option, string) result
(** [image d b] loads the metallib [b] and makes a compute pipeline, usable from
    an indirect command buffer, for each function it holds. Metal places the
    code itself, so the result has no upload: [Ok (i, None)].

    The result is [Error msg] if [b] is no metallib, or if Metal makes no
    pipeline of one of its functions. It releases the domain lock while Metal
    compiles. *)

val entry : image -> string -> int option
(** [entry i f] is the address of the [MTLComputePipelineState] of [i]'s
    function [f], or [None] if [i] has no function [f]. *)

val unload : t -> image -> unit
(** [unload d i] releases [i]'s pipelines. An indirect command buffer made with
    one of them keeps it until its own release ({!Device_metal_abi.icb}). The
    caller unloads once no work that names a pipeline of [i] directly is in
    flight, and never twice. *)

(** {1:work Work} *)

type part
(** The type for work on the device's queue. *)

val part :
  t ->
  queue:string ->
  ?after:int array ->
  [ `Words of int array
  | `Fill of nativeint * nativeint * int * int
  | `Copy of (region * int) * (region * int) * int ] ->
  part
(** [part d ~queue ~after w] is the work [w] on [queue]. The device runs one
    kind: [`Fill (f, arg, 0, 0)], the fill at address [f], called with [arg]
    ({!Device_metal_abi}). A fill may start any number of command buffers, so it
    declares no ring units and no segment bytes.

    [after] (defaults to [[||]]) lists the earlier parts of the submission this
    part waits for. The device runs a submission's parts in order, each after
    the one before, so [after] adds nothing to that order.

    Raises [Invalid_argument] if [queue] is not ["COMPUTE:0"], if [w] is
    [`Words] or [`Copy], which the device does not run (its memory is the
    host's, which copies it), if a fill declares ring units or segment bytes, or
    if an index of [after] is negative. *)

val room : t -> part array -> [ `Fits | `Later | `Never ]
(** [room d ps] is [`Fits]: {!submit} takes any parts {!part} makes, waiting
    inside for command buffers when the queue is full. The C form answers
    [NX_NEVER] for a part that is no fill, which {!part} never makes. *)

val submit :
  t ->
  v:int ->
  waits:([ `Word | `Equal | `Object ] * int * int) array ->
  handles:nativeint array ->
  part array ->
  [ `Ok | `Failed of string ]
(** [submit d ~v ~waits ~handles ps] runs [ps] in order as the work of value
    [v], after [d]'s earlier work: each fill in a compute encoder of its own,
    which runs after the encoders before it, and returns once every command
    buffer of [v] is committed. [v] is observable in {!word} once they all
    completed. With no part, [v] is observable once the work before it
    completed. [handles] is ignored: every region of [d] is resident.

    The caller holds [d]'s turn (Domains, above) and calls {!room} before.

    The result is [`Failed why] if a fill returned a failure, or Metal made no
    command buffer or no encoder. Then [v] is never observable, {!sleep} raises
    {!Fault}, and every later [submit] answers the same [`Failed why] and runs
    nothing.

    Raises [Invalid_argument] if [v] is not the value after the last one
    [submit] received ([1] first), if [waits] is not empty, or if an [after]
    index of a part is not below its own. It releases the domain lock while it
    runs. *)

val room_entry : nativeint
(** [room_entry] is the address of [device_metal_room], {!room} for C, with the
    prototype [device_metal.h] states. *)

val submit_entry : nativeint
(** [submit_entry] is the address of [device_metal_submit], {!submit} for C,
    with the prototype [device_metal.h] states. It calls no function of the
    OCaml runtime: its caller releases the domain lock. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word d] is [d]'s timeline word: eight bytes of host memory holding, as an
    unsigned 64-bit integer in the host's byte order, the last value [v] such
    that every submission up to [v] completed. The device's completion handler
    writes it with release order, and never lowers it. After {!stop} answered
    [`Stopped], it holds the last value {!submit} received, whatever that work
    did. It lives while the process runs. *)

val signaled : t -> int
(** [signaled d] is the value in {!word}, read with acquire order: every
    submission up to it completed, and its writes are visible to the reader,
    unless {!stop} answered [`Stopped]. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep d ~seen ~still_ms] returns once {!word} holds a value other than
    [seen], at once if it already does, or after [still_ms] milliseconds,
    whichever comes first. It blocks on a condition the completion handler
    signals, using no processor time while it waits. It may raise the calling
    thread's scheduling class for the wait, and it restores the thread's own
    class before it returns. It releases the domain lock while it waits.

    Raises {!Fault} if a submission of [d] failed, with the failure's reason:
    Metal's for a failed command buffer, such as
    ["Impacting Interactivity
     (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"].

    Raises [Invalid_argument] if [still_ms < 0]. *)

(** {1:loss Loss} *)

exception Fault of string
(** [Fault why] is raised by {!sleep} once a submission failed. *)

val stop : t -> [ `Stopped | `Unknown ]
(** [stop d] answers [`Stopped] if every command buffer [d] committed has
    completed: it then writes the last value {!submit} received into {!word},
    releases [d]'s queue, and no work of [d] writes memory again. It answers
    [`Unknown] if work is still in flight, which may write memory for as long as
    it runs, and keeps the queue. It never waits. Regions end at {!free} and
    {!unmap}, which may follow. *)
