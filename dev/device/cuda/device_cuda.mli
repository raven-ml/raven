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

    A program that uses a GPU alone opens it, allocates and submits, then waits
    for the word:
    {[
    let g = Result.get_ok (Device_cuda.open_ 0) in
    let src = Option.get (Device_cuda.alloc g `Pinned 4096) in
    let dst = Option.get (Device_cuda.alloc g `Device 4096) in
    let copy = `Copy ((dst, 0), (src, 0), 4096) in
    let p = Device_cuda.part g ~queue:"COPY:0" copy in
    match Device_cuda.submit g ~v:1 ~waits:[||] ~handles:[||] [| p |] with
    | `Ok ->
        let rec wait () =
          let seen = Device_cuda.signaled g in
          if seen < 1 then (
            Device_cuda.sleep g ~seen ~still_ms:200;
            wait ())
        in
        wait ()
    | `Failed why -> prerr_endline why
    ]}

    {b The CUDA library} ([libcuda]) is loaded at the first call of {!count} or
    {!open_}, by its standard name from the platform's library search path:
    [libcuda.so.1] (or [libcuda.so]) on Linux, [nvcuda.dll] on Windows,
    [libcuda.dylib] on macOS. A library of that name earlier on the path is
    loaded in its place, for the whole process. Building and linking this
    library needs no CUDA installation.

    {b Requirements.} The device writes its timeline from its streams with
    64-bit stream memory operations, and its memory and the host's share one
    address space (unified addressing). A GPU without either does not open;
    there is no fallback.

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
    {!val-image}, {!entry}, {!unload} and {!sleep}; {!submit} answers [`Failed]
    instead. Misuse, such as a region of another device or a value out of order,
    raises [Invalid_argument].

    {b Domains.} Every value may be called from any domain, at the same time as
    others, with three exceptions. {!room} and {!submit}, and their C forms, run
    one call at a time, in value order: their caller serialises them. {!stop} is
    called once, after every other call returned; after it only {!free} and the
    [symbol] function of {!val-capability} are called. {!sleep} may run while
    another domain submits.

    {b References.}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}: Primary
      Context Management, Stream Management, Event Management, Stream Memory
      Operations ([cuStreamWaitValue64], [cuStreamWriteValue64],
      [cuStreamBatchMemOp], [CUstreamWaitValue_flags]), Memory Management
      ([cuMemHostAlloc], [cuMemHostRegister]), Unified Addressing, Peer Context
      Memory Access, Module Management, Error Handling. *)

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
    initialised, if [i >= count ()], if the GPU lacks 64-bit stream memory
    operations or unified addressing, while a device of GPU [i] is open or was
    stopped while its work still ran and that work runs on, or with CUDA's
    error, such as the error a fault left in the GPU's context. A GPU has one
    device at a time: two would wait on each other's words through the context's
    shared hardware queues, ordering CUDA does not see.

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

val blocks : t -> [ `Returns | `May_block ]
(** [blocks g] is [`May_block]: {!submit} calls CUDA, which may block. *)

type capability = Device_cuda_abi.t
(** The type for what compiled code needs from a device. *)

val capability : t -> capability
(** [capability g] finds the functions of the CUDA library [g] was opened with.
*)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Device_cuda_abi.key}. *)

val self : t -> nativeint
(** [self g] is the address of [g]'s state, the first argument of
    [device_cuda_room] and [device_cuda_submit]. It is valid while the process
    runs: a device's C state holds its {!word}, which other devices may read
    after [g] is gone, so neither is ever freed. *)

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
    that overlaps them is [None].

    Raises [Invalid_argument] if [r] is another device's or {!word}, or was
    freed. *)

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
    page-locked for every CUDA device and mapped, unless CUDA refuses to
    page-lock them. CUDA refuses read-only memory and memory whose pages another
    page-locked range shares. The host memory must stay mapped until [r] is
    freed.

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
    itself. CUDA may wait for all of the GPU's work before it loads [bin], and
    [image] lets other domains run meanwhile. The result is [Error msg] with
    CUDA's error if CUDA refuses [bin], for instance a cubin for another GPU. *)

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

(** {1:work Work} *)

type part
(** The type for work for one queue of a device. *)

val part :
  t ->
  queue:string ->
  ?after:int array ->
  [ `Words of int array
  | `Fill of nativeint * nativeint * int * int
  | `Copy of (region * int) * (region * int) * int ] ->
  part
(** [part g ~queue ~after w] is the work [w] for [g]'s queue [queue], one of
    {!queues}. [after] (defaults to [[||]]) holds the indices, in the array
    given to {!submit}, of the parts of its submission that it runs after, each
    smaller than its own. Parts on one queue run in array order; parts on two
    queues that [after] does not order may run at once. [w] is:
    - [`Fill (f, arg, units, bytes)], a C function [f] that enqueues its work on
      the queue's stream. {!submit} calls it as [f stream arg v], with [g]'s
      context current on the calling thread and [v] the submission's value, as
      {!Device_cuda_abi} states. Nothing bounds what a fill enqueues, so it
      declares no room: [units] and [bytes] are [0], and {!val-part} refuses any
      other;
    - [`Copy ((dst, o), (src, o'), n)], a copy of the [n] bytes of [src] at
      offset [o'] to [dst] at offset [o], any two regions of [g]. The caller
      keeps the two ranges apart: a copy between overlapping ranges writes
      undefined bytes.

    No part writes a timeline word, [g]'s or another device's: only the devices'
    streams write their words, and a word the work wrote could move backwards or
    claim work that has not completed. Nothing checks it, since a fill can store
    anywhere.

    Raises [Invalid_argument] if [queue] is not a queue of [g], if [w] is
    [`Words _], which names ring words a CUDA device has not, if [units] or
    [bytes] is not [0], if a copy's range lies outside its region, if a region
    is of another device or was freed, or if an index of [after] is negative.

    A part names its regions until it is submitted: the caller frees none of
    them before. *)

val room : t -> part array -> [ `Fits | `Later | `Never ]
(** [room g ps] is [`Fits]: CUDA's streams take any amount of work, and a call
    of {!submit} that finds a stream full waits for earlier work to free it. Its
    C form, [device_cuda_room], answers [NX_NEVER] for a part {!val-part}
    refuses: one with words, ring units or segment bytes, or on no queue of the
    device. *)

val submit :
  t ->
  v:int ->
  waits:([ `Word | `Object ] * int * int) array ->
  handles:nativeint array ->
  part array ->
  [ `Ok | `Failed of string ]
(** [submit g ~v ~waits ~handles ps] hands over [ps] as [g]'s value [v], the
    value after the last one [g] was given. Each wait [(`Word, a, w)] holds the
    work back until the aligned 64-bit word at address [a], which [g]'s work
    addresses, holds at least [w], compared circularly: [x] is at least [w] if
    [x - w], as a signed 64-bit integer, is not negative. The work runs after
    every earlier value of [g] and after the waits; once it completed, the
    timeline word holds [v]. A submission of no parts writes [v] after its waits
    and after every earlier value. [handles] is ignored: CUDA's work names its
    memory by address.

    The result is [`Ok] once every part is enqueued, or [`Failed why] with the
    step and the error of the first CUDA call that failed, a fill's included, as
    ["running a fill: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was
     encountered"]. The parts enqueued before the failure may run; the others
    never do. A failed device stays failed, since CUDA may keep the context's
    error for the process: every later [submit] enqueues none of its parts and
    answers the same [`Failed].

    The timeline word still reaches every value, so that work waiting on it
    elsewhere runs on: a failed [submit], and every later one, writes [v] after
    its waits, every earlier value and the work enqueued for [v]. It writes
    nothing once the context failed, whose work no longer runs, or if CUDA
    refuses a call that orders the write.

    [submit] may block while a stream is full, until the device's earlier work
    completes, and lets other domains run meanwhile.

    Raises [Invalid_argument] if [v] is not the value after the last one, if a
    part is another device's, if a part's [after] names a part at or after its
    own index, or if a wait is [`Object]: the device waits only for words to
    reach a value. *)

val room_entry : nativeint
(** [room_entry] is the address of the C function [device_cuda_room], {!room}
    for C, which [device_cuda.h] declares. *)

val submit_entry : nativeint
(** [submit_entry] is the address of the C function [device_cuda_submit],
    {!submit} for C, which [device_cuda.h] declares. It is called without the
    domain lock, and calls no function of the OCaml runtime. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word g] is [g]'s timeline word: eight bytes of page-locked host memory
    holding, as an unsigned 64-bit integer in the host's byte order, the last
    value [v] such that the work of every value up to [v] completed. [g]'s
    streams write it after a fence that makes the work's writes visible
    ([cuStreamWriteValue64] with [CU_STREAM_WRITE_VALUE_DEFAULT], a system-wide
    memory fence before the write); it never decreases. Other devices may map it
    and wait on it. It is never freed: another device's work may still read it
    after [g] is stopped or collected. *)

val signaled : t -> int
(** [signaled g] is the value in {!word}, read with acquire order: the work of
    every value up to it completed, and its writes are visible to the reader. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep g ~seen ~still_ms] returns once [g]'s timeline word differs from
    [seen], at once if it does already, or after [still_ms] milliseconds;
    [still_ms] is not negative. It asks CUDA each millisecond whether [g]'s
    streams met an error ([cuStreamQuery]), so it finds a fault at most a
    millisecond after CUDA reports it. It lets other domains run while it waits,
    and may run while {!submit} does.

    Raises {!exception-Fault} with CUDA's error if [g]'s work met one. *)

(** {1:loss Loss} *)

exception Fault of string
(** The exception for a fault of a device's work, with CUDA's error. *)

val stop : t -> unit
(** [stop g] stops [g] for good, without waiting. If [g]'s work no longer writes
    memory (the work of every value it was given completed, its streams are
    idle, or a fault ended the context's work), the timeline word holds at least
    the last value {!submit} was given when [stop] returns, so work of other
    devices that waits on it runs on, and [g]'s streams are destroyed. Otherwise
    the timeline word reaches the last value {!submit} was given once that work
    ends, unless it waits on a word of another device that never reaches its
    value; the GPU opens again once that work ends. After [stop], only {!free}
    and the [symbol] function of {!val-capability} may be called on [g]; neither
    raises {!exception-Fault}. *)
