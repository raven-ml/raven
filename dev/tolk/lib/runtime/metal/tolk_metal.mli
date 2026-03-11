(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Metal GPU device backend.

    [Tolk_metal] provides a {!Tolk.Device.t} that executes compiled kernels on
    Apple Metal GPUs. Construct a device with {!create} and interact with it
    through the {!Tolk.Device} interface.

    For batched multi-kernel execution, {!Icb} encodes a sequence of compute
    dispatches into a Metal indirect command buffer that can be replayed with a
    single GPU submission.

    {1:compilation Kernel compilation}

    Kernels are compiled in two stages. The compiler first attempts offline
    compilation via Apple's private MTLCompiler framework (source to MTLB
    binary). When that framework is unavailable, it falls back to runtime source
    compilation through the Metal API.

    {1:env Environment variables}

    - [METAL_FAST_MATH] — when set to a non-zero integer, enables fast-math mode
      for runtime source compilation (the fallback path). Defaults to [0]
      (disabled). *)

(** {1:device Device} *)

val create : string -> Tolk.Device.t
(** [create name] is a Metal device identified by [name].

    The device uses the system default Metal GPU, an LRU-cached shared-memory
    allocator with blit-based buffer transfers, and the {!Tolk.Cstyle.metal}
    renderer. An {!Stdlib.at_exit} handler synchronizes in-flight work and
    releases the underlying Metal device and command queue.

    Raises [Failure] if no Metal GPU is available (e.g. running in a VM or on
    unsupported hardware). *)

(** {1:state Device state} *)

module State : sig
  type t
  (** The type for Metal device state. Holds the GPU device handle, command
      queue, shared timeline event, and in-flight command buffer list. *)

  val create : unit -> t
  (** [create ()] initializes the system default Metal device, command queue,
      and shared event.

      Raises [Failure] if no Metal GPU is available. *)

  val synchronize : t -> unit
  (** [synchronize t] blocks until all in-flight command buffers complete. After
      return, the in-flight list is empty.

      Raises [Failure] if any command buffer completed with an error. *)

  val shutdown : t -> unit
  (** [shutdown t] synchronizes and releases all Metal resources (command queue,
      shared event, device). Subsequent calls are no-ops. *)

  val is_virtual : t -> bool
  (** [is_virtual t] is [true] iff the device name contains ["virtual"],
      indicating a paravirtualized Metal device (e.g. macOS VM). ICB-based graph
      execution is unreliable on virtual devices. *)
end

(** {1:icb Indirect command buffers}

    An indirect command buffer (ICB) pre-encodes a fixed sequence of compute
    dispatches that can be replayed with a single GPU submission. Buffers and
    dispatch dimensions can be updated between replays without re-encoding the
    full command sequence.

    Typical usage:
    + {!Icb.create} to allocate the ICB.
    + {!Icb.encode} for each kernel in the batch.
    + {!Icb.execute} to submit.
    + {!Icb.update_buffer} / {!Icb.update_dispatch} then {!Icb.execute} for
      subsequent iterations.
    + {!Icb.release} when done. *)

module Icb : sig
  type t
  (** The type for indirect command buffers. *)

  val create : State.t -> count:int -> t
  (** [create state ~count] allocates an ICB with capacity for [count] compute
      commands.

      Raises [Failure] if Metal cannot allocate the ICB. *)

  val encode :
    t ->
    index:int ->
    program:nativeint ->
    buffers:nativeint array ->
    arg_buf:nativeint ->
    arg_offsets:int array ->
    global:int array ->
    local:int array ->
    unit
  (** [encode t ~index ~program ~buffers ~arg_buf ~arg_offsets ~global ~local]
      encodes a compute dispatch at command [index] with:
      - [program] — pipeline handle from {!Tolk.Device.Program.entry_addr}.
      - [buffers] — kernel buffer bindings (array of Metal buffer addresses).
      - [arg_buf] — Metal buffer holding packed [int32] variable parameters, or
        [0n] if there are none.
      - [arg_offsets] — byte offsets into [arg_buf] for each variable parameter.
      - [global] — threadgroup grid dimensions, length 3.
      - [local] — threads per threadgroup, length 3.

      A memory barrier is inserted after the dispatch so commands execute in
      order.

      Raises [Failure] if [local] threads exceed the pipeline's maximum. *)

  val update_buffer : t -> index:int -> buf_index:int -> buf:nativeint -> unit
  (** [update_buffer t ~index ~buf_index ~buf] replaces the buffer at binding
      [buf_index] for command [index]. The buffer offset is set to [0]. *)

  val update_dispatch :
    t -> index:int -> global:int array -> local:int array -> unit
  (** [update_dispatch t ~index ~global ~local] updates the threadgroup
      dimensions for command [index]. Both arrays must have length 3. *)

  val execute :
    State.t ->
    t ->
    resources:nativeint array ->
    pipelines:nativeint array ->
    unit
  (** [execute state t ~resources ~pipelines] submits the ICB for GPU execution.

      [resources] are Metal buffer handles marked for read and write access by
      the GPU. Every buffer referenced by encoded commands must appear here.

      [pipelines] are pipeline handles for the M1/M2 ICB workaround: on pre-M3
      GPUs (AGXG family < 15), a zero-size dummy dispatch is issued per pipeline
      before executing the ICB to prevent
      [kIOGPUCommandBufferCallbackErrorInvalidResource] crashes. On M3+ the
      array is ignored.

      The resulting command buffer is appended to the in-flight list. *)

  val release : t -> unit
  (** [release t] frees the underlying Metal ICB. *)
end
