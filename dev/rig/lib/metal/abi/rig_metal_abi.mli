(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a Metal device.

    Code compiled for a Metal device runs a step's kernels from an
    {e indirect command buffer}: dispatches recorded once, when the step is
    linked, and run on each submission of the step. Making one calls Metal on
    the device's objects, which only the device's driver, the library that
    opened the device, holds. Two things connect compiled code to the driver: a
    record, {!t}, that makes indirect command buffers, gives the address of
    {!field-split} and states the device's argument alignment, and a C calling
    convention, the fill, by which the driver runs the compiled code's work.

    This library holds only that agreement. The driver makes the record when it
    opens the device and declares it under {!key}; compiled code finds it there.
    Neither links the other.

    {b Fills.} Work for the device's queue can be a C function, a {e fill}:
    [int fill(void *queue, void *arg, uint64_t v)]. The driver calls it with:
    - [queue], a pointer to a word that holds the open compute command encoder,
      an [id] the driver made before the call. [queue] is valid only during the
      call;
    - [arg], the argument the work was given with;
    - [v], the value the work completes on the device's timeline.

    The encoder runs its commands in order, each after the one before completed
    ([MTLDispatchTypeSerial]), after the device's earlier work, and every memory
    of the device is resident while it runs. The encoder and its command buffer
    may already hold commands of earlier work, and later work may add to them,
    so a fill may find state that earlier commands set: it sets every state its
    commands read, such as a dispatch's pipeline, buffers and threadgroup memory
    lengths.

    The fill encodes into that encoder, for instance
    [executeCommandsInBuffer:withRange:], and does nothing else: it ends no
    encoder and makes no other encoder or command buffer, and waiting for
    earlier work and signalling [v] are the driver's. To start a new command
    buffer it calls {!field-split}; nothing bounds how many command buffers a
    fill makes. A fill declares no room: its ring units and segment bytes are
    [0], and the driver refuses work that declares any other. It stops at the
    first [split] that fails and returns its failure, and returns [0] otherwise.

    The driver ends encoders and commits command buffers when it chooses: a
    [split] is the only boundary a fill sets. It commits a work's commands no
    later than when the command buffers committed before them complete, so [v]
    is reached without a later submission, once every command buffer holding the
    work completed. The driver loses the device if a fill returns a failure, or,
    with Metal's reason, if a command buffer holding the work fails.

    {b References.}
    - Apple's Metal framework headers (macOS 26 SDK):
      [MTLIndirectCommandBuffer.h], [MTLIndirectCommandEncoder.h]
      ([setBarrier]), [MTLComputeCommandEncoder.h]
      ([executeCommandsInBuffer:withRange:], [useResources:count:usage:]),
      [MTLCommandBuffer.h] ([GPUStartTime], [GPUEndTime], [MTLDispatchType],
      [computeCommandEncoderWithDispatchType:]).
    - [MTLCommandQueue.h] ([commandBuffer], [maxCommandBufferCount]) and
      {{:https://developer.apple.com/documentation/metal/mtlcommandqueue/makecommandbuffer()}
       makeCommandBuffer()}: a full queue blocks until the GPU finishes a
      command buffer.
    - {{:https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime}
       gpuStartTime}: host time relative to mach time; [clock_gettime(3)]:
      [CLOCK_UPTIME_RAW] is mach time.
    - {{:https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu}
       Encoding indirect command buffers on the CPU}: the buffers that the
      commands of an indirect command buffer use must be declared resident.
    - {{:https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf}Metal
       feature set tables} (May 21, 2026), Resources: minimum constant buffer
      offset alignment. *)

(** {1:icbs Indirect command buffers} *)

type dispatch = {
  pipeline : int;
      (** Its [MTLComputePipelineState]: the address an entry of an image the
          device loaded gives. *)
  offset : int;
      (** Where its arguments start in the argument buffer, which it binds as
          its kernel buffer [0]: bytes from the buffer's first byte, a multiple
          of the record's {!field-align}. *)
  groups : int * int * int;
      (** Its threadgroups per grid, in x, y and z, each at least [1]. *)
  threads : int * int * int;
      (** Its threads per threadgroup, in x, y and z, each at least [1]. *)
}
(** The type for the dispatches an indirect command buffer records. *)

type icb = {
  handle : nativeint;  (** The [MTLIndirectCommandBuffer]. *)
  commands : nativeint array;
      (** The [MTLIndirectComputeCommand] of each dispatch, in order. A fill may
          change one before it first runs [handle], while no earlier work that
          runs [handle] is in flight: for instance its threadgroups per grid,
          with [concurrentDispatchThreadgroups:threadsPerThreadgroup:]. *)
  release : unit -> unit;
      (** [release ()] releases [handle], [commands] and the pipelines they
          hold. The owner of the linked step that made them calls it once, after
          the last work that ran [handle] completed: until then they live,
          whatever happens to their pipelines' image. Raises [Invalid_argument]
          if called twice. Any domain may call it, also after the driver stopped
          the device. *)
}
(** The type for indirect command buffers. *)

(** {1:record The record} *)

type t = {
  align : int;
      (** [align] is the device's minimum constant buffer offset alignment, in
          bytes, at least [1]. Metal requires a buffer a kernel reads as
          constant data to start at a multiple of it, so every dispatch's
          {!field-offset} is a multiple of it. *)
  icb : nativeint -> dispatch array -> (icb, string) result;
      (** [icb buffer ds] is [Ok b] with [b] an indirect command buffer that
          records one dispatch per element of [ds], in order, each run after the
          one before it completed. [buffer] is the argument buffer, an
          [MTLBuffer] of the device. [ds] may be empty.

          The result is [Error msg] if a dispatch asks for more threads per
          threadgroup than its pipeline allows
          ([maxTotalThreadsPerThreadgroup]), if Metal cannot make the indirect
          command buffer, or once the driver began to stop the device; a stop
          waits for an [icb] call in flight.

          Raises [Invalid_argument] if [buffer] or a pipeline belongs to another
          [MTLDevice], or if a dispatch's offset lies outside [buffer] or is not
          a multiple of {!field-align}, or one of its sizes is less than [1].
          Any domain may call it at any time. *)
  split : nativeint;
      (** [split] is the address of
          [int split(void *queue, uint64_t *start, uint64_t *end)], which a fill
          calls to end the open command buffer and start a new one. It ends the
          open encoder, commits its command buffer, and stores at [queue] the
          encoder of a new command buffer, which runs as the first did. When the
          queue holds as many command buffers as it can, [split] waits until one
          of them completes, so one work may make more command buffers than the
          queue holds.

          Unless [start] is [NULL], [split] writes at [start] the time the
          committed command buffer started on the GPU; unless [end] is [NULL],
          the time it ended at [end]. The command buffer may hold commands of
          earlier work before the fill's, and its times cover them too. Both are
          written before [v] is reached. Times are nanoseconds of the host clock
          ([CLOCK_UPTIME_RAW], the clock of Metal's [GPUStartTime]), as unsigned
          64-bit integers in the host's byte order.

          [split] returns [0], or a nonzero failure that the fill returns as its
          own. It fails if:
          - Metal makes no new command buffer ([commandBuffer] returns [nil]);
          - Metal makes no encoder for it
            ([computeCommandEncoderWithDispatchType:] returns [nil]).

          The address is valid while the device is open. Only a fill, during its
          call, may call [split]. *)
}
(** The type for what compiled code needs from a Metal device. *)

val key : t Type.Id.t
(** [key] is the key of a Metal device's {!t}. The device's driver declares its
    record under [key]; a device of another kind has none. *)
