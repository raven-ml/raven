(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a Metal device.

    Compiled code drives a Metal device through two things its driver
    provides. The record {!t} makes indirect command buffers, gives the
    address of {!field-split} and states the device's argument alignment. A
    {e fill} is a C function of the compiled code that the driver's [submit]
    calls to encode the code's work. Making an indirect command buffer calls
    Metal on the device's objects, which only the driver holds.

    This library holds only that agreement. The driver makes the record when
    it opens the device and declares it ({!declare}), compiled code finds it
    under {!key}, and neither links the other. *)

(** {1:fills Fills}

    A fill is work for the device's queue given as a C function,
    [int fill(void *queue, void *arg, uint64_t v)]. The driver calls it with:
    - [queue], a pointer to a word that holds the open compute command
      encoder, an [id<MTLComputeCommandEncoder>] the driver made before the
      call. [queue] is valid only during the call;
    - [arg], the argument the work was given;
    - [v], the value the work completes on the device's timeline.

    The encoder runs its commands in order, each after the one before has
    completed ([MTLDispatchTypeSerial]), and after the device's earlier work.
    Every allocation of the device is resident while they run. The encoder and
    its command buffer may already hold commands of earlier work, and later
    work may add to them, so a fill must set every state its commands read,
    such as a dispatch's pipeline, buffers and threadgroup memory lengths.

    A fill must encode into that encoder only, for instance with
    [executeCommandsInBuffer:withRange:]. It must not end the encoder or make
    another encoder or command buffer. The driver orders the fill's work after
    earlier work and signals [v], so a fill does neither. To start a new
    command buffer, a fill calls {!field-split}, as often as it needs. It must
    stop at the first [split] that fails and return that failure, and return
    [0] otherwise.

    The driver ends encoders and commits command buffers when it chooses, and
    a [split] is the only boundary a fill sets. The driver commits a fill's
    commands no later than when the command buffers committed before them
    have completed. [v] is therefore reached once every command buffer
    holding the fill's commands has completed, without a later submission.
    The driver loses the device if a fill returns a failure, or, with Metal's
    reason, if a command buffer holding the fill's commands fails. *)

(** {1:icbs Indirect command buffers}

    An {e indirect command buffer} is a sequence of dispatches recorded once
    and run by every submit of the submission that holds it. Its [handle],
    [commands] and their pipelines live while its value is reachable, even
    after the pipelines' image is unloaded. Once the value is collected, the
    driver releases them, even if it has stopped the device. *)

type dispatch = {
  pipeline : int;
      (** Its [MTLComputePipelineState]: the [code] of an entry of an image
          loaded on the device. *)
  offset : int;
      (** Where its arguments start in the argument buffer, which it binds as
          its kernel buffer [0], in bytes from the [MTLBuffer]'s first byte:
          for a rig buffer [a], [Rig.Buffer.offset a] plus the offset in [a].
          It must be a multiple of {!field-align}, which [icb] checks, and of
          the alignment of the kernel's argument structure, that of its
          largest member, which nothing checks, such as [8] for a structure
          that holds device pointers. *)
  groups : int * int * int;
      (** Its threadgroups per grid, in x, y and z, each at least [1]. *)
  threads : int * int * int;
      (** Its threads per threadgroup, in x, y and z, each at least [1]. *)
}
(** The type for the dispatches an indirect command buffer records. *)

type icb = private {
  handle : nativeint;  (** The [MTLIndirectCommandBuffer]. *)
  commands : nativeint array;
      (** The [MTLIndirectComputeCommand] of each dispatch, in order. A fill
          may change a command, such as its threadgroups per grid with
          [concurrentDispatchThreadgroups:threadsPerThreadgroup:], before its
          first run of [handle], and only while no earlier work that runs
          [handle] is in flight. *)
}
(** The type for indirect command buffers.

    {b Warning.} A submission whose work runs [handle] must keep the value
    ([Rig.Submission.make ~keep:icb]), or the buffer may be released while
    its work runs. A buffer collected between two of the submission's submits
    is released with its pipelines, and the next submit's fill encodes the
    released buffer. *)

val icb :
  handle:nativeint -> commands:nativeint array -> collect:(unit -> unit) -> icb
(** [icb ~handle ~commands ~collect] is the value of a native indirect command
    buffer that a driver made. The driver owns [handle], [commands] and their
    pipelines until this call returns. From then on [collect], which runs once
    the value is collected, releases them. A driver must pass each native
    buffer at most once.

    [collect] must keep no reference to the value, must neither block nor
    raise, and must leave inherited Metal objects untouched in a child of
    [fork]. *)

(** {1:record The record} *)

type t = {
  align : int;
      (** [align] is the device's minimum constant buffer offset alignment, in
          bytes, at least [1]. Metal requires a buffer a kernel reads as
          constant data to start at a multiple of it. *)
  icb : nativeint -> dispatch array -> (icb, string) result;
      (** [icb buffer ds] is [Ok b] with [b] an indirect command buffer that
          records one dispatch per element of [ds], in order, each run after
          the one before it has completed. [buffer] is the argument buffer, an
          [MTLBuffer] of the device, such as a rig buffer's
          [Rig.Buffer.handle]. [ds] may be empty. Any domain may call [icb] at
          any time, and {!Rig_edge.Driver.stop} waits for a call of [icb] in
          flight.

          {b Warning.} [b] does not keep [buffer]. The caller keeps it until
          the work that runs [b] is done: for a rig buffer [args], by keeping
          [args] with that work's submission
          ([Rig.Submission.make ~keep:(b, args)]).

          The result is [Error] with the reason if a dispatch asks for more
          threads per threadgroup than its pipeline allows, if Metal cannot
          make the indirect command buffer, or once the driver has begun to
          stop the device.

          Raises [Invalid_argument] if:
          {ul
          {- [buffer] is not a live region of the device;}
          {- a pipeline belongs to no image loaded on the device;}
          {- a dispatch's offset is negative, lies outside [buffer] or is not
             a multiple of {!field-align};}
          {- one of a dispatch's sizes is less than [1].}}

          A call from a callback inside a lock ([Rig]'s domains section)
          raises [Invalid_argument], having changed nothing. *)
  split : nativeint;
      (** [split] is the address of
          [int split(void *queue, uint64_t *start, uint64_t *end)], which ends
          the open command buffer and starts a new one. It ends the open
          encoder, commits its command buffer, and stores at [queue] the
          encoder of a new command buffer, which runs as the first did. While
          the device's queue holds as many command buffers as it can, [split]
          waits until one of them completes, so a fill may make more command
          buffers than the queue holds. Only a fill, during its call, may call
          [split], and the address is valid while the device is open.

          Unless [start] is [NULL], [split] writes at [start] the time the
          committed command buffer started on the GPU, and unless [end] is
          [NULL], the time it ended at [end]. The command buffer may hold
          commands of earlier work before the fill's, and its times cover them
          too. Both are written before [v] is reached. Times are nanoseconds
          of the host clock ([CLOCK_UPTIME_RAW], the clock of Metal's
          [GPUStartTime]), as unsigned 64-bit integers in the host's byte
          order.

          [split] returns [0], or a nonzero failure if Metal cannot make a new
          command buffer ([commandBuffer] returns [nil]) or an encoder for it
          ([computeCommandEncoderWithDispatchType:] returns [nil]). *)
}
(** The type for what compiled code needs from a Metal device. *)

val key : t Type.Id.t
(** [key] is the key of a Metal device's record. Compiled code finds the record
    of a device [d] with [Rig.capability d key], which is [None] if [d] is
    not a Metal device. *)

val declare : t -> Rig_edge.capability
(** [declare c] is [c] under {!key}, as the facts of a Metal device carry it
    ({!Rig_edge.type-facts}). *)

(** {1:references References}
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
       feature set tables}, Resources: minimum constant buffer offset
      alignment. *)
