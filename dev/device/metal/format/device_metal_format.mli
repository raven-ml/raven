(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a Metal device.

    Code compiled for a Metal device runs a step's kernels from an
    {e indirect command buffer}: dispatches recorded once, when the step is
    linked, and run on each submission of the step. Only the driver can make
    one, since making it calls Metal on the device's objects, so the driver
    gives the means as a {!t} when it opens the device, under {!key}.

    {b Fills.} A piece of work on the device's queue is a C function
    [int fill(void *queue, void *arg, uint64_t v)]. [queue] points at a word
    that holds the open compute command encoder, an [id] the driver made before
    the call: its command buffer runs after the device's earlier work, and every
    memory of the device is resident while it runs. The fill encodes into that
    encoder, for instance [executeCommandsInBuffer:withRange:], and nothing
    else: it ends no encoder and makes no other encoder or command buffer, and
    waiting for earlier work and signalling [v] are the driver's. {!field-split}
    starts a new command buffer. The fill returns [0], or a failure, and then
    encodes nothing after it. [queue] is valid only during the call.

    After the fill returns, the driver ends and commits the last command buffer.
    The value [v] is reached once every command buffer of the work completed; if
    one fails, Metal's reason loses the device. A fill that splits [n] times
    makes [n + 1] command buffers, the work's ring units, which the driver
    counts against the command buffers its queue holds.

    {b References.}
    - Apple's Metal framework headers (macOS 26 SDK):
      [MTLIndirectCommandBuffer.h], [MTLIndirectCommandEncoder.h]
      ([setBarrier]), [MTLComputeCommandEncoder.h]
      ([executeCommandsInBuffer:withRange:], [useResources:count:usage:]),
      [MTLCommandBuffer.h] ([GPUStartTime], [GPUEndTime]).
    - {{:https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu}
       Encoding indirect command buffers on the CPU}: the buffers that the
      commands of an indirect command buffer use must be declared resident. *)

type command = {
  pipeline : nativeint;
      (** Its [MTLComputePipelineState]: an entry of an image the device loaded.
      *)
  offset : int;
      (** The offset, in bytes, of its arguments in the indirect command
          buffer's buffer, its kernel buffer [0]. *)
  groups : int * int * int;  (** Its threadgroups per grid. *)
  threads : int * int * int;  (** Its threads per threadgroup. *)
}
(** The type for the dispatches of an indirect command buffer. *)

type t = {
  icb :
    nativeint -> command array -> (nativeint * nativeint array, string) result;
      (** [icb buffer cmds] is [Ok (icb, commands)]: [icb], an
          [MTLIndirectCommandBuffer] with one concurrent dispatch per command of
          [cmds], in order, each on its arguments in [buffer] and run after the
          one before it completed; and [commands], the
          [MTLIndirectComputeCommand] of each dispatch, in the same order, which
          a fill may change before it runs [icb], for instance with
          [concurrentDispatchThreadgroups:threadsPerThreadgroup:]. [buffer] is
          the [MTLBuffer] of a memory of the device. [icb], [commands] and the
          pipelines they hold live until that memory is freed to the driver,
          which happens after the work that ran them.

          The result is [Error msg] if a command asks for more threads per
          threadgroup than its pipeline allows
          ([maxTotalThreadsPerThreadgroup]), or if Metal cannot make [icb].

          Raises [Invalid_argument] if an offset lies outside [buffer], or a
          size is less than [1]. Any domain may call it. *)
  split : nativeint;
      (** [split] is the address of [int split(void *queue, uint64_t *times)],
          which a fill calls to end the open command buffer and start a new one.
          It commits the open command buffer and stores at [queue] the encoder
          of the next, made as the first was. Unless [times] is [NULL], it
          writes the committed command buffer's GPU start and end times into
          [times[0]] and [times[1]] before [v] is reached: nanoseconds of the
          host's monotonic clock, as unsigned integers in the host's byte order.
          It returns [0], or a failure that the fill returns as its own. Only a
          fill, during its call, may call it. *)
}
(** The type for what compiled code needs from a Metal device. *)

val key : t Type.Id.t
(** [key] is the key a Metal device's {!t} is found under. *)
