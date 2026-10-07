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

    {b Fills.} Work for the device's queue can be a C function, a {e fill}:
    [int fill(void *queue, void *arg, uint64_t v)]. [queue] points at a word
    that holds the open compute command encoder, an [id] the driver made before
    the call: its command buffer runs after the device's earlier work, and every
    memory of the device is resident while it runs. The fill encodes into that
    encoder, for instance [executeCommandsInBuffer:withRange:], and nothing
    else: it ends no encoder and makes no other encoder or command buffer, and
    waiting for earlier work and signalling [v] are the driver's. {!field-split}
    starts a new command buffer. The fill stops at the first call that fails and
    returns the failure [split] returned, or returns [0] once every call
    succeeded. [queue] is valid only during the call.

    After the fill returns, the driver ends and commits the last command buffer.
    The value [v] is reached once every command buffer of the work completed; if
    one fails, Metal's reason loses the device. A fill whose work declares [r]
    ring units makes at most [r] command buffers: it splits at most [r - 1]
    times.

    {b References.}
    - Apple's Metal framework headers (macOS 26 SDK):
      [MTLIndirectCommandBuffer.h], [MTLIndirectCommandEncoder.h]
      ([setBarrier]), [MTLComputeCommandEncoder.h]
      ([executeCommandsInBuffer:withRange:], [useResources:count:usage:]),
      [MTLCommandBuffer.h] ([GPUStartTime], [GPUEndTime]).
    - {{:https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu}
       Encoding indirect command buffers on the CPU}: the buffers that the
      commands of an indirect command buffer use must be declared resident. *)

type dispatch = {
  pipeline : nativeint;
      (** Its [MTLComputePipelineState]: an entry of an image the device loaded.
      *)
  offset : int;
      (** Where its arguments start in the indirect command buffer's argument
          buffer, bound as its kernel buffer [0]: bytes from the argument
          buffer's first byte. *)
  groups : int * int * int;  (** Its threadgroups per grid. *)
  threads : int * int * int;  (** Its threads per threadgroup. *)
}
(** The type for the dispatches an indirect command buffer records. *)

type icb = {
  handle : nativeint;  (** The [MTLIndirectCommandBuffer]. *)
  commands : nativeint array;
      (** The [MTLIndirectComputeCommand] of each dispatch, in order. A fill may
          change one before it runs [handle], while no earlier work that runs
          [handle] is in flight: for instance its threadgroups per grid, with
          [concurrentDispatchThreadgroups:threadsPerThreadgroup:]. *)
  release : unit -> unit;
      (** [release ()] releases [handle], [commands] and the pipelines they
          hold. The linked step that made them calls it once, after the last
          work that ran [handle] completed; until then they live, whatever
          happens to their pipelines' image. Raises [Invalid_argument] if called
          twice. Any domain may call it. *)
}
(** The type for indirect command buffers. *)

type t = {
  icb : nativeint -> dispatch array -> (icb, string) result;
      (** [icb buffer ds] is an indirect command buffer with one concurrent
          dispatch per element of [ds], in order, each on its arguments in the
          argument buffer [buffer], an [MTLBuffer] of the device, and run after
          the one before it completed. [ds] may be empty.

          The result is [Error msg] if a dispatch asks for more threads per
          threadgroup than its pipeline allows
          ([maxTotalThreadsPerThreadgroup]), or if Metal cannot make the
          indirect command buffer.

          Raises [Invalid_argument] if [buffer] or a pipeline belongs to another
          [MTLDevice], an offset lies outside [buffer], or a size is less than
          [1]. Any domain may call it. *)
  split : nativeint;
      (** [split] is the address of
          [int split(void *queue, uint64_t *start, uint64_t *end)], which a fill
          calls to end the open command buffer and start a new one. It commits
          the open command buffer and stores at [queue] the encoder of the next,
          made as the first was. When the queue holds as many command buffers as
          it can, it waits until the work's own earlier command buffers
          complete, so one work may make more command buffers than the queue
          holds. Unless [start] is [NULL], it writes the time the committed
          command buffer started on the GPU at [start] before [v] is reached;
          likewise the time it ended at [end]. Times are nanoseconds of the host
          clock ([CLOCK_UPTIME_RAW], the clock of Metal's [GPUStartTime]), as
          unsigned 64-bit integers in the host's byte order.

          It returns [0], or a failure that the fill returns as its own: a
          failure if the fill already made as many command buffers as its work
          declared ring units. Only a fill, during its call, may call it. *)
}
(** The type for what compiled code needs from a Metal device. *)

val key : t Type.Id.t
(** [key] is the key a Metal device's {!t} is found under. *)
