(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a CUDA device.

    Code compiled for a CUDA device runs on the host. It enqueues the device's
    work on a stream by calling the CUDA library the device was opened with
    ([libcuda]): [cuLaunchKernel] for a kernel, [cuMemcpyAsync] for a copy,
    [cuLaunchHostFunc] for a timestamp. Two things connect it to the device's
    driver, the library that opened the device: a record, {!t}, that finds the
    CUDA library's functions, and a C calling convention, the fill, by which the
    driver runs the compiled code's work.

    This library holds only that agreement. The driver makes the record when it
    opens the device and declares it under {!key}; compiled code finds it there.
    Neither links the other.

    {b Fills.} Work for one of the device's queues can be a C function, a
    {e fill}: [int fill(void *queue, void *arg, uint64_t v)]. The driver calls
    it with:
    - [queue], the queue's stream ([CUstream]), with the device's context
      current on the calling thread;
    - [arg], the argument the work was given with;
    - [v], the value the work completes on the device's timeline.

    The fill enqueues its work on that stream, and does nothing else: waiting
    for earlier work and signalling [v] are the driver's. It stops at the first
    call that fails and returns that call's [CUresult], or returns [0]
    ([CUDA_SUCCESS]) once every call succeeded. A call can fail for earlier work
    on the device, which CUDA reports at a later call. The driver loses the
    device on any failure, returned by the fill or met by its work after the
    fill returned.

    {b References.}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}:
      Execution Control ([cuLaunchKernel], [cuLaunchHostFunc]), Memory
      Management ([cuMemcpyAsync]) and Data types used by CUDA driver
      ([CUstream], [CUresult]). *)

type t = {
  symbol : string -> nativeint option;
      (** [symbol name] is [Some a] if the CUDA library exports a function named
          [name], with [a] its address, and [None] otherwise. [name] is the
          exported name, whose suffix selects a version of the function's
          interface: ["cuStreamWaitValue64_v2"] is version 2 of
          ["cuStreamWaitValue64"]. [a] is valid while the device is open. Any
          domain may call [symbol]. *)
}
(** The type for what compiled code needs from a CUDA device. *)

val key : t Type.Id.t
(** [key] is the key of a CUDA device's {!t}. The device's driver declares its
    record under [key]; a device of another kind has none. *)
