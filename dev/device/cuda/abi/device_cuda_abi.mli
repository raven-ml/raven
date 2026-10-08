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

    The fill enqueues its work on that stream and does nothing else. It may call
    [cuLaunchKernel], [cuGraphLaunch], [cuMemcpyAsync] and [cuLaunchHostFunc] on
    [queue] any number of times. Each call may block until the device's earlier
    work frees room in the stream; nothing bounds what a fill enqueues, so work
    given as a fill declares no room: [0] ring units and [0] segment bytes. The
    driver refuses work that declares any other. The fill does not wait for work
    ([cuStreamSynchronize], [cuEventSynchronize], [cuCtxSynchronize]), does not
    enqueue on another stream, and does not change the current context. Waiting
    for earlier work and signalling [v] are the driver's.

    A host function the fill enqueues runs on a thread of the CUDA library. It
    calls no CUDA function and takes no lock that a fill, or the code that calls
    the device's driver, may hold while a CUDA call blocks.

    The fill stops at the first call that fails and returns that call's
    [CUresult], or returns [0] ([CUDA_SUCCESS]) once every call succeeded. A
    call can fail because of earlier work on the device, which CUDA reports at a
    later call. The driver loses the device on any failure, whether the fill
    returned it or its work met it after the fill returned.

    {b References.}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}:
      Execution Control ([cuLaunchKernel], [cuLaunchHostFunc]), Graph Management
      ([cuGraphLaunch]), Memory Management ([cuMemcpyAsync]), Stream Management
      ([cuStreamSynchronize]), Event Management ([cuEventSynchronize]), Context
      Management ([cuCtxSynchronize]) and Data types used by CUDA driver
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
