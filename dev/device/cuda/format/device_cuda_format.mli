(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a CUDA device.

    Code compiled for a CUDA device runs on the host: it enqueues the device's
    work on a stream by calling the CUDA driver API, [cuLaunchKernel] for a
    kernel, [cuMemcpyAsync] for a copy, [cuLaunchHostFunc] for a timestamp. It
    calls the driver the device was opened with, so it needs the addresses of
    the driver's functions and of a host function that reads the clock. The
    driver gives them as a {!t} when it opens the device, under {!key}.

    {b Fills.} A piece of work on one of the device's queues is a C function
    [int fill(void *queue, void *arg, uint64_t v)]. It receives the queue's
    stream ([CUstream]) as [queue], with the device's context current on the
    calling thread, and the value [v] the work completes on the device's
    timeline. It enqueues its work on that stream, and nothing else: waiting for
    earlier work and signalling [v] are the driver's. It returns [0]
    ([CUDA_SUCCESS]), or the [CUresult] of the first call that failed, and then
    enqueues nothing after it.

    {b References.}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}: Stream
      Management ([CUstream]), Execution Control ([cuLaunchKernel],
      [cuLaunchHostFunc], [CUhostFn]) and Driver Entry Point Access (versioned
      function names). *)

type t = {
  symbol : string -> nativeint option;
      (** [symbol name] is the address of the driver's exported function [name],
          if the driver has one. The name selects the version of the function's
          interface, as the driver exports it: ["cuLaunchKernel"],
          ["cuStreamWaitValue64_v2"]. The address stays valid while the device
          is open. Any domain may call it. *)
  timestamp : nativeint;
      (** [timestamp] is the address of a host function ([CUhostFn]) that stores
          the host clock into the 64-bit word its argument points at:
          nanoseconds of the system's monotonic clock, as an unsigned integer in
          the host's byte order. A stream that runs it with [cuLaunchHostFunc]
          stamps the word once the stream's earlier work completed. It takes no
          lock and calls no CUDA function: a host function that did either could
          wait for a thread that waits for it. *)
}
(** The type for what compiled code needs from a CUDA device. *)

val key : t Type.Id.t
(** [key] is the key a CUDA device's {!t} is found under. *)
