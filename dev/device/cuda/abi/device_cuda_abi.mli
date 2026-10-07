(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a CUDA device.

    Code compiled for a CUDA device runs on the host: it enqueues the device's
    work on a stream by calling the CUDA library the device was opened with
    ([libcuda]): [cuLaunchKernel] for a kernel, [cuMemcpyAsync] for a copy,
    [cuLaunchHostFunc] for a timestamp. The device's driver gives the addresses
    it needs as a {!t} when it opens the device, under {!key}.

    {b Fills.} Work for one of the device's queues can be a C function, a
    {e fill}: [int fill(void *queue, void *arg, uint64_t v)]. It receives the
    queue's stream ([CUstream]) as [queue], with the device's context current on
    the calling thread, and the value [v] the work completes on the device's
    timeline. It enqueues its work on that stream, and nothing else: waiting for
    earlier work and signalling [v] are the driver's. It stops at the first call
    that fails and returns its [CUresult], or returns [0] ([CUDA_SUCCESS]) once
    every call succeeded. A call can fail for earlier work on the device, which
    CUDA reports at a later call; the driver loses the device on any failure,
    returned here or met by the work after the fill returned.

    {b References.}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}:
      Execution Control ([cuLaunchKernel], [cuLaunchHostFunc]), Memory
      Management ([cuMemcpyAsync]) and Data types used by CUDA driver
      ([CUstream], [CUresult]). *)

type t = {
  symbol : string -> nativeint option;
      (** [symbol name] is the address of the function [name] the CUDA library
          exports, if it exports one. The name selects the version of the
          function's interface: ["cuLaunchKernel"], ["cuStreamWaitValue64_v2"].
          The address stays valid while the device is open. Any domain may call
          it. *)
}
(** The type for what compiled code needs from a CUDA device. *)

val key : t Type.Id.t
(** [key] is the key a CUDA device's {!t} is found under. *)
