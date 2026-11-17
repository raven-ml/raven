(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a CUDA device.

    Compiled code drives a CUDA device from the host, through the CUDA
    library the device was opened with ([libcuda]). It runs a kernel with
    [cuLaunchKernel] and a copy with [cuMemcpyAsync]. It takes a timestamp
    with [cuLaunchHostFunc].

    Compiled code reaches the device's driver in two ways. The record {!t}
    finds the CUDA library's functions. A {e fill} is a C function of the
    compiled code that the driver's [submit] calls to enqueue the code's
    work.

    This library holds only that agreement. The driver makes the record when
    it opens the device and declares it ({!declare}), compiled code finds it
    under {!key}, and neither links the other. *)

(** {1:fills Fills}

    A fill is work for one of the device's queues given as a C function,
    [int fill(void *queue, void *arg, uint64_t v)]. The driver calls it with:
    - [queue], the queue's stream ([CUstream]), with the device's context
      current on the calling thread;
    - [arg], the argument the work was given;
    - [v], the value the work completes on the device's timeline.

    A fill must enqueue on [queue] only. It may call [cuLaunchKernel],
    [cuMemcpyAsync] and [cuLaunchHostFunc] on [queue], any number of times.
    Each call may block until the device's earlier work frees room in the
    stream. A fill must not wait for work ([cuStreamSynchronize],
    [cuEventSynchronize], [cuCtxSynchronize]) or change the current
    context. The driver orders the fill's work after
    earlier work and signals [v], so a fill does neither.

    A host function that a fill enqueues runs on a thread of the CUDA
    library. It must call no CUDA function, and must take no lock that a
    fill, or the code that calls the device's driver, may hold while a CUDA
    call blocks.

    A fill must stop at the first call that fails and return that call's
    [CUresult], and return [0] ([CUDA_SUCCESS]) once every call has
    succeeded. A call can fail because of earlier work on the device, which
    CUDA reports at a later call. The driver loses the device on any failure,
    whether the fill returned it or its work failed after the fill
    returned. *)

(** {1:record The record} *)

type t = {
  symbol : string -> nativeint option;
      (** [symbol name] is [Some a] if the CUDA library exports a function
          named [name], with [a] its address, and [None] otherwise. [name] is
          the exported name, whose suffix selects a version of the function's
          interface: ["cuStreamWaitValue64_v2"] is version 2 of
          ["cuStreamWaitValue64"]. [a] is valid while the device is open. Any
          domain may call [symbol]. *)
}
(** The type for what compiled code needs from a CUDA device. *)

val key : t Type.Id.t
(** [key] is the key of a CUDA device's record. Compiled code finds the record
    of a device [d] with [Rig.capability d key], which is [None] if [d] is
    not a CUDA device. *)

val declare : t -> Rig_edge.capability
(** [declare c] is [c] under {!key}, as the facts of a CUDA device carry it
    ({!Rig_edge.type-facts}). *)

(** {1:references References}
    - {{:https://docs.nvidia.com/cuda/cuda-driver-api/}CUDA Driver API}:
      Execution Control ([cuLaunchKernel], [cuLaunchHostFunc]), Memory
      Management ([cuMemcpyAsync]), Stream Management ([cuStreamSynchronize]),
      Event Management ([cuEventSynchronize]), Context Management
      ([cuCtxSynchronize]) and Data types used by CUDA driver ([CUstream],
      [CUresult]). *)
