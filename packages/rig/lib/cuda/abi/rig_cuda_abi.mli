(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a CUDA device.

    Compiled code drives a CUDA device from the host, through the CUDA
    library the device was opened with ([libcuda]). It runs a kernel with
    [cuLaunchKernel] and a copy with [cuMemcpyAsync]. It takes a timestamp
    with [cuLaunchHostFunc]. It runs a graph, a sequence of kernels recorded
    once and run by every submit of its submission, with [cuGraphLaunch].

    Compiled code reaches the device's driver in two ways. The record {!t}
    finds the CUDA library's functions and makes graphs in the device's
    context. A {e fill} is a C function of the compiled code that the
    driver's [submit] calls to enqueue the code's work.

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
    [cuMemcpyAsync] and [cuLaunchHostFunc] on [queue], and [cuGraphLaunch]
    on [queue] of a graph the device's record made ({!field-graph}), after
    updating its nodes, any number of times. Each call may block until the
    device's earlier work frees room in the stream. A fill must not wait for
    work ([cuStreamSynchronize], [cuEventSynchronize], [cuCtxSynchronize]) or
    change the current context. The driver orders the fill's work after
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

(** {1:graphs Graphs}

    A graph records a sequence of kernel launches, whose functions and order
    are fixed when the graph is made. To change the code a node runs, make
    another graph. A fill launches a graph with [cuGraphLaunch], and only a
    fill, during its call, updates or launches a graph. The driver calls a
    device's fills one at a time, so CUDA never sees two calls on one graph
    concurrently.

    A launch runs each node with its parameters at the launch, those it was
    made with or those of its last update. They are the work of the fill
    that launches the graph, and must name only what that fill's work may
    name: memory of its submission, or memory that lives as long as the
    graph. A node that names memory or a size of one run must therefore be
    updated by every fill that launches the graph, before the launch. A node
    that still holds an earlier run's parameters uses that run's memory,
    which may have been freed.

    Once collected, a graph is queued for destruction. The driver's next
    open, image load, image unload or graph construction on that GPU
    destroys the queued graphs before it unloads their images, even after the
    device has stopped or its context has failed, so reopening the GPU
    reclaims its previous devices' collected graphs. A cleanup already
    running on another domain is left to it. Collection and stopping alone
    call no CUDA destructor. A child of [fork] destroys no graph inherited
    from its parent. *)

type kernel = {
  func : int;
      (** Its [CUfunction]: the [code] of an entry of an image loaded on the
          device. *)
  grid : int * int * int;
      (** Its blocks per grid, in x, y and z, each at least [1]. *)
  block : int * int * int;
      (** Its threads per block, in x, y and z, each at least [1]. *)
  shared : int;  (** Its dynamic shared memory, in bytes, at least [0]. *)
  args : string;
      (** Its arguments, laid out as the kernel reads them: the buffer
          [cuLaunchKernel] takes as [CU_LAUNCH_PARAM_BUFFER_POINTER]. *)
}
(** The type for the kernel launches a graph records. Each size is at most
    [2]{^ [32]}[ - 1]. *)

type graph = private {
  handle : nativeint;  (** The [CUgraphExec]. *)
  nodes : nativeint array;
      (** The [CUgraphNode] of each kernel, in order. A fill may update a
          node's argument bytes, grid, block and shared memory size before it
          launches [handle], with [cuGraphExecKernelNodeSetParams_v2]
          ([CUDA_KERNEL_NODE_PARAMS_v2]). An update reaches the launches after
          it and none enqueued before it, so it waits for nothing. *)
}
(** The type for graphs. Its [handle], [nodes] and the CUDA modules of its
    kernels live while the value is reachable.

    {b Warning.} A submission whose work launches [handle] must keep the
    value ([Rig.Submission.make ~keep:graph]). Otherwise the graph may be
    destroyed, and its modules unloaded, while its work runs. A graph
    collected between two of the submission's submits is destroyed at the
    next operation {!section-graphs} names, and the next submit's fill
    launches the destroyed graph. *)

val graph :
  handle:nativeint -> nodes:nativeint array -> collect:(unit -> unit) -> graph
(** [graph ~handle ~nodes ~collect] is the value of a native graph that a
    driver made. The driver owns [handle], [nodes] and their kernels' modules
    until this call returns. From then on [collect], which runs once the value
    is collected, arranges their destruction. A driver must pass each native
    graph at most once.

    [collect] must keep no reference to the value, and must queue the
    destruction without blocking or calling CUDA. The driver destroys the
    graph, then its modules, at the operations {!section-graphs} names. *)

(** {1:record The record} *)

type t = {
  symbol : string -> nativeint option;
      (** [symbol name] is [Some a] if the CUDA library exports a function
          named [name], with [a] its address, and [None] otherwise. [name] is
          the exported name, whose suffix selects a version of the function's
          interface: ["cuStreamWaitValue64_v2"] is version 2 of
          ["cuStreamWaitValue64"]. [a] is valid while the device is open. Any
          domain may call [symbol]. *)
  graph : kernel array -> (graph, string) result;
      (** [graph ks] is [Ok g] with [g] a graph of the device's context that
          records one launch per element of [ks], in order, each run after the
          one before it has completed. [ks] may be empty. Any domain may call
          [graph] at any time, and it lets other domains run while CUDA
          instantiates the graph. The driver's stop waits for a call in
          flight.

          The result is [Error] with CUDA's reason if CUDA cannot make or
          instantiate the graph, such as for too much shared memory. It is
          [Error] with the context's error once a fault has failed the
          context, and [Error] once the driver has begun to stop the device.

          Raises [Invalid_argument] if a kernel's function belongs to no image
          loaded on the device, or a size is outside the range that
          {!kernel} gives.

          A call from a signal handler, finaliser or [Gc.Memprof] callback
          may raise [Rig_cuda.Refused], having changed nothing, when the
          callback interrupts a [graph] call, or [Rig_cuda]'s [count],
          [map_host], [free], [entry], [stop] or an opener's [make], of any
          CUDA device on the same thread. *)
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
      Execution Control ([cuLaunchKernel], [cuLaunchHostFunc]), Graph
      Management ([cuGraphLaunch], [cuGraphExecKernelNodeSetParams]), Memory
      Management ([cuMemcpyAsync]), Stream Management ([cuStreamSynchronize]),
      Event Management ([cuEventSynchronize]), Context Management
      ([cuCtxSynchronize]) and Data types used by CUDA driver ([CUstream],
      [CUresult]). *)
