(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from a CUDA device.

    Code compiled for a CUDA device runs on the host. It enqueues the device's
    work on a stream by calling the CUDA library the device was opened with
    ([libcuda]): [cuLaunchKernel] for a kernel, [cuMemcpyAsync] for a copy,
    [cuLaunchHostFunc] for a timestamp, [cuGraphLaunch] for a step recorded
    once, when it is linked, and run on each submission of it. Two things
    connect it to the device's driver, the library that opened the device: a
    record, {!t}, that finds the CUDA library's functions and makes graphs in
    the device's context, and a C calling convention, the fill, by which the
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
    [cuLaunchKernel], [cuMemcpyAsync] and [cuLaunchHostFunc] on [queue], and
    [cuGraphLaunch] on [queue] of a graph the device's record made
    ({!field-graph}), after updating its nodes, any number of times. Each call
    may block until the device's earlier work frees room in the stream; nothing
    bounds what a fill enqueues. A fill declares no room: its ring units and
    segment bytes are [0], and the driver refuses work that declares any other.
    The fill does not wait for work ([cuStreamSynchronize],
    [cuEventSynchronize], [cuCtxSynchronize]), does not enqueue on another
    stream, and does not change the current context. Waiting for earlier work
    and signalling [v] are the driver's.

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
      ([cuGraphLaunch], [cuGraphExecKernelNodeSetParams]), Memory Management
      ([cuMemcpyAsync]), Stream Management ([cuStreamSynchronize]), Event
      Management ([cuEventSynchronize]), Context Management ([cuCtxSynchronize])
      and Data types used by CUDA driver ([CUstream], [CUresult]). *)

(** {1:graphs Graphs} *)

type kernel = {
  func : int;
      (** Its [CUfunction]: the address an entry of an image the device loaded
          gives. *)
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

type graph = {
  handle : nativeint;  (** The [CUgraphExec]. *)
  nodes : nativeint array;
      (** The [CUgraphNode] of each kernel, in order. A fill may update one
          before it launches [handle], with [cuGraphExecKernelNodeSetParams_v2]
          ([CUDA_KERNEL_NODE_PARAMS_v2]). An update reaches the launches after
          it and none enqueued before it, so it waits for nothing. CUDA refuses
          an update that moves a kernel to another context; the fill then fails
          with CUDA's error. *)
  release : unit -> unit;
      (** [release ()] destroys [handle] and the graph [nodes] belong to. The
          owner of the linked step that made them calls it once, after the last
          work that launched [handle] completed. A graph keeps no image loaded:
          its owner keeps the images of its kernels loaded until [release]
          returns, unless the driver stopped the device first. Raises
          [Invalid_argument] if called twice. Any domain may call it, also after
          the driver stopped the device or a fault failed its context, when it
          ignores CUDA's answer. *)
}
(** The type for graphs.

    A launch of [handle] runs each node with its parameters at the launch: those
    it was made with, or those of its last update. They are the work of the fill
    that launches it, and name only what that fill's work may name: memory of
    its submission, or memory that lives as long as the graph. A node that names
    memory or a size of one run is therefore updated by every fill that launches
    the graph, before the launch; a node left as an earlier run set it uses that
    run's memory, which may have been freed.

    Only a fill, during its call, updates or launches a graph. The driver calls
    a device's fills one at a time, so CUDA never sees two calls on one graph at
    once. *)

(** {1:record The record} *)

type t = {
  symbol : string -> nativeint option;
      (** [symbol name] is [Some a] if the CUDA library exports a function named
          [name], with [a] its address, and [None] otherwise. [name] is the
          exported name, whose suffix selects a version of the function's
          interface: ["cuStreamWaitValue64_v2"] is version 2 of
          ["cuStreamWaitValue64"]. [a] is valid while the device is open. Any
          domain may call [symbol]. *)
  graph : kernel array -> (graph, string) result;
      (** [graph ks] is [Ok g] with [g] a graph of the device's context that
          records one launch per element of [ks], in order, each run after the
          one before it completed. [ks] may be empty.

          The result is [Error msg] with CUDA's reason if CUDA cannot make or
          instantiate the graph, such as for too much shared memory, with the
          context's error once a fault failed it, and once the driver began to
          stop the device; a stop waits for a [graph] call in flight.

          Raises [Invalid_argument] if a kernel is of no image the device
          loaded, or a size is out of its range: a grid or block size below [1],
          [shared] below [0], any above [2]{^ [32]}[ - 1]. Any domain may call
          it at any time; it lets other domains run while CUDA instantiates the
          graph. *)
}
(** The type for what compiled code needs from a CUDA device. *)

val key : t Type.Id.t
(** [key] is the key of a CUDA device's {!t}. The device's driver declares its
    record under [key]; a device of another kind has none. *)
