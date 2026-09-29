(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** CUDA devices.

    Opens NVIDIA GPUs as {!Nx_device.t}s named ["CUDA"], ["CUDA:1"], ["CUDA:2"],
    ... in the driver's order, each on its GPU's primary context.

    {b Memory.} Buffers are GPU memory, which the host does not address.
    {!Nx_device.Buffer.copy} moves their bytes on the device's copy stream:
    directly from and to host memory that the GPU addresses, and through the
    host's staging memory from and to other host memory. Host memory that the
    GPU addresses is page-locked:
    - {!Nx_device.Buffer.create}[ ~host:true] allocates it, and it counts in the
      device's budget, which defaults to the GPU's memory size;
    - {!Nx_device.Buffer.borrow} registers host memory, whole pages of it, for
      every CUDA device at once, and unregisters it when the last device's
      borrows of it are unreachable. It counts in no budget. The memory must be
      writable: memory mapped read-only cannot be borrowed;
    - the host's staging memory, 128 MiB, is registered at the first copy that
      needs it and kept for the life of the process. It counts in no budget.

    {b Requirements.} The GPU writes 64-bit values from its streams to signal
    its timeline, so a device opens only where the driver supports 64-bit stream
    memory operations and unified addressing, on every platform, Windows
    included. There is no fallback.

    {b Programs} are functions of CUDA modules: cubins, fatbins, or PTX, which
    the driver compiles when it loads it.

    {b Faults and hangs.} A fault on the GPU, such as an illegal address, fails
    the device with the driver's error when a wait finds it; so does a driver
    error while the runtime enqueues a copy, since the context may then be
    unusable. Work that does not signal within the device's
    {!Nx_device.timeout}, 30 seconds unless {!Nx_device.set_timeout} sets
    another, fails the device too: raise the timeout before submitting kernels
    that run longer.

    {b Other CUDA libraries.} Devices use the GPU's primary context, which the
    CUDA runtime API and the libraries over it share. The runtime makes it
    current only for the duration of its own driver calls, and leaves the
    calling thread's current context as it found it.

    {b The driver} is loaded at the first call of {!count} or {!get}, by its
    standard name, from the platform's library search path: [libcuda.so.1] (or
    [libcuda.so]) on Linux, [nvcuda.dll] on Windows, and [libcuda.dylib] on
    macOS. A library of that name earlier on the path is loaded in its place,
    for the whole process. Building and linking this library needs no CUDA
    installation. *)

val count : unit -> int
(** [count ()] is the number of CUDA devices: [0] if the driver cannot be loaded
    or initialized, or reports none. *)

val get : int -> (Nx_device.t, string) result
(** [get i] is CUDA device [i], opened by the first call that succeeds; every
    later call returns the same value. [Error msg] says why it cannot be opened,
    for example that the driver cannot be loaded, that [i >= count ()], or that
    the GPU cannot write 64-bit values from its streams. Opening requires the
    driver's 64-bit stream memory operations and unified addressing on every
    platform, Windows included; there is no fallback without them.

    Raises [Invalid_argument] if [i < 0]. *)

val v : int -> Nx_device.t
(** [v i] is like {!get} but raises [Invalid_argument] with [get]'s message when
    the device cannot be opened. *)

val of_address :
  Nx_device.t -> nativeint -> Nx_dtype.Scalar.t -> int -> Nx_device.Buffer.t
(** [of_address d a s n] is a borrowed buffer of [n] elements of format [s] at
    device address [a] of [d], without a copy: memory that another library
    allocated on [d]'s primary context. It is a view, at [a], of the whole
    allocation under [a], whose start is its {!Nx_device.Buffer.handle}. Nothing
    frees it: its owner keeps it allocated for as long as the buffer and its
    views are reachable. The host addresses it when the allocation is
    page-locked host memory.

    Raises [Invalid_argument] if [d] is not a CUDA device, if the driver knows
    no memory at [a] or it is memory of another context, or as
    {!Nx_device.Buffer.view} does if the [n] elements at [a] do not lie in the
    allocation or are not aligned. *)

(** {1:low Low-level}

    For the libraries that submit work to a CUDA device, inside
    {!Nx_device.submit}. They reach the driver by loading it by its standard
    name, which gives them the library this one loaded, and make [context]
    current on the submitting thread.

    Work for the timeline value [v] first waits on its stream for the signal
    word to reach [v - 1] ([cuStreamWaitValue64_v2] with
    [CU_STREAM_WAIT_VALUE_GEQ]), and ends by writing [v] into it
    ([cuStreamWriteValue64_v2]). The values thus complete in order across the
    two streams. The device's own copies follow the same rule. *)

type handles = {
  context : nativeint;  (** The GPU's primary [CUcontext]. *)
  compute : nativeint;  (** The non-blocking [CUstream] for kernels. *)
  copy : nativeint;
      (** The non-blocking [CUstream] for copies, which {!Nx_device.Buffer.copy}
          uses. *)
  signal : nativeint;
      (** The device address of the signal word, the first word of
          {!Nx_device.timeline}: page-locked host memory that holds the last
          value the device signaled. *)
}
(** The type for the driver objects of a device. *)

val handles : Nx_device.t -> handles
(** [handles d] is the driver objects of [d].

    Raises [Invalid_argument] if [d] is not a CUDA device. *)
