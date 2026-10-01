(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Eager computation on devices, one compiled program per operation.

    {!backend} computes each operation nx dispatches to it as a program that
    tolk compiles for the target of the device the operands lie on: the lowering
    of the operation ({!Lower}), over the operands as their views read their
    buffers, strided, broadcast or offset, stored into the destination. A
    program is compiled the first time the process meets its {e key}: the
    operation and its static arguments (axes, padding, fill value, ...); each
    operand's and the destination's dtype, shape and strides, where its first
    element lies within 16 bytes of the run of its buffer the program reads, and
    where that run starts within 16 bytes of memory, to which the program aligns
    its vector accesses; and the target of the destination's device. It is kept
    for the life of the process, and linked a few times on each device that runs
    it; each run takes the next link in turn, since runs of one link are
    serialized.

    A kernel queues its program on the device and returns: a read of the result
    waits for it, as for any work on the device. While a profile is taken
    ({!Nx_device.Profile}), each compilation is a span of the host named
    [compile] and the kernel's name, such as ["compile binary"].

    nx computes every placement's values on the host until it hands backends the
    arrays of their devices: through nx, a placement on a GPU with {!backend}
    computes on the host, with the host's programs, and the kernels
    ({!Nx_backend.kernels}) run on a GPU's arrays when they are given them.

    {b Numerics.} Each operation computes what nx documents, to the class of its
    row in the lowering's table: exactly, within a rounded sum, or within a
    budget of units in the last place. Linear algebra runs a number of steps
    that shapes fix and raises no {!Nx_backend.Linalg_error}: where nx.cpu
    raises one, the result holds the non-finite values the steps produce
    ({!Lower_linalg}).

    {b Refusals.} A kernel raises {!Nx_backend.Refused} before any work, naming
    the operation, for the Fourier transforms, [eig] and [eigh]; for [int4],
    [uint4], [complex64] and [complex128]; for arrays of a device it does not
    run on (see {!backend}); and for a dtype the device's target does not
    compute, such as [float64] on Metal, which [svd]'s singular values need. *)

val backend : Nx_backend.t
(** [backend] is the backend of compiled programs, named ["compiled"]. It runs
    on the host, on devices that share its memory, and on devices whose command
    queues tolk encodes: Metal and CUDA ({!Tolk_next_engine.device}). *)
