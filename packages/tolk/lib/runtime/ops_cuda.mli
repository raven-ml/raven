(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CUDA's command queues, as batches encode them.

    A CUDA device runs a batch's work on two streams of its context, one for
    kernels and one for copies. The batch's host program enqueues the work
    itself, calling the functions of the CUDA driver (C functions of the library
    ["cuda"], {!Hcq2.ccall}), with the context and the streams it loads from a
    placeholder of the device tagged ["cuda"]: its four words are the context,
    the compute stream, the copy stream, and the status the last call returned,
    which the host program stores there.

    A launch's arguments are laid out after their size, in a buffer of arguments
    ({!Op.Linear} ["kernargs"]), and the launch passes the driver their address
    and the address of their size in the words of the queue's command buffer. A
    kernel's function, and the host function that stamps a slot, are read from
    words of the device tagged [("function", lib, name)] and ["stamp"], in
    memory the host reads, which the engine fills when it links the batch. *)

val queues : host:string -> reaches:(string -> bool) -> Hcq2.queues
(** [queues ~host ~reaches] is the command queues of a CUDA device, whose
    batches are submitted by host programs of [host], and which addresses the
    memory of the devices [reaches] holds for. It has a compute queue and a copy
    queue, each a stream, and each submission first makes the context current on
    the calling thread. Its commands are:
    - [exec call prg], which launches [prg]'s function ([cuLaunchKernel]) on the
      addresses of its buffers on the device and its variables, with its launch
      sizes;
    - [copy dst src n], which copies [n] bytes ([cuMemcpyAsync]);
    - [wait signal v], which makes the stream wait until the 64-bit word
      [signal] holds at least [v] ([cuStreamWaitValue64_v2]);
    - [signal word v], which writes [v] into [word] once the stream reached it
      ([cuStreamWriteValue64_v2]);
    - [timestamp slot], which stores the host clock into the slot's second word
      once the stream reached it ([cuLaunchHostFunc]);
    - [memory_barrier], which encodes nothing: a stream sees the memory its
      earlier work wrote;
    - [submit], whose host program makes the calls in order and stores the
      status of the last into the ["cuda"] placeholder. *)
