(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Metal devices.

    Opens the Mac's GPU as an {!Nx_device.t} named ["METAL"]. Its buffers are
    memory that the processor and the GPU share, so copies between them and host
    buffers are memory copies, and it borrows host memory when the GPU shares
    the host's memory. Its programs are functions of metallib binaries. Its
    budget defaults to the working set size Metal recommends for the GPU.

    Its timestamps are readings of the host clock ({!Nx_device.Host_clock}).

    A fault on the GPU surfaces as a hang: the runtime reads no command buffer's
    status, so work that faults is found when its signal does not arrive in
    time.

    This library exists on macOS only. *)

val count : unit -> int
(** [count ()] is the number of Metal devices: [1] on a Mac whose GPU supports
    Metal, [0] otherwise. *)

val get : int -> (Nx_device.t, string) result
(** [get i] is Metal device [i], opened by the first call that succeeds; every
    later call returns the same value. [Error msg] says why it cannot be opened,
    for example that [i >= count ()], after the device's name, such as
    ["METAL:1: no such device; there is one Metal device"].

    Raises [Invalid_argument] if [i < 0]. *)

val v : int -> Nx_device.t
(** [v i] is like {!get} but raises [Failure] with [get]'s message when the
    device cannot be opened. *)

(** {1:low Low-level}

    For the libraries that submit work to a Metal device, inside
    {!Nx_device.submit}.

    Metal times its work by command buffer, on the host clock
    ({!Nx_device.Profile.now}). To profile a command buffer, a submitter writes
    it, retained, into the first word of the stamps it gives
    {!Nx_device.Profile.record}, and [0] into the second. Once the command
    buffer completed, the device writes its GPU start and end times over them
    and releases it. Stamps recorded again before they were read still hold the
    command buffer of the earlier record: the submitter releases it before it
    writes its new one. *)

type handles = {
  device : nativeint;  (** The [MTLDevice]. *)
  queue : nativeint;  (** The [MTLCommandQueue] all work is submitted to. *)
  event : nativeint;
      (** The [MTLSharedEvent] submitted work signals its timeline value on. *)
  fence : nativeint;
      (** The [MTLFence] that orders compute encoders on the queue. *)
  residency_set : nativeint option;
      (** The [MTLResidencySet] of the device's buffers, added to the queue,
          where Metal has residency sets. *)
}
(** The type for the Metal objects of a device. *)

val handles : Nx_device.t -> handles
(** [handles d] is the Metal objects of [d].

    Raises [Invalid_argument] if [d] is not a Metal device. *)

val resources : Nx_device.t -> nativeint array
(** [resources d] is the [MTLBuffer]s of [d]'s memory, which work must declare
    resident with [useResources:count:usage:] when [d] has no residency set. It
    is empty when [d] has one. Read it inside {!Nx_device.submit}, where no
    allocation changes it.

    Raises [Invalid_argument] if [d] is not a Metal device. *)
