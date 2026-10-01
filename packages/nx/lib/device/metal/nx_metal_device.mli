(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Metal devices.

    Opens the Mac's GPU as an {!Nx_device.t} named ["METAL"]. Its buffers are
    memory that the processor and the GPU share, coherently, so copies between
    them and host buffers are memory copies, its pinned and mapped memory
    ({!Nx_device.Buffer.memory}) are its own, and it borrows host
    memory when the GPU shares the host's memory. Its programs are functions of
    metallib binaries. Its budget defaults to the working set size Metal
    recommends for the GPU.

    Its timestamps are readings of the host clock
    ({!Nx_device.Driver.Host_clock}).

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
    {!Nx_device.submit}. Work for the value [v] is a command buffer of {!queue}
    whose compute encoders wait for and update {!fence}, which orders them after
    the device's earlier work, and which ends by encoding a signal of [v] on
    {!event}. Metal signals through its event alone
    ({!Nx_device.Driver.Signal}): other devices' work cannot wait for it on
    their queues.

    Metal times its work by command buffer, on the host clock
    ({!Nx_device.Profile.now}). To profile a command buffer, a submitter writes
    it, retained, into the second word of the stamps it gives
    {!Nx_device.Submission.record}, and [0] into the fourth. Once the command
    buffer completed, the device writes its GPU start and end times over them
    and releases it. Stamps recorded again before they were read still hold the
    command buffer of the earlier record: the submitter releases it before it
    writes its new one. *)

type t
(** The type for the Metal objects of a device. *)

val of_device : Nx_device.t -> t option
(** [of_device d] is the Metal objects of [d], if [d] is a Metal device. *)

val queue : t -> nativeint
(** [queue m] is the [MTLCommandQueue] all work is submitted to. *)

val event : t -> nativeint
(** [event m] is the [MTLSharedEvent] that work signals its values on. *)

val fence : t -> nativeint
(** [fence m] is the [MTLFence] that orders compute encoders on the queue. *)

val residency_set : t -> nativeint option
(** [residency_set m] is the [MTLResidencySet] of the device's buffers, added to
    the queue, where Metal has residency sets. *)

val resources : t -> nativeint array
(** [resources m] is the [MTLBuffer]s of the device's memory, which work must
    declare resident with [useResources:count:usage:] when the device has no
    residency set. It is empty when it has one. Read it inside
    {!Nx_device.submit}, where no allocation changes it. *)

val msg_send : nativeint
(** [msg_send] is the address of [objc_msgSend], which a compiled host program
    calls to send the Objective-C messages that encode and commit its work:
    with the type of the method it sends, receiver and selector first. *)

val selector : string -> nativeint
(** [selector name] is the selector [name], such as ["commandBuffer"] or
    ["encodeSignalEvent:value:"], registered with the Objective-C runtime: the
    word a compiled host program passes {!msg_send} for that message. *)

(** {2:icb Indirect command buffers} *)

type command = {
  program : Nx_device.Program.t;  (** Its pipeline: a program of the device. *)
  offset : int;
      (** The byte offset of its arguments in the buffer of arguments, bound as
          its kernel buffer 0. *)
  global : int * int * int;  (** Its threadgroups per grid. *)
  local : int * int * int;  (** Its threads per threadgroup. *)
}
(** The type for the dispatches of an indirect command buffer. *)

val indirect_commands :
  t ->
  Nx_device.Buffer.t ->
  command list ->
  (nativeint * nativeint list, string) result
(** [indirect_commands m args cmds] is an [MTLIndirectCommandBuffer] of [cmds],
    each a concurrent dispatch of its program on its arguments in [args], a
    buffer of [m]'s device, that runs after the dispatches before it; and the
    [MTLIndirectComputeCommand] of each dispatch, in order. They are released
    once [args] is unreachable, and the programs of [cmds] stay loaded until
    then. Work that runs the indirect command buffer lists [args] in its
    {!Nx_device.submit}'s [touches].

    [Error msg] if a command's threads per threadgroup exceed its pipeline's
    maximum, or if Metal cannot make the indirect command buffer.

    Raises [Invalid_argument] if [args] is not a buffer of [m]'s device or if a
    program of [cmds] is not loaded on it. *)
