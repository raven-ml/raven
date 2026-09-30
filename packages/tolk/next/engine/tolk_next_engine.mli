(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Running compiled schedules on devices.

    The compiler ({!Tolk_next}) turns a schedule into data: programs, copies and
    batches of command queues ({!Tolk_next.Hcq2.compile_linear}). This library
    runs that data on the devices of [nx.device]. It describes each device to
    the compiler ({!device}), links a compiled schedule once: it allocates the
    storage the schedule names and writes every word known before it runs
    ({!link}). It then runs the linked schedule as often as it is asked to, on
    the buffers bound to its parameters ({!run}).

    A schedule names its devices by strings ({!Tolk_next.Ops.device}); the
    caller maps each name to a device, and no registry does. A program the host
    runs, the only program outside batches, is loaded on the host of its
    device's machine ({!Nx_device.host_of}).

    Everything a run needs is allocated when the schedule is linked, so a run
    allocates no device memory, and runs of one linked schedule are serialized:
    they share its storage. *)

open Tolk_next

(** {1:devices Devices} *)

val target : Nx_device.t -> Helpers.Target.t
(** [target d] is what [d]'s programs are compiled for. A Metal, CUDA, AMD or NV
    device ([Nx_metal_device.of_device] and the others) has its vendor's device
    kind and its {!Nx_device.arch}. Any other device whose memory its host
    addresses, such as the host itself or a test device over the host's memory
    ({!Nx_device.Driver.host_memory}), runs host programs: device ["CPU"],
    renderer ["CLANG"], and architecture its host's {!Nx_device.arch} with the
    processor [native] on this machine and [generic] on another.

    Raises [Invalid_argument] if [d] runs no program: the disk, or a device
    whose memory its host does not address and no vendor claims. *)

val device : (string * Nx_device.t) list -> string -> Hcq2.device
(** [device devices name] is the device [devices] maps [name] to, as the
    compiler sees it ({!Tolk_next.Hcq2.compile_linear}'s [devices]): its
    {!target}, and the command queues of a Metal, CUDA, AMD or NV device, whose
    host programs the host of its machine runs and whose queues reach the memory
    of the named devices it maps ({!Nx_device.Buffer.borrow}).

    Raises [Invalid_argument] if [devices] does not map [name]. *)

(** {1:programs Host programs} *)

(** Programs the host runs. *)
module Program : sig
  type t
  (** The type for loaded host programs. *)

  val load : Nx_device.t -> Ops.t -> t
  (** [load d prg] is the compiled program [prg] ({!Tolk_next.Op.Program}) of
      the device [d], loaded on [d]'s host. A program is loaded once for each
      host and binary: a later load of it returns the same program.

      Raises [Invalid_argument] if [d]'s {!target} runs no host programs or if
      [prg] is not a compiled program, and [Failure] with the host's reason if
      the host refuses its binary. *)

  val run : ?vars:(string * int) list -> t -> Nx_device.Buffer.t list -> unit
  (** [run ~vars p buffers] runs [p] once on [buffers], its buffer parameters in
      order ({!Tolk_next.Ops.program_info.globals}), and returns once it
      returned. Each variable of [p] is the value [vars] binds to its name, or
      its bound value ([vars] defaults to [[]]). The call is outside the
      devices' ordering, as {!Nx_device.Program.call} is: synchronize the
      devices whose work touches [buffers] first.

      Raises [Invalid_argument] if [buffers] are not as many as [p]'s buffer
      parameters, if one holds fewer bytes than its parameter, or if a variable
      is unbound, and as {!Nx_device.Program.call} does. *)
end

(** {1:schedules Linked schedules} *)

type t
(** The type for linked schedules. *)

val link :
  devices:(string * Nx_device.t) list ->
  ?bound:(Ops.t * Nx_device.Buffer.t list) list ->
  Ops.t ->
  t
(** [link ~devices ~bound linear] is the compiled schedule [linear]
    ({!Tolk_next.Hcq2.compile_linear}) linked on [devices], which maps each
    device name of [linear] to its device.
    - Each storage node ({!Tolk_next.Op.Buffer}) is bound to its buffers in
      [bound] (default [[]]), one per device of its placement, and allocated
      otherwise ({!Nx_device.Buffer.create}).
    - Each placeholder of a batch is allocated on its device: a volatile one in
      pinned memory ({!Nx_device.Buffer.create}[ ~pinned:true]), which the host
      and the device see coherently. The signal word placeholder of a device is
      that device's {!Nx_device.signal_word}, a C function's the function's
      address, and the placeholders a vendor's commands name are the vendor's
      objects, from its library's low-level section.
    - Each program is loaded once for each device and binary, and the words
      known at link, the addresses of linked storage among them, are written
      into the placeholders.

    The linked schedule keeps its storage, its programs and the buffers of
    [bound] while it is reachable.

    Raises [Invalid_argument] if [devices] does not map a device of [linear], if
    [bound] gives a storage node buffers of other devices, sizes or number than
    its placement, or if [linear] is not a compiled schedule;
    {!Nx_device.Out_of_memory} if a device cannot allocate its storage; and
    [Failure] if a device refuses a program's binary. *)

val run :
  ?vars:(string * int) list -> t -> Nx_device.Buffer.t list array -> unit
(** [run ~vars s slots] runs [s] with [slots.(i)] bound to its parameter [i] (an
    untagged {!Tolk_next.Op.Param}), one buffer per device of the parameter's
    placement, and each variable of the schedule bound to its value in [vars]
    (default [[]]). It runs the calls of [s] in order:
    - a host program is {!Program.run} on each device of its first argument;
    - a copy is {!Nx_device.Buffer.copy};
    - a batch is one {!Nx_device.submit} over its devices that touches every
      buffer it reaches. It first waits for the work of its previous run on each
      of its devices ({!Nx_device.Submission.wait}), since the runs share its
      memory, and on the host for the work of devices outside the batch
      ({!Nx_device.Submission.waits}). It then writes its inputs' addresses into
      its address table and calls its host program with each device's last
      submitted value and the value its work signals
      ({!Nx_device.Submission.value}). While a profile is taken, it records each
      kernel's span on each of its devices ({!Nx_device.Submission.record}).

    [run] returns once every call is queued: a read of a result waits for the
    work that wrote it, as {!Nx_device.synchronize} does. Runs of [s] are
    serialized: a run starts once the previous one returned.

    Raises [Invalid_argument] if [slots] does not bind each parameter of [s] to
    buffers of its placement's devices, each holding its parameter's bytes, or
    if a variable is unbound, and {!Nx_device.Lost} as the devices do. *)

(** {1:measuring Measuring} *)

val measure :
  ?cold:bool ->
  ?vars:(string * int) list ->
  devices:(string * Nx_device.t) list ->
  string ->
  Ops.t ->
  float
(** [measure ~cold ~vars ~devices name prg] is the time in seconds of one run of
    the compiled program [prg] on the device [devices] maps [name] to, on
    scratch buffers of its parameters' sizes, as the device times it: the span
    between its stamps on a device with queues, the call on the host otherwise.
    With [cold] (default [false]), the device's caches are invalidated first
    where its vendor can ([Nx_nv_device.invalidate_caches]). It is the
    measurement the compiler's search of kernel optimisations times its
    candidates with.

    Raises as {!link} and {!run} do. *)
