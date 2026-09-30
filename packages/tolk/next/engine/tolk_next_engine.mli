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
    kind and its {!Nx_device.arch}. A host, of this machine or another, and a
    device that shares its host's memory ({!Nx_device.shares_host_memory}), such
    as a test device over the host's memory ({!Nx_device.Driver.host_memory})
    that maps the host's, run host programs: device ["CPU"], renderer ["CLANG"],
    and architecture its host's {!Nx_device.arch} with the processor [native] on
    this machine and [generic] on another.

    Raises [Invalid_argument] if [d] runs no program: the disk, or a device no
    vendor claims that is no host and does not share its host's memory. *)

type device = {
  device : Nx_device.t;  (** The device. *)
  compiler : Hcq2.device;
      (** The device as the compiler sees it ({!Tolk_next.Hcq2.compile_linear}'s
          [devices]): its target, and the command queues its vendor encodes, if
          it runs work from queues. *)
  placeholder : Ops.t -> Nx_device.Buffer.t option;
      (** [placeholder u] is the storage of the placeholder [u] of a batch on
          this device if the vendor's commands name it, such as the objects of
          the vendor library's low-level section or a word holding the address
          of a C function they call, and [None] for the others. It is asked for
          the placeholders on the device and for those of the batches the device
          runs, wherever they are, such as a C function's word on the host. *)
  submitting : unit -> unit;
      (** [submitting ()] runs inside each submission of a batch on the device,
          once the batch waited for its previous run and before its host
          program, when the device's objects cannot change: the time to write
          the words that change between runs, such as the buffers Metal's work
          declares resident. *)
}
(** The type for devices as the engine runs work on them. *)

val device : (string * Nx_device.t) list -> string -> device
(** [device devices name] is the device [devices] maps [name] to, with its
    {!target}, or, for the disk, which runs no program, the target of the device
    ["DISK"]. A Metal, CUDA, AMD or NV device has the command queues its
    vendor's encoder writes, submitted by host programs of the host of its
    machine, which [devices] must name; no such encoder exists yet, so every
    device runs its calls one by one. A caller that runs work from queues of its
    own, such as a test of the compiler, makes a {!device} of its own.

    Raises [Invalid_argument] if [devices] does not map [name]. *)

(** {1:programs Host programs} *)

(** Programs the host runs. *)
module Program : sig
  type t
  (** The type for loaded host programs. *)

  val load : Nx_device.t -> Ops.t -> t
  (** [load d prg] is the compiled program [prg] ({!Tolk_next.Op.Program}) of
      the device [d], loaded on [d]'s host. The host loads a binary once: later
      loads of it share the loaded code.

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
  devices:(string -> device) ->
  ?bound:(Ops.t * Nx_device.Buffer.t list) list ->
  Ops.t ->
  t
(** [link ~devices ~bound linear] is the compiled schedule [linear]
    ({!Tolk_next.Hcq2.compile_linear}, compiled for
    [fun n -> (devices n).compiler]) linked on [devices], which maps each device
    name of [linear] to its device.
    - Each storage node ({!Tolk_next.Op.Buffer}) is bound to its buffers in
      [bound] (default [[]]), one per device of its placement, and allocated
      otherwise ({!Nx_device.Buffer.create}).
    - Each placeholder of a batch is the storage that its device's
      [placeholder], or that of a device of the batch, gives it, if any.
      Otherwise the signal word placeholder of a device is that device's
      {!Nx_device.signal_word}, and any other is allocated in pinned memory of
      its device ({!Nx_device.Buffer.create}[ ~pinned:true]), which the host and
      the device see coherently, since the batch's host program writes it.
    - Each program is loaded once for each device and binary, and the words
      known at link, the addresses of linked storage among them, are written
      into the placeholders.

    The linked schedule keeps its storage, its programs and the buffers of
    [bound] while it is reachable.

    Raises [Invalid_argument] if [devices] does not map a device of [linear], if
    [bound] gives a storage node buffers of other devices, sizes or number than
    its placement, if a batch names a C function of a library the engine does
    not know, or if [linear] is not a compiled schedule;
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
    - a range around calls runs them once for each combination of the ranges'
      values, the last range varying fastest, with each range's variable
      ({!Tolk_next.Hcq2.range_value}) bound to its value;
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
  devices:(string -> device) ->
  string ->
  Ops.t ->
  float
(** [measure ~cold ~vars ~devices name prg] is the time in seconds of one run of
    the compiled program [prg] on the device [devices] maps [name] to, on
    scratch buffers of its parameters' sizes. Each run is timed by a profile of
    it ({!Nx_device.Profile}): the span of its kernel, which a device with
    queues stamps and the host records around its call otherwise. While a
    profile is taken already, it is the run and the device's synchronization on
    the host clock. The time is the mean of as many runs as take at least 10
    microseconds in all, up to 1000 runs, so that a kernel shorter than a tick
    of its clock is not measured as taking no time. With [cold] (default
    [false]), the device's caches are invalidated before each run where its
    vendor can ([Nx_nv_device.invalidate_caches]). It is the measurement the
    compiler's search of kernel optimisations times its candidates with.

    Raises as {!link} and {!run} do. *)
