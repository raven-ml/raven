(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Running compiled schedules on devices.

    The compiler ({!Tolk}) turns a schedule into data: programs, copies and
    batches of command queues ({!Tolk.Hcq2.compile_linear}). This library runs
    that data on the devices of [nx.device]. It describes each device to the
    compiler ({!device}), links a compiled schedule once: it allocates the
    storage the schedule names and writes every word known before it runs
    ({!link}). It then runs the linked schedule as often as it is asked to, on
    the buffers bound to its parameters ({!run}).

    A schedule names its devices by strings ({!Tolk.Ops.device}); the caller
    maps each name to a device, and no registry does. A program the host runs,
    the only program outside batches, is loaded on the host of its device's
    machine ({!Nx_device.host_of}).

    Everything a run needs is allocated when the schedule is linked, so a run
    allocates no device memory, and runs of one linked schedule are serialized:
    they share its storage. *)

open Tolk

(** {1:devices Devices} *)

val target : Nx_device.t -> Helpers.Target.t
(** [target d] is what [d]'s programs are compiled for, a function of [d] alone.
    A Metal, CUDA, AMD or NV device ([Nx_metal_device.of_device] and the others)
    has its vendor's device kind and its {!Nx_device.arch}, and names no
    renderer: its programs are rendered by the first of its kind's renderers
    that suits it ({!Tolk.Device.renderer}). A host, of this machine or another,
    and a device that shares its host's memory
    ({!Nx_device.shares_host_memory}), such as a test device over the host's
    memory ({!Nx_device.Driver.host_memory}) that maps the host's, run host
    programs: device ["CPU"], renderer ["CLANG"], and architecture its host's
    {!Nx_device.arch} with the processor [native] on this machine and [generic]
    on another.

    Raises [Invalid_argument] if [d] runs no program: the disk, or a device no
    vendor claims that is no host and does not share its host's memory. *)

val renderer : Nx_device.t -> Renderer.t
(** [renderer d] is the renderer of [d]'s {!target}, which writes the source of
    its programs, made once per target ({!Tolk.Device.renderer}).

    Raises [Invalid_argument] as {!target} does, and [Failure] with the reason
    when the target's renderer is not available, such as a compiler the machine
    lacks. *)

type device = {
  device : Nx_device.t;  (** The device. *)
  compiler : Hcq2.device;
      (** The device as the compiler sees it ({!Tolk.Hcq2.compile_linear}'s
          [devices]): its target, and the command queues its vendor encodes, if
          it runs work from queues. *)
  placeholder : Ops.t -> Nx_device.Buffer.t option;
      (** [placeholder u] is the storage of the placeholder [u] of a batch on
          this device if the vendor's commands name it, such as the objects of
          the vendor library's low-level section or a word holding the address
          of a C function they call, and [None] for the others. It holds at
          least the placeholder's bytes, or {!link} raises. It is asked for the
          placeholders on the device and for those of the batches the device
          runs, wherever they are, such as a C function's word on the host. *)
  submitting : unit -> unit;
      (** [submitting ()] runs inside each submission of a batch on the device,
          once the batch waited for its previous run and before its host
          program, when the device's objects cannot change: the time to write
          the words that change between runs, such as the buffers Metal's work
          declares resident. It raises [Invalid_argument], before anything is
          submitted, if the batch cannot run now: an AMD batch encoded for
          another profile request than the one being taken
          ({!Nx_device.Profile.counters}, {!Nx_device.Profile.traced}). *)
}
(** The type for devices as the engine runs work on them. *)

val device : (string * Nx_device.t) list -> string -> device
(** [device devices name] is the device [devices] maps [name] to, with its
    {!target}, or, for the disk, which runs no program, the target of the device
    ["DISK"]. A Metal, CUDA, AMD or NV device has the command queues of its
    vendor ({!Tolk.Ops_metal}, {!Tolk.Ops_cuda}, {!Tolk.Ops_amd},
    {!Tolk.Ops_nv}), submitted by host programs of the host of its machine
    ({!Nx_device.host_of}), under the name [devices] gives it, or under its own
    ({!Nx_device.name}) when [devices] gives it none, and gives the storage its
    commands name: Metal's objects, selectors, indirect command buffers and
    stamps and the address of [objc_msgSend]; CUDA's context, streams,
    functions, stamping function and the driver's entry points; AMD's rings and
    their words, its programs' code objects, which it loads, growing the
    device's scratch memory for their kernels, and its scratch memory; NV's
    channel rings and their words, its programs' cubins, which it loads, and a
    word of the bytes per thread of the device's local memory, which it grows
    for the kernels of each batch it links. A CUDA, AMD or NV device's queues
    reach the memory of the devices of [devices] that the device reaches
    ({!Nx_device.reaches}), and copy other memory through the host's. A caller
    that runs work from queues of its own, such as a test of the compiler, makes
    a {!device} of its own.

    Raises [Invalid_argument] if [devices] does not map [name], gives one name
    to two devices, or gives the name of an unnamed host of a Metal, CUDA, AMD
    or NV device to another device, [Failure] with nx.device's reason when an
    AMD or NV device refuses a program's code object at link, and
    {!Nx_device.Out_of_memory} when an NV device cannot grow its local memory
    for a batch it links. *)

(** {1:programs Host programs} *)

(** Programs the host runs. *)
module Program : sig
  type t
  (** The type for loaded host programs. *)

  val load : Nx_device.t -> Ops.t -> t
  (** [load d prg] is the compiled program [prg] ({!Tolk.Op.Program}) of the
      device [d], loaded on [d]'s host. The host loads a binary once: later
      loads of it share the loaded code.

      Raises [Invalid_argument] if [d]'s {!target} runs no host programs or if
      [prg] is not a compiled program, and [Failure] with the host's reason if
      the host refuses its binary. *)

  val run : ?vars:(string * int) list -> t -> Nx_device.Buffer.t list -> unit
  (** [run ~vars p buffers] runs [p] once on [buffers], its buffer parameters in
      order ({!Tolk.Ops.program_info.globals}), and returns once it returned.
      Each variable of [p] is the value [vars] binds to its name, or its bound
      value ([vars] defaults to [[]]). The call is outside the devices'
      ordering, as {!Nx_device.Program.call} is: synchronize the devices whose
      work touches [buffers] first.

      Raises [Invalid_argument] if [buffers] are not as many as [p]'s buffer
      parameters, if one holds fewer bytes than its parameter, or if a variable
      is unbound, and as {!Nx_device.Program.call} does. *)

  val blocks : ?vars:(string * int) list -> t -> int
  (** [blocks ~vars p] is the number of blocks a {!run} of [p] with [vars]
      splits its work into, which the host's cores run at once
      ({!Nx_device.Program.workers}). It is [1] for a program whose iterations
      depend on each other, or that does too little work to repay waking the
      threads.

      Raises [Invalid_argument] if a variable is unbound. *)
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
    ({!Tolk.Hcq2.compile_linear}, compiled for [fun n -> (devices n).compiler])
    linked on [devices], which maps each device name of [linear] to its device.
    - Each storage node ({!Tolk.Op.Buffer}) is bound to its buffers in [bound]
      (default [[]]), one per device of its placement, and allocated otherwise
      ({!Nx_device.Buffer.create}).
    - Each placeholder of a batch is the storage that its device's
      [placeholder], or that of a device of the batch, gives it, if any.
      Otherwise the signal word placeholder of a device is that device's
      {!Nx_device.signal_word}; the staging placeholder of a host (tagged
      ["staging"]) is that host's {!Nx_device.staging}, which every linked
      schedule and nx.device's copies share. Any other, which the host writes,
      is allocated on its device: a volatile placeholder, which the device
      writes too, and a command buffer (tagged ["cmdbuf_…"]), which the device
      fetches, in pinned memory ({!Nx_device.Buffer.create}[ ~memory:Pinned]),
      which both see coherently; the others, such as kernel arguments, in mapped
      memory ([~memory:Mapped]), which the device reads as its own, and which is
      pinned memory on a device that has none.
    - Each program is loaded once for each device and binary, and the words
      known at link, the addresses of linked storage among them, are written
      into the placeholders.
    - A device of a batch reaches the storage its words address as
      {!Nx_device.Buffer.reach} gives it, once per memory, for reading and
      writing when the batch writes that storage
      ({!Tolk.Ops.hcq_info}[.writes]); host memory the device does not map is
      then staged. A placeholder's storage is borrowed, never staged.

    The linked schedule keeps its storage, its programs and the buffers of
    [bound] while it is reachable. The runs of schedules that stage copies
    through one host take turns with its staging memory, whatever their devices,
    and with nx.device's copies through it: a run waits for the staged work of
    earlier runs on other devices before it submits its own, and a copy waits
    for the runs' work before it fills a slot.

    Raises [Invalid_argument] if [devices] does not map a device of [linear], if
    [bound] gives a storage node buffers of other devices, sizes or number than
    its placement, if a vendor gives a placeholder storage of fewer bytes than
    the placeholder, if a batch names a C function of a library the engine does
    not know, if a batch that profiles was encoded for another profile request
    than the one being taken, or if [linear] is not a compiled schedule;
    {!Nx_device.Out_of_memory} if a device cannot allocate its storage; and
    [Failure] if a device refuses a program's binary. *)

val run :
  ?vars:(string * int) list -> t -> Nx_device.Buffer.t list array -> unit
(** [run ~vars s slots] runs [s] with [slots.(i)] bound to its parameter [i] (an
    untagged {!Tolk.Op.Param}), one buffer per device of the parameter's
    placement, and each variable of the schedule bound to its value in [vars]
    (default [[]]). It runs the calls of [s] in order:
    - a host program is {!Program.run} on each device of its first argument;
    - a copy is {!Nx_device.Buffer.copy};
    - a range around calls runs them once for each combination of the ranges'
      values, the last range varying fastest, with each range's variable
      ({!Tolk.Hcq2.range_value}) bound to its value: host programs and copies,
      and batches around a back edge, since a batch holds the ranges of its
      calls as loops;
    - a back edge of calls ({!Tolk.Ops.backedge}) runs them once per trip of its
      range, with the range's variable bound to the trip, while its flag holds:
      before each trip it reads the flag's boolean with
      {!Nx_device.Buffer.copy}, which first synchronizes the flag's device, and
      stops once the flag is false or the trips are done;
    - a batch is one {!Nx_device.submit} over its devices that touches every
      buffer it reaches. It first waits for the work of its previous run on each
      of its devices ({!Nx_device.Submission.wait}), since the runs share its
      memory, and on the host for the work of the devices whose memory it
      reaches but that are not the batch's, such as the source of a copy from a
      device without queues ({!Nx_device.Submission.waits}): no queue of the
      batch waits for them. It then writes its inputs' addresses into its
      address table and calls its host program with each device's last submitted
      value and the value its work signals ({!Nx_device.Submission.value}). Its
      devices reach each run's inputs as {!Nx_device.Buffer.reach} gives them,
      once per memory, for reading and writing when the batch writes any input
      over it. While a profile is taken, it records each kernel's span on each
      of its devices ({!Nx_device.Submission.record}).

    [run] returns once every call is queued: a read of a result waits for the
    work that wrote it, as {!Nx_device.synchronize} does. A batch that writes a
    staged buffer is the exception: its submission returns once its work is done
    and the buffer copied back ({!Nx_device.submit}). Runs of [s] are
    serialized: a run starts once the previous one returned.

    When the setting {!Tolk.Setting.debug} is [1] or more and [s] runs ten calls
    or more, [run] first prints ["jit execs n calls"] on standard output, [n]
    its number of calls. When it is [2] or more ({!profile}), [run] prints a
    line for each kernel on standard output: its device, how many kernels ran
    before it, its name, its number of arguments, and its time and throughput. A
    host program and a copy are timed on the host clock. A batch synchronizes
    its devices once submitted, and its kernels' times are their spans as the
    devices stamp them, if its schedule stamps them, unless a profile is being
    taken elsewhere, whose spans they are: their lines then have no time.

    Raises [Invalid_argument] if [slots] does not bind each parameter of [s] to
    buffers of its placement's devices, each holding its parameter's bytes, if a
    variable is unbound, or if a device refuses to run a batch now (its
    [submitting]), and {!Nx_device.Lost} as the devices do. *)

val profile : unit -> Tolk.Hcq2.profile
(** [profile ()] is the profile that a schedule compiled now needs for {!run}'s
    reports ({!Tolk.Hcq2.compile_linear}'s [profile]): {!Tolk.Hcq2.Stamped}
    while {!Tolk.Setting.debug} is [2] or more, when [run] prints each kernel's
    time from its stamps, and {!Tolk.Hcq2.Unstamped} otherwise. A caller that
    keeps compiled schedules keeps one for each value of [profile ()]. *)

val slots : t -> Nx_device.Buffer.t list array
(** [slots s] is, for each slot that a parameter of [s] takes, a buffer on each
    device of the parameter's placement holding the parameter's bytes, and no
    buffer for a slot no parameter takes: slots {!run} and {!time} accept. Their
    contents are unspecified.

    Raises {!Nx_device.Out_of_memory} if a device cannot allocate a buffer. *)

(** {1:timing Timing}

    A search times many programs of one kernel, each a few times: it links each
    call ({!link_call}), allocates slots once, for the first ({!slots}), and
    times the runs of each on them ({!time}). *)

val link_call : devices:(string -> device) -> Ops.t -> t
(** [link_call ~devices call] is the call [call] of a compiled program
    ({!Tolk.Op.Program}) linked as a schedule of one call that records its
    kernel's span while a profile is taken
    ({!Tolk.Hcq2.compile_linear}[ ~profile:Stamped]). [call]'s arguments supply
    its buffers and positional scalars; {!time}'s [vars] bind free variables.

    Raises [Invalid_argument] if [call] is not a call of a compiled program, and
    as {!link} does. *)

val clock : t -> Tolk.Search.clock
(** [clock s] is the clock {!time} times [s] on: {!Tolk.Search.Device} if every
    kernel of [s] runs in a batch of a device with queues, which stamps it, and
    {!Tolk.Search.Host} otherwise. *)

val time :
  ?vars:(string * int) list -> t -> Nx_device.Buffer.t list array -> float
(** [time ~vars s slots] is the time in seconds of one run of [s] on [slots]
    with [vars] ({!run}), from cold caches. It invalidates the caches of the
    devices that run [s]'s kernels where their vendor can
    ([Nx_nv_device.invalidate_caches]), then runs [s] once, reporting nothing on
    standard output whatever {!Tolk.Setting.debug} holds. The time is the sum of
    the spans of [s]'s kernels in a profile of the run ({!Nx_device.Profile}),
    which a device with queues stamps and the host records around each call of a
    host program. The profile is the time's own, whatever profiles are taken
    around it, which record the run's events too. A time allocates no device
    memory and loads nothing.

    Raises as {!run} does, {!Nx_device.Lost} if a device that runs a kernel of
    [s] is lost during the run, and [Invalid_argument] if a kernel of [s]
    records no span in the profile: one of a batch compiled without a profile,
    or one under a range of no trips. *)
