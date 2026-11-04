(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Running compiled schedules on rig's devices.

    The compiler ({!Tolk}) turns a schedule into calls of compiled programs and
    copies, with the ranges and loops around them
    ({!Tolk.Realize.compile_linear}). This library describes rig's devices to
    the compiler ({!target}), and turns a compiled schedule into a program of
    rig ({!Rig_program}), which it loads once ({!link}) and runs as often as it
    is asked to, on the buffers bound to its parameters ({!run}).

    A schedule names its devices by strings ({!Tolk.Ops.type-device}); the
    caller maps each name to a device, and no registry does. Every device of a
    schedule is of one machine ({!Rig.host_of}). *)

open Tolk

(** {1:devices Devices} *)

val target : Rig.t -> Helpers.Target.t
(** [target d] is what [d]'s programs are compiled for, a function of [d] alone.
    A Metal, CUDA, AMD or NV device, as its driver declares it
    ({!Rig.capability}), has its vendor's device kind and its {!Rig.arch}, and
    names no renderer: its programs are rendered by its kind's renderer
    ({!Tolk.Device.renderer}). A host, of this machine
    or another, and a device whose work is this process's code
    ({!Rig.runs_on_host}), such as a {!Rig.memory_device}, run host programs:
    device ["CPU"], renderer ["CLANG"], and architecture its host's {!Rig.arch}
    with the processor [native] on this machine and [generic] on another. On
    this machine under Windows the architecture names the system, as in
    ["x86_64-windows,native"], under which an x86_64 program is compiled for
    Windows' calling convention ({!Tolk.Compiler_cpu.clang}). Another
    machine's architecture names no system.

    Raises [Invalid_argument] if [d] runs no program: an io device, or another
    device that no vendor claims. *)

val renderer : Rig.t -> Renderer.t
(** [renderer d] is the renderer of [d]'s {!target}, which writes the source of
    its programs, made once per target ({!Tolk.Device.renderer}).

    Raises [Invalid_argument] as {!target} does, and [Failure] with the reason
    when the target's renderer is not available, such as a compiler the machine
    lacks. *)

(** {1:schedules Linked schedules} *)

type t
(** The type for linked schedules. *)

(** The type for where a schedule's storage that no buffer is bound to lives. *)
type storage =
  | Linked
      (** Made at link, in its device's memory, which the runs share and the
          schedule holds while it is reachable. *)
  | Per_run
      (** Made by each run in its device's memory before its first call, and
          returned after its last ({!Rig.Buffer.free}), its memory coming back
          once the work on it is done, so that the schedule holds none of it
          between runs. A run holds its storage whole: a schedule lowered by
          {!Tolk.Jit.jit_lower} plans one arena per device
          ({!Tolk.Memory.memory_plan_rewrite}), so a run then holds what its
          plan places at once. *)

val link :
  devices:(string -> Rig.t) ->
  ?bound:(Ops.t * Rig.Buffer.t) list ->
  ?storage:storage ->
  Ops.t ->
  t
(** [link ~devices ~bound ~storage linear] is the compiled schedule [linear]
    ({!Tolk.Realize.compile_linear}, compiled for the {!target} of each device)
    loaded on [devices], which maps each device name of [linear] to its device.
    - Each storage node ({!Tolk.Op.Buffer}) is bound to its buffer in [bound]
      (default [[]]), of its device, and is made otherwise as [storage]
      (default {!Linked}) says, starting at its phase ({!run}). The schedule
      holds the memory of every bound buffer a kernel or copy accesses
      ({!Rig.Claim.hold}) while it is reachable, so no call consumes it. A link
      that raises holds none of them.
    - A kernel is launched on the device of its first argument. A host program
      is launched on its device's host, which splits its loop over its cores
      where the program splits one ({!Tolk.Ops.kernel_info.split}): into a
      block for each 2{^18} operations the program's estimates count
      ({!Tolk.Ops.type-estimates}), or for each 2{^18} bytes of memory they
      count up to 8 blocks, whichever gives more, at least one block and at
      most one per iteration.
    - A copy is a copy part of a submission of its source's device, or else of
      its destination's, where that device copies and reaches both sides and
      they are both GPU memory or both memory the host computes on
      ({!Rig.runs_on_host}); otherwise it is a move of the program
      ({!Rig_program.step}), one {!Rig.Buffer.submit_copy}.
    - Consecutive kernels and copies of one device are one submission
      ({!Rig.Submission}), and each of them is a submission of its own while
      {!Tolk.Setting.debug} is [2] or more, so that a submission's work in a
      profile ({!Rig.Profile.Work}) is one kernel's.
    - A range around calls is a loop of the program, and so is a back edge,
      which reads its flag before each trip.

    The variables a run binds, and the values computed from them, are words of
    the run's ints. A value under a range may move with its trips, by a constant
    factor per range: an offset into a buffer, the offset a copy reads or
    writes, or a scalar argument. Everything else, such as a grid or a
    range's trips, is fixed for the run.

    Raises [Invalid_argument] if [devices] does not map a device of [linear], if
    its devices are of several machines, if a kernel's device is of another
    kind than its program's target, if [bound] binds a node that is not storage,
    or a buffer of another device or of fewer bytes than the node's, a buffer
    a kernel or copy accesses that is dead or whose memory is held exclusive,
    or a buffer a kernel accesses that does not start at the node's phase ({!run}),
    if a computing device starts its memory on fewer bytes ({!Rig.alignment})
    than the alignment of a storage node made there at link, if kernels
    access a parameter at two alignments or phases, if a value under a range
    moves with its trips in another way than above, if a back edge's flag
    moves with the run's words, or if [linear] is not a compiled schedule;
    {!Rig.Out_of_memory} if a device cannot allocate its storage at link; and
    [Failure] with the reason if the program does not load, such as a device
    that refuses a program's binary. *)

val run : ?vars:(string * int) list -> t -> Rig.Buffer.t option array -> unit
(** [run ~vars s slots] runs [s] with the buffer of [slots.(i)] bound to its
    parameter [i] (an untagged {!Tolk.Op.Param}), on the parameter's device,
    and each variable of the schedule bound to its value in [vars]
    (default [[]]), or else to the value it is bound to. It runs the calls of
    [s] in order, with each range's trips, and each back edge's while its flag
    holds ({!Rig_program.run}).

    A kernel's vector accesses are aligned to their width only where each buffer
    it accesses starts at its parameter's phase: its first byte lies the
    parameter's [phase] bytes past a multiple of its [align]
    ({!Tolk.Ops.type-param_arg}), as {!Rig.Buffer.aligned} tells. [run] checks
    this of each buffer a kernel of [s] accesses, before any work runs. A copy
    moves bytes wherever they start, so a buffer that only copies access may
    start anywhere.

    [run] returns once every call is submitted: a read of a result waits for the
    work that wrote it ({!Rig.Buffer.read}). [s] keeps none of the buffers of
    [slots] once [run] returns. Runs of [s] are serialized: a run starts once
    the previous one returned.

    When the setting {!Tolk.Setting.debug} is [1] or more and [s] runs ten calls
    or more, [run] first prints ["jit execs n calls"] on standard output, [n]
    its number of calls. When it is [2] or more, [run] prints a line for each
    kernel on standard output: its device, how many kernels ran before it, its
    name, its number of arguments, and its time and throughput, where its device
    records its submission's work ({!Rig.Profile.Work}).

    Raises [Invalid_argument] if [slots] does not bind each parameter of [s] to
    a buffer of its device holding its parameter's bytes, if a
    buffer a kernel accesses does not start at its parameter's phase, if a
    variable is unbound, and as {!Rig_program.run} does; {!Rig.Out_of_memory}
    if a device cannot allocate the storage of a schedule linked {!Per_run}:
    no call has run, and the storage the run made is returned. *)

(** {1:timing Timing}

    A search times many programs of one kernel, each a few times: it links each
    call ({!link_call}), allocates slots once, for the first ({!slots}), and
    times the runs of each on them ({!time}), each from a clean, cold cache
    ({!scrub}). *)

val link_call : devices:(string -> Rig.t) -> Ops.t -> t
(** [link_call ~devices call] is the call [call] of a compiled program
    ({!Tolk.Op.Program}) linked as a schedule of one call, whose kernel is a
    submission of its own. [call]'s arguments supply its buffers and positional
    scalars; {!time}'s [vars] bind free variables.

    Raises [Invalid_argument] if [call] is not a call of a compiled program, and
    as {!link} does. *)

val slots : t -> Rig.Buffer.t option array
(** [slots s] is, for each slot that a parameter of [s] takes, a buffer of the
    parameter's device holding the parameter's bytes, starting at its phase,
    and [None] for a slot no parameter takes: slots {!run} and
    {!time} accept. Their contents are unspecified.

    Raises {!Rig.Out_of_memory} if a device cannot allocate a buffer. *)

type scrub
(** The type for what empties the caches of a schedule's devices before a
    time. *)

val scrub : t -> scrub
(** [scrub s] is, for each device that runs a kernel of [s] and whose
    last-level cache's bytes its driver states ({!Rig.cache_bytes}), a buffer
    of twice those bytes in the device's memory, written once, and a kernel
    compiled for the device that reads it whole. It serves {!time} of every
    schedule whose kernels run on those devices. A device whose driver states
    no cache bytes, such as a Metal device or an AMD GPU over PCI, has none.

    Raises {!Rig.Out_of_memory} if a device cannot allocate its buffer, and
    [Failure] with the reason if a kernel does not compile or load. *)

val time :
  ?vars:(string * int) list ->
  ?scrub:scrub ->
  t ->
  Rig.Buffer.t option array ->
  float
(** [time ~vars s slots] is the time in seconds of one run of [s] on [slots]
    with [vars] ({!run}), reporting nothing on standard output whatever
    {!Tolk.Setting.debug} holds. The time is the sum, over the run's
    submissions that run a kernel of [s], of their work in a profile of the
    run ({!Rig.Profile.Work}). A submission of several kernels counts once,
    and one of copies alone counts nothing. The run's submissions are found
    by the points its steps report ({!Rig_program.type-frame}), so work of
    other domains and of profiles taken around it adds nothing. A time loads
    nothing, and allocates no device memory but the storage of a schedule
    linked {!Per_run}.

    With [scrub], the run starts from a clean, cold cache on each device of
    [scrub] that runs a kernel of [s]: before the run, and outside its time,
    [time] reads that device's buffer and waits for the read, which leaves
    none of the run's buffers in the device's last-level cache and none of
    their lines to write back. A read of twice the cache's bytes evicts every
    line wherever the run's buffers lie. On the host this holds for the cache
    the system states, the largest of its first core's: a cache it does not
    state, such as Apple silicon's system cache, and the last-level caches of
    other core complexes, which the read shares among them, may keep lines.
    Without [scrub], or on a device it has none for, the run finds what the
    caches hold. A read of twice a cache of 64 MiB takes about 0.25 ms on a
    GPU of 576 GB/s.

    Raises as {!run} does, and {!Rig.Lost} if a device that runs a kernel of [s]
    is lost during the run, or if another domain's device whose work the
    run's profile recorded is lost before the profile reads it
    ({!Rig.Profile.take}). *)

(**/**)

val blocks : extent:int -> ops:int -> bytes:int -> int * int
(** [blocks ~extent ~ops ~bytes] is the number of blocks a host launch of a
    split loop of [extent] iterations runs in, the loop estimated to do [ops]
    operations and move [bytes] bytes of memory, and the iterations of each
    block but the last ({!link}). For the suites. *)
