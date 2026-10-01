(** The NULL device: test devices whose command queues run on the host.

    The devices ["CPU:1"], ["CPU:2"] and ["CPU:3"] are {!Run.devices}' test
    devices of the host's memory, whose work signals by storing its value into
    its signal word. They compile for the host's processor, and their queues are
    encoded with {!Null_queue}'s commands, with two changes that make them
    runnable: a copy carries its size in place of its event, and a queue is
    submitted by a call of the host program to the device's [submit] function,
    which hands the command buffer's address and size to the device.

    The device runs each submitted queue on a domain of its own, as hardware
    runs its queues behind the host: each command in turn, a wait holding its
    queue until its signal is reached, while the other queues go on. An [exec]
    calls the program's host entry on the addresses and values of its arguments;
    a [timestamp] writes {!Nx_device.Profile.now}. *)

val devices :
  ?copy_queue:bool ->
  ?reaches:(string -> bool) ->
  ?ring:int ->
  unit ->
  string ->
  Tolk_engine.device
(** [devices ~copy_queue ~reaches ~ring ()] maps ["CPU"] to the host
    ({!Tolk_engine.device}) and each NULL device's name to it, with copy queues
    iff [copy_queue] (default [true]). Its compiler's view is that of
    {!Tolk_engine.target}, with queues whose host is ["CPU"] and which address
    the memory of the devices [reaches] holds for (default all): a copy between
    memory they do not address stages through the host's
    ({!Tolk.Hcq2.queues.reaches}). With [ring], a queue holds at most [ring]
    bytes of commands in one submission, as a queue that runs them from its ring
    does, and refuses more ({!Tolk.Hcq2.Over_capacity}).

    A C function a batch calls from a NULL device's memory
    ({!Tolk.Hcq2.ccall}[ ~host]) is any the process's libraries define.

    Raises [Invalid_argument] for any other name. *)

val device : string -> Nx_device.t
(** [device name] is the NULL device [name].

    Raises [Invalid_argument] if [name] is none. *)

val with_latency : float -> (unit -> 'a) -> 'a
(** [with_latency s f] is [f ()], during which a NULL device starts running a
    queue [s] seconds after it is submitted, as a busy device would. *)

val synchronize : unit -> unit
(** [synchronize ()] waits until every queue submitted to a NULL device has run,
    and each NULL device {!Nx_device.synchronize}d.

    Raises [Failure] with the device's reason if a queue failed, such as on a
    command of an unknown code. *)
