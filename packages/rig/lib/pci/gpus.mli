(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A vendor's GPUs, and the process's hold on them.

    A machine's GPUs of one vendor are its PCI functions that the vendor's
    driver recognizes, numbered in bus order. GPU [i] is the same function
    whether the process reaches it through its kernel driver or over PCI, and
    whichever driver holds it.

    This library knows three states of a GPU besides its kernel driver's:
    - A GPU is {e detached} once {!detach} has unbound its kernel driver and
      the driver has let go of it. A process can then take its function.
    - A GPU is {e held} by this process while a driver drives it ({!type-hold}).
      No other open of the GPU succeeds, and no change to the machine touches
      it.
    - A GPU is {e lost} once a hold has ended without a clean stop, or a reset
      of it has failed. Its next open renews it ({!renew}).

    A process makes one {!t} per vendor, and the holds and lost GPUs above are
    that value's. The opens, resets and changes of one vendor's GPUs run one at
    a time, the driver's [f] and {!detach}'s wait included, so each delays the
    vendor's other opens and changes. {!stop} waits for none of them. Every
    [Error] about GPU [i] starts with its name ({!val-name}). Each value that
    takes a GPU number [i] raises [Invalid_argument] if [i < 0]. *)

type t
(** The type for a vendor's GPUs. *)

val make :
  name:string ->
  memory_bar:int ->
  nodes:(root:string -> string -> string list) ->
  unreleased:(root:string -> string -> string option) ->
  teardown_ms:int ->
  reset:(Function.t -> (unit, string) result) ->
  (Machine.id -> bool) ->
  t
(** [make ~name ~memory_bar ~nodes ~unreleased ~teardown_ms ~reset is_gpu] is
    the GPUs of a vendor, the functions [f] with [is_gpu f], named after [name]
    ({!val-name}). [nodes] and [unreleased] read the machine's files under
    [root], which is ["/"] for {!Machine.this}.
    - [memory_bar] is the BAR through which the process reaches the GPUs'
      memory, which {!detach} enlarges.
    - [nodes ~root bus] is the character devices through which the kernel
      driver serves the GPU at [bus] without a [dev] file under the GPU's
      directory in [/sys/bus/pci], as paths from [root], such as
      [["dev/nvidia0"]].
    - [unreleased ~root bus] is [Some why] if the kernel driver, no longer
      bound to the GPU at [bus], has not let go of it, and [None] if it has or
      the vendor cannot tell. It is asked only of an unbound GPU. It must raise
      nothing, and returns [None] for a file it cannot read.
    - [teardown_ms] is the longest the kernel driver takes to drop the files of
      the GPU's devices that it holds for a process after the process has let
      go of them, as for a compute runtime after its process exits.
    - [reset fn] must stop whatever runs on the GPU of the taken function [fn],
      whatever ran on it before, its kernel driver included, and reset the GPU
      as its vendor does. It returns [Ok ()] once the GPU answers again with
      nothing running on it, or [Error why]. The function's bus mastering is
      off when [reset] is called. *)

val name : t -> int -> string
(** [name g i] is the name of GPU [i]. It is [name] of {!make} for [0], such as
    ["AMD-PCI"], and that name followed by [":i"] otherwise, such as
    ["AMD-PCI:1"]. *)

val buses : t -> Machine.t -> string list
(** [buses g m] is the bus addresses of [g]'s GPUs on [m], in bus order. GPU [i]
    is the [i]th. *)

(** {1:holds Opening}

    An open is a bracket. It holds the GPU and calls the driver's [f], which
    starts the GPU. Before the GPU can be left running, [f] must give the hold
    the vendor's stop ({!set_stop}). If [f] returns [Ok _], the process keeps
    the hold until {!stop}. If [f] returns [Error _] or raises before
    {!set_stop}, the open releases the GPU as it found it. If it does so after
    {!set_stop}, the open stops the GPU through the hold, and the GPU is lost.
    An exception raised by [f] passes through. A driver reports what a request
    refused as an [Error], since the library's requests return them and its
    accesses raise nothing ({{!Rig_pci.errors}errors}).

    At its exit the process stops each GPU it still holds as {!stop} does,
    before its functions' bus mastering is turned off and their files closed. A
    child of [fork] stops none of its parent's GPUs, which are still the
    parent's. An exception raised by a vendor's stop at exit is printed on
    standard error, and the other GPUs are stopped. *)

type hold
(** The type for the process's hold on one GPU. *)

val open_ :
  t ->
  Machine.t ->
  int ->
  (hold -> Function.t -> ('a, string) result) ->
  ('a, string) result
(** [open_ g m i f] is [f h fn], with the process holding GPU [i] of [m] by [h]
    and its function [fn] taken ({!Function.take}). [f]'s [Error] messages get
    the GPU's name as a prefix too. A lost GPU, or one a dead process left
    reaching memory, is renewed before [f] runs ({!renew}).

    [Error why] without calling [f] if [m] has no GPU [i], saying how many it
    has, if the process holds it already, if it is unbound and its kernel
    driver has not let go of it yet ([unreleased] of {!make}), or if its
    function cannot be taken, with {!Function.take}'s reason. [Error why] with
    the vendor's reason if its renewal fails, and the GPU is then lost. An
    exception raised by the vendor's reset passes through, and the GPU is lost.

    Raises [Invalid_argument], with the GPU released, if [f] returns [Ok _]
    without {!set_stop}. *)

val set_stop : hold -> (unit -> [ `Clean | `Lost | `Unknown ]) -> unit
(** [set_stop h stop] makes [stop] the vendor's stop of the GPU [h] holds, which
    {!stop} calls. The vendor must call [set_stop] inside the open's [f],
    before the GPU can be left running. That is before [f]'s first write to
    the GPU, unless [f] itself stops the GPU when it fails after writing.
    [stop ()] must end the GPU's work and return [`Clean] if the GPU may be
    released as it is, [`Lost] if its work stopped but the GPU must be
    renewed, and [`Unknown] if its work may still run.

    Raises [Invalid_argument] if [h] has a stop already or was stopped. *)

val stop : hold -> [ `Stopped | `Unknown ]
(** [stop h] calls the vendor's stop ({!set_stop}), then releases the GPU's
    function. The GPU may then be opened again, and its next open renews it if
    the vendor's stop returned [`Lost] or [`Unknown]. [stop h] is [`Unknown]
    iff the vendor's stop returned [`Unknown], and [`Stopped] otherwise. Later
    calls return what the first returned and call nothing, and a call while
    another runs waits for it and returns the same. An exception raised by the
    vendor's stop passes through, and the GPU is lost. A device calls [stop]
    once its work is done or lost. *)

val renew : hold -> (unit, string) result
(** [renew h] resets the GPU [h] holds as its vendor does ([reset] of {!make}),
    with its bus mastering off, and gives back the memory dead processes left
    for it. The memory this process allocated for it stays. A driver calls
    [renew] from its [f], before it keeps any of the GPU's state, when it
    finds the GPU running firmware it cannot continue from.

    [Error why] with the reset's reason. The open then loses the GPU, whatever
    its [f] returns. An exception raised by the reset passes through, and the
    GPU is lost the same way. *)

val hang_ms : int
(** [hang_ms] is [30_000], 30 seconds. Work that a GPU driven without a kernel
    driver has not advanced for that long is hung, since no kernel driver
    bounds it. *)

(** {1:changes Changes to the machine}

    {!detach}, {!attach} and {!reset} change the machine, and their changes
    persist after the process. Each runs under the lock a physical take holds,
    so no process takes the GPU meanwhile. {!detach} and {!attach} write the
    machine's [/sys/bus/pci], so they act only on a machine the process reaches
    without a transport, such as {!Machine.this}. They need [CAP_SYS_ADMIN] and
    write access to the files they write, which root has:
    - {!detach} writes the function's [driver_override], its driver's
      [unbind], the [remove] file of each other function of its device, and
      the function's [enable], [config] (its command register) and
      [resourceN_resize] for its memory BAR;
    - {!attach} writes [vfio-pci]'s [unbind], the function's [enable] and
      [driver_override], [/sys/bus/pci/rescan] and
      [/sys/bus/pci/drivers_probe].

    Each of them returns [Error why], changing nothing, if [m] has no GPU [i] or
    a process holds it. {!detach} and {!attach} also return [Error why] on a
    machine reached through a transport, and {!attach} when the process may
    not write a file, naming the file. {!attach} and {!reset} also return
    [Error why] if the GPU's function cannot be taken, or if the GPU is
    unbound and its kernel driver has not let go of it yet.

    A kernel driver lets go of a GPU, writing to it as it does, once the last
    file of the GPU's character devices has gone. Its unbind either waits for
    that or, as a DRM driver's does, returns first and lets go later. A file
    remains while a process holds it open or mapped, and while the kernel holds
    it for a process, as for a compute runtime for a while after its process
    exits. The GPU's character devices are those with a [dev] file under its
    directory in [/sys/bus/pci], such as its DRM nodes, and those that [nodes]
    names ({!make}).

    So {!detach} unbinds a driver only once no file of the GPU remains, and
    the driver lets go inside the unbind. It refuses the GPU immediately
    while a process holds one of its files, waits up to [teardown_ms]
    ({!make}) for those the kernel holds, and after the unbind asks
    [unreleased] ({!make}) whether the driver has let go.

    The library finds the files that processes hold in [/proc/PID/fd] and
    [/proc/PID/map_files], for the processes of its PID namespace. It finds
    the files of a DRM device in debugfs's [/sys/kernel/debug/dri/BUS/clients],
    which lists a file until its last reference goes. The library misses a
    process that opens a device after its last look, and memory of the GPU
    exported as a dma-buf to another device or process, which Linux lists under
    no device. The driver then lets go when that holder does.

    A process must not take an unbound GPU before its driver has let go, since
    the driver's last writes would race the take's. {!open_}, {!reset} and
    {!attach} refuse a GPU whose [unreleased] ({!make}) says so. *)

(** The type for how far {!detach} took a GPU. *)
type progress =
  | Detached  (** A process can take the GPU's function. *)
  | Unreleased of string
      (** The GPU is unbound, but its kernel driver has not let go of it, for
          the vendor's reason ([unreleased] of {!make}), and its memory BAR is
          as it was. The driver lets go, writing to the GPU, once the last
          holder of the GPU's files lets go of them. {!detach} called again
          waits for that up to [teardown_ms] and finishes, or returns
          [Error] if the driver still holds the GPU. *)
  | Part_way of string
      (** {!detach} failed after it had started to change the machine, for
          the reason, which names the file if the process may not write one.
          The GPU may be neither detached nor as it was. {!attach} gives it
          back, and {!detach} called again tries again. *)

val detach : t -> Machine.t -> int -> (progress, string) result
(** [detach g m i] detaches GPU [i] of [m] from its kernel driver, so that a
    process can take its function. A GPU bound to [vfio-pci] stays bound.
    Otherwise [detach] sets the function's [driver_override] to no driver,
    which keeps every kernel driver off the GPU through probes, rescans and
    module loads until {!attach} or a reboot, and unbinds the driver. It
    removes the other functions of the GPU's device, such as its audio
    function. Once unbound, the function is left:
    - enabled;
    - with its bus mastering off, which a kernel driver's unbind may leave on;
    - with its memory BAR at the largest size the BAR and its bridge take.

    The kernel driver's users, a display among them, lose the GPU until
    {!attach} or a reboot.

    It returns [Error] only before its first write to the machine, and after
    that write [Ok p], where [p] is how far it got ({!type-progress}).

    [Error why], changing nothing, if:
    - no process could take its function once detached, such as when an
      IOMMU translates its addresses and it is not bound to [vfio-pci];
    - a process, this one included, holds a file of one of its devices,
      naming it, or the kernel still holds one after [teardown_ms];
    - [/proc] cannot be read, or, for a GPU with a DRM device, its list of
      files in debugfs;
    - the process may not write [driver_override];
    - the GPU is already unbound and its kernel driver has still not let go
      of it after [teardown_ms].

    [Ok Detached] leaves the memory BAR small on [vfio-pci], or where the
    kernel refuses every larger size. *)

val attach : t -> Machine.t -> int -> (unit, string) result
(** [attach g m i] gives GPU [i] of [m] back to its kernel driver. A GPU bound
    to no kernel driver, or to [vfio-pci], is first reset as {!reset} does,
    since its kernel driver expects it as its vendor's reset leaves it,
    whatever ran on it. [attach] then unbinds [vfio-pci], disables the
    function, has Linux rescan the bus, which brings back the functions
    {!detach} removed, clears the function's [driver_override] and has Linux
    probe drivers, which binds the GPU's driver. A GPU bound to its kernel
    driver stays bound. If the function's PCI driver override
    ([driver_override]) is set, whether a {!detach} that stopped part-way
    left it or an administrator set it, [attach] has Linux rescan the bus
    and clears the override.

    [Error why] if no driver takes the GPU, such as when the driver's module is
    not loaded. [Error why], with the GPU left as it was, if its reset fails,
    and [why] says how to proceed. An exception raised by the vendor's reset
    passes through, with the GPU left as it was. *)

val reset : t -> Machine.t -> int -> (unit, string) result
(** [reset g m i] takes the function of GPU [i] of [m], turns its bus mastering
    off, resets the GPU as its vendor does ({!make}) and releases the function,
    whatever the vendor's reset returns or raises. After a reset that returns
    [Ok ()], the memory dead processes left for the GPU goes
    ({{!Function.dma}system memory}), and the next open does not renew the GPU.
    A reset that fails leaves the GPU lost. An exception raised by the vendor's
    reset passes through, as in an open.

    [Error why] with the vendor's reset's reason. *)
