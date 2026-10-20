(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** User-space drivers of a machine's PCI functions.

    A {e driver} drives, from the process, the PCI functions that its match
    accepts, such as a vendor's GPUs or network adapters ({!make}). A machine's
    functions of one driver are numbered in bus order. Function [i] of a driver
    is the same function whether the process reaches it through its kernel
    driver or over PCI, and whichever driver holds it. A driver states which
    functions it drives, the process's hold on each, and how a function is
    reset, detached from its kernel driver and given back, as Linux's
    [struct pci_driver] states its [id_table], [probe] and [remove].

    This library knows three states of a function besides its kernel driver's:
    - A function is {e detached} once {!detach} has unbound its kernel driver
      and the kernel driver has let go of it. A process can then take it.
    - A function is {e held} by this process while a driver drives it
      ({!type-hold}). No other open of the function succeeds, and no change to
      the machine touches it.
    - A function is {e lost} once a hold has ended without a clean stop, or a
      reset of it has failed. Its next open renews it ({!renew}).

    A process makes one {!t} per driver, and the holds and lost functions above
    are that value's. The opens, resets and changes of one driver's functions
    run one at a time, the driver's [f] and {!detach}'s wait included, so each
    delays the driver's other opens and changes. Each holds a lock for its whole
    run, so a call of rig's or of a rig driver's from a callback on its thread
    meanwhile is made inside a lock and raises [Invalid_argument] ([Rig]'s
    domains section). {!stop} waits for none of them. Every [Error] about
    function [i] starts with its name ({!val-name}), and names its bus address
    where it has one. Each value that takes a function number [i] raises
    [Invalid_argument] if [i < 0]. *)

type t
(** The type for a driver. *)

(** The type for what {!detach} does with the other functions of a device, such
    as a GPU's audio function or a network adapter's other ports. A function
    taken physically that Linux resets through its bus must be alone on its
    device ({!Function.take}). A driver whose reset reaches the whole device
    removes them. *)
type siblings =
  | Remove
      (** {!detach} removes them from the machine until {!attach} or a reboot.
          Behind an IOMMU that translates, it binds them to [vfio-pci] with the
          function instead, and {!attach} gives them back to their drivers.

          A kernel driver may read the data of the drivers of its device's
          other functions as its own driver's for as long as it is bound, as
          NVIDIA's reads an audio function's as a sound card's. So {!attach}
          never binds one of them to [vfio-pci] while the function's kernel
          driver is bound, one that fails included, and takes each off
          [vfio-pci] before that driver's probe. *)
  | Keep
      (** {!detach} leaves them, and refuses a function that is not bound to
          [vfio-pci] while its device has any and Linux resets it through its
          bus, naming them and their drivers. The caller then frees them as it
          decides. *)

val make :
  name:string ->
  memory_bar:int option ->
  siblings:siblings ->
  nodes:(root:string -> string -> string list) ->
  unreleased:(root:string -> string -> string option) ->
  teardown_ms:int ->
  reset:(Function.t -> (unit, string) result) ->
  (Machine.id -> bool) ->
  t
(** [make ~name ~memory_bar ~siblings ~nodes ~unreleased ~teardown_ms ~reset
     matches] is the driver of the functions [f] with [matches f], named after
    [name] ({!val-name}). [nodes] and [unreleased] read the machine's files
    under [root], which is ["/"] for {!Machine.this}.
    - [memory_bar] is the BAR through which the process reaches the function's
      memory, which {!detach} enlarges, or [None] for a function with no memory
      to reach.
    - [siblings] is what {!detach} does with the other functions of a function's
      device.
    - [nodes ~root bus] is the character devices through which the kernel driver
      serves the function at [bus] without a [dev] file under the function's
      directory in [/sys/bus/pci], as paths from [root], such as
      [["dev/nvidia0"]].
    - [unreleased ~root bus] is [Some why] if the kernel driver, no longer bound
      to the function at [bus], has not let go of it, and [None] if it has or
      the driver cannot tell. It is asked only of an unbound function. It must
      raise nothing, and returns [None] for a file it cannot read.
    - [teardown_ms] is the longest the kernel driver takes to drop the files of
      the function's devices that it holds for a process after the process has
      let go of them, as for a compute runtime after its process exits.
    - [reset fn] must stop whatever runs on the taken function [fn], whatever
      ran on it before, its kernel driver included, and reset it as its vendor
      does. It returns [Ok ()] once the function answers again with nothing
      running on it, or [Error why]. The function's bus mastering is off when
      [reset] is called. *)

val name : t -> int -> string
(** [name d i] is the name of function [i]. It is [name] of {!make} for [0],
    such as ["AMD-PCI"], and that name followed by [":i"] otherwise, such as
    ["AMD-PCI:1"]. *)

val buses : t -> Machine.t -> (string list, string) result
(** [buses d m] is the bus addresses of [d]'s functions on [m], in bus order.
    Function [i] is the [i]th.

    [Error why] if [m]'s functions cannot be listed ({!Machine.functions}). *)

(** {1:holds Opening}

    An open is a bracket. It holds the function and calls the driver's [f],
    which starts the function. Before the function can be left running, [f] must
    give the hold the driver's stop ({!set_stop}). If [f] returns [Ok _], the
    process keeps the hold until {!stop}. If [f] returns [Error _] or raises
    before {!set_stop}, the open releases the function as it found it. If it
    does so after {!set_stop}, the open stops the function through the hold, and
    the function is lost. An exception raised by [f] passes through. A driver
    reports what a request refused as an [Error], since the library's requests
    return them and its accesses raise nothing ({{!Rig_pci.errors}errors}).

    At its exit the process stops each function it still holds as {!stop} does,
    before their bus mastering is turned off and their files closed. A child of
    [fork] stops none of its parent's functions, which are still the parent's.
    An exception raised by a driver's stop at exit is printed on standard error,
    and the other functions are stopped. An exit from a callback inside a lock
    may hold a lock a function's stop takes. That stop raises
    [Invalid_argument], printed as above, and leaves its function as a killed
    process does, for its next open to renew. *)

type hold
(** The type for the process's hold on one function. *)

val open_ :
  t ->
  Machine.t ->
  int ->
  (hold -> Function.t -> ('a, string) result) ->
  ('a, string) result
(** [open_ d m i f] is [f h fn], with the process holding function [i] of [m] by
    [h] and the function [fn] taken ({!Function.take}). [f]'s [Error] messages
    get the function's name as a prefix too. A lost function, or one a dead
    process left reaching memory, is renewed before [f] runs ({!renew}).

    [Error why] without calling [f] if [m]'s functions cannot be listed, if [m]
    has no function [i] of [d], saying how many [d] drives there, if the
    process holds it already, if it is unbound and its kernel driver has not
    let go of it yet ([unreleased] of {!make}), or if it cannot be taken, with
    {!Function.take}'s reason. [Error why] with the driver's reason if its
    renewal fails, and the function is then lost. An exception raised by the
    driver's reset passes through, and the function is lost.

    Raises [Invalid_argument], with the function released, if [f] returns [Ok _]
    without {!set_stop}. *)

val set_stop : hold -> (unit -> [ `Clean | `Lost | `Unknown ]) -> unit
(** [set_stop h stop] makes [stop] the driver's stop of the function [h] holds,
    which {!stop} calls. The driver must call [set_stop] inside the open's [f],
    before the function can be left running. That is before [f]'s first write to
    the function, unless [f] itself stops the function when it fails after
    writing. [stop ()] must end the function's work and return [`Clean] if the
    function may be released as it is, [`Lost] if its work stopped but the
    function must be renewed, and [`Unknown] if its work may still run.

    Raises [Invalid_argument] if [h] has a stop already or was stopped. *)

val stop : hold -> [ `Stopped | `Unknown ]
(** [stop h] calls the driver's stop ({!set_stop}), then releases the function.
    The function may then be opened again, and its next open renews it if the
    driver's stop returned [`Lost] or [`Unknown]. [stop h] is [`Unknown] iff the
    driver's stop returned [`Unknown], and [`Stopped] otherwise. Later calls
    return what the first returned and call no driver's stop, and a call while
    another runs waits for it and returns the same. Where the first raised as it
    released the function, as one from a callback inside a lock of this
    library's on the same thread does, the next call releases it. An exception
    raised by the driver's stop passes through, and the function is lost. A
    device calls [stop] once its work is done or lost. *)

val renew : hold -> (unit, string) result
(** [renew h] resets the function [h] holds as its vendor does ([reset] of
    {!make}), with its bus mastering off, and gives back the memory dead
    processes left for it. The memory this process allocated for it stays. A
    driver calls [renew] from its [f], before it keeps any of the function's
    state, when it finds the function running firmware it cannot continue from.

    [Error why] with the reset's reason. The open then loses the function,
    whatever its [f] returns. An exception raised by the reset passes through,
    and the function is lost the same way. *)

val hang_ms : int
(** [hang_ms] is [30_000], 30 seconds. Work that a function driven without a
    kernel driver has not advanced for that long is hung, since no kernel driver
    bounds it. *)

(** {1:changes Changes to the machine}

    {!detach}, {!attach} and {!reset} change the machine, and their changes
    persist after the process. Each runs under the lock a physical take holds,
    so no process takes the function meanwhile. {!detach} and {!attach} write
    the machine's [/sys/bus/pci], so they act only on a machine the process
    reaches without a transport, such as {!Machine.this}. They need
    [CAP_SYS_ADMIN] and write access to the files they write, which root has:
    - {!detach} writes the function's [driver_override], its driver's [unbind],
      the [remove] file of each other function of its device if it removes them
      ({!type-siblings}), and the function's [enable], [config] (its command
      register) and [resourceN_resize] for its memory BAR, if it has one. Behind
      an IOMMU that translates, it writes the [driver_override] and the driver's
      [unbind] of each other function of the device in place of [remove], and
      [/sys/bus/pci/drivers_probe];
    - {!attach} writes [vfio-pci]'s [unbind], the function's [enable] and
      [driver_override], [/sys/bus/pci/rescan] and [/sys/bus/pci/drivers_probe],
      and, with [Remove], [vfio-pci]'s [unbind] and the [driver_override] of
      each other function of the device that it gives back.

    Each of them returns [Error why], changing nothing, if [m]'s functions
    cannot be listed, if [m] has no function [i] of the driver or if a process
    holds it. {!detach} and {!attach} also return [Error why] on a machine
    reached through a transport, and {!attach} when the process may not write a
    file, naming the file. {!attach} and {!reset} also return [Error why] if
    the function cannot be taken, or if it is unbound and its kernel driver has
    not let go of it yet.

    A kernel driver lets go of a function, writing to it as it does, once the
    last file of the function's character devices has gone. Its unbind either
    waits for that or, as a DRM driver's does, returns first and lets go later.
    A file remains while a process holds it open or mapped, and while the kernel
    holds it for a process, as for a compute runtime for a while after its
    process exits. The function's character devices are those with a [dev] file
    under its directory in [/sys/bus/pci], such as a GPU's DRM nodes, and those
    that [nodes] names ({!make}).

    So {!detach} unbinds a kernel driver only once no file of the function
    remains, and the kernel driver lets go inside the unbind. It refuses the
    function immediately while a process holds one of its files, waits up to
    [teardown_ms] ({!make}) for those the kernel holds, and after the unbind
    asks [unreleased] ({!make}) whether the kernel driver has let go.

    The library finds the files that processes hold in [/proc/PID/fd] and
    [/proc/PID/map_files], for the processes of its PID namespace. It finds the
    files of a DRM device in debugfs's [/sys/kernel/debug/dri/BUS/clients],
    which lists a file until its last reference goes. The library misses a
    process that opens a device after its last look, and memory of the function
    exported as a dma-buf to another device or process, which Linux lists under
    no device. The kernel driver then lets go when that holder does.

    A process must not take an unbound function before its kernel driver has let
    go, since the kernel driver's last writes would race the take's. {!open_},
    {!reset} and {!attach} refuse a function whose [unreleased] ({!make}) says
    so. *)

(** The type for how far {!detach} took a function. *)
type progress =
  | Detached  (** A process can take the function. *)
  | Unreleased of string
      (** The function is unbound, but its kernel driver has not let go of it,
          for the driver's reason ([unreleased] of {!make}), and its memory BAR
          is as it was. The kernel driver lets go, writing to the function, once
          the last holder of the function's files lets go of them. {!detach}
          called again waits for that up to [teardown_ms] and finishes, or
          returns [Error] if the kernel driver still holds the function. *)
  | Part_way of string
      (** {!detach} failed after it had started to change the machine, for the
          reason, which names the file if the process may not write one. The
          function may be neither detached nor as it was. {!attach} gives it
          back, and {!detach} called again tries again. *)

val detach : t -> Machine.t -> int -> (progress, string) result
(** [detach d m i] detaches function [i] of [m] from its kernel driver, so that
    a process can take it. A function bound to [vfio-pci] stays bound. Otherwise
    [detach] sets the function's [driver_override] to no driver, which keeps
    every kernel driver off the function through probes, rescans and module
    loads until {!attach} or a reboot, and unbinds the kernel driver. It removes
    or keeps the other functions of the function's device, as [siblings] of
    {!make} says.

    Behind an IOMMU that translates the function's addresses, a process takes it
    through VFIO, which takes its IOMMU group whole. [detach] then sets the
    [driver_override] of the function and, with [Remove], of the device's other
    functions to [vfio-pci], unbinds their drivers, resizes the memory BAR while
    they are unbound, and has Linux probe them, which binds them to [vfio-pci].
    The [vfio-pci] module must be loaded.

    Either way the function is left:
    - enabled, if unbound;
    - with its bus mastering off, which a kernel driver's unbind may leave on;
    - with its memory BAR, if it has one, at the largest size the BAR and its
      bridge take.

    The kernel driver's users, a display among them, lose the function until
    {!attach} or a reboot.

    It returns [Error] only before its first write to the machine, and after
    that write [Ok p], where [p] is how far it got ({!type-progress}).

    [Error why], changing nothing, if:
    - no process could take the function once detached, such as when its IOMMU
      group holds functions that detach leaves bound to other drivers (other
      devices' or, with [Keep], its own device's), when [vfio-pci] is not loaded
      behind an IOMMU that translates, or when its device keeps other functions
      ([Keep]) and Linux resets it through its bus;
    - a process, this one included, holds a file of one of its devices, naming
      the file and the process with its command, such as a vendor's daemon to
      stop first (nvidia-persistenced), or the kernel still holds one after
      [teardown_ms];
    - [/proc] cannot be read, or, for a function with a DRM device, its list of
      files in debugfs;
    - the process may not write [driver_override];
    - the function is already unbound and its kernel driver has still not let go
      of it after [teardown_ms].

    [Ok Detached] leaves the memory BAR small on a function that was on
    [vfio-pci] already, or where the kernel refuses every larger size. *)

val attach : t -> Machine.t -> int -> (unit, string) result
(** [attach d m i] gives function [i] of [m] back to its kernel driver. A
    function bound to no kernel driver, or to [vfio-pci], is first reset as
    {!reset} does, since its kernel driver expects it as its vendor's reset
    leaves it, whatever ran on it. Behind an IOMMU that translates, an unbound
    function, as a {!detach} that stopped part-way leaves it, is bound to
    [vfio-pci] for that reset once its kernel driver has let go. [attach] then
    unbinds [vfio-pci], disables the function, has Linux rescan the bus, which
    brings back the functions {!detach} removed, clears the function's
    [driver_override] and has Linux probe drivers, which binds the function's
    kernel driver. A function bound to its kernel driver stays bound. If the
    function's PCI driver override ([driver_override]) is set, whether a
    {!detach} that stopped part-way left it or an administrator set it, [attach]
    has Linux rescan the bus and clears the override. With [Remove]
    ({!type-siblings}), each other function of the device on [vfio-pci], or
    whose [driver_override] names it, as {!detach} behind a translating IOMMU
    leaves them, is unbound from [vfio-pci] before the function's drivers are
    probed, and goes back to its driver after. An [attach] that fails at that
    probe leaves the function unbound and disabled, and each such other
    function unbound with its [driver_override] naming [vfio-pci]. Loading
    the kernel driver's module then binds the function, and the next [attach]
    gives the others back.

    [Error why] if [vfio-pci] stays bound to such another function, or if no
    kernel driver takes the function or such another function, such as when
    the kernel driver's module is not loaded. [Error why], with the function
    left as it was but bound to [vfio-pci] where it was bound for the reset,
    if [vfio-pci] or the reset fails, and [why] says how to proceed. An
    exception raised by the driver's reset passes through, with the function
    left the same way. *)

val reset : t -> Machine.t -> int -> (unit, string) result
(** [reset d m i] takes function [i] of [m], turns its bus mastering off, resets
    it as its vendor does ({!make}) and releases it, whatever the driver's reset
    returns or raises. After a reset that returns [Ok ()], the memory dead
    processes left for the function goes ({{!Function.dma}system memory}), and
    the next open does not renew the function. A reset that fails leaves the
    function lost. An exception raised by the driver's reset passes through, as
    in an open.

    [Error why] with the driver's reset's reason. *)
