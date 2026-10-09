(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A vendor's GPUs, and the process's hold on them.

    A machine's GPUs of one vendor are its PCI functions the vendor's driver
    recognizes, numbered in bus order: GPU [i] is the same function whether the
    process reaches it through its kernel driver or over PCI, and whichever
    driver holds it.

    While a driver holds a GPU, the process holds it ({!hold}): no other open of
    it succeeds, and no change to the machine touches it. A GPU the process lost
    is renewed by its next open ({!renew}).

    These facts belong to a value of {!t}: a process makes one per vendor.
    Opening and changing GPUs of one vendor are serialized, the driver's start
    and {!detach}'s wait included, so either delays the vendor's other opens and
    changes; {!stop} waits for neither. Every [Error] about GPU [i] starts with
    its name ({!val-name}). A function given a GPU number [i] raises
    [Invalid_argument] if [i < 0]. *)

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
    the GPUs of a vendor: the functions [f] with [is_gpu f], named after [name]
    ({!val-name}). The vendor's facts read the machine's files under [root],
    ["/"] for {!Machine.this}.
    - [memory_bar] is the BAR through which the process reaches their memory,
      which {!detach} enlarges.
    - [nodes ~root bus] is the character devices through which the kernel driver
      serves the GPU at [bus] without a [dev] file under the GPU's directory in
      [/sys/bus/pci], by path from [root], such as [["dev/nvidia0"]].
    - [unreleased ~root bus] is [Some why] if the kernel driver, no longer bound
      to the GPU at [bus], has not let go of it, and [None] if it has or the
      vendor cannot tell. {!detach} and {!open_} ask it only of an unbound GPU.
      It raises nothing: a file it cannot read is [None].
    - [teardown_ms] is the longest the kernel driver takes to drop the files of
      the GPU's devices that it holds for a process after the process let go of
      them, such as a compute runtime's after its process exits: {!detach} waits
      that long for them. A file a process still holds is not waited for.
    - [reset fn] stops whatever runs on the GPU of the taken function [fn],
      whatever ran on it before, its kernel driver included, and resets it as
      its vendor does, returning [Ok ()] once the GPU answers again with nothing
      running on it, or [Error why]. Its bus mastering is off when [reset] is
      called. *)

val name : t -> int -> string
(** [name g i] is the name of GPU [i]: [name] of {!make} for [0], such as
    ["AMD-PCI"], and that name followed by [":i"] otherwise, such as
    ["AMD-PCI:1"]. *)

val buses : t -> Machine.t -> string list
(** [buses g m] is the bus addresses of [g]'s GPUs on [m], in bus order: GPU [i]
    is the [i]th. *)

(** {1:holds Opening}

    An open is a bracket. It holds the GPU and calls the driver's [f], which
    starts the GPU. Before the GPU can be left running, [f] gives the hold the
    vendor's stop ({!set_stop}). If [f] is [Ok _], the process keeps the hold
    until {!stop}. If [f] answers [Error _] or raises before {!set_stop}, the
    open releases the GPU as it found it; after, the open stops the GPU through
    the hold, and the GPU is lost, to be renewed by its next open. An exception
    raised by [f] passes through: a driver reports the world's failures as
    [Error]s, the library's requests returning them and its accesses raising
    nothing ({{!Rig_pci.errors}errors}).

    At its exit the process stops each GPU it still holds as {!stop} does,
    before its functions' bus mastering is turned off and their files closed. A
    child of [fork] stops none of its parent's GPUs: they are still the
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
(** [open_ g m i f] is [f h fn], the process holding GPU [i] of [m] by [h] and
    its function [fn] taken ({!Function.take}), with an [Error]'s message, [f]'s
    included, after the GPU's name ({!val-name}). A GPU this process lost, or
    one a process that died left reaching memory, is renewed first ({!renew}).

    [Error why] without calling [f] if [m] has no GPU [i], saying how many it
    has, if the process holds it already, if it is unbound and its kernel driver
    has not let go of it yet, which writes to it ([unreleased] of {!make}), if
    its function cannot be taken, [why] being {!Function.take}'s, or if its
    renewal fails, [why] being the vendor's, the GPU then lost. An exception
    raised by the vendor's reset passes through, the GPU lost.

    Raises [Invalid_argument], the GPU released, if [f] answers [Ok _] without
    {!set_stop}. *)

val set_stop : hold -> (unit -> [ `Clean | `Lost | `Unknown ]) -> unit
(** [set_stop h stop] makes [stop] the vendor's stop of the GPU [h] holds, which
    {!stop} calls. The vendor calls it inside the open's [f], before the GPU can
    be left running: before its first write, or after it, for a start that stops
    a GPU it wrote to before it fails. [stop ()] ends the GPU's work and answers
    [`Clean] if the GPU may be released as it is, [`Lost] if its work stopped
    but it must be renewed, and [`Unknown] if its work may still run.

    Raises [Invalid_argument] if [h] has a stop already or was stopped. *)

val stop : hold -> [ `Stopped | `Unknown ]
(** [stop h] calls the vendor's stop ({!set_stop}), then releases the GPU's
    function: the GPU may be opened again, and is renewed first if the vendor's
    stop answered [`Lost] or [`Unknown]. It is [`Unknown] iff the vendor's stop
    answered [`Unknown], and [`Stopped] otherwise. [stop h] again answers what
    the first answered and calls nothing; a call while another runs waits for it
    and answers the same. A device calls it once its work is done or lost; the
    open's bracket and the process's exit call it too, the first call stopping
    the GPU. *)

val renew : hold -> (unit, string) result
(** [renew h] resets the GPU [h] holds as its vendor does ([reset] of {!make}),
    its bus mastering off, and gives back the memory processes that died left
    for it; the memory this process allocated for it stays. The open renews a
    GPU this process lost or a dead process left; a driver calls it from its
    start, before it keeps any of the GPU's state, when it finds the GPU running
    firmware it cannot continue from. [Error why] with the reset's reason: the
    open then loses the GPU, whatever its [f] answers. An exception the reset
    raises passes through, the GPU lost the same way. *)

val hang_ms : int
(** [hang_ms] is 30 seconds: work a GPU driven without a kernel driver has not
    advanced for that long is hung, since no kernel driver bounds it. *)

(** {1:changes Changes to the machine}

    These change the machine and persist after the process. Each refuses a GPU
    the process holds. Changes run under the lock a physical take holds;
    {!detach} leaves a function bound to [vfio-pci] as it is. {!detach} and
    {!attach} write the machine's [/sys/bus/pci], so they act on a machine the
    process reaches without a transport, such as {!Machine.this}, and need
    [CAP_SYS_ADMIN] and write access to the files they write, which root has. An
    [Error] for a file the process may not write names it. *)

val detach : t -> Machine.t -> int -> (unit, string) result
(** [detach g m i] detaches GPU [i] of [m] from its kernel driver, so that a
    process can take its function. Unless it is bound to [vfio-pci], it keeps
    every kernel driver off the GPU until {!attach} or a reboot: it sets the
    function's [driver_override] to no driver, which probes, rescans and module
    loads obey, and unbinds the driver. It removes the other functions of its
    device, such as its audio function, and, unbound, enables the function,
    turns its bus mastering off, which a kernel driver's unbind may leave on,
    and makes its memory BAR the largest size the BAR and its bridge take. The
    kernel driver's users, a display among them, lose the GPU until {!attach} or
    a reboot.

    A kernel driver lets go of a GPU, writing to it as it does, once the last
    file of the GPU's character devices goes. Its unbind either waits for that
    or, as a DRM driver's does, returns first and lets go later. So [detach]
    unbinds the driver only once no file remains, and the driver lets go inside
    the unbind. A file remains while a process holds it open or mapped, and
    while the kernel holds it for a process, as a compute runtime does for a
    while after the process exits. The GPU's character devices are those with a
    [dev] file under its directory in [/sys/bus/pci], such as its DRM nodes, and
    those [nodes] names ({!make}). [detach] reads the files processes hold from
    [/proc/PID/fd] and [/proc/PID/map_files], the processes of its PID
    namespace, and those of a DRM device from debugfs's
    [/sys/kernel/debug/dri/BUS/clients], which lists a file until its last
    reference goes. It waits up to [teardown_ms] ({!make}) for the files the
    kernel holds. After the unbind, [unreleased] ({!make}) confirms that the
    driver let go.

    Two holders escape the wait: a process that opens a device after its last
    check, and memory of the GPU exported to another device or process (a
    dma-buf), which Linux lists by no device. The driver then lets go when the
    holder does, after [detach] returns, unless [unreleased] sees it.

    [Error why], changing nothing, if [m] is reached through a transport, if [i]
    is no GPU, if a process could not take its function once detached, such as
    when an IOMMU translates its addresses and it is not bound to [vfio-pci], if
    a process holds a file of one of its devices, this one included, naming it,
    if the kernel still holds one after [teardown_ms], or if [/proc] or, for a
    GPU with a DRM device, its list of files in debugfs cannot be read.
    [Error why] with the GPU detached and its memory BAR as it was if
    [unreleased] answers [Some _] after the unbind: the driver lets go of the
    GPU when the holder does, writing to it then, and a process that takes it
    before that races those writes; [detach] called again waits for it up to
    [teardown_ms] and finishes once it let go. [Error why] if the process may
    not write a file, or if the GPU is still not detached, saying why. A memory
    BAR left small is no error: on [vfio-pci], or where the kernel refuses every
    larger size. *)

val attach : t -> Machine.t -> int -> (unit, string) result
(** [attach g m i] gives GPU [i] of [m] back to its kernel driver. A GPU bound
    to no kernel driver, or to [vfio-pci], is first reset as {!reset} does:
    whatever ran on it, in this process or another, its kernel driver expects it
    as its vendor's reset leaves it. [attach] then clears a [vfio-pci] binding
    and its [driver_override], and Linux rescans the bus, which brings back the
    functions {!detach} removed, and binds the GPU's driver. It writes
    [/sys/bus/pci/rescan], the function's [driver_override],
    [/sys/bus/pci/drivers_probe] and, for [vfio-pci], the driver's [unbind]. A
    GPU bound to its kernel driver is left as it is.

    [Error why] if [m] is reached through a transport, if [i] is no GPU, if a
    process holds it, if the process may not write a file, or if no driver takes
    it, such as when the driver's module is not loaded. If its function cannot
    be taken, such as an unbound GPU behind an IOMMU, or its reset fails, the
    GPU is left as it was and [why] says how to proceed (bind it to [vfio-pci],
    power cycle).

    Exceptions raised by the vendor's reset pass through, the GPU left as it
    was.

    On Linux 6.12, an AMD GPU given back to amdgpu after {!detach} serves its
    render node, but KFD refuses every process until the amdgpu module reloads:
    the unbind left KFD locked. *)

val reset : t -> Machine.t -> int -> (unit, string) result
(** [reset g m i] takes the function of GPU [i] of [m], turns its bus mastering
    off, resets the GPU as its vendor does ({!make}) and releases it, whatever
    the vendor's reset returns or raises. After a reset that is [Ok ()], the
    memory a process that died left for the GPU goes ({!Function.alloc_dma}),
    and the next open does not renew it. Exceptions raised by the vendor's reset
    pass through, as in {{!holds}an open}.

    [Error why] if [i] is no GPU, if a process holds it, if its function cannot
    be taken, or the vendor's reset's. *)
