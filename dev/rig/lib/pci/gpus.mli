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
    it succeeds, and no change to the machine touches it. A GPU lost opens again
    only after a {!reset}, which {!attach} runs too.

    These facts belong to a value of {!t}: a process makes one per vendor.
    Opening and changing GPUs of one vendor are serialized, the driver's start
    and {!detach}'s wait included, so either delays the vendor's other opens and
    changes; {!release} and {!lose} wait for none. The driver puts the GPU's
    name in front of an [Error]'s message. A function given a GPU number [i]
    raises [Invalid_argument] if [i < 0]. *)

type t
(** The type for a vendor's GPUs. *)

val make :
  memory_bar:int ->
  nodes:(root:string -> string -> string list) ->
  unreleased:(root:string -> string -> string option) ->
  teardown_ms:int ->
  reset:(Function.t -> (unit, string) result) ->
  (Machine.id -> bool) ->
  t
(** [make ~memory_bar ~nodes ~unreleased ~teardown_ms ~reset is_gpu] is the GPUs
    of a vendor: the functions [f] with [is_gpu f]. The vendor's facts read the
    machine's files under [root], ["/"] for {!Machine.this}.
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

val buses : t -> Machine.t -> string list
(** [buses g m] is the bus addresses of [g]'s GPUs on [m], in bus order: GPU [i]
    is the [i]th. *)

(** {1:holds Opening}

    An open is a bracket. It holds the GPU and calls the driver's [f], which
    starts the GPU. If [f] is [Ok _], the process keeps the hold until
    {!release} or {!lose}; otherwise the open gives back what it took. An
    exception raised by [f] passes through, the GPU given back: a driver reports
    the world's failures as [Error]s, the library's requests returning them and
    its accesses raising nothing ({{!Rig_pci.errors}errors}).

    A driver whose start fails after it wrote to the GPU gives it back itself,
    inside [f]: with {!lose}, the GPU then opens again only after a {!reset},
    which a failure that left it in a state the driver cannot know needs. [f]
    then answers [Error _] or raises, and the open gives back nothing more.

    At its exit the process stops each GPU it still holds, as the open's
    [at_exit] says, before its functions' bus mastering is turned off and their
    files closed. A child of [fork] stops none of its parent's GPUs: they are
    still the parent's. An exception raised by [at_exit] is printed on standard
    error, and the other GPUs are stopped. *)

type hold
(** The type for the process's hold on one GPU. *)

val bus : hold -> string
(** [bus h] is the bus address of the GPU [h] holds. *)

val open_ :
  t ->
  Machine.t ->
  int ->
  at_exit:('a -> unit) ->
  (hold -> Function.t -> ('a, string) result) ->
  ('a, string) result
(** [open_ g m i ~at_exit f] is [f h fn], the process holding GPU [i] of [m] by
    [h] and its function [fn] taken ({!Function.take}). The hold keeps [fn]. If
    [f h fn] is [Ok v], the process calls [at_exit v] at its exit if it holds
    the GPU then.

    A GPU a process that died left reaching memory, which it may still write
    ({!Function.alloc_dma}), is reset as {!reset} does before [f] runs, under
    the same take: its bus mastering off, the vendor's reset, and that memory
    given back. An exception raised by the vendor's reset of such a GPU passes
    through, the GPU given back and lost.

    [Error why] without calling [f] if [m] has no GPU [i], saying how many it
    has, if the process holds it already, if it was lost and not {!reset} since,
    if it is unbound and its kernel driver has not let go of it yet, which
    writes to it ([unreleased] of {!make}), if its function cannot be taken,
    [why] being {!Function.take}'s, if the memory processes that died left
    cannot be read, or if the reset of a GPU a process that died left fails,
    [why] being the vendor's, the GPU then lost.

    Raises [Invalid_argument] if [f] gave the GPU back and answered [Ok _]. *)

val renew : hold -> (unit, string) result
(** [renew h] resets the GPU [h] holds as its vendor does ([reset] of {!make}),
    its bus mastering off, and gives back the memory processes that died left
    for it; the memory this process allocated for it stays. A driver calls it
    from its start, before it keeps any of the GPU's state, when it finds the
    GPU running firmware it cannot continue from, such as one a process that
    died left. [Error why] with the reset's reason, [h] then given back as
    {!lose} does: the GPU opens again only after a {!reset}. An exception the
    reset raises passes through, [h] given back the same way. *)

val release : hold -> unit
(** [release h] gives the GPU [h] holds back: it releases its function, and the
    GPU may be opened again. The driver stops its use of the GPU first.

    Raises [Invalid_argument] if [h] was given back already. *)

val lose : hold -> unit
(** [lose h] is {!release}, for a GPU the driver lost. The GPU then opens again
    only after a {!reset}: the driver lost it in a state it cannot know, perhaps
    still running and reaching memory, which only the vendor's reset clears,
    since the function's own reset does not reset every GPU.

    Raises [Invalid_argument] if [h] was given back already. *)

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
    device, such as its audio function, and, unbound, enables the function and
    makes its memory BAR the largest size the BAR and its bridge take. The
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
    the vendor's reset returns or raises. A GPU lost opens again after a reset
    that is [Ok ()], and the memory a process that died left for it goes then
    ({!Function.alloc_dma}). Exceptions raised by the vendor's reset pass
    through, as in {{!holds}an open}.

    [Error why] if [i] is no GPU, if a process holds it, if its function cannot
    be taken, or the vendor's reset's. *)
