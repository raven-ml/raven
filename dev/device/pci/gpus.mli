(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A vendor's GPUs, and the process's hold on them.

    A machine's GPUs of one vendor are its PCI functions the vendor's driver
    recognizes, numbered in bus order: GPU [i] is the same function whether the
    process reaches it through its kernel driver or over PCI, and whichever
    driver holds it.

    The process reaches a vendor's GPUs of {!Machine.this} through one
    interface, the kernel driver or PCI, fixed by its first successful open;
    another machine's GPUs only over PCI. While a driver holds a GPU, the
    process holds it ({!hold}): no other open of it succeeds, and no change to
    the machine touches it. A GPU lost while driven over PCI opens again only
    after a {!reset}.

    These facts belong to a value of {!t}: a process makes one per vendor.
    Opening and changing GPUs of one vendor are serialized, the driver's start
    included, so a start that takes seconds delays the vendor's other opens and
    changes; {!release} and {!lose} wait for none. The driver puts the GPU's
    name in front of an [Error]'s message. A function given a GPU number [i]
    raises [Invalid_argument] if [i < 0]. *)

type t
(** The type for a vendor's GPUs. *)

val make : memory_bar:int -> (Machine.id -> bool) -> t
(** [make ~memory_bar is_gpu] is the GPUs of a vendor: the functions [f] with
    [is_gpu f]. [memory_bar] is the BAR through which the process reaches their
    memory, which {!detach} enlarges. *)

val buses : t -> Machine.t -> string list
(** [buses g m] is the bus addresses of [g]'s GPUs on [m], in bus order: GPU [i]
    is the [i]th. *)

(** {1:holds Opening}

    An open is a bracket. It holds the GPU and calls the driver's [f], which
    starts the GPU. If [f] is [Ok _], the process keeps the hold until
    {!release} or {!lose}; otherwise the open gives back what it took. An
    exception raised by [f] passes through, the GPU given back: a driver reports
    the world's failures as [Error]s, the library's requests returning them and
    its accesses raising nothing ({{!Device_pci.errors}errors}). *)

type hold
(** The type for the process's hold on one GPU. *)

val bus : hold -> string
(** [bus h] is the bus address of the GPU [h] holds. *)

val open_kernel :
  t -> Machine.t -> int -> (hold -> ('a, string) result) -> ('a, string) result
(** [open_kernel g m i f] is [f h], the process holding GPU [i] of [m] by [h]
    for its kernel driver to drive. If [f h] is [Ok _], the process reaches
    [g]'s GPUs through their kernel driver from then on.

    [Error why] without calling [f] if [m] is not {!Machine.this}, if the
    process reaches [g]'s GPUs over PCI, if [m] has no GPU [i], saying how many
    it has, or if the process holds it already. *)

val open_pci :
  t ->
  Machine.t ->
  int ->
  (hold -> Function.t -> ('a, string) result) ->
  ('a, string) result
(** [open_pci g m i f] is [f h fn], the process holding GPU [i] of [m] by [h]
    and its function [fn] taken ({!Function.take}). The hold keeps [fn]. If
    [f h fn] is [Ok _] and [m] is {!Machine.this}, the process reaches [g]'s
    GPUs of this machine over PCI from then on.

    [Error why] without calling [f] if the process reaches [g]'s GPUs of
    {!Machine.this} through their kernel driver, if [m] has no GPU [i], saying
    how many it has, if the process holds it already, if it was lost over PCI
    and not {!reset} since, or if its function cannot be taken, [why] being
    {!Function.take}'s. *)

val release : hold -> unit
(** [release h] gives the GPU [h] holds back: it releases its function if
    {!open_pci} took it, and the GPU may be opened again. The driver stops its
    use of the GPU first.

    Raises [Invalid_argument] if [h] was given back already. *)

val lose : hold -> unit
(** [lose h] is {!release}, for a GPU the driver lost. A GPU {!open_pci} took
    then opens again only after a {!reset}.

    Raises [Invalid_argument] if [h] was given back already. *)

(** {1:changes Changes to the machine}

    These change the machine and persist after the process. Each refuses a GPU
    the process holds. Changes run under the lock a physical take holds; they
    leave a function bound to [vfio-pci] as it is. {!detach} and {!attach} write
    the machine's [/sys/bus/pci], so they act on a machine the process reaches
    without a transport, such as {!Machine.this}, and need [CAP_SYS_ADMIN] and
    write access to the files they write, which root has. An [Error] for a file
    the process may not write names it. *)

val detach : t -> Machine.t -> int -> (unit, string) result
(** [detach g m i] detaches GPU [i] of [m] from its kernel driver, so that a
    process can take its function, unless it is detached already. It unbinds the
    driver unless that is [vfio-pci], removes the other functions of its device,
    such as its audio function, and, unbound, enables the function and makes its
    memory BAR the largest size the BAR and its bridge take. The kernel driver's
    users, a display among them, lose the GPU until {!attach} or a reboot.

    [Error why] if [m] is reached through a transport, if [i] is no GPU, if the
    process holds it, if the process may not write a file, or if the GPU is
    still not detached, saying why, such as when an IOMMU translates its
    addresses and it is not bound to [vfio-pci]. A memory BAR left small on
    [vfio-pci] is no error; the message of the open that needs it names the
    unbind, detach and bind that enlarge it. *)

val attach : t -> Machine.t -> int -> (unit, string) result
(** [attach g m i] gives GPU [i] of [m] back to its kernel driver: Linux rescans
    the bus, which brings back the functions {!detach} removed, and binds the
    GPU's driver. It writes [/sys/bus/pci/rescan] and
    [/sys/bus/pci/drivers_probe].

    [Error why] if [m] is reached through a transport, if [i] is no GPU, if the
    process holds it, if the process may not write a file, if it is bound to
    [vfio-pci], whose [driver_override] must be cleared first, or if no driver
    takes it, such as when the driver's module is not loaded. *)

val reset :
  t ->
  Machine.t ->
  int ->
  (Function.t -> (unit, string) result) ->
  (unit, string) result
(** [reset g m i f] takes the function of GPU [i] of [m], calls [f] on it to
    reset the GPU as its vendor does, and releases it, whatever [f] returns or
    raises. A GPU lost over PCI opens again after a reset whose [f] is [Ok ()].
    Exceptions raised by [f] pass through, as in {{!holds}an open}.

    [Error why] if [i] is no GPU, if the process holds it, if its function
    cannot be taken, or [f]'s. *)
