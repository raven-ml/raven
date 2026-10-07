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

    Opening and changing GPUs of one vendor are serialized. Every [Error] names
    what to do where something can be done; the driver puts the GPU's name in
    front. *)

type t
(** The type for a vendor's GPUs. *)

val make :
  name:string -> lock:string -> memory_bar:int -> (Machine.id -> bool) -> t
(** [make ~name ~lock ~memory_bar is_gpu] is the GPUs of the vendor named [name]
    in messages, such as ["AMD"]: the functions [f] with [is_gpu f]. [lock]
    names the lock file this process takes for them ({!Function.take}), and
    [memory_bar] the BAR through which the process reaches their memory, which
    {!detach} enlarges. *)

val buses : t -> Machine.t -> string list
(** [buses g m] is the bus addresses of [g]'s GPUs on [m], in bus order: GPU [i]
    is the [i]th. *)

(** {1:holds Opening} *)

type hold
(** The type for the process's hold on one GPU. *)

val bus : hold -> string
(** [bus h] is the bus address of the GPU [h] holds. *)

val open_kernel :
  t -> Machine.t -> int -> (hold -> ('a, string) result) -> ('a, string) result
(** [open_kernel g m i f] is [f h], the process holding GPU [i] of [m] by [h]
    for its kernel driver to drive. If [f h] is [Ok _], the process reaches
    [g]'s GPUs through their kernel driver from then on, and holds GPU [i] until
    {!release} or {!lose}; otherwise [h] is released. [Failure], [Sys_error] and
    [Unix.Unix_error] raised by [f] are [Error]s.

    [Error why] without calling [f] if [m] is not {!Machine.this}, if [m] is not
    Linux, if the process reaches [g]'s GPUs over PCI, if [m] has no GPU [i],
    saying how many it has, or if the process holds it already.

    Raises [Invalid_argument] if [i < 0]. *)

val open_pci :
  t ->
  Machine.t ->
  int ->
  (hold -> Function.t -> ('a, string) result) ->
  ('a, string) result
(** [open_pci g m i f] is [f h fn], the process holding GPU [i] of [m] by [h]
    and its function [fn] taken ({!Function.take}). If [f h fn] is [Ok _], the
    process reaches [g]'s GPUs of {!Machine.this} over PCI from then on, if [m]
    is that machine, and holds GPU [i] and [fn] until {!release} or {!lose};
    otherwise [fn] and [h] are released. [Failure], [Sys_error] and
    [Unix.Unix_error] raised by [f] are [Error]s.

    [Error why] without calling [f] if the process reaches [g]'s GPUs of
    {!Machine.this} through their kernel driver, if [m] has no GPU [i], saying
    how many it has, if the process holds it already, if it was lost over PCI
    and not {!reset} since, or if its function cannot be taken.

    Raises [Invalid_argument] if [i < 0]. *)

val release : hold -> unit
(** [release h] gives the GPU [h] holds back: it releases its function if
    {!open_pci} took it, and the GPU may be opened again. The driver stops its
    use of the GPU first. *)

val lose : hold -> unit
(** [lose h] is {!release}, for a GPU the driver lost. A GPU {!open_pci} took
    then opens again only after a {!reset}. *)

(** {1:changes Changes to the machine}

    These change the machine and persist after the process. Each refuses a GPU
    the process holds. *)

val detach : t -> int -> (unit, string) result
(** [detach g i] detaches GPU [i] of {!Machine.this} from its kernel driver, so
    that a process can take its function, unless it is detached already: it
    unbinds the driver, unless that is [vfio-pci], removes the other functions
    of its device, such as its audio function, enables it, and makes its memory
    BAR as large as the platform allows. The kernel driver's users, a display
    among them, lose it until {!attach} or a reboot. It needs root, or write
    access to the files under [/sys/bus/pci] it writes.

    [Error why] if [i] is no GPU, if the process holds it, if the process may
    not write a file, naming it, or if the GPU is still not detached, saying
    why, such as when an IOMMU translates its addresses and it is not bound to
    [vfio-pci].

    Raises [Invalid_argument] if [i < 0]. *)

val attach : t -> int -> (unit, string) result
(** [attach g i] gives GPU [i] of {!Machine.this} back to its kernel driver:
    Linux rescans the bus, which brings back the functions {!detach} removed,
    and binds the GPU's driver. It needs root, or write access to
    [/sys/bus/pci/rescan] and [/sys/bus/pci/drivers_probe].

    [Error why] if [i] is no GPU, if the process holds it, if it is bound to
    [vfio-pci], naming the [driverctl] command that unbinds it, or if no driver
    takes it, such as when the driver's module is not loaded.

    Raises [Invalid_argument] if [i < 0]. *)

val reset :
  t -> Machine.t -> int -> (Function.t -> unit) -> (unit, string) result
(** [reset g m i f] takes the function of GPU [i] of [m], calls [f] on it to
    reset the GPU as its vendor does, and releases it. A GPU lost over PCI opens
    again afterwards. [Failure] raised by [f] is an [Error].

    [Error why] if [i] is no GPU, if the process holds it, or if its function
    cannot be taken.

    Raises [Invalid_argument] if [i < 0]. *)

val fetch :
  t -> int -> (string -> (unit, string) result) -> (unit, string) result
(** [fetch g i f] is [f bus], the vendor fetching the firmware of GPU [i] of
    {!Machine.this}, at [bus] ({!Firmware.fetch}).

    [Error why] if [i] is no GPU or if the process holds it.

    Raises [Invalid_argument] if [i < 0]. *)
