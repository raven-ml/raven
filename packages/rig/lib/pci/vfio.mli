(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functions opened through VFIO, and the containers that map system memory for
    them behind an IOMMU.

    A request the kernel refuses raises {!Fail.Failed}, naming the step, the
    subject and the system's cause, with the remedy where one exists. *)

(** {1:functions Functions} *)

(** The type for IOMMU models. *)
type model =
  | Type1v2  (** Behind an IOMMU. *)
  | No_iommu  (** VFIO's no-IOMMU mode, which stands in for none. *)

val open_function :
  Sysfs.t ->
  Unix.file_descr list ref ->
  string ->
  model ->
  Unix.file_descr * Unix.file_descr * Unix.file_descr
(** [open_function files fds bus m] is the container, the function's
    descriptor and an eventfd. It opens [bus]'s group, on the machine whose
    files are [files], in a container of its own with model [m], then the
    function, whose first MSI vector it routes to the eventfd. Each descriptor
    goes on [fds] once open. *)

val bar_offset : string -> Unix.file_descr -> int -> int -> int -> int
(** [bar_offset bus fd i off n] is the offset of BAR [i] in the function's
    descriptor [fd], if VFIO maps its [n] bytes from [off]. *)

val config_offset : string -> Unix.file_descr -> int
(** [config_offset bus fd] is the offset of configuration space in [fd]. *)

val reset : Unix.file_descr -> unit
(** [reset fd] resets the function of [fd]. *)

val wait : Unix.file_descr -> int -> bool
(** [wait efd ms] waits at most [ms] milliseconds for the eventfd [efd], with
    the runtime released, and is [true] iff it was signalled. *)

(** {1:containers Containers} *)

type t
(** The type for a function behind an IOMMU, in its container. One container
    per machine holds every function the process takes behind an IOMMU there,
    unless Linux refuses a function's group to it or the group's IOMMU maps
    less of the container's device addresses: that function joins the next
    container that takes it, or a new one. *)

val open_ : Sysfs.t -> Unix.file_descr list ref -> string -> t * Unix.file_descr
(** [open_ files fds bus] is the function [bus], in its container, and the
    eventfd its interrupts signal, their descriptors on [fds]. The container's
    descriptor is not on [fds]: {!close} closes it with its last function. *)

val device : t -> Unix.file_descr
(** [device t] is the descriptor of [t]'s function. *)

val map_dma : string -> string -> t -> int -> int -> int option
(** [map_dma fn bus t a n] is the device address at which [t]'s container maps
    the [n] bytes at [a]. Maps are counted, and each map of the same [(a, n)],
    for [t] or another function of the container, adds a count. It is [None]
    if the process reached its locked-memory limit, or the container holds as
    many mappings as [dma_entry_limit] allows or has no device addresses left.
    [fn] is the public function that asks, named in the [Invalid_argument]
    for a released function. *)

val unmap_dma : string -> t -> int -> int -> unit
(** [unmap_dma bus t a n] drops a count of [(a, n)], unmapping it with the
    last. *)

val close : t -> unit
(** [close t] marks [t] released at its function's release, and closes its
    container with the container's last function, which removes every mapping
    the container holds. *)
