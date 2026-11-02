(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functions opened through VFIO, and the containers that map system memory for
    them behind an IOMMU.

    A request the kernel refuses raises {!Fail.Failed}, naming the step, the
    subject and the system's cause, with the remedy where one exists. *)

(** {1:descriptors Descriptors} *)

type fds
(** The type for the descriptors a take opens, at most six, in the order
    they open. *)

val fds : unit -> fds
(** [fds ()] holds no descriptor. *)

val opened : fds -> Unix.file_descr -> unit
(** [opened fds fd] adds [fd] to [fds]. It allocates nothing, so that a
    descriptor added as the step after its open is never lost to an
    exception. Raises [Invalid_argument] past six descriptors. *)

val last : fds -> Unix.file_descr
(** [last fds] is the descriptor added last. [fds] must hold one. *)

val close_fds : fds -> unit
(** [close_fds fds] closes the descriptors of [fds], the last first, each
    dropped from [fds] as its close returns, with nothing between that
    allocates or polls: called again after an exception ended it, it closes
    the rest, and none twice. Raises [Failure], having dropped it, for a
    descriptor that was not open. *)

(** {1:functions Functions} *)

(** The type for IOMMU models. *)
type model =
  | Type1v2  (** Behind an IOMMU. *)
  | No_iommu  (** VFIO's no-IOMMU mode, which stands in for none. *)

val open_function :
  Sysfs.t ->
  fds ->
  string ->
  model ->
  Unix.file_descr * Unix.file_descr * Unix.file_descr
(** [open_function files fds bus m] is the container, the function's
    descriptor and an eventfd. It opens [bus]'s group, on the machine whose
    files are [files], in a container of its own with model [m], then the
    function, whose first MSI vector it routes to the eventfd. Each descriptor
    goes on [fds] as it opens. *)

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

val open_ : Sysfs.t -> fds -> string -> t * Unix.file_descr
(** [open_ files fds bus] is the function [bus], in its container, and the
    eventfd its interrupts signal, their descriptors and their group's on
    [fds] as each opens. The container's descriptor is not on [fds]: {!close}
    closes it with its last function. *)

val device : t -> Unix.file_descr
(** [device t] is the descriptor of [t]'s function. *)

val map_dma : string -> string -> t -> below:int -> int -> int -> int option
(** [map_dma fn bus t ~below a n] is the device address at which [t]'s
    container maps the [n] bytes at [a], ending at or below [below]. Maps are
    counted, each by the function it is for, and each map of the same
    [(a, n)], for [t] or another function of the container, adds a count. [t]
    counts the map from the step that makes it, so an exception that ends the
    call leaves it to {!close} of [t]. It is [None] if the process reached its
    locked-memory limit, or the container holds as many mappings as
    [dma_entry_limit] allows, has no device addresses left below [below], or
    maps [(a, n)] already past [below]. [fn] is the public function that asks,
    named in the [Invalid_argument] for a released function. *)

val unmap_dma : string -> t -> int -> int -> unit
(** [unmap_dma bus t a n] drops one of [t]'s counts of [(a, n)], unmapping
    it with the container's last. It drops nothing if [t] counts none, as
    after {!close}. *)

val close : t -> unit
(** [close t] marks [t] released at its function's release, drops each of
    [t]'s counts, unmapping the pairs no other function of the container
    counts, and closes its container with the container's last function, which
    removes every mapping the container holds. A [close] called again after an
    exception ended one finishes it, and counts [t] out of its container once.
    Raises [Failure], having counted [t] out, if the container's descriptor was
    not open. *)
