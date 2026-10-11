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
(** The type for a function's container behind an IOMMU. *)

val open_ : Sysfs.t -> Unix.file_descr list ref -> string -> t * Unix.file_descr
(** [open_ files fds bus] is the container of [bus] and the eventfd its
    interrupts signal, their descriptors on [fds]. *)

val device : t -> Unix.file_descr
(** [device c] is the descriptor of [c]'s function. *)

val map_dma : string -> string -> t -> int -> int -> int option
(** [map_dma fn bus c a n] is the device address at which [c] maps the [n] bytes
    at [a]. Maps are counted, and each map of the same [(a, n)] adds a count.
    It is [None] if the process reached its locked-memory limit or [c] has no
    device addresses left. [fn] is the public function that asks, named in the
    [Invalid_argument] for a released function. *)

val unmap_dma : string -> t -> int -> int -> unit
(** [unmap_dma bus c a n] drops a count of [(a, n)], unmapping it with the last.
*)

val close : t -> unit
(** [close c] marks [c] closed at its function's release, whose closing of
    the container's descriptor then removes every mapping. *)
