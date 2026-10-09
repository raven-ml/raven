(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The requests the path makes of KFD and of a GPU's render node: each its
    number and its parameters, laid out as the kernel reads them, and the
    readers of what the kernel writes back. *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for a request's bytes, outside the OCaml heap, which the kernel
    reads and writes while the runtime is released. *)

type t = private {
  mutable number : int;  (** The ioctl's number. *)
  mutable size : int;  (** The bytes of its parameters. *)
  params : params;  (** Its parameters, in the first [size] bytes. *)
  data : params;  (** The memory a pointer in the parameters names. *)
  data_address : int;  (** The address of [data]. *)
}
(** The type for requests. A request is made in place by one of the functions
    below, which set every byte of its parameters, and read back once its ioctl
    returned. Each domain keeps one, so that a request allocates nothing. *)

val take : unit -> t
(** [take ()] is the domain's request, or a new one if another thread of the
    domain holds it. *)

val give : t -> unit
(** [give r] gives [r] back to the domain, once its ioctl returned and its
    answer was read: the domain's next {!take} is [r].

    A request an exception kept from its [give] costs the domain one new
    request, at its next {!take}, whose [give] makes it the domain's. *)

(** {1:kfd KFD} *)

val version : t -> unit
(** [version r] asks for KFD's version. *)

val version_of : t -> int
(** [version_of r] is the version [r] answered, major * 1000 + minor. *)

val acquire_vm : t -> drm:int -> gpu:int -> unit
(** [acquire_vm r ~drm ~gpu] gives the process the address space of GPU [gpu],
    whose render node is open as [drm]. *)

val runtime_enable : t -> unit
(** [runtime_enable r] enables the process's runtime. *)

type memory = [ `Gpu | `Bar | `System | `Userptr | `Mmio ]
(** The type for the kinds of memory the path allocates: GPU memory, GPU memory
    the host maps through the BAR, system memory the kernel driver owns, the
    process's own memory, and the page of registers the driver remaps. *)

val alloc : t -> gpu:int -> va:int -> bytes:int -> memory -> unit
(** [alloc r ~gpu ~va ~bytes m] allocates [bytes] bytes of [m] for GPU [gpu] at
    the address [va]. All but [`Mmio] are executable. *)

val handle : t -> int
(** [handle r] is the handle of the memory the {!alloc} [r] allocated. *)

val mmap_offset : t -> int64
(** [mmap_offset r] is the offset to map the memory the {!alloc} [r] allocated
    at. *)

val free : t -> int -> unit
(** [free r handle] gives back the memory [handle]. *)

val map : t -> gpu:int -> int -> unit
(** [map r ~gpu handle] maps [handle] for GPU [gpu]. *)

val unmap : t -> gpu:int -> int -> unit
(** [unmap r ~gpu handle] unmaps [handle] for GPU [gpu]. *)

val mapped : t -> int
(** [mapped r] is the GPUs a {!map} or an {!unmap} [r] reached. *)

val event : t -> [ `Signal | `Memory | `Hardware ] -> page:int -> unit
(** [event r k ~page] makes an event of kind [k] (a signal, a memory exception
    or a hardware exception) on the event page whose handle is [page], or 0 for
    events after the page's first. A signal event resets as a wait takes it. *)

val event_id : t -> int
(** [event_id r] is the id of the event [r] made. *)

val destroy_event : t -> int -> unit
(** [destroy_event r id] destroys the event [id]. *)

val reset_event : t -> int -> unit
(** [reset_event r id] resets the event [id]. *)

val queue :
  t ->
  [ `Pm4 | `Aql | `Sdma ] ->
  gpu:int ->
  ring:int ->
  ring_bytes:int ->
  eop:int ->
  eop_bytes:int ->
  save:int ->
  save_bytes:int ->
  ctl_stack:int ->
  write:int ->
  read:int ->
  unit
(** [queue r k ~gpu ~ring ~ring_bytes ~eop ~eop_bytes ~save ~save_bytes
     ~ctl_stack ~write ~read] makes a queue of kind [k] on GPU [gpu], from its
    ring, end-of-pipe buffer and context save area (addresses and bytes), its
    control stack's bytes, and the addresses of its write and read positions. *)

val queue_id : t -> int
(** [queue_id r] is the id of the queue the {!queue} [r] made. *)

val doorbell_offset : t -> int64
(** [doorbell_offset r] is the offset of the doorbell of the queue the {!queue}
    [r] made. *)

val destroy_queue : t -> int -> unit
(** [destroy_queue r id] destroys the queue [id]. *)

val wait : t -> int array -> ms:int -> unit
(** [wait r ids ~ms] waits at most [ms] for any of the events [ids], a signal,
    then a memory and a hardware exception event, or the two exception events
    alone. Raises [Invalid_argument] for more than three events. *)

val exception_id : t -> [ `Memory | `Hardware ] -> int
(** [exception_id r k] is the id of the exception event of kind [k] the {!wait}
    [r] waited for. *)

val exception_gpu : t -> [ `Memory | `Hardware ] -> int
(** [exception_gpu r k] is the GPU whose fault set the exception event of kind
    [k] of the {!wait} [r], 0 if none. *)

val fault : t -> [ `Memory | `Hardware ] -> string
(** [fault r k] describes the fault the exception event of kind [k] of the
    {!wait} [r] reports. *)

(** {1:render The render node} *)

val device_info : t -> unit
(** [device_info r] asks for the GPU's facts. *)

val clock_khz : t -> int
(** [clock_khz r] is the GPU's clock, in kHz, of the {!device_info} [r]. *)

val compute_units : t -> int array
(** [compute_units r] is the GPU's active compute units, 16 masks engine-major
    as cu_bitmap lays them out, of the {!device_info} [r]. *)

val alloc_context : t -> unit
(** [alloc_context r] makes a context. *)

val context : t -> int
(** [context r] is the id of the context [r] made. *)

val stable_pstate : t -> int -> unit
(** [stable_pstate r ctx] holds the GPU in its stable power state for the
    context [ctx]. *)

val free_context : t -> int -> unit
(** [free_context r ctx] frees the context [ctx]. *)
