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

type t = { number : int; params : params; data : params }
(** The type for requests: the ioctl's number, its parameters, and the memory a
    pointer in them names, empty if none. The caller keeps [data] alive until
    the ioctl returns. *)

(** {1:kfd KFD} *)

val version : unit -> t
(** [version ()] asks for KFD's version. *)

val version_of : t -> int
(** [version_of r] is the version [r] answered, major * 1000 + minor. *)

val acquire_vm : drm:int -> gpu:int -> t
(** [acquire_vm ~drm ~gpu] gives the process the address space of GPU [gpu],
    whose render node is open as [drm]. *)

val runtime_enable : unit -> t
(** [runtime_enable ()] enables the process's runtime. *)

type memory = [ `Gpu | `Bar | `System | `Userptr | `Mmio ]
(** The type for the kinds of memory the path allocates: GPU memory, GPU memory
    the host maps through the BAR, system memory the kernel driver owns, the
    process's own memory, and the page of registers the driver remaps. *)

val alloc : gpu:int -> va:int -> bytes:int -> memory -> t
(** [alloc ~gpu ~va ~bytes m] allocates [bytes] bytes of [m] for GPU [gpu] at
    the address [va]. All but [`Mmio] are executable. *)

val allocation : t -> int * int64
(** [allocation r] is the memory [r] allocated: its handle and the offset to map
    it at. *)

val free : int -> t
(** [free handle] gives back the memory [handle]. *)

val map : gpu:int -> int -> t
(** [map ~gpu handle] maps [handle] for GPU [gpu]. *)

val unmap : gpu:int -> int -> t
(** [unmap ~gpu handle] unmaps [handle] for GPU [gpu]. *)

val mapped : t -> int
(** [mapped r] is the GPUs a {!map} or an {!unmap} [r] reached. *)

val event : [ `Signal | `Memory | `Hardware ] -> page:int -> t
(** [event k ~page] makes an event of kind [k] (a signal, a memory exception or
    a hardware exception) on the event page whose handle is [page], or 0 for
    events after the page's first. A signal event resets as a wait takes it. *)

val event_id : t -> int
(** [event_id r] is the id of the event [r] made. *)

val destroy_event : int -> t
(** [destroy_event id] destroys the event [id]. *)

val reset_event : int -> t
(** [reset_event id] resets the event [id]. *)

val queue :
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
  t
(** [queue k ~gpu ~ring ~ring_bytes ~eop ~eop_bytes ~save ~save_bytes ~ctl_stack
     ~write ~read] makes a queue of kind [k] on GPU [gpu], from its ring,
    end-of-pipe buffer and context save area (addresses and bytes), its control
    stack's bytes, and the addresses of its write and read positions. *)

val queue_made : t -> int * int64
(** [queue_made r] is the queue [r] made: its id and its doorbell's offset. *)

val destroy_queue : int -> t
(** [destroy_queue id] destroys the queue [id]. *)

val wait : int array -> ms:int -> t
(** [wait ids ~ms] waits at most [ms] for any of the events [ids], a signal,
    then a memory and a hardware exception event, or the two exception events
    alone. *)

val exception_event : t -> [ `Memory | `Hardware ] -> int * int
(** [exception_event r k] is the id of the exception event of kind [k] the
    {!wait} [r] waited for, and the GPU whose fault set it, 0 if none. *)

val fault : t -> [ `Memory | `Hardware ] -> string
(** [fault r k] describes the fault the exception event of kind [k] of the
    {!wait} [r] reports. *)

(** {1:render The render node} *)

val device_info : unit -> t
(** [device_info ()] asks for the GPU's facts. *)

val clock_khz : t -> int
(** [clock_khz r] is the GPU's clock, in kHz, of the {!device_info} [r]. *)

val compute_units : t -> int array
(** [compute_units r] is the GPU's active compute units, 16 masks engine-major
    as cu_bitmap lays them out, of the {!device_info} [r]. *)

val alloc_context : unit -> t
(** [alloc_context ()] makes a context. *)

val context : t -> int
(** [context r] is the id of the context [r] made. *)

val stable_pstate : int -> t
(** [stable_pstate ctx] holds the GPU in its stable power state for the context
    [ctx]. *)

val free_context : int -> t
(** [free_context ctx] frees the context [ctx]. *)
