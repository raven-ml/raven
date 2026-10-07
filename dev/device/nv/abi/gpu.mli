(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs, as their formats depend on them.

    A GPU's launch descriptors follow its compute engine's class, its kernels'
    machine code its streaming multiprocessors' version, and its local memory
    the number of its parts. A launch also depends on its {e windows}: the
    addresses at which the GPU's channels show kernels their shared and local
    memory. A driver reads the class and the parts from the GPU when it opens
    it, chooses the windows and sets them on each channel it opens
    ({!Method.shared_memory_window}), and gives compiled code the whole as a
    {!t}, under {!key}.

    {b Memory.} Every address at which the GPU reaches memory the driver gives
    it is below [2{^40}], where ring segments, launch descriptors and semaphores
    must lie ({!Method}). The windows are at or above [2{^40}].

    {b Work.} Work that compiled code hands one of the GPU's queues is words:
    ring entries ({!Gpfifo.entry}), each as two 32-bit words, low first. Each
    entry names a segment of the GPU's memory that compiled code wrote and keeps
    until the work completes. The driver places the entries in the queue's ring
    after its wait for earlier work and before its signal of the work's value,
    so the segments hold neither. The GPU's queues take no fills.

    {b Times.} The GPU writes times ({!Method.release_stamp},
    {!Qmd.release_stamp}, {!Method.copy_stamp}) as nanoseconds of its own timer,
    unsigned 64-bit integers in little-endian order. The timer is a clock of the
    GPU: a reader converts its times to the host clock ([CLOCK_MONOTONIC]) by
    the offset between the two, measured with a time the GPU writes between two
    readings of the host clock. *)

type t = {
  compute_class : int;
      (** The class of its compute engine, such as [0xc9c0] for Ada's. *)
  sass_version : int;
      (** The version of the machine code its multiprocessors run. *)
  gpcs : int;  (** Its graphics processing clusters. *)
  tpcs_per_gpc : int;  (** The texture processing clusters of one. *)
  sms_per_tpc : int;  (** The streaming multiprocessors of one of those. *)
  warps_per_sm : int;  (** The most warps a multiprocessor runs at once. *)
  shared_window : int;
      (** The address at which its kernels see their shared memory. *)
  local_window : int;
      (** The address at which its kernels see their local memory. *)
}
(** The type for GPUs. Every count is positive. *)

val key : t Type.Id.t
(** [key] is the key an NVIDIA GPU's {!t} is found under. *)
