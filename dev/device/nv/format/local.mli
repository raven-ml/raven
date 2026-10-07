(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels' local memory.

    Each thread of a launch has local memory of its own, for its stack and
    spills, which kernels see through the local memory window
    ({!Method.local_memory_window}). A GPU takes it from one allocation, given
    with {!Method.local_memory}, which holds as much for every thread every
    multiprocessor may run at once. Each launch says how much one thread has
    ({!Qmd.set_local_memory}). *)

type t = {
  per_thread : int;  (** The bytes of each thread, a multiple of 32. *)
  per_tpc : int;
      (** The bytes of each texture processing cluster, a multiple of 32 KiB, as
          {!Method.local_memory} takes them. *)
  bytes : int;  (** The allocation's size, a multiple of 128 KiB. *)
}
(** The type for local memory. *)

val make : Gpu.t -> int -> t
(** [make g n] is the local memory of [g] for kernels whose threads need [n]
    bytes each ({!Launch.local_bytes}): [n] rounded up to 32 bytes per thread,
    for the warps of 32 threads of every multiprocessor. It is all [0] for
    [n = 0].

    Raises [Invalid_argument] if [n] is negative. *)
