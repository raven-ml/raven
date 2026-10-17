(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The library's locks.

    Each lock counts among the locks its thread holds, from before its take to
    after its give ([rig_hold] in [rig_edge.h]). A signal handler, finaliser or
    [Gc.Memprof] callback runs where the code it interrupts allocates or polls,
    so it may run inside one, and a call of rig's or of a driver's that it makes
    is then refused as the call starts, before it could wait for the lock its
    own thread holds.

    Locks nest in this order. A vendor's lock ({!Gpus}) is held over an open, a
    reset or a change, and a hold's lock over its stop. Under either come the
    GPUs' holds ({!Gpus}) and the vendor's driver's own locks. Under those come
    a container's lock ({!Vfio}), then a space's ({!Space}); and system memory's
    state ({!Sysmem}), then its reservations' lock. A function's lock
    ({!Function}), a seek ({!Local}) and the firmware found ({!Firmware}) hold
    no other.

    A vendor's lock is held across a whole open, a boot included, and a
    reset, which wait for the GPU: two of them on one GPU would interleave
    their writes. A container's and system memory's locks are held across
    the system calls that map memory for a device. A call of rig's from a
    callback during one raises. *)

type t
(** The type for locks. *)

val create : unit -> t
(** [create ()] is a new lock. *)

val protect : t -> (unit -> 'a) -> 'a
(** [protect l f] is [f ()] run holding [l], which it gives back as [f] returns
    or raises. An exception raised as it takes [l], such as by a signal handler
    that runs while it waits for [l], leaves the count as it was.

    Raises [Invalid_argument], having taken nothing, if the calling thread holds
    [l]. *)
