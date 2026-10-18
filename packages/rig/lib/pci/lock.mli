(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The library's locks.

    Each lock counts among the locks its thread holds, from before its take to
    after its give ([rig_hold] in [rig_edge.h]), so that a call from a
    callback inside one is refused as {{!Rig_pci.errors}errors} states.

    Locks nest in this order. A driver's lock ({!Driver}) is held over an
    open, a reset or a change, and a hold's lock over its stop. Under either
    come the driver's holds ({!Driver}) and the driver's own locks. Under
    those come VFIO's list of containers, then a container's lock ({!Vfio}),
    then a space's ({!Space}); and system memory's state ({!Sysmem}), then
    its reservations' lock. A function's lock
    ({!Function}), a seek ({!Local}) and the firmware found ({!Firmware}) hold
    no other.

    A vendor's lock is held across a whole open, a boot included, and a
    reset, which wait for the GPU: two of them on one GPU would interleave
    their writes. A container's and system memory's locks are held across
    the system calls that map memory for a device. *)

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
