(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's PCI functions: taken through VFIO behind an IOMMU, or
    physically through [/sys/bus/pci]. *)

val ops : Sysfs.t -> Ops.ops
(** [ops files] is the operations of the functions of the machine whose files
    are [files]: {!Machine.this}'s for [Sysfs.v "/"]. *)

val take : 's Sysfs.lock -> (Ops.fn, string) result
(** [take l] is the function [l] locks, taken as {!Sysfs.access} says, or
    [Error why]. Taken physically, the function keeps the lock ({!Sysfs.keep})
    until its release. *)

val command : int
(** [command] is the offset of a function's command register in its
    configuration space. *)

val bus_master : int
(** [bus_master] is the command register's bit that lets the function master the
    bus, reaching system memory by DMA. *)
