(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's PCI functions: taken through VFIO behind an IOMMU, or
    physically through [/sys/bus/pci]. *)

val ops : Sysfs.t -> Ops.ops
(** [ops files] is the operations of the functions of the machine whose files
    are [files]: {!Machine.this}'s for [Sysfs.v "/"]. *)

val locked :
  Sysfs.t -> string -> (unit -> ('a, string) result) -> ('a, string) result
(** [locked files bus f] is [f ()] while the process holds the lock a physical
    take of [bus] holds, or [Error why] if another holder has it or the lock's
    file cannot be opened. A take of [bus] inside [f], by the same domain,
    shares the lock. *)

val command : int
(** [command] is the offset of a function's command register in its
    configuration space. *)

val bus_master : int
(** [bus_master] is the command register's bit that lets the function master the
    bus, reaching system memory by DMA. *)
