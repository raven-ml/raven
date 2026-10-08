(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's PCI functions (private): taken through VFIO behind an IOMMU,
    or physically through [/sys/bus/pci]. *)

val ops : Sysfs.t -> Ops.ops
(** [ops h] is the operations of the host [h]'s functions: {!Machine.this}'s for
    [Sysfs.v "/"]. *)

val locked :
  Sysfs.t -> string -> (unit -> ('a, string) result) -> ('a, string) result
(** [locked h bus f] is [f ()] while the process holds the lock a physical take
    of [bus] holds, or [Error why] if another holder has it or the lock's file
    cannot be opened. *)
