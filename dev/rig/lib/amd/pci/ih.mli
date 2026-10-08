(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's interrupt rings: where its blocks report releases, faults and
    errors, as 32-byte entries the GPU writes into its own memory and the
    process reads through the memory BAR. The ring's read and write pointers are
    byte offsets.

    The GPU's owner serializes calls. *)

type t
(** The type for the rings. *)

(** The type for what an entry reports. *)
type report =
  | Page_fault  (** GC's page walker faulted: {!Gmc.fault} reads the rest. *)
  | Fault of string  (** A shader error or another block's error. *)

val bytes : int
(** [bytes] is the length of each ring, 256 KiB. *)

val make :
  Regs.t -> Gmc.t -> Rig_pci.Window.t -> rings:int * int -> wptr:int -> t
(** [make r gmc vram ~rings ~wptr] is the GPU's two rings, at the physical
    addresses [rings] of its memory, the first ring's write pointer written back
    to the physical address [wptr]. *)

val start : t -> unit
(** [start ih] programs and enables both rings, empty. *)

val pending : t -> bool
(** [pending ih] is [true] iff the GPU wrote entries the process has not read:
    it reads the write pointer the GPU writes back. *)

val read : t -> report list
(** [read ih] is the reports of the entries the GPU wrote since the last [read],
    in order. The ring is then empty: the read pointer is written back, and an
    overflow is cleared, reading on from the oldest entry not overwritten.
    Raises {!Regs.Stuck} if the GPU's function or machine failed, whose pointer
    reads all ones. *)

val decode :
  gc:Discovery.version -> sdma:Discovery.version -> int array -> report option
(** [decode ~gc ~sdma e] is what the eight words [e] of an entry report on a GPU
    of GC [gc] and SDMA [sdma]: [None] for a release (a queue's end of pipe, a
    copy engine's trap) and for interrupts that report no error, such as a
    shader's interrupt of another kind. An error is reported with its client,
    source, ring, VMID, PASID, node and context words. Pure.

    Raises [Invalid_argument] if [e] does not hold 8 words. *)
