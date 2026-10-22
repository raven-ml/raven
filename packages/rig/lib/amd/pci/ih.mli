(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's two interrupt rings, where its blocks report releases, faults
    and errors.

    Each entry is 32 bytes, which the GPU writes into its own memory and the
    process reads through the memory BAR. The rings signal no interrupt, so
    the process polls the first ring ({!pending}) and reads its entries
    ({!read}). Calls for one GPU run one at a time. *)

(** {1:entries Entries}

    An entry is eight 32-bit words, which name the block that raised it
    (its client) and what it reports (its source). Every value is pure, and
    any domain may call it. *)

(** The type for what an entry reports. *)
type report =
  | Page_fault  (** GC's page walker faulted. *)
  | Fault of string  (** A shader error or another block's error. *)

val decode :
  gc:Rig_amd_abi.Gpu.version ->
  sdma:Rig_amd_abi.Gpu.version ->
  int array ->
  report option
(** [decode ~gc ~sdma e] is what the eight words [e] of an entry report on a
    GPU of GC [gc] and SDMA [sdma]. It is [None] for a release (a queue's end
    of pipe, a copy engine's trap) and for any interrupt that reports no
    error, such as a shader's interrupt of another kind. An error's report
    names its client, source, ring, VMID, PASID, node and context words.

    Raises [Invalid_argument] if [e] does not hold 8 words. *)

(** {1:rings Rings} *)

type t
(** The type for the rings. *)

val bytes : int
(** [bytes] is the length of each ring, 256 KiB. *)

val make :
  Regs.t -> Gmc.t -> Rig_pci.Window.t -> rings:int * int -> wptr:int -> t
(** [make r gmc vram ~rings ~wptr] is the GPU's two rings, at the physical
    addresses [rings] of its memory. The first ring writes its write pointer
    back to the physical address [wptr]. *)

val start : t -> unit
(** [start ih] programs and enables both rings, empty. The first ring writes
    its write pointer back to memory, and neither signals an interrupt. *)

val running : Regs.t -> bool
(** [running r] is [true] iff either interrupt ring of [r]'s GPU is
    enabled. *)

val halt : Regs.t -> unit
(** [halt r] disables both interrupt rings of [r]'s GPU. *)

val pending : t -> bool
(** [pending ih] is [true] iff the write pointer the GPU wrote back shows
    entries of the first ring the process has not read. *)

val read : t -> report list
(** [read ih] is the reports of the first ring's entries the GPU wrote since
    the last {!consume}, in order. It leaves them in the ring, so a caller
    records what they report before it consumes them. After an overflow it
    clears the overflow and reads on from the oldest entry the GPU did not
    overwrite.

    Raises {!Regs.Stuck} if the GPU's function or its machine failed
    ({!Rig_pci.Function.failed}). *)

val consume : t -> bool
(** [consume ih] writes the read pointer back past the entries the last
    {!read} reported, which no later [read] reports. It is [true] iff that
    read found entries, so that the pointer moved. *)
