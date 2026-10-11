(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The entries of a GPU's interrupt rings, where its blocks report
    releases, faults and errors.

    An entry is eight 32-bit words, which name the block that raised it
    (its client) and what it reports (its source). Every value is pure, and
    any domain may call it. *)

(** The type for what an entry reports. *)
type report =
  | Page_fault  (** GC's page walker faulted. *)
  | Fault of string  (** A shader error or another block's error. *)

val decode : gc:Gpu.version -> sdma:Gpu.version -> int array -> report option
(** [decode ~gc ~sdma e] is what the eight words [e] of an entry report on a
    GPU of GC [gc] and SDMA [sdma]. It is [None] for a release (a queue's end
    of pipe, a copy engine's trap) and for any interrupt that reports no
    error, such as a shader's interrupt of another kind. An error's report
    names its client, source, ring, VMID, PASID, node and context words.

    Raises [Invalid_argument] if [e] does not hold 8 words. *)
