(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GPU's interrupt rings: where its blocks report releases, faults and
   errors, as 32-byte entries the GPU writes into the GPU's memory and the
   process reads through the memory BAR.

   The GPU's owner serializes calls. *)

type t

(* The type for what an entry reports. *)
type report =
  | Release (* a queue's end of work or copy trap: a wake-up *)
  | Page_fault (* GC's page walker faulted: Gmc.fault reads the rest *)
  | Fault of string (* a shader error or another block's report *)

(* [bytes] is the length of each ring, 256 KiB. *)
val bytes : int

(* [make r gmc vram ~rings ~wptr] is the GPU's two rings, at the physical
   addresses [rings] of its memory, the first ring's write pointer written back
   to [wptr]. *)
val make :
  Regs.t -> Gmc.t -> Device_pci.Window.t -> rings:int * int -> wptr:int -> t

(* [start ih] programs and enables both rings, empty. *)
val start : t -> unit

(* [pending ih] is [true] iff the GPU wrote entries the process has not read. It
   reads the write pointer through the memory BAR. *)
val pending : t -> bool

(* [read ih] is the reports of the entries the GPU wrote since the last [read],
   in order; the ring is then empty, an overflow cleared. *)
val read : t -> report list

(* [decode ~gc ~sdma e] is what the eight words [e] of an entry report, on a GPU
   of GC [gc] and SDMA [sdma]. Pure. *)
val decode :
  gc:Discovery.version -> sdma:Discovery.version -> int array -> report
