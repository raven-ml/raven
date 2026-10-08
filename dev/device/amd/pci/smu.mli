(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GPU's power manager (MP1): messages to its firmware through three
   registers, its clocks, the GPU's whole reset (mode 1) and its machine-check
   banks.

   The GPU's owner serializes calls. *)

type t

(* [make r gmc ~table] is the power manager, whose driver table is at the
   physical address [table] of the GPU's memory. *)
val make : Regs.t -> Gmc.t -> table:int -> t

(* [alive s] is [true] iff the power manager's firmware answers. *)
val alive : t -> bool

(* [start s] gives the firmware its driver table and enables its features. *)
val start : t -> unit

(* [clocks s level] holds the memory, fabric, SoC and graphics clocks at their
   lowest or highest level. *)
val clocks : t -> [ `Lowest | `Highest ] -> unit

(* [reset s] resets the GPU whole (mode 1) and waits for its function to answer
   again: 500 ms, then its vendor ID read from configuration space, which fails
   fast where a register read would stall the bus, for at most 2 s. Outside a
   fabric only; a fabric's GPUs reset together. Raises Regs.Stuck if the
   function does not answer. *)
val reset : t -> unit

(* [banks s] is the machine-check banks the firmware holds, uncorrectable then
   correctable, for the report of a fatal hardware error. *)
val banks : t -> string
