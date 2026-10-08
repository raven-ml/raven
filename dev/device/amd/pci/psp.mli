(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GPU's security processor (MP0). Its bootloader starts the processor's own
   OS from the SOS components; the OS then loads every other block's firmware,
   which the process hands it through a ring in the GPU's memory, and keeps the
   trusted memory region (TMR) the firmware runs from.

   The GPU's owner serializes calls. *)

type t

(* The type for the GPU memory the processor works in, laid out by Boot in the
   boot pool, so that a partial boot finds it where the last one left it.
   Addresses are physical. *)
type memory = {
  message : int; (* 1 MiB, aligned on 1 MiB: what a command loads *)
  command : int; (* the command buffer *)
  fence : int; (* the fence the ring writes *)
  ring : int; (* the ring of frames, 64 KiB *)
}

(* [make r gmc vram tables m] is the processor, whose memory is [m]. *)
val make :
  Regs.t ->
  Gmc.t ->
  Device_pci.Window.t ->
  Device_pci.Page_table.t ->
  memory ->
  t

(* [alive p] is [true] iff the processor's OS runs. *)
val alive : t -> bool

(* [start p images ~partial] starts the processor and loads [images]: a full
   boot loads the SOS components, makes the ring, sets up the TMR and loads
   every piece; a partial boot finds the TMR the last boot left, whose size it
   reads from a scratch register, and keeps its address, the first runtime
   allocation of the GPU's memory in both. Raises Regs.Stuck if the processor
   does not answer, or answers a command with an error status, naming the
   command and the status. *)
val start : t -> Images.t -> partial:bool -> unit

(* [tmr p] is the TMR's size in bytes. *)
val tmr : t -> int

(* [partition p mode] sets the GPU's compute partition mode, one partition for
   its dies on a GPU of several. *)
val partition : t -> int -> unit
