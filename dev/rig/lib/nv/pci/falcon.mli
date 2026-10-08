(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's security processors, which start its GSP (private).

    On Ampere and Ada the GSP's falcon runs FWSEC from the VBIOS, which sets up
    the protected region of memory (WPR2); SEC2 then runs the booter, which
    loads the GSP's firmware into it and starts the GSP's RISC-V core. On
    Blackwell the process asks the FSP, through its message queue, to boot the
    GSP from the FMC (the chain of trust, COT). The GSP's CPU sequence ({!Gsp})
    also resets and starts the GSP's falcon on Ampere and Ada.

    A heavy-secure ucode is copied into a falcon from the GPU's memory, 256
    bytes per DMA command, and checked by the falcon's boot ROM against its
    signature. *)

val gsp : int
(** [gsp] is the base of the GSP's falcon registers. *)

val sec2 : int
(** [sec2] is the base of SEC2's falcon registers. *)

val wait_reset : Chip.t -> (unit, string) result
(** [wait_reset c] waits for the GPU's own boot after a reset to finish: its
    firmware writes [0xff] into a scratch register. *)

val reset : Chip.t -> int -> [ `Falcon | `Riscv ] -> (unit, string) result
(** [reset c base core] resets the falcon at [base], waits for it to scrub its
    memories, and selects its [core]. *)

val start : Chip.t -> int -> unit
(** [start c base] starts the falcon's CPU at its boot vector. *)

val wait_halt : Chip.t -> int -> (unit, string) result
(** [wait_halt c base] waits for the falcon's CPU to halt. *)

val boot :
  Chip.t ->
  [ `Legacy of fwsec:int * Vbios.fwsec * booter:int * Images.booter
  | `Cot of args:int * fmc:int * Images.fmc ] ->
  libos:int ->
  wpr_meta:int ->
  (unit, string) result
(** [boot c start ~libos ~wpr_meta] starts the GSP with its libos arguments at
    [libos] and its WPR metadata at [wpr_meta], bus addresses the GSP reads.
    [`Legacy (fwsec, f, booter, b)] runs FWSEC [f] and the booter [b], whose
    images are in the GPU's memory at [fwsec] and [booter];
    [`Cot (args, fmc, m)] sends the FSP the COT payload of the FMC [m], whose
    image and boot arguments are at the bus addresses [fmc] and [args]. [Error]
    names the step that failed and the mailboxes it left. *)
