(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's security processors (falcons), which start its GSP: a boot's
    register accesses as data ({!op}), which {!run} performs. Addresses are
    BAR 0's, a falcon's within its unit ({!gsp}, {!sec2}). *)

(** {1:ops Register sequences} *)

(** The type for what a poll or a check expects of a register's bits. *)
type cond =
  | Is of int  (** The masked bits are this value. *)
  | Not of int  (** The masked bits are not this value. *)
  | Differs of int
      (** The register differs from the register at this address, unmasked. *)

(** The type for register accesses. *)
type op =
  | Write of int * int  (** [Write (r, x)] writes [x] to [r]. *)
  | Modify of int * int * int
      (** [Modify (r, mask, x)] sets the bits [mask] of [r] to those of [x]. *)
  | Copy of int * int  (** [Copy (r, r')] writes [r]'s value to [r']. *)
  | Poll of string * int * int * cond
      (** [Poll (what, r, mask, c)] waits for [r]'s bits [mask] to meet [c], for
          at most 30 seconds; [what] names what is waited for. *)
  | Delay of int  (** [Delay us] waits [us] microseconds. *)
  | When of int * int * cond * op list
      (** [When (r, mask, c, ops)] runs [ops] if [r]'s bits [mask] meet [c]. *)
  | Expect of string * int * int * cond
      (** [Expect (what, r, mask, c)] fails, naming [what] and [r]'s value,
          unless [r]'s bits [mask] meet [c]. *)

val run : Chip.t -> op list -> (unit, string) result
(** [run c ops] performs [ops] in order on [c]'s registers. [Error] names the
    poll that timed out or the check that failed, or the GPU's failure. *)

(** {1:falcons Falcons} *)

val gsp : int
(** [gsp] is the base of the GSP's falcon, [NV_PGSP]'s first address. *)

val sec2 : int
(** [sec2] is the base of SEC2's falcon, [NV_PSEC]'s first address. *)

val wait_reset : Chip.family -> op list
(** [wait_reset f] waits for the GPU's own boot after a reset to finish: its
    firmware reports progress [GFW_BOOT_PROGRESS_COMPLETED] (Ampere, Ada) or the
    FSP [FSP_BOOT_COMPLETE_STATUS_SUCCESS] (Blackwell). *)

val reset : int -> [ `Falcon | `Riscv ] -> op list
(** [reset base core] resets the falcon at [base] (Ampere and Ada), holding the
    reset 100 ms, waits for it to scrub its memories, and selects [core]. *)

val start : int -> op list
(** [start base] starts the falcon's CPU at its boot vector, through its alias
    register if the falcon enables it. *)

val wait_halt : int -> op list
(** [wait_halt base] waits for the falcon's CPU to halt. *)

type section = {
  off : int;  (** Its offset in the image. *)
  pa : int;  (** Its address in the falcon's memory. *)
  va : int;
      (** Its virtual address, which tags the falcon's instruction memory and is
          the code's boot vector. *)
  size : int;  (** Its bytes. *)
}
(** The type for a ucode's code or data. *)

type hs = {
  image : int;  (** The image's address in the GPU's memory. *)
  code : section;
  data : section;
  pkc : int;  (** Where in its data its signature is. *)
  engines : int;  (** The engines it runs on, as a mask. *)
  ucode : int;  (** Its ucode ID. *)
}
(** The type for heavy-secure ucodes in the GPU's memory. *)

(** {1:boots Boots} *)

val legacy : fwsec:hs -> booter:hs -> libos:int -> wpr_meta:int -> op list
(** [legacy ~fwsec ~booter ~libos ~wpr_meta] starts the GSP on Ampere and Ada:
    FWSEC on the GSP's falcon, a check that it set up WPR2, the GSP's RISC-V
    core given its libos arguments at [libos], the booter on SEC2 given the WPR
    metadata at [wpr_meta] (both bus addresses), a check of the booter's
    mailbox, and a check that the GSP's core runs. *)

val cot_args : libos:int -> wpr_meta:Fb_layout.image -> string
(** [cot_args ~libos ~wpr_meta] is the FMC's boot arguments
    ([GSP_FMC_BOOT_PARAMS]): the WPR metadata [wpr_meta] and the libos arguments
    at the bus address [libos], both in coherent system memory. *)

val cot_payload : args:int -> fmc:int -> Images.fmc -> string
(** [cot_payload ~args ~fmc m] is the COT payload ([NVDM_PAYLOAD_COT], version
    2) that has the FSP boot the GSP from the FMC [m], whose image and boot
    arguments are at the bus addresses [fmc] and [args], its FRTS region where
    {!Fb_layout.cot_frts} puts it. *)

val cot : string -> op list
(** [cot payload] sends the FSP the COT [payload] and waits for the GSP's boot
    to release its lockdown (Blackwell). *)

(** {1:stops Stops} *)

val suspended : Chip.family -> op list
(** [suspended f] waits for the GSP's firmware to suspend after its unload:
    its libos reports it in the GSP's first mailbox (Ampere, Ada), or its
    core halts once the FMC has finished the shutdown (Blackwell). *)

val fwsec_sb : hs -> op list
(** [fwsec_sb fwsec] runs FWSEC with its SB command ({!Vbios.Sb}) on the
    GSP's reset falcon, which puts back the VBIOS's applications, then checks
    that the GPU's own boot status is unlocked and complete, and that FWSEC's
    error code is zero (Ampere, Ada). *)

val unload : hs -> op list
(** [unload booter] takes the GPU's protected memory (WPR2) down, if it is
    up, with the unloading booter on SEC2, and checks the booter's mailbox and
    that WPR2 is down (Ampere, Ada). *)
