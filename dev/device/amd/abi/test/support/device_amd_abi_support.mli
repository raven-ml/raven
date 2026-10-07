(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the AMD ABI suites share: GPUs, words, and PM4 packets read back. *)

open Device_amd_abi

val timeout : float
(** [timeout] is the hang guard of each suite's top tests, in seconds. Every
    suite runs in milliseconds. *)

(** {1:gpus GPUs} *)

val gpu :
  ?target:Gpu.version ->
  ?sdma:Gpu.version ->
  ?xccs:int ->
  ?shader_engines:int ->
  ?compute_units:int ->
  ?scratch_slots:int ->
  Gpu.version ->
  Gpu.t
(** [gpu gc] is a GPU of GC [gc]: target [gc], SDMA 6.0.0, one die of 4 shader
    engines and 32 compute units of 32 scratch slots, unless said otherwise. *)

val families : Gpu.version list
(** [families] is the GC versions the register tables define. *)

val version : Gpu.version -> string
(** [version v] is ["a.b.c"]. *)

(** {1:words Words} *)

val words : string -> int list
(** [words s] is the 32-bit little-endian words of [s], as unsigned integers. *)

val encode : int Packet.t -> int list
(** [encode p] is the words of [p] over integers. *)

(** {1:pm4 PM4 packets} *)

val set_sh_reg : int
(** [set_sh_reg] is PACKET3_SET_SH_REG's opcode, whose first word is an offset
    from {!sh_start}. Likewise {!set_uconfig_reg} from {!uconfig_start}, and
    {!pred_exec}, whose second word is the dies' mask from bit 24 and the count
    of the words it predicates (soc15d.h, nvd.h). *)

val set_uconfig_reg : int
val pred_exec : int
val sh_start : int
val uconfig_start : int

val packets : int list -> (int * int list) list
(** [packets ws] is the type 3 packets of [ws], each as its opcode and its body.
    Raises [Failure] on a word that starts no type 3 packet, or a packet that
    passes [ws]'s end. *)

val writes : int list -> (int * int) list
(** [writes ws] is the register writes of the [SET_SH_REG] and [SET_UCONFIG_REG]
    packets of [ws], in order, each as its register's address and its value. *)

val pm4 : Gpu.t -> int list -> string
(** [pm4 g ws] is [ws] as text, one packet a line: its name, then the registers
    a register write sets, by [g]'s names where it has one, or its body in
    hexadecimal. *)
