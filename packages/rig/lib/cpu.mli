(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host's processor: the arch {!Rig.host} states ({!Rig.arch}), from the
    instruction set this code is compiled for and, on x86-64, the CPUID and
    XCR0 words the x86-64 psABI's levels read. *)

val host_arch : string
(** [host_arch] is this machine's host's {!Rig.arch}, read when rig starts. *)

val instruction_set : string -> string
(** [instruction_set arch] is ["x86_64"] for a level of {!x86_levels} and
    [arch] for any other. *)

val x86_levels : string list
(** [x86_levels] is the x86-64 psABI's levels, in order. *)

type words = { ecx1 : int; ebx7 : int; ecx81 : int; xcr0 : int }
(** The type for the words the levels read: CPUID.01H:ECX, CPUID.07H.00H:EBX,
    CPUID.80000001H:ECX and XCR0. *)

val x86_level : words -> string
(** [x86_level w] is the highest level of {!x86_levels} whose features, and
    those of the levels below it, [w] states. *)
